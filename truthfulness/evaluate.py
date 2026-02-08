#!/usr/bin/env python3
"""
Generic evaluation scaffold (currently implements TruthfulQA).

TruthfulQA evaluation by *length-normalized conditional logprob*:

For each example:
  score(best)  = avg_t log p(token_t | chat(user_prompt) + best_prefix)
  score(wrong) = avg_t log p(token_t | chat(user_prompt) + wrong_prefix)
(where wrong = incorrect_answers[0])

Correct = 1 if score(best) > score(wrong) else 0

Supports:
  - Base model (HF id/path) OR "edited_models/<base>/<edit_dir_or_file>"
  - Optional steering hooks via llm.apply_model(...)
  - vLLM prompt_logprobs scoring (no answer verifier, no MC prompts)

Output:
  evaluate_results/<dataset>/<model_out_name>/1_run.json
"""

import os
import re
import json
import csv
import argparse
import numpy as np
import torch
import tempfile
import atexit
import warnings
from collections import Counter
from functools import partial
from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

os.environ["VLLM_USE_V1"] = "0"
from vllm import LLM, SamplingParams

from utils import MODEL_MAP  # keep your existing mapping


# -------------------------
# Dataset registry (extend later)
# -------------------------
DATASET_REGISTRY = {
    "truthfulqa": {
        "loader": "load_from_disk",
        "default_path": "data/truthfulqa/test",
    },
    "factor_news": {
        "loader": "local_csv",
        "default_path": "factor/news_factor.csv",
    },
    # Add more datasets later...
}


# -------------------------
# Chat template helpers
# -------------------------
def chat_prefix(user_prompt: str, tokenizer) -> str:
    """Chat template ending right at assistant start (assistant content empty)."""
    conv = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": ""},
    ]
    return tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)


def chat_with_assistant_content(user_prompt: str, assistant_text: str, tokenizer) -> str:
    """Chat template that includes assistant_text as assistant message content."""
    conv = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_text},
    ]
    return tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)


# -------------------------
# prompt_logprobs warning utilities
# -------------------------
_MISSING_TOPK = Counter()
_MAX_CANDIDATE_PREVIEW = 12

def _warn_token_not_in_topk(
    *,
    token_id: int,
    position: int | None = None,
    prompt_logprobs_k: int | None = None,
    prefix_len_tokens: int | None = None,
    candidate_map=None,
) -> None:
    """
    Warn when the gold token isn't present in the returned prompt_logprobs dict
    (i.e., not in top-k for that position). This usually indicates either:
      - prompt_logprobs_k too small, OR
      - offset/prefix_len token alignment mismatch, OR
      - prompt used for tokenization differs from prompt used by vLLM.
    """
    cand_ids = []
    if isinstance(candidate_map, dict):
        cand_ids = list(candidate_map.keys())

    _MISSING_TOPK[token_id] += 1

    preview = cand_ids[:_MAX_CANDIDATE_PREVIEW]
    preview_str = ", ".join(map(str, preview)) + ("..." if len(cand_ids) > len(preview) else "")

    warnings.warn(
        "Token not present in vLLM prompt_logprobs top-k candidates. "
        "This indicates a scoring/candidate mismatch (e.g., wrong offset/prefix_len, "
        "or prompt_logprobs_k too small). Returning -inf.\n"
        f"  token_id={token_id}, pos={position}, prefix_len={prefix_len_tokens}, k={prompt_logprobs_k}\n"
        f"  candidates({len(cand_ids)}): [{preview_str}]",
        RuntimeWarning,
        stacklevel=3,
    )


def _extract_token_logprob(
    prompt_logprob_item,
    token_id: int,
    *,
    pos: int | None = None,
    prompt_logprobs_k: int | None = None,
    prefix_len_tokens: int | None = None,
) -> float:
    """
    vLLM returns prompt_logprobs as a per-position mapping: token_id -> Logprob-like object.
    Handle common shapes robustly.
    """
    if prompt_logprob_item is None:
        return float("-inf")

    if isinstance(prompt_logprob_item, dict):
        obj = prompt_logprob_item.get(token_id, None)
        if obj is None:
            _warn_token_not_in_topk(
                token_id=token_id,
                position=pos,
                prompt_logprobs_k=prompt_logprobs_k,
                prefix_len_tokens=prefix_len_tokens,
                candidate_map=prompt_logprob_item,
            )
            return float("-inf")

        if hasattr(obj, "logprob"):
            return float(obj.logprob)
        if isinstance(obj, dict) and "logprob" in obj:
            return float(obj["logprob"])
        try:
            return float(obj)
        except Exception:
            return float("-inf")

    return float("-inf")


def length_normalized_answer_logprob(
    *,
    prefix_len_tokens: int,
    prompt_logprobs_list,
    full_ids,
    prompt_logprobs_k: int | None = None,
) -> float:
    """
    Compute avg logprob over answer tokens only, using returned prompt_logprobs.

    prefix_len_tokens: number of tokens in the prefix (user + assistant-start) region.
    """
    ans_start = prefix_len_tokens
    if ans_start >= len(full_ids):
        return float("-inf")

    s = 0.0
    n = 0

    # vLLM sometimes returns prompt_logprobs aligned to its internal tokenization length,
    # so we compute an offset to index the right positions.
    offset = len(prompt_logprobs_list) - len(full_ids)
    L = min(len(full_ids), len(prompt_logprobs_list))
    for p in range(ans_start, L):
        tok_id = full_ids[p]
        vpos = p + offset
        if vpos < 0 or vpos >= len(prompt_logprobs_list):
            return float("-inf")

        lp = _extract_token_logprob(
            prompt_logprobs_list[vpos],
            tok_id,
            pos=vpos,
            prompt_logprobs_k=prompt_logprobs_k,
            prefix_len_tokens=prefix_len_tokens,
        )
        if lp == float("-inf"):
            return float("-inf")
        s += lp
        n += 1

    return s / max(n, 1)


# -------------------------
# Steering hook (your style)
# -------------------------
class LinearInterventionHook:
    def __init__(self, direction, weight):
        self.direction = direction
        self.weight = weight

    def __call__(self, module, input, output):
        ref = output[0] if isinstance(output, tuple) else output
        self.direction = self.direction.type_as(ref)
        if isinstance(output, tuple):
            modified = ref + self.direction.to(ref.device) * self.weight
            return (modified,) + output[1:]
        else:
            return output + self.direction.to(output.device) * self.weight


def add_steering(model, directions, weight, components=None):
    if not components:
        return
    for component in components:
        if component not in directions:
            continue
        steering_vector = directions[component]
        hook = LinearInterventionHook(steering_vector, weight)
        eval(f"model.{component}.register_forward_hook(hook)")


def parse_components_spec(spec: str, num_layers: int):
    """
    Parse strings like:
      'attn0-1', 'mlp1-3', 'attn', 'mlp', 'attn0-1mlp1-3', 'attnmlp'
    Returns component paths:
      'model.layers[i].self_attn.o_proj' and 'model.layers[i].mlp.down_proj'
    """
    if not spec:
        return []

    pattern = re.compile(r"(attn|mlp)(\d+(?:-\d+)?)?")
    matches = list(pattern.finditer(spec))
    if not matches:
        raise ValueError(f"Unrecognized component spec: {spec}")

    layers_to_attn = set()
    layers_to_mlp = set()

    def expand_range(tok):
        # tok: None | 'k' | 'k-m' (end exclusive)
        if tok is None:
            return list(range(num_layers))
        if "-" in tok:
            a, b = tok.split("-", 1)
            s = int(a)
            e = int(b)
        else:
            s = int(tok)
            e = s + 1
        s = max(0, s)
        e = min(num_layers, e)
        if s >= e:
            return []
        return list(range(s, e))

    for m in matches:
        kind = m.group(1)
        rng = m.group(2)
        layer_idxs = expand_range(rng)
        if kind == "attn":
            layers_to_attn.update(layer_idxs)
        else:
            layers_to_mlp.update(layer_idxs)

    components = []
    for i in sorted(layers_to_attn):
        components.append(f"model.layers[{i}].self_attn.o_proj")
    for i in sorted(layers_to_mlp):
        components.append(f"model.layers[{i}].mlp.down_proj")
    return components


def register_steering(hf_model, *, direction_path, alpha, components_spec):
    inner = getattr(hf_model, "model", None)
    if inner is None or not hasattr(inner, "layers"):
        raise RuntimeError("Unexpected model structure: missing .model.layers")
    num_layers = len(inner.layers)

    directions = torch.load(direction_path, map_location="cpu")["post"]["direction"]
    comps = parse_components_spec(components_spec, num_layers)
    add_steering(hf_model, directions=directions, weight=alpha, components=comps)


# -------------------------
# Optional: rank-1 edit pack application
# -------------------------
@torch.no_grad()
def apply_rank1_edits_to_hf_model(hf_model, edit_pack: dict) -> None:
    inner = getattr(hf_model, "model", hf_model)
    layers = inner.layers
    cfg = hf_model.config

    H = int(edit_pack["meta"].get("hidden_size") or cfg.hidden_size)
    A = int(edit_pack["meta"].get("num_heads") or cfg.num_attention_heads)
    D = int(
        edit_pack["meta"].get("head_dim")
        or getattr(cfg, "head_dim", None)
        or (H // A)
    )

    vhat_attn_meta = edit_pack["meta"]["vhat_attn"]
    vhat_mlp_meta = edit_pack["meta"]["vhat_mlp"]

    for (layer_idx, head_idx), payload in edit_pack.get("attn", {}).items():
        lam = float(payload["lambda"])
        if lam == 0.0:
            continue
        oW = layers[layer_idx].self_attn.o_proj.weight
        dtype = oW.dtype
        vhat = vhat_attn_meta[layer_idx].to(device=oW.device, dtype=dtype)
        khat = payload["k_hat"].to(device=oW.device, dtype=dtype)
        h0 = head_idx * D
        h1 = h0 + D
        oW.data[:, h0:h1].add_(lam * torch.outer(vhat, khat))

    for (layer_idx, neuron_idx), payload in edit_pack.get("mlp", {}).items():
        lam = float(payload["lambda"])
        if lam == 0.0:
            continue
        dW = layers[layer_idx].mlp.down_proj.weight
        dtype = dW.dtype
        vhat = vhat_mlp_meta[layer_idx].to(device=dW.device, dtype=dtype)
        dW.data[:, neuron_idx].add_(lam * vhat)


# -------------------------
# TruthfulQA scoring eval (UNCHANGED)
# -------------------------
def eval_truthfulqa_score(*, llm, tokenizer, ds, prompt_logprobs_k: int, batch_size: int, limit: int):
    qs, bests, wrongs = [], [], []
    for ex in ds:
        q = (ex.get("question") or "").strip()
        best = (ex.get("best_answer") or "").strip()
        inc = ex.get("incorrect_answers") or []
        wrong0 = ""
        if len(inc) > 0 and isinstance(inc[0], str):
            wrong0 = inc[0].strip()
        if not q or not best or not wrong0:
            continue
        qs.append(q)
        bests.append(best)
        wrongs.append(wrong0)

    if limit and limit > 0:
        qs = qs[:limit]
        bests = bests[:limit]
        wrongs = wrongs[:limit]

    print(f"[TruthfulQA] scoring pairs: {len(qs)}")

    prefix_lens = []
    for q in qs:
        pref_ids = tokenizer.encode(chat_prefix(q, tokenizer), add_special_tokens=False)
        prefix_lens.append(len(pref_ids))

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=max(1, prompt_logprobs_k),
    )

    def batched(lst, bs):
        for i in range(0, len(lst), bs):
            yield i, lst[i:i + bs]

    full_texts_best = [chat_with_assistant_content(q, a, tokenizer) for q, a in zip(qs, bests)]
    full_texts_wrong = [chat_with_assistant_content(q, a, tokenizer) for q, a in zip(qs, wrongs)]
    full_ids_best = [tokenizer.encode(t, add_special_tokens=False) for t in full_texts_best]
    full_ids_wrong = [tokenizer.encode(t, add_special_tokens=False) for t in full_texts_wrong]

    best_scores = [float("-inf")] * len(qs)
    for start, batch in batched(full_texts_best, batch_size):
        outs = llm.generate(batch, sp)
        for j, out in enumerate(outs):
            idx = start + j
            plps = getattr(out, "prompt_logprobs", None)
            if plps is None:
                raise RuntimeError("vLLM did not return prompt_logprobs; check vLLM version.")
            best_scores[idx] = length_normalized_answer_logprob(
                prefix_len_tokens=prefix_lens[idx],
                prompt_logprobs_list=plps,
                full_ids=full_ids_best[idx],
                prompt_logprobs_k=prompt_logprobs_k,
            )

    wrong_scores = [float("-inf")] * len(qs)
    for start, batch in batched(full_texts_wrong, batch_size):
        outs = llm.generate(batch, sp)
        for j, out in enumerate(outs):
            idx = start + j
            plps = getattr(out, "prompt_logprobs", None)
            if plps is None:
                raise RuntimeError("vLLM did not return prompt_logprobs; check vLLM version.")
            wrong_scores[idx] = length_normalized_answer_logprob(
                prefix_len_tokens=prefix_lens[idx],
                prompt_logprobs_list=plps,
                full_ids=full_ids_wrong[idx],
                prompt_logprobs_k=prompt_logprobs_k,
            )

    records, corrects, margins = [], [], []
    for i in range(len(qs)):
        sb = best_scores[i]
        sw = wrong_scores[i]
        correct = 1 if (sb > sw) else 0
        margin = sb - sw if (np.isfinite(sb) and np.isfinite(sw)) else float("nan")
        corrects.append(correct)
        margins.append(margin)
        records.append(
            {
                "question": qs[i],
                "best_answer": bests[i],
                "wrong_answer": wrongs[i],
                "score_best_avg_logprob": float(sb) if np.isfinite(sb) else sb,
                "score_wrong_avg_logprob": float(sw) if np.isfinite(sw) else sw,
                "margin": float(margin) if np.isfinite(margin) else margin,
                "correct": bool(correct),
                "best_answer_len_tokens": int(len(tokenizer.encode(bests[i], add_special_tokens=False))),
                "wrong_answer_len_tokens": int(len(tokenizer.encode(wrongs[i], add_special_tokens=False))),
            }
        )

    acc = float(np.mean(corrects) * 100.0) if corrects else 0.0
    mean_margin = float(np.nanmean(margins)) if margins else float("nan")
    std_margin = float(np.nanstd(margins, ddof=1)) if len(margins) > 1 else 0.0

    aggregate = {
        "accuracy": acc,
        "mean_margin": mean_margin,
        "std_margin": std_margin,
    }
    return aggregate, records


# -------------------------
# FACTOR-NEWS eval (NEW)
# -------------------------
def _factor_news_user_prompt(prefix: str) -> str:
    # Minimal instruct wrapper, consistent across candidates.
    return (
        "Continue the following news passage with exactly one next sentence.\n\n"
        f"Passage:\n{prefix}\n\n"
        "Next sentence:"
    )


def load_factor_news_csv(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"factor_news CSV not found: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        raise RuntimeError(f"No rows read from: {path}")
    return rows


def eval_factor_news_score(*, llm, tokenizer, rows, prompt_logprobs_k: int, batch_size: int, limit: int):
    """
    Each row needs:
      - full_prefix
      - completion
      - contradiction_0, contradiction_1, contradiction_2
    """
    prompts = []
    candidates = []  # list of [pos, neg0, neg1, neg2]
    raw_prefixes = []

    for r in rows:
        prefix = (r.get("full_prefix") or "").strip()
        pos = (r.get("completion") or "").strip()
        neg0 = (r.get("contradiction_0") or "").strip()
        neg1 = (r.get("contradiction_1") or "").strip()
        neg2 = (r.get("contradiction_2") or "").strip()

        if not prefix or not pos or not neg0 or not neg1 or not neg2:
            continue

        user_prompt = _factor_news_user_prompt(prefix)
        prompts.append(user_prompt)
        candidates.append([pos, neg0, neg1, neg2])
        raw_prefixes.append(prefix)

    if limit and limit > 0:
        prompts = prompts[:limit]
        candidates = candidates[:limit]
        raw_prefixes = raw_prefixes[:limit]

    print(f"[FACTOR-NEWS] examples: {len(prompts)}")

    # prefix lens for each example (same for all 4 candidates)
    prefix_lens = []
    for up in prompts:
        pref_ids = tokenizer.encode(chat_prefix(up, tokenizer), add_special_tokens=False)
        prefix_lens.append(len(pref_ids))

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=max(1, prompt_logprobs_k),
    )

    def batched(lst, bs):
        for i in range(0, len(lst), bs):
            yield i, lst[i:i + bs]

    # Flatten 4 candidates per example into one big list for vLLM
    flat_full_texts = []
    flat_full_ids = []
    flat_meta = []  # (ex_idx, cand_idx)
    for i, up in enumerate(prompts):
        for j in range(4):
            cand = candidates[i][j]
            ft = chat_with_assistant_content(up, cand, tokenizer)
            flat_full_texts.append(ft)
            flat_full_ids.append(tokenizer.encode(ft, add_special_tokens=False))
            flat_meta.append((i, j))

    flat_scores = [float("-inf")] * len(flat_full_texts)
    for start, batch in batched(flat_full_texts, batch_size):
        outs = llm.generate(batch, sp)
        for j, out in enumerate(outs):
            idx = start + j
            plps = getattr(out, "prompt_logprobs", None)
            if plps is None:
                raise RuntimeError("vLLM did not return prompt_logprobs; check vLLM version.")
            ex_idx, cand_idx = flat_meta[idx]
            flat_scores[idx] = length_normalized_answer_logprob(
                prefix_len_tokens=prefix_lens[ex_idx],
                prompt_logprobs_list=plps,
                full_ids=flat_full_ids[idx],
                prompt_logprobs_k=prompt_logprobs_k,
            )

    # Reconstruct per-example candidate scores
    per_ex_scores = [[float("-inf")] * 4 for _ in range(len(prompts))]
    for k, (ex_idx, cand_idx) in enumerate(flat_meta):
        per_ex_scores[ex_idx][cand_idx] = flat_scores[k]

    records, corrects, margins = [], [], []
    for i in range(len(prompts)):
        sc = per_ex_scores[i]
        # pos is index 0; negatives are 1..3
        pos_score = sc[0]
        best_neg = max(sc[1], sc[2], sc[3])
        pred = int(np.argmax(sc)) if all([np.isfinite(x) for x in sc]) else int(np.argmax([(-1e30 if not np.isfinite(x) else x) for x in sc]))
        correct = 1 if pred == 0 else 0
        margin = pos_score - best_neg if (np.isfinite(pos_score) and np.isfinite(best_neg)) else float("nan")

        corrects.append(correct)
        margins.append(margin)

        records.append(
            {
                "full_prefix": raw_prefixes[i],
                "user_prompt": prompts[i],
                "completion": candidates[i][0],
                "contradiction_0": candidates[i][1],
                "contradiction_1": candidates[i][2],
                "contradiction_2": candidates[i][3],
                "score_completion_avg_logprob": float(pos_score) if np.isfinite(pos_score) else pos_score,
                "score_contradiction_0_avg_logprob": float(sc[1]) if np.isfinite(sc[1]) else sc[1],
                "score_contradiction_1_avg_logprob": float(sc[2]) if np.isfinite(sc[2]) else sc[2],
                "score_contradiction_2_avg_logprob": float(sc[3]) if np.isfinite(sc[3]) else sc[3],
                "pred_choice": int(pred),  # 0=completion, 1/2/3=contradiction_{idx-1}
                "margin_vs_best_neg": float(margin) if np.isfinite(margin) else margin,
                "correct": bool(correct),
                "completion_len_tokens": int(len(tokenizer.encode(candidates[i][0], add_special_tokens=False))),
                "best_neg_len_tokens": int(
                    max(
                        len(tokenizer.encode(candidates[i][1], add_special_tokens=False)),
                        len(tokenizer.encode(candidates[i][2], add_special_tokens=False)),
                        len(tokenizer.encode(candidates[i][3], add_special_tokens=False)),
                    )
                ),
            }
        )

    acc = float(np.mean(corrects) * 100.0) if corrects else 0.0
    mean_margin = float(np.nanmean(margins)) if margins else float("nan")
    std_margin = float(np.nanstd(margins, ddof=1)) if len(margins) > 1 else 0.0

    aggregate = {
        "accuracy": acc,
        "mean_margin": mean_margin,
        "std_margin": std_margin,
    }
    return aggregate, records


def main():
    parser = argparse.ArgumentParser(description="Evaluation (TruthfulQA scoring + FACTOR-NEWS).")

    # Keep dataset arg for future expansion
    parser.add_argument("--dataset", type=str, default="truthfulqa", choices=list(DATASET_REGISTRY.keys()))

    # Dataset path (TruthfulQA)
    parser.add_argument(
        "--truthfulqa_test_path",
        type=str,
        default=DATASET_REGISTRY["truthfulqa"]["default_path"],
        help="load_from_disk path to saved TruthfulQA test split",
    )

    # Dataset path (FACTOR-NEWS)
    parser.add_argument(
        "--factor_news_path",
        type=str,
        default=DATASET_REGISTRY["factor_news"]["default_path"],
        help="Path to local factor/news_factor.csv",
    )

    parser.add_argument("--model", type=str, default="llama3-8b",
                        help="HF model id/path OR edited_models/<base>/<dir_or_pt>")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--max_model_len", type=int, default=1024)

    # Steering
    parser.add_argument("--do_steering", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--components_spec", type=str, default="attnmlp")
    parser.add_argument("--direction_path", type=str, default="",
                        help="Path to directions .pt (expects ['post']['direction'] dict)")

    # Scoring params
    parser.add_argument("--prompt_logprobs_k", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=-1)

    # Output root
    parser.add_argument("--out_dir", type=str, default="evaluate_results")

    args = parser.parse_args()

    # -------------------------
    # Load dataset
    # -------------------------
    if args.dataset == "truthfulqa":
        ds = load_from_disk(args.truthfulqa_test_path)
        ds_factor_rows = None
    elif args.dataset == "factor_news":
        ds = None
        ds_factor_rows = load_factor_news_csv(args.factor_news_path)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented yet.")

    # -------------------------
    # Detect edited model path
    # -------------------------
    parts = args.model.strip("/").split("/")
    is_edited = len(parts) >= 3 and parts[0] == "edited_models"
    edit_pack = None

    if is_edited:
        base_model_name = parts[1]
        base_model_path = MODEL_MAP.get(base_model_name, base_model_name)

        edited_dir = os.path.join(*parts[:3])
        edit_pt_path = None

        if os.path.isdir(edited_dir):
            for f in os.listdir(edited_dir):
                if f.endswith(".pt"):
                    edit_pt_path = os.path.join(edited_dir, f)
                    break

        if (edit_pt_path is None) and args.model.endswith(".pt") and os.path.isfile(args.model):
            edit_pt_path = args.model

        if edit_pt_path is None:
            raise FileNotFoundError(f"No .pt edit file found under: {edited_dir}")

        print(f"[Edited model detected] base={base_model_name}")
        print(f"  edits: {edit_pt_path}")

        model_for_tokenizer_id = base_model_path
        edit_pack = torch.load(edit_pt_path, map_location="cpu")
        base_for_dirs = base_model_name
    else:
        base_model_name = args.model
        model_for_tokenizer_id = MODEL_MAP.get(args.model, args.model)
        base_for_dirs = base_model_name

    # -------------------------
    # Load tokenizer/config from base
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_for_tokenizer_id)
    base_cfg = AutoConfig.from_pretrained(model_for_tokenizer_id)

    # If edited: materialize edited model into temp dir for vLLM
    tmpdir_handle = None
    if is_edited and edit_pack is not None:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_for_tokenizer_id,
            torch_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        apply_rank1_edits_to_hf_model(hf_model, edit_pack)

        tmpdir_handle = tempfile.TemporaryDirectory(prefix=f"edited_{base_for_dirs}_{os.getpid()}_")
        edited_path = tmpdir_handle.name

        hf_model.save_pretrained(edited_path, safe_serialization=True)
        tokenizer.save_pretrained(edited_path)
        base_cfg.save_pretrained(edited_path)
        print(f"✅ Saved edited model to temp folder: {edited_path}")

        atexit.register(lambda: tmpdir_handle.cleanup())
        model_for_vllm = edited_path
    else:
        model_for_vllm = model_for_tokenizer_id

    # -------------------------
    # Init vLLM
    # -------------------------
    llm = LLM(
        model=model_for_vllm,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_logprobs=1000,
        dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16,
    )

    # Optional: steering hooks
    if args.do_steering:
        if not args.direction_path:
            args.direction_path = f"directions/{base_for_dirs}_truthful_dirs.pt"
        if not os.path.exists(args.direction_path):
            raise FileNotFoundError(f"Direction checkpoint not found at {args.direction_path}.")
        print(f"[Steering] Loaded direction file from: {args.direction_path}")

        steering = partial(
            register_steering,
            direction_path=args.direction_path,
            alpha=args.alpha,
            components_spec=args.components_spec,
        )
        llm.apply_model(steering)

    # -------------------------
    # Evaluate (dispatch)
    # -------------------------
    if args.dataset == "truthfulqa":
        aggregate, records = eval_truthfulqa_score(
            llm=llm,
            tokenizer=tokenizer,
            ds=ds,
            prompt_logprobs_k=args.prompt_logprobs_k,
            batch_size=args.batch_size,
            limit=args.limit,
        )
    elif args.dataset == "factor_news":
        aggregate, records = eval_factor_news_score(
            llm=llm,
            tokenizer=tokenizer,
            rows=ds_factor_rows,
            prompt_logprobs_k=args.prompt_logprobs_k,
            batch_size=args.batch_size,
            limit=args.limit,
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented yet.")

    # Print a tiny summary of top-k misses (optional, but helpful)
    if _MISSING_TOPK:
        top = _MISSING_TOPK.most_common(10)
        print("[WARN] token_id not-in-topk counts (top 10):", top)

    summary = {
        "dataset": args.dataset,
        "mode": "score_avg_logprob",
        "model": args.model,
        "do_steering": bool(args.do_steering),
        "alpha": float(args.alpha),
        "components_spec": args.components_spec,
        "direction_path": args.direction_path if args.do_steering else "",
        "prompt_logprobs_k": int(args.prompt_logprobs_k),
        "n_examples": int(len(records)),
        "aggregate": aggregate,
        "records": records,
    }

    # -------------------------
    # Save
    # -------------------------
    model_out_name = (
        f"{args.model}"
        + (f"_steering_{args.alpha}_{args.components_spec}" if args.do_steering else "")
    )
    out_path = os.path.join(args.out_dir, args.dataset, model_out_name, "1_run.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Accuracy: {aggregate['accuracy']:.2f}%  (N={len(records)})")
    print(f"Mean margin: {aggregate['mean_margin']:.4f} ± {aggregate['std_margin']:.4f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
