import os
import json
import argparse
import numpy as np
import re
from functools import partial
import torch
import tempfile
import atexit
from datasets import load_dataset
from transformers import (
    AutoConfig,
    GenerationConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)

os.environ["VLLM_USE_V1"] = "0"
from vllm import LLM, SamplingParams
from utils import extract_answer, verify_answer, DATASET_MAP, MODEL_MAP


# Set a default budget for reasoning tokens
DEFAULT_BUDGET = 512

def apply_chat(prompt: str, tokenizer):
    """
    Wraps a user prompt in the vLLM chat template.
    """
    conversations = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=True,
    )


def make_params(n: int, budget: int, cfg) -> SamplingParams:
    """
    Build SamplingParams from model config and given budget.
    """
    kw = {"n": n, "max_tokens": budget}
    if hasattr(cfg, "temperature") and cfg.temperature is not None:
        kw["temperature"] = cfg.temperature
    if hasattr(cfg, "top_k") and cfg.top_k is not None:
        kw["top_k"] = cfg.top_k
    if hasattr(cfg, "top_p") and cfg.top_p is not None:
        kw["top_p"] = cfg.top_p
    return SamplingParams(**kw)


class LinearInterventionHook:
    def __init__(self, direction, weight):
        self.direction = direction
        self.weight = weight

    def __call__(self, module, input, output):
        # align dtype/device
        ref = output[0] if isinstance(output, tuple) else output
        self.direction = self.direction.type_as(ref)
        if isinstance(output, tuple):
            # reconstruct tuple: (modified_first, *rest)
            modified = ref + self.direction.to(ref.device) * self.weight
            return (modified,) + output[1:]
        else:
            return output + self.direction.to(output.device) * self.weight


def add_steering(model, directions, weight, components=None):
    if not components:
        return
    for component in components:
        if component not in directions:
            # silently skip if direction not provided for this component
            continue
        steering_vector = directions[component]
        hook = LinearInterventionHook(steering_vector, weight)
        eval(f"model.{component}.register_forward_hook(hook)")


def parse_components_spec(spec: str, num_layers: int):
    """
    Parse strings like:
      - 'attn0-1' (layers [0])
      - 'mlp1-3'  (layers [1,2])
      - 'attn'    (all attn layers)
      - 'mlp'     (all mlp layers)
      - 'attn0-1mlp1-3' (both)
      - 'attnmlp' (everything)
    Returns: sorted unique list of component paths:
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
    add_steering(
        hf_model,
        directions=directions,
        weight=alpha,
        components=comps,
    )


# ---------- NEW: rank-1 edits applied to a plain HF model on CPU ----------
@torch.no_grad()
def apply_rank1_edits_to_hf_model(hf_model, edit_pack: dict) -> None:
    """
    Mutate HF model in-place (unsharded).
      attn: ΔW[:, head_slice] += λ * vhat @ k_hat^T
      mlp : ΔW[:, neuron_idx] += λ * vhat
    """
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

    vhat_attn_meta = edit_pack["meta"]["vhat_attn"]  # {layer:int: [H]}
    vhat_mlp_meta = edit_pack["meta"]["vhat_mlp"]  # {layer:int: [H]}

    # ATTENTION o_proj: [H, H]
    for (layer_idx, head_idx), payload in edit_pack.get("attn", {}).items():
        lam = float(payload["lambda"])
        if lam == 0.0:
            continue
        oW = layers[layer_idx].self_attn.o_proj.weight  # [H, H]
        dtype = oW.dtype
        vhat = vhat_attn_meta[layer_idx].to(device=oW.device, dtype=dtype)  # [H]
        khat = payload["k_hat"].to(device=oW.device, dtype=dtype)  # [D]
        h0 = head_idx * D
        h1 = h0 + D
        delta = lam * torch.outer(vhat, khat)  # [H, D]
        oW.data[:, h0:h1].add_(delta)

    # MLP down_proj: [H, I]
    for (layer_idx, neuron_idx), payload in edit_pack.get("mlp", {}).items():
        lam = float(payload["lambda"])
        if lam == 0.0:
            continue
        dW = layers[layer_idx].mlp.down_proj.weight  # [H, I]
        dtype = dW.dtype
        vhat = vhat_mlp_meta[layer_idx].to(device=dW.device, dtype=dtype)  # [H]
        dW.data[:, neuron_idx].add_(lam * vhat)


def main():
    parser = argparse.ArgumentParser(
        description="Inference with optional steering vectors; outputs JSON with per-run stats"
    )
    parser.add_argument("--dataset", choices=DATASET_MAP.keys(), default="gsm8k")
    parser.add_argument("--model", default="llama3-8b")
    parser.add_argument("--n_sample", type=int, default=10)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)

    # ------------ Steering toggles ------------
    parser.add_argument(
        "--do_steering",
        action="store_true",
        help="If set, register steering hooks on the vLLM model before generation.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="steering strength.",
    )
    parser.add_argument(
        "--components_spec",
        type=str,
        default="attnmlp",
        help="Compact spec like 'attn0-1mlp1-3', 'attn', 'mlp', or 'attnmlp'.",
    )
    # ---------------------------------------------------------

    args = parser.parse_args()

    # 1) Load dataset
    dataset_name, split = DATASET_MAP[args.dataset]["args"]
    dataset_args = dataset_name if isinstance(dataset_name, tuple) else (dataset_name,)
    ds = load_dataset(*dataset_args, split=split)
    question_key = DATASET_MAP[args.dataset]["question_key"]
    answer_key = DATASET_MAP[args.dataset]["answer_key"]

    if args.dataset == "AIME2024":
        override_28 = r"""Torus $T$ is the surface produced by revolving a circle with radius $3$ around an axis in the plane of the circle that is a distance $6$ from the center of the circle (so like a donut). Let $S$ be a sphere with a radius $11$. When $T$ rests on the inside of $S$, it is internally tangent to $S$ along a circle with radius $r_i$, and when $T$ rests on the outside of $S$, it is externally tangent to $S$ along a circle with radius $r_o$. The difference $r_i-r_o$ can be written as $\tfrac{m}{n}$, where $m$ and $n$ are relatively prime positive integers. Find $m+n$. 
[asy] unitsize(0.3 inch); draw(ellipse((0,0), 3, 1.75)); draw((-1.2,0.1)..(-0.8,-0.03)..(-0.4,-0.11)..(0,-0.15)..(0.4,-0.11)..(0.8,-0.03)..(1.2,0.1)); draw((-1,0.04)..(-0.5,0.12)..(0,0.16)..(0.5,0.12)..(1,0.04)); draw((0,2.4)--(0,-0.15)); draw((0,-0.15)--(0,-1.75), dashed); draw((0,-1.75)--(0,-2.25)); draw(ellipse((2,0), 1, 0.9)); draw((2.03,-0.02)--(2.9,-0.4)); [/asy]"""
        ds = ds.map(
            lambda example, idx: {"problem": override_28} if idx == 28 else example,
            with_indices=True,
        )

    # -------------------------------
    # Detect edited_models/ path and locate edit pack
    # -------------------------------
    parts = args.model.strip("/").split("/")
    is_edited = len(parts) >= 3 and parts[0] == "edited_models"
    if is_edited:
        # Example: edited_models/llama3-8b/rank1__...
        base_model_name = parts[1]
        base_model_path = MODEL_MAP.get(base_model_name, base_model_name)
        edited_dir = os.path.join(*parts[:3])

        # Find a .pt edit file inside edited_dir (or allow passing a .pt path directly)
        edit_pt_path = None
        if os.path.isdir(edited_dir):
            for f in os.listdir(edited_dir):
                if f.endswith(".pt"):
                    edit_pt_path = os.path.join(edited_dir, f)
                    break
        if (edit_pt_path is None) and args.model.endswith(".pt") and os.path.isfile(
            args.model
        ):
            edit_pt_path = args.model

        if edit_pt_path is None:
            raise FileNotFoundError(f"No .pt edit file found under: {edited_dir}")

        print(f"[Edited model detected] base={base_model_name}")
        print(f"  edits: {edit_pt_path}")

        model_for_tokenizer_id = base_model_path  # base model for cfg/tokenizer/HF
        edit_pack = torch.load(edit_pt_path, map_location="cpu")
    else:
        base_model_name = args.model
        model_for_tokenizer_id = MODEL_MAP.get(args.model, args.model)
        edit_pack = None  # no edits

    # 2) Load config/tokenizer from base
    # Patch older Mistral configs missing head_dim
    if base_model_name == "mistral-7b":
        _orig_from_pretrained = AutoConfig.from_pretrained

        def patched_from_pretrained(*args, **kwargs):
            res = _orig_from_pretrained(*args, **kwargs)

            # HF sometimes returns (config, unused_kwargs)
            if isinstance(res, tuple):
                config, unused = res
            else:
                config, unused = res, None

            # Patch head_dim if missing
            if getattr(config, "head_dim", None) is None and getattr(config, "hidden_size", None) is not None:
                config.head_dim = config.hidden_size // config.num_attention_heads

            # Preserve original return type
            return (config, unused) if unused is not None else config

        AutoConfig.from_pretrained = patched_from_pretrained
        

    max_pos = AutoConfig.from_pretrained(
        model_for_tokenizer_id
    ).max_position_embeddings
    if base_model_name == "deepseek-qwen3-8b":
        cfg = GenerationConfig.from_pretrained("deepseek-ai/DeepSeek-R1-0528")
    else:
        cfg = GenerationConfig.from_pretrained(model_for_tokenizer_id)
    tokenizer = AutoTokenizer.from_pretrained(model_for_tokenizer_id)

    # 3) If edited: build an edited HF model in a **unique TemporaryDirectory**
    #    Otherwise, point vLLM to the base model id/path directly.
    tmpdir_handle = None
    if is_edited and edit_pack is not None:
        base_cfg = AutoConfig.from_pretrained(model_for_tokenizer_id)
        if (
            getattr(base_cfg, "head_dim", None) is None
            and getattr(base_cfg, "num_attention_heads", None) is not None
        ):
            base_cfg.head_dim = base_cfg.hidden_size // base_cfg.num_attention_heads

        hf_model = AutoModelForCausalLM.from_pretrained(
            model_for_tokenizer_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )

        # Apply rank-1 edits in place
        apply_rank1_edits_to_hf_model(hf_model, edit_pack)

        # Create a per-process unique temp dir; clean up at exit
        tmpdir_handle = tempfile.TemporaryDirectory(
            prefix=f"edited_{base_model_name}_{os.getpid()}_"
        )
        edited_path = tmpdir_handle.name

        # Save edited weights/config/tokenizer for vLLM to load
        hf_model.save_pretrained(edited_path, safe_serialization=True)
        tokenizer.save_pretrained(edited_path)
        base_cfg.save_pretrained(edited_path)
        print(f"✅ Saved edited model to temp folder: {edited_path}")

        # Ensure cleanup even on normal exit
        atexit.register(lambda: tmpdir_handle.cleanup())

        model_for_vllm = edited_path
    else:
        model_for_vllm = model_for_tokenizer_id

    # 4) Initialize vLLM with the (possibly edited) folder
    llm = LLM(
        model=model_for_vllm,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=min(DEFAULT_BUDGET + 2048, max_pos),
        dtype=torch.bfloat16,
    )

    # 4.5) (Optional) Register steering hooks AFTER edits (if requested)
    if args.do_steering:
        base_for_dirs = base_model_name if is_edited else args.model
        direction_ckpt_path = f"directions/{base_for_dirs}_truthful_dirs.pt"
        if not os.path.exists(direction_ckpt_path):
            raise FileNotFoundError(
                f"Direction checkpoint not found at {direction_ckpt_path}. "
                "Make sure you have precomputed and saved directions with that name."
            )
        print(f"[Steering] Loaded direction file from: {direction_ckpt_path}")

        steering = partial(
            register_steering,
            direction_path=direction_ckpt_path,
            alpha=args.alpha,
            components_spec=args.components_spec,
        )
        llm.apply_model(steering)

    # 5) Prepare sampling parameters
    sampling_params = make_params(
        args.n_sample, min(DEFAULT_BUDGET, max_pos - 512), cfg
    )

    # 6) Build prompts
    prompts = []
    for ex in ds:
        prompt = ex[question_key]
        if args.dataset == "commonsense":
            for option, content in zip(ex["choices"]["label"], ex["choices"]["text"]):
                prompt += f"\n{option}. {content}"
            prompt += (
                "\nPlease write your final answer in the form of "
                "\\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, \\boxed{E}"
            )
        elif args.dataset == "arc-challenge":
            for option, content in zip(ex["choices"]["label"], ex["choices"]["text"]):
                prompt += f"\n{option}. {content}"
            prompt += (
                "\nPlease write your final answer in the form of "
                "\\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}"
            )
        elif args.dataset == "code-mmlu":
            instruction = (
                "\n\nPlease choose the correct option to complete the below code.\n\n"
                "Code to complete:\n\n"
            )
            prompt = instruction + ex[question_key] + "\n\n"
            for option, content in zip(["A", "B", "C", "D"], ex["choices"]):
                prompt += f"Option {option}:\n\n{content}\n\n"
            prompt += (
                "\nPlease write your final answer in the form of "
                "\\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}"
            )
        prompts.append(apply_chat(prompt, tokenizer))

    # 7) Generate
    results = llm.generate(prompts=prompts, sampling_params=sampling_params)

    # 8) Collect runs and compute stats
    runs = {rid: [] for rid in range(args.n_sample)}
    for idx, gen in enumerate(results):
        gold = ds[idx][answer_key]
        if args.dataset == "gpqa":
            gold = extract_answer(gold)
        for rid, out in enumerate(gen.outputs):
            text = out.text.strip()
            pred = extract_answer(text)

            # ---- Fallback for multiple-choice datasets ----
            if args.dataset in ("commonsense", "code-mmlu"):
                # valid choices depend on dataset
                if args.dataset == "commonsense":
                    valid_choices = ["A", "B", "C", "D", "E"]
                else:  # code-mmlu
                    valid_choices = ["A", "B", "C", "D"]

                # if extract_answer didn't give a clean choice label, fall back
                if pred not in valid_choices:
                    # find first standalone A/B/C/D(/E) in the text
                    m = re.search(r"\b([A-E])\b", text)
                    if m and m.group(1) in valid_choices:
                        pred = m.group(1)
                        
            correct = False
            try:
                correct = verify_answer(gold, pred)
            except Exception:
                pass
            reasoning_length = len(
                tokenizer.encode(text, add_special_tokens=False)
            )

            runs[rid].append(
                {
                    "question": ds[idx][
                        DATASET_MAP[args.dataset]["question_key"]
                    ],
                    "full_response": text,
                    "reasoning_length": reasoning_length,
                    "prediction": pred,
                    "gold": gold,
                    "correct": correct,
                }
            )

    summary = {"runs": [], "aggregate": {}}
    accs = []
    lengths = []
    for run_id, recs in runs.items():
        acc = sum(1 for r in recs if r["correct"]) / len(recs) * 100
        accs.append(acc)
        avg_len = sum(r["reasoning_length"] for r in recs) / len(recs)
        lengths.append(avg_len)
        summary["runs"].append(
            {
                "run_id": run_id,
                "accuracy": acc,
                "avg_length": avg_len,
                "records": recs,
            }
        )

    summary["aggregate"]["mean_accuracy"] = float(np.mean(accs))
    summary["aggregate"]["std_accuracy"] = (
        float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
    )
    summary["aggregate"]["mean_length"] = float(np.mean(lengths))
    summary["aggregate"]["std_length"] = (
        float(np.std(lengths, ddof=1)) if len(lengths) > 1 else 0.0
    )

    # 9) Save JSON (preserve original args.model string in path)
    model_out_name = f"{args.model}{f'_steering_{args.alpha}_{args.components_spec}' if args.do_steering else ''}"
    output_path = (
        f"evaluate_results/{args.dataset}/{model_out_name}/{args.n_sample}_runs.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"Per‐run accuracies: {accs}")
    print(
        f"Mean ± std accuracy: "
        f"{summary['aggregate']['mean_accuracy']:.2f}% "
        f"± {summary['aggregate']['std_accuracy']:.2f}%"
    )
    print(
        f"Mean ± std length: "
        f"{summary['aggregate']['mean_length']:.1f} "
        f"± {summary['aggregate']['std_length']:.1f} tokens"
    )
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
