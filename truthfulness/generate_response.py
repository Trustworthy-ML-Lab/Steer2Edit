#!/usr/bin/env python3
"""
[1/3] Generate 1 candidate per TruthfulQA probe question (model's own generation).

- One inference pass (n=1) using model GenerationConfig (optionally override temp/top_p/top_k)
- Stores full correct_answers and incorrect_answers for downstream judging
"""

import os
import json
import argparse
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoConfig, GenerationConfig, AutoTokenizer

os.environ["VLLM_USE_V1"] = "0"
from vllm import LLM, SamplingParams

from utils import MODEL_MAP
from transformers.utils import logging
logging.set_verbosity_error()

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

DEFAULT_BUDGET = 128


def apply_chat(prompt: str, tokenizer):
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        return f"{prompt}\n\nResponse:"


def make_params_from_cfg(cfg: GenerationConfig, *, budget: int, n: int = 1,
                         temperature_override=None, top_p_override=None, top_k_override=None) -> SamplingParams:
    kw = {"n": n, "max_tokens": budget}

    if getattr(cfg, "temperature", None) is not None:
        kw["temperature"] = float(cfg.temperature)
    if getattr(cfg, "top_k", None) is not None:
        kw["top_k"] = int(cfg.top_k)
    if getattr(cfg, "top_p", None) is not None:
        kw["top_p"] = float(cfg.top_p)

    if temperature_override is not None:
        kw["temperature"] = float(temperature_override)
    if top_p_override is not None:
        kw["top_p"] = float(top_p_override)
    if top_k_override is not None:
        kw["top_k"] = int(top_k_override)

    return SamplingParams(**kw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-7b", choices=list(MODEL_MAP.keys()))
    parser.add_argument("--probe_path", default="data/truthfulqa/train")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--out_dir", default="responses")

    # optional overrides
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)

    args = parser.parse_args()

    ds = load_from_disk(args.probe_path)

    questions = ds["question"]
    correct_answers = ds["best_answer"]
    incorrect_answers = ds["incorrect_answers"]

    if args.max_samples > 0:
        questions = questions[:args.max_samples]
        correct_answers = correct_answers[:args.max_samples]
        incorrect_answers = incorrect_answers[:args.max_samples]

    model_id = MODEL_MAP[args.model]
    max_pos = AutoConfig.from_pretrained(model_id).max_position_embeddings
    cfg = GenerationConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    prompts = [apply_chat(q, tokenizer) for q in questions]

    llm = LLM(
        model=model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=min(args.budget + 512, max_pos),
        dtype=torch.bfloat16,
    )

    budget = min(args.budget, max_pos - 512)
    sampling_params = make_params_from_cfg(
        cfg,
        budget=budget,
        n=1,
        temperature_override=args.temperature,
        top_p_override=args.top_p,
        top_k_override=args.top_k,
    )

    print(f"Generating 1 candidate for {len(prompts)} questions...")
    outs = llm.generate(prompts=prompts, sampling_params=sampling_params)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.model}_truthfulqa_responses.json")

    rows = []
    for i, (q, ca, ia, out) in enumerate(zip(questions, correct_answers, incorrect_answers, outs)):
        text = (out.outputs[0].text or "").strip()
        rows.append({
            "id": i,
            "question": q,
            "correct_answers": ca if isinstance(ca, list) else [ca],
            "incorrect_answers": ia if isinstance(ia, list) else [],
            "candidate": text,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"Saved generations to {out_path}")


if __name__ == "__main__":
    main()
