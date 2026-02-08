#!/usr/bin/env python3
"""
Extract TruthfulQA "truthfulness" directions (pos - neg) on probing split.

- Prompt = dataset["question"]
- Positive trace = rows with parsed_label == "TRUE"
- Negative trace = rows with parsed_label == "FALSE"
- Ignore parsed_label == "NONE"
- No system prompt in chat template
- Save format UNCHANGED:
    directions["pre"/"post"]["pos_mean"/"neg_mean"/"direction"][module_key] = tensor[D]

Expected dataset columns:
  - id
  - question
  - answer
  - judge_output
  - parsed_label   ("TRUE"/"FALSE"/"NONE")
  - cand_idx
"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import MODEL_MAP
from transformers.utils import logging

logging.set_verbosity_error()

# ----------------------
# Reproducibility
# ----------------------
np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

# ----------------------
# Args
# ----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="qwen2.5-3b", choices=list(MODEL_MAP.keys()))
parser.add_argument("--max_samples", type=int, default=-1, help="For debugging; -1 means use all.")
parser.add_argument("--out_dir", type=str, default="directions")
args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# probe path derived from model name (as requested)
probe_path = os.path.join("data", "truthfulqa", "probe", args.model)

# ----------------------
# Chat wrapper (NO system prompt)
# ----------------------
def apply_chat(prompt: str, tokenizer) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{prompt}\n\nResponse:"

# ----------------------
# Load probing dataset
# ----------------------
ds = load_from_disk(probe_path)

pos_questions, pos_traces = [], []
neg_questions, neg_traces = [], []

for q, a, lab in zip(ds["question"], ds["answer"], ds["parsed_label"]):
    if lab == "TRUE":
        pos_questions.append(q)
        pos_traces.append(a)
    elif lab == "FALSE":
        neg_questions.append(q)
        neg_traces.append(a)

if args.max_samples > 0:
    pos_questions = pos_questions[:args.max_samples]
    pos_traces    = pos_traces[:args.max_samples]
    neg_questions = neg_questions[:args.max_samples]
    neg_traces    = neg_traces[:args.max_samples]

assert len(pos_questions) > 0, f"No TRUE samples found in {probe_path}"
assert len(neg_questions) > 0, f"No FALSE samples found in {probe_path}"

print(f"Loaded probe dataset from: {probe_path}")
print(f"Using {len(pos_questions)} TRUE and {len(neg_questions)} FALSE samples")

# ----------------------
# Load model/tokenizer
# ----------------------
model_path = MODEL_MAP[args.model]
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ----------------------
# Layers
# ----------------------
layers = model.model.layers
num_layers = len(layers)

# ----------------------
# Hook collectors (unified pre/post)
# ----------------------
attn_o_proj_pre, attn_o_proj_post = [], []
mlp_down_proj_pre, mlp_down_proj_post = [], []

def make_io_collector(pre_list, post_list):
    def hook_fn(module, inputs, output):
        pre_list.append(inputs[0].detach())
        post_list.append(output.detach())
    return hook_fn

attn_hooks, mlp_hooks = [], []
for lyr in layers:
    attn_hooks.append(
        lyr.self_attn.o_proj.register_forward_hook(
            make_io_collector(attn_o_proj_pre, attn_o_proj_post)
        )
    )
    mlp_hooks.append(
        lyr.mlp.down_proj.register_forward_hook(
            make_io_collector(mlp_down_proj_pre, mlp_down_proj_post)
        )
    )

# ----------------------
# Utilities
# ----------------------
def span_mean_per_layer(collected, start, end):
    stk = torch.stack(collected, dim=0)                        # [L,B,S,D]
    span = stk[:, :, start-1:end-1, :].mean(dim=2).squeeze(1)  # [L,D]
    return span

def collect_embeddings(q_list, t_list):
    attn_post_list, attn_pre_list = [], []
    mlp_post_list, mlp_pre_list = [], []

    for q, t in tqdm(zip(q_list, t_list), total=len(q_list), ncols=80):
        base_str = apply_chat(q, tokenizer).strip()
        start = len(tokenizer(base_str).input_ids)
        full_str = base_str + t
        end = len(tokenizer(full_str).input_ids)

        toks = tokenizer(full_str, return_tensors="pt")
        with torch.no_grad():
            _ = model(
                input_ids=toks["input_ids"].to(device),
                attention_mask=toks["attention_mask"].to(device),
            )

        if end <= start:
            print(f"[warn] empty span for sample; start={start} end={end}")

        attn_post_list.append(span_mean_per_layer(attn_o_proj_post, start, end).cpu())
        attn_pre_list.append(span_mean_per_layer(attn_o_proj_pre, start, end).cpu())
        mlp_post_list.append(span_mean_per_layer(mlp_down_proj_post, start, end).cpu())
        mlp_pre_list.append(span_mean_per_layer(mlp_down_proj_pre, start, end).cpu())

        del attn_o_proj_post[:]
        del attn_o_proj_pre[:]
        del mlp_down_proj_post[:]
        del mlp_down_proj_pre[:]

    return attn_post_list, attn_pre_list, mlp_post_list, mlp_pre_list

# ----------------------
# Capture embeddings
# ----------------------
print(f"\n[1/2] Capturing POSITIVE embeddings ({len(pos_questions)} samples) from TRUE answers...")
pos_attn_post, pos_attn_pre, pos_mlp_post, pos_mlp_pre = collect_embeddings(pos_questions, pos_traces)

print(f"\n[2/2] Capturing NEGATIVE embeddings ({len(neg_questions)} samples) from FALSE answers...")
neg_attn_post, neg_attn_pre, neg_mlp_post, neg_mlp_pre = collect_embeddings(neg_questions, neg_traces)

# ----------------------
# Means over samples: [num_layers, D]
# ----------------------
pos_attn_post_mean = torch.stack(pos_attn_post, dim=0).mean(dim=0)
pos_attn_pre_mean  = torch.stack(pos_attn_pre,  dim=0).mean(dim=0)
pos_mlp_post_mean  = torch.stack(pos_mlp_post,  dim=0).mean(dim=0)
pos_mlp_pre_mean   = torch.stack(pos_mlp_pre,   dim=0).mean(dim=0)

neg_attn_post_mean = torch.stack(neg_attn_post, dim=0).mean(dim=0)
neg_attn_pre_mean  = torch.stack(neg_attn_pre,  dim=0).mean(dim=0)
neg_mlp_post_mean  = torch.stack(neg_mlp_post,  dim=0).mean(dim=0)
neg_mlp_pre_mean   = torch.stack(neg_mlp_pre,   dim=0).mean(dim=0)

# ----------------------
# Build directions dict with top-level 'pre' and 'post' (SAME FORMAT)
# ----------------------
directions = {
    "pre":  {"pos_mean": {}, "neg_mean": {}, "direction": {}},
    "post": {"pos_mean": {}, "neg_mean": {}, "direction": {}},
}

for i in range(num_layers):
    attn_key = f"model.layers[{i}].self_attn.o_proj"
    mlp_key  = f"model.layers[{i}].mlp.down_proj"

    # POST
    directions["post"]["pos_mean"][attn_key] = pos_attn_post_mean[i]
    directions["post"]["neg_mean"][attn_key] = neg_attn_post_mean[i]
    directions["post"]["direction"][attn_key] = pos_attn_post_mean[i] - neg_attn_post_mean[i]

    directions["post"]["pos_mean"][mlp_key] = pos_mlp_post_mean[i]
    directions["post"]["neg_mean"][mlp_key] = neg_mlp_post_mean[i]
    directions["post"]["direction"][mlp_key] = pos_mlp_post_mean[i] - neg_mlp_post_mean[i]

    # PRE
    directions["pre"]["pos_mean"][attn_key] = pos_attn_pre_mean[i]
    directions["pre"]["neg_mean"][attn_key] = neg_attn_pre_mean[i]
    directions["pre"]["direction"][attn_key] = pos_attn_pre_mean[i] - neg_attn_pre_mean[i]

    directions["pre"]["pos_mean"][mlp_key] = pos_mlp_pre_mean[i]
    directions["pre"]["neg_mean"][mlp_key] = neg_mlp_pre_mean[i]
    directions["pre"]["direction"][mlp_key] = pos_mlp_pre_mean[i] - neg_mlp_pre_mean[i]

# ----------------------
# Save (SAME FORMAT)
# ----------------------
os.makedirs(args.out_dir, exist_ok=True)
out_path = os.path.join(args.out_dir, f"{args.model}_truthful_dirs.pt")
torch.save(directions, out_path)
print(f"Saved means and directions (grouped by pre/post) to: {out_path}")

# ----------------------
# Cleanup hooks
# ----------------------
for h in attn_hooks + mlp_hooks:
    h.remove()
