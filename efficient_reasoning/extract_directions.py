import os
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import MODEL_MAP
from tqdm import tqdm
import json

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
parser.add_argument("--model", type=str, default="deepseek-qwen-7b")
parser.add_argument("--percent", type=float, default=0.05, help="tail fraction for shortest/longest")
parser.add_argument("--only_correct", action="store_true", help="only use correct GSM8K samples")
parser.add_argument("--max_seq_len", type=int, default=4096, help="right-truncate full prompt+response to this many tokens")
args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ----------------------
# Chat wrapper (keep your original)
# ----------------------
def apply_chat(prompt: str, tokenizer, model_name: str = None) -> str:
    system_prompts = {
        "llama": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information.",
        "mistral": "Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.",
        "vicuna": "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    }

    alias_map = {
        "llama2-7b": "llama",
        "llama3-8b": "llama",
        "mistral-7b": "mistral",
        "vicuna-7b": "vicuna",
        "guanaco-7b": "vicuna",
        # deepseek/qwen typically have chat_template; sys_prompt usually ignored
    }

    base_key = alias_map.get(model_name, None)
    sys_prompt = system_prompts.get(base_key, None)

    has_template = getattr(tokenizer, "chat_template", None)
    if has_template:
        conversations = []
        if sys_prompt:
            conversations.append({"role": "system", "content": sys_prompt})
        conversations.append({"role": "user", "content": prompt})

        return tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        prefix = f"System: {sys_prompt}\n\n" if sys_prompt else ""
        return f"{prefix}{prompt}\n\nResponse:"

# ----------------------
# Load GSM8K responses JSON
# ----------------------
resp_json = f"responses/{args.model}_gsm8k_responses.json"
with open(resp_json, "r") as f:
    data = json.load(f)

# ----------------------
# Load model/tokenizer
# ----------------------
model_path = MODEL_MAP[args.model]
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# ----------------------
# Split into 5% shortest / 5% longest by token length of FULL response
# Prefer JSON's response_length (already tokens); fallback to tokenizer.
# ----------------------
def get_resp_len_tokens(ex):
    L = ex.get("response_length", None)
    if isinstance(L, int) and L >= 0:
        return L
    r = ex.get("response", "")
    if not isinstance(r, str):
        r = str(r)
    return len(tokenizer(r, add_special_tokens=False).input_ids)

rows = []
for i, ex in enumerate(data):
    if args.only_correct and not ex.get("correct", False):
        continue
    if not isinstance(ex.get("question", None), str):
        continue
    if not isinstance(ex.get("response", None), str):
        continue
    L = get_resp_len_tokens(ex)
    if L <= 0:
        continue
    rows.append((i, L))

if len(rows) < 20:
    raise RuntimeError(f"Too few usable samples: {len(rows)} (try without --only_correct)")

rows.sort(key=lambda x: x[1])  # ascending
n = len(rows)
k = max(1, int(np.floor(args.percent * n)))

short_ids = [idx for idx, _ in rows[:k]]
long_ids  = [idx for idx, _ in rows[-k:]]

short_data = [data[i] for i in short_ids]
long_data  = [data[i] for i in long_ids]

# efficient direction = SHORT - LONG, keep direction = pos - neg => POS=SHORT, NEG=LONG
pos_questions = [ex["question"] for ex in short_data]
pos_traces    = [ex["response"] for ex in short_data]
neg_questions = [ex["question"] for ex in long_data]
neg_traces    = [ex["response"] for ex in long_data]

print(f"Loaded: {resp_json}")
print(f"Usable: {n} | percent={args.percent} => k={k}")
print(f"Using SHORT(pos)={len(pos_traces)} LONG(neg)={len(neg_traces)} | only_correct={args.only_correct}")
print(f"Len(min/median/max) = {rows[0][1]} / {rows[n//2][1]} / {rows[-1][1]}")
print(f"Cutoffs: short_max={rows[k-1][1]} long_min={rows[-k][1]}")
print(f"Max seq len cap (RIGHT trunc): {args.max_seq_len}")

# ----------------------
# Layers
# ----------------------
layers = model.model.layers
num_layers = len(layers)

# ----------------------
# Hook collectors (unified pre/post)
# ----------------------
attn_o_proj_pre,  attn_o_proj_post  = [], []
mlp_down_proj_pre, mlp_down_proj_post = [], []

def make_io_collector(pre_list, post_list):
    def hook_fn(module, inputs, output):
        pre_list.append(inputs[0].detach())
        post_list.append(output.detach())
    return hook_fn

attn_hooks = []
mlp_hooks  = []
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
    """
    collected: list length = num_layers, each tensor [B, S, D]
    Returns: [num_layers, D] averaged over tokens in [start-1:end-1)
    (kept identical to your script)
    """
    stk = torch.stack(collected, dim=0)                        # [L, B, S, D]
    span = stk[:, :, start-1:end-1, :].mean(dim=2).squeeze(1)  # [L, D]
    return span

def collect_embeddings(q_list, t_list):
    attn_post_list, attn_pre_list = [], []
    mlp_post_list,  mlp_pre_list  = [], []

    for q, t in tqdm(zip(q_list, t_list), total=len(q_list), ncols=80):
        base_str = apply_chat(q, tokenizer, args.model).strip()

        # Tokenize base and full explicitly so we can right-truncate safely
        base_ids = tokenizer(base_str, add_special_tokens=False).input_ids
        full_ids = tokenizer(base_str + t, add_special_tokens=False).input_ids

        start = len(base_ids)   # boundary between prompt and response
        end = len(full_ids)

        # RIGHT truncation: keep prefix (prompt + beginning of response)
        if end > args.max_seq_len:
            full_ids = full_ids[:args.max_seq_len]
            end = len(full_ids)

        # If truncation removed most/all of response, the "response span" might be empty
        if end <= start:
            print(f"[warn] empty span after right-truncation; start={start} end={end}")
            # append zeros without doing forward to save memory
            d = model.model.layers[0].self_attn.o_proj.weight.shape[0]
            zeros = torch.zeros((num_layers, d), dtype=torch.float32)
            attn_post_list.append(zeros)
            attn_pre_list.append(zeros)
            mlp_post_list.append(zeros)
            mlp_pre_list.append(zeros)
            continue

        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, device=device)

        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

        attn_post_mean = span_mean_per_layer(attn_o_proj_post, start, end).cpu()
        attn_pre_mean  = span_mean_per_layer(attn_o_proj_pre,  start, end).cpu()
        mlp_post_mean  = span_mean_per_layer(mlp_down_proj_post, start, end).cpu()
        mlp_pre_mean   = span_mean_per_layer(mlp_down_proj_pre,  start, end).cpu()

        attn_post_list.append(attn_post_mean)
        attn_pre_list.append(attn_pre_mean)
        mlp_post_list.append(mlp_post_mean)
        mlp_pre_list.append(mlp_pre_mean)

        del attn_o_proj_post[:]
        del attn_o_proj_pre[:]
        del mlp_down_proj_post[:]
        del mlp_down_proj_pre[:]

    return attn_post_list, attn_pre_list, mlp_post_list, mlp_pre_list

# ----------------------
# Capture embeddings
# ----------------------
print(f"\n[1/2] Capturing SHORT (POS, efficient) embeddings ({len(pos_questions)} samples)...")
pos_attn_post, pos_attn_pre, pos_mlp_post, pos_mlp_pre = collect_embeddings(pos_questions, pos_traces)

print(f"\n[2/2] Capturing LONG (NEG, inefficient) embeddings ({len(neg_questions)} samples)...")
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
# Build directions dict
#   direction = POS - NEG = SHORT - LONG  (efficient reasoning)
# ----------------------
directions = {
    "pre":  {"pos_mean": {}, "neg_mean": {}, "direction": {}},
    "post": {"pos_mean": {}, "neg_mean": {}, "direction": {}},
    "meta": {
        "model": args.model,
        "dataset": "gsm8k",
        "responses": resp_json,
        "percent": args.percent,
        "only_correct": args.only_correct,
        "n_usable": n,
        "n_short": len(short_ids),
        "n_long": len(long_ids),
        "cutoff_short_max": rows[k-1][1],
        "cutoff_long_min": rows[-k][1],
        "direction_def": "SHORT - LONG (efficient reasoning)",
        "max_seq_len": args.max_seq_len,
        "truncation": "right",
    }
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
# Save
# ----------------------
os.makedirs("directions/", exist_ok=True)
out_path = f"directions/{args.model}_efficient_reasoning_dirs.pt"
torch.save(directions, out_path)
print(f"Saved means and directions (grouped by pre/post) to: {out_path}")

# ----------------------
# Cleanup hooks (optional)
# ----------------------
for h in attn_hooks + mlp_hooks:
    h.remove()
