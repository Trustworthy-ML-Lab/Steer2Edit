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
parser.add_argument("--model", type=str, default="llama3-8b")
args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ----------------------
# Chat wrapper
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

    # Map variants to base keys
    alias_map = {
        "llama2-7b": "llama",
        "llama3-8b": "llama",
        "mistral-7b": "mistral",
        "vicuna-7b": "vicuna",
        "guanaco-7b": "vicuna",
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
        return (
            f"{prefix}{prompt}\n\n"
            f"Response:"
        )

# ----------------------
# Load response JSONs (negative vs positive)
#   - We treat 'positive' as the behavior you want (previously 'target', e.g., correct)
#   - We treat 'negative' as the behavior you don't want (previously 'original', e.g., incorrect)
# ----------------------
neg_json = f"responses/{args.model}_alpaca.json"   # negative behavior examples
pos_json = f"responses/{args.model}_malicious.json"       # positive behavior examples

with open(neg_json, "r") as f:
    negative_data = json.load(f)
with open(pos_json, "r") as f:
    positive_data = json.load(f)

questions   = [ex["question"] for ex in positive_data]
pos_traces  = [ex["response"] for ex in positive_data]
neg_traces  = [ex["response"] for ex in negative_data]

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
        # nn.Linear forward: inputs is a single-tensor tuple
        pre_list.append(inputs[0].detach())
        post_list.append(output.detach())
    return hook_fn

# Register a single hook per target module
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
    """
    stk = torch.stack(collected, dim=0)                        # [L, B, S, D]
    span = stk[:, :, start-1:end-1, :].mean(dim=2).squeeze(1)  # [L, D]
    return span

def collect_embeddings(q_list, t_list):
    """
    For each (question, trace), run one forward and return:
      attn_post_list, attn_pre_list, mlp_post_list, mlp_pre_list
    Each element is [num_layers, D]
    """
    attn_post_list, attn_pre_list = [], []
    mlp_post_list,  mlp_pre_list  = [], []

    for q, t in tqdm(zip(q_list, t_list), total=len(q_list), ncols=80):
        base_str = apply_chat(q, tokenizer, args.model).strip()
        start = len(tokenizer(base_str).input_ids)
        full_str = base_str + t
        end = len(tokenizer(full_str).input_ids)

        toks = tokenizer(full_str, return_tensors="pt")
        with torch.no_grad():
            _ = model(
                input_ids=toks["input_ids"].to(device),
                attention_mask=toks["attention_mask"].to(device)
            )

        if end <= start:
            print(f"[warn] empty span for sample; start={start} end={end}")

        # per-layer span means
        attn_post_mean = span_mean_per_layer(attn_o_proj_post, start, end).cpu()
        attn_pre_mean  = span_mean_per_layer(attn_o_proj_pre,  start, end).cpu()
        mlp_post_mean  = span_mean_per_layer(mlp_down_proj_post, start, end).cpu()
        mlp_pre_mean   = span_mean_per_layer(mlp_down_proj_pre,  start, end).cpu()

        attn_post_list.append(attn_post_mean)
        attn_pre_list.append(attn_pre_mean)
        mlp_post_list.append(mlp_post_mean)
        mlp_pre_list.append(mlp_pre_mean)

        # clear buffers for next sample
        del attn_o_proj_post[:]
        del attn_o_proj_pre[:]
        del mlp_down_proj_post[:]
        del mlp_down_proj_pre[:]

    return attn_post_list, attn_pre_list, mlp_post_list, mlp_pre_list

# ----------------------
# Capture embeddings
# ----------------------
print(f"\n[1/2] Capturing POSITIVE embeddings ({len(questions)} samples)...")
pos_attn_post, pos_attn_pre, pos_mlp_post, pos_mlp_pre = collect_embeddings(questions, pos_traces)

print(f"\n[2/2] Capturing NEGATIVE embeddings ({len(questions)} samples)...")
neg_attn_post, neg_attn_pre, neg_mlp_post, neg_mlp_pre = collect_embeddings(questions, neg_traces)

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
# Build directions dict with top-level 'pre' and 'post'
#   - Store pos_mean, neg_mean, and direction = pos_mean - neg_mean
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
# Sanity print
# ----------------------
print("Example (layer 0):")
print("  o_proj pre  pos_mean:", directions['pre']['pos_mean']['model.layers[0].self_attn.o_proj'].size())
print("  o_proj pre  neg_mean:", directions['pre']['neg_mean']['model.layers[0].self_attn.o_proj'].size())
print("  o_proj pre  direction:", directions['pre']['direction']['model.layers[0].self_attn.o_proj'].size())
print("  o_proj post pos_mean:", directions['post']['pos_mean']['model.layers[0].self_attn.o_proj'].size())
print("  o_proj post neg_mean:", directions['post']['neg_mean']['model.layers[0].self_attn.o_proj'].size())
print("  o_proj post direction:", directions['post']['direction']['model.layers[0].self_attn.o_proj'].size())

print("  down_proj pre  pos_mean:", directions['pre']['pos_mean']['model.layers[0].mlp.down_proj'].size())
print("  down_proj pre  neg_mean:", directions['pre']['neg_mean']['model.layers[0].mlp.down_proj'].size())
print("  down_proj pre  direction:", directions['pre']['direction']['model.layers[0].mlp.down_proj'].size())
print("  down_proj post pos_mean:", directions['post']['pos_mean']['model.layers[0].mlp.down_proj'].size())
print("  down_proj post neg_mean:", directions['post']['neg_mean']['model.layers[0].mlp.down_proj'].size())
print("  down_proj post direction:", directions['post']['direction']['model.layers[0].mlp.down_proj'].size())

# ----------------------
# Save
# ----------------------
os.makedirs(f"directions/", exist_ok=True)
out_path = f"directions/{args.model}_refusal_dirs.pt"
torch.save(directions, out_path)
print(f"Saved means and directions (grouped by pre/post) to: {out_path}")

# ----------------------
# Cleanup hooks (optional)
# ----------------------
for h in attn_hooks + mlp_hooks:
    h.remove()
