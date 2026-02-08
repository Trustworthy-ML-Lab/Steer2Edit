import os
import json
import re
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoConfig, GenerationConfig, AutoTokenizer
from utils import MODEL_MAP, DATASET_MAP, extract_answer, verify_answer

# ----------------------
# Argument parsing
# ----------------------
parser = argparse.ArgumentParser(
    description="Evaluate thinking length of LLM responses on GSM8K using offline vLLM"
)
parser.add_argument(
    "--model", type=str,
    default="qwen3-4b-thinking",
    help="Model to evaluate"
)
parser.add_argument(
    "--dataset", type=str,
    default="gsm8k",
    help="Dataset to evaluate"
)
parser.add_argument(
    "--probe_path",
    type=str,
    default="data/gsm8k_probe.jsonl",
    help="Local probe dataset (JSONL with {question, answer}); used if file exists",
)
parser.add_argument(
    "--max_samples",
    type=int,
    default=-1,
    help="Optional cap on number of probe examples",
)
parser.add_argument(
    "--n_sample", 
    type=int, 
    default=1, 
    help="Number of samples to generate per prompt"
)
parser.add_argument(
    "--tensor_parallel_size", type=int,
    default=4,
    help="Tensor parallel size for vLLM"
)

args = parser.parse_args()

# Set random seed for reproducibility
np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

def apply_chat(prompt: str, tokenizer):
    """
    Wraps a user prompt in the vLLM chat template.
    """
    has_template = getattr(tokenizer, "chat_template", None)
    if has_template:
        conversations = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        return (
            f"{prompt}\n\n"
            f"Answer:"
        )

def make_params(n: int, budget: int, cfg: GenerationConfig) -> SamplingParams:
    """
    Build SamplingParams from model config and given budget.
    If bad_reasoning is True, override with model-specific high-randomness settings.
    """
    kw = {"n": n, "max_tokens": budget}

    # Respect model's generation_config if present
    if hasattr(cfg, "temperature") and cfg.temperature is not None:
        kw["temperature"] = float(cfg.temperature)
    if hasattr(cfg, "top_k") and cfg.top_k is not None:
        kw["top_k"] = int(cfg.top_k)
    if hasattr(cfg, "top_p") and cfg.top_p is not None:
        kw["top_p"] = float(cfg.top_p)

    return SamplingParams(**kw)

# ----------------------
# Load dataset
# ----------------------
questions = []
answers = []

if args.probe_path and os.path.isfile(args.probe_path):
    with open(args.probe_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            questions.append(row["question"])
            answers.append(row["answer"])
else:
    dataset_name, _ = DATASET_MAP[args.dataset]["args"]
    ds = load_dataset(dataset_name, split="train")
    question_key = DATASET_MAP[args.dataset]["question_key"]
    answer_key = DATASET_MAP[args.dataset]["answer_key"]
    for ex in ds:
        questions.append(ex[question_key])
        answers.append(ex[answer_key])

if args.max_samples > 0:
    questions = questions[:args.max_samples]
    answers = answers[:args.max_samples]

model_id  = MODEL_MAP[args.model]
max_pos = AutoConfig.from_pretrained(model_id).max_position_embeddings
cfg = GenerationConfig.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# ----------------------
# Prepare prompts with model-specific template
# ----------------------
prompts = [apply_chat(q, tokenizer) for q in questions]

# ----------------------
# Initialize vLLM offline LLM
# ----------------------
llm = LLM(
    model=model_id,
    tensor_parallel_size=args.tensor_parallel_size,
    max_model_len=max_pos,
    dtype=torch.bfloat16
)

# 4) Prepare sampling parameters
sampling_params = make_params(args.n_sample, 8192, cfg)

# ----------------------
# Helper to extract thinking section
# ----------------------
def extract_thinking(response_text):
    response_length = len(tokenizer(response_text, return_tensors='np')['input_ids'][0])
    match = re.search(r"(<think>.*?</think>)", response_text, re.DOTALL)

    if match:
        thinking = match.group(1).strip()
        thinking_length = len(tokenizer(thinking, return_tensors='np')['input_ids'][0])
        return thinking, int(response_length), int(thinking_length)
    return "", int(response_length), -1

# ----------------------
# Run offline batch inference
# ----------------------
print(f"Running offline batch inference on {len(prompts)} examples with model {args.model}...")
outputs = llm.generate(prompts, sampling_params)
# ----------------------
# Process outputs
# ----------------------
responses_data = []
for q, gold, batch_result in zip(questions, answers, outputs):
    text = batch_result.outputs[0].text.strip()
    thinking, response_length, thinking_length = extract_thinking(text)
    pred = extract_answer(text)
    # correctness
    correct = False
    try:
        correct = verify_answer(gold, pred)
    except:
        pass
    responses_data.append({
        "question": q,
        "response": text,
        "thinking": thinking,
        "response_length": response_length,
        "thinking_length": thinking_length,
        "prediction": pred,
        "gold": gold,
        "correct": correct
    })

# ----------------------
# Save results using original filenames
# ----------------------
os.makedirs(f"responses/", exist_ok=True)
json_path = f"responses/{args.model}_{args.dataset}_responses.json"
with open(json_path, 'w') as f:
    json.dump(responses_data, f, indent=4)
print(f"Saved JSON results to {json_path}")
