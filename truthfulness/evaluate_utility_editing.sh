#!/usr/bin/env bash
set -euo pipefail

# ---------------- Config ----------------
MODEL="${MODEL:-gemma2-2b}"         # override via --model or MODEL=
MODEL_ROOT="${MODEL_ROOT:-}"        # optional; default set after parsing
N_SAMPLE=10
TENSOR_PARALLEL_SIZE=4
LOG_DIR="logs"

# Fixed hyper-params (evaluate a chosen edit)
# Use -1.0 to disable attn/mlp edits (keep naming convention)
RHO_ATTN="${RHO_ATTN:-0.3}"
RHO_MLP="${RHO_MLP:--1.0}"
ALPHA="${ALPHA:-0.75}"

DATASETS=(
  "gsm8k"
)
# ---------------------------------------

# -------- Parse args --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --model=*) MODEL="${1#*=}"; shift ;;
    --model-root) MODEL_ROOT="$2"; shift 2 ;;
    --model-root=*) MODEL_ROOT="${1#*=}"; shift ;;
    --rho-attn) RHO_ATTN="$2"; shift 2 ;;
    --rho-attn=*) RHO_ATTN="${1#*=}"; shift ;;
    --rho-mlp) RHO_MLP="$2"; shift 2 ;;
    --rho-mlp=*) RHO_MLP="${1#*=}"; shift ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    --alpha=*) ALPHA="${1#*=}"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Default MODEL_ROOT if not provided
if [[ -z "$MODEL_ROOT" ]]; then
  MODEL_ROOT="edited_models/${MODEL}"
fi

mkdir -p "$LOG_DIR"

# Normalize "-1" -> "-1.0" for rm tag (to match existing checkpoints)
rm_tag="$RHO_MLP"
[[ "$rm_tag" == "-1" ]] && rm_tag="-1.0"

# If both edits disabled, nothing to evaluate (matches previous logic)
if [[ ("$RHO_ATTN" == "-1" || "$RHO_ATTN" == "-1.0") && \
      ("$RHO_MLP" == "-1" || "$RHO_MLP" == "-1.0") ]]; then
  echo "Both rho_attn and rho_mlp are disabled (-1.0); nothing to evaluate."
  exit 0
fi

model_path="${MODEL_ROOT}/rank1__ra${RHO_ATTN}_rm${rm_tag}_a${ALPHA}.pt"

echo "Model      : $MODEL"
echo "Model root : $MODEL_ROOT"
echo "rho_attn   : $RHO_ATTN"
echo "rho_mlp    : $RHO_MLP (tag: $rm_tag)"
echo "alpha      : $ALPHA"
echo "datasets   : ${DATASETS[*]}"
echo "checkpoint : $model_path"
echo

if [[ ! -f "$model_path" ]]; then
  echo "ERROR: model checkpoint not found: $model_path"
  exit 1
fi

total=0
for dataset in "${DATASETS[@]}"; do
  log_safe="${model_path//\//_}"
  log_prefix="${log_safe}__${dataset}"
  echo "[Eval] python evaluate_utility.py --model $model_path --dataset $dataset --n_sample $N_SAMPLE"

  python evaluate_utility.py \
    --model "$model_path" \
    --dataset "$dataset" \
    --n_sample "$N_SAMPLE" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    > "$LOG_DIR/${log_prefix}.log" 2>&1

  ((total++)) || true
done

echo
echo "Finished evaluations: $total runs."
