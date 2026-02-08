#!/usr/bin/env bash
set -euo pipefail

# ---------------- Config ----------------
MODEL="${MODEL:-qwen3-4b-thinking}"     # can override via --model or MODEL=
MODEL_ROOT="${MODEL_ROOT:-}"            # optional override
N_SAMPLE=3
TENSOR_PARALLEL_SIZE=4
LOG_DIR="logs"

RHO_ATTN_RANGE="${RHO_ATTN_RANGE:-}"
RHO_MLP_RANGE="${RHO_MLP_RANGE:-}"
ALPHA_RANGE="${ALPHA_RANGE:-}"

# default dataset(s)
DATASET_ARG="MATH-500,gpqa,code-mmlu,gsm8k"
# ----------------------------------------

# -------- Parse args --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --model=*) MODEL="${1#*=}"; shift ;;
    --model-root) MODEL_ROOT="$2"; shift 2 ;;
    --model-root=*) MODEL_ROOT="${1#*=}"; shift ;;
    --dataset) DATASET_ARG="$2"; shift 2 ;;
    --dataset=*) DATASET_ARG="${1#*=}"; shift ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

# parse datasets (comma-separated)
IFS=',' read -r -a DATASETS <<<"$DATASET_ARG"

# Default model root if not given
if [[ -z "$MODEL_ROOT" ]]; then
  MODEL_ROOT="edited_models/${MODEL}"
fi

MODEL_NAME="$(basename "$MODEL_ROOT")"

# Default grid ranges from summary.md (only if not set by user)
if [[ -z "$RHO_ATTN_RANGE" || -z "$RHO_MLP_RANGE" || -z "$ALPHA_RANGE" ]]; then
  case "$MODEL" in
    qwen3-4b-thinking)
      RHO_ATTN_RANGE="${RHO_ATTN_RANGE:--1.0}"
      RHO_MLP_RANGE="${RHO_MLP_RANGE:-0.65,0.8,0.05}"
      ALPHA_RANGE="${ALPHA_RANGE:-0.05,0.2,0.05}"
      ;;
    nemotron-7b)
      RHO_ATTN_RANGE="${RHO_ATTN_RANGE:-0.2,0.3,0.05}"
      RHO_MLP_RANGE="${RHO_MLP_RANGE:-0.8,0.9,0.05}"
      ALPHA_RANGE="${ALPHA_RANGE:-0.1,0.2,0.05}"
      ;;
  esac
fi

mkdir -p "$LOG_DIR"

# -------- Utility functions --------
normf() {
  if [[ "$1" == "-1.0" ]]; then
    echo "-1.0"
    return
  fi
  local x
  x=$(printf "%.3f" "$1")
  x="${x%%0}"
  while [[ "$x" =~ \.[0-9]*0$ ]]; do x="${x%0}"; done
  [[ "$x" =~ \.$ ]] && x="${x%.}"
  echo "$x"
}

mk_seq() {
  local range="$1"
  if [[ "$range" == "-1.0" ]]; then
    echo "-1.0"
    return
  fi
  IFS=',' read -r start end step <<<"$range"
  [[ -z "${start:-}" || -z "${end:-}" || -z "${step:-}" ]] && {
    echo "ERROR: bad range '$range' (need start,end,step)" >&2
    exit 1
  }
  awk -v s="$start" -v e="$end" -v d="$step" '
    BEGIN { for (x = s; x <= e + (d/2); x += d) printf "%.9f\n", x }
  ' | while read -r v; do normf "$v"; done
}

mapfile -t RA_LIST < <(mk_seq "$RHO_ATTN_RANGE")
mapfile -t RM_LIST < <(mk_seq "$RHO_MLP_RANGE")
mapfile -t A_LIST  < <(mk_seq "$ALPHA_RANGE")

echo "Model          : $MODEL"
echo "Model root     : $MODEL_ROOT"
echo "Model name     : $MODEL_NAME"
echo "Datasets       : ${DATASETS[*]}"
echo "rho_attn list  : ${RA_LIST[*]}"
echo "rho_mlp  list  : ${RM_LIST[*]}"
echo "alpha    list  : ${A_LIST[*]}"
echo

total=0
skipped_missing=0
skipped_done=0

for ra in "${RA_LIST[@]}"; do
  for rm in "${RM_LIST[@]}"; do
    for a in "${A_LIST[@]}"; do
      if [[ "$ra" == "-1" || "$ra" == "-1.0" ]]; then
        [[ "$rm" == "-1" || "$rm" == "-1.0" ]] && continue
      fi

      rm_tag="$rm"
      [[ "$rm" == "-1" ]] && rm_tag="-1.0"

      RUN_DIR="rank1__ra${ra}_rm${rm_tag}_a${a}.pt"
      model_path="${MODEL_ROOT}/${RUN_DIR}"

      if [[ ! -f "$model_path" ]]; then
        echo "[SKIP] missing model: $model_path"
        ((skipped_missing++)) || true
        continue
      fi

      for dataset in "${DATASETS[@]}"; do
        RESULT_JSON="evaluate_results/${dataset}/edited_models/${MODEL_NAME}/${RUN_DIR}/${N_SAMPLE}_runs.json"

        if [[ -f "$RESULT_JSON" ]]; then
          echo "[SKIP] already evaluated: $RESULT_JSON"
          ((skipped_done++)) || true
          continue
        fi

        log_safe="${model_path//\//_}"
        log_prefix="${log_safe}__${dataset}"

        echo "[RUN ] python evaluate.py --model $model_path --dataset $dataset --n_sample $N_SAMPLE"

        python evaluate.py \
          --model "$model_path" \
          --dataset "$dataset" \
          --n_sample "$N_SAMPLE" \
          --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
          > "$LOG_DIR/${log_prefix}.log" 2>&1

        ((total++)) || true
      done
    done
  done
done

echo
echo "Finished evaluations : $total"
echo "Skipped (missing)    : $skipped_missing"
echo "Skipped (already run): $skipped_done"
