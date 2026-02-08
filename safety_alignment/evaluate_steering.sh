#!/usr/bin/env bash
set -euo pipefail

N_SAMPLE=10

MODELS=(
  "llama2-7b"
  "mistral-7b"
)

DATASETS=(
  "gcg"
  "advllm"
)

ALPHAS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5)

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

for MODEL in "${MODELS[@]}"; do
  TENSOR_PARALLEL_SIZE=4
  LOG_SAFE_MODEL="${MODEL//\//_}"

  for DATASET in "${DATASETS[@]}"; do

    # ================= Normal evaluation =================
    RESULT_JSON="evaluate_results/${DATASET}/${MODEL}/${N_SAMPLE}_runs.json"
    LOG_PREFIX="${LOG_SAFE_MODEL}__${DATASET}"

    if [[ -f "$RESULT_JSON" ]]; then
      echo "[SKIP][Normal] exists: $RESULT_JSON"
    else
      echo "[RUN ][Normal] evaluate.py --model $MODEL --dataset $DATASET --n_sample $N_SAMPLE"
      python evaluate.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --n_sample "$N_SAMPLE" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        > "$LOG_DIR/${LOG_PREFIX}.log" 2>&1
    fi

    # ================= Steering evaluations =================
    for ALPHA in "${ALPHAS[@]}"; do
      STEER_DIR="${MODEL}_steering_${ALPHA}_attnmlp"
      RESULT_JSON="evaluate_results/${DATASET}/${STEER_DIR}/${N_SAMPLE}_runs.json"
      LOG_PREFIX="${LOG_SAFE_MODEL}__${DATASET}__steer_alpha${ALPHA}"

      if [[ -f "$RESULT_JSON" ]]; then
        echo "[SKIP][Steer] exists: $RESULT_JSON"
        continue
      fi

      echo "[RUN ][Steer] evaluate.py --model $MODEL --dataset $DATASET --alpha $ALPHA"
      python evaluate.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --n_sample "$N_SAMPLE" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --do_steering \
        --alpha "$ALPHA" \
        > "$LOG_DIR/${LOG_PREFIX}.log" 2>&1
    done

  done
done
