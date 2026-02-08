#!/usr/bin/env bash
set -euo pipefail

MODELS=(
  "gemma2-2b"
  "llama3-8b"
)

DATASETS=(
  "truthfulqa"
)

ALPHAS=(0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0)
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

for MODEL in "${MODELS[@]}"; do
  TENSOR_PARALLEL_SIZE=4
  LOG_SAFE_MODEL="${MODEL//\//_}"  # replace '/' with '_'

  for DATASET in "${DATASETS[@]}"; do

    # ================= Normal evaluation =================
    RESULT_JSON="evaluate_results/${DATASET}/${MODEL}/1_run.json"
    LOG_PREFIX="${LOG_SAFE_MODEL}__${DATASET}"

    if [[ -f "$RESULT_JSON" ]]; then
      echo "[SKIP][Normal] exists: $RESULT_JSON"
    else
      echo "[RUN ][Normal] evaluate.py --model $MODEL --dataset $DATASET"
      python evaluate.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        > "$LOG_DIR/${LOG_PREFIX}.log" 2>&1
    fi

    # ================= Steering evaluations =================
    for ALPHA in "${ALPHAS[@]}"; do
      STEER_DIR="${MODEL}_steering_${ALPHA}_attnmlp"
      RESULT_JSON="evaluate_results/${DATASET}/${STEER_DIR}/1_run.json"
      LOG_PREFIX="${LOG_SAFE_MODEL}__${DATASET}__steer_alpha${ALPHA}"

      if [[ -f "$RESULT_JSON" ]]; then
        echo "[SKIP][Steer ] exists: $RESULT_JSON"
        continue
      fi

      echo "[RUN ][Steer ] evaluate.py --model $MODEL --dataset $DATASET --alpha $ALPHA"
      python evaluate.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --do_steering \
        --alpha "$ALPHA" \
        > "$LOG_DIR/${LOG_PREFIX}.log" 2>&1
    done

  done
done
