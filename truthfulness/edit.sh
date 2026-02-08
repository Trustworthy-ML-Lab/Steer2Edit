#!/usr/bin/env bash
set -euo pipefail

# -------- Config (defaults can be overridden by env or flags) --------
MODEL="${MODEL:-llama3-8b}"
DIRECTIONS_PATH="${DIRECTIONS_PATH:-}"
SAVE_ROOT="${SAVE_ROOT:-edited_models}"

RHO_ATTN_RANGE="${RHO_ATTN_RANGE:-}"
RHO_MLP_RANGE="${RHO_MLP_RANGE:-}"
ALPHA_RANGE="${ALPHA_RANGE:-}"

print_usage() {
  cat <<EOF
Usage: ./grid_edit.sh [--model NAME] [--dirs PATH] [--save-root DIR]

Examples:
  MODEL=mistral-7b ./grid_edit.sh
  MODEL=mistral-7b RHO_ATTN_RANGE="0.42,0.50,0.02" RHO_MLP_RANGE="0.40,0.60,0.05" ALPHA_RANGE="0.65,0.85,0.05" ./grid_edit.sh

Skip behavior:
  - If the output .pt file already exists, it will be skipped.
EOF
}

# -------- Parse flags --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --dirs|--directions) DIRECTIONS_PATH="$2"; shift 2;;
    --save-root) SAVE_ROOT="$2"; shift 2;;
    -h|--help) print_usage; exit 0;;
    *) echo "Unknown arg: $1"; print_usage; exit 1;;
  esac
done

# Auto-adjust directions path if not explicitly set
if [[ -z "$DIRECTIONS_PATH" ]]; then
  DIRECTIONS_PATH="directions/${MODEL}_truthful_dirs.pt"
fi

# Default grid ranges from summary.md (only if not set by user)
if [[ -z "$RHO_ATTN_RANGE" || -z "$RHO_MLP_RANGE" || -z "$ALPHA_RANGE" ]]; then
  case "$MODEL" in
    gemma2-2b)
      RHO_ATTN_RANGE="${RHO_ATTN_RANGE:-0.3,0.5,0.05}"
      RHO_MLP_RANGE="${RHO_MLP_RANGE:--1.0}"
      ALPHA_RANGE="${ALPHA_RANGE:-0.75,0.95,0.05}"
      ;;
    llama3-8b)
      RHO_ATTN_RANGE="${RHO_ATTN_RANGE:-0.1,0.14,0.01}"
      RHO_MLP_RANGE="${RHO_MLP_RANGE:-0.3,0.5,0.05}"
      ALPHA_RANGE="${ALPHA_RANGE:-0.3,0.7,0.1}"
      ;;
  esac
fi

SAVE_DIR="$SAVE_ROOT/$MODEL"
mkdir -p "$SAVE_DIR"

# -------- Utility: build numeric sequences --------
mk_seq() {
  local range="$1"
  if [[ "$range" == "-1.0" ]]; then
    echo "-1.0"
  else
    IFS=',' read -r start end step <<<"$range"
    if [[ -z "${start:-}" || -z "${end:-}" || -z "${step:-}" ]]; then
      echo "ERROR: bad range '$range' (need start,end,step)" >&2
      exit 1
    fi
    LC_NUMERIC=C seq "$start" "$step" "$end"
  fi
}

# Normalize floats to match your saved filenames like 0.42 (not 0.420000)
normf() {
  # Keep sentinel exactly "-1.0" so filenames are consistent and skips work.
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

out_file_for() {
  local ra rm a
  ra="$(normf "$1")"
  rm="$(normf "$2")"
  a="$(normf "$3")"
  echo "${SAVE_DIR}/rank1__ra${ra}_rm${rm}_a${a}.pt"
}

mapfile -t RHO_ATTN_LIST < <(mk_seq "$RHO_ATTN_RANGE")
mapfile -t RHO_MLP_LIST  < <(mk_seq "$RHO_MLP_RANGE")
mapfile -t ALPHA_LIST    < <(mk_seq "$ALPHA_RANGE")

echo "Model          : $MODEL"
echo "Directions     : $DIRECTIONS_PATH"
echo "Save dir       : $SAVE_DIR"
echo "rho_attn list  : ${RHO_ATTN_LIST[*]}"
echo "rho_mlp  list  : ${RHO_MLP_LIST[*]}"
echo "alpha    list  : ${ALPHA_LIST[*]}"
echo

# -------- Main sweep --------
total=0
skipped=0

for rho_attn in "${RHO_ATTN_LIST[@]}"; do
  for rho_mlp in "${RHO_MLP_LIST[@]}"; do
    for alpha in "${ALPHA_LIST[@]}"; do
      if [[ "$rho_attn" == "-1.0" && "$rho_mlp" == "-1.0" ]]; then
        echo "Skipping (rho_attn=-1.0 && rho_mlp=-1.0): alpha=$alpha"
        continue
      fi

      OUT_PT="$(out_file_for "$rho_attn" "$rho_mlp" "$alpha")"

      if [[ -f "$OUT_PT" ]]; then
        echo "[SKIP] exists: $OUT_PT"
        skipped=$((skipped+1))
        continue
      fi

      printf '[RUN ] edit.py --model %s --rho_attn %s --rho_mlp %s --alpha %s\n' \
        "$MODEL" "$rho_attn" "$rho_mlp" "$alpha"
      echo "       -> $OUT_PT"

      python edit.py \
        --model "$MODEL" \
        --directions_path "$DIRECTIONS_PATH" \
        --rho_attn "$rho_attn" \
        --rho_mlp "$rho_mlp" \
        --alpha "$alpha" \
        --save_root "$SAVE_ROOT"

      total=$((total+1))
    done
  done
done

echo
echo "Completed $total edit configurations."
echo "Skipped   $skipped (already exists)."
