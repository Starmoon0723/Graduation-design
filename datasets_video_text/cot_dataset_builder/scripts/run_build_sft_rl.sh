#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/../configs/qwen36_27b_dataset.yaml}"
STEP="${STEP:-both}"
DATASET="${DATASET:-all}"
SPLIT="${SPLIT:-train}"
INCLUDE_INCORRECT_SFT="${INCLUDE_INCORRECT_SFT:-0}"

steps=()
case "${STEP}" in
  visual) steps=(visual) ;;
  dialogue) steps=(dialogue) ;;
  both) steps=(visual dialogue) ;;
  *) echo "STEP must be visual, dialogue, or both" >&2; exit 1 ;;
esac

datasets=()
case "${DATASET}" in
  meld) datasets=(meld) ;;
  iemocap) datasets=(iemocap) ;;
  all) datasets=(meld iemocap) ;;
  *) echo "DATASET must be meld, iemocap, or all" >&2; exit 1 ;;
esac

for step in "${steps[@]}"; do
  for dataset in "${datasets[@]}"; do
    args=(--config "${CONFIG}" --step "${step}" --dataset "${dataset}" --split "${SPLIT}")
    if [[ "${INCLUDE_INCORRECT_SFT}" == "1" ]]; then
      args+=(--include-incorrect-sft)
    fi
    python3 "${SCRIPT_DIR}/build_sft_rl_dataset.py" "${args[@]}"
  done
done
