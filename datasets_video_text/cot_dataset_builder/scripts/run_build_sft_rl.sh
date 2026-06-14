#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/../configs/qwen36_27b_dataset.yaml}"
DATASET="${DATASET:-all}"
SPLIT="${SPLIT:-train}"
BUILD_WEAK_PREFERENCES="${BUILD_WEAK_PREFERENCES:-0}"
SKIP_VIDEO_EXISTS_CHECK="${SKIP_VIDEO_EXISTS_CHECK:-0}"

datasets=()
case "${DATASET}" in
  meld) datasets=(meld) ;;
  iemocap) datasets=(iemocap) ;;
  all) datasets=(meld iemocap) ;;
  *) echo "DATASET must be meld, iemocap, or all" >&2; exit 1 ;;
esac

for dataset in "${datasets[@]}"; do
  args=(--config "${CONFIG}" --dataset "${dataset}" --split "${SPLIT}")
  if [[ "${BUILD_WEAK_PREFERENCES}" == "1" ]]; then
    args+=(--build-weak-preferences)
  fi
  if [[ "${SKIP_VIDEO_EXISTS_CHECK}" == "1" ]]; then
    args+=(--skip-video-exists-check)
  fi
  python3 "${SCRIPT_DIR}/build_sft_rl_dataset.py" "${args[@]}"
done
