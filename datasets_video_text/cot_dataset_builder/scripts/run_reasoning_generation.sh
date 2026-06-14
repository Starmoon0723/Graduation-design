#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/../configs/qwen36_27b_dataset.yaml}"
PROJECT_ROOT="${PROJECT_ROOT:-/XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/Graduation-design}"
ENV_FILE="${ENV_FILE:-/XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/cache_env_new.sh}"
STEP="${STEP:-both}"
DATASET="${DATASET:-all}"
SPLIT="${SPLIT:-train}"
SERVERS="${SERVERS:-http://127.0.0.1:18000/v1,http://127.0.0.1:18001/v1}"
NUM_SHARDS="${NUM_SHARDS:-2}"
LIMIT="${LIMIT:-0}"
RESUME="${RESUME:-1}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/datasets_video_text/cot_dataset_builder/logs/qwen36_27b/generate}"
ACTIVE_PIDS=()

cleanup_workers() {
  if [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; then
    echo "[cot-generate] Stopping workers: ${ACTIVE_PIDS[*]}" >&2
    kill -TERM "${ACTIVE_PIDS[@]}" 2>/dev/null || true
    sleep 2
    kill -KILL "${ACTIVE_PIDS[@]}" 2>/dev/null || true
  fi
}
trap 'cleanup_workers; exit 130' INT TERM

if [[ -n "${ENV_FILE}" && -f "${ENV_FILE}" ]]; then
  echo "[cot-generate] Sourcing environment: ${ENV_FILE}"
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
fi

mkdir -p "${LOG_DIR}"

run_one() {
  local step="$1"
  local dataset="$2"
  echo "[cot-generate] step=${step} dataset=${dataset} split=${SPLIT} shards=${NUM_SHARDS}"
  local pids=()
  ACTIVE_PIDS=()
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    local log_file="${LOG_DIR}/${step}_${dataset}_${SPLIT}_shard${shard}.log"
    local args=(
      --config "${CONFIG}"
      --step "${step}"
      --dataset "${dataset}"
      --split "${SPLIT}"
      --servers "${SERVERS}"
      --shard-index "${shard}"
      --num-shards "${NUM_SHARDS}"
      --limit "${LIMIT}"
    )
    if [[ "${RESUME}" == "1" ]]; then
      args+=(--resume)
    fi
    python3 "${SCRIPT_DIR}/generate_reasoning_with_vllm.py" "${args[@]}" > "${log_file}" 2>&1 &
    pids+=("$!")
    ACTIVE_PIDS+=("$!")
    echo "[cot-generate] shard=${shard} pid=${pids[$shard]} log=${log_file}"
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  ACTIVE_PIDS=()
  if [[ "${failed}" != "0" ]]; then
    echo "[cot-generate] failed: step=${step} dataset=${dataset}. Check ${LOG_DIR}" >&2
    exit 1
  fi
}

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
    run_one "${step}" "${dataset}"
  done
done

echo "[cot-generate] Done."
