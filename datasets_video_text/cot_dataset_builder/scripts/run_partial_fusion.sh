#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/../configs/qwen36_27b_dataset.yaml}"
PROJECT_ROOT="${PROJECT_ROOT:-/XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/Graduation-design}"
ENV_FILE="${ENV_FILE:-/XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/cache_env_new.sh}"
DATASET="${DATASET:-meld}"
SPLIT="${SPLIT:-train}"
SERVERS="${SERVERS:-http://127.0.0.1:18000/v1,http://127.0.0.1:18001/v1}"
NUM_SHARDS="${NUM_SHARDS:-2}"
LIMIT="${LIMIT:-0}"
RESUME="${RESUME:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/datasets_video_text/cot_dataset_builder/results/qwen36_27b/step3_fusion_reason_partial}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/datasets_video_text/cot_dataset_builder/logs/qwen36_27b/partial_fusion}"
ACTIVE_PIDS=()

cleanup_workers() {
  if [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; then
    echo "[partial-fusion] Stopping workers: ${ACTIVE_PIDS[*]}" >&2
    kill -TERM "${ACTIVE_PIDS[@]}" 2>/dev/null || true
    sleep 2
    kill -KILL "${ACTIVE_PIDS[@]}" 2>/dev/null || true
  fi
}
trap 'cleanup_workers; exit 130' INT TERM

if [[ -n "${ENV_FILE}" && -f "${ENV_FILE}" ]]; then
  echo "[partial-fusion] Sourcing environment: ${ENV_FILE}"
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
fi

VISUAL_DIR="${PROJECT_ROOT}/datasets_video_text/cot_dataset_builder/results/qwen36_27b/step1_visual_reason/${DATASET}"
DIALOGUE_DIR="${PROJECT_ROOT}/datasets_video_text/cot_dataset_builder/results/qwen36_27b/step2_dialogue_reason/${DATASET}"
OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

shopt -s nullglob
if [[ -n "${VISUAL_REASON_FILES:-}" ]]; then
  # shellcheck disable=SC2206
  visual_files=(${VISUAL_REASON_FILES})
else
  visual_files=("${VISUAL_DIR}/${SPLIT}_shard"*.jsonl "${VISUAL_DIR}/${SPLIT}_shard"*.json)
fi

if [[ -n "${DIALOGUE_REASON_FILES:-}" ]]; then
  # shellcheck disable=SC2206
  dialogue_files=(${DIALOGUE_REASON_FILES})
else
  dialogue_files=("${DIALOGUE_DIR}/${SPLIT}_shard"*.jsonl "${DIALOGUE_DIR}/${SPLIT}_shard"*.json)
fi
shopt -u nullglob

if [[ "${#visual_files[@]}" -eq 0 ]]; then
  echo "[partial-fusion] No visual reason files found under ${VISUAL_DIR}" >&2
  exit 1
fi
if [[ "${#dialogue_files[@]}" -eq 0 ]]; then
  echo "[partial-fusion] No dialogue reason files found under ${DIALOGUE_DIR}" >&2
  exit 1
fi

echo "[partial-fusion] visual files:"
printf '  %s\n' "${visual_files[@]}"
echo "[partial-fusion] dialogue files:"
printf '  %s\n' "${dialogue_files[@]}"
echo "[partial-fusion] output dir: ${OUTPUT_DIR}"

common_args=(
  --config "${CONFIG}"
  --step fusion
  --dataset "${DATASET}"
  --split "${SPLIT}"
  --servers "${SERVERS}"
  --num-shards "${NUM_SHARDS}"
  --limit "${LIMIT}"
  --only-available-reasons
)
for file in "${visual_files[@]}"; do
  common_args+=(--visual-reason-file "${file}")
done
for file in "${dialogue_files[@]}"; do
  common_args+=(--dialogue-reason-file "${file}")
done
if [[ "${RESUME}" == "1" ]]; then
  common_args+=(--resume)
fi

pids=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  log_file="${LOG_DIR}/${DATASET}_${SPLIT}_shard${shard}.log"
  output_file="${OUTPUT_DIR}/${SPLIT}_shard${shard}.jsonl"
  python3 "${SCRIPT_DIR}/generate_reasoning_with_vllm.py" \
    "${common_args[@]}" \
    --shard-index "${shard}" \
    --output "${output_file}" \
    > "${log_file}" 2>&1 &
  pids+=("$!")
  ACTIVE_PIDS+=("$!")
  echo "[partial-fusion] shard=${shard} pid=${pids[$shard]} output=${output_file} log=${log_file}"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
ACTIVE_PIDS=()

if [[ "${failed}" != "0" ]]; then
  echo "[partial-fusion] At least one worker failed. Check ${LOG_DIR}." >&2
  exit 1
fi

echo "[partial-fusion] Done. Outputs are in ${OUTPUT_DIR}"
