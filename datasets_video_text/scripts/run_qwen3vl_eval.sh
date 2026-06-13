#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/Graduation-design}"
MODEL_PATH="${MODEL_PATH:-/XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/hfmodel/qwen3vl_8b}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/datasets_video_text/results/qwen3vl_8b}"
DATASET="${DATASET:-all}"
SPLIT="${SPLIT:-test}"
GPUS="${GPUS:-0,1,2,3}"
FPS="${FPS:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
FLASH_ATTN="${FLASH_ATTN:-1}"
RESUME="${RESUME:-1}"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
WORLD_SIZE="${#GPU_ARRAY[@]}"

common_args=(
  --project-root "${PROJECT_ROOT}"
  --model-path "${MODEL_PATH}"
  --split "${SPLIT}"
  --output-dir "${OUTPUT_DIR}"
  --fps "${FPS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
)

if [[ "${FLASH_ATTN}" == "1" ]]; then
  common_args+=(--flash-attn)
fi
if [[ "${RESUME}" == "1" ]]; then
  common_args+=(--resume)
fi

run_dataset() {
  local dataset="$1"
  echo "[Qwen3VL] Running ${dataset}/${SPLIT} on GPUs ${GPUS} with fps=${FPS}"
  local pids=()
  for rank in "${!GPU_ARRAY[@]}"; do
    local gpu="${GPU_ARRAY[$rank]}"
    local log_file="${LOG_DIR}/${dataset}_${SPLIT}_rank${rank}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" python3 "${SCRIPT_DIR}/evaluate_qwen3vl.py" \
      --mode infer \
      --dataset "${dataset}" \
      --rank "${rank}" \
      --world-size "${WORLD_SIZE}" \
      "${common_args[@]}" \
      > "${log_file}" 2>&1 &
    pids+=("$!")
    echo "[Qwen3VL] rank=${rank} gpu=${gpu} pid=${pids[$rank]} log=${log_file}"
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "[Qwen3VL] At least one worker failed for ${dataset}. Check ${LOG_DIR}." >&2
    exit 1
  fi

  python3 "${SCRIPT_DIR}/evaluate_qwen3vl.py" \
    --mode aggregate \
    --dataset "${dataset}" \
    "${common_args[@]}"
}

if [[ "${DATASET}" == "all" ]]; then
  run_dataset "meld"
  run_dataset "iemocap"
elif [[ "${DATASET}" == "meld" || "${DATASET}" == "iemocap" ]]; then
  run_dataset "${DATASET}"
else
  echo "DATASET must be one of: all, meld, iemocap" >&2
  exit 1
fi

echo "[Qwen3VL] Done. Results are in ${OUTPUT_DIR}"
