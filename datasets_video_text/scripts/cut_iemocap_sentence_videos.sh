#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/Graduation-design}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IEMOCAP_ROOT="${IEMOCAP_ROOT:-${PROJECT_ROOT}/datasets_video_text/data/iemocap}"
INPUT_DIR="${IEMOCAP_PROCESSED_DIR:-${IEMOCAP_ROOT}/processed}"
OUTPUT_DIR="${IEMOCAP_SENTENCE_PROCESSED_DIR:-${IEMOCAP_ROOT}/processed_sentence}"
VIDEO_ROOT="${IEMOCAP_SENTENCE_VIDEO_ROOT:-${IEMOCAP_ROOT}/sentence_videos}"
WORKERS="${WORKERS:-4}"

python3 "${SCRIPT_DIR}/cut_iemocap_sentence_videos.py" \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --video-root "${VIDEO_ROOT}" \
  --workers "${WORKERS}" \
  "$@"

