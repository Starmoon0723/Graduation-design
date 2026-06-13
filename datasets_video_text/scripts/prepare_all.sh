#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/Graduation-design}"

echo "[ALL] Preparing MELD"
bash "${PROJECT_ROOT}/datasets_video_text/scripts/download_meld.sh"

echo "[ALL] Preparing IEMOCAP"
bash "${PROJECT_ROOT}/datasets_video_text/scripts/download_iemocap.sh"

echo "[ALL] Finished"

