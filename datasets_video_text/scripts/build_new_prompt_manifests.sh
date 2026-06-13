#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/Graduation-design}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${DATASET:-all}"

python3 "${SCRIPT_DIR}/build_new_prompt_manifests.py" \
  --project-root "${PROJECT_ROOT}" \
  --dataset "${DATASET}" \
  "$@"

