#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/Graduation-design}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets_video_text/data}"
MELD_ROOT="${MELD_ROOT:-${DATA_ROOT}/meld}"
RAW_DIR="${MELD_RAW_DIR:-${MELD_ROOT}/raw}"
ARCHIVE_DIR="${MELD_ARCHIVE_DIR:-${MELD_ROOT}/archives}"
PROCESSED_DIR="${MELD_PROCESSED_DIR:-${MELD_ROOT}/processed}"
ARCHIVE="${ARCHIVE_DIR}/MELD.Raw.tar.gz"

MELD_URLS=(
  "https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz"
  "https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz"
)

mkdir -p "${RAW_DIR}" "${ARCHIVE_DIR}" "${PROCESSED_DIR}"

download_file() {
  local url="$1"
  local output="$2"
  echo "[MELD] Trying ${url}"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 8 -s 8 --continue=true -o "$(basename "${output}")" -d "$(dirname "${output}")" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "${output}" "${url}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --retry 5 --continue-at - -o "${output}" "${url}"
  else
    echo "No downloader found. Install aria2, wget, or curl." >&2
    return 127
  fi
}

if [[ ! -s "${ARCHIVE}" ]]; then
  ok=0
  for url in "${MELD_URLS[@]}"; do
    if download_file "${url}" "${ARCHIVE}.part"; then
      mv "${ARCHIVE}.part" "${ARCHIVE}"
      ok=1
      break
    fi
  done
  if [[ "${ok}" != "1" ]]; then
    echo "Failed to download MELD.Raw.tar.gz from all configured URLs." >&2
    exit 1
  fi
fi

if [[ ! -d "${RAW_DIR}/MELD.Raw" ]]; then
  echo "[MELD] Extracting ${ARCHIVE}"
  tar -xzf "${ARCHIVE}" -C "${RAW_DIR}"
fi

echo "[MELD] Removing audio-only files if present"
find "${RAW_DIR}" -type f \( -iname '*.wav' -o -iname '*.mp3' -o -iname '*.flac' -o -iname '*.aac' \) -delete

echo "[MELD] Building processed manifests"
python3 "${SCRIPT_DIR}/prepare_meld.py" \
  --raw-dir "${RAW_DIR}" \
  --output-dir "${PROCESSED_DIR}"

echo "[MELD] Done. Processed files are in ${PROCESSED_DIR}"

