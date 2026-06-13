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

extract_archive_once() {
  local archive="$1"
  local dest="$2"
  local marker="${dest}/.$(basename "${archive}").extracted"
  if [[ -f "${marker}" ]]; then
    return 0
  fi

  echo "[MELD] Extracting nested archive ${archive}"
  case "${archive}" in
    *.tar.gz|*.tgz)
      tar -xzf "${archive}" -C "${dest}"
      ;;
    *.tar)
      tar -xf "${archive}" -C "${dest}"
      ;;
    *.zip)
      unzip -q -o "${archive}" -d "${dest}"
      ;;
    *)
      echo "[MELD] Unsupported nested archive: ${archive}" >&2
      return 1
      ;;
  esac
  touch "${marker}"
}

ensure_annotation_csv() {
  local name="$1"
  local target=""

  target="$(find "${RAW_DIR}" -type f -iname "${name}" | head -n 1 || true)"
  if [[ -n "${target}" ]]; then
    return 0
  fi

  local annotation_dir="${RAW_DIR}/MELD.Raw"
  mkdir -p "${annotation_dir}"
  target="${annotation_dir}/${name}"

  local url="https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/${name}"
  echo "[MELD] Missing ${name}; downloading official annotation from ${url}"
  if command -v wget >/dev/null 2>&1; then
    wget -c -O "${target}" "${url}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --retry 5 -o "${target}" "${url}"
  else
    cat >&2 <<EOF
[MELD] Missing ${name}, and neither wget nor curl is available.
[MELD] Please download it manually from:
       ${url}
[MELD] Then place it below:
       ${annotation_dir}
EOF
    exit 1
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

echo "[MELD] Extracting inner split archives if present"
while IFS= read -r nested_archive; do
  extract_archive_once "${nested_archive}" "$(dirname "${nested_archive}")"
done < <(find "${RAW_DIR}" -type f \( -iname '*.tar.gz' -o -iname '*.tgz' -o -iname '*.tar' -o -iname '*.zip' \))

echo "[MELD] Ensuring official annotation CSV files exist"
ensure_annotation_csv "train_sent_emo.csv"
ensure_annotation_csv "dev_sent_emo.csv"
ensure_annotation_csv "test_sent_emo.csv"

echo "[MELD] Removing audio-only files if present"
find "${RAW_DIR}" -type f \( -iname '*.wav' -o -iname '*.mp3' -o -iname '*.flac' -o -iname '*.aac' \) -delete

echo "[MELD] Building processed manifests"
python3 "${SCRIPT_DIR}/prepare_meld.py" \
  --raw-dir "${RAW_DIR}" \
  --output-dir "${PROCESSED_DIR}"

echo "[MELD] Done. Processed files are in ${PROCESSED_DIR}"
