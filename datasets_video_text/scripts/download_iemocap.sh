#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/XYFS01/HDD_POOL/hitsz_mszhang/hitsz_mszhang_1/MRC/MRC/MRC_project/others/AAA/vlm/Graduation-design}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets_video_text/data}"
IEMOCAP_ROOT="${IEMOCAP_ROOT:-${DATA_ROOT}/iemocap}"
RAW_DIR="${IEMOCAP_RAW_DIR:-${IEMOCAP_ROOT}/raw}"
ARCHIVE_DIR="${IEMOCAP_ARCHIVE_DIR:-${IEMOCAP_ROOT}/archives}"
PROCESSED_DIR="${IEMOCAP_PROCESSED_DIR:-${IEMOCAP_ROOT}/processed}"

mkdir -p "${RAW_DIR}" "${ARCHIVE_DIR}" "${PROCESSED_DIR}"

echo "[IEMOCAP] This dataset is license-restricted."
echo "[IEMOCAP] Put official archive files in: ${ARCHIVE_DIR}"
echo "[IEMOCAP] Or provide private authorized URLs via IEMOCAP_URLS_FILE."

download_authorized_urls() {
  local urls_file="$1"
  [[ -f "${urls_file}" ]] || return 0
  while IFS= read -r url; do
    [[ -z "${url}" || "${url}" =~ ^# ]] && continue
    local name
    name="$(basename "${url%%\?*}")"
    local out="${ARCHIVE_DIR}/${name}"
    [[ -s "${out}" ]] && continue
    echo "[IEMOCAP] Downloading authorized URL: ${url}"
    if command -v aria2c >/dev/null 2>&1; then
      aria2c -x 8 -s 8 --continue=true -o "${name}" -d "${ARCHIVE_DIR}" "${url}"
    elif command -v wget >/dev/null 2>&1; then
      wget -c -O "${out}" "${url}"
    elif command -v curl >/dev/null 2>&1; then
      curl -L --retry 5 --continue-at - -o "${out}" "${url}"
    else
      echo "No downloader found. Install aria2, wget, or curl." >&2
      return 127
    fi
  done < "${urls_file}"
}

if [[ -n "${IEMOCAP_URLS_FILE:-}" ]]; then
  download_authorized_urls "${IEMOCAP_URLS_FILE}"
fi

shopt -s nullglob
archives=("${ARCHIVE_DIR}"/*.zip "${ARCHIVE_DIR}"/*.tar "${ARCHIVE_DIR}"/*.tar.gz "${ARCHIVE_DIR}"/*.tgz)
if [[ "${#archives[@]}" -eq 0 && ! -d "${RAW_DIR}/IEMOCAP_full_release" ]]; then
  cat >&2 <<EOF
No IEMOCAP archive found.

Steps:
1. Obtain IEMOCAP from the official USC SAIL distribution after approval.
2. Copy the official zip/tar files to:
   ${ARCHIVE_DIR}
3. Re-run:
   IEMOCAP_ARCHIVE_DIR=${ARCHIVE_DIR} bash ${SCRIPT_DIR}/download_iemocap.sh
EOF
  exit 1
fi

for archive in "${archives[@]}"; do
  marker="${RAW_DIR}/.$(basename "${archive}").extracted"
  [[ -f "${marker}" ]] && continue
  echo "[IEMOCAP] Extracting $(basename "${archive}")"
  case "${archive}" in
    *.zip)
      unzip -q -o "${archive}" -d "${RAW_DIR}"
      ;;
    *.tar)
      tar -xf "${archive}" -C "${RAW_DIR}"
      ;;
    *.tar.gz|*.tgz)
      tar -xzf "${archive}" -C "${RAW_DIR}"
      ;;
    *)
      echo "Unsupported archive: ${archive}" >&2
      exit 1
      ;;
  esac
  touch "${marker}"
done

echo "[IEMOCAP] Removing audio-only files if present"
find "${RAW_DIR}" -type f \( -iname '*.wav' -o -iname '*.mp3' -o -iname '*.flac' -o -iname '*.aac' \) -delete

echo "[IEMOCAP] Building processed manifests"
python3 "${SCRIPT_DIR}/prepare_iemocap.py" \
  --raw-dir "${RAW_DIR}" \
  --output-dir "${PROCESSED_DIR}"

echo "[IEMOCAP] Done. Processed files are in ${PROCESSED_DIR}"

