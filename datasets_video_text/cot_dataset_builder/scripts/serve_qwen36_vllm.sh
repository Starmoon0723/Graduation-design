#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/../configs/qwen36_27b_dataset.yaml}"
ACTION="${1:-start}"
ENV_FILE="${ENV_FILE:-}"

if [[ -z "${ENV_FILE}" ]]; then
  ENV_FILE="$(SCRIPT_DIR="${SCRIPT_DIR}" python3 - "${CONFIG}" <<'PY'
import os
import sys
from pathlib import Path
sys.path.insert(0, os.environ["SCRIPT_DIR"])
from config_utils import load_config
cfg = load_config(sys.argv[1])
print(cfg.get("project", {}).get("env_file", ""))
PY
)"
fi

if [[ -n "${ENV_FILE}" && -f "${ENV_FILE}" ]]; then
  echo "[qwen36-vllm] Sourcing environment: ${ENV_FILE}"
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
elif [[ -n "${ENV_FILE}" ]]; then
  echo "[qwen36-vllm] ENV_FILE not found, continuing without sourcing: ${ENV_FILE}" >&2
fi

python3 "${SCRIPT_DIR}/serve_qwen36_vllm.py" "${ACTION}" --config "${CONFIG}"
