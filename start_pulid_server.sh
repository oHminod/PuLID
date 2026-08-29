#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PULID_PROJECT_ROOT="${PROJECT_DIR}"
VENV_ACTIVATE="${PROJECT_DIR}/.venv/bin/activate"

if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "Environnement virtuel introuvable : ${VENV_ACTIVATE}" >&2
  echo "Créez-le avec : uv venv --python 3.11" >&2
  exit 1
fi

cd "${PROJECT_DIR}"
source "${VENV_ACTIVATE}"

exec pulid-server \
  --host 127.0.0.1 \
  --port 12693 \
  --device mps \
  --cors-origin http://localhost:8800 \
  "$@"
