#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${PROJECT_DIR}/.venv/bin/python"

if [[ -x "${VENV_PYTHON}" ]]; then
  PYTHON_BIN="${VENV_PYTHON}"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "Python 3 introuvable. Exécutez d'abord : ./install_macos.sh" >&2
  exit 1
fi

cd "${PROJECT_DIR}"
exec "${PYTHON_BIN}" frontend/server.py --host 127.0.0.1 --port 8888 "$@"
