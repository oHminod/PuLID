#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"
LLAMA_CPP_METAL_INDEX="https://abetlen.github.io/llama-cpp-python/whl/metal"

normalize_models_root() {
  local selected="${1%/}"
  if [[ "${selected}" == "~" ]]; then
    selected="${HOME}"
  elif [[ "${selected}" == "~/"* ]]; then
    selected="${HOME}/${selected:2}"
  elif [[ "${selected}" != /* ]]; then
    selected="${PROJECT_DIR}/${selected}"
  fi
  if [[ "$(basename -- "${selected}" | tr '[:upper:]' '[:lower:]')" == "pulid_models" ]]; then
    printf '%s\n' "${selected}"
  else
    printf '%s\n' "${selected}/PuLID_models"
  fi
}

read_configured_models_root() {
  local config_file="${PROJECT_DIR}/config/local.yaml"
  local line=""
  local value=""
  [[ -f "${config_file}" ]] || return 0
  while IFS= read -r line; do
    if [[ "${line}" == models_root:* ]]; then
      value="${line#models_root:}"
      value="${value#"${value%%[![:space:]]*}"}"
      value="${value%"${value##*[![:space:]]}"}"
      if [[ ${#value} -ge 2 ]]; then
        if [[ "${value:0:1}" == '"' && "${value: -1}" == '"' ]] ||
          [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
          value="${value:1:${#value}-2}"
        fi
      fi
      if [[ -n "${value}" && "${value}" != /* ]]; then
        value="${PROJECT_DIR}/${value}"
      fi
      printf '%s\n' "${value%/}"
      return 0
    fi
  done < "${config_file}"
}

DEFAULT_MODELS_ROOT="${PROJECT_DIR}/PuLID_models"
REQUESTED_MODELS_ROOT="${PULID_MODELS_ROOT:-}"
PULID_MODELS_ROOT=""

if [[ -n "${REQUESTED_MODELS_ROOT}" ]]; then
  REQUESTED_MODELS_ROOT="$(normalize_models_root "${REQUESTED_MODELS_ROOT}")"
  if [[ -d "${REQUESTED_MODELS_ROOT}" ]]; then
    PULID_MODELS_ROOT="${REQUESTED_MODELS_ROOT}"
  fi
fi

if [[ -z "${PULID_MODELS_ROOT}" ]]; then
  CONFIGURED_MODELS_ROOT="$(read_configured_models_root)"
  if [[ -n "${CONFIGURED_MODELS_ROOT}" && -d "${CONFIGURED_MODELS_ROOT}" ]]; then
    PULID_MODELS_ROOT="${CONFIGURED_MODELS_ROOT}"
  fi
fi

if [[ -z "${PULID_MODELS_ROOT}" && -d "${DEFAULT_MODELS_ROOT}" ]]; then
  PULID_MODELS_ROOT="${DEFAULT_MODELS_ROOT}"
fi

if [[ -n "${PULID_MODELS_ROOT}" ]]; then
  echo "Installation existante détectée : ${PULID_MODELS_ROOT}"
else
  while true; do
    read -r -p "Utiliser l'emplacement par défaut ${DEFAULT_MODELS_ROOT} ? [O/n] " USE_DEFAULT
    case "${USE_DEFAULT:-o}" in
      o|O|oui|OUI|y|Y|yes|YES)
        PULID_MODELS_ROOT="${DEFAULT_MODELS_ROOT}"
        break
        ;;
      n|N|non|NON|no|NO)
        read -r -p "Chemin du dossier parent (ou d'un dossier déjà nommé PuLID_models) : " CUSTOM_MODELS_ROOT
        if [[ -n "${CUSTOM_MODELS_ROOT}" ]]; then
          PULID_MODELS_ROOT="$(normalize_models_root "${CUSTOM_MODELS_ROOT}")"
          break
        fi
        ;;
      *) echo "Répondez oui ou non." ;;
    esac
  done
fi

mkdir -p "${PULID_MODELS_ROOT}"

export PULID_MODELS_ROOT
export HF_HOME="${PULID_MODELS_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${PULID_MODELS_ROOT}/huggingface/hub"
export TRANSFORMERS_CACHE="${PULID_MODELS_ROOT}/huggingface/transformers"
export TORCH_HOME="${PULID_MODELS_ROOT}/torch"
export XDG_CACHE_HOME="${PULID_MODELS_ROOT}/other"
export MPLCONFIGDIR="${PULID_MODELS_ROOT}/other/matplotlib"
export UV_CACHE_DIR="${PULID_MODELS_ROOT}/other/uv-macos"
export UV_PYTHON_INSTALL_DIR="${PULID_MODELS_ROOT}/other/uv-python-macos"
export UV_LINK_MODE=copy
export NO_ALBUMENTATIONS_UPDATE=1

on_error() {
  local exit_code=$?
  echo >&2
  echo "[ERREUR] Installation macOS interrompue à la ligne ${BASH_LINENO[0]}." >&2
  echo "Corrigez l'erreur affichée ci-dessus puis relancez install_macos.sh." >&2
  exit "${exit_code}"
}
trap on_error ERR

cd "${PROJECT_DIR}"

mkdir -p \
  "${HF_HOME}" \
  "${HUGGINGFACE_HUB_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${TORCH_HOME}" \
  "${XDG_CACHE_HOME}" \
  "${MPLCONFIGDIR}" \
  "${UV_CACHE_DIR}" \
  "${UV_PYTHON_INSTALL_DIR}"

UV_EXE="$(command -v uv || true)"
if [[ -z "${UV_EXE}" ]]; then
  UV_INSTALL_DIR="${PULID_MODELS_ROOT}/other/uv-macos-bin"
  export UV_INSTALL_DIR
  mkdir -p "${UV_INSTALL_DIR}"
  echo "Installation de uv sous ${UV_INSTALL_DIR}..."
  curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/install.sh | sh
  UV_EXE="${UV_INSTALL_DIR}/uv"
fi

if [[ ! -x "${UV_EXE}" ]]; then
  echo "[ERREUR] Exécutable uv introuvable : ${UV_EXE}" >&2
  exit 1
fi

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Création de l'environnement Python 3.11..."
  "${UV_EXE}" venv --python 3.11 "${VENV_DIR}"
fi

PYTHON_VERSION="$("${VENV_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "${PYTHON_VERSION}" in
  3.11|3.12|3.13) ;;
  *)
    echo "[ERREUR] .venv utilise Python ${PYTHON_VERSION}; Python 3.11 à 3.13 est requis." >&2
    echo "Déplacez l'ancien .venv hors du projet puis relancez ce script." >&2
    exit 1
    ;;
esac

HOST_ARCH="$(uname -m)"
PYTHON_ARCH="$("${VENV_PYTHON}" -c 'import platform; print(platform.machine())')"
if [[ "${HOST_ARCH}" == "arm64" && "${PYTHON_ARCH}" != "arm64" ]]; then
  echo "[ERREUR] Le Mac est arm64 mais .venv utilise Python ${PYTHON_ARCH}." >&2
  echo "Déplacez l'ancien .venv hors du projet puis relancez ce script." >&2
  exit 1
fi

echo "Installation ou mise à jour du runtime GGUF Metal..."
if ! "${UV_EXE}" pip install \
  --python "${VENV_PYTHON}" \
  --extra-index-url "${LLAMA_CPP_METAL_INDEX}" \
  --only-binary llama-cpp-python \
  --reinstall-package llama-cpp-python \
  "llama-cpp-python>=0.3.16,<0.4"; then
  echo "La wheel Metal n'a pas pu être installée ; compilation locale Metal..."
  CMAKE_ARGS="-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_APPLE_SILICON_PROCESSOR=arm64 -DGGML_METAL=ON -DGGML_ACCELERATE=ON" \
    FORCE_CMAKE=1 \
    "${UV_EXE}" pip install \
      --python "${VENV_PYTHON}" \
      --index-url "https://pypi.org/simple" \
      --no-binary llama-cpp-python \
      --no-cache \
      "llama-cpp-python>=0.3.16,<0.4"
fi

echo "Installation ou mise à jour de PuLID et de ses extras..."
"${UV_EXE}" pip install \
  --python "${VENV_PYTHON}" \
  --extra-index-url "${LLAMA_CPP_METAL_INDEX}" \
  --only-binary llama-cpp-python \
  -e ".[inference,pulid,server,embeddings,dev]"

echo
echo "Installation ou réparation des modèles et configurations..."
"${VENV_DIR}/bin/pulid-install" \
  --models-root "${PULID_MODELS_ROOT}" \
  --sdxl ask

echo "Vérification des composants Python..."
"${VENV_PYTHON}" -c "import diffusers, fastapi, llama_cpp, torch, transformers; info = llama_cpp.llama_print_system_info().decode(); assert 'MTL' in info, 'Backend Metal absent de llama-cpp-python'; print('Python', '${PYTHON_VERSION}', '-', '${PYTHON_ARCH}'); print('PyTorch', torch.__version__, '- MPS disponible :', torch.backends.mps.is_available()); print('llama-cpp-python', llama_cpp.__version__, '- Metal OK')"
"${VENV_DIR}/bin/pulid-gen" --version
"${VENV_PYTHON}" -c "from pulid_app.config import load_config; config = load_config(); embedding = config.text_embedding; assert embedding is not None; assert embedding.checkpoint.is_file(), embedding.checkpoint; print('GGUF configuré :', embedding.checkpoint)"
"${VENV_DIR}/bin/pulid-gen" doctor
"${VENV_PYTHON}" scripts/inspect_models.py --show-cache-env --fail-on-internal-cache

trap - ERR
echo
echo "Installation macOS terminée."
echo "Activez l'environnement avec : source .venv/bin/activate"
echo "Tests unitaires : .venv/bin/python -m pytest"
echo "Serveur local : ./start_pulid_server.sh"
