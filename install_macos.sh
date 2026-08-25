#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PULID_MODELS_ROOT="${PULID_MODELS_ROOT:-/Volumes/SSD/Documents/PuLID_models}"
VENV_DIR="${PROJECT_DIR}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"
LLAMA_CPP_CPU_INDEX="https://abetlen.github.io/llama-cpp-python/whl/cpu"

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

if [[ ! -d "${PULID_MODELS_ROOT}" ]]; then
  echo "[ERREUR] Racine de modèles introuvable :" >&2
  echo "  ${PULID_MODELS_ROOT}" >&2
  echo "Montez le SSD ou surchargez PULID_MODELS_ROOT avant de relancer." >&2
  exit 1
fi

EMBEDDING_CHECKPOINT="${PULID_MODELS_ROOT}/text_embedding/bge-m3-Q8_0.gguf"
if [[ ! -f "${EMBEDDING_CHECKPOINT}" ]]; then
  echo "[ERREUR] Modèle d'embedding GGUF introuvable :" >&2
  echo "  ${EMBEDDING_CHECKPOINT}" >&2
  echo "Placez le fichier dans text_embedding sous models_root." >&2
  exit 1
fi

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

echo "Installation ou mise à jour du runtime GGUF CPU..."
if ! "${UV_EXE}" pip install \
  --python "${VENV_PYTHON}" \
  --extra-index-url "${LLAMA_CPP_CPU_INDEX}" \
  --only-binary llama-cpp-python \
  "llama-cpp-python>=0.3.16,<0.4"; then
  echo "La wheel CPU n'a pas pu être installée ; compilation locale CPU via Accelerate..."
  CMAKE_ARGS="-DGGML_METAL=OFF -DGGML_ACCELERATE=ON" \
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
  --extra-index-url "${LLAMA_CPP_CPU_INDEX}" \
  --only-binary llama-cpp-python \
  -e ".[inference,pulid,server,embeddings,dev]"

echo "Vérification des composants Python..."
"${VENV_PYTHON}" -c "import diffusers, fastapi, llama_cpp, torch, transformers; print('Python', '${PYTHON_VERSION}', '-', '${PYTHON_ARCH}'); print('PyTorch', torch.__version__, '- MPS disponible :', torch.backends.mps.is_available()); print('llama-cpp-python', llama_cpp.__version__)"
"${VENV_DIR}/bin/pulid-gen" --version
"${VENV_PYTHON}" -c "from pulid_app.config import load_config; config = load_config(); embedding = config.text_embedding; assert embedding is not None; assert embedding.checkpoint.is_file(), embedding.checkpoint; print('GGUF configuré :', embedding.checkpoint)"

trap - ERR
echo
echo "Installation macOS terminée."
echo "Activez l'environnement avec : source .venv/bin/activate"
echo "Tests unitaires : .venv/bin/python -m pytest"
echo "Serveur local : ./start_pulid_server.sh"
