"""Chemins des modèles, caches et artefacts locaux."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Mapping

from pulid_app.config import AppConfig


ANTELOPEV2_REQUIRED_FILES = frozenset(
    {
        "1k3d68.onnx",
        "2d106det.onnx",
        "genderage.onnx",
        "glintr100.onnx",
        "scrfd_10g_bnkps.onnx",
    }
)


@dataclass(frozen=True)
class ModelInventory:
    pulid_checkpoints: tuple[Path, ...]
    antelope_dir: Path | None
    antelope_missing_files: tuple[str, ...]
    sdxl_candidates: tuple[Path, ...]


def external_cache_paths(models_root: Path) -> dict[str, Path]:
    """Retourne les emplacements imposés aux bibliothèques de modèles."""

    return {
        "HF_HOME": models_root / "huggingface",
        "HUGGINGFACE_HUB_CACHE": models_root / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": models_root / "huggingface" / "transformers",
        "TORCH_HOME": models_root / "torch",
        "XDG_CACHE_HOME": models_root / "other",
        "MPLCONFIGDIR": models_root / "other" / "matplotlib",
    }


def configure_external_model_caches(models_root: Path) -> Mapping[str, str]:
    """Redirige les caches lourds avant tout import d'une bibliothèque ML.

    La fonction est idempotente et remplace volontairement toute valeur héritée :
    la racine de modèles configurée reste l'unique source de vérité.
    """

    configured: dict[str, str] = {}
    for name, path in external_cache_paths(models_root).items():
        value = str(path.resolve(strict=False))
        os.environ[name] = value
        configured[name] = value
    # Albumentations effectue sinon une requête PyPI à chaque nouvel environnement.
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    configured["NO_ALBUMENTATIONS_UPDATE"] = "1"
    return configured


def cache_env_violations(
    models_root: Path,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Signale les caches effectifs absents ou situés hors de ``models_root``."""

    selected = os.environ if environ is None else environ
    root = models_root.expanduser().resolve(strict=False)
    violations: list[str] = []
    for name in external_cache_paths(root):
        raw_value = selected.get(name)
        if not raw_value:
            violations.append(f"{name} n'est pas défini")
            continue
        path = Path(raw_value).expanduser().resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError:
            violations.append(f"{name} pointe hors de models_root : {path}")
    return tuple(violations)


def _unique_sorted(paths: list[Path]) -> tuple[Path, ...]:
    return tuple(sorted(set(paths), key=lambda item: str(item).casefold()))


def inspect_models(config: AppConfig) -> ModelInventory:
    """Inventorie les modèles locaux sans les charger ni accéder au réseau."""

    root = config.models_root
    if not root.is_dir():
        return ModelInventory((), None, (), ())

    safetensors = list(root.rglob("*.safetensors"))

    pulid_paths = [path for path in safetensors if "pulid" in path.name.casefold()]
    if config.pulid.checkpoint.is_file():
        pulid_paths.append(config.pulid.checkpoint)

    sdxl_paths = [
        path
        for path in safetensors
        if "pulid" not in path.name.casefold()
        and "vae" not in path.name.casefold().replace("bakedvae", "")
    ]
    if config.sdxl.checkpoint.is_file():
        sdxl_paths.append(config.sdxl.checkpoint)

    antelope_dir = config.insightface.model_dir
    if not antelope_dir.is_dir():
        matches = sorted(
            (path for path in root.rglob(config.insightface.model_name) if path.is_dir()),
            key=lambda item: str(item).casefold(),
        )
        antelope_dir = matches[0] if matches else None

    missing: tuple[str, ...] = ()
    if antelope_dir is not None:
        present = {path.name for path in antelope_dir.glob("*.onnx")}
        missing = tuple(sorted(ANTELOPEV2_REQUIRED_FILES - present))

    return ModelInventory(
        pulid_checkpoints=_unique_sorted(pulid_paths),
        antelope_dir=antelope_dir,
        antelope_missing_files=missing,
        sdxl_candidates=_unique_sorted(sdxl_paths),
    )


def ensure_writable_directory(path: Path) -> None:
    """Crée le dossier si nécessaire et vérifie réellement son écriture."""

    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise NotADirectoryError(f"Le chemin n'est pas un dossier : {path}")
    try:
        with tempfile.NamedTemporaryFile(prefix=".pulid-write-test-", dir=path):
            pass
    except OSError as exc:
        raise PermissionError(f"Le dossier n'est pas accessible en écriture : {path}") from exc


def resolve_sdxl_checkpoint(config: AppConfig, model_name: str | None) -> Path:
    """Résout un checkpoint nommé à côté du modèle SDXL configuré."""

    if model_name is None:
        return config.sdxl.checkpoint

    selected = model_name.strip()
    if not selected:
        raise ValueError("L'option --model ne peut pas être vide.")
    if selected in {".", ".."} or "/" in selected or "\\" in selected:
        raise ValueError(
            "L'option --model attend uniquement un nom de modèle, pas un chemin."
        )

    lowered = selected.casefold()
    for extension in (".safetensors", ".safetensor"):
        if lowered.endswith(extension):
            selected = selected[: -len(extension)]
            break
    if not selected:
        raise ValueError("Le nom fourni à --model est invalide.")

    return (
        config.sdxl.checkpoint.parent / f"{selected}.safetensors"
    ).resolve(strict=False)
