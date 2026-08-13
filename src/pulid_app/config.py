"""Chargement et validation de la configuration de l'application."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
DEFAULT_PULID_REVISION = "1aa2fc7df4bf51080df39f355f9abdc1cbfefbaa"


class ConfigError(ValueError):
    """Configuration absente, invalide ou incohérente."""


@dataclass(frozen=True)
class SDXLConfig:
    checkpoint: Path
    config_dir: Path | None = None


@dataclass(frozen=True)
class PuLIDConfig:
    checkpoint: Path
    source_dir: Path | None = None
    revision: str = DEFAULT_PULID_REVISION
    eva_clip_model: str = "EVA02-CLIP-L-14-336"
    eva_clip_pretrained: str = "eva_clip"
    facexlib_root: Path | None = None


@dataclass(frozen=True)
class InsightFaceConfig:
    model_root: Path
    model_name: str

    @property
    def model_dir(self) -> Path:
        return self.model_root / self.model_name


@dataclass(frozen=True)
class DeviceConfig:
    preferred: str = "mps"
    dtype: str = "float16"


@dataclass(frozen=True)
class AppConfig:
    models_root: Path
    sdxl: SDXLConfig
    pulid: PuLIDConfig
    insightface: InsightFaceConfig
    outputs_dir: Path
    identity_cache_dir: Path
    device: DeviceConfig
    source_path: Path


def _require_mapping(value: Any, key: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"La section '{key}' doit être un objet YAML.")
    return value


def _require_text(mapping: Mapping[str, Any], key: str, context: str = "") -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        qualified = f"{context}.{key}" if context else key
        raise ConfigError(f"La clé '{qualified}' doit contenir une chaîne non vide.")
    return value.strip()


def _absolute(path: str | Path, base: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve(strict=False)


def _model_path(path: str, models_root: Path) -> Path:
    return _absolute(path, models_root)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Charge la configuration et résout tous les chemins de façon déterministe."""

    selected = path or os.environ.get("PULID_CONFIG") or DEFAULT_CONFIG_PATH
    source_path = _absolute(selected, PROJECT_ROOT)
    if not source_path.is_file():
        raise ConfigError(f"Fichier de configuration introuvable : {source_path}")

    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML invalide dans {source_path} : {exc}") from exc

    root = _require_mapping(raw, "racine")
    raw_models_root = os.environ.get("PULID_MODELS_ROOT") or _require_text(
        root, "models_root"
    )
    models_root = _absolute(raw_models_root, PROJECT_ROOT)

    sdxl = _require_mapping(root.get("sdxl"), "sdxl")
    pulid = _require_mapping(root.get("pulid"), "pulid")
    insightface = _require_mapping(root.get("insightface"), "insightface")
    device = _require_mapping(root.get("device", {}), "device")

    outputs_dir = _absolute(_require_text(root, "outputs_dir"), PROJECT_ROOT)
    identity_cache_dir = _absolute(
        _require_text(root, "identity_cache_dir"), PROJECT_ROOT
    )
    insightface_root = _model_path(
        _require_text(insightface, "model_root", "insightface"), models_root
    )

    return AppConfig(
        models_root=models_root,
        sdxl=SDXLConfig(
            checkpoint=_model_path(
                _require_text(sdxl, "checkpoint", "sdxl"), models_root
            ),
            config_dir=(
                _model_path(str(sdxl["config_dir"]), models_root)
                if isinstance(sdxl.get("config_dir"), str)
                and str(sdxl["config_dir"]).strip()
                else None
            ),
        ),
        pulid=PuLIDConfig(
            checkpoint=_model_path(
                _require_text(pulid, "checkpoint", "pulid"), models_root
            ),
            source_dir=_model_path(
                str(pulid.get("source_dir", "sources/PuLID")), models_root
            ),
            revision=str(
                os.environ.get("PULID_OFFICIAL_REVISION")
                or pulid.get("revision", DEFAULT_PULID_REVISION)
            ).strip(),
            eva_clip_model=str(
                pulid.get("eva_clip_model", "EVA02-CLIP-L-14-336")
            ).strip(),
            eva_clip_pretrained=str(
                pulid.get("eva_clip_pretrained", "eva_clip")
            ).strip(),
            facexlib_root=_model_path(
                str(pulid.get("facexlib_root", "facexlib/weights")), models_root
            ),
        ),
        insightface=InsightFaceConfig(
            model_root=insightface_root,
            model_name=_require_text(insightface, "model_name", "insightface"),
        ),
        outputs_dir=outputs_dir,
        identity_cache_dir=identity_cache_dir,
        device=DeviceConfig(
            preferred=str(device.get("preferred", "mps")),
            dtype=str(device.get("dtype", "float16")),
        ),
        source_path=source_path,
    )
