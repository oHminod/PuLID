from __future__ import annotations

from pathlib import Path

from pulid_app.config import (
    AppConfig,
    DeviceConfig,
    InsightFaceConfig,
    PuLIDConfig,
    SDXLConfig,
)
from pulid_app.paths import (
    ANTELOPEV2_REQUIRED_FILES,
    configure_external_model_caches,
    ensure_writable_directory,
    inspect_models,
)


def _config(tmp_path: Path) -> AppConfig:
    models = tmp_path / "models"
    return AppConfig(
        models_root=models,
        sdxl=SDXLConfig(models / "realvisxl.safetensors"),
        pulid=PuLIDConfig(models / "pulid_v1.1.safetensors"),
        insightface=InsightFaceConfig(models, "antelopev2"),
        outputs_dir=tmp_path / "outputs",
        identity_cache_dir=tmp_path / "cache" / "identity",
        device=DeviceConfig(),
        source_path=tmp_path / "config.yaml",
    )


def test_inspect_models_finds_expected_inventory(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.models_root.mkdir()
    config.sdxl.checkpoint.touch()
    config.pulid.checkpoint.touch()
    config.insightface.model_dir.mkdir()
    for name in ANTELOPEV2_REQUIRED_FILES:
        (config.insightface.model_dir / name).touch()

    inventory = inspect_models(config)

    assert inventory.sdxl_candidates == (config.sdxl.checkpoint,)
    assert inventory.pulid_checkpoints == (config.pulid.checkpoint,)
    assert inventory.antelope_dir == config.insightface.model_dir
    assert inventory.antelope_missing_files == ()


def test_inspect_models_reports_missing_antelope_file(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.insightface.model_dir.mkdir(parents=True)
    missing = sorted(ANTELOPEV2_REQUIRED_FILES)[0]
    for name in ANTELOPEV2_REQUIRED_FILES - {missing}:
        (config.insightface.model_dir / name).touch()

    inventory = inspect_models(config)

    assert inventory.antelope_missing_files == (missing,)


def test_configure_external_model_caches(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HF_HOME", "/an/internal/cache")

    configured = configure_external_model_caches(tmp_path)

    assert configured["HF_HOME"] == str(tmp_path / "huggingface")
    assert configured["TORCH_HOME"] == str(tmp_path / "torch")
    assert configured["MPLCONFIGDIR"] == str(tmp_path / "other" / "matplotlib")
    assert configured["NO_ALBUMENTATIONS_UPDATE"] == "1"


def test_ensure_writable_directory_creates_directory(tmp_path: Path) -> None:
    target = tmp_path / "new" / "outputs"

    ensure_writable_directory(target)

    assert target.is_dir()
    assert list(target.iterdir()) == []
