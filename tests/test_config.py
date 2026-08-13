from __future__ import annotations

from pathlib import Path

import pytest

from pulid_app.config import ConfigError, load_config


def _write_config(path: Path, models_root: Path) -> None:
    path.write_text(
        f"""
models_root: {models_root}
sdxl:
  checkpoint: sdxl/model.safetensors
  config_dir: sdxl/config
pulid:
  checkpoint: pulid/model.safetensors
  source_dir: sources/PuLID
  revision: 0123456789abcdef0123456789abcdef01234567
  eva_clip_model: EVA02-CLIP-L-14-336
  eva_clip_pretrained: eva_clip
  facexlib_root: facexlib
insightface:
  model_root: insightface
  model_name: antelopev2
outputs_dir: ./outputs
identity_cache_dir: ./cache/identity
device:
  preferred: cpu
  dtype: float32
  offload_strategy: model_cpu_offload
""",
        encoding="utf-8",
    )


def test_load_config_resolves_model_paths_from_models_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    models_root = tmp_path / "models"
    _write_config(config_path, models_root)
    monkeypatch.delenv("PULID_MODELS_ROOT", raising=False)

    config = load_config(config_path)

    assert config.models_root == models_root
    assert config.sdxl.checkpoint == models_root / "sdxl/model.safetensors"
    assert config.sdxl.config_dir == models_root / "sdxl/config"
    assert config.pulid.checkpoint == models_root / "pulid/model.safetensors"
    assert config.pulid.source_dir == models_root / "sources/PuLID"
    assert config.pulid.revision == "0123456789abcdef0123456789abcdef01234567"
    assert config.pulid.facexlib_root == models_root / "facexlib"
    assert config.insightface.model_dir == models_root / "insightface/antelopev2"
    assert config.device.offload_strategy == "model_cpu_offload"


def test_models_root_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, tmp_path / "original")
    override = tmp_path / "override"
    monkeypatch.setenv("PULID_MODELS_ROOT", str(override))

    config = load_config(config_path)

    assert config.models_root == override
    assert config.sdxl.checkpoint == override / "sdxl/model.safetensors"


def test_missing_required_key_is_actionable(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("models_root: /tmp/models\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="sdxl"):
        load_config(config_path)
