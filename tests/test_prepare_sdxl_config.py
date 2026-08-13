from __future__ import annotations

from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
import sys

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prepare_sdxl_config import _validate_config_tree  # noqa: E402


REQUIRED_CONFIG_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder_2/config.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/vocab.json",
    "tokenizer_2/merges.txt",
    "unet/config.json",
    "vae/config.json",
)


def _create_config_tree(root: Path) -> None:
    for relative in REQUIRED_CONFIG_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")


def test_validate_config_tree_accepts_config_only(tmp_path: Path) -> None:
    _create_config_tree(tmp_path)

    count, size = _validate_config_tree(tmp_path)

    assert count == len(REQUIRED_CONFIG_FILES)
    assert size == len(REQUIRED_CONFIG_FILES) * 2


def test_validate_config_tree_rejects_model_weights(tmp_path: Path) -> None:
    _create_config_tree(tmp_path)
    (tmp_path / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")

    with pytest.raises(RuntimeError, match="poids interdits"):
        _validate_config_tree(tmp_path)
