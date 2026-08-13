"""Tests d'intégration opt-in utilisant les modèles locaux du SSD externe."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import pytest

from pulid_app.config import PROJECT_ROOT, load_config
from pulid_app.device import get_best_device
from pulid_app.models.identity_encoder import IdentityEncoder
from pulid_app.models.sdxl import SDXLModel
from pulid_app.pipeline.generator import ImageGenerator


REFERENCE_IMAGE = PROJECT_ROOT / "inputs" / "noemie.webp"


def _require_opt_in(variable: str) -> None:
    if os.environ.get(variable) != "1":
        pytest.skip(f"Définissez {variable}=1 pour exécuter ce test d'intégration.")


def _require_accelerator() -> str:
    device = get_best_device()
    if device not in {"mps", "cuda"}:
        pytest.skip("Ce test nécessite un accélérateur MPS ou CUDA.")
    return device


@pytest.mark.integration
def test_insightface_encodes_example_image(tmp_path: Path) -> None:
    _require_opt_in("PULID_RUN_INTEGRATION")
    config = replace(load_config(), identity_cache_dir=tmp_path / "identity")

    identity = IdentityEncoder.from_config(config).encode_image(
        REFERENCE_IMAGE,
        identity_id="noemie",
        force_recompute=True,
    )

    assert identity.id == "noemie"
    assert identity.face_embedding.shape == (512,)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.gpu
def test_sdxl_generates_minimal_image() -> None:
    _require_opt_in("PULID_RUN_SLOW")
    config = load_config()
    model = SDXLModel.from_config(config, device=_require_accelerator())
    try:
        result = model.generate(
            prompt="portrait photo",
            steps=1,
            width=512,
            height=512,
            seed=42,
        )
    finally:
        model.close()

    assert result.image.size == (512, 512)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.gpu
def test_pulid_generates_minimal_image(tmp_path: Path) -> None:
    _require_opt_in("PULID_RUN_SLOW")
    config = replace(
        load_config(),
        outputs_dir=tmp_path / "outputs",
        identity_cache_dir=tmp_path / "identity",
    )
    generator = ImageGenerator(
        config,
        device=_require_accelerator(),
        allow_downloads=False,
    )
    try:
        identity = generator.encode_identity(
            REFERENCE_IMAGE,
            identity_id="noemie",
        )
        result = generator.generate(
            prompt="portrait photo",
            identity=identity,
            steps=1,
            width=512,
            height=512,
            seed=42,
        )
    finally:
        generator.close()

    assert result.png_path.is_file()
    assert result.json_path.is_file()
