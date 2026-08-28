from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from pulid_app.config import (
    AppConfig,
    DeviceConfig,
    InsightFaceConfig,
    PuLIDConfig,
    SDXLConfig,
    TextEmbeddingConfig,
)
from pulid_app.doctor import (
    FACEXLIB_REQUIRED_FILES,
    _nearest_mount,
    build_doctor_report,
    print_doctor_report,
)
from pulid_app.models.pulid_assets import REQUIRED_SOURCE_FILES, SOURCE_MARKER
from pulid_app.models.sdxl import REQUIRED_SDXL_CONFIG_FILES
from pulid_app.paths import ANTELOPEV2_REQUIRED_FILES


def _ready_config(tmp_path: Path) -> AppConfig:
    models = tmp_path / "models"
    models.mkdir()
    sdxl_checkpoint = models / "sdxl.safetensors"
    pulid_checkpoint = models / "pulid.safetensors"
    sdxl_checkpoint.touch()
    pulid_checkpoint.touch()

    antelope = models / "antelopev2"
    antelope.mkdir()
    for name in ANTELOPEV2_REQUIRED_FILES:
        (antelope / name).touch()

    sdxl_config = models / "sdxl-config"
    for relative in REQUIRED_SDXL_CONFIG_FILES:
        path = sdxl_config / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    revision = "a" * 40
    source = models / "sources" / "PuLID"
    for relative in REQUIRED_SOURCE_FILES:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (source / SOURCE_MARKER).write_text(
        json.dumps({"revision": revision}),
        encoding="utf-8",
    )

    facexlib = models / "facexlib" / "weights"
    facexlib.mkdir(parents=True)
    for name in FACEXLIB_REQUIRED_FILES:
        (facexlib / name).touch()

    eva = (
        models
        / "huggingface"
        / "hub"
        / "models--QuanSun--EVA-CLIP"
        / "snapshots"
        / "revision"
        / "eva.pt"
    )
    eva.parent.mkdir(parents=True)
    eva.touch()

    text_embedding = models / "text_embedding" / "bge-m3-Q8_0.gguf"
    text_embedding.parent.mkdir()
    text_embedding.touch()

    return AppConfig(
        models_root=models,
        sdxl=SDXLConfig(sdxl_checkpoint, sdxl_config),
        pulid=PuLIDConfig(
            pulid_checkpoint,
            source_dir=source,
            revision=revision,
            facexlib_root=facexlib,
        ),
        insightface=InsightFaceConfig(models, "antelopev2"),
        outputs_dir=tmp_path / "outputs",
        identity_cache_dir=tmp_path / "cache" / "identity",
        device=DeviceConfig(),
        source_path=tmp_path / "config.yaml",
        text_embedding=TextEmbeddingConfig(text_embedding),
    )


def test_doctor_reports_ready_environment(tmp_path: Path) -> None:
    config = _ready_config(tmp_path)
    device = SimpleNamespace(
        selected_device="mps",
        mps_available=True,
        cuda_available=False,
    )

    report = build_doctor_report(
        config,
        device_reporter=lambda: device,
        version_resolver=lambda _distribution: "1.2.3",
    )

    assert report.healthy is True
    assert not report.errors
    assert {check.name for check in report.checks} >= {
        "SSD / models_root",
        "Caches externes",
        "Checkpoint SDXL",
        "Checkpoint PuLID",
        "Modèle embedding GGUF",
        "AntelopeV2",
        "Configuration SDXL",
        "Runtime PuLID",
        "FaceXLib",
        "EVA-CLIP",
        "Dossier outputs",
        "Accélérateur",
        "Version torch",
        "Version llama-cpp-python",
    }

    output = StringIO()
    print_doctor_report(report, Console(file=output, force_terminal=False))
    assert "Doctor réussi" in output.getvalue()


def test_nearest_mount_uses_anchor_when_is_mount_is_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unsupported_is_mount(_path: Path) -> bool:
        raise NotImplementedError("Path.is_mount() is unsupported on this system")

    monkeypatch.setattr(Path, "is_mount", unsupported_is_mount)

    assert _nearest_mount(tmp_path / "models") == Path(tmp_path.anchor)


def test_doctor_reports_missing_models_root(tmp_path: Path) -> None:
    config = _ready_config(tmp_path)
    missing = tmp_path / "missing-models"
    config = AppConfig(
        models_root=missing,
        sdxl=SDXLConfig(missing / "sdxl.safetensors", missing / "config"),
        pulid=PuLIDConfig(missing / "pulid.safetensors"),
        insightface=InsightFaceConfig(missing, "antelopev2"),
        outputs_dir=tmp_path / "outputs-2",
        identity_cache_dir=tmp_path / "cache-2",
        device=config.device,
        source_path=config.source_path,
    )
    device = SimpleNamespace(
        selected_device="cpu",
        mps_available=False,
        cuda_available=False,
    )

    report = build_doctor_report(
        config,
        device_reporter=lambda: device,
        version_resolver=lambda _distribution: "1.2.3",
    )

    assert report.healthy is False
    assert any(check.name == "SSD / models_root" for check in report.errors)
    assert any(check.name == "Checkpoint SDXL" for check in report.errors)


def test_doctor_can_tolerate_deferred_sdxl_during_installation(
    tmp_path: Path,
) -> None:
    config = _ready_config(tmp_path)
    config.sdxl.checkpoint.unlink()
    device = SimpleNamespace(
        selected_device="cpu",
        mps_available=False,
        cuda_available=False,
    )

    report = build_doctor_report(
        config,
        device_reporter=lambda: device,
        version_resolver=lambda _distribution: "1.2.3",
        allow_missing_sdxl=True,
    )

    sdxl_check = next(
        check for check in report.checks if check.name == "Checkpoint SDXL"
    )
    assert report.healthy is True
    assert sdxl_check.status == "warning"
    assert "Relancez `pulid-install`" in sdxl_check.details
