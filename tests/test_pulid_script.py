from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

from pulid_app.config import (
    AppConfig,
    DeviceConfig,
    InsightFaceConfig,
    PuLIDConfig,
    SDXLConfig,
)


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from test_pulid import DEFAULT_REFERENCE, build_metadata, build_parser  # noqa: E402


def test_parser_defaults_to_noemie_webp() -> None:
    args = build_parser().parse_args([])

    assert args.reference == DEFAULT_REFERENCE
    assert args.reference.name == "noemie.webp"
    assert args.strength == 0.8
    assert args.steps == 20


def test_metadata_contains_phase_9_manifest(tmp_path: Path) -> None:
    config = AppConfig(
        models_root=tmp_path / "models",
        sdxl=SDXLConfig(tmp_path / "models" / "realvisxl.safetensors"),
        pulid=PuLIDConfig(
            tmp_path / "models" / "pulid_v1.1.safetensors",
            source_dir=tmp_path / "models" / "sources" / "PuLID",
            revision="a" * 40,
        ),
        insightface=InsightFaceConfig(tmp_path / "models", "antelopev2"),
        outputs_dir=tmp_path / "outputs",
        identity_cache_dir=tmp_path / "cache",
        device=DeviceConfig(),
        source_path=tmp_path / "config.yaml",
    )
    args = build_parser().parse_args(
        [
            "--reference",
            "noemie.webp",
            "--prompt",
            "portrait",
            "--seed",
            "7",
            "--strength",
            "0.9",
        ]
    )
    result = SimpleNamespace(
        seed=7,
        device="mps",
        dtype="float16",
        dtype_fallback_used=False,
        duration_seconds=12.5,
    )

    metadata = build_metadata(
        args=args,
        config=config,
        reference=(tmp_path / "noemie.webp"),
        result=result,
        identity_duration_seconds=3.0,
        total_duration_seconds=15.5,
    )

    assert metadata["reference_image"] == str(tmp_path / "noemie.webp")
    assert metadata["prompt"] == "portrait"
    assert metadata["seed"] == 7
    assert metadata["identity_strength"] == 0.9
    assert metadata["sdxl_checkpoint"] == str(config.sdxl.checkpoint)
    assert metadata["pulid_checkpoint"] == str(config.pulid.checkpoint)
    assert metadata["device"] == "mps"
    assert metadata["vae"] == "integrated"
