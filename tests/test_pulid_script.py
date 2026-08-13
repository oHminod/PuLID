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

from test_pulid import (  # noqa: E402
    DEFAULT_REFERENCE,
    build_metadata,
    build_parser,
    resolve_sdxl_checkpoint,
)


def test_parser_defaults_to_noemie_webp() -> None:
    args = build_parser().parse_args([])

    assert args.reference == DEFAULT_REFERENCE
    assert args.reference.name == "noemie.webp"
    assert args.strength == 0.8
    assert args.steps == 20
    assert args.model is None
    assert args.method is None
    assert args.guidance_scale == 7.0


def test_parser_accepts_model_name_without_extension() -> None:
    args = build_parser().parse_args(["--model", "reaxl_v30"])

    assert args.model == "reaxl_v30"


def test_parser_accepts_custom_sampling_and_cfg() -> None:
    args = build_parser().parse_args(
        ["--method", "dpmpp_2m_sde_karras", "--cfg", "4.5"]
    )

    assert args.method == "dpmpp_2m_sde_karras"
    assert args.guidance_scale == 4.5


def test_resolve_sdxl_checkpoint_uses_configured_model_directory(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    checkpoint = resolve_sdxl_checkpoint(config, "reaxl_v30")

    assert checkpoint == tmp_path / "models" / "reaxl_v30.safetensors"


def test_resolve_sdxl_checkpoint_tolerates_extension(tmp_path: Path) -> None:
    config = _config(tmp_path)

    checkpoint = resolve_sdxl_checkpoint(config, "reaxl_v30.safetensors")

    assert checkpoint == tmp_path / "models" / "reaxl_v30.safetensors"


def test_resolve_sdxl_checkpoint_rejects_a_path(tmp_path: Path) -> None:
    config = _config(tmp_path)

    try:
        resolve_sdxl_checkpoint(config, "sdxl/reaxl_v30")
    except ValueError as exc:
        assert "uniquement un nom" in str(exc)
    else:
        raise AssertionError("Un chemin de modèle aurait dû être refusé.")


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
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


def test_metadata_contains_phase_9_manifest(tmp_path: Path) -> None:
    config = _config(tmp_path)
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
    assert metadata["guidance_scale"] == 7.0
    assert metadata["sampling_method"] == "default"
    assert metadata["sdxl_checkpoint"] == str(config.sdxl.checkpoint)
    assert metadata["pulid_checkpoint"] == str(config.pulid.checkpoint)
    assert metadata["device"] == "mps"
    assert metadata["vae"] == "integrated"
