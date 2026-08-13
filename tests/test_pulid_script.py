from __future__ import annotations

from pathlib import Path
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
    assert args.sigmas == "normal"
    assert args.guidance_scale == 7.0


def test_parser_accepts_model_name_without_extension() -> None:
    args = build_parser().parse_args(["--model", "reaxl_v30"])

    assert args.model == "reaxl_v30"


def test_parser_accepts_custom_sampling_and_cfg() -> None:
    args = build_parser().parse_args(
        [
            "--method",
            "dpmpp_2m_sde",
            "--sigmas",
            "karras",
            "--cfg",
            "4.5",
        ]
    )

    assert args.method == "dpmpp_2m_sde"
    assert args.sigmas == "karras"
    assert args.guidance_scale == 4.5


def test_resolve_sdxl_checkpoint_uses_configured_model_directory(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    checkpoint = resolve_sdxl_checkpoint(config, "reaxl_v30")

    assert checkpoint == tmp_path / "models" / "checkpoints" / "reaxl_v30.safetensors"


def test_resolve_sdxl_checkpoint_tolerates_extension(tmp_path: Path) -> None:
    config = _config(tmp_path)

    checkpoint = resolve_sdxl_checkpoint(config, "reaxl_v30.safetensors")

    assert checkpoint == tmp_path / "models" / "checkpoints" / "reaxl_v30.safetensors"


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
        sdxl=SDXLConfig(
            tmp_path / "models" / "checkpoints" / "realvisxl.safetensors"
        ),
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
