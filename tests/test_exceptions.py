from __future__ import annotations

from io import StringIO
from pathlib import Path

from rich.console import Console

from pulid_app.cli import build_parser, run_generate
from pulid_app.exceptions import (
    EmbeddingError,
    ExternalDriveNotMountedError,
    FaceNotDetectedError,
    GenerationError,
    ModelLoadError,
    ModelNotFoundError,
    MultipleFacesDetectedError,
    UnsupportedDeviceError,
    actionable_error,
)
from pulid_app.paths import require_models_root


def _write_config(path: Path, models_root: Path) -> None:
    path.write_text(
        f"""
models_root: {models_root}
sdxl:
  checkpoint: missing.safetensors
  config_dir: config
pulid:
  checkpoint: pulid.safetensors
insightface:
  model_root: .
  model_name: antelopev2
outputs_dir: ./outputs
identity_cache_dir: ./cache/identity
device:
  preferred: cpu
  dtype: float32
""",
        encoding="utf-8",
    )


def test_public_error_types_are_instantiable() -> None:
    errors = (
        ModelNotFoundError("modèle absent"),
        ExternalDriveNotMountedError("/Volumes/SSD/Documents/PuLID_models"),
        FaceNotDetectedError("aucun visage"),
        MultipleFacesDetectedError(2),
        UnsupportedDeviceError("device inconnu"),
        ModelLoadError("chargement impossible"),
        GenerationError("génération impossible"),
        EmbeddingError("embedding impossible"),
    )

    assert all(isinstance(error, RuntimeError) for error in errors)
    assert isinstance(errors[3], MultipleFacesDetectedError)
    assert errors[3].count == 2
    assert isinstance(errors[4], ValueError)


def test_actionable_error_prefers_root_domain_cause() -> None:
    cause = ModelNotFoundError("checkpoint absent")
    try:
        raise GenerationError("pipeline interrompu") from cause
    except GenerationError as exc:
        label, selected = actionable_error(exc)

    assert label == "ModelNotFoundError"
    assert selected is cause


def test_require_models_root_reports_unmounted_external_drive() -> None:
    missing = Path("/Volumes/PuLID-test-absent/Documents/PuLID_models")

    try:
        require_models_root(missing)
    except ExternalDriveNotMountedError as exc:
        message = str(exc)
    else:  # pragma: no cover - ce volume ne doit pas exister en CI
        raise AssertionError("Le volume de test inattendu est monté.")

    assert str(missing) in message
    assert "SSD externe est monté" in message


def test_generate_cli_names_missing_model_and_suggests_correction(
    tmp_path: Path,
) -> None:
    models_root = tmp_path / "models"
    models_root.mkdir()
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, models_root)
    args = build_parser().parse_args(
        [
            "generate",
            "--config",
            str(config_path),
            "--reference",
            str(tmp_path / "noemie.webp"),
            "--prompt",
            "portrait",
        ]
    )
    output = StringIO()

    exit_code = run_generate(
        args,
        Console(file=output, force_terminal=False, width=200),
    )

    assert exit_code == 2
    rendered = output.getvalue()
    assert "ModelNotFoundError:" in rendered
    assert "Corrigez sdxl.checkpoint" in rendered
