from __future__ import annotations

from io import StringIO
from pathlib import Path

from rich.console import Console

from pulid_app.cli import run_inspection
from pulid_app.paths import ANTELOPEV2_REQUIRED_FILES


def test_full_inspection_succeeds(tmp_path: Path) -> None:
    models = tmp_path / "models"
    antelope = models / "antelopev2"
    antelope.mkdir(parents=True)
    (models / "realvisxlV50_v50LightningBakedvae.safetensors").touch()
    (models / "pulid_v1.1.safetensors").touch()
    for name in ANTELOPEV2_REQUIRED_FILES:
        (antelope / name).touch()

    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
models_root: {models}
sdxl:
  checkpoint: realvisxlV50_v50LightningBakedvae.safetensors
pulid:
  checkpoint: pulid_v1.1.safetensors
insightface:
  model_root: .
  model_name: antelopev2
outputs_dir: {tmp_path / 'outputs'}
identity_cache_dir: {tmp_path / 'cache' / 'identity'}
device:
  preferred: mps
  dtype: float16
""",
        encoding="utf-8",
    )
    output = StringIO()

    result = run_inspection(config, Console(file=output, force_terminal=False))

    assert result == 0
    assert "Inspection réussie" in output.getvalue()
    assert "VAE intégré" in output.getvalue()

