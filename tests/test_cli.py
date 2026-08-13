from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from rich.console import Console

from pulid_app.cli import (
    build_parser,
    run_benchmark,
    run_encode,
    run_generate,
    run_inspection,
)
from pulid_app.identity import CharacterIdentity
from pulid_app.paths import ANTELOPEV2_REQUIRED_FILES


def _write_cli_config(tmp_path: Path) -> tuple[Path, Path]:
    models = tmp_path / "models"
    models.mkdir()
    checkpoints = models / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "sdxl.safetensors").touch()
    (models / "pulid.safetensors").touch()
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
models_root: {models}
sdxl:
  checkpoint: checkpoints/sdxl.safetensors
  config_dir: sdxl-config
pulid:
  checkpoint: pulid.safetensors
  source_dir: sources/PuLID
insightface:
  model_root: .
  model_name: antelopev2
outputs_dir: {tmp_path / 'outputs'}
identity_cache_dir: {tmp_path / 'cache' / 'identity'}
device:
  preferred: cpu
  dtype: float32
""",
        encoding="utf-8",
    )
    return config, models


def test_full_inspection_succeeds(tmp_path: Path) -> None:
    models = tmp_path / "models"
    checkpoints = models / "checkpoints"
    antelope = models / "antelopev2"
    antelope.mkdir(parents=True)
    checkpoints.mkdir()
    (checkpoints / "realvisxlV50_v50LightningBakedvae.safetensors").touch()
    (models / "pulid_v1.1.safetensors").touch()
    for name in ANTELOPEV2_REQUIRED_FILES:
        (antelope / name).touch()

    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
models_root: {models}
sdxl:
  checkpoint: checkpoints/realvisxlV50_v50LightningBakedvae.safetensors
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


def test_inspection_can_show_and_validate_external_caches(tmp_path: Path) -> None:
    models = tmp_path / "models"
    checkpoints = models / "checkpoints"
    antelope = models / "antelopev2"
    antelope.mkdir(parents=True)
    checkpoints.mkdir()
    (checkpoints / "sdxl.safetensors").touch()
    (models / "pulid_v1.1.safetensors").touch()
    for name in ANTELOPEV2_REQUIRED_FILES:
        (antelope / name).touch()
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
models_root: {models}
sdxl:
  checkpoint: checkpoints/sdxl.safetensors
pulid:
  checkpoint: pulid_v1.1.safetensors
insightface:
  model_root: .
  model_name: antelopev2
outputs_dir: {tmp_path / 'outputs'}
identity_cache_dir: {tmp_path / 'cache'}
""",
        encoding="utf-8",
    )
    output = StringIO()

    result = run_inspection(
        config,
        Console(file=output, force_terminal=False),
        show_cache_env=True,
        fail_on_internal_cache=True,
    )

    assert result == 0
    assert "Caches de modèles effectifs" in output.getvalue()
    assert "Tous les caches sont sous models_root" in output.getvalue()


def test_main_parser_exposes_phase_11_subcommands() -> None:
    parser = build_parser()

    generate = parser.parse_args(
        [
            "generate",
            "--reference",
            "inputs/noemie.webp",
            "--prompt",
            "portrait",
            "--method",
            "dpmpp_2m_sde_karras",
            "--cfg",
            "4.5",
            "--offload",
            "model_cpu_offload",
        ]
    )
    benchmark = parser.parse_args(
        [
            "benchmark",
            "--reference",
            "inputs/noemie.webp",
            "--prompt",
            "portrait",
        ]
    )

    assert generate.command == "generate"
    assert generate.guidance_scale == 4.5
    assert generate.method == "dpmpp_2m_sde_karras"
    assert generate.offload == "model_cpu_offload"
    assert benchmark.command == "benchmark"


def test_encode_command_uses_content_cache(tmp_path: Path) -> None:
    config_path, _models = _write_cli_config(tmp_path)
    reference = tmp_path / "noemie.webp"
    reference.write_bytes(b"image")
    cache_path = tmp_path / "cache" / "identity" / "noemie.npz"
    cache_path.parent.mkdir(parents=True)
    cache_path.touch()

    class FakeEncoder:
        def cache_path_for(self, *_args, **_kwargs):
            return cache_path

        def encode_image(self, image, *, identity_id, **_kwargs):
            return CharacterIdentity(
                id=identity_id,
                source_images=[str(image)],
                face_embedding=np.ones(512, dtype=np.float32),
            )

    args = build_parser().parse_args(
        [
            "encode",
            "--config",
            str(config_path),
            "--reference",
            str(reference),
            "--character",
            "noemie",
        ]
    )
    output = StringIO()

    result = run_encode(
        args,
        Console(file=output, force_terminal=False),
        encoder_factory=lambda _config: FakeEncoder(),
    )

    assert result == 0
    assert "Cache réutilisé" in output.getvalue()
    assert "shape=(512,)" in output.getvalue()


def test_generate_command_forwards_options_and_closes_generator(tmp_path: Path) -> None:
    config_path, models = _write_cli_config(tmp_path)
    (models / "checkpoints" / "reaxl_v30.safetensors").touch()
    reference = tmp_path / "noemie.webp"
    reference.write_bytes(b"image")
    calls: dict[str, object] = {}

    class FakeGenerator:
        device = "cpu"

        def __init__(self, config, **kwargs):
            calls["config"] = config
            calls["constructor"] = kwargs

        def encode_identity(self, image, **kwargs):
            calls["encode"] = {"image": image, **kwargs}
            return object()

        def generate(self, **kwargs):
            calls["generate"] = kwargs
            return SimpleNamespace(
                png_path=tmp_path / "outputs" / "pulid.png",
                json_path=tmp_path / "outputs" / "pulid.json",
            )

        def close(self):
            calls["closed"] = True

    args = build_parser().parse_args(
        [
            "generate",
            "--config",
            str(config_path),
            "--reference",
            str(reference),
            "--prompt",
            "portrait",
            "--model",
            "reaxl_v30",
            "--method",
            "dpmpp_2m_sde_karras",
            "--cfg",
            "4.5",
            "--seed",
            "7",
            "--offload",
            "model_cpu_offload",
        ]
    )
    output = StringIO()

    result = run_generate(
        args,
        Console(file=output, force_terminal=False),
        generator_factory=FakeGenerator,
    )

    assert result == 0
    assert calls["config"].sdxl.checkpoint == (
        models / "checkpoints" / "reaxl_v30.safetensors"
    )
    assert calls["generate"]["guidance_scale"] == 4.5
    assert calls["generate"]["sampling_method"] == "dpmpp_2m_sde_karras"
    assert calls["generate"]["seed"] == 7
    assert calls["constructor"]["offload_strategy"] == "model_cpu_offload"
    assert calls["closed"] is True
    assert "Image générée" in output.getvalue()


def test_benchmark_command_delegates_to_runner(tmp_path: Path) -> None:
    config_path, _models = _write_cli_config(tmp_path)
    reference = tmp_path / "noemie.webp"
    reference.write_bytes(b"image")
    calls: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, config, **kwargs):
            calls["config"] = config
            calls["constructor"] = kwargs

        def run(self, **kwargs):
            calls["run"] = kwargs
            return SimpleNamespace(
                json_path=tmp_path / "outputs" / "benchmarks" / "benchmark.json",
                report={
                    "summary_seconds": {
                        "total": {"mean": 1.25},
                    }
                },
            )

    args = build_parser().parse_args(
        [
            "benchmark",
            "--config",
            str(config_path),
            "--reference",
            str(reference),
            "--prompt",
            "portrait",
            "--runs",
            "2",
            "--steps",
            "3",
            "--offload",
            "model_cpu_offload",
        ]
    )
    output = StringIO()

    result = run_benchmark(
        args,
        Console(file=output, force_terminal=False),
        runner_factory=FakeRunner,
    )

    assert result == 0
    assert calls["run"]["runs"] == 2
    assert calls["run"]["steps"] == 3
    assert calls["constructor"]["offload_strategy"] == "model_cpu_offload"
    assert "Benchmark enregistré" in output.getvalue()
