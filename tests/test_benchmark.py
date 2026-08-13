from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pulid_app.config import (
    AppConfig,
    DeviceConfig,
    InsightFaceConfig,
    PuLIDConfig,
    SDXLConfig,
)
from pulid_app.pipeline.benchmark import (
    BENCHMARK_METRICS,
    BenchmarkError,
    BenchmarkRunner,
)


def _config(tmp_path: Path) -> AppConfig:
    models = tmp_path / "models"
    return AppConfig(
        models_root=models,
        sdxl=SDXLConfig(models / "sdxl.safetensors", models / "config"),
        pulid=PuLIDConfig(models / "pulid.safetensors"),
        insightface=InsightFaceConfig(models, "antelopev2"),
        outputs_dir=tmp_path / "outputs",
        identity_cache_dir=tmp_path / "cache" / "identity",
        device=DeviceConfig(preferred="cpu", dtype="float32"),
        source_path=tmp_path / "config.yaml",
    )


class FakeBenchmarkGenerator:
    instances: list["FakeBenchmarkGenerator"] = []

    def __init__(self, _config, **kwargs) -> None:
        self.device = "cpu"
        self.constructor_kwargs = kwargs
        self.closed = False
        self.__class__.instances.append(self)

    def load_identity_encoder(self):
        return object()

    def load_identity_adapter(self):
        return object()

    def load_sdxl(self):
        return SimpleNamespace(active_dtype_name="float32")

    def encode_identity(self, *_args, **_kwargs):
        return object()

    def generate(self, **kwargs):
        output_dir = Path(kwargs["output_dir"])
        return SimpleNamespace(
            png_path=output_dir / f"{kwargs['output_prefix']}.png",
            json_path=output_dir / f"{kwargs['output_prefix']}.json",
            save_duration_seconds=0.01,
            metadata={
                "prompt_preparation_duration_seconds": 0.02,
                "diffusion_duration_seconds": 0.03,
                "vae_duration_seconds": 0.04,
            },
        )

    def close(self):
        self.closed = True


def test_benchmark_writes_runs_and_summary(tmp_path: Path) -> None:
    FakeBenchmarkGenerator.instances.clear()
    runner = BenchmarkRunner(
        _config(tmp_path),
        device="cpu",
        offload_strategy="none",
        generator_factory=FakeBenchmarkGenerator,
    )

    result = runner.run(
        reference=tmp_path / "noemie.webp",
        prompt="portrait",
        runs=2,
        steps=3,
        width=64,
        height=64,
    )

    assert result.json_path.parent == tmp_path / "outputs" / "benchmarks"
    assert result.json_path.is_file()
    assert len(result.report["runs"]) == 2
    assert set(result.report["summary_seconds"]) == set(BENCHMARK_METRICS)
    assert result.report["summary_seconds"]["save"]["mean"] == 0.01
    assert all(instance.closed for instance in FakeBenchmarkGenerator.instances)
    assert all(
        instance.constructor_kwargs["offload_strategy"] == "none"
        for instance in FakeBenchmarkGenerator.instances
    )
    persisted = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert persisted["parameters"]["runs"] == 2
    assert persisted["environment"]["offload_strategy"] == "none"
    assert persisted["runs"][0]["seed"] == persisted["runs"][1]["seed"] == 42


def test_benchmark_rejects_non_positive_run_count(tmp_path: Path) -> None:
    runner = BenchmarkRunner(
        _config(tmp_path),
        device="cpu",
        generator_factory=FakeBenchmarkGenerator,
    )

    with pytest.raises(BenchmarkError, match="strictement positif"):
        runner.run(reference=tmp_path / "noemie.webp", prompt="portrait", runs=0)
