"""Benchmark reproductible du pipeline complet PuLID + SDXL."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import statistics
import time
from typing import Any, Callable

from pulid_app.config import AppConfig
from pulid_app.exceptions import GenerationError
from pulid_app.io.metadata import save_json_metadata
from pulid_app.paths import configure_external_model_caches, ensure_writable_directory
from pulid_app.pipeline.generator import ImageGenerator, ImageGeneratorError
from pulid_app.pipeline.memory import MemoryManager, MemoryManagerError


BENCHMARK_METRICS = (
    "load_sdxl",
    "load_pulid",
    "load_insightface",
    "identity_extraction",
    "prompt_preparation",
    "diffusion",
    "vae",
    "save",
    "total",
)


class BenchmarkError(GenerationError):
    """Le benchmark n'a pas pu produire un rapport complet."""


@dataclass(frozen=True)
class BenchmarkResult:
    """Rapport sérialisé et chemin de sortie adjacent."""

    report: dict[str, Any]
    json_path: Path


def _measure(callable_: Callable[[], Any]) -> tuple[Any, float]:
    started = time.monotonic()
    result = callable_()
    return result, time.monotonic() - started


def _reset_cuda_peak(device: str, models_root: Path) -> None:
    if device.split(":", maxsplit=1)[0] != "cuda":
        return
    configure_external_model_caches(models_root)
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
    except (AttributeError, RuntimeError):
        return


def _memory_metrics(config: AppConfig, device: str) -> dict[str, int | None]:
    metrics: dict[str, int | None] = {
        "allocated_bytes": None,
        "reserved_bytes": None,
        "limit_bytes": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }
    try:
        snapshot = MemoryManager(config.models_root, device=device).snapshot()
    except MemoryManagerError:
        return metrics
    metrics.update(
        {
            "allocated_bytes": snapshot.allocated_bytes,
            "reserved_bytes": snapshot.reserved_bytes,
            "limit_bytes": snapshot.limit_bytes,
        }
    )

    if device.split(":", maxsplit=1)[0] != "cuda":
        return metrics
    configure_external_model_caches(config.models_root)
    try:
        import torch

        if torch.cuda.is_available():
            metrics["peak_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(device)
            )
            metrics["peak_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(device)
            )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    return metrics


def _summary(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for metric in BENCHMARK_METRICS:
        values = [float(run["durations_seconds"][metric]) for run in runs]
        result[metric] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "pstdev": statistics.pstdev(values),
        }
    return result


class BenchmarkRunner:
    """Exécute des runs froids identiques et agrège leurs durées."""

    def __init__(
        self,
        config: AppConfig,
        *,
        device: str | None = None,
        dtype_name: str | None = None,
        offload_strategy: str | None = None,
        generator_factory: Callable[..., Any] = ImageGenerator,
    ) -> None:
        self.config = config
        configure_external_model_caches(config.models_root)
        self.device = device
        self.dtype_name = dtype_name
        self.offload_strategy = offload_strategy
        self.generator_factory = generator_factory

    def run(
        self,
        *,
        reference: str | Path,
        prompt: str,
        identity_id: str | None = None,
        face_index: int | None = None,
        runs: int = 3,
        negative_prompt: str | None = None,
        seed: int = 42,
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        identity_strength: float = 0.8,
        guidance_scale: float = 7.0,
        sampling_method: str | None = None,
        sigma_schedule: str | None = None,
    ) -> BenchmarkResult:
        if runs <= 0:
            raise BenchmarkError("Le nombre de runs doit être strictement positif.")

        source = Path(reference).expanduser().resolve(strict=False)
        character = (identity_id or source.stem).strip()
        benchmark_dir = self.config.outputs_dir / "benchmarks"
        run_output_dir = benchmark_dir / "runs"
        try:
            ensure_writable_directory(run_output_dir)
        except (OSError, PermissionError) as exc:
            raise BenchmarkError(f"Dossier de benchmark inutilisable : {exc}") from exc

        run_reports: list[dict[str, Any]] = []
        effective_device: str | None = None
        effective_dtype: str | None = None
        for index in range(1, runs + 1):
            generator: Any | None = None
            run_started = time.monotonic()
            try:
                generator = self.generator_factory(
                    self.config,
                    device=self.device,
                    dtype_name=self.dtype_name,
                    offload_strategy=self.offload_strategy,
                    allow_downloads=False,
                )
                effective_device = generator.device
                _reset_cuda_peak(effective_device, self.config.models_root)

                _encoder, load_insightface = _measure(
                    generator.load_identity_encoder
                )
                _adapter, load_pulid = _measure(generator.load_identity_adapter)
                sdxl, load_sdxl = _measure(generator.load_sdxl)
                effective_dtype = sdxl.active_dtype_name

                identity, identity_extraction = _measure(
                    lambda: generator.encode_identity(
                        source,
                        identity_id=character,
                        face_index=face_index,
                        force_recompute=True,
                    )
                )
                generated = generator.generate(
                    prompt=prompt,
                    identity=identity,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    width=width,
                    height=height,
                    steps=steps,
                    identity_strength=identity_strength,
                    guidance_scale=guidance_scale,
                    sampling_method=sampling_method,
                    sigma_schedule=sigma_schedule,
                    output_prefix=f"benchmark_run_{index:03d}",
                    output_dir=run_output_dir,
                    collect_timings=True,
                )
                memory = _memory_metrics(self.config, effective_device)
                stage_metadata = generated.metadata
                durations = {
                    "load_sdxl": load_sdxl,
                    "load_pulid": load_pulid,
                    "load_insightface": load_insightface,
                    "identity_extraction": identity_extraction,
                    "prompt_preparation": stage_metadata[
                        "prompt_preparation_duration_seconds"
                    ],
                    "diffusion": stage_metadata["diffusion_duration_seconds"],
                    "vae": stage_metadata["vae_duration_seconds"],
                    "save": generated.save_duration_seconds,
                    "total": 0.0,
                }
            except (ImageGeneratorError, OSError, RuntimeError, ValueError) as exc:
                raise BenchmarkError(f"Run {index}/{runs} impossible : {exc}") from exc
            finally:
                if generator is not None:
                    try:
                        generator.close()
                    except ImageGeneratorError as exc:
                        raise BenchmarkError(
                            f"Cleanup du run {index}/{runs} impossible : {exc}"
                        ) from exc

            durations["total"] = time.monotonic() - run_started
            run_reports.append(
                {
                    "index": index,
                    "seed": seed,
                    "durations_seconds": durations,
                    "memory": memory,
                    "outputs": {
                        "image": str(generated.png_path),
                        "metadata": str(generated.json_path),
                    },
                }
            )

        created_at = datetime.now(timezone.utc)
        report = {
            "schema_version": 1,
            "created_at": created_at.isoformat(),
            "parameters": {
                "reference": str(source),
                "identity_id": character,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "runs": runs,
                "seed": seed,
                "width": width,
                "height": height,
                "steps": steps,
                "identity_strength": identity_strength,
                "guidance_scale": guidance_scale,
                "sampling_method": sampling_method or "default",
                "sigma_schedule": sigma_schedule or "normal",
            },
            "environment": {
                "device": effective_device,
                "dtype": effective_dtype,
                "offload_strategy": (
                    self.offload_strategy or self.config.device.offload_strategy
                ),
                "sdxl_checkpoint": str(self.config.sdxl.checkpoint),
                "pulid_checkpoint": str(self.config.pulid.checkpoint),
            },
            "runs": run_reports,
            "summary_seconds": _summary(run_reports),
        }
        timestamp = created_at.strftime("%Y%m%dT%H%M%S_%fZ")
        json_path = save_json_metadata(
            benchmark_dir / f"benchmark_{timestamp}.json",
            report,
        )
        return BenchmarkResult(report=report, json_path=json_path)
