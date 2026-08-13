"""Chargement hors ligne et génération avec un checkpoint SDXL monofichier."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Iterator, TYPE_CHECKING

from pulid_app.config import AppConfig
from pulid_app.device import get_best_device
from pulid_app.exceptions import (
    GenerationError,
    ModelLoadError,
    ModelNotFoundError,
    PuLIDAppError,
)
from pulid_app.paths import configure_external_model_caches
from pulid_app.pipeline.memory import MemoryManager, MemoryManagerError

if TYPE_CHECKING:
    from PIL.Image import Image


REQUIRED_SDXL_CONFIG_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder_2/config.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/vocab.json",
    "tokenizer_2/merges.txt",
    "unet/config.json",
    "vae/config.json",
)
SUPPORTED_SAMPLING_METHODS = frozenset({"dpmpp_2m_sde_karras"})
SUPPORTED_OFFLOAD_STRATEGIES = frozenset({"none", "model_cpu_offload"})


class SDXLError(PuLIDAppError):
    """Erreur métier du pipeline SDXL local."""


class SDXLConfigurationError(SDXLError):
    """Un chemin ou un paramètre SDXL est invalide."""


class SDXLLoadError(SDXLError, ModelLoadError):
    """Le checkpoint SDXL ne peut pas être chargé."""


class SDXLGenerationError(SDXLError, GenerationError):
    """La génération SDXL a échoué."""


class SDXLModelNotFoundError(SDXLConfigurationError, ModelNotFoundError):
    """Un fichier local requis par SDXL est absent."""


@dataclass(frozen=True)
class SDXLGenerationResult:
    """Image SDXL et détails effectifs de son exécution."""

    image: "Image"
    seed: int
    device: str
    dtype: str
    dtype_fallback_used: bool
    duration_seconds: float
    stage_durations_seconds: Mapping[str, float]


def _dtype_name(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


def _is_out_of_memory(exc: RuntimeError) -> bool:
    message = str(exc).casefold()
    return "out of memory" in message or "mps backend out of memory" in message


@contextmanager
def _measure_pipeline_stages(
    pipeline: Any,
    *,
    enabled: bool,
) -> Iterator[dict[str, float]]:
    """Instrumente les méthodes Diffusers sans modifier leur implémentation."""

    timings = {
        "prompt_preparation": 0.0,
        "diffusion": 0.0,
        "vae": 0.0,
    }
    if not enabled:
        yield timings
        return

    restorations: list[tuple[Any, str, bool, Any]] = []
    diffusion_started: float | None = None

    def wrap_accumulated(target: Any, attribute: str, key: str) -> None:
        original = getattr(target, attribute)
        target_attributes = vars(target)
        had_instance_attribute = attribute in target_attributes
        previous_instance_value = target_attributes.get(attribute)

        def measured(*args: Any, **kwargs: Any) -> Any:
            started = time.monotonic()
            try:
                return original(*args, **kwargs)
            finally:
                timings[key] += time.monotonic() - started

        restorations.append(
            (target, attribute, had_instance_attribute, previous_instance_value)
        )
        setattr(target, attribute, measured)

    def wrap_diffusion(target: Any) -> None:
        original = getattr(target, "forward")
        target_attributes = vars(target)
        had_instance_attribute = "forward" in target_attributes
        previous_instance_value = target_attributes.get("forward")

        def measured(*args: Any, **kwargs: Any) -> Any:
            nonlocal diffusion_started
            started = time.monotonic()
            if diffusion_started is None:
                diffusion_started = started
            try:
                return original(*args, **kwargs)
            finally:
                timings["diffusion"] = time.monotonic() - diffusion_started

        restorations.append(
            (target, "forward", had_instance_attribute, previous_instance_value)
        )
        setattr(target, "forward", measured)

    try:
        wrap_accumulated(pipeline, "encode_prompt", "prompt_preparation")
        wrap_diffusion(pipeline.unet)
        wrap_accumulated(pipeline.vae, "decode", "vae")
        yield timings
    finally:
        for target, attribute, had_instance_attribute, previous in reversed(
            restorations
        ):
            if had_instance_attribute:
                setattr(target, attribute, previous)
            else:
                delattr(target, attribute)


class SDXLModel:
    """Pipeline SDXL chargé depuis un `.safetensors` et des configs locales."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_dir: str | Path,
        *,
        models_root: str | Path,
        device: str | None = None,
        dtype_name: str | None = None,
        allow_dtype_fallback: bool = True,
        offload_strategy: str = "none",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=False)
        self.config_dir = Path(config_dir).expanduser().resolve(strict=False)
        self.models_root = Path(models_root).expanduser().resolve(strict=False)
        self.device = (device or get_best_device()).casefold()
        self.requested_dtype_name = dtype_name.casefold() if dtype_name else None
        self.allow_dtype_fallback = allow_dtype_fallback
        self.offload_strategy = offload_strategy.strip().casefold()
        self._validate_offload_strategy()
        self.memory_manager = MemoryManager(self.models_root, device=self.device)
        self.pipeline: Any | None = None
        self.active_dtype: Any | None = None
        self.dtype_fallback_used = False
        self.sampling_method: str | None = None
        self._default_scheduler: Any | None = None

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        *,
        device: str | None = None,
        dtype_name: str | None = None,
        allow_dtype_fallback: bool = True,
        offload_strategy: str | None = None,
    ) -> "SDXLModel":
        if config.sdxl.config_dir is None:
            raise SDXLConfigurationError(
                "La clé sdxl.config_dir est absente. Exécutez d'abord "
                "`python scripts/prepare_sdxl_config.py`."
            )
        return cls(
            config.sdxl.checkpoint,
            config.sdxl.config_dir,
            models_root=config.models_root,
            device=device,
            dtype_name=dtype_name or config.device.dtype,
            allow_dtype_fallback=allow_dtype_fallback,
            offload_strategy=offload_strategy or config.device.offload_strategy,
        )

    def _validate_offload_strategy(self) -> None:
        if self.offload_strategy not in SUPPORTED_OFFLOAD_STRATEGIES:
            supported = ", ".join(sorted(SUPPORTED_OFFLOAD_STRATEGIES))
            raise SDXLConfigurationError(
                f"Stratégie d'offload inconnue : {self.offload_strategy!r}. "
                f"Valeurs acceptées : {supported}."
            )
        device_type = self.device.split(":", maxsplit=1)[0]
        if self.offload_strategy != "none" and device_type != "cuda":
            raise SDXLConfigurationError(
                "model_cpu_offload est réservé à CUDA ; utilisez offload_strategy=none "
                f"avec le device {self.device!r}."
            )

    def _place_pipeline(self, pipeline: Any) -> Any:
        """Place SDXL selon la stratégie choisie, sans API CUDA hors CUDA."""

        device_type = self.device.split(":", maxsplit=1)[0]
        if self.offload_strategy == "model_cpu_offload":
            if device_type != "cuda":  # garde redondante, volontairement locale
                raise SDXLConfigurationError(
                    "L'offload SDXL ne peut être activé que sur CUDA."
                )
            offload = getattr(pipeline, "enable_model_cpu_offload", None)
            if not callable(offload):
                raise SDXLLoadError(
                    "Cette version de Diffusers ne fournit pas "
                    "enable_model_cpu_offload(). Réinstallez les dépendances d'inférence."
                )
            offload(device=self.device)
            return pipeline
        return self.memory_manager.move_to_device(pipeline, self.device)

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None

    @property
    def active_dtype_name(self) -> str | None:
        return _dtype_name(self.active_dtype) if self.active_dtype is not None else None

    def _validate_local_files(self) -> None:
        if not self.checkpoint_path.is_file():
            raise SDXLModelNotFoundError(
                f"Checkpoint SDXL introuvable : {self.checkpoint_path}. "
                "Corrigez sdxl.checkpoint ou sélectionnez un modèle local existant."
            )
        if self.checkpoint_path.suffix.casefold() != ".safetensors":
            raise SDXLConfigurationError(
                f"Le checkpoint SDXL doit être un .safetensors : {self.checkpoint_path}"
            )
        if not self.config_dir.is_dir():
            raise SDXLModelNotFoundError(
                f"Dossier de configuration SDXL introuvable : {self.config_dir}. "
                "Exécutez `python scripts/prepare_sdxl_config.py`."
            )
        missing = [
            self.config_dir / relative
            for relative in REQUIRED_SDXL_CONFIG_FILES
            if not (self.config_dir / relative).is_file()
        ]
        if missing:
            raise SDXLModelNotFoundError(
                "Configuration SDXL locale incomplète ; fichiers manquants : "
                + ", ".join(str(path) for path in missing)
                + ". Exécutez `python scripts/prepare_sdxl_config.py`."
            )

    def _import_ml(self) -> tuple[Any, Any]:
        configure_external_model_caches(self.models_root)
        try:
            import torch
            from diffusers import StableDiffusionXLPipeline
        except ImportError as exc:
            raise SDXLLoadError(
                "Les dépendances SDXL sont absentes. Activez .venv puis exécutez "
                "`uv pip install -e '.[inference,dev]'`."
            ) from exc
        return torch, StableDiffusionXLPipeline

    def _import_dpm_scheduler(self) -> Any:
        configure_external_model_caches(self.models_root)
        try:
            from diffusers import DPMSolverMultistepScheduler
        except ImportError as exc:
            raise SDXLLoadError(
                "Le scheduler DPM++ est indisponible. Activez .venv puis "
                "réinstallez les dépendances d'inférence."
            ) from exc
        return DPMSolverMultistepScheduler

    def _replace_scheduler(self, pipeline: Any, method: str) -> None:
        if method != "dpmpp_2m_sde_karras":
            supported = ", ".join(sorted(SUPPORTED_SAMPLING_METHODS))
            raise SDXLConfigurationError(
                f"Méthode de sampling inconnue : {method!r}. "
                f"Valeurs acceptées : {supported}."
            )
        scheduler_class = self._import_dpm_scheduler()
        try:
            pipeline.scheduler = scheduler_class.from_config(
                pipeline.scheduler.config,
                algorithm_type="sde-dpmsolver++",
                solver_order=2,
                use_karras_sigmas=True,
            )
        except Exception as exc:
            raise SDXLConfigurationError(
                "Impossible de configurer le sampling "
                f"{method!r} depuis le scheduler du checkpoint : {exc}"
            ) from exc

    def set_sampling_method(self, method: str | None) -> "SDXLModel":
        """Remplace le scheduler chargé, ou conserve celui du modèle si absent."""

        if method is None:
            if self.is_loaded and self.sampling_method is not None:
                assert self.pipeline is not None
                if self._default_scheduler is None:
                    raise SDXLConfigurationError(
                        "Le scheduler d'origine du checkpoint n'est plus disponible."
                    )
                self.pipeline.scheduler = self._default_scheduler
            self.sampling_method = None
            return self
        normalized = method.strip().casefold()
        if normalized not in SUPPORTED_SAMPLING_METHODS:
            supported = ", ".join(sorted(SUPPORTED_SAMPLING_METHODS))
            raise SDXLConfigurationError(
                f"Méthode de sampling inconnue : {method!r}. "
                f"Valeurs acceptées : {supported}."
            )
        self.load()
        assert self.pipeline is not None
        self._replace_scheduler(self.pipeline, normalized)
        self.sampling_method = normalized
        return self

    def _resolve_dtype(self, torch_module: Any) -> Any:
        if self.device == "cpu":
            return torch_module.float32
        configured = self.requested_dtype_name or "float16"
        aliases = {
            "float16": torch_module.float16,
            "fp16": torch_module.float16,
            "float32": torch_module.float32,
            "fp32": torch_module.float32,
        }
        try:
            return aliases[configured]
        except KeyError as exc:
            raise SDXLConfigurationError(
                f"Dtype SDXL non pris en charge : {configured!r}. "
                "Valeurs acceptées : float16, float32."
            ) from exc

    def _load_pipeline(self, torch_module: Any, pipeline_class: Any, dtype: Any) -> Any:
        pipeline: Any | None = None
        try:
            pipeline = pipeline_class.from_single_file(
                str(self.checkpoint_path),
                config=str(self.config_dir),
                cache_dir=str(self.models_root / "huggingface" / "hub"),
                local_files_only=True,
                torch_dtype=dtype,
                add_watermarker=False,
            )
            pipeline = self._place_pipeline(pipeline)
            self._default_scheduler = pipeline.scheduler
            if self.sampling_method is not None:
                # Un éventuel rechargement contrôlé doit conserver le scheduler
                # demandé au lieu de revenir silencieusement à celui du modèle.
                self._replace_scheduler(pipeline, self.sampling_method)
            if self.device == "mps":
                # Réduit le pic mémoire du calcul d'attention au prix d'un peu de débit.
                pipeline.enable_attention_slicing("auto")
            return pipeline
        except Exception as exc:
            cleanup_error: MemoryManagerError | None = None
            try:
                self.memory_manager.unload(pipeline, cleanup=False)
                # Le chargement des poids peut avoir alloué avant que le module
                # ne soit visible par le gestionnaire.
                self.memory_manager.cleanup(force=True)
            except MemoryManagerError as memory_exc:
                cleanup_error = memory_exc
            cleanup_details = (
                f" Nettoyage mémoire également impossible : {cleanup_error}."
                if cleanup_error is not None
                else ""
            )
            raise SDXLLoadError(
                f"Impossible de charger {self.checkpoint_path} sur {self.device} "
                f"en {_dtype_name(dtype)} : {exc}.{cleanup_details}"
            ) from exc

    def load(self) -> "SDXLModel":
        """Charge le checkpoint sans accès réseau et le déplace sur le device choisi."""

        if self.is_loaded:
            return self
        self._validate_local_files()
        torch_module, pipeline_class = self._import_ml()
        self.memory_manager.bind_torch(torch_module)
        dtype = self._resolve_dtype(torch_module)
        try:
            self.pipeline = self._load_pipeline(torch_module, pipeline_class, dtype)
            self.active_dtype = dtype
        except SDXLLoadError as exc:
            cause = exc.__cause__
            can_fallback = (
                self.allow_dtype_fallback
                and self.device == "mps"
                and dtype == torch_module.float16
                and isinstance(cause, RuntimeError)
                and not _is_out_of_memory(cause)
            )
            if not can_fallback:
                raise
            self._cleanup_pipeline()
            try:
                self.pipeline = self._load_pipeline(
                    torch_module, pipeline_class, torch_module.float32
                )
            except SDXLLoadError as fallback_exc:
                raise SDXLLoadError(
                    f"Le chargement MPS a échoué en float16 puis en float32 : {fallback_exc}"
                ) from fallback_exc
            self.active_dtype = torch_module.float32
            self.dtype_fallback_used = True
        return self

    def _cleanup_pipeline(self) -> None:
        pipeline = self.pipeline
        self.pipeline = None
        self.active_dtype = None
        self._default_scheduler = None
        try:
            self.memory_manager.unload(pipeline)
        except MemoryManagerError as exc:
            raise SDXLLoadError(f"Impossible de libérer le pipeline SDXL : {exc}") from exc

    def close(self) -> None:
        """Décharge le pipeline ; les appels répétés sont sans effet."""

        self._cleanup_pipeline()

    def __enter__(self) -> "SDXLModel":
        return self.load()

    def __exit__(self, *_args: Any) -> None:
        self.close()

    @staticmethod
    def _validate_generation_parameters(
        prompt: str,
        seed: int,
        steps: int,
        width: int,
        height: int,
        guidance_scale: float,
    ) -> None:
        if not prompt.strip():
            raise SDXLConfigurationError("Le prompt ne peut pas être vide.")
        if seed < 0:
            raise SDXLConfigurationError("La seed doit être positive ou nulle.")
        if steps <= 0:
            raise SDXLConfigurationError("Le nombre de steps doit être strictement positif.")
        if width <= 0 or height <= 0 or width % 8 != 0 or height % 8 != 0:
            raise SDXLConfigurationError(
                "La largeur et la hauteur doivent être positives et divisibles par 8."
            )
        if guidance_scale < 0:
            raise SDXLConfigurationError("guidance_scale doit être positif ou nul.")

    def _run_generation(
        self,
        torch_module: Any,
        *,
        prompt: str,
        negative_prompt: str | None,
        seed: int,
        steps: int,
        width: int,
        height: int,
        guidance_scale: float,
        cross_attention_kwargs: Mapping[str, Any] | None,
        collect_timings: bool,
    ) -> tuple["Image", dict[str, float]]:
        assert self.pipeline is not None
        generator = torch_module.Generator(device="cpu").manual_seed(seed)
        generation_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "generator": generator,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "num_images_per_prompt": 1,
        }
        if cross_attention_kwargs:
            # Canal générique : SDXL ne connaît ni PuLID ni le producteur de ces
            # paramètres, il les transmet seulement aux processeurs installés.
            generation_kwargs["cross_attention_kwargs"] = dict(
                cross_attention_kwargs
            )
        with _measure_pipeline_stages(
            self.pipeline,
            enabled=collect_timings,
        ) as stage_durations:
            with torch_module.inference_mode():
                output = self.pipeline(**generation_kwargs)
        if not output.images:
            raise SDXLGenerationError("SDXL n'a retourné aucune image.")
        return output.images[0], stage_durations

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str | None = None,
        seed: int = 42,
        steps: int = 20,
        width: int = 1024,
        height: int = 1024,
        guidance_scale: float = 7.0,
        cross_attention_kwargs: Mapping[str, Any] | None = None,
        collect_timings: bool = False,
    ) -> SDXLGenerationResult:
        """Génère une image avec un éventuel second essai FP32 contrôlé sur MPS."""

        self._validate_generation_parameters(
            prompt, seed, steps, width, height, guidance_scale
        )
        self.load()
        torch_module, pipeline_class = self._import_ml()
        self.memory_manager.bind_torch(torch_module)
        started = time.monotonic()
        try:
            image, stage_durations = self._run_generation(
                torch_module,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                steps=steps,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                cross_attention_kwargs=cross_attention_kwargs,
                collect_timings=collect_timings,
            )
        except RuntimeError as exc:
            can_fallback = (
                self.allow_dtype_fallback
                and self.device == "mps"
                and self.active_dtype == torch_module.float16
                and not self.dtype_fallback_used
                and not _is_out_of_memory(exc)
                and not cross_attention_kwargs
            )
            if not can_fallback:
                raise SDXLGenerationError(f"Génération SDXL impossible : {exc}") from exc

            self._cleanup_pipeline()
            try:
                self.pipeline = self._load_pipeline(
                    torch_module, pipeline_class, torch_module.float32
                )
                self.active_dtype = torch_module.float32
                self.dtype_fallback_used = True
                image, stage_durations = self._run_generation(
                    torch_module,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    steps=steps,
                    width=width,
                    height=height,
                    guidance_scale=guidance_scale,
                    cross_attention_kwargs=None,
                    collect_timings=collect_timings,
                )
            except (SDXLLoadError, RuntimeError) as fallback_exc:
                raise SDXLGenerationError(
                    "La génération MPS a échoué en float16 puis en float32 : "
                    f"{fallback_exc}"
                ) from fallback_exc

        return SDXLGenerationResult(
            image=image,
            seed=seed,
            device=self.device,
            dtype=self.active_dtype_name or "inconnu",
            dtype_fallback_used=self.dtype_fallback_used,
            duration_seconds=time.monotonic() - started,
            stage_durations_seconds=stage_durations,
        )
