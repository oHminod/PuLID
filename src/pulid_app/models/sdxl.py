"""Chargement hors ligne et génération avec un checkpoint SDXL monofichier."""

from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path
import time
from typing import Any, TYPE_CHECKING

from pulid_app.config import AppConfig
from pulid_app.device import get_best_device
from pulid_app.paths import configure_external_model_caches

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


class SDXLError(RuntimeError):
    """Erreur métier du pipeline SDXL local."""


class SDXLConfigurationError(SDXLError):
    """Un chemin ou un paramètre SDXL est invalide."""


class SDXLLoadError(SDXLError):
    """Le checkpoint SDXL ne peut pas être chargé."""


class SDXLGenerationError(SDXLError):
    """La génération SDXL a échoué."""


@dataclass(frozen=True)
class SDXLGenerationResult:
    """Image SDXL et détails effectifs de son exécution."""

    image: "Image"
    seed: int
    device: str
    dtype: str
    dtype_fallback_used: bool
    duration_seconds: float


def _dtype_name(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


def _is_out_of_memory(exc: RuntimeError) -> bool:
    message = str(exc).casefold()
    return "out of memory" in message or "mps backend out of memory" in message


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
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=False)
        self.config_dir = Path(config_dir).expanduser().resolve(strict=False)
        self.models_root = Path(models_root).expanduser().resolve(strict=False)
        self.device = (device or get_best_device()).casefold()
        self.requested_dtype_name = dtype_name.casefold() if dtype_name else None
        self.allow_dtype_fallback = allow_dtype_fallback
        self.pipeline: Any | None = None
        self.active_dtype: Any | None = None
        self.dtype_fallback_used = False

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        *,
        device: str | None = None,
        dtype_name: str | None = None,
        allow_dtype_fallback: bool = True,
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
        )

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None

    @property
    def active_dtype_name(self) -> str | None:
        return _dtype_name(self.active_dtype) if self.active_dtype is not None else None

    def _validate_local_files(self) -> None:
        if not self.checkpoint_path.is_file():
            raise SDXLConfigurationError(
                f"Checkpoint SDXL introuvable : {self.checkpoint_path}"
            )
        if self.checkpoint_path.suffix.casefold() != ".safetensors":
            raise SDXLConfigurationError(
                f"Le checkpoint SDXL doit être un .safetensors : {self.checkpoint_path}"
            )
        if not self.config_dir.is_dir():
            raise SDXLConfigurationError(
                f"Dossier de configuration SDXL introuvable : {self.config_dir}. "
                "Exécutez `python scripts/prepare_sdxl_config.py`."
            )
        missing = [
            self.config_dir / relative
            for relative in REQUIRED_SDXL_CONFIG_FILES
            if not (self.config_dir / relative).is_file()
        ]
        if missing:
            raise SDXLConfigurationError(
                "Configuration SDXL locale incomplète ; fichiers manquants : "
                + ", ".join(str(path) for path in missing)
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
        try:
            pipeline = pipeline_class.from_single_file(
                str(self.checkpoint_path),
                config=str(self.config_dir),
                cache_dir=str(self.models_root / "huggingface" / "hub"),
                local_files_only=True,
                torch_dtype=dtype,
                add_watermarker=False,
            )
            pipeline.to(self.device)
            if self.device == "mps":
                # Réduit le pic mémoire du calcul d'attention au prix d'un peu de débit.
                pipeline.enable_attention_slicing("auto")
            return pipeline
        except Exception as exc:
            raise SDXLLoadError(
                f"Impossible de charger {self.checkpoint_path} sur {self.device} "
                f"en {_dtype_name(dtype)} : {exc}"
            ) from exc

    def load(self) -> "SDXLModel":
        """Charge le checkpoint sans accès réseau et le déplace sur le device choisi."""

        if self.is_loaded:
            return self
        self._validate_local_files()
        torch_module, pipeline_class = self._import_ml()
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
            self._cleanup_pipeline(torch_module)
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

    def _cleanup_pipeline(self, torch_module: Any) -> None:
        pipeline = self.pipeline
        self.pipeline = None
        if pipeline is not None:
            del pipeline
        gc.collect()
        if self.device == "mps" and hasattr(torch_module, "mps"):
            torch_module.mps.empty_cache()
        elif self.device == "cuda" and hasattr(torch_module, "cuda"):
            torch_module.cuda.empty_cache()

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
    ) -> "Image":
        assert self.pipeline is not None
        generator = torch_module.Generator(device="cpu").manual_seed(seed)
        with torch_module.inference_mode():
            output = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                generator=generator,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                num_images_per_prompt=1,
            )
        if not output.images:
            raise SDXLGenerationError("SDXL n'a retourné aucune image.")
        return output.images[0]

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
    ) -> SDXLGenerationResult:
        """Génère une image avec un éventuel second essai FP32 contrôlé sur MPS."""

        self._validate_generation_parameters(
            prompt, seed, steps, width, height, guidance_scale
        )
        self.load()
        torch_module, pipeline_class = self._import_ml()
        started = time.monotonic()
        try:
            image = self._run_generation(
                torch_module,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                steps=steps,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
            )
        except RuntimeError as exc:
            can_fallback = (
                self.allow_dtype_fallback
                and self.device == "mps"
                and self.active_dtype == torch_module.float16
                and not self.dtype_fallback_used
                and not _is_out_of_memory(exc)
            )
            if not can_fallback:
                raise SDXLGenerationError(f"Génération SDXL impossible : {exc}") from exc

            self._cleanup_pipeline(torch_module)
            try:
                self.pipeline = self._load_pipeline(
                    torch_module, pipeline_class, torch_module.float32
                )
                self.active_dtype = torch_module.float32
                self.dtype_fallback_used = True
                image = self._run_generation(
                    torch_module,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    steps=steps,
                    width=width,
                    height=height,
                    guidance_scale=guidance_scale,
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
        )
