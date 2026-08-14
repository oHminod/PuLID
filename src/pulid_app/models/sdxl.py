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
    PromptTooLongError,
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
NORMAL_SIGMA_SCHEDULE = "normal"
SIGMA_SCHEDULE_ARGUMENTS: dict[str, dict[str, bool]] = {
    NORMAL_SIGMA_SCHEDULE: {},
    "karras": {"use_karras_sigmas": True},
    "exponential": {"use_exponential_sigmas": True},
    "beta": {"use_beta_sigmas": True},
}
SUPPORTED_SIGMA_SCHEDULES = frozenset(SIGMA_SCHEDULE_ARGUMENTS)
SIGMA_SCHEDULE_FLAGS = (
    "use_karras_sigmas",
    "use_exponential_sigmas",
    "use_beta_sigmas",
)
MAX_PROMPT_TOKENS = 255


@dataclass(frozen=True)
class SamplingMethodSpec:
    """Implémentation Diffusers et courbes de sigmas compatibles."""

    label: str
    scheduler_class_name: str
    scheduler_arguments: Mapping[str, Any]
    supported_sigma_schedules: frozenset[str]


_ALL_SIGMA_SCHEDULES = frozenset(SIGMA_SCHEDULE_ARGUMENTS)
_NORMAL_SIGMAS_ONLY = frozenset({NORMAL_SIGMA_SCHEDULE})
SAMPLING_METHOD_SPECS: dict[str, SamplingMethodSpec] = {
    "dpmpp_2m": SamplingMethodSpec(
        label="DPM++ 2M",
        scheduler_class_name="DPMSolverMultistepScheduler",
        scheduler_arguments={"algorithm_type": "dpmsolver++", "solver_order": 2},
        supported_sigma_schedules=_ALL_SIGMA_SCHEDULES,
    ),
    "dpmpp_2m_sde": SamplingMethodSpec(
        label="DPM++ 2M SDE",
        scheduler_class_name="DPMSolverMultistepScheduler",
        scheduler_arguments={"algorithm_type": "sde-dpmsolver++", "solver_order": 2},
        supported_sigma_schedules=_ALL_SIGMA_SCHEDULES,
    ),
    "dpmpp_3m_sde": SamplingMethodSpec(
        label="DPM++ 3M SDE",
        scheduler_class_name="DPMSolverMultistepScheduler",
        scheduler_arguments={"algorithm_type": "sde-dpmsolver++", "solver_order": 3},
        supported_sigma_schedules=_ALL_SIGMA_SCHEDULES,
    ),
    "euler": SamplingMethodSpec(
        label="Euler",
        scheduler_class_name="EulerDiscreteScheduler",
        scheduler_arguments={},
        supported_sigma_schedules=_ALL_SIGMA_SCHEDULES,
    ),
    "euler_ancestral": SamplingMethodSpec(
        label="Euler ancestral",
        scheduler_class_name="EulerAncestralDiscreteScheduler",
        scheduler_arguments={},
        supported_sigma_schedules=_NORMAL_SIGMAS_ONLY,
    ),
    "heun": SamplingMethodSpec(
        label="Heun",
        scheduler_class_name="HeunDiscreteScheduler",
        scheduler_arguments={},
        supported_sigma_schedules=_ALL_SIGMA_SCHEDULES,
    ),
    "lms": SamplingMethodSpec(
        label="LMS",
        scheduler_class_name="LMSDiscreteScheduler",
        scheduler_arguments={},
        supported_sigma_schedules=_ALL_SIGMA_SCHEDULES,
    ),
    "ddim": SamplingMethodSpec(
        label="DDIM",
        scheduler_class_name="DDIMScheduler",
        scheduler_arguments={},
        supported_sigma_schedules=_NORMAL_SIGMAS_ONLY,
    ),
}
SUPPORTED_SAMPLING_METHODS = frozenset(SAMPLING_METHOD_SPECS)
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


@dataclass(frozen=True)
class _PromptTokens:
    """Jetons utiles d'un texte pour l'un des deux encodeurs CLIP de SDXL."""

    tokenizer: Any
    token_ids: tuple[int, ...]
    model_max_length: int
    chunk_capacity: int

    @property
    def chunk_count(self) -> int:
        return max(
            1,
            (len(self.token_ids) + self.chunk_capacity - 1) // self.chunk_capacity,
        )


def _dtype_name(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


def _is_out_of_memory(exc: RuntimeError) -> bool:
    message = str(exc).casefold()
    return "out of memory" in message or "mps backend out of memory" in message


def _tokenize_prompt(
    tokenizer: Any,
    text: str,
    *,
    prompt_kind: str,
    encoder_index: int,
) -> _PromptTokens:
    """Tokenise sans troncature et valide la limite applicative."""

    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )
        raw_token_ids = getattr(encoded, "input_ids", None)
        if raw_token_ids is None and isinstance(encoded, Mapping):
            raw_token_ids = encoded.get("input_ids")
        token_ids = tuple(int(token_id) for token_id in raw_token_ids)
        model_max_length = int(tokenizer.model_max_length)
        special_token_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    except (AttributeError, TypeError, ValueError) as exc:
        raise SDXLConfigurationError(
            f"Impossible de tokeniser le {prompt_kind} avec l'encodeur CLIP "
            f"{encoder_index} : {exc}"
        ) from exc

    chunk_capacity = model_max_length - special_token_count
    if model_max_length <= 0 or chunk_capacity <= 0:
        raise SDXLConfigurationError(
            f"Fenêtre CLIP invalide pour l'encodeur {encoder_index} : "
            f"model_max_length={model_max_length}, "
            f"jetons spéciaux={special_token_count}."
        )
    if len(token_ids) > MAX_PROMPT_TOKENS:
        raise PromptTooLongError(
            prompt_kind=prompt_kind,
            token_count=len(token_ids),
            max_tokens=MAX_PROMPT_TOKENS,
            encoder_index=encoder_index,
        )
    return _PromptTokens(
        tokenizer=tokenizer,
        token_ids=token_ids,
        model_max_length=model_max_length,
        chunk_capacity=chunk_capacity,
    )


def _build_token_blocks(tokens: _PromptTokens, block_count: int) -> list[list[int]]:
    """Ajoute BOS/EOS et aligne tous les blocs sur la fenêtre CLIP."""

    chunks = [
        list(tokens.token_ids[start : start + tokens.chunk_capacity])
        for start in range(0, len(tokens.token_ids), tokens.chunk_capacity)
    ] or [[]]
    if len(chunks) > block_count:
        raise SDXLConfigurationError(
            f"Nombre de blocs CLIP insuffisant : {block_count} pour {len(chunks)} requis."
        )
    chunks.extend([] for _ in range(block_count - len(chunks)))

    pad_token_id = getattr(tokens.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokens.tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        raise SDXLConfigurationError(
            "Le tokenizer CLIP ne définit ni pad_token_id ni eos_token_id."
        )

    blocks: list[list[int]] = []
    for chunk in chunks:
        build_with_special_tokens = getattr(
            tokens.tokenizer,
            "build_inputs_with_special_tokens",
            None,
        )
        if callable(build_with_special_tokens):
            block = list(build_with_special_tokens(chunk))
        else:
            bos_token_id = getattr(tokens.tokenizer, "bos_token_id", None)
            eos_token_id = getattr(tokens.tokenizer, "eos_token_id", None)
            if bos_token_id is None or eos_token_id is None:
                raise SDXLConfigurationError(
                    "Le tokenizer CLIP ne permet pas de construire les marqueurs BOS/EOS."
                )
            block = [int(bos_token_id), *chunk, int(eos_token_id)]
        if len(block) > tokens.model_max_length:
            raise SDXLConfigurationError(
                "Le tokenizer CLIP a produit un bloc plus long que sa fenêtre : "
                f"{len(block)} > {tokens.model_max_length}."
            )
        block.extend([int(pad_token_id)] * (tokens.model_max_length - len(block)))
        blocks.append(block)
    return blocks


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
        self.sigma_schedule: str | None = None
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

    def _import_scheduler_class(self, class_name: str) -> Any:
        configure_external_model_caches(self.models_root)
        try:
            import diffusers
        except ImportError as exc:
            raise SDXLLoadError(
                "Les schedulers Diffusers sont indisponibles. Activez .venv puis "
                "réinstallez les dépendances d'inférence."
            ) from exc
        try:
            return getattr(diffusers, class_name)
        except AttributeError as exc:
            raise SDXLLoadError(
                f"Le scheduler Diffusers {class_name} est indisponible. "
                "Réinstallez les dépendances d'inférence."
            ) from exc

    @staticmethod
    def _normalize_sigma_schedule(sigma_schedule: str | None) -> str:
        normalized = (
            sigma_schedule.strip().casefold()
            if sigma_schedule is not None
            else NORMAL_SIGMA_SCHEDULE
        )
        if normalized not in SUPPORTED_SIGMA_SCHEDULES:
            supported = ", ".join(sorted(SUPPORTED_SIGMA_SCHEDULES))
            raise SDXLConfigurationError(
                f"Courbe de sigmas inconnue : {sigma_schedule!r}. "
                f"Valeurs acceptées : {supported}."
            )
        return normalized

    @staticmethod
    def supported_sigmas_for(method: str | None) -> frozenset[str]:
        """Retourne les courbes compatibles sans charger de modèle."""

        if method is None:
            return _NORMAL_SIGMAS_ONLY
        normalized = method.strip().casefold()
        try:
            return SAMPLING_METHOD_SPECS[normalized].supported_sigma_schedules
        except KeyError as exc:
            supported = ", ".join(sorted(SUPPORTED_SAMPLING_METHODS))
            raise SDXLConfigurationError(
                f"Méthode de sampling inconnue : {method!r}. "
                f"Valeurs acceptées : {supported}."
            ) from exc

    def _replace_scheduler(
        self,
        pipeline: Any,
        method: str,
        sigma_schedule: str,
    ) -> None:
        try:
            spec = SAMPLING_METHOD_SPECS[method]
        except KeyError as exc:
            supported = ", ".join(sorted(SUPPORTED_SAMPLING_METHODS))
            raise SDXLConfigurationError(
                f"Méthode de sampling inconnue : {method!r}. "
                f"Valeurs acceptées : {supported}."
            ) from exc
        if sigma_schedule not in spec.supported_sigma_schedules:
            supported = ", ".join(sorted(spec.supported_sigma_schedules))
            raise SDXLConfigurationError(
                f"La courbe de sigmas {sigma_schedule!r} est incompatible avec "
                f"la méthode {method!r}. Valeurs acceptées : {supported}."
            )

        scheduler_class = self._import_scheduler_class(spec.scheduler_class_name)
        arguments = dict(spec.scheduler_arguments)
        if spec.supported_sigma_schedules != _NORMAL_SIGMAS_ONLY:
            arguments.update({flag: False for flag in SIGMA_SCHEDULE_FLAGS})
            arguments.update(SIGMA_SCHEDULE_ARGUMENTS[sigma_schedule])
        source_scheduler = self._default_scheduler or pipeline.scheduler
        try:
            pipeline.scheduler = scheduler_class.from_config(
                source_scheduler.config,
                **arguments,
            )
        except Exception as exc:
            raise SDXLConfigurationError(
                "Impossible de configurer le sampling "
                f"{method!r} avec les sigmas {sigma_schedule!r} depuis le "
                f"scheduler du checkpoint : {exc}"
            ) from exc

    def set_sampling(
        self,
        method: str | None,
        sigma_schedule: str | None = None,
    ) -> "SDXLModel":
        """Sélectionne indépendamment l'algorithme et sa courbe de sigmas."""

        normalized_sigma = self._normalize_sigma_schedule(sigma_schedule)
        if method is None:
            if normalized_sigma != NORMAL_SIGMA_SCHEDULE:
                raise SDXLConfigurationError(
                    f"La courbe de sigmas {normalized_sigma!r} nécessite une "
                    "méthode de sampling explicite."
                )
            if self.is_loaded and self.sampling_method is not None:
                assert self.pipeline is not None
                if self._default_scheduler is None:
                    raise SDXLConfigurationError(
                        "Le scheduler d'origine du checkpoint n'est plus disponible."
                    )
                self.pipeline.scheduler = self._default_scheduler
            self.sampling_method = None
            self.sigma_schedule = None
            return self
        normalized = method.strip().casefold()
        compatible_sigmas = self.supported_sigmas_for(normalized)
        if normalized_sigma not in compatible_sigmas:
            supported = ", ".join(sorted(compatible_sigmas))
            raise SDXLConfigurationError(
                f"La courbe de sigmas {normalized_sigma!r} est incompatible avec "
                f"la méthode {normalized!r}. Valeurs acceptées : {supported}."
            )
        self.load()
        assert self.pipeline is not None
        self._replace_scheduler(self.pipeline, normalized, normalized_sigma)
        self.sampling_method = normalized
        self.sigma_schedule = normalized_sigma
        return self

    def set_sampling_method(self, method: str | None) -> "SDXLModel":
        """Compatibilité Python : modifie la méthode en conservant les sigmas."""

        return self.set_sampling(
            method,
            self.sigma_schedule if method is not None else None,
        )

    def set_sigma_schedule(self, sigma_schedule: str | None) -> "SDXLModel":
        """Compatibilité Python : modifie les sigmas de la méthode courante."""

        return self.set_sampling(self.sampling_method, sigma_schedule)

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
                self._replace_scheduler(
                    pipeline,
                    self.sampling_method,
                    self.sigma_schedule or NORMAL_SIGMA_SCHEDULE,
                )
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

    def _encode_prompt_token_sets(
        self,
        torch_module: Any,
        token_sets: tuple[_PromptTokens, _PromptTokens],
        *,
        block_count: int,
        clip_skip: int | None,
    ) -> tuple[Any, Any]:
        """Encode puis concatène les blocs des deux CLIP de SDXL."""

        assert self.pipeline is not None
        text_encoders = (
            getattr(self.pipeline, "text_encoder", None),
            getattr(self.pipeline, "text_encoder_2", None),
        )
        if any(encoder is None for encoder in text_encoders):
            raise SDXLConfigurationError(
                "Le pipeline SDXL ne fournit pas ses deux encodeurs texte CLIP."
            )

        device = getattr(self.pipeline, "_execution_device", self.device)
        hidden_states: list[Any] = []
        pooled_embeddings: Any | None = None
        for tokens, text_encoder in zip(token_sets, text_encoders, strict=True):
            input_ids = torch_module.tensor(
                _build_token_blocks(tokens, block_count),
                dtype=torch_module.long,
                device=device,
            )
            encoded = text_encoder(input_ids, output_hidden_states=True)
            encoder_hidden_states = getattr(encoded, "hidden_states", None)
            hidden_state_offset = (clip_skip + 2) if clip_skip is not None else 2
            if (
                not encoder_hidden_states
                or len(encoder_hidden_states) < hidden_state_offset
            ):
                raise SDXLConfigurationError(
                    "Un encodeur CLIP n'a pas retourné assez d'états cachés "
                    f"pour clip_skip={clip_skip}."
                )
            selected_hidden_state = encoder_hidden_states[-hidden_state_offset]
            if len(selected_hidden_state.shape) != 3:
                raise SDXLConfigurationError(
                    "Shape CLIP inattendue pour les états cachés : "
                    f"{tuple(selected_hidden_state.shape)}."
                )
            batch_size, sequence_length, hidden_size = selected_hidden_state.shape
            hidden_states.append(
                selected_hidden_state.reshape(
                    1,
                    int(batch_size) * int(sequence_length),
                    int(hidden_size),
                )
            )

            first_output = encoded[0]
            if getattr(first_output, "ndim", 0) == 2:
                # SDXL prend le pooling du second CLIP. Pour un texte plus court
                # que l'autre conditionnement, conserver son dernier bloc réel
                # plutôt que le dernier bloc de padding.
                pooled_index = tokens.chunk_count - 1
                pooled_embeddings = first_output[pooled_index : pooled_index + 1]

        if pooled_embeddings is None:
            raise SDXLConfigurationError(
                "Le second encodeur CLIP n'a pas retourné d'embedding groupé."
            )
        return torch_module.cat(hidden_states, dim=-1), pooled_embeddings

    def _prepare_long_prompt_kwargs(
        self,
        torch_module: Any,
        *,
        prompt: str,
        negative_prompt: str | None,
        guidance_scale: float,
        clip_skip: int | None,
    ) -> dict[str, Any] | None:
        """Prépare des embeddings segmentés seulement lorsqu'un texte dépasse un bloc."""

        assert self.pipeline is not None
        tokenizers = (
            getattr(self.pipeline, "tokenizer", None),
            getattr(self.pipeline, "tokenizer_2", None),
        )
        if any(tokenizer is None for tokenizer in tokenizers):
            raise SDXLConfigurationError(
                "Le pipeline SDXL ne fournit pas ses deux tokenizers CLIP."
            )

        positive_tokens = tuple(
            _tokenize_prompt(
                tokenizer,
                prompt,
                prompt_kind="prompt positif",
                encoder_index=index,
            )
            for index, tokenizer in enumerate(tokenizers, start=1)
        )
        do_classifier_free_guidance = guidance_scale > 1.0
        pipeline_config = getattr(self.pipeline, "config", None)
        if isinstance(pipeline_config, Mapping):
            force_zeros = bool(
                pipeline_config.get("force_zeros_for_empty_prompt", False)
            )
        else:
            force_zeros = bool(
                getattr(pipeline_config, "force_zeros_for_empty_prompt", False)
            )
        zero_negative_prompt = (
            do_classifier_free_guidance and negative_prompt is None and force_zeros
        )

        negative_tokens: tuple[_PromptTokens, _PromptTokens] | None = None
        if do_classifier_free_guidance and not zero_negative_prompt:
            negative_text = negative_prompt or ""
            negative_tokens = tuple(
                _tokenize_prompt(
                    tokenizer,
                    negative_text,
                    prompt_kind="prompt négatif",
                    encoder_index=index,
                )
                for index, tokenizer in enumerate(tokenizers, start=1)
            )

        all_tokens = (*positive_tokens, *(negative_tokens or ()))
        block_count = max(tokens.chunk_count for tokens in all_tokens)
        if block_count == 1:
            return None

        prompt_embeds, pooled_prompt_embeds = self._encode_prompt_token_sets(
            torch_module,
            positive_tokens,
            block_count=block_count,
            clip_skip=clip_skip,
        )
        prepared = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }
        if do_classifier_free_guidance:
            if zero_negative_prompt:
                negative_prompt_embeds = torch_module.zeros_like(prompt_embeds)
                negative_pooled_prompt_embeds = torch_module.zeros_like(
                    pooled_prompt_embeds
                )
            else:
                assert negative_tokens is not None
                (
                    negative_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = self._encode_prompt_token_sets(
                    torch_module,
                    negative_tokens,
                    block_count=block_count,
                    clip_skip=clip_skip,
                )
            prepared.update(
                {
                    "negative_prompt_embeds": negative_prompt_embeds,
                    "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
                }
            )
        return prepared

    def _run_generation(
        self,
        torch_module: Any,
        *,
        prompt: str,
        negative_prompt: str | None,
        clip_skip: int | None,
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
        if clip_skip is not None:
            generation_kwargs["clip_skip"] = clip_skip
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
                prompt_started = time.monotonic()
                long_prompt_kwargs = self._prepare_long_prompt_kwargs(
                    torch_module,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale,
                    clip_skip=clip_skip,
                )
                if long_prompt_kwargs is not None:
                    generation_kwargs.pop("prompt")
                    generation_kwargs.pop("negative_prompt")
                    generation_kwargs.pop("clip_skip", None)
                    generation_kwargs.update(long_prompt_kwargs)
                    stage_durations["prompt_preparation"] += (
                        time.monotonic() - prompt_started
                    )
                output = self.pipeline(**generation_kwargs)
        if not output.images:
            raise SDXLGenerationError("SDXL n'a retourné aucune image.")
        return output.images[0], stage_durations

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str | None = None,
        clip_skip: int | None = None,
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
        if clip_skip is not None and (
            isinstance(clip_skip, bool) or not isinstance(clip_skip, int) or clip_skip < 1
        ):
            raise SDXLConfigurationError(
                "clip_skip doit être un entier strictement positif ou None."
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
                clip_skip=clip_skip,
                seed=seed,
                steps=steps,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                cross_attention_kwargs=cross_attention_kwargs,
                collect_timings=collect_timings,
            )
        except PromptTooLongError:
            raise
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
                    clip_skip=clip_skip,
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
