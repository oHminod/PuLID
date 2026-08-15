"""Petit serveur HTTP pour l'inventaire SDXL et la génération en mémoire."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import secrets
import unicodedata
import re
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError

from pulid_app import __version__
from pulid_app.config import AppConfig, load_config
from pulid_app.exceptions import (
    FaceNotDetectedError,
    ModelNotFoundError,
    MultipleFacesDetectedError,
    PuLIDAppError,
    UnsupportedDeviceError,
    actionable_error,
)
from pulid_app.models.identity_encoder import SUPPORTED_IMAGE_FORMATS
from pulid_app.models.sdxl import (
    NORMAL_SIGMA_SCHEDULE,
    SAMPLING_METHOD_SPECS,
    SUPPORTED_SAMPLING_METHODS,
    SUPPORTED_SIGMA_SCHEDULES,
)
from pulid_app.paths import (
    configure_external_model_caches,
    require_models_root,
    resolve_sdxl_checkpoint,
)
from pulid_app.pipeline.generator import DEFAULT_NEGATIVE_PROMPT, ImageGenerator


DEFAULT_METHOD = "default"
DEFAULT_SIGMAS = NORMAL_SIGMA_SCHEDULE
SIGMA_SCHEDULE_ORDER = tuple(
    name
    for name in ("normal", "karras", "exponential", "beta")
    if name in SUPPORTED_SIGMA_SCHEDULES
)
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_IDENTITY_STRENGTH = 0.8
MAX_REFERENCE_BYTES = 20 * 1024 * 1024
MAX_SEED = 2**63 - 1


@dataclass(frozen=True)
class GeneratedPayload:
    content: bytes
    filename: str
    seed: int
    model: str
    method: str
    sigmas: str


def list_sdxl_models(config: AppConfig) -> list[dict[str, Any]]:
    """Liste les checkpoints du dossier SDXL configuré, sans les charger."""

    directory = config.sdxl.checkpoint.parent
    if not directory.is_dir():
        raise ModelNotFoundError(
            f"Dossier de checkpoints SDXL introuvable : {directory}. "
            "Vérifiez le montage du SSD et la clé sdxl.checkpoint."
        )
    configured = config.sdxl.checkpoint.resolve(strict=False)
    return [
        {
            "name": path.stem,
            "filename": path.name,
            "default": path.resolve(strict=False) == configured,
        }
        for path in sorted(directory.glob("*.safetensors"), key=lambda item: item.name.casefold())
        if path.is_file()
    ]


def list_sampling_methods() -> list[dict[str, Any]]:
    """Expose le scheduler du checkpoint et les remplacements disponibles."""

    methods = [
        {
            "name": DEFAULT_METHOD,
            "label": "Scheduler du checkpoint",
            "default": True,
            "supported_sigma_schedules": [DEFAULT_SIGMAS],
        }
    ]
    methods.extend(
        {
            "name": name,
            "label": spec.label,
            "default": False,
            "supported_sigma_schedules": [
                schedule
                for schedule in SIGMA_SCHEDULE_ORDER
                if schedule in spec.supported_sigma_schedules
            ],
        }
        for name, spec in SAMPLING_METHOD_SPECS.items()
    )
    return methods


def list_sigma_schedules() -> list[dict[str, Any]]:
    """Expose les courbes de sigmas indépendamment des méthodes de sampling."""

    labels = {
        NORMAL_SIGMA_SCHEDULE: "Normal / natif",
        "karras": "Karras",
        "exponential": "Exponentiel",
        "beta": "Beta",
    }
    method_names = [DEFAULT_METHOD, *SAMPLING_METHOD_SPECS]
    schedules: list[dict[str, Any]] = []
    for name in SIGMA_SCHEDULE_ORDER:
        compatible_methods = [
            method
            for method in method_names
            if name in _supported_sigmas_for_http_method(method)
        ]
        schedules.append(
            {
                "name": name,
                "label": labels[name],
                "default": name == DEFAULT_SIGMAS,
                "supported_sampling_methods": compatible_methods,
            }
        )
    return schedules


def _supported_sigmas_for_http_method(method: str) -> frozenset[str]:
    if method == DEFAULT_METHOD:
        return frozenset({DEFAULT_SIGMAS})
    return SAMPLING_METHOD_SPECS[method].supported_sigma_schedules


def resolve_generation_seed(
    requested: int,
    random_seed: Callable[[], int] | None = None,
) -> int:
    """Transforme 0/-1 en seed aléatoire et valide les seeds explicites."""

    if requested in {0, -1}:
        factory = random_seed or (lambda: secrets.randbelow(MAX_SEED) + 1)
        selected = int(factory())
        if selected <= 0 or selected > MAX_SEED:
            raise ValueError(f"La seed aléatoire doit être comprise entre 1 et {MAX_SEED}.")
        return selected
    if requested < -1 or requested > MAX_SEED:
        raise ValueError(
            f"La seed doit valoir -1, 0, ou être comprise entre 1 et {MAX_SEED}."
        )
    return requested


def decode_reference_image(content: bytes) -> Image.Image:
    """Décode une référence entièrement en mémoire et valide son format."""

    if not content:
        raise ValueError("L'image de référence est vide.")
    if len(content) > MAX_REFERENCE_BYTES:
        raise ValueError(
            f"L'image de référence dépasse la limite de {MAX_REFERENCE_BYTES // 1024 // 1024} Mio."
        )
    try:
        with Image.open(BytesIO(content)) as source:
            image_format = (source.format or "").upper()
            if image_format not in SUPPORTED_IMAGE_FORMATS:
                supported = ", ".join(sorted(SUPPORTED_IMAGE_FORMATS))
                raise ValueError(
                    f"Format {image_format or 'inconnu'} non pris en charge. "
                    f"Formats acceptés : {supported}."
                )
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
    except ValueError:
        raise
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Image de référence illisible : {exc}") from exc
    return image


def _filename_slug(character: str) -> str:
    normalized = unicodedata.normalize("NFKD", character.strip())
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii").casefold()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name).strip("-")
    return slug or "personnage"


def generated_filename(character: str, created_at: datetime) -> str:
    """Construit un nom sûr avec le personnage et l'horodatage UTC."""

    utc = created_at.astimezone(timezone.utc)
    return f"{_filename_slug(character)}_{utc.strftime('%Y%m%dT%H%M%S_%fZ')}.png"


def _sampling_method(value: str) -> str | None:
    normalized = value.strip().casefold()
    if normalized == DEFAULT_METHOD:
        return None
    if normalized not in SUPPORTED_SAMPLING_METHODS:
        accepted = ", ".join([DEFAULT_METHOD, *sorted(SUPPORTED_SAMPLING_METHODS)])
        raise ValueError(
            f"Méthode de sampling inconnue : {value!r}. Valeurs acceptées : {accepted}."
        )
    return normalized


def _sigma_schedule(value: str) -> str | None:
    normalized = value.strip().casefold()
    if normalized not in SUPPORTED_SIGMA_SCHEDULES:
        accepted = ", ".join(SIGMA_SCHEDULE_ORDER)
        raise ValueError(
            f"Courbe de sigmas inconnue : {value!r}. Valeurs acceptées : {accepted}."
        )
    return None if normalized == DEFAULT_SIGMAS else normalized


def _sampling_selection(method: str, sigmas: str) -> tuple[str | None, str | None]:
    sampling_method = _sampling_method(method)
    sigma_schedule = _sigma_schedule(sigmas)
    effective_method = sampling_method or DEFAULT_METHOD
    effective_sigmas = sigma_schedule or DEFAULT_SIGMAS
    supported = _supported_sigmas_for_http_method(effective_method)
    if effective_sigmas not in supported:
        accepted = ", ".join(sorted(supported))
        raise ValueError(
            f"La courbe de sigmas {effective_sigmas!r} est incompatible avec "
            f"la méthode {effective_method!r}. Valeurs acceptées : {accepted}."
        )
    return sampling_method, sigma_schedule


class GenerationService:
    """Orchestre une requête et ne persiste que le petit cache d'identité."""

    def __init__(
        self,
        config: AppConfig,
        *,
        device: str | None = None,
        dtype_name: str | None = None,
        offload_strategy: str | None = None,
        generator_factory: Callable[..., Any] = ImageGenerator,
        now_factory: Callable[[], datetime] | None = None,
        random_seed: Callable[[], int] | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self.dtype_name = dtype_name
        self.offload_strategy = offload_strategy
        self.generator_factory = generator_factory
        self.now_factory = now_factory or (lambda: datetime.now(timezone.utc))
        self.random_seed = random_seed

    def generate(
        self,
        *,
        reference_content: bytes,
        character: str,
        prompt: str,
        negative_prompt: str | None,
        clip_skip_2: bool,
        model: str,
        cfg: float,
        steps: int,
        strength: float,
        method: str,
        sigmas: str,
        seed: int,
    ) -> GeneratedPayload:
        selected_character = character.strip()
        selected_prompt = prompt.strip()
        selected_negative_prompt = (
            DEFAULT_NEGATIVE_PROMPT
            if negative_prompt is None
            else negative_prompt.strip()
        )
        if not selected_character:
            raise ValueError("Le nom du personnage ne peut pas être vide.")
        if not selected_prompt:
            raise ValueError("Le prompt ne peut pas être vide.")

        checkpoint = resolve_sdxl_checkpoint(self.config, model)
        if not checkpoint.is_file():
            raise ModelNotFoundError(
                f"Checkpoint SDXL introuvable : {checkpoint}. "
                "Choisissez un modèle renvoyé par GET /models."
            )
        sampling_method, sigma_schedule = _sampling_selection(method, sigmas)
        effective_method = sampling_method or DEFAULT_METHOD
        effective_sigmas = sigma_schedule or DEFAULT_SIGMAS
        effective_seed = resolve_generation_seed(seed, self.random_seed)
        reference_image = decode_reference_image(reference_content)
        request_config = replace(
            self.config,
            sdxl=replace(self.config.sdxl, checkpoint=checkpoint),
        )

        generator: Any | None = None
        generation_error: Exception | None = None
        try:
            generator = self.generator_factory(
                request_config,
                device=self.device,
                dtype_name=self.dtype_name,
                offload_strategy=self.offload_strategy,
                allow_downloads=False,
            )
            identity = generator.encode_identity_memory(
                reference_image,
                identity_id=selected_character,
                source_name="<http-upload>",
                use_cache=True,
            )
            generated = generator.generate_in_memory(
                prompt=selected_prompt,
                negative_prompt=selected_negative_prompt,
                clip_skip_2=clip_skip_2,
                identity=identity,
                seed=effective_seed,
                steps=steps,
                guidance_scale=cfg,
                sampling_method=sampling_method,
                sigma_schedule=sigma_schedule,
                width=DEFAULT_WIDTH,
                height=DEFAULT_HEIGHT,
                identity_strength=strength,
            )
            output = BytesIO()
            generated.image.save(output, format="PNG")
            content = output.getvalue()
        except Exception as exc:
            generation_error = exc
            raise
        finally:
            if generator is not None:
                try:
                    generator.close()
                except PuLIDAppError:
                    if generation_error is None:
                        raise

        return GeneratedPayload(
            content=content,
            filename=generated_filename(selected_character, self.now_factory()),
            seed=effective_seed,
            model=checkpoint.stem,
            method=effective_method,
            sigmas=effective_sigmas,
        )


def _http_error(exc: BaseException) -> HTTPException:
    label, cause = actionable_error(exc)
    client_errors = (
        FaceNotDetectedError,
        MultipleFacesDetectedError,
        ModelNotFoundError,
        UnsupportedDeviceError,
        ValueError,
    )
    status_code = 422 if isinstance(cause, client_errors) else 500
    return HTTPException(
        status_code=status_code,
        detail={"error": label, "message": str(cause)},
    )


def create_app(
    config_path: str | Path | None = None,
    *,
    device: str | None = None,
    dtype_name: str | None = None,
    offload_strategy: str | None = None,
    generator_factory: Callable[..., Any] = ImageGenerator,
    now_factory: Callable[[], datetime] | None = None,
    random_seed: Callable[[], int] | None = None,
    cors_origins: Sequence[str] = (),
) -> FastAPI:
    """Construit l'application et sérialise les générations lourdes."""

    config = load_config(config_path)
    require_models_root(config.models_root)
    configure_external_model_caches(config.models_root)
    service = GenerationService(
        config,
        device=device,
        dtype_name=dtype_name,
        offload_strategy=offload_strategy,
        generator_factory=generator_factory,
        now_factory=now_factory,
        random_seed=random_seed,
    )
    app = FastAPI(
        title="PuLID API",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
            expose_headers=[
                "Content-Disposition",
                "X-Generation-Seed",
                "X-SDXL-Model",
                "X-Sampling-Method",
                "X-Sigma-Schedule",
            ],
        )
    generation_lock = asyncio.Lock()

    @app.get("/models")
    async def models() -> dict[str, Any]:
        try:
            return {
                "models": list_sdxl_models(config),
                "sampling_methods": list_sampling_methods(),
                "sigma_schedules": list_sigma_schedules(),
            }
        except PuLIDAppError as exc:
            raise _http_error(exc) from exc

    @app.post("/generate", response_class=Response)
    async def generate(
        reference: Annotated[bytes, File(description="Image JPEG, PNG, WebP, BMP ou TIFF")],
        character: Annotated[str, Form(min_length=1, max_length=100)],
        prompt: Annotated[str, Form(min_length=1, max_length=4000)],
        model: Annotated[str, Form(min_length=1, max_length=255)],
        negative_prompt: Annotated[str | None, Form(max_length=4000)] = None,
        clip_skip_2: Annotated[bool, Form()] = False,
        cfg: Annotated[float, Form(ge=0, le=30)] = 7.0,
        steps: Annotated[int, Form(ge=1, le=200)] = 20,
        strength: Annotated[
            float,
            Form(ge=0, allow_inf_nan=False),
        ] = DEFAULT_IDENTITY_STRENGTH,
        method: Annotated[str, Form(min_length=1)] = DEFAULT_METHOD,
        sigmas: Annotated[str, Form(min_length=1)] = DEFAULT_SIGMAS,
        seed: Annotated[int, Form(ge=-1, le=MAX_SEED)] = 0,
    ) -> Response:
        try:
            async with generation_lock:
                payload = await run_in_threadpool(
                    service.generate,
                    reference_content=reference,
                    character=character,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    clip_skip_2=clip_skip_2,
                    model=model,
                    cfg=cfg,
                    steps=steps,
                    strength=strength,
                    method=method,
                    sigmas=sigmas,
                    seed=seed,
                )
        except (PuLIDAppError, OSError, RuntimeError, ValueError) as exc:
            raise _http_error(exc) from exc

        return Response(
            content=payload.content,
            media_type="image/png",
            headers={
                "Content-Disposition": f'attachment; filename="{payload.filename}"',
                "Cache-Control": "no-store",
                "X-Generation-Seed": str(payload.seed),
                "X-SDXL-Model": payload.model,
                "X-Sampling-Method": payload.method,
                "X-Sigma-Schedule": payload.sigmas,
            },
        )

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serveur HTTP PuLID local.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"))
    parser.add_argument("--dtype", choices=("float16", "float32"))
    parser.add_argument(
        "--offload",
        choices=("none", "model_cpu_offload"),
    )
    parser.add_argument(
        "--cors-origin",
        action="append",
        default=[],
        help="Origin frontend autorisée, répétable (ex. http://localhost:3000).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import uvicorn

    uvicorn.run(
        create_app(
            args.config,
            device=args.device,
            dtype_name=args.dtype,
            offload_strategy=args.offload,
            cors_origins=args.cors_origin,
        ),
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
