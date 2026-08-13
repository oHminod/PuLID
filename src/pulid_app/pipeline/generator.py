"""Orchestration haut niveau d'une génération PuLID + SDXL locale."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, TYPE_CHECKING

from pulid_app.config import AppConfig
from pulid_app.identity import CharacterIdentity
from pulid_app.io.images import save_image_with_metadata
from pulid_app.paths import configure_external_model_caches

if TYPE_CHECKING:
    from PIL.Image import Image
    from pulid_app.models.identity_encoder import IdentityEncoder
    from pulid_app.models.pulid_adapter import PuLIDAdapter, PuLIDIdentityFeatures
    from pulid_app.models.sdxl import SDXLModel


DEFAULT_NEGATIVE_PROMPT = (
    "flaws in the eyes, flaws in the face, low quality, worst quality, "
    "artifacts, text, watermark, deformed, mutated, disfigured, blurry"
)
SUPPORTED_DEVICE_TYPES = frozenset({"cpu", "cuda", "mps"})


class ImageGeneratorError(RuntimeError):
    """Une étape du pipeline haut niveau n'a pas pu aboutir."""


@dataclass(frozen=True)
class EncodedIdentity:
    """Identité générique mise en cache et conditionnement PuLID prêt à l'emploi."""

    character: CharacterIdentity
    conditioning: "PuLIDIdentityFeatures"
    cache_path: Path
    cache_hit: bool
    duration_seconds: float

    @property
    def id(self) -> str:
        return self.character.id

    @property
    def source_images(self) -> tuple[str, ...]:
        return tuple(self.character.source_images)


@dataclass(frozen=True)
class ImageGenerationResult:
    """Image générée, fichiers adjacents et manifeste effectif."""

    image: "Image"
    png_path: Path
    json_path: Path
    metadata: dict[str, Any]


def _normalize_device(device: str) -> str:
    normalized = device.strip().casefold()
    device_type = normalized.split(":", maxsplit=1)[0]
    if device_type not in SUPPORTED_DEVICE_TYPES:
        supported = ", ".join(sorted(SUPPORTED_DEVICE_TYPES))
        raise ImageGeneratorError(
            f"Device non pris en charge : {device!r}. Valeurs acceptées : {supported}."
        )
    return normalized


class ImageGenerator:
    """Façade lazy et réutilisable autour d'InsightFace, PuLID et SDXL."""

    def __init__(
        self,
        config: AppConfig,
        *,
        device: str | None = None,
        dtype_name: str | None = None,
        allow_downloads: bool = False,
        identity_encoder: "IdentityEncoder | None" = None,
        adapter: "PuLIDAdapter | None" = None,
        sdxl: "SDXLModel | None" = None,
    ) -> None:
        self.config = config
        configure_external_model_caches(config.models_root)
        self._device = _normalize_device(device) if device is not None else None
        self.dtype_name = dtype_name or config.device.dtype
        self.allow_downloads = allow_downloads
        self._identity_encoder = identity_encoder
        self._adapter = adapter
        self._sdxl = sdxl

    @property
    def device(self) -> str:
        """Résout le meilleur backend une seule fois, au premier usage."""

        if self._device is None:
            from pulid_app.device import get_best_device

            self._device = _normalize_device(get_best_device())
        return self._device

    def _get_identity_encoder(self) -> "IdentityEncoder":
        if self._identity_encoder is None:
            from pulid_app.models.identity_encoder import IdentityEncoder

            self._identity_encoder = IdentityEncoder.from_config(self.config)
        return self._identity_encoder

    def _get_adapter(self) -> "PuLIDAdapter":
        if self._adapter is None:
            from pulid_app.models.pulid_adapter import PuLIDAdapter

            self._adapter = PuLIDAdapter.from_config(
                self.config,
                device=self.device,
                dtype_name=self.dtype_name,
                allow_downloads=self.allow_downloads,
                identity_encoder=self._get_identity_encoder(),
            )
        return self._adapter

    def _get_sdxl(self) -> "SDXLModel":
        if self._sdxl is None:
            from pulid_app.models.sdxl import SDXLModel

            self._sdxl = SDXLModel.from_config(
                self.config,
                device=self.device,
                dtype_name=self.dtype_name,
                # Un rechargement automatique perdrait les processeurs PuLID.
                allow_dtype_fallback=False,
            )
        return self._sdxl

    def encode_identity(
        self,
        image: str | Path,
        *,
        identity_id: str | None = None,
        face_index: int | None = None,
        force_recompute: bool = False,
    ) -> EncodedIdentity:
        """Encode une référence, réutilise ArcFace puis prépare les tokens PuLID."""

        from pulid_app.models.identity_encoder import IdentityEncoderError
        from pulid_app.models.pulid_adapter import PuLIDError

        source = Path(image).expanduser().resolve(strict=False)
        selected_id = (identity_id or source.stem).strip()
        started = time.monotonic()
        try:
            encoder = self._get_identity_encoder()
            cache_path = encoder.cache_path_for(
                source,
                identity_id=selected_id,
                face_index=face_index,
            )
            cache_hit = cache_path.is_file() and not force_recompute
            character = encoder.encode_image(
                source,
                identity_id=selected_id,
                face_index=face_index,
                force_recompute=force_recompute,
            )
            conditioning = self._get_adapter().prepare_identity(
                source,
                face_embedding=character.face_embedding,
                face_index=face_index,
            )
        except (IdentityEncoderError, PuLIDError, OSError, ValueError) as exc:
            raise ImageGeneratorError(
                f"Impossible d'encoder l'identité depuis {source} : {exc}"
            ) from exc
        return EncodedIdentity(
            character=character,
            conditioning=conditioning,
            cache_path=cache_path,
            cache_hit=cache_hit,
            duration_seconds=time.monotonic() - started,
        )

    def _metadata(
        self,
        *,
        identity: EncodedIdentity,
        prompt: str,
        negative_prompt: str | None,
        seed: int,
        steps: int,
        guidance_scale: float,
        sampling_method: str | None,
        identity_strength: float,
        width: int,
        height: int,
        result: Any,
        pipeline_duration_seconds: float,
    ) -> dict[str, Any]:
        source_images = list(identity.source_images)
        return {
            "reference_image": source_images[0],
            "reference_images": source_images,
            "identity_id": identity.id,
            "identity_cache_path": str(identity.cache_path),
            "identity_cache_hit": identity.cache_hit,
            "identity_encoding_duration_seconds": identity.duration_seconds,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": result.seed,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "sampling_method": sampling_method or "default",
            "identity_strength": identity_strength,
            "width": width,
            "height": height,
            "sdxl_checkpoint": str(self.config.sdxl.checkpoint),
            "pulid_checkpoint": str(self.config.pulid.checkpoint),
            "pulid_source_dir": (
                str(self.config.pulid.source_dir)
                if self.config.pulid.source_dir is not None
                else None
            ),
            "pulid_revision": self.config.pulid.revision,
            "vae": "integrated",
            "device": result.device,
            "dtype": result.dtype,
            "dtype_fallback_used": result.dtype_fallback_used,
            "generation_duration_seconds": result.duration_seconds,
            "pipeline_duration_seconds": pipeline_duration_seconds,
            "total_duration_seconds": (
                identity.duration_seconds + pipeline_duration_seconds
            ),
        }

    def generate(
        self,
        *,
        prompt: str,
        identity: EncodedIdentity,
        negative_prompt: str | None = DEFAULT_NEGATIVE_PROMPT,
        seed: int = 42,
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        identity_strength: float = 0.8,
        guidance_scale: float = 7.0,
        sampling_method: str | None = None,
        output_prefix: str = "pulid",
    ) -> ImageGenerationResult:
        """Génère puis sauvegarde automatiquement le PNG et son manifeste JSON."""

        if not isinstance(identity, EncodedIdentity):
            raise ImageGeneratorError(
                "identity doit être produit par ImageGenerator.encode_identity()."
            )

        from pulid_app.models.pulid_adapter import PuLIDError
        from pulid_app.models.sdxl import SDXLError

        started = time.monotonic()
        adapter = self._get_adapter()
        try:
            sdxl = self._get_sdxl().load()
            sdxl.set_sampling_method(sampling_method)
            assert sdxl.pipeline is not None
            adapter.apply(sdxl.pipeline)
            adapter.set_identity(identity.conditioning, strength=identity_strength)
            try:
                cross_attention_kwargs = adapter.cross_attention_kwargs(
                    classifier_free_guidance=guidance_scale > 1.0
                )
                result = sdxl.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    steps=steps,
                    width=width,
                    height=height,
                    guidance_scale=guidance_scale,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            finally:
                adapter.clear_identity()

            pipeline_duration = time.monotonic() - started
            metadata = self._metadata(
                identity=identity,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                steps=steps,
                guidance_scale=guidance_scale,
                sampling_method=sampling_method,
                identity_strength=identity_strength,
                width=width,
                height=height,
                result=result,
                pipeline_duration_seconds=pipeline_duration,
            )
            png_path, json_path = save_image_with_metadata(
                result.image,
                metadata,
                self.config.outputs_dir,
                prefix=output_prefix,
            )
        except (PuLIDError, SDXLError, OSError, RuntimeError, ValueError) as exc:
            if isinstance(exc, ImageGeneratorError):
                raise
            raise ImageGeneratorError(f"Génération PuLID impossible : {exc}") from exc

        return ImageGenerationResult(
            image=result.image,
            png_path=png_path,
            json_path=json_path,
            metadata=metadata,
        )

    def close(self) -> None:
        """Libère les modèles détenus ; les appels répétés sont sans effet."""

        errors: list[str] = []
        if self._sdxl is not None:
            try:
                self._sdxl.close()
            except Exception as exc:
                errors.append(f"SDXL : {exc}")
        if self._adapter is not None:
            try:
                self._adapter.close()
            except Exception as exc:
                errors.append(f"PuLID : {exc}")
        self._sdxl = None
        self._adapter = None
        self._identity_encoder = None
        if errors:
            raise ImageGeneratorError(
                "Nettoyage mémoire incomplet : " + "; ".join(errors)
            )

    def __enter__(self) -> "ImageGenerator":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
