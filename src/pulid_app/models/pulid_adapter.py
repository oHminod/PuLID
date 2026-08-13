"""Adaptateur modulaire autour de l'implémentation officielle PuLID v1.1."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping
import weakref

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError

from pulid_app.config import AppConfig, DEFAULT_PULID_REVISION
from pulid_app.exceptions import (
    GenerationError,
    ModelLoadError,
    ModelNotFoundError,
    PuLIDAppError,
)
from pulid_app.models.identity_encoder import IdentityEncoder, IdentityEncoderError
from pulid_app.models.pulid_assets import (
    PuLIDAssetError,
    ensure_official_source,
)
from pulid_app.paths import configure_external_model_caches
from pulid_app.pipeline.memory import MemoryManager, MemoryManagerError


SUPPORTED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "BMP", "TIFF"})
EXPECTED_CHECKPOINT_MODULES = frozenset({"id_adapter", "id_adapter_attn_layers"})


class PuLIDError(PuLIDAppError):
    """Erreur métier de l'adaptateur PuLID."""


class PuLIDConfigurationError(PuLIDError):
    """La configuration, le checkpoint ou le runtime PuLID est invalide."""


class PuLIDLoadError(PuLIDError, ModelLoadError):
    """Les modules ou les poids PuLID ne peuvent pas être chargés."""


class PuLIDIdentityError(PuLIDError, GenerationError):
    """Les traits d'identité PuLID ne peuvent pas être préparés."""


class PuLIDModelNotFoundError(PuLIDConfigurationError, ModelNotFoundError):
    """Le checkpoint PuLID local configuré est absent."""


@dataclass(frozen=True)
class PuLIDIdentityFeatures:
    """Conditionnements positif et négatif attendus par les cross-attentions."""

    conditional: Any
    unconditional: Any
    source_images: tuple[str, ...]


def split_checkpoint_state(
    state_dict: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Sépare strictement le checkpoint selon ses deux modules officiels."""

    grouped: dict[str, dict[str, Any]] = {}
    malformed: list[str] = []
    for key, value in state_dict.items():
        module, separator, nested_key = key.partition(".")
        if not separator or not nested_key:
            malformed.append(key)
            continue
        grouped.setdefault(module, {})[nested_key] = value
    if malformed:
        raise PuLIDConfigurationError(
            "Clé(s) de checkpoint PuLID invalide(s) : " + ", ".join(malformed)
        )
    modules = frozenset(grouped)
    if modules != EXPECTED_CHECKPOINT_MODULES:
        missing = sorted(EXPECTED_CHECKPOINT_MODULES - modules)
        unexpected = sorted(modules - EXPECTED_CHECKPOINT_MODULES)
        details: list[str] = []
        if missing:
            details.append("modules absents : " + ", ".join(missing))
        if unexpected:
            details.append("modules inattendus : " + ", ".join(unexpected))
        raise PuLIDConfigurationError(
            "Structure du checkpoint PuLID incompatible (" + "; ".join(details) + ")."
        )
    return grouped


def _module_is_below(module: Any, root: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False
    try:
        Path(module_file).resolve().relative_to(root)
    except (OSError, ValueError):
        return False
    return True


class PuLIDAdapter:
    """Charge PuLID v1.1 sans exposer ses classes au pipeline métier."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        models_root: str | Path,
        source_dir: str | Path | None = None,
        revision: str = DEFAULT_PULID_REVISION,
        insightface_model_dir: str | Path | None = None,
        facexlib_root: str | Path | None = None,
        eva_clip_model: str = "EVA02-CLIP-L-14-336",
        eva_clip_pretrained: str = "eva_clip",
        device: str = "cpu",
        dtype_name: str = "float16",
        allow_downloads: bool = True,
        identity_encoder: IdentityEncoder | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=False)
        self.models_root = Path(models_root).expanduser().resolve(strict=False)
        self.source_dir = Path(
            source_dir or self.models_root / "sources" / "PuLID"
        ).expanduser().resolve(strict=False)
        self.revision = revision
        self.insightface_model_dir = (
            Path(insightface_model_dir).expanduser().resolve(strict=False)
            if insightface_model_dir is not None
            else self.models_root / "antelopev2"
        )
        self.facexlib_root = Path(
            facexlib_root or self.models_root / "facexlib" / "weights"
        ).expanduser().resolve(strict=False)
        self.eva_clip_model = eva_clip_model.strip()
        self.eva_clip_pretrained = eva_clip_pretrained.strip()
        self.device = device.strip().casefold()
        self.dtype_name = dtype_name.strip().casefold()
        self.allow_downloads = allow_downloads
        self.identity_encoder = identity_encoder

        self.memory_manager = MemoryManager(self.models_root, device=self.device)
        self._runtime: SimpleNamespace | None = None
        self._id_adapter: Any | None = None
        self._attention_state: dict[str, Any] | None = None
        self._attention_layers: Any | None = None
        self._applied_unet: weakref.ReferenceType[Any] | None = None
        self._eva_clip: Any | None = None
        self._face_helper: Any | None = None
        self._active_dtype: Any | None = None
        self._identity_features: PuLIDIdentityFeatures | None = None
        self._identity_strength = 1.0

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        *,
        device: str | None = None,
        dtype_name: str | None = None,
        allow_downloads: bool = True,
        identity_encoder: IdentityEncoder | None = None,
    ) -> "PuLIDAdapter":
        return cls(
            config.pulid.checkpoint,
            models_root=config.models_root,
            source_dir=config.pulid.source_dir,
            revision=config.pulid.revision,
            insightface_model_dir=config.insightface.model_dir,
            facexlib_root=config.pulid.facexlib_root,
            eva_clip_model=config.pulid.eva_clip_model,
            eva_clip_pretrained=config.pulid.eva_clip_pretrained,
            device=device or config.device.preferred,
            dtype_name=dtype_name or config.device.dtype,
            allow_downloads=allow_downloads,
            identity_encoder=identity_encoder,
        )

    @property
    def is_loaded(self) -> bool:
        return self._id_adapter is not None and self._attention_state is not None

    @property
    def is_applied(self) -> bool:
        return self._applied_unet is not None and self._applied_unet() is not None

    @property
    def has_identity(self) -> bool:
        return self._identity_features is not None

    @property
    def identity_strength(self) -> float:
        return self._identity_strength

    def _validate_configuration(self) -> None:
        if not self.checkpoint_path.is_file():
            raise PuLIDModelNotFoundError(
                f"Checkpoint PuLID v1.1 introuvable : {self.checkpoint_path}. "
                "Corrigez pulid.checkpoint dans la configuration."
            )
        if self.checkpoint_path.suffix.casefold() != ".safetensors":
            raise PuLIDConfigurationError(
                f"Le checkpoint PuLID doit être un .safetensors : {self.checkpoint_path}"
            )
        if not self.eva_clip_model or not self.eva_clip_pretrained:
            raise PuLIDConfigurationError(
                "eva_clip_model et eva_clip_pretrained doivent être configurés."
            )

    def _import_runtime(self) -> SimpleNamespace:
        if self._runtime is not None:
            return self._runtime
        configure_external_model_caches(self.models_root)
        try:
            official = ensure_official_source(
                self.source_dir,
                self.revision,
                allow_download=self.allow_downloads,
            )
        except PuLIDAssetError as exc:
            raise PuLIDLoadError(str(exc)) from exc

        source_text = str(official.path)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        try:
            import torch
            from safetensors.torch import load_file

            attention = importlib.import_module("pulid.attention_processor")
            encoders = importlib.import_module("pulid.encoders_transformer")
            eva_clip = importlib.import_module("eva_clip")
        except ImportError as exc:
            raise PuLIDLoadError(
                "Dépendance PuLID absente. Activez .venv puis exécutez "
                "`uv pip install -e '.[inference,pulid,dev]'`."
            ) from exc

        for module in (attention, encoders, eva_clip):
            if not _module_is_below(module, official.path):
                raise PuLIDLoadError(
                    f"Le module {module.__name__} provient de {module.__file__}, pas du "
                    f"snapshot officiel configuré {official.path}. Redémarrez le processus "
                    "après avoir retiré l'autre paquet PuLID du PYTHONPATH."
                )
        self.memory_manager.bind_torch(torch)
        self._runtime = SimpleNamespace(
            torch=torch,
            load_file=load_file,
            attention=attention,
            IDFormer=encoders.IDFormer,
            eva_clip=eva_clip,
        )
        return self._runtime

    def _resolve_dtype(self, torch_module: Any) -> Any:
        if self.device == "cpu":
            return torch_module.float32
        aliases = {
            "float16": torch_module.float16,
            "fp16": torch_module.float16,
            "float32": torch_module.float32,
            "fp32": torch_module.float32,
        }
        try:
            return aliases[self.dtype_name]
        except KeyError as exc:
            raise PuLIDConfigurationError(
                f"Dtype PuLID non pris en charge : {self.dtype_name!r}. "
                "Valeurs acceptées : float16, float32."
            ) from exc

    def load(self) -> "PuLIDAdapter":
        """Charge IDFormer et conserve les poids d'attention jusqu'à ``apply``."""

        if self.is_loaded:
            return self
        self._validate_configuration()
        runtime = self._import_runtime()
        dtype = self._resolve_dtype(runtime.torch)
        try:
            state = runtime.load_file(str(self.checkpoint_path), device="cpu")
            grouped = split_checkpoint_state(state)
            id_adapter = runtime.IDFormer()
            id_adapter.load_state_dict(grouped["id_adapter"], strict=True)
            id_adapter.eval()
            id_adapter.to(dtype=dtype)
            id_adapter = self.memory_manager.move_to_device(id_adapter, self.device)
        except (PuLIDConfigurationError, MemoryManagerError):
            raise
        except Exception as exc:
            raise PuLIDLoadError(
                f"Impossible de charger PuLID v1.1 depuis {self.checkpoint_path} : {exc}"
            ) from exc

        self._id_adapter = id_adapter
        self._attention_state = grouped["id_adapter_attn_layers"]
        self._active_dtype = dtype
        return self

    @staticmethod
    def _hidden_size(unet: Any, processor_name: str) -> int:
        channels = tuple(unet.config.block_out_channels)
        if processor_name.startswith("mid_block"):
            return int(channels[-1])
        if processor_name.startswith("up_blocks"):
            block_id = int(processor_name.removeprefix("up_blocks.").split(".", 1)[0])
            return int(tuple(reversed(channels))[block_id])
        if processor_name.startswith("down_blocks"):
            block_id = int(processor_name.removeprefix("down_blocks.").split(".", 1)[0])
            return int(channels[block_id])
        raise PuLIDConfigurationError(
            f"Nom de processeur d'attention SDXL non reconnu : {processor_name}"
        )

    def apply(self, pipeline: Any) -> Any:
        """Injecte les processeurs officiels PuLID dans l'UNet SDXL fourni."""

        if pipeline is None or getattr(pipeline, "unet", None) is None:
            raise PuLIDConfigurationError("Le pipeline fourni ne contient pas d'UNet SDXL.")
        self.load()
        assert self._runtime is not None
        assert self._attention_state is not None
        unet = pipeline.unet
        if self._applied_unet is not None and self._applied_unet() is unet:
            return pipeline

        runtime = self._runtime
        processors: dict[str, Any] = {}
        try:
            for name in unet.attn_processors:
                is_cross_attention = not name.endswith("attn1.processor")
                if is_cross_attention:
                    processor = runtime.attention.IDAttnProcessor2_0(
                        hidden_size=self._hidden_size(unet, name),
                        cross_attention_dim=unet.config.cross_attention_dim,
                    )
                else:
                    processor = runtime.attention.AttnProcessor2_0()
                processor.to(device=unet.device, dtype=unet.dtype)
                processors[name] = processor
            unet.set_attn_processor(processors)
            layers = runtime.torch.nn.ModuleList(unet.attn_processors.values())
            layers.load_state_dict(self._attention_state, strict=True)
            layers.eval()
        except Exception as exc:
            raise PuLIDLoadError(
                f"Impossible d'injecter PuLID dans l'UNet SDXL : {exc}"
            ) from exc

        self._attention_layers = layers
        self._applied_unet = weakref.ref(unet)
        return pipeline

    def _ensure_identity_encoder(self) -> IdentityEncoder:
        if self.identity_encoder is None:
            self.identity_encoder = IdentityEncoder(
                self.insightface_model_dir,
                models_root=self.models_root,
            )
        return self.identity_encoder

    def _resolve_eva_pretrained(self, runtime: SimpleNamespace) -> str:
        """Résout EVA-CLIP localement en mode hors ligne, sans requête Hub."""

        if self.allow_downloads:
            return self.eva_clip_pretrained
        try:
            pretrained = runtime.eva_clip.get_pretrained_cfg(
                self.eva_clip_model,
                self.eva_clip_pretrained,
            )
            hub_reference = pretrained.get("hf_hub", "")
            repository, filename = os.path.split(hub_reference)
            if not repository or not filename:
                raise ValueError(
                    f"référence Hugging Face absente pour {self.eva_clip_model}/"
                    f"{self.eva_clip_pretrained}"
                )
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(
                repository,
                filename,
                cache_dir=str(self.models_root / "huggingface" / "hub"),
            )
        except (ImportError, OSError, TypeError, ValueError) as exc:
            raise PuLIDLoadError(
                f"Impossible de résoudre EVA-CLIP dans le cache local : {exc}"
            ) from exc
        if not isinstance(cached, str) or not Path(cached).is_file():
            raise PuLIDLoadError(
                f"Poids EVA-CLIP absents du cache local pour {repository}/{filename}. "
                "Relancez sans --offline une première fois."
            )
        return cached

    def _load_image_bgr(
        self, image: str | Path | Image.Image | NDArray[np.uint8]
    ) -> tuple[NDArray[np.uint8], str]:
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise PuLIDIdentityError(
                    "Le tableau image doit être un RGB uint8 non vide de shape (H, W, 3)."
                )
            return np.ascontiguousarray(image[:, :, ::-1]), "<numpy>"
        if isinstance(image, Image.Image):
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            return np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1]), "<PIL>"

        path = Path(image).expanduser().resolve(strict=False)
        if not path.is_file():
            raise PuLIDIdentityError(f"Image de référence introuvable : {path}")
        try:
            with Image.open(path) as source:
                image_format = (source.format or "").upper()
                if image_format not in SUPPORTED_IMAGE_FORMATS:
                    supported = ", ".join(sorted(SUPPORTED_IMAGE_FORMATS))
                    raise PuLIDIdentityError(
                        f"Format {image_format or 'inconnu'} non pris en charge pour "
                        f"{path}. Formats acceptés : {supported}."
                    )
                rgb = ImageOps.exif_transpose(source).convert("RGB")
                bgr = np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1])
        except PuLIDIdentityError:
            raise
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise PuLIDIdentityError(f"Image de référence illisible : {path} ({exc})") from exc
        return bgr, str(path)

    def _ensure_preprocessors(self) -> None:
        if self._eva_clip is not None and self._face_helper is not None:
            return
        runtime = self._import_runtime()
        if not self.allow_downloads:
            required_facexlib = (
                "detection_Resnet50_Final.pth",
                "parsing_parsenet.pth",
                "parsing_bisenet.pth",
            )
            missing = [
                self.facexlib_root / name
                for name in required_facexlib
                if not (self.facexlib_root / name).is_file()
            ]
            if missing:
                raise PuLIDLoadError(
                    "Poids FaceXLib absents en mode hors ligne : "
                    + ", ".join(str(path) for path in missing)
                )
        try:
            from facexlib.parsing import init_parsing_model
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper

            face_helper = FaceRestoreHelper(
                upscale_factor=1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model="retinaface_resnet50",
                save_ext="png",
                device=runtime.torch.device("cpu"),
                model_rootpath=str(self.facexlib_root),
            )
            face_helper.face_parse = init_parsing_model(
                model_name="bisenet",
                device=runtime.torch.device("cpu"),
                model_rootpath=str(self.facexlib_root),
            )
            eva_pretrained = self._resolve_eva_pretrained(runtime)
            model, _, _ = runtime.eva_clip.create_model_and_transforms(
                self.eva_clip_model,
                eva_pretrained,
                force_custom_clip=True,
                device="cpu",
                precision="fp32",
                cache_dir=str(self.models_root / "huggingface" / "hub"),
            )
            eva_clip = model.visual.eval()
        except Exception as exc:
            action = (
                "Vérifiez l'accès réseau et que les caches affichés par "
                "`python scripts/inspect_models.py --show-cache-env` pointent vers le SSD."
                if self.allow_downloads
                else "Relancez avec allow_downloads=True pour préparer EVA-CLIP/FaceXLib."
            )
            raise PuLIDLoadError(
                f"Impossible de préparer EVA-CLIP ou FaceXLib : {exc}. {action}"
            ) from exc
        self._face_helper = face_helper
        self._eva_clip = eva_clip

    @staticmethod
    def _to_gray(tensor: Any) -> Any:
        gray = 0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3]
        return gray.repeat(1, 3, 1, 1)

    def prepare_identity(
        self,
        image: str | Path | Image.Image | NDArray[np.uint8],
        face_embedding: NDArray[np.float32] | None = None,
        *,
        face_index: int | None = None,
    ) -> PuLIDIdentityFeatures:
        """Construit les 32 tokens PuLID depuis une référence et ArcFace."""

        self.load()
        self._ensure_preprocessors()
        assert self._runtime is not None
        assert self._id_adapter is not None
        assert self._eva_clip is not None
        assert self._face_helper is not None
        runtime = self._runtime
        torch = runtime.torch
        image_bgr, source_name = self._load_image_bgr(image)

        if face_embedding is None:
            try:
                encoded = self._ensure_identity_encoder().encode(
                    image_bgr,
                    face_index=face_index,
                )
            except IdentityEncoderError as exc:
                raise PuLIDIdentityError(f"Embedding InsightFace impossible : {exc}") from exc
            embedding = encoded.embedding
        else:
            embedding = np.asarray(face_embedding, dtype=np.float32).reshape(-1)
        if embedding.shape != (512,) or not np.isfinite(embedding).all():
            raise PuLIDIdentityError(
                f"L'embedding facial doit avoir shape (512,), reçu {embedding.shape}."
            )

        face_helper = self._face_helper
        try:
            face_helper.clean_all()
            face_helper.read_image(image_bgr)
            face_helper.get_face_landmarks_5(only_center_face=True)
            face_helper.align_warp_face()
            if not face_helper.cropped_faces:
                raise RuntimeError("FaceXLib n'a aligné aucun visage")
            aligned_bgr = np.ascontiguousarray(face_helper.cropped_faces[0])

            face_tensor = torch.from_numpy(aligned_bgr[:, :, ::-1].copy())
            face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0
            from torchvision.transforms import InterpolationMode
            from torchvision.transforms.functional import normalize, resize

            parsing = face_helper.face_parse(
                normalize(
                    face_tensor,
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                )
            )[0].argmax(dim=1, keepdim=True)
            background = torch.zeros_like(parsing, dtype=torch.bool)
            for label in (0, 16, 18, 7, 8, 9, 14, 15):
                background |= parsing == label
            face_features = torch.where(
                background,
                torch.ones_like(face_tensor),
                self._to_gray(face_tensor),
            )

            self._eva_clip.to(dtype=self._active_dtype)
            eva_clip = self.memory_manager.move_to_device(self._eva_clip, self.device)
            face_features = face_features.to(
                device=self.device,
                dtype=self._active_dtype,
            )
            interpolation = (
                InterpolationMode.NEAREST
                if self.device == "mps"
                else InterpolationMode.BICUBIC
            )
            face_features = resize(
                face_features,
                eva_clip.image_size,
                interpolation,
            )
            mean = getattr(eva_clip, "image_mean")
            std = getattr(eva_clip, "image_std")
            if not isinstance(mean, (tuple, list)):
                mean = (mean,) * 3
            if not isinstance(std, (tuple, list)):
                std = (std,) * 3
            face_features = normalize(face_features, mean, std)

            with torch.inference_mode():
                id_cond_vit, hidden = eva_clip(
                    face_features,
                    return_all_features=False,
                    return_hidden=True,
                    shuffle=False,
                )
                id_cond_vit = id_cond_vit / torch.norm(
                    id_cond_vit,
                    p=2,
                    dim=1,
                    keepdim=True,
                ).clamp_min(1e-12)
                face_tensor_embedding = torch.from_numpy(embedding).unsqueeze(0).to(
                    device=self.device,
                    dtype=self._active_dtype,
                )
                id_condition = torch.cat(
                    (face_tensor_embedding, id_cond_vit), dim=-1
                ).unsqueeze(1)
                unconditional_condition = torch.zeros_like(id_condition[:, 0])
                unconditional_hidden = [torch.zeros_like(item) for item in hidden]
                conditional = self._id_adapter(id_condition, hidden)
                unconditional = self._id_adapter(
                    unconditional_condition,
                    unconditional_hidden,
                )
        except PuLIDIdentityError:
            raise
        except Exception as exc:
            raise PuLIDIdentityError(
                f"Préparation des traits PuLID impossible pour {source_name} : {exc}"
            ) from exc
        finally:
            if self._eva_clip is not None:
                try:
                    self.memory_manager.unload(self._eva_clip)
                except MemoryManagerError:
                    pass

        return PuLIDIdentityFeatures(
            conditional=conditional,
            unconditional=unconditional,
            source_images=(source_name,),
        )

    def set_identity(
        self,
        identity_features: PuLIDIdentityFeatures,
        strength: float = 1.0,
    ) -> None:
        """Active un conditionnement d'identité et sa force d'injection."""

        if not isinstance(identity_features, PuLIDIdentityFeatures):
            raise PuLIDIdentityError(
                "identity_features doit être produit par prepare_identity()."
            )
        if not math.isfinite(strength) or strength < 0:
            raise PuLIDIdentityError("La force d'identité doit être finie et positive ou nulle.")
        self._identity_features = identity_features
        self._identity_strength = float(strength)

    def cross_attention_kwargs(
        self,
        *,
        classifier_free_guidance: bool = True,
    ) -> dict[str, Any]:
        """Expose les seuls kwargs PuLID nécessaires à une future génération."""

        if self._identity_features is None:
            return {}
        runtime = self._import_runtime()
        features = self._identity_features
        embedding = features.conditional
        if classifier_free_guidance:
            embedding = runtime.torch.cat(
                (features.unconditional, features.conditional), dim=0
            )
        return {
            "id_embedding": embedding,
            "id_scale": self._identity_strength,
        }

    def clear_identity(self) -> None:
        """Retire le conditionnement courant sans désinstaller l'adaptateur."""

        had_identity = self._identity_features is not None
        self._identity_features = None
        self._identity_strength = 1.0
        if had_identity:
            try:
                self.memory_manager.cleanup(force=True)
            except MemoryManagerError:
                pass

    def close(self) -> None:
        """Décharge les encodeurs détenus par l'adaptateur."""

        self.clear_identity()
        try:
            self.memory_manager.unload(self._eva_clip, cleanup=False)
            self.memory_manager.unload(self._id_adapter)
        except MemoryManagerError as exc:
            raise PuLIDLoadError(f"Impossible de libérer PuLID : {exc}") from exc
        self._eva_clip = None
        self._id_adapter = None
        self._attention_state = None
        self._attention_layers = None
        self._applied_unet = None
        self._active_dtype = None
