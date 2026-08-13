"""Détection faciale et extraction d'embeddings avec AntelopeV2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Sequence
import unicodedata

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError

from pulid_app.config import AppConfig
from pulid_app.identity import (
    CharacterIdentity,
    IdentitySerializationError,
    sha256_file,
)
from pulid_app.paths import configure_external_model_caches


DETECTION_MODEL = "scrfd_10g_bnkps.onnx"
RECOGNITION_MODEL = "glintr100.onnx"
DEFAULT_PROVIDERS = ("CPUExecutionProvider",)
SUPPORTED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "BMP", "TIFF"})

ImageInput = str | Path | NDArray[np.generic]


class IdentityEncoderError(RuntimeError):
    """Erreur métier de l'encodeur d'identité."""


class ModelLoadError(IdentityEncoderError):
    """Les modèles locaux ne peuvent pas être chargés."""


class ImageReadError(IdentityEncoderError):
    """L'image est absente, illisible ou dans un format incorrect."""


class NoFaceDetectedError(IdentityEncoderError):
    """Aucun visage n'a été détecté."""


class MultipleFacesDetectedError(IdentityEncoderError):
    """Plusieurs visages exigent une sélection explicite."""

    def __init__(self, count: int) -> None:
        self.count = count
        super().__init__(
            f"{count} visages détectés. Sélectionnez-en un explicitement avec "
            "face_index (ou --face-index dans le script)."
        )


class FaceIndexError(IdentityEncoderError):
    """L'index de visage demandé n'existe pas."""


class IdentityCacheError(IdentityEncoderError):
    """Le cache d'identité est absent, incohérent ou corrompu."""


@dataclass(frozen=True)
class FaceDetection:
    """Résultat géométrique d'une détection faciale."""

    bbox: tuple[float, float, float, float]
    score: float
    keypoints: NDArray[np.float32] | None


@dataclass(frozen=True)
class EncodedFace:
    """Embedding associé à un visage sélectionné dans une image."""

    embedding: NDArray[np.float32]
    detection: FaceDetection
    face_count: int
    face_index: int

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.embedding))

    @property
    def normalized_embedding(self) -> NDArray[np.float32]:
        norm = self.norm
        if norm == 0:
            raise IdentityEncoderError("L'embedding facial a une norme nulle.")
        return self.embedding / norm


@dataclass(frozen=True)
class ImageFileInfo:
    """Informations obtenues pendant le décodage d'une image locale."""

    path: Path
    format: str
    width: int
    height: int


class IdentityEncoder:
    """Charge SCRFD et ArcFace depuis un dossier AntelopeV2 strictement local."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        models_root: str | Path | None = None,
        providers: Sequence[str] = DEFAULT_PROVIDERS,
        detection_size: tuple[int, int] = (640, 640),
        detection_threshold: float = 0.5,
        identity_cache_dir: str | Path | None = None,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve(strict=False)
        self.models_root = Path(models_root or self.model_dir.parent).expanduser().resolve(
            strict=False
        )
        self.providers = tuple(providers)
        self.detection_size = detection_size
        self.detection_threshold = detection_threshold
        self.identity_cache_dir = (
            Path(identity_cache_dir).expanduser().resolve(strict=False)
            if identity_cache_dir is not None
            else None
        )
        self._detector: Any | None = None
        self._recognizer: Any | None = None

    @classmethod
    def from_config(cls, config: AppConfig) -> "IdentityEncoder":
        return cls(
            config.insightface.model_dir,
            models_root=config.models_root,
            providers=DEFAULT_PROVIDERS,
            identity_cache_dir=config.identity_cache_dir,
        )

    @property
    def is_loaded(self) -> bool:
        return self._detector is not None and self._recognizer is not None

    def _required_model_paths(self) -> tuple[Path, Path]:
        return (
            self.model_dir / DETECTION_MODEL,
            self.model_dir / RECOGNITION_MODEL,
        )

    def load(self) -> "IdentityEncoder":
        """Charge les deux sessions ONNX sans résolution distante de modèles."""

        if self.is_loaded:
            return self
        if not self.model_dir.is_dir():
            raise ModelLoadError(
                f"Dossier AntelopeV2 introuvable : {self.model_dir}. "
                "Corrigez insightface.model_root/model_name dans la configuration."
            )

        missing = [path for path in self._required_model_paths() if not path.is_file()]
        if missing:
            missing_text = ", ".join(str(path) for path in missing)
            raise ModelLoadError(f"Modèle(s) ONNX requis introuvable(s) : {missing_text}")
        if not self.providers:
            raise ModelLoadError("Au moins un ONNX Execution Provider doit être configuré.")

        configure_external_model_caches(self.models_root)
        try:
            import onnxruntime
            from insightface.model_zoo import get_model
        except ImportError as exc:
            raise ModelLoadError(
                "InsightFace/ONNX Runtime n'est pas installé. Activez .venv puis "
                "exécutez `uv pip install -e '.[dev]'`."
            ) from exc

        available = set(onnxruntime.get_available_providers())
        unavailable = [provider for provider in self.providers if provider not in available]
        if unavailable:
            raise ModelLoadError(
                "ONNX Execution Provider indisponible : "
                f"{', '.join(unavailable)}. Providers disponibles : "
                f"{', '.join(sorted(available)) or 'aucun'}."
            )

        detector_path, recognizer_path = self._required_model_paths()
        try:
            detector = get_model(str(detector_path), providers=list(self.providers))
            recognizer = get_model(str(recognizer_path), providers=list(self.providers))
            if detector is None or recognizer is None:
                raise RuntimeError("InsightFace n'a pas reconnu un modèle ONNX local.")
            detector.prepare(
                ctx_id=-1,
                input_size=self.detection_size,
                det_thresh=self.detection_threshold,
            )
            recognizer.prepare(ctx_id=-1)
        except Exception as exc:
            raise ModelLoadError(
                f"Échec du chargement d'AntelopeV2 depuis {self.model_dir} : {exc}"
            ) from exc

        self._detector = detector
        self._recognizer = recognizer
        return self

    def _load_image_file(
        self, image: str | Path
    ) -> tuple[NDArray[np.uint8], ImageFileInfo]:
        path = Path(image).expanduser().resolve(strict=False)
        if not path.is_file():
            raise ImageReadError(f"Image introuvable : {path}")
        try:
            with Image.open(path) as source:
                image_format = (source.format or "").upper()
                if image_format not in SUPPORTED_IMAGE_FORMATS:
                    supported = ", ".join(sorted(SUPPORTED_IMAGE_FORMATS))
                    raise ImageReadError(
                        f"Format d'image {image_format or 'inconnu'} non pris en charge "
                        f"pour {path}. Formats acceptés : {supported}."
                    )
                rgb = ImageOps.exif_transpose(source).convert("RGB")
                width, height = rgb.size
                rgb_array = np.asarray(rgb, dtype=np.uint8)
        except ImageReadError:
            raise
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise ImageReadError(f"Image illisible : {path} ({exc})") from exc

        bgr = np.ascontiguousarray(rgb_array[:, :, ::-1])
        return bgr, ImageFileInfo(
            path=path,
            format=image_format,
            width=width,
            height=height,
        )

    def _load_image(self, image: ImageInput) -> NDArray[np.uint8]:
        if isinstance(image, np.ndarray):
            array = image
            if array.ndim != 3 or array.shape[2] != 3 or array.size == 0:
                raise ImageReadError(
                    "Le tableau image doit être un tableau BGR non vide de shape (H, W, 3)."
                )
            if array.dtype != np.uint8:
                raise ImageReadError("Le tableau image doit utiliser le dtype uint8.")
            return np.ascontiguousarray(array)

        # InsightFace/OpenCV attend du BGR pour JPEG, PNG, WebP, BMP et TIFF.
        return self._load_image_file(image)[0]

    def _detect_array(self, image_bgr: NDArray[np.uint8]) -> tuple[FaceDetection, ...]:
        self.load()
        assert self._detector is not None
        try:
            bboxes, keypoints = self._detector.detect(
                image_bgr,
                max_num=0,
                metric="default",
            )
        except Exception as exc:
            raise IdentityEncoderError(f"Échec de la détection faciale : {exc}") from exc

        if bboxes is None or len(bboxes) == 0:
            return ()

        detections: list[FaceDetection] = []
        for index, raw_bbox in enumerate(np.asarray(bboxes)):
            if raw_bbox.shape[0] < 4:
                raise IdentityEncoderError(
                    f"Bounding box InsightFace invalide à l'index {index}: {raw_bbox}."
                )
            bbox = tuple(float(value) for value in raw_bbox[:4])
            score = float(raw_bbox[4]) if raw_bbox.shape[0] >= 5 else 1.0
            face_keypoints: NDArray[np.float32] | None = None
            if keypoints is not None and index < len(keypoints):
                face_keypoints = np.asarray(keypoints[index], dtype=np.float32)
            detections.append(
                FaceDetection(
                    bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    score=score,
                    keypoints=face_keypoints,
                )
            )
        return tuple(detections)

    def detect(self, image: ImageInput) -> tuple[FaceDetection, ...]:
        """Retourne toutes les faces détectées, triées comme par SCRFD."""

        return self._detect_array(self._load_image(image))

    def _encode_array(
        self,
        image_bgr: NDArray[np.uint8],
        face_index: int | None,
    ) -> EncodedFace:
        detections = self._detect_array(image_bgr)
        face_count = len(detections)
        if face_count == 0:
            raise NoFaceDetectedError("Aucun visage détecté dans l'image.")
        if face_count > 1 and face_index is None:
            raise MultipleFacesDetectedError(face_count)

        selected_index = 0 if face_index is None else face_index
        if selected_index < 0 or selected_index >= face_count:
            raise FaceIndexError(
                f"Index de visage {selected_index} invalide ; utilisez une valeur entre "
                f"0 et {face_count - 1}."
            )
        selected = detections[selected_index]
        if selected.keypoints is None:
            raise IdentityEncoderError(
                "Les points de repère faciaux sont absents ; l'alignement ArcFace est impossible."
            )

        assert self._recognizer is not None
        face_record = SimpleNamespace(
            bbox=np.asarray(selected.bbox, dtype=np.float32),
            kps=selected.keypoints,
            det_score=selected.score,
            embedding=None,
        )
        try:
            returned = self._recognizer.get(image_bgr, face_record)
        except Exception as exc:
            raise IdentityEncoderError(f"Échec de l'extraction de l'embedding : {exc}") from exc

        raw_embedding = face_record.embedding
        if raw_embedding is None and returned is not None:
            raw_embedding = returned
        if raw_embedding is None:
            raise IdentityEncoderError("InsightFace n'a retourné aucun embedding facial.")

        embedding = np.asarray(raw_embedding, dtype=np.float32).reshape(-1)
        if embedding.size == 0 or not np.isfinite(embedding).all():
            raise IdentityEncoderError("L'embedding facial est vide ou contient des valeurs invalides.")
        return EncodedFace(
            embedding=embedding,
            detection=selected,
            face_count=face_count,
            face_index=selected_index,
        )

    def encode(
        self,
        image: ImageInput,
        *,
        face_index: int | None = None,
    ) -> EncodedFace:
        """Extrait l'embedding d'une face unique ou explicitement sélectionnée."""

        return self._encode_array(self._load_image(image), face_index)

    def _encoder_fingerprint(self) -> str:
        model_files: list[dict[str, int | str | None]] = []
        for path in self._required_model_paths():
            try:
                stat = path.stat()
            except OSError:
                model_files.append({"name": path.name, "size": None, "mtime_ns": None})
            else:
                model_files.append(
                    {
                        "name": path.name,
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                )
        payload = {
            "models": model_files,
            "providers": self.providers,
            "detection_size": self.detection_size,
            "detection_threshold": self.detection_threshold,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _identity_slug(identity_id: str) -> str:
        cleaned = identity_id.strip()
        if not cleaned:
            raise IdentityCacheError("L'identifiant du personnage est vide.")
        normalized = unicodedata.normalize("NFKD", cleaned)
        ascii_name = normalized.encode("ascii", "ignore").decode("ascii").casefold()
        slug = re.sub(r"[^a-z0-9]+", "-", ascii_name).strip("-")
        if slug:
            return slug
        digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:12]
        return f"identity-{digest}"

    def _effective_cache_dir(self, cache_dir: str | Path | None) -> Path:
        selected = cache_dir if cache_dir is not None else self.identity_cache_dir
        if selected is None:
            raise IdentityCacheError(
                "Aucun dossier de cache d'identité n'est configuré. Fournissez "
                "cache_dir ou construisez l'encodeur avec IdentityEncoder.from_config()."
            )
        return Path(selected).expanduser().resolve(strict=False)

    def cache_path_for(
        self,
        image: str | Path,
        *,
        identity_id: str,
        face_index: int | None = None,
        cache_dir: str | Path | None = None,
    ) -> Path:
        """Retourne le chemin adressé par le contenu et la configuration d'encodage."""

        identity_id = identity_id.strip()
        self._identity_slug(identity_id)
        source = Path(image).expanduser().resolve(strict=False)
        try:
            source_hash = sha256_file(source)
        except IdentitySerializationError as exc:
            raise IdentityCacheError(str(exc)) from exc
        selector = "auto" if face_index is None else f"face-{face_index}"
        filename = (
            f"{self._identity_slug(identity_id)}_{source_hash[:24]}_{selector}_"
            f"{self._encoder_fingerprint()[:12]}.npz"
        )
        return self._effective_cache_dir(cache_dir) / filename

    def _validate_cached_identity(
        self,
        identity: CharacterIdentity,
        *,
        identity_id: str,
        source_hash: str,
        requested_face_index: int | None,
        encoder_fingerprint: str,
        cache_path: Path,
    ) -> None:
        expected_index = requested_face_index
        checks = {
            "id": (identity.id, identity_id),
            "source_sha256": (identity.metadata.get("source_sha256"), source_hash),
            "requested_face_index": (
                identity.metadata.get("requested_face_index"),
                expected_index,
            ),
            "encoder_fingerprint": (
                identity.metadata.get("encoder_fingerprint"),
                encoder_fingerprint,
            ),
        }
        mismatches = [
            name for name, (actual, expected) in checks.items() if actual != expected
        ]
        if mismatches:
            raise IdentityCacheError(
                f"Cache d'identité incohérent {cache_path} ; champs invalides : "
                f"{', '.join(mismatches)}. Supprimez ce fichier pour le recalculer."
            )

    def encode_image(
        self,
        image: str | Path,
        *,
        identity_id: str,
        face_index: int | None = None,
        cache_dir: str | Path | None = None,
        force_recompute: bool = False,
    ) -> CharacterIdentity:
        """Encode une image et réutilise son cache NPZ adressé par SHA-256."""

        identity_id = identity_id.strip()
        self._identity_slug(identity_id)
        image_bgr, image_info = self._load_image_file(image)
        try:
            source_hash = sha256_file(image_info.path)
        except IdentitySerializationError as exc:
            raise IdentityCacheError(str(exc)) from exc
        encoder_fingerprint = self._encoder_fingerprint()
        selector = "auto" if face_index is None else f"face-{face_index}"
        cache_path = self._effective_cache_dir(cache_dir) / (
            f"{self._identity_slug(identity_id)}_{source_hash[:24]}_{selector}_"
            f"{encoder_fingerprint[:12]}.npz"
        )

        if cache_path.is_file() and not force_recompute:
            try:
                cached = CharacterIdentity.load(cache_path)
            except IdentitySerializationError as exc:
                raise IdentityCacheError(
                    f"Cache d'identité illisible {cache_path} : {exc}. "
                    "Supprimez ce fichier pour le recalculer."
                ) from exc
            self._validate_cached_identity(
                cached,
                identity_id=identity_id,
                source_hash=source_hash,
                requested_face_index=face_index,
                encoder_fingerprint=encoder_fingerprint,
                cache_path=cache_path,
            )
            if cached.source_images != [str(image_info.path)]:
                cached = CharacterIdentity(
                    id=cached.id,
                    source_images=[str(image_info.path)],
                    face_embedding=cached.face_embedding,
                    metadata=cached.metadata,
                )
            return cached

        encoded = self._encode_array(image_bgr, face_index)
        identity = CharacterIdentity(
            id=identity_id,
            source_images=[str(image_info.path)],
            face_embedding=encoded.embedding,
            metadata={
                "source_sha256": source_hash,
                "source_format": image_info.format,
                "source_width": image_info.width,
                "source_height": image_info.height,
                "model_dir": str(self.model_dir),
                "providers": list(self.providers),
                "encoder_fingerprint": encoder_fingerprint,
                "requested_face_index": face_index,
                "selected_face_index": encoded.face_index,
                "face_count": encoded.face_count,
                "bounding_box": list(encoded.detection.bbox),
                "detection_score": encoded.detection.score,
                "embedding_norm_l2": encoded.norm,
            },
        )
        try:
            identity.save(cache_path)
        except IdentitySerializationError as exc:
            raise IdentityCacheError(str(exc)) from exc
        return identity
