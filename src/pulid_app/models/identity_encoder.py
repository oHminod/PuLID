"""Détection faciale et extraction d'embeddings avec AntelopeV2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError

from pulid_app.config import AppConfig
from pulid_app.paths import configure_external_model_caches


DETECTION_MODEL = "scrfd_10g_bnkps.onnx"
RECOGNITION_MODEL = "glintr100.onnx"
DEFAULT_PROVIDERS = ("CPUExecutionProvider",)

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
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve(strict=False)
        self.models_root = Path(models_root or self.model_dir.parent).expanduser().resolve(
            strict=False
        )
        self.providers = tuple(providers)
        self.detection_size = detection_size
        self.detection_threshold = detection_threshold
        self._detector: Any | None = None
        self._recognizer: Any | None = None

    @classmethod
    def from_config(cls, config: AppConfig) -> "IdentityEncoder":
        return cls(
            config.insightface.model_dir,
            models_root=config.models_root,
            providers=DEFAULT_PROVIDERS,
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

        path = Path(image).expanduser().resolve(strict=False)
        if not path.is_file():
            raise ImageReadError(f"Image introuvable : {path}")
        try:
            with Image.open(path) as source:
                rgb = ImageOps.exif_transpose(source).convert("RGB")
                rgb_array = np.asarray(rgb, dtype=np.uint8)
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise ImageReadError(f"Image illisible : {path} ({exc})") from exc

        # InsightFace/OpenCV attend du BGR, y compris pour une source WebP.
        return np.ascontiguousarray(rgb_array[:, :, ::-1])

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

    def encode(
        self,
        image: ImageInput,
        *,
        face_index: int | None = None,
    ) -> EncodedFace:
        """Extrait l'embedding d'une face unique ou explicitement sélectionnée."""

        image_bgr = self._load_image(image)
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

