from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from pulid_app.models.identity_encoder import (
    FaceIndexError,
    IdentityEncoder,
    ImageReadError,
    ModelLoadError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)


def _face_data(count: int) -> tuple[np.ndarray, np.ndarray]:
    bboxes = np.asarray(
        [[10 + index, 20, 110 + index, 120, 0.95] for index in range(count)],
        dtype=np.float32,
    )
    keypoints = np.ones((count, 5, 2), dtype=np.float32) * 42
    return bboxes, keypoints


class FakeDetector:
    def __init__(self, count: int) -> None:
        self.count = count

    def detect(self, _image, *, max_num: int, metric: str):
        assert max_num == 0
        assert metric == "default"
        return _face_data(self.count)


class FakeRecognizer:
    def get(self, _image, face):
        face.embedding = np.asarray([3.0, 4.0], dtype=np.float32)


def _loaded_encoder(tmp_path: Path, face_count: int) -> IdentityEncoder:
    encoder = IdentityEncoder(tmp_path)
    encoder._detector = FakeDetector(face_count)
    encoder._recognizer = FakeRecognizer()
    return encoder


def test_webp_image_is_read_and_encoded(tmp_path: Path) -> None:
    image_path = tmp_path / "reference.webp"
    Image.new("RGB", (32, 32), (120, 80, 40)).save(image_path, format="WEBP")
    encoder = _loaded_encoder(tmp_path, face_count=1)

    result = encoder.encode(image_path)

    assert result.face_count == 1
    assert result.embedding.shape == (2,)
    assert result.norm == pytest.approx(5.0)
    np.testing.assert_allclose(result.normalized_embedding, [0.6, 0.8])


def test_no_face_is_rejected(tmp_path: Path) -> None:
    encoder = _loaded_encoder(tmp_path, face_count=0)
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    with pytest.raises(NoFaceDetectedError, match="Aucun visage"):
        encoder.encode(image)


def test_multiple_faces_require_explicit_index(tmp_path: Path) -> None:
    encoder = _loaded_encoder(tmp_path, face_count=2)
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    with pytest.raises(MultipleFacesDetectedError) as raised:
        encoder.encode(image)

    assert raised.value.count == 2
    assert encoder.encode(image, face_index=1).face_index == 1


def test_invalid_face_index_is_rejected(tmp_path: Path) -> None:
    encoder = _loaded_encoder(tmp_path, face_count=1)
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    with pytest.raises(FaceIndexError, match="Index de visage"):
        encoder.encode(image, face_index=2)


def test_unreadable_image_is_rejected(tmp_path: Path) -> None:
    image_path = tmp_path / "broken.webp"
    image_path.write_text("not an image", encoding="utf-8")
    encoder = _loaded_encoder(tmp_path, face_count=1)

    with pytest.raises(ImageReadError, match="Image illisible"):
        encoder.detect(image_path)


def test_missing_model_directory_is_rejected(tmp_path: Path) -> None:
    model_dir = tmp_path / "missing-antelopev2"

    with pytest.raises(ModelLoadError, match="Dossier AntelopeV2 introuvable"):
        IdentityEncoder(model_dir).load()

