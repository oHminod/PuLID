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
    def __init__(self) -> None:
        self.calls = 0

    def get(self, _image, face):
        self.calls += 1
        face.embedding = np.asarray([3.0, 4.0], dtype=np.float32)


def _loaded_encoder(tmp_path: Path, face_count: int) -> IdentityEncoder:
    encoder = IdentityEncoder(tmp_path)
    encoder._detector = FakeDetector(face_count)
    encoder._recognizer = FakeRecognizer()
    return encoder


@pytest.mark.parametrize(
    ("image_format", "suffix"),
    [("JPEG", ".jpg"), ("PNG", ".png"), ("WEBP", ".webp")],
)
def test_supported_image_formats_are_read_and_encoded(
    tmp_path: Path, image_format: str, suffix: str
) -> None:
    image_path = tmp_path / f"reference{suffix}"
    Image.new("RGB", (32, 32), (120, 80, 40)).save(
        image_path, format=image_format
    )
    encoder = _loaded_encoder(tmp_path, face_count=1)

    result = encoder.encode(image_path)

    assert result.face_count == 1
    assert result.embedding.shape == (2,)
    assert result.norm == pytest.approx(5.0)
    np.testing.assert_allclose(result.normalized_embedding, [0.6, 0.8])


def test_unsupported_image_format_is_rejected(tmp_path: Path) -> None:
    image_path = tmp_path / "animated.gif"
    Image.new("RGB", (32, 32), (120, 80, 40)).save(image_path, format="GIF")
    encoder = _loaded_encoder(tmp_path, face_count=1)

    with pytest.raises(ImageReadError, match="Format d'image GIF non pris en charge"):
        encoder.encode(image_path)


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


def test_second_encode_image_call_reuses_content_cache(tmp_path: Path) -> None:
    model_dir = tmp_path / "antelopev2"
    model_dir.mkdir()
    (model_dir / "scrfd_10g_bnkps.onnx").write_bytes(b"detector")
    (model_dir / "glintr100.onnx").write_bytes(b"recognizer")
    cache_dir = tmp_path / "cache" / "identity"
    image_path = tmp_path / "noemie.png"
    Image.new("RGB", (32, 32), (120, 80, 40)).save(image_path, format="PNG")

    encoder = _loaded_encoder(model_dir, face_count=1)
    encoder.identity_cache_dir = cache_dir
    recognizer = encoder._recognizer
    first = encoder.encode_image(image_path, identity_id="noemie")

    assert isinstance(recognizer, FakeRecognizer)
    assert recognizer.calls == 1
    cache_path = encoder.cache_path_for(image_path, identity_id="noemie")
    assert cache_path.is_file()
    assert cache_path.name.startswith("noemie_")

    cold_encoder = IdentityEncoder(model_dir, identity_cache_dir=cache_dir)
    second = cold_encoder.encode_image(image_path, identity_id="noemie")

    assert cold_encoder.is_loaded is False
    assert second.id == "noemie"
    assert second.metadata["source_format"] == "PNG"
    np.testing.assert_array_equal(second.face_embedding, first.face_embedding)


def test_cache_normalizes_character_name(tmp_path: Path) -> None:
    model_dir = tmp_path / "antelopev2"
    model_dir.mkdir()
    (model_dir / "scrfd_10g_bnkps.onnx").write_bytes(b"detector")
    (model_dir / "glintr100.onnx").write_bytes(b"recognizer")
    image_path = tmp_path / "noemie.webp"
    Image.new("RGB", (32, 32)).save(image_path, format="WEBP")
    encoder = _loaded_encoder(model_dir, face_count=1)
    encoder.identity_cache_dir = tmp_path / "cache"

    first = encoder.encode_image(image_path, identity_id="  noemie  ")
    cold_encoder = IdentityEncoder(model_dir, identity_cache_dir=encoder.identity_cache_dir)
    second = cold_encoder.encode_image(image_path, identity_id="noemie")

    assert first.id == "noemie"
    assert second.id == "noemie"
    assert cold_encoder.is_loaded is False
