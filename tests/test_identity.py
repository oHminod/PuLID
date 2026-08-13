from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pulid_app.identity import (
    CharacterIdentity,
    IdentitySerializationError,
    sha256_file,
)


def test_character_identity_npz_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "noemie.webp"
    source.write_bytes(b"webp-content")
    identity = CharacterIdentity(
        id="noemie",
        source_images=[str(source)],
        face_embedding=np.asarray([1.5, -2.0, 3.25], dtype=np.float32),
        metadata={"source_format": "WEBP", "face_count": 1},
    )
    cache_path = tmp_path / "cache" / "identity" / "noemie.npz"

    saved_path = identity.save(cache_path)
    loaded = CharacterIdentity.load(saved_path)

    assert loaded.id == "noemie"
    assert loaded.source_images == [str(source)]
    assert loaded.metadata == {"source_format": "WEBP", "face_count": 1}
    np.testing.assert_array_equal(loaded.face_embedding, identity.face_embedding)


def test_character_identity_rejects_corrupted_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "corrupted.npz"
    cache_path.write_bytes(b"not-an-npz")

    with pytest.raises(IdentitySerializationError, match="Cache d'identité illisible"):
        CharacterIdentity.load(cache_path)


def test_sha256_depends_on_file_content_not_filename(tmp_path: Path) -> None:
    first = tmp_path / "noemie.png"
    second = tmp_path / "copy.webp"
    first.write_bytes(b"same-content")
    second.write_bytes(b"same-content")

    assert sha256_file(first) == sha256_file(second)

