from __future__ import annotations

from io import BytesIO
from pathlib import Path
import json
import zipfile

import pytest

from pulid_app.models.pulid_assets import (
    OFFICIAL_ARCHIVE_TEMPLATE,
    PuLIDAssetError,
    REQUIRED_SOURCE_FILES,
    ensure_official_source,
)


REVISION = "0123456789abcdef0123456789abcdef01234567"


def _archive(files: tuple[str, ...] = REQUIRED_SOURCE_FILES) -> bytes:
    stream = BytesIO()
    with zipfile.ZipFile(stream, mode="w") as archive:
        for relative in files:
            archive.writestr(f"PuLID-{REVISION}/{relative}", "# test\n")
    return stream.getvalue()


def test_ensure_official_source_downloads_atomically(tmp_path: Path) -> None:
    calls: list[str] = []

    def opener(url: str) -> BytesIO:
        calls.append(url)
        return BytesIO(_archive())

    result = ensure_official_source(
        tmp_path / "models" / "sources" / "PuLID",
        REVISION,
        opener=opener,
    )

    assert result.downloaded is True
    assert result.path.is_dir()
    assert (result.path / "pulid/encoders_transformer.py").is_file()
    assert (result.path / ".pulid-source.json").is_file()
    assert calls == [OFFICIAL_ARCHIVE_TEMPLATE.format(revision=REVISION)]


def test_existing_official_source_is_not_downloaded_again(tmp_path: Path) -> None:
    source = tmp_path / "PuLID"
    for relative in REQUIRED_SOURCE_FILES:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (source / ".pulid-source.json").write_text(
        json.dumps({"revision": REVISION}), encoding="utf-8"
    )

    result = ensure_official_source(
        source,
        REVISION,
        opener=lambda _url: pytest.fail("aucun téléchargement attendu"),
    )

    assert result.downloaded is False


def test_existing_source_revision_must_match(tmp_path: Path) -> None:
    source = tmp_path / "PuLID"
    for relative in REQUIRED_SOURCE_FILES:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (source / ".pulid-source.json").write_text(
        json.dumps({"revision": "f" * 40}), encoding="utf-8"
    )

    with pytest.raises(PuLIDAssetError, match="Révision PuLID incohérente"):
        ensure_official_source(source, REVISION)


def test_check_only_reports_missing_source(tmp_path: Path) -> None:
    with pytest.raises(PuLIDAssetError, match="prepare_pulid.py"):
        ensure_official_source(tmp_path / "PuLID", REVISION, allow_download=False)


def test_incomplete_existing_source_is_never_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "PuLID"
    source.mkdir()
    user_file = source / "user.txt"
    user_file.write_text("preserve", encoding="utf-8")

    with pytest.raises(PuLIDAssetError, match="incomplet"):
        ensure_official_source(source, REVISION)

    assert user_file.read_text(encoding="utf-8") == "preserve"


def test_revision_must_be_full_git_sha(tmp_path: Path) -> None:
    with pytest.raises(PuLIDAssetError, match="SHA Git complet"):
        ensure_official_source(tmp_path / "PuLID", "main")
