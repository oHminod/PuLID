"""Acquisition atomique du code officiel PuLID hors du dépôt applicatif."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import BinaryIO, Callable
from urllib.request import urlopen
import zipfile


OFFICIAL_REPOSITORY = "https://github.com/ToTheBeginning/PuLID"
OFFICIAL_ARCHIVE_TEMPLATE = OFFICIAL_REPOSITORY + "/archive/{revision}.zip"
REQUIRED_SOURCE_FILES = (
    "LICENSE",
    "eva_clip/__init__.py",
    "eva_clip/factory.py",
    "pulid/attention_processor.py",
    "pulid/encoders_transformer.py",
)
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SOURCE_MARKER = ".pulid-source.json"


class PuLIDAssetError(RuntimeError):
    """Le runtime officiel PuLID est absent, incomplet ou non téléchargeable."""


@dataclass(frozen=True)
class OfficialSource:
    path: Path
    revision: str
    downloaded: bool


def validate_official_source(path: str | Path) -> tuple[Path, ...]:
    """Retourne les fichiers attendus absents du snapshot officiel."""

    source = Path(path).expanduser().resolve(strict=False)
    return tuple(source / relative for relative in REQUIRED_SOURCE_FILES if not (source / relative).is_file())


def _validate_revision(revision: str) -> str:
    normalized = revision.strip().casefold()
    if not REVISION_PATTERN.fullmatch(normalized):
        raise PuLIDAssetError(
            "La révision PuLID officielle doit être un SHA Git complet de 40 caractères : "
            f"{revision!r}."
        )
    return normalized


def _validate_source_marker(source: Path, revision: str) -> None:
    marker_path = source / SOURCE_MARKER
    if not marker_path.is_file():
        raise PuLIDAssetError(
            f"Le snapshot PuLID {source} n'a pas de marqueur de révision. "
            "Déplacez-le puis relancez `python scripts/prepare_pulid.py`."
        )
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PuLIDAssetError(f"Marqueur PuLID illisible : {marker_path} ({exc})") from exc
    recorded = marker.get("revision") if isinstance(marker, dict) else None
    if recorded != revision:
        raise PuLIDAssetError(
            f"Révision PuLID incohérente dans {marker_path} : "
            f"{recorded!r}, attendue {revision!r}."
        )


def _safe_archive_root(archive: zipfile.ZipFile) -> PurePosixPath:
    roots: set[str] = set()
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts:
            raise PuLIDAssetError(
                f"L'archive PuLID contient un chemin non sûr : {member.filename!r}."
            )
        if path.parts:
            roots.add(path.parts[0])
    if len(roots) != 1:
        raise PuLIDAssetError("L'archive PuLID ne contient pas une racine unique.")
    return PurePosixPath(next(iter(roots)))


def _download_archive(
    url: str,
    destination: Path,
    opener: Callable[[str], BinaryIO],
) -> None:
    try:
        with opener(url) as source, destination.open("wb") as output:
            shutil.copyfileobj(source, output)
    except (OSError, ValueError) as exc:
        raise PuLIDAssetError(
            f"Impossible de télécharger le code officiel PuLID depuis {url} : {exc}"
        ) from exc


def ensure_official_source(
    source_dir: str | Path,
    revision: str,
    *,
    allow_download: bool = True,
    opener: Callable[[str], BinaryIO] = urlopen,
) -> OfficialSource:
    """Valide ou télécharge atomiquement le snapshot officiel épinglé."""

    destination = Path(source_dir).expanduser().resolve(strict=False)
    normalized_revision = _validate_revision(revision)
    missing = validate_official_source(destination)
    if not missing:
        _validate_source_marker(destination, normalized_revision)
        return OfficialSource(destination, normalized_revision, downloaded=False)
    if destination.exists():
        raise PuLIDAssetError(
            f"Dossier PuLID officiel incomplet : {destination}. Fichiers manquants : "
            + ", ".join(str(path) for path in missing)
            + ". Supprimez ou déplacez ce dossier, puis relancez la préparation."
        )
    if not allow_download:
        raise PuLIDAssetError(
            f"Code officiel PuLID absent : {destination}. Exécutez "
            "`python scripts/prepare_pulid.py` avec l'accès réseau."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = OFFICIAL_ARCHIVE_TEMPLATE.format(revision=normalized_revision)
    try:
        with tempfile.TemporaryDirectory(
            prefix=".pulid-source-", dir=destination.parent
        ) as temporary_name:
            temporary = Path(temporary_name)
            archive_path = temporary / "pulid.zip"
            _download_archive(url, archive_path, opener)
            try:
                with zipfile.ZipFile(archive_path) as archive:
                    archive_root = _safe_archive_root(archive)
                    archive.extractall(temporary / "extracted")
            except (OSError, zipfile.BadZipFile) as exc:
                raise PuLIDAssetError(f"Archive PuLID invalide : {exc}") from exc

            extracted = temporary / "extracted" / Path(*archive_root.parts)
            extracted_missing = validate_official_source(extracted)
            if extracted_missing:
                raise PuLIDAssetError(
                    "Snapshot PuLID officiel incomplet ; fichiers manquants : "
                    + ", ".join(str(path) for path in extracted_missing)
                )
            marker = {
                "repository": OFFICIAL_REPOSITORY,
                "revision": normalized_revision,
            }
            (extracted / SOURCE_MARKER).write_text(
                json.dumps(marker, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(extracted, destination)
    except PuLIDAssetError:
        raise
    except OSError as exc:
        raise PuLIDAssetError(
            f"Impossible d'installer le code officiel PuLID dans {destination} : {exc}"
        ) from exc

    return OfficialSource(destination, normalized_revision, downloaded=True)
