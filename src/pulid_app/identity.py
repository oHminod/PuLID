"""Représentation générique et sérialisable d'une identité de personnage."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from numpy.typing import NDArray


IDENTITY_FORMAT_VERSION = 1


class IdentitySerializationError(RuntimeError):
    """Une identité ne peut pas être validée, enregistrée ou relue."""


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Calcule le SHA-256 du contenu sans charger le fichier entier en mémoire."""

    source = Path(path).expanduser().resolve(strict=False)
    if not source.is_file():
        raise IdentitySerializationError(f"Fichier source introuvable : {source}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as stream:
            while chunk := stream.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise IdentitySerializationError(
            f"Impossible de calculer le hash de {source} : {exc}"
        ) from exc
    return digest.hexdigest()


@dataclass
class CharacterIdentity:
    """Identité indépendante de PuLID et de tout pipeline de génération."""

    id: str
    source_images: list[str]
    face_embedding: NDArray[np.float32]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = self.id.strip()
        if not self.id:
            raise IdentitySerializationError("L'identifiant du personnage est vide.")
        self.source_images = [str(path) for path in self.source_images]
        if not self.source_images:
            raise IdentitySerializationError("Une identité doit référencer au moins une image.")
        embedding = np.asarray(self.face_embedding, dtype=np.float32).reshape(-1)
        if embedding.size == 0 or not np.isfinite(embedding).all():
            raise IdentitySerializationError(
                "L'embedding facial est vide ou contient des valeurs invalides."
            )
        self.face_embedding = np.ascontiguousarray(embedding)
        self.metadata = dict(self.metadata)

    def save(self, path: str | Path) -> Path:
        """Enregistre l'identité dans une archive NPZ atomique sans pickle."""

        destination = Path(path).expanduser().resolve(strict=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            metadata_json = json.dumps(
                self.metadata,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise IdentitySerializationError(
                f"Les métadonnées de l'identité {self.id!r} ne sont pas sérialisables : {exc}"
            ) from exc

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                np.savez_compressed(
                    temporary,
                    format_version=np.asarray(IDENTITY_FORMAT_VERSION, dtype=np.int64),
                    identity_id=np.asarray(self.id, dtype=np.str_),
                    source_images=np.asarray(self.source_images, dtype=np.str_),
                    face_embedding=self.face_embedding,
                    metadata_json=np.asarray(metadata_json, dtype=np.str_),
                )
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, destination)
        except (OSError, TypeError, ValueError) as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise IdentitySerializationError(
                f"Impossible d'enregistrer l'identité dans {destination} : {exc}"
            ) from exc
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "CharacterIdentity":
        """Relit et valide une archive créée par :meth:`save`."""

        source = Path(path).expanduser().resolve(strict=False)
        if not source.is_file():
            raise IdentitySerializationError(f"Cache d'identité introuvable : {source}")
        required = {
            "format_version",
            "identity_id",
            "source_images",
            "face_embedding",
            "metadata_json",
        }
        try:
            with np.load(source, allow_pickle=False) as archive:
                missing = required - set(archive.files)
                if missing:
                    raise IdentitySerializationError(
                        "Archive d'identité incomplète ; clés manquantes : "
                        f"{', '.join(sorted(missing))}."
                    )
                version = int(np.asarray(archive["format_version"]).item())
                if version != IDENTITY_FORMAT_VERSION:
                    raise IdentitySerializationError(
                        f"Version de cache non prise en charge : {version} "
                        f"(attendue : {IDENTITY_FORMAT_VERSION})."
                    )
                identity_id = str(np.asarray(archive["identity_id"]).item())
                source_images = [
                    str(value) for value in np.asarray(archive["source_images"]).tolist()
                ]
                embedding = np.array(
                    archive["face_embedding"], dtype=np.float32, copy=True
                )
                metadata_json = str(np.asarray(archive["metadata_json"]).item())
        except IdentitySerializationError:
            raise
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise IdentitySerializationError(
                f"Cache d'identité illisible : {source} ({exc})"
            ) from exc

        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError as exc:
            raise IdentitySerializationError(
                f"Métadonnées JSON invalides dans {source} : {exc}"
            ) from exc
        if not isinstance(metadata, dict):
            raise IdentitySerializationError(
                f"Les métadonnées de {source} doivent être un objet JSON."
            )
        return cls(
            id=identity_id,
            source_images=source_images,
            face_embedding=embedding,
            metadata=metadata,
        )
