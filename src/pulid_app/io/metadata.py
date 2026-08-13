"""Écriture déterministe des métadonnées JSON locales."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


def save_json_metadata(path: str | Path, metadata: Mapping[str, Any]) -> Path:
    """Écrit atomiquement un objet JSON et retourne son chemin absolu."""

    destination = Path(path).expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            json.dump(metadata, temporary, ensure_ascii=False, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, destination)
    except (OSError, TypeError, ValueError) as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Impossible d'écrire les métadonnées dans {destination} : {exc}"
        ) from exc
    return destination

