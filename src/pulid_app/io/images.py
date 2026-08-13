"""Écriture atomique des images générées et de leurs métadonnées."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from PIL import Image

from pulid_app.io.metadata import save_json_metadata


def save_png(path: str | Path, image: Image.Image) -> Path:
    destination = Path(path).expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}.",
            suffix=".png",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            image.save(temporary, format="PNG")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    except (OSError, ValueError) as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Impossible d'écrire l'image PNG {destination} : {exc}") from exc
    return destination


def save_image_with_metadata(
    image: Image.Image,
    metadata: Mapping[str, Any],
    output_dir: str | Path,
    *,
    prefix: str = "sdxl_test",
) -> tuple[Path, Path]:
    """Écrit `<prefix>_<timestamp>.png` et son JSON adjacent."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    base = Path(output_dir).expanduser().resolve(strict=False) / f"{prefix}_{timestamp}"
    png_path = save_png(base.with_suffix(".png"), image)
    json_path = save_json_metadata(base.with_suffix(".json"), metadata)
    return png_path, json_path
