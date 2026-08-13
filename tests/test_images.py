from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from pulid_app.io.images import save_image_with_metadata, save_png


def test_save_png_round_trip(tmp_path: Path) -> None:
    destination = tmp_path / "output.png"
    image = Image.new("RGB", (16, 12), (10, 20, 30))

    saved = save_png(destination, image)

    with Image.open(saved) as reloaded:
        assert reloaded.format == "PNG"
        assert reloaded.size == (16, 12)


def test_save_image_with_adjacent_metadata(tmp_path: Path) -> None:
    image = Image.new("RGB", (8, 8))

    png_path, json_path = save_image_with_metadata(
        image, {"seed": 42}, tmp_path, prefix="sdxl_test"
    )

    assert png_path.stem == json_path.stem
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"seed": 42}
