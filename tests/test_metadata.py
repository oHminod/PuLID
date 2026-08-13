from __future__ import annotations

import json
from pathlib import Path

from pulid_app.io.metadata import save_json_metadata


def test_save_json_metadata_creates_valid_json(tmp_path: Path) -> None:
    destination = tmp_path / "cache" / "identity" / "face.json"

    result = save_json_metadata(destination, {"shape": [512], "score": 0.99})

    assert result == destination
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "shape": [512],
        "score": 0.99,
    }

