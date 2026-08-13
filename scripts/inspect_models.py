#!/usr/bin/env python3
"""Inspecte les modèles locaux sans charger les bibliothèques de génération."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pulid_app.cli import inspect_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(inspect_main())
