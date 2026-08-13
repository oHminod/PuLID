#!/usr/bin/env python3
"""Prépare le snapshot officiel requis par l'adaptateur PuLID."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pulid_app.config import ConfigError, load_config  # noqa: E402
from pulid_app.models.pulid_assets import (  # noqa: E402
    PuLIDAssetError,
    ensure_official_source,
)
from pulid_app.paths import configure_external_model_caches  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Télécharge ou valide le code officiel PuLID épinglé."
    )
    parser.add_argument("--config", type=Path, help="Configuration YAML alternative.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Valide le snapshot présent sans accès réseau.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Configuration invalide : {exc}", file=sys.stderr)
        return 2
    configure_external_model_caches(config.models_root)

    if config.pulid.source_dir is None:
        print("pulid.source_dir est absent de la configuration.", file=sys.stderr)
        return 2
    try:
        source = ensure_official_source(
            config.pulid.source_dir,
            config.pulid.revision,
            allow_download=not args.check_only,
        )
    except PuLIDAssetError as exc:
        print(f"Préparation PuLID impossible : {exc}", file=sys.stderr)
        return 1

    action = "téléchargé" if source.downloaded else "déjà présent"
    print(f"Snapshot officiel PuLID {action} : {source.path}")
    print(f"Révision : {source.revision}")
    print(
        "EVA-CLIP et les poids FaceXLib seront téléchargés automatiquement dans "
        f"{config.models_root} au premier prepare_identity(); aucun doublon local n'est créé."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
