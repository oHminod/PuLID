#!/usr/bin/env python3
"""Valide le chargement et un calcul réel avec le GGUF d'embedding local."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pulid_app.config import ConfigError, load_config  # noqa: E402
from pulid_app.exceptions import PuLIDAppError  # noqa: E402
from pulid_app.models.text_embedding import TextEmbeddingService  # noqa: E402
from pulid_app.paths import configure_external_model_caches  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Charge BGE-M3 et calcule un embedding de contrôle."
    )
    parser.add_argument("--config", type=Path, help="Configuration YAML alternative.")
    parser.add_argument(
        "--device",
        choices=("mps", "cuda", "cpu"),
        default="cpu",
        help="Backend à tester (cpu par défaut).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Configuration invalide : {exc}", file=sys.stderr)
        return 2

    embedding_config = config.text_embedding
    if embedding_config is None:
        print(
            "La section text_embedding est absente de la configuration.",
            file=sys.stderr,
        )
        return 2

    configure_external_model_caches(config.models_root)
    service = TextEmbeddingService(embedding_config, device=args.device)
    try:
        response = service.create_embedding(
            model=embedding_config.model_id,
            input_value="test",
        )
    except (PuLIDAppError, ValueError) as exc:
        print(f"Échec du test BGE-M3 sur {args.device.upper()}.", file=sys.stderr)
        traceback.print_exception(exc)
        return 1
    finally:
        service.close()

    dimensions = len(response["data"][0]["embedding"])
    if dimensions != embedding_config.dimensions:
        print(
            "Dimension d'embedding inattendue : "
            f"{dimensions} au lieu de {embedding_config.dimensions}.",
            file=sys.stderr,
        )
        return 1

    print(
        f"BGE-M3 {args.device.upper()} OK : {dimensions} dimensions, "
        f"contexte {embedding_config.context_size} tokens."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
