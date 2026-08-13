#!/usr/bin/env python3
"""Crée ou réutilise le cache d'identité générique d'un personnage."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pulid_app.config import ConfigError, load_config  # noqa: E402
from pulid_app.paths import configure_external_model_caches  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Encode une identité et réutilise son cache adressé par contenu."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=PROJECT_ROOT / "inputs" / "reference.webp",
        help="Image JPEG, PNG, WebP, BMP ou TIFF (défaut : inputs/reference.webp).",
    )
    parser.add_argument(
        "--character",
        default="noemie",
        help="Identifiant du personnage (défaut : noemie).",
    )
    parser.add_argument("--config", type=Path, help="Configuration YAML alternative.")
    parser.add_argument(
        "--face-index",
        type=int,
        help="Index explicite si plusieurs visages sont détectés (base 0).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore le cache existant et recalcule l'embedding.",
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

    from rich.console import Console

    from pulid_app.models.identity_encoder import IdentityEncoder, IdentityEncoderError

    console = Console()
    encoder = IdentityEncoder.from_config(config)
    try:
        cache_path = encoder.cache_path_for(
            args.image,
            identity_id=args.character,
            face_index=args.face_index,
        )
        cache_hit = cache_path.is_file() and not args.force
        identity = encoder.encode_image(
            args.image,
            identity_id=args.character,
            face_index=args.face_index,
            force_recompute=args.force,
        )
    except IdentityEncoderError as exc:
        console.print(f"[bold red]Échec du cache d'identité :[/] {exc}")
        return 1

    metadata = identity.metadata
    state = "réutilisé" if cache_hit else "créé"
    color = "cyan" if cache_hit else "green"
    console.print(f"Personnage : [bold]{identity.id}[/]")
    console.print(f"Image : {identity.source_images[0]}")
    console.print(f"Format : {metadata.get('source_format', 'inconnu')}")
    console.print(f"SHA-256 : {metadata.get('source_sha256', 'inconnu')}")
    console.print(f"Embedding shape : {identity.face_embedding.shape}")
    console.print(
        f"Embedding norme L2 : {float(metadata.get('embedding_norm_l2', 0.0)):.6f}"
    )
    console.print(f"Cache {state} : [{color}]{cache_path}[/]")
    console.print("[bold green]Identité prête.[/]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

