#!/usr/bin/env python3
"""Valide AntelopeV2 sur une image de référence locale."""

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
        description="Détecte une face et extrait son embedding avec AntelopeV2."
    )
    parser.add_argument("--image", type=Path, required=True, help="Image JPG, PNG ou WebP.")
    parser.add_argument("--config", type=Path, help="Configuration YAML alternative.")
    parser.add_argument(
        "--face-index",
        type=int,
        help="Index explicite si plusieurs visages sont détectés (base 0).",
    )
    parser.add_argument(
        "--save-metadata",
        action="store_true",
        help="Écrit un résumé JSON dans cache/identity/.",
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

    # InsightFace et ONNX Runtime ne sont importés qu'après la redirection des caches.
    from rich.console import Console

    from pulid_app.io.metadata import save_json_metadata
    from pulid_app.models.identity_encoder import (
        IdentityEncoder,
        IdentityEncoderError,
    )

    console = Console()
    encoder = IdentityEncoder.from_config(config)
    try:
        encoder.load()
        result = encoder.encode(args.image, face_index=args.face_index)
    except IdentityEncoderError as exc:
        console.print(f"[bold red]Échec InsightFace :[/] {exc}")
        return 1

    bbox = result.detection.bbox
    console.print(f"Modèles AntelopeV2 : [cyan]{encoder.model_dir}[/]")
    console.print(f"ONNX provider(s) : {', '.join(encoder.providers)}")
    console.print(f"Nombre de visages : [bold]{result.face_count}[/]")
    console.print(f"Visage sélectionné : {result.face_index}")
    console.print(
        "Bounding box : "
        f"[{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}]"
    )
    console.print(f"Score : {result.detection.score:.6f}")
    console.print(f"Embedding shape : {result.embedding.shape}")
    console.print(f"Embedding norme L2 : {result.norm:.6f}")

    if args.save_metadata:
        image_path = args.image.expanduser().resolve(strict=False)
        metadata_path = config.identity_cache_dir / f"{image_path.stem}_insightface.json"
        metadata = {
            "image": str(image_path),
            "model_dir": str(encoder.model_dir),
            "providers": list(encoder.providers),
            "face_count": result.face_count,
            "selected_face_index": result.face_index,
            "bounding_box": list(result.detection.bbox),
            "detection_score": result.detection.score,
            "embedding_shape": list(result.embedding.shape),
            "embedding_norm_l2": result.norm,
        }
        try:
            saved_path = save_json_metadata(metadata_path, metadata)
        except RuntimeError as exc:
            console.print(f"[bold red]Métadonnées non enregistrées :[/] {exc}")
            return 1
        console.print(f"Métadonnées : [green]{saved_path}[/]")

    console.print("[bold green]Validation InsightFace réussie.[/]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

