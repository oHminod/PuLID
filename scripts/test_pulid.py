#!/usr/bin/env python3
"""Exécute le premier pipeline complet PuLID v1.1 + SDXL local."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pulid_app.config import ConfigError, load_config  # noqa: E402
from pulid_app.models.sdxl import (  # noqa: E402
    SAMPLING_METHOD_SPECS,
    SIGMA_SCHEDULE_ARGUMENTS,
)
from pulid_app.paths import (  # noqa: E402
    configure_external_model_caches,
    resolve_sdxl_checkpoint,
)


DEFAULT_PROMPT = "cinematic portrait of a woman standing in Tokyo at night"
DEFAULT_NEGATIVE_PROMPT = (
    "flaws in the eyes, flaws in the face, low quality, worst quality, "
    "artifacts, text, watermark, deformed, mutated, disfigured, blurry"
)
DEFAULT_REFERENCE = PROJECT_ROOT / "inputs" / "noemie.webp"
SAMPLING_METHODS = tuple(SAMPLING_METHOD_SPECS)
SIGMA_SCHEDULES = tuple(SIGMA_SCHEDULE_ARGUMENTS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Génère une image SDXL conditionnée par PuLID v1.1."
    )
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--guidance-scale",
        "--cfg",
        dest="guidance_scale",
        type=float,
        default=7.0,
        help="CFG numérique (7.0 par défaut).",
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--face-index", type=int)
    parser.add_argument("--config", type=Path, help="Configuration YAML alternative.")
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"))
    parser.add_argument("--dtype", choices=("float16", "float32"))
    parser.add_argument(
        "--model",
        help=(
            "Nom d'un checkpoint SDXL du dossier checkpoints configuré, "
            "sans l'extension .safetensors."
        ),
    )
    parser.add_argument(
        "--method",
        choices=SAMPLING_METHODS,
        help="Méthode de sampling personnalisée ; scheduler du modèle par défaut.",
    )
    parser.add_argument(
        "--sigmas",
        choices=SIGMA_SCHEDULES,
        default="normal",
        help="Courbe de sigmas, sélectionnée indépendamment de la méthode.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        selected_checkpoint = resolve_sdxl_checkpoint(config, args.model)
    except (ConfigError, ValueError) as exc:
        print(f"Configuration invalide : {exc}", file=sys.stderr)
        return 2

    if args.model is not None and not selected_checkpoint.is_file():
        print(
            "Modèle SDXL demandé introuvable : "
            f"{selected_checkpoint}. Placez le fichier .safetensors dans "
            f"{selected_checkpoint.parent} ou omettez --model pour utiliser "
            "le modèle configuré.",
            file=sys.stderr,
        )
        return 2
    config = replace(config, sdxl=replace(config.sdxl, checkpoint=selected_checkpoint))

    reference = args.reference.expanduser().resolve(strict=False)
    configure_external_model_caches(config.models_root)

    from rich.console import Console

    from pulid_app.pipeline.generator import ImageGenerator, ImageGeneratorError

    console = Console()
    generator: ImageGenerator | None = None
    try:
        generator = ImageGenerator(
            config,
            device=args.device,
            dtype_name=args.dtype,
            allow_downloads=False,
        )
        console.print(f"Référence : [cyan]{reference}[/]")
        console.print(f"Modèle SDXL : [cyan]{config.sdxl.checkpoint.name}[/]")
        console.print(f"Device : [bold]{generator.device}[/]")
        console.print("1/2 — Encodage lazy et cache de l'identité…")
        identity = generator.encode_identity(
            reference,
            identity_id=reference.stem,
            face_index=args.face_index,
        )
        cache_state = "réutilisé" if identity.cache_hit else "créé"
        console.print(
            "[green]✓ Identité prête[/] — "
            f"cache={cache_state}, durée={identity.duration_seconds:.2f} s"
        )
        console.print(f"Sampling : [cyan]{args.method or 'scheduler du modèle'}[/]")
        console.print(f"Sigmas : [cyan]{args.sigmas}[/]")
        console.print("2/2 — Génération et sauvegarde automatiques…")
        generated = generator.generate(
            prompt=args.prompt,
            identity=identity,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            width=args.width,
            height=args.height,
            steps=args.steps,
            identity_strength=args.strength,
            guidance_scale=args.guidance_scale,
            sampling_method=args.method,
            sigma_schedule=args.sigmas,
        )
    except ImageGeneratorError as exc:
        console.print(f"[bold red]Échec du pipeline PuLID :[/] {exc}")
        return 1
    finally:
        if generator is not None:
            try:
                generator.close()
            except ImageGeneratorError as exc:
                console.print(f"[yellow]Avertissement de nettoyage : {exc}[/]")

    console.print(f"[bold green]Image PuLID générée :[/] {generated.png_path}")
    console.print(f"[bold green]Métadonnées :[/] {generated.json_path}")
    console.print(
        f"Durée totale : {generated.metadata['total_duration_seconds']:.2f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
