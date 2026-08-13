#!/usr/bin/env python3
"""Valide le checkpoint SDXL local par une génération image + JSON."""

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


DEFAULT_PROMPT = "portrait photo of a woman, tropical beach, studio lighting"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Génère une image de test SDXL locale.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--config", type=Path, help="Configuration YAML alternative.")
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"))
    parser.add_argument("--dtype", choices=("float16", "float32"))
    parser.add_argument(
        "--no-dtype-fallback",
        action="store_true",
        help="Désactive le second essai float32 en cas d'erreur MPS float16.",
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

    from pulid_app.io.images import save_image_with_metadata
    from pulid_app.models.sdxl import SDXLError, SDXLModel

    console = Console()
    try:
        model = SDXLModel.from_config(
            config,
            device=args.device,
            dtype_name=args.dtype,
            allow_dtype_fallback=not args.no_dtype_fallback,
        )
        console.print(f"Checkpoint : [cyan]{model.checkpoint_path}[/]")
        console.print(f"Configuration locale : [cyan]{model.config_dir}[/]")
        console.print(f"Device : [bold]{model.device}[/]")
        console.print("Chargement SDXL hors ligne…")
        model.load()
        console.print(f"Dtype chargé : [bold]{model.active_dtype_name}[/]")
        result = model.generate(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            steps=args.steps,
            width=args.width,
            height=args.height,
            guidance_scale=args.guidance_scale,
        )
        metadata = {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "seed": args.seed,
            "steps": args.steps,
            "width": args.width,
            "height": args.height,
            "guidance_scale": args.guidance_scale,
            "checkpoint": str(model.checkpoint_path),
            "config_dir": str(model.config_dir),
            "vae": "integrated",
            "device": result.device,
            "dtype": result.dtype,
            "dtype_fallback_used": result.dtype_fallback_used,
            "duration_seconds": result.duration_seconds,
        }
        png_path, json_path = save_image_with_metadata(
            result.image,
            metadata,
            config.outputs_dir,
            prefix="sdxl_test",
        )
    except (SDXLError, RuntimeError) as exc:
        console.print(f"[bold red]Échec SDXL :[/] {exc}")
        return 1

    console.print(f"[bold green]Image générée :[/] {png_path}")
    console.print(f"[bold green]Métadonnées :[/] {json_path}")
    console.print(f"Durée d'inférence : {result.duration_seconds:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
