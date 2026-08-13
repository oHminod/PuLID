#!/usr/bin/env python3
"""Exécute le premier pipeline complet PuLID v1.1 + SDXL local."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pulid_app.config import AppConfig, ConfigError, load_config  # noqa: E402
from pulid_app.paths import configure_external_model_caches  # noqa: E402


DEFAULT_PROMPT = "cinematic portrait of a woman standing in Tokyo at night"
DEFAULT_NEGATIVE_PROMPT = (
    "flaws in the eyes, flaws in the face, low quality, worst quality, "
    "artifacts, text, watermark, deformed, mutated, disfigured, blurry"
)
DEFAULT_REFERENCE = PROJECT_ROOT / "inputs" / "noemie.webp"


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
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--face-index", type=int)
    parser.add_argument("--config", type=Path, help="Configuration YAML alternative.")
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"))
    parser.add_argument("--dtype", choices=("float16", "float32"))
    return parser


def build_metadata(
    *,
    args: argparse.Namespace,
    config: AppConfig,
    reference: Path,
    result: Any,
    identity_duration_seconds: float,
    total_duration_seconds: float,
) -> dict[str, Any]:
    """Construit le manifeste adjacent sans dépendre des modèles lourds."""

    return {
        "reference_image": str(reference),
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": result.seed,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "identity_strength": args.strength,
        "width": args.width,
        "height": args.height,
        "sdxl_checkpoint": str(config.sdxl.checkpoint),
        "pulid_checkpoint": str(config.pulid.checkpoint),
        "pulid_source_dir": str(config.pulid.source_dir),
        "pulid_revision": config.pulid.revision,
        "vae": "integrated",
        "device": result.device,
        "dtype": result.dtype,
        "dtype_fallback_used": result.dtype_fallback_used,
        "identity_duration_seconds": identity_duration_seconds,
        "generation_duration_seconds": result.duration_seconds,
        "total_duration_seconds": total_duration_seconds,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Configuration invalide : {exc}", file=sys.stderr)
        return 2

    reference = args.reference.expanduser().resolve(strict=False)
    configure_external_model_caches(config.models_root)

    from rich.console import Console

    from pulid_app.device import get_best_device
    from pulid_app.io.images import save_image_with_metadata
    from pulid_app.models.pulid_adapter import PuLIDAdapter, PuLIDError
    from pulid_app.models.sdxl import SDXLError, SDXLModel

    console = Console()
    device = args.device or get_best_device()
    adapter: PuLIDAdapter | None = None
    sdxl: SDXLModel | None = None
    started = time.monotonic()
    try:
        adapter = PuLIDAdapter.from_config(
            config,
            device=device,
            dtype_name=args.dtype,
            # La phase 8 doit être préparée explicitement ; une génération ne
            # déclenche jamais un téléchargement inattendu.
            allow_downloads=False,
        )
        console.print(f"Référence : [cyan]{reference}[/]")
        console.print(f"Device : [bold]{device}[/]")
        console.print("1/4 — Préparation InsightFace, FaceXLib, EVA-CLIP et IDFormer…")
        identity_started = time.monotonic()
        identity_features = adapter.prepare_identity(
            reference,
            face_index=args.face_index,
        )
        identity_duration = time.monotonic() - identity_started
        adapter.set_identity(identity_features, strength=args.strength)
        console.print(
            "[green]✓ Identité prête[/] — "
            f"tokens={tuple(identity_features.conditional.shape)}, "
            f"durée={identity_duration:.2f} s"
        )

        console.print("2/4 — Chargement hors ligne du checkpoint SDXL…")
        sdxl = SDXLModel.from_config(
            config,
            device=device,
            dtype_name=args.dtype,
            # Un rechargement automatique perdrait les processeurs injectés.
            allow_dtype_fallback=False,
        ).load()
        assert sdxl.pipeline is not None

        console.print("3/4 — Injection de PuLID dans les cross-attentions SDXL…")
        adapter.apply(sdxl.pipeline)
        cross_attention_kwargs = adapter.cross_attention_kwargs(
            classifier_free_guidance=args.guidance_scale > 1.0
        )

        console.print("4/4 — Dénoyautage SDXL et décodage du VAE intégré…")
        result = sdxl.generate(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            steps=args.steps,
            width=args.width,
            height=args.height,
            guidance_scale=args.guidance_scale,
            cross_attention_kwargs=cross_attention_kwargs,
        )
        total_duration = time.monotonic() - started
        metadata = build_metadata(
            args=args,
            config=config,
            reference=reference,
            result=result,
            identity_duration_seconds=identity_duration,
            total_duration_seconds=total_duration,
        )
        png_path, json_path = save_image_with_metadata(
            result.image,
            metadata,
            config.outputs_dir,
            prefix="pulid",
        )
    except (PuLIDError, SDXLError, RuntimeError, ValueError) as exc:
        console.print(f"[bold red]Échec du pipeline PuLID :[/] {exc}")
        return 1
    finally:
        if sdxl is not None:
            try:
                sdxl.close()
            except SDXLError as exc:
                console.print(f"[yellow]Avertissement SDXL : {exc}[/]")
        if adapter is not None:
            try:
                adapter.close()
            except PuLIDError as exc:
                console.print(f"[yellow]Avertissement PuLID : {exc}[/]")

    console.print(f"[bold green]Image PuLID générée :[/] {png_path}")
    console.print(f"[bold green]Métadonnées :[/] {json_path}")
    console.print(f"Durée totale : {total_duration:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
