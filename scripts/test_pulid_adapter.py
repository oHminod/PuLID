#!/usr/bin/env python3
"""Valide le chargement PuLID et, en option, l'identité de référence."""

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
        description="Charge PuLID v1.1 sans lancer de génération SDXL."
    )
    parser.add_argument("--config", type=Path, help="Configuration YAML alternative.")
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"))
    parser.add_argument("--dtype", choices=("float16", "float32"))
    parser.add_argument(
        "--reference",
        type=Path,
        help="Prépare aussi les traits PuLID de cette image (JPEG/PNG/WebP/BMP/TIFF).",
    )
    parser.add_argument("--face-index", type=int)
    parser.add_argument(
        "--apply-sdxl",
        action="store_true",
        help="Charge aussi le checkpoint SDXL local et injecte les processeurs PuLID.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Refuse tout téléchargement de runtime ou poids auxiliaires.",
    )
    return parser


def _shape(value: object) -> tuple[int, ...] | str:
    shape = getattr(value, "shape", None)
    return tuple(int(item) for item in shape) if shape is not None else "inconnue"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Configuration invalide : {exc}", file=sys.stderr)
        return 2
    configure_external_model_caches(config.models_root)

    from rich.console import Console

    from pulid_app.device import get_best_device
    from pulid_app.models.pulid_adapter import PuLIDAdapter, PuLIDError

    console = Console()
    device = args.device or get_best_device()
    adapter = PuLIDAdapter.from_config(
        config,
        device=device,
        dtype_name=args.dtype,
        allow_downloads=not args.offline,
    )
    sdxl = None
    try:
        adapter.load()
        console.print(f"[green]✓ PuLID v1.1 chargé[/] sur [bold]{device}[/]")
        console.print(f"Checkpoint : [cyan]{adapter.checkpoint_path}[/]")
        console.print(f"Snapshot officiel : [cyan]{adapter.source_dir}[/]")
        console.print(
            f"Tenseurs d'attention : [bold]{len(adapter._attention_state or {})}[/]"
        )
        if args.apply_sdxl:
            from pulid_app.models.sdxl import SDXLModel

            sdxl = SDXLModel.from_config(
                config,
                device=device,
                dtype_name=args.dtype,
            ).load()
            assert sdxl.pipeline is not None
            adapter.apply(sdxl.pipeline)
            injected = sum(
                type(processor).__name__.startswith("IDAttnProcessor")
                for processor in sdxl.pipeline.unet.attn_processors.values()
            )
            console.print(
                f"[green]✓ PuLID injecté dans SDXL[/] — "
                f"cross-attentions={injected}"
            )
        if args.reference is not None:
            features = adapter.prepare_identity(
                args.reference,
                face_index=args.face_index,
            )
            adapter.set_identity(features, strength=0.8)
            console.print(
                "[green]✓ Traits d'identité préparés[/] — "
                f"conditionnel={_shape(features.conditional)}, "
                f"inconditionnel={_shape(features.unconditional)}"
            )
    except (PuLIDError, RuntimeError) as exc:
        console.print(f"[bold red]Échec PuLID :[/] {exc}")
        return 1
    finally:
        try:
            adapter.close()
        except PuLIDError as exc:
            console.print(f"[yellow]Avertissement de libération : {exc}[/]")
        if sdxl is not None:
            try:
                sdxl.close()
            except RuntimeError as exc:
                console.print(f"[yellow]Avertissement SDXL : {exc}[/]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
