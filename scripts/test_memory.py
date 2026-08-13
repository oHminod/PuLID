#!/usr/bin/env python3
"""Valide une allocation puis une libération mémoire sur le backend local."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pulid_app.config import ConfigError, load_config  # noqa: E402
from pulid_app.paths import configure_external_model_caches  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Teste le nettoyage mémoire PyTorch sans charger de modèle."
    )
    parser.add_argument("--config", type=Path, help="Configuration YAML alternative.")
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"))
    parser.add_argument(
        "--size",
        type=int,
        default=4096,
        help="Côté de la matrice float32 allouée (4096 = 64 Mio).",
    )
    return parser


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "indisponible"
    return f"{value / (1024**2):.1f} Mio"


def _synchronize(torch_module: Any, device: str) -> None:
    backend = getattr(torch_module, device, None)
    synchronize = getattr(backend, "synchronize", None)
    if callable(synchronize):
        synchronize()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.size <= 0:
        print("--size doit être strictement positif.", file=sys.stderr)
        return 2

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Configuration invalide : {exc}", file=sys.stderr)
        return 2

    configure_external_model_caches(config.models_root)
    try:
        import torch
        from rich.console import Console

        from pulid_app.device import get_best_device
        from pulid_app.pipeline.memory import MemoryManager, MemoryManagerError
    except ImportError as exc:
        print(
            "Dépendance absente. Activez .venv puis exécutez "
            "`uv pip install -e '.[dev]'`.",
            file=sys.stderr,
        )
        return 2

    console = Console()
    device = args.device or get_best_device()
    manager = MemoryManager(
        config.models_root,
        device=device,
        torch_module=torch,
    )

    try:
        before = manager.snapshot()
        tensor = torch.ones((args.size, args.size), dtype=torch.float32, device=device)
        tensor.mul_(2.0)
        _synchronize(torch, device)
        during = manager.snapshot()
        del tensor
        manager.cleanup(force=True)
        _synchronize(torch, device)
        after = manager.snapshot()
    except (MemoryManagerError, RuntimeError) as exc:
        console.print(f"[bold red]Échec du test mémoire :[/] {exc}")
        return 1

    console.print(f"Device : [bold]{device}[/]")
    console.print(
        "Mémoire allouée : "
        f"avant={_format_bytes(before.allocated_bytes)}, "
        f"pendant={_format_bytes(during.allocated_bytes)}, "
        f"après={_format_bytes(after.allocated_bytes)}"
    )
    console.print(
        "Mémoire réservée/pilote : "
        f"avant={_format_bytes(before.reserved_bytes)}, "
        f"pendant={_format_bytes(during.reserved_bytes)}, "
        f"après={_format_bytes(after.reserved_bytes)}"
    )

    if before.allocated_bytes is not None and during.allocated_bytes is not None:
        if during.allocated_bytes <= before.allocated_bytes:
            console.print("[bold red]L'allocation n'est pas visible dans les compteurs.[/]")
            return 1
        if after.allocated_bytes is None or after.allocated_bytes >= during.allocated_bytes:
            console.print("[bold red]La libération n'est pas visible dans les compteurs.[/]")
            return 1

    console.print("[bold green]Allocation et libération mémoire validées.[/]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
