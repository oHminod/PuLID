#!/usr/bin/env python3
"""Test d'intégration minimal du backend MPS."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pulid_app.config import ConfigError, load_config  # noqa: E402
from pulid_app.paths import configure_external_model_caches  # noqa: E402


def main() -> int:
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Configuration invalide : {exc}", file=sys.stderr)
        return 2

    configure_external_model_caches(config.models_root)

    # L'import intervient seulement après la redirection de tous les caches.
    try:
        import torch
    except ImportError as exc:
        print(
            "PyTorch n'est pas installé. Activez .venv puis exécutez "
            "`uv pip install -e '.[dev]'`.",
            file=sys.stderr,
        )
        return 2

    from rich.console import Console

    from pulid_app.device import get_best_device, print_device_report

    console = Console()
    report = print_device_report(console)
    if get_best_device() != "mps":
        if not report.mps_built:
            reason = "cette installation de PyTorch n'a pas été compilée avec MPS"
        else:
            reason = "MPS n'est pas disponible sur cette machine"
        console.print(f"[bold red]Échec du test MPS :[/] {reason}.")
        return 1

    try:
        with torch.inference_mode():
            left = torch.arange(1, 10, dtype=torch.float32, device="mps").reshape(3, 3)
            right = torch.eye(3, dtype=torch.float32, device="mps")
            result = left @ right
            torch.mps.synchronize()
            expected = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3)
            valid = torch.allclose(result.cpu(), expected)
    except RuntimeError as exc:
        console.print(f"[bold red]Opération MPS impossible :[/] {exc}")
        return 1

    if not valid:
        console.print("[bold red]Le résultat tensoriel MPS est incorrect.[/]")
        return 1

    console.print(
        "[bold green]Test MPS réussi :[/] multiplication matricielle 3×3 correcte."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

