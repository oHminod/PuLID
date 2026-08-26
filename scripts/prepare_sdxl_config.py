#!/usr/bin/env python3
"""Télécharge explicitement les seules configurations requises par SDXL."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pulid_app.config import ConfigError, load_config  # noqa: E402
from pulid_app.installer import (  # noqa: E402
    SDXL_CONFIG_PATTERNS as ALLOWED_CONFIG_PATTERNS,
    SDXL_CONFIG_REPOSITORY as DEFAULT_CONFIG_REPO,
    validate_sdxl_config_tree,
)
from pulid_app.paths import configure_external_model_caches  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prépare les configs/tokenizers SDXL locaux sans télécharger de poids."
        )
    )
    parser.add_argument("--config", type=Path, help="Configuration YAML alternative.")
    parser.add_argument(
        "--repo",
        default=DEFAULT_CONFIG_REPO,
        help=f"Dépôt de configuration source (défaut : {DEFAULT_CONFIG_REPO}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Actualise les fichiers de configuration déjà présents.",
    )
    return parser


def _validate_config_tree(path: Path) -> tuple[int, int]:
    """Alias historique conservé pour les scripts et tests existants."""

    return validate_sdxl_config_tree(path)


def _format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("o", "Kio", "Mio", "Gio"):
        if amount < 1024 or unit == "Gio":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} Gio"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Configuration invalide : {exc}", file=sys.stderr)
        return 2
    if config.sdxl.config_dir is None:
        print("La clé sdxl.config_dir est absente de la configuration.", file=sys.stderr)
        return 2

    configure_external_model_caches(config.models_root)
    from huggingface_hub import snapshot_download
    from rich.console import Console

    console = Console()
    console.print(f"Source : [cyan]{args.repo}[/]")
    console.print(f"Destination : [cyan]{config.sdxl.config_dir}[/]")
    try:
        snapshot_download(
            repo_id=args.repo,
            cache_dir=config.models_root / "huggingface" / "hub",
            local_dir=config.sdxl.config_dir,
            allow_patterns=list(ALLOWED_CONFIG_PATTERNS),
            force_download=args.force,
        )
        file_count, total_size = _validate_config_tree(config.sdxl.config_dir)
    except Exception as exc:
        console.print(f"[bold red]Préparation SDXL impossible :[/] {exc}")
        return 1

    console.print(
        f"[bold green]Configuration SDXL prête :[/] {file_count} fichiers, "
        f"{_format_bytes(total_size)}, aucun poids téléchargé."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
