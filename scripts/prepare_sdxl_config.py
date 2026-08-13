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
from pulid_app.paths import configure_external_model_caches  # noqa: E402


DEFAULT_CONFIG_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
ALLOWED_CONFIG_PATTERNS = (
    "*.json",
    "*.txt",
    "*.model",
    "**/*.json",
    "**/*.txt",
    "**/*.model",
)
FORBIDDEN_WEIGHT_SUFFIXES = frozenset(
    {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}
)


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
    required = (
        path / "model_index.json",
        path / "scheduler" / "scheduler_config.json",
        path / "text_encoder" / "config.json",
        path / "text_encoder_2" / "config.json",
        path / "tokenizer" / "tokenizer_config.json",
        path / "tokenizer" / "vocab.json",
        path / "tokenizer" / "merges.txt",
        path / "tokenizer_2" / "tokenizer_config.json",
        path / "tokenizer_2" / "vocab.json",
        path / "tokenizer_2" / "merges.txt",
        path / "unet" / "config.json",
        path / "vae" / "config.json",
    )
    missing = [file for file in required if not file.is_file()]
    if missing:
        raise RuntimeError(
            "Configuration SDXL incomplète ; fichiers manquants : "
            + ", ".join(str(file) for file in missing)
        )

    all_files = [file for file in path.rglob("*") if file.is_file()]
    forbidden = [
        file for file in all_files if file.suffix.casefold() in FORBIDDEN_WEIGHT_SUFFIXES
    ]
    if forbidden:
        raise RuntimeError(
            "Des poids interdits ont été trouvés dans le dossier de configuration : "
            + ", ".join(str(file) for file in forbidden)
        )
    config_files = [
        file
        for file in all_files
        if ".cache" not in file.relative_to(path).parts
    ]
    return len(config_files), sum(file.stat().st_size for file in config_files)


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
