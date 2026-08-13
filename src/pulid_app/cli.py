"""Interface de ligne de commande de la phase de bootstrap."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

from rich.console import Console
from rich.table import Table

from pulid_app.config import AppConfig, ConfigError, load_config
from pulid_app.paths import (
    cache_env_violations,
    configure_external_model_caches,
    ensure_writable_directory,
    external_cache_paths,
    inspect_models,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspecte les checkpoints locaux requis par PuLID."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Fichier YAML à utiliser à la place de config/default.yaml.",
    )
    parser.add_argument(
        "--show-cache-env",
        action="store_true",
        help="Affiche les emplacements effectifs des caches de modèles.",
    )
    parser.add_argument(
        "--fail-on-internal-cache",
        action="store_true",
        help="Échoue si un cache effectif se trouve hors de models_root.",
    )
    return parser


def _format_bytes(value: int) -> str:
    units = ("o", "Kio", "Mio", "Gio", "Tio")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} Tio"


def _print_cache_env(config: AppConfig, console: Console) -> None:
    table = Table(title="Caches de modèles effectifs")
    table.add_column("Variable")
    table.add_column("Chemin", overflow="fold")
    for name in external_cache_paths(config.models_root):
        table.add_row(name, os.environ.get(name, "absent"))
    console.print(table)


def run_inspection(
    config_path: Path | None,
    console: Console,
    *,
    show_cache_env: bool = False,
    fail_on_internal_cache: bool = False,
) -> int:
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        console.print(f"[bold red]Configuration invalide :[/] {exc}")
        return 2

    configure_external_model_caches(config.models_root)
    console.print(f"Configuration : [cyan]{config.source_path}[/]")
    console.print(f"Racine modèles : [cyan]{config.models_root}[/]")

    failures: list[str] = []
    if show_cache_env or fail_on_internal_cache:
        _print_cache_env(config, console)
    cache_failures = cache_env_violations(config.models_root)
    if fail_on_internal_cache and cache_failures:
        console.print("[red]✗ Cache(s) hors du SSD configuré :[/]")
        for failure in cache_failures:
            console.print(f"  • {failure}")
        failures.append("cache_env")
    elif fail_on_internal_cache:
        console.print("[green]✓ Tous les caches sont sous models_root.[/]")
    if not config.models_root.is_dir():
        console.print("[red]✗ La racine des modèles n'existe pas ou n'est pas un dossier.[/]")
        failures.append("models_root")
    else:
        usage = shutil.disk_usage(config.models_root)
        console.print(
            f"[green]✓ Racine disponible[/] — espace libre : "
            f"[bold]{_format_bytes(usage.free)}[/] / {_format_bytes(usage.total)}"
        )

    inventory = inspect_models(config)

    if inventory.pulid_checkpoints:
        console.print("[green]✓ Checkpoint(s) PuLID :[/]")
        for path in inventory.pulid_checkpoints:
            console.print(f"  • {path}")
    else:
        console.print(f"[red]✗ Checkpoint PuLID introuvable :[/] {config.pulid.checkpoint}")
        failures.append("pulid")

    if inventory.antelope_dir is None:
        console.print(
            f"[red]✗ AntelopeV2 introuvable :[/] {config.insightface.model_dir}"
        )
        failures.append("antelopev2")
    elif inventory.antelope_missing_files:
        console.print(f"[red]✗ AntelopeV2 incomplet :[/] {inventory.antelope_dir}")
        for name in inventory.antelope_missing_files:
            console.print(f"  • fichier manquant : {name}")
        failures.append("antelopev2")
    else:
        console.print(f"[green]✓ AntelopeV2 complet :[/] {inventory.antelope_dir}")

    table = Table(title="Candidats SDXL (.safetensors)")
    table.add_column("Chemin", overflow="fold")
    table.add_column("Configuré", justify="center")
    for path in inventory.sdxl_candidates:
        configured = "✓" if path == config.sdxl.checkpoint else ""
        table.add_row(str(path), configured)
    if inventory.sdxl_candidates:
        console.print(table)
    else:
        console.print("[red]✗ Aucun candidat SDXL .safetensors détecté.[/]")
        failures.append("sdxl")

    if not config.sdxl.checkpoint.is_file():
        console.print(f"[red]✗ Checkpoint SDXL configuré absent :[/] {config.sdxl.checkpoint}")
        failures.append("sdxl_configured")
    else:
        console.print(
            "[green]✓ Checkpoint SDXL configuré (VAE intégré) :[/] "
            f"{config.sdxl.checkpoint}"
        )

    try:
        ensure_writable_directory(config.outputs_dir)
    except (OSError, PermissionError) as exc:
        console.print(f"[red]✗ Sorties non accessibles :[/] {exc}")
        failures.append("outputs")
    else:
        console.print(f"[green]✓ Sorties accessibles en écriture :[/] {config.outputs_dir}")

    if failures:
        console.print(f"[bold red]Inspection échouée ({len(failures)} contrôle(s)).[/]")
        return 1
    console.print("[bold green]Inspection réussie.[/]")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_inspection(
        args.config,
        Console(),
        show_cache_env=args.show_cache_env,
        fail_on_internal_cache=args.fail_on_internal_cache,
    )
