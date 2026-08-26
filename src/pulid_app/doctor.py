"""Diagnostic local des prérequis nécessaires à une génération PuLID."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import os
from pathlib import Path
from typing import Any, Callable, Literal

from rich.console import Console
from rich.table import Table

from pulid_app.config import AppConfig
from pulid_app.models.pulid_assets import (
    PuLIDAssetError,
    ensure_official_source,
)
from pulid_app.models.sdxl import REQUIRED_SDXL_CONFIG_FILES
from pulid_app.paths import (
    ANTELOPEV2_REQUIRED_FILES,
    cache_env_violations,
    configure_external_model_caches,
    ensure_writable_directory,
)


DoctorStatus = Literal["ok", "warning", "error"]
FACEXLIB_REQUIRED_FILES = (
    "detection_Resnet50_Final.pth",
    "parsing_parsenet.pth",
    "parsing_bisenet.pth",
)
CRITICAL_DISTRIBUTIONS = (
    "torch",
    "diffusers",
    "transformers",
    "safetensors",
    "insightface",
    "onnxruntime",
    "Pillow",
    "huggingface-hub",
)


@dataclass(frozen=True)
class DoctorCheck:
    """Résultat d'un contrôle unique et actionnable."""

    name: str
    status: DoctorStatus
    details: str


@dataclass(frozen=True)
class DoctorReport:
    """Ensemble des contrôles d'environnement de la CLI."""

    checks: tuple[DoctorCheck, ...]

    @property
    def errors(self) -> tuple[DoctorCheck, ...]:
        return tuple(check for check in self.checks if check.status == "error")

    @property
    def healthy(self) -> bool:
        return not self.errors


def _expected_volume_mount(path: Path) -> Path | None:
    parts = path.resolve(strict=False).parts
    if len(parts) >= 3 and parts[1] == "Volumes":
        return Path("/Volumes") / parts[2]
    return None


def _nearest_mount(path: Path) -> Path:
    candidate = path.resolve(strict=False)
    while candidate.parent != candidate:
        try:
            if candidate.is_mount():
                return candidate
        except NotImplementedError:
            # Path.is_mount() n'est pris en charge sous Windows qu'à partir
            # de Python 3.12. L'ancre représente le volume (par ex. D:\\).
            return Path(candidate.anchor)
        candidate = candidate.parent
    return candidate


def _readable_file_check(name: str, path: Path) -> DoctorCheck:
    if not path.is_file():
        return DoctorCheck(
            name,
            "error",
            f"Absent : {path}",
        )
    if not os.access(path, os.R_OK):
        return DoctorCheck(
            name,
            "error",
            f"Lecture refusée : {path}",
        )
    return DoctorCheck(name, "ok", str(path))


def _all_readable_files_check(
    name: str,
    paths: tuple[Path, ...],
    *,
    corrective_action: str,
) -> DoctorCheck:
    missing = [path for path in paths if not path.is_file()]
    unreadable = [path for path in paths if path.is_file() and not os.access(path, os.R_OK)]
    if missing or unreadable:
        details: list[str] = []
        if missing:
            details.append("absents : " + ", ".join(str(path) for path in missing))
        if unreadable:
            details.append(
                "illisibles : " + ", ".join(str(path) for path in unreadable)
            )
        details.append(corrective_action)
        return DoctorCheck(name, "error", "; ".join(details))
    return DoctorCheck(name, "ok", f"{len(paths)} fichier(s) lisible(s)")


def _eva_clip_check(config: AppConfig) -> DoctorCheck:
    cache_root = (
        config.models_root
        / "huggingface"
        / "hub"
        / "models--QuanSun--EVA-CLIP"
        / "snapshots"
    )
    candidates = tuple(cache_root.glob("*/*.pt")) if cache_root.is_dir() else ()
    readable = tuple(path for path in candidates if os.access(path, os.R_OK))
    if not readable:
        return DoctorCheck(
            "EVA-CLIP",
            "error",
            "Poids EVA-CLIP absents du cache externe. Relancez `pulid-install` "
            "pour réparer les modèles manquants.",
        )
    return DoctorCheck("EVA-CLIP", "ok", str(readable[0]))


def build_doctor_report(
    config: AppConfig,
    *,
    device_reporter: Callable[[], Any] | None = None,
    version_resolver: Callable[[str], str] | None = None,
) -> DoctorReport:
    """Exécute les contrôles sans charger de poids de modèles."""

    configure_external_model_caches(config.models_root)
    checks: list[DoctorCheck] = []

    expected_mount = _expected_volume_mount(config.models_root)
    if not config.models_root.is_dir():
        checks.append(
            DoctorCheck(
                "SSD / models_root",
                "error",
                f"Racine absente : {config.models_root}. Montez le SSD configuré.",
            )
        )
    elif expected_mount is not None and not expected_mount.is_mount():
        checks.append(
            DoctorCheck(
                "SSD / models_root",
                "error",
                f"Le volume attendu n'est pas monté : {expected_mount}",
            )
        )
    else:
        mount = expected_mount or _nearest_mount(config.models_root)
        checks.append(
            DoctorCheck(
                "SSD / models_root",
                "ok",
                f"{config.models_root} (volume : {mount})",
            )
        )

    violations = cache_env_violations(config.models_root)
    checks.append(
        DoctorCheck(
            "Caches externes",
            "error" if violations else "ok",
            "; ".join(violations) if violations else f"Sous {config.models_root}",
        )
    )

    checks.append(_readable_file_check("Checkpoint SDXL", config.sdxl.checkpoint))
    checks.append(_readable_file_check("Checkpoint PuLID", config.pulid.checkpoint))
    if config.text_embedding is not None:
        checks.append(
            _readable_file_check(
                "Modèle embedding GGUF",
                config.text_embedding.checkpoint,
            )
        )
    antelope_paths = tuple(
        config.insightface.model_dir / name
        for name in sorted(ANTELOPEV2_REQUIRED_FILES)
    )
    checks.append(
        _all_readable_files_check(
            "AntelopeV2",
            antelope_paths,
            corrective_action="Complétez insightface.model_root/model_name.",
        )
    )

    if config.sdxl.config_dir is None:
        checks.append(
            DoctorCheck(
                "Configuration SDXL",
                "error",
                "sdxl.config_dir est absent. Exécutez scripts/prepare_sdxl_config.py.",
            )
        )
    else:
        sdxl_config_paths = tuple(
            config.sdxl.config_dir / relative
            for relative in REQUIRED_SDXL_CONFIG_FILES
        )
        checks.append(
            _all_readable_files_check(
                "Configuration SDXL",
                sdxl_config_paths,
                corrective_action="Exécutez scripts/prepare_sdxl_config.py.",
            )
        )

    if config.pulid.source_dir is None:
        checks.append(
            DoctorCheck("Runtime PuLID", "error", "pulid.source_dir est absent.")
        )
    else:
        try:
            source = ensure_official_source(
                config.pulid.source_dir,
                config.pulid.revision,
                allow_download=False,
            )
        except PuLIDAssetError as exc:
            checks.append(DoctorCheck("Runtime PuLID", "error", str(exc)))
        else:
            checks.append(
                DoctorCheck(
                    "Runtime PuLID",
                    "ok",
                    f"Révision {source.revision} — {source.path}",
                )
            )

    facexlib_root = config.pulid.facexlib_root
    if facexlib_root is None:
        checks.append(
            DoctorCheck("FaceXLib", "error", "pulid.facexlib_root est absent.")
        )
    else:
        checks.append(
            _all_readable_files_check(
                "FaceXLib",
                tuple(facexlib_root / name for name in FACEXLIB_REQUIRED_FILES),
                corrective_action="Préparez les poids FaceXLib depuis le test PuLID.",
            )
        )
    checks.append(_eva_clip_check(config))

    try:
        ensure_writable_directory(config.outputs_dir)
    except (OSError, PermissionError) as exc:
        checks.append(DoctorCheck("Dossier outputs", "error", str(exc)))
    else:
        checks.append(DoctorCheck("Dossier outputs", "ok", str(config.outputs_dir)))

    if device_reporter is None:
        from pulid_app.device import get_device_report

        device_reporter = get_device_report
    try:
        device = device_reporter()
    except Exception as exc:
        checks.append(DoctorCheck("Accélérateur", "error", str(exc)))
    else:
        accelerator_available = bool(device.mps_available or device.cuda_available)
        checks.append(
            DoctorCheck(
                "Accélérateur",
                "ok" if accelerator_available else "warning",
                f"sélection={device.selected_device}, "
                f"MPS={'oui' if device.mps_available else 'non'}, "
                f"CUDA={'oui' if device.cuda_available else 'non'}",
            )
        )

    resolver = version_resolver or metadata.version
    distributions = list(CRITICAL_DISTRIBUTIONS)
    if config.text_embedding is not None:
        distributions.append("llama-cpp-python")
    for distribution in distributions:
        try:
            version = resolver(distribution)
        except metadata.PackageNotFoundError:
            checks.append(
                DoctorCheck(
                    f"Version {distribution}",
                    "error",
                    "Dépendance absente ; réinstallez les extras "
                    "inference,pulid,server,embeddings.",
                )
            )
        else:
            checks.append(DoctorCheck(f"Version {distribution}", "ok", version))

    return DoctorReport(tuple(checks))


def print_doctor_report(report: DoctorReport, console: Console) -> None:
    """Affiche un rapport compact et un résumé final."""

    symbols = {"ok": "✓", "warning": "!", "error": "✗"}
    styles = {"ok": "green", "warning": "yellow", "error": "red"}
    table = Table(title="Diagnostic PuLID")
    table.add_column("État", justify="center")
    table.add_column("Contrôle")
    table.add_column("Détails", overflow="fold")
    for check in report.checks:
        table.add_row(
            f"[{styles[check.status]}]{symbols[check.status]}[/]",
            check.name,
            check.details,
        )
    console.print(table)
    if report.healthy:
        console.print("[bold green]Doctor réussi : environnement prêt.[/]")
    else:
        console.print(
            f"[bold red]Doctor échoué : {len(report.errors)} problème(s) à corriger.[/]"
        )
