"""Détection du backend PyTorch et rapport matériel léger."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, Any

from pulid_app.config import load_config
from pulid_app.exceptions import UnsupportedDeviceError
from pulid_app.paths import configure_external_model_caches

if TYPE_CHECKING:
    import torch
    from rich.console import Console


SUPPORTED_DEVICE_TYPES = frozenset({"cuda", "mps", "cpu"})


@dataclass(frozen=True)
class DeviceReport:
    """Informations utiles au diagnostic du backend d'inférence."""

    selected_device: str
    torch_version: str
    mps_available: bool
    mps_built: bool
    cuda_available: bool
    cuda_device_count: int
    dtype: str
    system_memory_bytes: int | None
    accelerator_memory_bytes: int | None


def _import_torch() -> Any:
    """Configure les caches externes avant l'import lourd de PyTorch."""

    config = load_config()
    configure_external_model_caches(config.models_root)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch n'est pas installé. Activez .venv puis exécutez "
            "`uv pip install -e '.[dev]'`."
        ) from exc
    return torch


def _mps_backend(torch_module: Any) -> Any | None:
    backends = getattr(torch_module, "backends", None)
    return getattr(backends, "mps", None)


def _mps_is_available(torch_module: Any) -> bool:
    backend = _mps_backend(torch_module)
    checker = getattr(backend, "is_available", None)
    return bool(checker()) if callable(checker) else False


def _mps_is_built(torch_module: Any) -> bool:
    backend = _mps_backend(torch_module)
    checker = getattr(backend, "is_built", None)
    return bool(checker()) if callable(checker) else False


def _cuda_is_available(torch_module: Any) -> bool:
    cuda = getattr(torch_module, "cuda", None)
    checker = getattr(cuda, "is_available", None)
    return bool(checker()) if callable(checker) else False


def get_best_device() -> str:
    """Sélectionne le backend disponible selon l'ordre CUDA, MPS, CPU."""

    torch_module = _import_torch()
    if _cuda_is_available(torch_module):
        return "cuda"
    if _mps_is_available(torch_module):
        return "mps"
    return "cpu"


def _device_type(device: str) -> str:
    device_type = device.strip().casefold().split(":", maxsplit=1)[0]
    if device_type not in SUPPORTED_DEVICE_TYPES:
        supported = ", ".join(sorted(SUPPORTED_DEVICE_TYPES))
        raise UnsupportedDeviceError(
            f"Device non pris en charge : {device!r}. Valeurs acceptées : {supported}."
        )
    return device_type


def get_default_dtype(device: str) -> "torch.dtype":
    """Retourne FP16 sur accélérateur et FP32 sur CPU."""

    torch_module = _import_torch()
    if _device_type(device) in {"cuda", "mps"}:
        return torch_module.float16
    return torch_module.float32


def _dtype_name(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


def _system_memory_bytes() -> int | None:
    """Lit la mémoire physique sans ajouter de dépendance système."""

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    if not isinstance(page_size, int) or not isinstance(page_count, int):
        return None
    if page_size <= 0 or page_count <= 0:
        return None
    return page_size * page_count


def _accelerator_memory_bytes(torch_module: Any, device: str) -> int | None:
    if device != "cuda":
        # Apple Silicon partage la mémoire système entre CPU et GPU.
        return _system_memory_bytes() if device == "mps" else None
    try:
        properties = torch_module.cuda.get_device_properties(0)
        return int(properties.total_memory)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def get_device_report() -> DeviceReport:
    """Construit un rapport sans allouer de tenseur sur l'accélérateur."""

    torch_module = _import_torch()
    cuda_available = _cuda_is_available(torch_module)
    mps_available = _mps_is_available(torch_module)
    if cuda_available:
        selected = "cuda"
    elif mps_available:
        selected = "mps"
    else:
        selected = "cpu"

    cuda_count = 0
    if cuda_available:
        try:
            cuda_count = int(torch_module.cuda.device_count())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            cuda_count = 0

    dtype = torch_module.float16 if selected in {"cuda", "mps"} else torch_module.float32
    return DeviceReport(
        selected_device=selected,
        torch_version=str(torch_module.__version__),
        mps_available=mps_available,
        mps_built=_mps_is_built(torch_module),
        cuda_available=cuda_available,
        cuda_device_count=cuda_count,
        dtype=_dtype_name(dtype),
        system_memory_bytes=_system_memory_bytes(),
        accelerator_memory_bytes=_accelerator_memory_bytes(torch_module, selected),
    )


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "indisponible"
    units = ("o", "Kio", "Mio", "Gio", "Tio")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} Tio"


def print_device_report(console: "Console | None" = None) -> DeviceReport:
    """Affiche et retourne le rapport du backend sélectionné."""

    if console is None:
        from rich.console import Console

        console = Console()

    report = get_device_report()
    console.print(f"Device sélectionné : [bold cyan]{report.selected_device}[/]")
    console.print(f"PyTorch : {report.torch_version}")
    console.print(
        f"MPS disponible : {'oui' if report.mps_available else 'non'} "
        f"(support compilé : {'oui' if report.mps_built else 'non'})"
    )
    console.print(
        f"CUDA disponible : {'oui' if report.cuda_available else 'non'} "
        f"(devices : {report.cuda_device_count})"
    )
    console.print(f"Dtype par défaut : {report.dtype}")
    console.print(f"Mémoire système : {_format_bytes(report.system_memory_bytes)}")
    if report.selected_device == "mps":
        console.print(
            "Mémoire accélérateur : "
            f"{_format_bytes(report.accelerator_memory_bytes)} (mémoire unifiée)"
        )
    elif report.selected_device == "cuda":
        console.print(
            f"Mémoire accélérateur : {_format_bytes(report.accelerator_memory_bytes)}"
        )
    return report
