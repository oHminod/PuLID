"""Cycle de vie explicite des modules et caches mémoire PyTorch."""

from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path
from typing import Any

from pulid_app.config import AppConfig
from pulid_app.paths import configure_external_model_caches


SUPPORTED_DEVICE_TYPES = frozenset({"cpu", "cuda", "mps"})


class MemoryManagerError(RuntimeError):
    """Un déplacement ou un nettoyage mémoire n'a pas pu aboutir."""


@dataclass(frozen=True)
class MemorySnapshot:
    """Mesures mémoire disponibles pour un device à un instant donné."""

    device: str
    allocated_bytes: int | None
    reserved_bytes: int | None
    limit_bytes: int | None


def _normalize_device(device: str) -> str:
    normalized = device.strip().casefold()
    device_type = normalized.split(":", maxsplit=1)[0]
    if device_type not in SUPPORTED_DEVICE_TYPES:
        supported = ", ".join(sorted(SUPPORTED_DEVICE_TYPES))
        raise MemoryManagerError(
            f"Device non pris en charge : {device!r}. Valeurs acceptées : {supported}."
        )
    return normalized


def _device_type(device: str) -> str:
    return _normalize_device(device).split(":", maxsplit=1)[0]


def _optional_int(callable_: Any, *args: Any) -> int | None:
    if not callable(callable_):
        return None
    try:
        return int(callable_(*args))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


class MemoryManager:
    """Déplace, décharge et nettoie les ressources d'un pipeline séquentiel.

    Le gestionnaire ne conserve jamais les modules eux-mêmes. Il suit seulement
    leur device par identifiant afin de vider une fois le cache de l'accélérateur
    après leur retour sur CPU.
    """

    def __init__(
        self,
        models_root: str | Path,
        *,
        device: str = "cpu",
        torch_module: Any | None = None,
    ) -> None:
        self.models_root = Path(models_root).expanduser().resolve(strict=False)
        self.device = _normalize_device(device)
        self._torch_module = torch_module
        self._module_devices: dict[int, str] = {}
        self._dirty_accelerators: set[str] = set()
        self._cleanup_pending = False

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        *,
        device: str | None = None,
        torch_module: Any | None = None,
    ) -> "MemoryManager":
        return cls(
            config.models_root,
            device=device or config.device.preferred,
            torch_module=torch_module,
        )

    def bind_torch(self, torch_module: Any) -> None:
        """Réutilise le module PyTorch déjà importé par le pipeline appelant."""

        if torch_module is None:
            raise MemoryManagerError("Le module PyTorch fourni ne peut pas être nul.")
        self._torch_module = torch_module

    def _import_torch(self) -> Any:
        if self._torch_module is not None:
            return self._torch_module
        configure_external_model_caches(self.models_root)
        try:
            import torch
        except ImportError as exc:
            raise MemoryManagerError(
                "PyTorch est requis pour nettoyer la mémoire. Activez .venv puis "
                "exécutez `uv pip install -e '.[dev]'`."
            ) from exc
        self._torch_module = torch
        return torch

    def move_to_device(self, module: Any, device: str | None = None) -> Any:
        """Déplace un module et retourne l'objet renvoyé par sa méthode ``to``."""

        if module is None:
            raise MemoryManagerError("Impossible de déplacer un module nul.")
        mover = getattr(module, "to", None)
        if not callable(mover):
            raise MemoryManagerError(
                f"L'objet {type(module).__name__} ne fournit pas de méthode to(device)."
            )

        target = _normalize_device(device or self.device)
        previous = self._module_devices.pop(id(module), None)
        if previous is not None and _device_type(previous) != "cpu":
            self._dirty_accelerators.add(_device_type(previous))
        if _device_type(target) != "cpu":
            # Même un déplacement interrompu peut avoir réservé de la mémoire.
            self._dirty_accelerators.add(_device_type(target))
        self._cleanup_pending = True

        try:
            moved = mover(target)
        except Exception as exc:
            raise MemoryManagerError(
                f"Impossible de déplacer {type(module).__name__} vers {target} : {exc}"
            ) from exc
        result = moved if moved is not None else module
        self._module_devices[id(result)] = target
        return result

    def unload(self, module: Any, *, cleanup: bool = True) -> None:
        """Replace un module sur CPU puis libère les caches devenus inutiles."""

        if module is not None:
            previous = self._module_devices.pop(id(module), self.device)
            previous_type = _device_type(previous)
            if previous_type != "cpu":
                self._dirty_accelerators.add(previous_type)
            self._cleanup_pending = True

            mover = getattr(module, "to", None)
            if not callable(mover):
                raise MemoryManagerError(
                    f"L'objet {type(module).__name__} ne fournit pas de méthode to(device)."
                )
            try:
                mover("cpu")
            except Exception as exc:
                raise MemoryManagerError(
                    f"Impossible de décharger {type(module).__name__} vers CPU : {exc}"
                ) from exc

        if cleanup:
            self.cleanup()
        return None

    def cleanup(self, *, force: bool = False) -> bool:
        """Collecte Python et vide au plus une fois chaque cache accélérateur sale.

        Retourne ``True`` lorsqu'un nettoyage a effectivement été exécuté.
        ``force`` sert aux allocations de tenseurs externes que le gestionnaire
        n'a pas pu observer.
        """

        if not force and not self._cleanup_pending and not self._dirty_accelerators:
            return False

        gc.collect()
        accelerators = set(self._dirty_accelerators)
        if force and _device_type(self.device) != "cpu":
            accelerators.add(_device_type(self.device))

        torch_module = self._import_torch() if accelerators else None
        remaining = set(accelerators)
        for accelerator in sorted(accelerators):
            backend = getattr(torch_module, accelerator, None)
            empty_cache = getattr(backend, "empty_cache", None)
            if not callable(empty_cache):
                continue
            try:
                empty_cache()
            except Exception as exc:
                self._dirty_accelerators = remaining
                self._cleanup_pending = True
                raise MemoryManagerError(
                    f"Impossible de vider le cache {accelerator.upper()} : {exc}"
                ) from exc
            remaining.discard(accelerator)

        self._dirty_accelerators = remaining
        self._cleanup_pending = bool(remaining)
        if remaining:
            missing = ", ".join(sorted(remaining))
            raise MemoryManagerError(
                f"PyTorch ne fournit pas empty_cache() pour : {missing}."
            )
        return True

    def snapshot(self, device: str | None = None) -> MemorySnapshot:
        """Retourne les compteurs exposés par PyTorch, sans allocation."""

        selected = _normalize_device(device or self.device)
        selected_type = _device_type(selected)
        if selected_type == "cpu":
            return MemorySnapshot(selected, None, None, None)

        torch_module = self._import_torch()
        backend = getattr(torch_module, selected_type, None)
        if selected_type == "mps":
            return MemorySnapshot(
                selected,
                _optional_int(getattr(backend, "current_allocated_memory", None)),
                _optional_int(getattr(backend, "driver_allocated_memory", None)),
                _optional_int(getattr(backend, "recommended_max_memory", None)),
            )

        properties = None
        get_properties = getattr(backend, "get_device_properties", None)
        if callable(get_properties):
            try:
                properties = get_properties(selected)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                properties = None
        limit = getattr(properties, "total_memory", None)
        return MemorySnapshot(
            selected,
            _optional_int(getattr(backend, "memory_allocated", None), selected),
            _optional_int(getattr(backend, "memory_reserved", None), selected),
            int(limit) if isinstance(limit, int) else None,
        )
