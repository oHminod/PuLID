from __future__ import annotations

from types import SimpleNamespace

import pytest

from pulid_app.pipeline.memory import MemoryManager, MemoryManagerError


class FakeModule:
    def __init__(self) -> None:
        self.moves: list[str] = []

    def to(self, device: str) -> "FakeModule":
        self.moves.append(device)
        return self


class FakeMPS:
    def __init__(self) -> None:
        self.empty_cache_calls = 0

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def current_allocated_memory(self) -> int:
        return 10

    def driver_allocated_memory(self) -> int:
        return 20

    def recommended_max_memory(self) -> int:
        return 30


class FakeCUDA:
    def __init__(self) -> None:
        self.empty_cache_calls = 0
        self.queried_devices: list[str] = []

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def memory_allocated(self, device: str) -> int:
        self.queried_devices.append(device)
        return 40

    def memory_reserved(self, device: str) -> int:
        self.queried_devices.append(device)
        return 50

    def get_device_properties(self, device: str) -> SimpleNamespace:
        self.queried_devices.append(device)
        return SimpleNamespace(total_memory=60)


def _fake_torch() -> SimpleNamespace:
    return SimpleNamespace(mps=FakeMPS(), cuda=FakeCUDA())


def test_move_and_unload_mps_module_only_empties_cache_once(tmp_path) -> None:
    torch_module = _fake_torch()
    manager = MemoryManager(tmp_path, device="mps", torch_module=torch_module)
    module = FakeModule()

    assert manager.move_to_device(module) is module
    assert manager.unload(module) is None

    assert module.moves == ["mps", "cpu"]
    assert torch_module.mps.empty_cache_calls == 1
    assert manager.cleanup() is False
    assert torch_module.mps.empty_cache_calls == 1


def test_multiple_unloads_can_be_grouped_into_one_cleanup(tmp_path) -> None:
    torch_module = _fake_torch()
    manager = MemoryManager(tmp_path, device="cuda", torch_module=torch_module)
    first = manager.move_to_device(FakeModule())
    second = manager.move_to_device(FakeModule())

    manager.unload(first, cleanup=False)
    manager.unload(second, cleanup=False)

    assert torch_module.cuda.empty_cache_calls == 0
    assert manager.cleanup() is True
    assert torch_module.cuda.empty_cache_calls == 1


def test_cpu_cleanup_does_not_call_accelerator_caches(tmp_path) -> None:
    torch_module = _fake_torch()
    manager = MemoryManager(tmp_path, device="cpu", torch_module=torch_module)
    module = manager.move_to_device(FakeModule())

    manager.unload(module)

    assert torch_module.mps.empty_cache_calls == 0
    assert torch_module.cuda.empty_cache_calls == 0


def test_snapshot_reports_mps_counters(tmp_path) -> None:
    manager = MemoryManager(tmp_path, device="mps", torch_module=_fake_torch())

    snapshot = manager.snapshot()

    assert snapshot.device == "mps"
    assert snapshot.allocated_bytes == 10
    assert snapshot.reserved_bytes == 20
    assert snapshot.limit_bytes == 30


def test_snapshot_preserves_indexed_cuda_device(tmp_path) -> None:
    torch_module = _fake_torch()
    manager = MemoryManager(tmp_path, device="cuda:1", torch_module=torch_module)

    snapshot = manager.snapshot()

    assert snapshot.device == "cuda:1"
    assert snapshot.allocated_bytes == 40
    assert snapshot.reserved_bytes == 50
    assert snapshot.limit_bytes == 60
    assert torch_module.cuda.queried_devices == ["cuda:1", "cuda:1", "cuda:1"]


def test_move_rejects_object_without_to_method(tmp_path) -> None:
    manager = MemoryManager(tmp_path, device="mps", torch_module=_fake_torch())

    with pytest.raises(MemoryManagerError, match="méthode to"):
        manager.move_to_device(object())


def test_invalid_device_is_rejected_before_importing_torch(tmp_path) -> None:
    with pytest.raises(MemoryManagerError, match="Device non pris en charge"):
        MemoryManager(tmp_path, device="metal")
