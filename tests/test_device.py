from __future__ import annotations

from types import SimpleNamespace

import pytest

from pulid_app import device


class Availability:
    def __init__(self, available: bool, built: bool = True) -> None:
        self.available = available
        self.built = built

    def is_available(self) -> bool:
        return self.available

    def is_built(self) -> bool:
        return self.built


def _fake_torch(*, cuda: bool, mps: bool) -> SimpleNamespace:
    return SimpleNamespace(
        __version__="test",
        float16="float16",
        float32="float32",
        cuda=SimpleNamespace(
            is_available=lambda: cuda,
            device_count=lambda: 1 if cuda else 0,
            get_device_properties=lambda _index: SimpleNamespace(total_memory=12_000),
        ),
        backends=SimpleNamespace(mps=Availability(mps)),
    )


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [
        (True, True, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_get_best_device_priority(
    monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool, expected: str
) -> None:
    monkeypatch.setattr(device, "_import_torch", lambda: _fake_torch(cuda=cuda, mps=mps))

    assert device.get_best_device() == expected


def test_get_default_dtype_uses_fp16_for_accelerators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        device, "_import_torch", lambda: _fake_torch(cuda=False, mps=True)
    )

    assert device.get_default_dtype("mps") == "float16"
    assert device.get_default_dtype("cuda:0") == "float16"
    assert device.get_default_dtype("cpu") == "float32"


def test_get_default_dtype_rejects_unknown_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        device, "_import_torch", lambda: _fake_torch(cuda=False, mps=False)
    )

    with pytest.raises(ValueError, match="Device non pris en charge"):
        device.get_default_dtype("tpu")


def test_device_report_describes_selected_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        device, "_import_torch", lambda: _fake_torch(cuda=False, mps=True)
    )
    monkeypatch.setattr(device, "_system_memory_bytes", lambda: 32_000)

    report = device.get_device_report()

    assert report.selected_device == "mps"
    assert report.mps_available is True
    assert report.cuda_available is False
    assert report.dtype == "float16"
    assert report.accelerator_memory_bytes == 32_000

