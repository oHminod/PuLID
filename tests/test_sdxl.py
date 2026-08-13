from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from pulid_app.models.sdxl import (
    REQUIRED_SDXL_CONFIG_FILES,
    SDXLConfigurationError,
    SDXLLoadError,
    SDXLModel,
)


class FakeInferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class FakeGenerator:
    def __init__(self, device: str) -> None:
        self.device = device
        self.seed = None

    def manual_seed(self, seed: int):
        self.seed = seed
        return self


class FakePipeline:
    load_kwargs = None

    @classmethod
    def from_single_file(cls, checkpoint: str, **kwargs):
        cls.load_kwargs = {"checkpoint": checkpoint, **kwargs}
        return cls()

    def to(self, device: str):
        self.device = device
        return self

    def enable_attention_slicing(self, mode: str) -> None:
        self.attention_slicing = mode

    def __call__(self, **kwargs):
        self.generation_kwargs = kwargs
        return SimpleNamespace(images=[Image.new("RGB", (kwargs["width"], kwargs["height"]))])


def _create_local_sdxl_files(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "realvisxl.safetensors"
    checkpoint.touch()
    config_dir = tmp_path / "config"
    for relative in REQUIRED_SDXL_CONFIG_FILES:
        path = config_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    return checkpoint, config_dir


def _fake_torch() -> SimpleNamespace:
    return SimpleNamespace(
        float16="float16",
        float32="float32",
        Generator=FakeGenerator,
        inference_mode=lambda: FakeInferenceMode(),
        mps=SimpleNamespace(empty_cache=lambda: None),
        cuda=SimpleNamespace(empty_cache=lambda: None),
    )


def test_load_uses_local_config_and_disables_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, config_dir = _create_local_sdxl_files(tmp_path)
    model = SDXLModel(
        checkpoint,
        config_dir,
        models_root=tmp_path,
        device="mps",
        dtype_name="float16",
    )
    monkeypatch.setattr(model, "_import_ml", lambda: (_fake_torch(), FakePipeline))

    model.load()

    assert FakePipeline.load_kwargs["checkpoint"] == str(checkpoint)
    assert FakePipeline.load_kwargs["config"] == str(config_dir)
    assert FakePipeline.load_kwargs["local_files_only"] is True
    assert FakePipeline.load_kwargs["add_watermarker"] is False
    assert model.active_dtype_name == "float16"


def test_generate_returns_image_and_effective_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, config_dir = _create_local_sdxl_files(tmp_path)
    model = SDXLModel(
        checkpoint,
        config_dir,
        models_root=tmp_path,
        device="cpu",
        dtype_name="float16",
    )
    monkeypatch.setattr(model, "_import_ml", lambda: (_fake_torch(), FakePipeline))

    result = model.generate(prompt="portrait", seed=42, steps=2, width=64, height=64)

    assert result.image.size == (64, 64)
    assert result.seed == 42
    assert result.device == "cpu"
    assert result.dtype == "float32"


def test_mps_load_retries_fp32_after_non_oom_fp16_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, config_dir = _create_local_sdxl_files(tmp_path)

    class FallbackPipeline(FakePipeline):
        dtypes: list[str] = []

        @classmethod
        def from_single_file(cls, checkpoint: str, **kwargs):
            cls.dtypes.append(kwargs["torch_dtype"])
            if len(cls.dtypes) == 1:
                raise RuntimeError("opération float16 MPS non prise en charge")
            return cls()

    model = SDXLModel(
        checkpoint,
        config_dir,
        models_root=tmp_path,
        device="mps",
        dtype_name="float16",
    )
    monkeypatch.setattr(model, "_import_ml", lambda: (_fake_torch(), FallbackPipeline))

    model.load()

    assert FallbackPipeline.dtypes == ["float16", "float32"]
    assert model.active_dtype_name == "float32"
    assert model.dtype_fallback_used is True


def test_mps_load_does_not_retry_fp32_after_out_of_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, config_dir = _create_local_sdxl_files(tmp_path)

    class OutOfMemoryPipeline(FakePipeline):
        calls = 0

        @classmethod
        def from_single_file(cls, checkpoint: str, **kwargs):
            cls.calls += 1
            raise RuntimeError("MPS backend out of memory")

    model = SDXLModel(
        checkpoint,
        config_dir,
        models_root=tmp_path,
        device="mps",
        dtype_name="float16",
    )
    monkeypatch.setattr(
        model, "_import_ml", lambda: (_fake_torch(), OutOfMemoryPipeline)
    )

    with pytest.raises(SDXLLoadError, match="out of memory"):
        model.load()

    assert OutOfMemoryPipeline.calls == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt": ""},
        {"prompt": "portrait", "seed": -1},
        {"prompt": "portrait", "steps": 0},
        {"prompt": "portrait", "width": 63},
        {"prompt": "portrait", "guidance_scale": -1.0},
    ],
)
def test_generate_rejects_invalid_parameters(tmp_path: Path, kwargs: dict) -> None:
    checkpoint, config_dir = _create_local_sdxl_files(tmp_path)
    model = SDXLModel(checkpoint, config_dir, models_root=tmp_path, device="cpu")

    with pytest.raises(SDXLConfigurationError):
        model.generate(**kwargs)
