from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from pulid_app.exceptions import PromptTooLongError
from pulid_app.models.sdxl import (
    MAX_PROMPT_TOKENS,
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


class FakeTokenizer:
    model_max_length = 77
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 2

    def __call__(self, text: str, **_kwargs):
        return SimpleNamespace(
            input_ids=[10 + index for index, _word in enumerate(text.split())]
        )

    def num_special_tokens_to_add(self, *, pair: bool) -> int:
        assert pair is False
        return 2

    def build_inputs_with_special_tokens(self, token_ids: list[int]) -> list[int]:
        return [self.bos_token_id, *token_ids, self.eos_token_id]


class FakeEncoderOutput:
    def __init__(self, first_output: np.ndarray, hidden: np.ndarray) -> None:
        self._first_output = first_output
        self.hidden_states = tuple(hidden * layer for layer in range(1, 6))

    def __getitem__(self, index: int) -> np.ndarray:
        if index != 0:
            raise IndexError(index)
        return self._first_output


class FakeTextEncoder:
    def __init__(self, hidden_size: int, *, pooled: bool) -> None:
        self.hidden_size = hidden_size
        self.pooled = pooled
        self.input_batches: list[np.ndarray] = []

    def __call__(self, input_ids: np.ndarray, *, output_hidden_states: bool):
        assert output_hidden_states is True
        self.input_batches.append(input_ids.copy())
        hidden = np.repeat(input_ids[..., None], self.hidden_size, axis=-1).astype(
            np.float32
        )
        if self.pooled:
            first_output = np.arange(
                input_ids.shape[0] * self.hidden_size,
                dtype=np.float32,
            ).reshape(input_ids.shape[0], self.hidden_size)
        else:
            first_output = hidden
        return FakeEncoderOutput(first_output, hidden)


class FakePipeline:
    load_kwargs = None

    def __init__(self) -> None:
        self.scheduler = SimpleNamespace(config={"scheduler": "checkpoint"})
        self.config = SimpleNamespace(force_zeros_for_empty_prompt=True)
        self.tokenizer = FakeTokenizer()
        self.tokenizer_2 = FakeTokenizer()
        self.to_calls: list[str] = []
        self.offload_calls: list[str] = []

    @classmethod
    def from_single_file(cls, checkpoint: str, **kwargs):
        cls.load_kwargs = {"checkpoint": checkpoint, **kwargs}
        return cls()

    def to(self, device: str):
        self.device = device
        self.to_calls.append(device)
        return self

    def enable_model_cpu_offload(self, *, device: str) -> None:
        self.offload_calls.append(device)

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
        long=np.int64,
        tensor=lambda value, **kwargs: np.asarray(value, dtype=kwargs.get("dtype")),
        cat=lambda values, dim: np.concatenate(values, axis=dim),
        zeros_like=np.zeros_like,
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


def test_cuda_model_cpu_offload_avoids_eager_pipeline_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, config_dir = _create_local_sdxl_files(tmp_path)
    model = SDXLModel(
        checkpoint,
        config_dir,
        models_root=tmp_path,
        device="cuda",
        offload_strategy="model_cpu_offload",
    )
    monkeypatch.setattr(model, "_import_ml", lambda: (_fake_torch(), FakePipeline))

    model.load()

    assert model.pipeline.offload_calls == ["cuda"]
    assert model.pipeline.to_calls == []


def test_cuda_default_strategy_keeps_eager_pipeline_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, config_dir = _create_local_sdxl_files(tmp_path)
    model = SDXLModel(
        checkpoint,
        config_dir,
        models_root=tmp_path,
        device="cuda:1",
    )
    monkeypatch.setattr(model, "_import_ml", lambda: (_fake_torch(), FakePipeline))

    model.load()

    assert model.pipeline.to_calls == ["cuda:1"]
    assert model.pipeline.offload_calls == []


@pytest.mark.parametrize("device", ["mps", "cpu"])
def test_model_cpu_offload_is_rejected_without_cuda(
    tmp_path: Path, device: str
) -> None:
    checkpoint, config_dir = _create_local_sdxl_files(tmp_path)

    with pytest.raises(SDXLConfigurationError, match="réservé à CUDA"):
        SDXLModel(
            checkpoint,
            config_dir,
            models_root=tmp_path,
            device=device,
            offload_strategy="model_cpu_offload",
        )


def test_unknown_offload_strategy_is_rejected(tmp_path: Path) -> None:
    checkpoint, config_dir = _create_local_sdxl_files(tmp_path)

    with pytest.raises(SDXLConfigurationError, match="Stratégie d'offload inconnue"):
        SDXLModel(
            checkpoint,
            config_dir,
            models_root=tmp_path,
            device="cuda",
            offload_strategy="sequential",
        )


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


def test_generate_forwards_generic_cross_attention_kwargs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, config_dir = _create_local_sdxl_files(tmp_path)
    model = SDXLModel(
        checkpoint,
        config_dir,
        models_root=tmp_path,
        device="cpu",
    )
    monkeypatch.setattr(model, "_import_ml", lambda: (_fake_torch(), FakePipeline))
    identity_embedding = object()

    model.generate(
        prompt="portrait",
        steps=2,
        width=64,
        height=64,
        cross_attention_kwargs={
            "id_embedding": identity_embedding,
            "id_scale": 0.8,
        },
    )

    assert model.pipeline.generation_kwargs["cross_attention_kwargs"] == {
        "id_embedding": identity_embedding,
        "id_scale": 0.8,
    }


@pytest.mark.parametrize(
    ("token_count", "expected_sequence_length"),
    [(76, 154), (MAX_PROMPT_TOKENS, 308)],
)
def test_generate_encodes_long_prompts_in_aligned_clip_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    token_count: int,
    expected_sequence_length: int,
) -> None:
    checkpoint, config_dir = _create_local_sdxl_files(tmp_path)

    class LongPromptPipeline(FakePipeline):
        def __init__(self) -> None:
            super().__init__()
            self._execution_device = "cpu"
            self.text_encoder = FakeTextEncoder(2, pooled=False)
            self.text_encoder_2 = FakeTextEncoder(3, pooled=True)

    model = SDXLModel(checkpoint, config_dir, models_root=tmp_path, device="cpu")
    monkeypatch.setattr(
        model,
        "_import_ml",
        lambda: (_fake_torch(), LongPromptPipeline),
    )
    prompt = " ".join(f"word-{index}" for index in range(token_count))

    model.generate(
        prompt=prompt,
        negative_prompt="bad anatomy",
        steps=2,
        width=64,
        height=64,
    )

    assert model.pipeline is not None
    kwargs = model.pipeline.generation_kwargs
    assert "prompt" not in kwargs
    assert "negative_prompt" not in kwargs
    assert kwargs["prompt_embeds"].shape == (1, expected_sequence_length, 5)
    assert kwargs["negative_prompt_embeds"].shape == (
        1,
        expected_sequence_length,
        5,
    )
    assert kwargs["pooled_prompt_embeds"].shape == (1, 3)
    assert kwargs["negative_pooled_prompt_embeds"].shape == (1, 3)

    positive_blocks = model.pipeline.text_encoder.input_batches[0]
    recovered_ids = [
        token_id
        for block in positive_blocks
        for token_id in block[1:76]
        if token_id != FakeTokenizer.eos_token_id
    ]
    assert recovered_ids == list(range(10, 10 + token_count))


def test_generate_keeps_native_diffusers_encoding_for_short_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, config_dir = _create_local_sdxl_files(tmp_path)
    model = SDXLModel(checkpoint, config_dir, models_root=tmp_path, device="cpu")
    monkeypatch.setattr(model, "_import_ml", lambda: (_fake_torch(), FakePipeline))

    model.generate(
        prompt="short portrait prompt",
        negative_prompt="bad anatomy",
        clip_skip=1,
        steps=2,
        width=64,
        height=64,
    )

    assert model.pipeline is not None
    assert model.pipeline.generation_kwargs["prompt"] == "short portrait prompt"
    assert model.pipeline.generation_kwargs["negative_prompt"] == "bad anatomy"
    assert model.pipeline.generation_kwargs["clip_skip"] == 1
    assert "prompt_embeds" not in model.pipeline.generation_kwargs


@pytest.mark.parametrize(
    ("clip_skip", "expected_hidden_value"),
    [(None, 4), (1, 3)],
)
def test_generate_selects_expected_hidden_state_for_long_prompts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clip_skip: int | None,
    expected_hidden_value: int,
) -> None:
    checkpoint, config_dir = _create_local_sdxl_files(tmp_path)

    class LongPromptPipeline(FakePipeline):
        def __init__(self) -> None:
            super().__init__()
            self._execution_device = "cpu"
            self.text_encoder = FakeTextEncoder(2, pooled=False)
            self.text_encoder_2 = FakeTextEncoder(3, pooled=True)

    model = SDXLModel(checkpoint, config_dir, models_root=tmp_path, device="cpu")
    monkeypatch.setattr(model, "_import_ml", lambda: (_fake_torch(), LongPromptPipeline))

    model.generate(
        prompt=" ".join(f"word-{index}" for index in range(76)),
        negative_prompt="bad anatomy",
        clip_skip=clip_skip,
        steps=2,
        width=64,
        height=64,
    )

    assert model.pipeline is not None
    kwargs = model.pipeline.generation_kwargs
    assert "clip_skip" not in kwargs
    assert np.all(kwargs["prompt_embeds"][0, 0] == expected_hidden_value)


@pytest.mark.parametrize(
    ("prompt", "negative_prompt", "prompt_kind"),
    [
        (" ".join(["word"] * (MAX_PROMPT_TOKENS + 1)), None, "prompt positif"),
        ("portrait", " ".join(["bad"] * (MAX_PROMPT_TOKENS + 1)), "prompt négatif"),
    ],
)
def test_generate_rejects_prompts_above_255_useful_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt: str,
    negative_prompt: str | None,
    prompt_kind: str,
) -> None:
    checkpoint, config_dir = _create_local_sdxl_files(tmp_path)
    model = SDXLModel(checkpoint, config_dir, models_root=tmp_path, device="cpu")
    monkeypatch.setattr(model, "_import_ml", lambda: (_fake_torch(), FakePipeline))

    with pytest.raises(PromptTooLongError, match=prompt_kind):
        model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=2,
            width=64,
            height=64,
        )


def test_generate_collects_prompt_diffusion_and_vae_timings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, config_dir = _create_local_sdxl_files(tmp_path)

    class Component:
        def forward(self, value):
            return value

    class VAE:
        def decode(self, value):
            return value

    class InstrumentedPipeline(FakePipeline):
        def __init__(self) -> None:
            super().__init__()
            self.unet = Component()
            self.vae = VAE()

        def encode_prompt(self, value):
            return value

        def __call__(self, **kwargs):
            self.encode_prompt(kwargs["prompt"])
            self.unet.forward("latents")
            self.unet.forward("latents")
            self.vae.decode("latents")
            return SimpleNamespace(
                images=[Image.new("RGB", (kwargs["width"], kwargs["height"]))]
            )

    model = SDXLModel(checkpoint, config_dir, models_root=tmp_path, device="cpu")
    monkeypatch.setattr(
        model,
        "_import_ml",
        lambda: (_fake_torch(), InstrumentedPipeline),
    )

    result = model.generate(
        prompt="portrait",
        steps=2,
        width=64,
        height=64,
        collect_timings=True,
    )

    assert result.stage_durations_seconds["prompt_preparation"] >= 0.0
    assert result.stage_durations_seconds["diffusion"] >= 0.0
    assert result.stage_durations_seconds["vae"] >= 0.0
    assert model.pipeline is not None
    assert "encode_prompt" not in vars(model.pipeline)
    assert "forward" not in vars(model.pipeline.unet)
    assert "decode" not in vars(model.pipeline.vae)


def test_set_sampling_configures_dpmpp_2m_sde_and_karras_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, config_dir = _create_local_sdxl_files(tmp_path)
    model = SDXLModel(checkpoint, config_dir, models_root=tmp_path, device="cpu")
    monkeypatch.setattr(model, "_import_ml", lambda: (_fake_torch(), FakePipeline))
    calls: list[tuple[object, dict]] = []

    class FakeScheduler:
        @classmethod
        def from_config(cls, config: object, **kwargs: object) -> object:
            calls.append((config, kwargs))
            return SimpleNamespace(config=config, configured=kwargs)

    imported_classes: list[str] = []

    def import_scheduler(class_name: str):
        imported_classes.append(class_name)
        return FakeScheduler

    monkeypatch.setattr(model, "_import_scheduler_class", import_scheduler)
    model.load()

    model.set_sampling("dpmpp_2m_sde", "karras")

    assert model.sampling_method == "dpmpp_2m_sde"
    assert model.sigma_schedule == "karras"
    assert imported_classes == ["DPMSolverMultistepScheduler"]
    assert calls == [
        (
            {"scheduler": "checkpoint"},
            {
                "algorithm_type": "sde-dpmsolver++",
                "solver_order": 2,
                "use_karras_sigmas": True,
                "use_exponential_sigmas": False,
                "use_beta_sigmas": False,
            },
        )
    ]


@pytest.mark.parametrize(
    ("method", "scheduler_class"),
    [
        ("dpmpp_2m", "DPMSolverMultistepScheduler"),
        ("dpmpp_3m_sde", "DPMSolverMultistepScheduler"),
        ("euler", "EulerDiscreteScheduler"),
        ("euler_ancestral", "EulerAncestralDiscreteScheduler"),
        ("heun", "HeunDiscreteScheduler"),
        ("lms", "LMSDiscreteScheduler"),
        ("ddim", "DDIMScheduler"),
    ],
)
def test_set_sampling_supports_additional_methods(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    scheduler_class: str,
) -> None:
    checkpoint, config_dir = _create_local_sdxl_files(tmp_path)
    model = SDXLModel(checkpoint, config_dir, models_root=tmp_path, device="cpu")
    monkeypatch.setattr(model, "_import_ml", lambda: (_fake_torch(), FakePipeline))

    class FakeScheduler:
        @classmethod
        def from_config(cls, config: object, **kwargs: object) -> object:
            return SimpleNamespace(config=config, configured=kwargs)

    imported_classes: list[str] = []

    def import_scheduler(class_name: str):
        imported_classes.append(class_name)
        return FakeScheduler

    monkeypatch.setattr(model, "_import_scheduler_class", import_scheduler)

    model.set_sampling(method, "normal")

    assert imported_classes == [scheduler_class]
    assert model.sampling_method == method
    assert model.sigma_schedule == "normal"


@pytest.mark.parametrize(
    ("sigma_schedule", "enabled_flag"),
    [
        ("normal", None),
        ("karras", "use_karras_sigmas"),
        ("exponential", "use_exponential_sigmas"),
        ("beta", "use_beta_sigmas"),
    ],
)
def test_set_sampling_configures_each_sigma_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sigma_schedule: str,
    enabled_flag: str | None,
) -> None:
    checkpoint, config_dir = _create_local_sdxl_files(tmp_path)
    model = SDXLModel(checkpoint, config_dir, models_root=tmp_path, device="cpu")
    monkeypatch.setattr(model, "_import_ml", lambda: (_fake_torch(), FakePipeline))
    configured: dict[str, object] = {}

    class FakeScheduler:
        @classmethod
        def from_config(cls, config: object, **kwargs: object) -> object:
            configured.update(kwargs)
            return SimpleNamespace(config=config, configured=kwargs)

    monkeypatch.setattr(
        model,
        "_import_scheduler_class",
        lambda _class_name: FakeScheduler,
    )

    model.set_sampling("euler", sigma_schedule)

    sigma_flags = {
        name: configured[name]
        for name in (
            "use_karras_sigmas",
            "use_exponential_sigmas",
            "use_beta_sigmas",
        )
    }
    assert sum(value is True for value in sigma_flags.values()) == (
        0 if enabled_flag is None else 1
    )
    if enabled_flag is not None:
        assert sigma_flags[enabled_flag] is True


def test_set_sampling_method_none_keeps_checkpoint_scheduler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, config_dir = _create_local_sdxl_files(tmp_path)
    model = SDXLModel(checkpoint, config_dir, models_root=tmp_path, device="cpu")
    monkeypatch.setattr(model, "_import_ml", lambda: (_fake_torch(), FakePipeline))
    model.load()
    assert model.pipeline is not None
    scheduler = object()
    model.pipeline.scheduler = scheduler

    model.set_sampling_method(None)

    assert model.pipeline.scheduler is scheduler
    assert model.sampling_method is None


def test_set_sampling_method_none_restores_checkpoint_scheduler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, config_dir = _create_local_sdxl_files(tmp_path)
    model = SDXLModel(checkpoint, config_dir, models_root=tmp_path, device="cpu")
    monkeypatch.setattr(model, "_import_ml", lambda: (_fake_torch(), FakePipeline))

    class FakeScheduler:
        @classmethod
        def from_config(cls, config: object, **_kwargs: object) -> object:
            return SimpleNamespace(config=config, custom=True)

    monkeypatch.setattr(
        model,
        "_import_scheduler_class",
        lambda _class_name: FakeScheduler,
    )
    model.load()
    assert model.pipeline is not None
    checkpoint_scheduler = model.pipeline.scheduler
    model.set_sampling("dpmpp_2m_sde", "karras")

    model.set_sampling_method(None)

    assert model.pipeline.scheduler is checkpoint_scheduler
    assert model.sampling_method is None
    assert model.sigma_schedule is None


def test_set_sampling_method_none_clears_pending_method_before_reload(
    tmp_path: Path,
) -> None:
    checkpoint, config_dir = _create_local_sdxl_files(tmp_path)
    model = SDXLModel(checkpoint, config_dir, models_root=tmp_path, device="cpu")
    model.sampling_method = "dpmpp_2m_sde"
    model.sigma_schedule = "karras"

    model.set_sampling_method(None)

    assert model.sampling_method is None
    assert model.sigma_schedule is None


def test_set_sampling_method_rejects_unknown_name(tmp_path: Path) -> None:
    checkpoint, config_dir = _create_local_sdxl_files(tmp_path)
    model = SDXLModel(checkpoint, config_dir, models_root=tmp_path, device="cpu")

    with pytest.raises(SDXLConfigurationError, match="Méthode de sampling inconnue"):
        model.set_sampling_method("euler_custom")


def test_set_sampling_rejects_unknown_or_incompatible_sigmas(tmp_path: Path) -> None:
    checkpoint, config_dir = _create_local_sdxl_files(tmp_path)
    model = SDXLModel(checkpoint, config_dir, models_root=tmp_path, device="cpu")

    with pytest.raises(SDXLConfigurationError, match="Courbe de sigmas inconnue"):
        model.set_sampling("euler", "custom")
    with pytest.raises(SDXLConfigurationError, match="incompatible"):
        model.set_sampling("euler_ancestral", "karras")
    with pytest.raises(SDXLConfigurationError, match="nécessite une méthode"):
        model.set_sampling(None, "karras")


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
    torch_module = _fake_torch()
    empty_cache_calls: list[None] = []
    torch_module.mps.empty_cache = lambda: empty_cache_calls.append(None)
    monkeypatch.setattr(
        model, "_import_ml", lambda: (torch_module, OutOfMemoryPipeline)
    )

    with pytest.raises(SDXLLoadError, match="out of memory"):
        model.load()

    assert OutOfMemoryPipeline.calls == 1
    assert len(empty_cache_calls) == 1


def test_close_unloads_pipeline_and_only_empties_mps_cache_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, config_dir = _create_local_sdxl_files(tmp_path)
    torch_module = _fake_torch()
    empty_cache_calls: list[None] = []
    torch_module.mps.empty_cache = lambda: empty_cache_calls.append(None)
    model = SDXLModel(
        checkpoint,
        config_dir,
        models_root=tmp_path,
        device="mps",
        dtype_name="float16",
    )
    monkeypatch.setattr(model, "_import_ml", lambda: (torch_module, FakePipeline))
    model.load()
    pipeline = model.pipeline

    model.close()
    model.close()

    assert pipeline is not None
    assert pipeline.device == "cpu"
    assert model.is_loaded is False
    assert model.active_dtype_name is None
    assert len(empty_cache_calls) == 1


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
