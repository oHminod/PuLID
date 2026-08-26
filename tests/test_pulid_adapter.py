from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pulid_app.models.pulid_adapter import (
    PuLIDAdapter,
    PuLIDConfigurationError,
    PuLIDIdentityError,
    PuLIDIdentityFeatures,
    split_checkpoint_state,
)


def test_split_checkpoint_state_keeps_official_modules_separate() -> None:
    state = {
        "id_adapter.latents": "id",
        "id_adapter_attn_layers.1.id_to_k.weight": "attention",
    }

    grouped = split_checkpoint_state(state)

    assert grouped == {
        "id_adapter": {"latents": "id"},
        "id_adapter_attn_layers": {"1.id_to_k.weight": "attention"},
    }


def test_offline_eva_pretrained_is_resolved_from_local_cache(
    tmp_path: Path, monkeypatch
) -> None:
    cached = tmp_path / "eva.pt"
    cached.touch()
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda repository, filename, cache_dir: str(cached),
    )
    adapter = PuLIDAdapter(
        tmp_path / "pulid.safetensors",
        models_root=tmp_path,
        allow_downloads=False,
    )
    runtime = SimpleNamespace(
        eva_clip=SimpleNamespace(
            get_pretrained_cfg=lambda model, tag: {
                "hf_hub": "QuanSun/EVA-CLIP/eva.pt"
            }
        )
    )

    assert adapter._resolve_eva_pretrained(runtime) == str(cached)


def test_offline_eva_pretrained_finds_revision_pinned_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    cached = (
        tmp_path
        / "huggingface"
        / "hub"
        / "models--QuanSun--EVA-CLIP"
        / "snapshots"
        / "11afd202f2ae80869d6cef18b1ec775e79bd8d12"
        / "EVA02_CLIP_L_336_psz14_s6B.pt"
    )
    cached.parent.mkdir(parents=True)
    cached.touch()
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda repository, filename, cache_dir: None,
    )
    adapter = PuLIDAdapter(
        tmp_path / "pulid.safetensors",
        models_root=tmp_path,
        allow_downloads=False,
    )
    runtime = SimpleNamespace(
        eva_clip=SimpleNamespace(
            get_pretrained_cfg=lambda model, tag: {
                "hf_hub": (
                    "QuanSun/EVA-CLIP/"
                    "EVA02_CLIP_L_336_psz14_s6B.pt"
                )
            }
        )
    )

    assert adapter._resolve_eva_pretrained(runtime) == str(cached)


def test_split_checkpoint_state_rejects_unknown_module() -> None:
    with pytest.raises(PuLIDConfigurationError, match="modules inattendus"):
        split_checkpoint_state(
            {
                "id_adapter.latents": object(),
                "id_adapter_attn_layers.1.weight": object(),
                "vae.weight": object(),
            }
        )


class FakeProcessor:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.moves: list[dict] = []

    def to(self, **kwargs):
        self.moves.append(kwargs)
        return self


class FakeModuleList(list):
    def load_state_dict(self, state, strict: bool):
        self.loaded = (state, strict)

    def eval(self):
        self.evaluated = True
        return self


class FakeUnet:
    def __init__(self) -> None:
        self.attn_processors = OrderedDict(
            (
                (
                    "down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor",
                    object(),
                ),
                (
                    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor",
                    object(),
                ),
            )
        )
        self.config = SimpleNamespace(
            block_out_channels=(320, 640),
            cross_attention_dim=2048,
        )
        self.device = "cpu"
        self.dtype = "float32"
        self.set_calls = 0

    def set_attn_processor(self, processors) -> None:
        self.attn_processors = OrderedDict(processors)
        self.set_calls += 1


def _loaded_adapter(tmp_path: Path) -> PuLIDAdapter:
    adapter = PuLIDAdapter(
        tmp_path / "pulid.safetensors",
        models_root=tmp_path,
        device="cpu",
    )
    adapter._id_adapter = object()
    adapter._attention_state = {"1.id_to_k.weight": object()}
    adapter._runtime = SimpleNamespace(
        torch=SimpleNamespace(
            nn=SimpleNamespace(ModuleList=FakeModuleList),
            cat=lambda values, dim: np.concatenate(values, axis=dim),
        ),
        attention=SimpleNamespace(
            AttnProcessor2_0=FakeProcessor,
            IDAttnProcessor2_0=FakeProcessor,
        ),
    )
    return adapter


def test_apply_injects_only_cross_attention_processors(tmp_path: Path) -> None:
    adapter = _loaded_adapter(tmp_path)
    pipeline = SimpleNamespace(unet=FakeUnet())

    assert adapter.apply(pipeline) is pipeline

    processors = list(pipeline.unet.attn_processors.values())
    assert processors[0].kwargs == {}
    assert processors[1].kwargs == {
        "hidden_size": 320,
        "cross_attention_dim": 2048,
    }
    assert adapter._attention_layers.loaded == (adapter._attention_state, True)
    assert pipeline.unet.set_calls == 1
    adapter.apply(pipeline)
    assert pipeline.unet.set_calls == 1


def test_identity_state_builds_cfg_batch_and_can_be_cleared(tmp_path: Path) -> None:
    adapter = _loaded_adapter(tmp_path)
    features = PuLIDIdentityFeatures(
        conditional=np.ones((1, 32, 2048), dtype=np.float32),
        unconditional=np.zeros((1, 32, 2048), dtype=np.float32),
        source_images=("reference.webp",),
    )

    adapter.set_identity(features, strength=0.8)
    kwargs = adapter.cross_attention_kwargs()

    assert kwargs["id_embedding"].shape == (2, 32, 2048)
    assert kwargs["id_scale"] == 0.8
    adapter.clear_identity()
    assert adapter.cross_attention_kwargs() == {}


@pytest.mark.parametrize("strength", [-1.0, float("nan"), float("inf")])
def test_identity_strength_must_be_finite_and_non_negative(
    tmp_path: Path, strength: float
) -> None:
    adapter = _loaded_adapter(tmp_path)
    features = PuLIDIdentityFeatures(object(), object(), ("reference.webp",))

    with pytest.raises(PuLIDIdentityError, match="force d'identité"):
        adapter.set_identity(features, strength)
