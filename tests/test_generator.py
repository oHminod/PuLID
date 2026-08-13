from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from pulid_app.config import (
    AppConfig,
    DeviceConfig,
    InsightFaceConfig,
    PuLIDConfig,
    SDXLConfig,
)
from pulid_app.identity import CharacterIdentity
from pulid_app.pipeline.generator import (
    EncodedIdentity,
    ImageGenerator,
    ImageGeneratorError,
)


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        models_root=tmp_path / "models",
        sdxl=SDXLConfig(
            tmp_path / "models" / "sdxl.safetensors",
            tmp_path / "models" / "sdxl-config",
        ),
        pulid=PuLIDConfig(
            tmp_path / "models" / "pulid.safetensors",
            source_dir=tmp_path / "models" / "sources" / "PuLID",
            revision="a" * 40,
        ),
        insightface=InsightFaceConfig(tmp_path / "models", "antelopev2"),
        outputs_dir=tmp_path / "outputs",
        identity_cache_dir=tmp_path / "cache" / "identity",
        device=DeviceConfig(),
        source_path=tmp_path / "config.yaml",
    )


class FakeIdentityEncoder:
    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self.cache_calls: list[dict] = []
        self.encode_calls: list[dict] = []

    def cache_path_for(self, image: Path, **kwargs: object) -> Path:
        self.cache_calls.append({"image": image, **kwargs})
        return self.cache_path

    def encode_image(self, image: Path, **kwargs: object) -> CharacterIdentity:
        self.encode_calls.append({"image": image, **kwargs})
        return CharacterIdentity(
            id=str(kwargs["identity_id"]),
            source_images=[str(image)],
            face_embedding=np.ones(512, dtype=np.float32),
            metadata={"source_format": "WEBP"},
        )


class FakeAdapter:
    def __init__(self) -> None:
        self.conditioning = SimpleNamespace(tokens="pulid")
        self.prepare_calls: list[dict] = []
        self.applied_to: list[object] = []
        self.identity_calls: list[tuple[object, float]] = []
        self.clear_calls = 0
        self.close_calls = 0

    def prepare_identity(self, image: Path, **kwargs: object) -> object:
        self.prepare_calls.append({"image": image, **kwargs})
        return self.conditioning

    def apply(self, pipeline: object) -> object:
        self.applied_to.append(pipeline)
        return pipeline

    def set_identity(self, identity: object, strength: float) -> None:
        self.identity_calls.append((identity, strength))

    def cross_attention_kwargs(self, **kwargs: object) -> dict[str, object]:
        return {"id_embedding": "tokens", "cfg": kwargs["classifier_free_guidance"]}

    def clear_identity(self) -> None:
        self.clear_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class FakeSDXL:
    def __init__(self) -> None:
        self.pipeline = SimpleNamespace(name="pipeline")
        self.load_calls = 0
        self.sampling_calls: list[str | None] = []
        self.generate_calls: list[dict] = []
        self.close_calls = 0

    def load(self) -> "FakeSDXL":
        self.load_calls += 1
        return self

    def set_sampling_method(self, method: str | None) -> "FakeSDXL":
        self.sampling_calls.append(method)
        return self

    def generate(self, **kwargs: object) -> object:
        self.generate_calls.append(kwargs)
        return SimpleNamespace(
            image=Image.new("RGB", (int(kwargs["width"]), int(kwargs["height"]))),
            seed=kwargs["seed"],
            device="cpu",
            dtype="float32",
            dtype_fallback_used=False,
            duration_seconds=0.25,
            stage_durations_seconds={
                "prompt_preparation": 0.01,
                "diffusion": 0.2,
                "vae": 0.04,
            },
        )

    def close(self) -> None:
        self.close_calls += 1


def _generator_with_fakes(
    tmp_path: Path,
) -> tuple[ImageGenerator, FakeIdentityEncoder, FakeAdapter, FakeSDXL]:
    encoder = FakeIdentityEncoder(tmp_path / "cache" / "identity" / "noemie.npz")
    adapter = FakeAdapter()
    sdxl = FakeSDXL()
    generator = ImageGenerator(
        _config(tmp_path),
        device="cpu",
        identity_encoder=encoder,
        adapter=adapter,
        sdxl=sdxl,
    )
    return generator, encoder, adapter, sdxl


def test_constructor_keeps_models_lazy(tmp_path: Path) -> None:
    generator = ImageGenerator(_config(tmp_path), device="cpu")

    assert generator._identity_encoder is None
    assert generator._adapter is None
    assert generator._sdxl is None


def test_encode_identity_reuses_generic_cache_and_prepares_pulid(tmp_path: Path) -> None:
    reference = tmp_path / "noemie.webp"
    Image.new("RGB", (16, 16)).save(reference, format="WEBP")
    generator, encoder, adapter, _sdxl = _generator_with_fakes(tmp_path)
    encoder.cache_path.parent.mkdir(parents=True)
    encoder.cache_path.touch()

    identity = generator.encode_identity(reference, identity_id="noemie")

    assert identity.id == "noemie"
    assert identity.cache_hit is True
    assert identity.cache_path == encoder.cache_path
    assert adapter.prepare_calls[0]["face_embedding"].shape == (512,)
    assert identity.conditioning is adapter.conditioning


def test_generate_saves_png_json_and_forwards_effective_parameters(
    tmp_path: Path,
) -> None:
    generator, _encoder, adapter, sdxl = _generator_with_fakes(tmp_path)
    reference = tmp_path / "noemie.webp"
    character = CharacterIdentity(
        id="noemie",
        source_images=[str(reference)],
        face_embedding=np.ones(512, dtype=np.float32),
    )
    identity = EncodedIdentity(
        character=character,
        conditioning=adapter.conditioning,
        cache_path=tmp_path / "cache" / "noemie.npz",
        cache_hit=True,
        duration_seconds=1.5,
    )

    generated = generator.generate(
        prompt="portrait",
        identity=identity,
        seed=7,
        width=64,
        height=64,
        steps=3,
        identity_strength=0.9,
        guidance_scale=4.5,
        sampling_method="dpmpp_2m_sde_karras",
    )

    assert generated.png_path.is_file()
    assert generated.json_path.is_file()
    assert generated.png_path.stem == generated.json_path.stem
    assert generated.image.size == (64, 64)
    assert sdxl.sampling_calls == ["dpmpp_2m_sde_karras"]
    assert sdxl.generate_calls[0]["seed"] == 7
    assert sdxl.generate_calls[0]["guidance_scale"] == 4.5
    assert sdxl.generate_calls[0]["cross_attention_kwargs"]["cfg"] is True
    assert adapter.identity_calls == [(adapter.conditioning, 0.9)]
    assert adapter.clear_calls == 1

    metadata = json.loads(generated.json_path.read_text(encoding="utf-8"))
    assert metadata["identity_id"] == "noemie"
    assert metadata["identity_cache_hit"] is True
    assert metadata["seed"] == 7
    assert metadata["sampling_method"] == "dpmpp_2m_sde_karras"
    assert metadata["sdxl_checkpoint"] == str(generator.config.sdxl.checkpoint)
    assert metadata["vae"] == "integrated"
    assert metadata["offload_strategy"] == "none"
    assert metadata["prompt_preparation_duration_seconds"] == 0.01
    assert metadata["diffusion_duration_seconds"] == 0.2
    assert metadata["vae_duration_seconds"] == 0.04
    assert metadata["save_duration_seconds"] >= 0.0
    assert generated.save_duration_seconds >= 0.0


def test_generate_rejects_identity_from_another_api(tmp_path: Path) -> None:
    generator, _encoder, _adapter, _sdxl = _generator_with_fakes(tmp_path)

    with pytest.raises(ImageGeneratorError, match="encode_identity"):
        generator.generate(prompt="portrait", identity=object())  # type: ignore[arg-type]


def test_close_is_idempotent(tmp_path: Path) -> None:
    generator, _encoder, adapter, sdxl = _generator_with_fakes(tmp_path)

    generator.close()
    generator.close()

    assert adapter.close_calls == 1
    assert sdxl.close_calls == 1
