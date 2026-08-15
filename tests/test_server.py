from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import httpx
from PIL import Image
import pytest

from pulid_app.exceptions import PromptTooLongError
from pulid_app.server import (
    create_app,
    generated_filename,
    resolve_generation_seed,
)


def _write_config(tmp_path: Path) -> tuple[Path, Path]:
    models = tmp_path / "models"
    checkpoints = models / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "realvisxl.safetensors").touch()
    (checkpoints / "reaxl_v30.safetensors").touch()
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
models_root: {models}
sdxl:
  checkpoint: checkpoints/realvisxl.safetensors
  config_dir: sdxl/config
pulid:
  checkpoint: pulid_v1.1.safetensors
insightface:
  model_root: .
  model_name: antelopev2
outputs_dir: {tmp_path / 'outputs'}
identity_cache_dir: {tmp_path / 'cache' / 'identity'}
device:
  preferred: cpu
  dtype: float32
  offload_strategy: none
""",
        encoding="utf-8",
    )
    return config, models


def _image_bytes(format_: str = "PNG") -> bytes:
    output = BytesIO()
    Image.new("RGB", (24, 24), "white").save(output, format=format_)
    return output.getvalue()


class FakeMemoryGenerator:
    instances: list["FakeMemoryGenerator"] = []

    def __init__(self, config, **kwargs) -> None:
        self.config = config
        self.constructor_kwargs = kwargs
        self.encode_kwargs = None
        self.generate_kwargs = None
        self.closed = False
        self.__class__.instances.append(self)

    def encode_identity_memory(self, image, **kwargs):
        self.encode_kwargs = {"image": image, **kwargs}
        return object()

    def generate_in_memory(self, **kwargs):
        self.generate_kwargs = kwargs
        return SimpleNamespace(
            image=Image.new("RGB", (32, 32), "navy"),
            metadata={},
        )

    def close(self) -> None:
        self.closed = True


def _app(tmp_path: Path, *, cors_origins: tuple[str, ...] = ()):
    config, _models = _write_config(tmp_path)
    FakeMemoryGenerator.instances.clear()
    return create_app(
        config,
        generator_factory=FakeMemoryGenerator,
        now_factory=lambda: datetime(2026, 8, 13, 20, 9, 47, 123456, tzinfo=timezone.utc),
        random_seed=lambda: 987654321,
        cors_origins=cors_origins,
    )


def _request(app, method: str, path: str, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_models_endpoint_lists_sampling_methods_and_sigmas_separately(
    tmp_path: Path,
) -> None:
    response = _request(_app(tmp_path), "GET", "/models")

    assert response.status_code == 200
    assert response.json() == {
        "models": [
            {
                "name": "realvisxl",
                "filename": "realvisxl.safetensors",
                "default": True,
            },
            {
                "name": "reaxl_v30",
                "filename": "reaxl_v30.safetensors",
                "default": False,
            },
        ],
        "sampling_methods": [
            {
                "name": "default",
                "label": "Scheduler du checkpoint",
                "default": True,
                "supported_sigma_schedules": ["normal"],
            },
            {
                "name": "dpmpp_2m",
                "label": "DPM++ 2M",
                "default": False,
                "supported_sigma_schedules": [
                    "normal",
                    "karras",
                    "exponential",
                    "beta",
                ],
            },
            {
                "name": "dpmpp_2m_sde",
                "label": "DPM++ 2M SDE",
                "default": False,
                "supported_sigma_schedules": [
                    "normal",
                    "karras",
                    "exponential",
                    "beta",
                ],
            },
            {
                "name": "dpmpp_3m_sde",
                "label": "DPM++ 3M SDE",
                "default": False,
                "supported_sigma_schedules": [
                    "normal",
                    "karras",
                    "exponential",
                    "beta",
                ],
            },
            {
                "name": "euler",
                "label": "Euler",
                "default": False,
                "supported_sigma_schedules": [
                    "normal",
                    "karras",
                    "exponential",
                    "beta",
                ],
            },
            {
                "name": "euler_ancestral",
                "label": "Euler ancestral",
                "default": False,
                "supported_sigma_schedules": ["normal"],
            },
            {
                "name": "heun",
                "label": "Heun",
                "default": False,
                "supported_sigma_schedules": [
                    "normal",
                    "karras",
                    "exponential",
                    "beta",
                ],
            },
            {
                "name": "lms",
                "label": "LMS",
                "default": False,
                "supported_sigma_schedules": [
                    "normal",
                    "karras",
                    "exponential",
                    "beta",
                ],
            },
            {
                "name": "ddim",
                "label": "DDIM",
                "default": False,
                "supported_sigma_schedules": ["normal"],
            },
        ],
        "sigma_schedules": [
            {
                "name": "normal",
                "label": "Normal / natif",
                "default": True,
                "supported_sampling_methods": [
                    "default",
                    "dpmpp_2m",
                    "dpmpp_2m_sde",
                    "dpmpp_3m_sde",
                    "euler",
                    "euler_ancestral",
                    "heun",
                    "lms",
                    "ddim",
                ],
            },
            {
                "name": "karras",
                "label": "Karras",
                "default": False,
                "supported_sampling_methods": [
                    "dpmpp_2m",
                    "dpmpp_2m_sde",
                    "dpmpp_3m_sde",
                    "euler",
                    "heun",
                    "lms",
                ],
            },
            {
                "name": "exponential",
                "label": "Exponentiel",
                "default": False,
                "supported_sampling_methods": [
                    "dpmpp_2m",
                    "dpmpp_2m_sde",
                    "dpmpp_3m_sde",
                    "euler",
                    "heun",
                    "lms",
                ],
            },
            {
                "name": "beta",
                "label": "Beta",
                "default": False,
                "supported_sampling_methods": [
                    "dpmpp_2m",
                    "dpmpp_2m_sde",
                    "dpmpp_3m_sde",
                    "euler",
                    "heun",
                    "lms",
                ],
            },
        ],
    }


def test_generate_returns_png_headers_and_enables_identity_cache(tmp_path: Path) -> None:
    app = _app(tmp_path)
    files_before = {path for path in tmp_path.rglob("*") if path.is_file()}

    response = _request(
        app,
        "POST",
        "/generate",
        files={"reference": ("noemie.webp", _image_bytes("WEBP"), "image/webp")},
        data={
            "character": "Noémie",
            "prompt": "cinematic portrait",
            "negative_prompt": "bad anatomy, watermark",
            "clip_skip_2": "true",
            "model": "reaxl_v30",
            "cfg": "4.5",
            "steps": "8",
            "strength": "1.25",
            "method": "dpmpp_2m_sde",
            "sigmas": "karras",
            "seed": "0",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-generation-seed"] == "987654321"
    assert response.headers["x-sdxl-model"] == "reaxl_v30"
    assert response.headers["x-sampling-method"] == "dpmpp_2m_sde"
    assert response.headers["x-sigma-schedule"] == "karras"
    assert response.headers["content-disposition"] == (
        'attachment; filename="noemie_20260813T200947_123456Z.png"'
    )
    with Image.open(BytesIO(response.content)) as generated:
        assert generated.size == (32, 32)

    instance = FakeMemoryGenerator.instances[0]
    assert instance.config.sdxl.checkpoint == (
        tmp_path / "models" / "checkpoints" / "reaxl_v30.safetensors"
    )
    assert instance.encode_kwargs["identity_id"] == "Noémie"
    assert instance.encode_kwargs["source_name"] == "<http-upload>"
    assert instance.encode_kwargs["use_cache"] is True
    assert instance.generate_kwargs["seed"] == 987654321
    assert instance.generate_kwargs["negative_prompt"] == "bad anatomy, watermark"
    assert instance.generate_kwargs["clip_skip_2"] is True
    assert instance.generate_kwargs["steps"] == 8
    assert instance.generate_kwargs["guidance_scale"] == 4.5
    assert instance.generate_kwargs["identity_strength"] == 1.25
    assert instance.generate_kwargs["sampling_method"] == "dpmpp_2m_sde"
    assert instance.generate_kwargs["sigma_schedule"] == "karras"
    assert instance.closed is True
    assert {path for path in tmp_path.rglob("*") if path.is_file()} == files_before
    assert not (tmp_path / "outputs").exists()


def test_generate_accepts_default_method_and_explicit_seed(tmp_path: Path) -> None:
    response = _request(
        _app(tmp_path),
        "POST",
        "/generate",
        files={"reference": ("noemie.png", _image_bytes(), "image/png")},
        data={
            "character": "noemie",
            "prompt": "portrait",
            "model": "realvisxl",
            "seed": "42",
        },
    )

    assert response.status_code == 200
    assert response.headers["x-generation-seed"] == "42"
    assert response.headers["x-sampling-method"] == "default"
    assert response.headers["x-sigma-schedule"] == "normal"
    assert FakeMemoryGenerator.instances[0].generate_kwargs["sampling_method"] is None
    assert FakeMemoryGenerator.instances[0].generate_kwargs["sigma_schedule"] is None
    assert FakeMemoryGenerator.instances[0].generate_kwargs["identity_strength"] == 0.8
    assert (
        FakeMemoryGenerator.instances[0].generate_kwargs["negative_prompt"]
        == "flaws in the eyes, flaws in the face, low quality, worst quality, "
        "artifacts, text, watermark, deformed, mutated, disfigured, blurry"
    )
    assert FakeMemoryGenerator.instances[0].generate_kwargs["clip_skip_2"] is False


@pytest.mark.parametrize("strength", ["-0.1", "nan", "inf"])
def test_generate_rejects_invalid_strength_before_loading_generator(
    tmp_path: Path,
    strength: str,
) -> None:
    response = _request(
        _app(tmp_path),
        "POST",
        "/generate",
        files={"reference": ("noemie.png", _image_bytes(), "image/png")},
        data={
            "character": "noemie",
            "prompt": "portrait",
            "model": "realvisxl",
            "strength": strength,
        },
    )

    assert response.status_code == 422
    assert FakeMemoryGenerator.instances == []


def test_generate_rejects_unknown_model_without_loading_generator(tmp_path: Path) -> None:
    response = _request(
        _app(tmp_path),
        "POST",
        "/generate",
        files={"reference": ("noemie.png", _image_bytes(), "image/png")},
        data={
            "character": "noemie",
            "prompt": "portrait",
            "model": "absent",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "ModelNotFoundError"
    assert FakeMemoryGenerator.instances == []


def test_generate_reports_long_prompt_as_a_422_client_error(tmp_path: Path) -> None:
    class LongPromptRejectingGenerator(FakeMemoryGenerator):
        def generate_in_memory(self, **kwargs):
            self.generate_kwargs = kwargs
            raise PromptTooLongError(
                prompt_kind="prompt positif",
                token_count=256,
                max_tokens=255,
                encoder_index=1,
            )

    config, _models = _write_config(tmp_path)
    app = create_app(config, generator_factory=LongPromptRejectingGenerator)

    response = _request(
        app,
        "POST",
        "/generate",
        files={"reference": ("noemie.png", _image_bytes(), "image/png")},
        data={
            "character": "noemie",
            "prompt": "very long prompt",
            "model": "realvisxl",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "PromptTooLongError"
    assert "256 jetons utiles" in response.json()["detail"]["message"]
    assert LongPromptRejectingGenerator.instances[-1].closed is True


def test_generate_rejects_invalid_image_sampling_method_and_sigmas(
    tmp_path: Path,
) -> None:
    app = _app(tmp_path)
    invalid_image = _request(
        app,
        "POST",
        "/generate",
        files={"reference": ("bad.png", b"not-an-image", "image/png")},
        data={
            "character": "noemie",
            "prompt": "portrait",
            "model": "realvisxl",
        },
    )
    invalid_method = _request(
        app,
        "POST",
        "/generate",
        files={"reference": ("noemie.png", _image_bytes(), "image/png")},
        data={
            "character": "noemie",
            "prompt": "portrait",
            "model": "realvisxl",
            "method": "unknown",
        },
    )
    invalid_sigmas = _request(
        app,
        "POST",
        "/generate",
        files={"reference": ("noemie.png", _image_bytes(), "image/png")},
        data={
            "character": "noemie",
            "prompt": "portrait",
            "model": "realvisxl",
            "method": "euler",
            "sigmas": "unknown",
        },
    )
    incompatible_sigmas = _request(
        app,
        "POST",
        "/generate",
        files={"reference": ("noemie.png", _image_bytes(), "image/png")},
        data={
            "character": "noemie",
            "prompt": "portrait",
            "model": "realvisxl",
            "method": "euler_ancestral",
            "sigmas": "karras",
        },
    )

    assert invalid_image.status_code == 422
    assert invalid_method.status_code == 422
    assert invalid_sigmas.status_code == 422
    assert incompatible_sigmas.status_code == 422
    assert "Méthode de sampling inconnue" in invalid_method.json()["detail"]["message"]
    assert "Courbe de sigmas inconnue" in invalid_sigmas.json()["detail"]["message"]
    assert "incompatible" in incompatible_sigmas.json()["detail"]["message"]
    assert FakeMemoryGenerator.instances == []


def test_seed_and_filename_helpers_are_deterministic() -> None:
    created = datetime(2026, 1, 2, 3, 4, 5, 6, tzinfo=timezone.utc)

    assert resolve_generation_seed(0, lambda: 123) == 123
    assert resolve_generation_seed(-1, lambda: 456) == 456
    assert resolve_generation_seed(42) == 42
    assert generated_filename("Noémie Test", created) == (
        "noemie-test_20260102T030405_000006Z.png"
    )


def test_configured_cors_origin_can_call_frontend_endpoints(tmp_path: Path) -> None:
    app = _app(tmp_path, cors_origins=("http://localhost:3000",))

    response = _request(
        app,
        "OPTIONS",
        "/generate",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "POST" in response.headers["access-control-allow-methods"]
