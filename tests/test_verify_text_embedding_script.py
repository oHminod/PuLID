from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pulid_app.config import TextEmbeddingConfig
from scripts import verify_text_embedding


def test_verify_text_embedding_runs_real_service_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    embedding_config = TextEmbeddingConfig(
        checkpoint=tmp_path / "bge-m3-Q8_0.gguf",
        dimensions=3,
    )
    config = SimpleNamespace(
        models_root=tmp_path,
        text_embedding=embedding_config,
    )
    captured: dict[str, object] = {}

    class FakeService:
        def __init__(self, selected_config, *, device: str) -> None:
            captured["config"] = selected_config
            captured["device"] = device

        def create_embedding(self, *, model: str, input_value: str):
            captured["model"] = model
            captured["input"] = input_value
            return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(verify_text_embedding, "load_config", lambda _path: config)
    monkeypatch.setattr(
        verify_text_embedding,
        "configure_external_model_caches",
        lambda models_root: captured.setdefault("models_root", models_root),
    )
    monkeypatch.setattr(verify_text_embedding, "TextEmbeddingService", FakeService)

    assert verify_text_embedding.main(["--device", "cuda"]) == 0
    assert captured == {
        "models_root": tmp_path,
        "config": embedding_config,
        "device": "cuda",
        "model": embedding_config.model_id,
        "input": "test",
        "closed": True,
    }


def test_verify_text_embedding_rejects_wrong_dimensions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    embedding_config = TextEmbeddingConfig(
        checkpoint=tmp_path / "bge-m3-Q8_0.gguf",
        dimensions=3,
    )
    config = SimpleNamespace(
        models_root=tmp_path,
        text_embedding=embedding_config,
    )

    class WrongDimensionService:
        def __init__(self, _config, *, device: str) -> None:
            pass

        def create_embedding(self, *, model: str, input_value: str):
            return {"data": [{"embedding": [0.1, 0.2]}]}

        def close(self) -> None:
            pass

    monkeypatch.setattr(verify_text_embedding, "load_config", lambda _path: config)
    monkeypatch.setattr(
        verify_text_embedding,
        "configure_external_model_caches",
        lambda _models_root: None,
    )
    monkeypatch.setattr(
        verify_text_embedding,
        "TextEmbeddingService",
        WrongDimensionService,
    )

    assert verify_text_embedding.main(["--device", "cuda"]) == 1
