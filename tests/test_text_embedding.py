from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from pulid_app.config import TextEmbeddingConfig
from pulid_app.models.text_embedding import (
    TextEmbeddingService,
    load_llama_cpp_embedding_model,
)


def test_llama_cpp_factory_forces_cpu_and_local_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "bge-m3-Q8_0.gguf"
    checkpoint.touch()
    captured = {}
    sentinel = object()

    def fake_llama(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=fake_llama))
    config = TextEmbeddingConfig(
        checkpoint=checkpoint,
        context_size=4096,
        batch_size=4096,
        threads=2,
    )

    result = load_llama_cpp_embedding_model(config)

    assert result is sentinel
    assert captured == {
        "model_path": str(checkpoint),
        "embedding": True,
        "n_gpu_layers": 0,
        "n_ctx": 4096,
        "n_batch": 4096,
        "n_ubatch": 4096,
        "n_threads": 2,
        "n_threads_batch": 2,
        "offload_kqv": False,
        "op_offload": False,
        "flash_attn": False,
        "use_mmap": True,
        "verbose": False,
    }


def test_embedding_service_rejects_token_overflow_without_truncating(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "bge-m3-Q8_0.gguf"
    checkpoint.touch()

    class TokenOverflowEngine:
        called = False

        def tokenize(self, _content):
            return list(range(129))

        def create_embedding(self, _inputs):
            self.called = True
            raise AssertionError("Le calcul ne doit pas démarrer.")

    engine = TokenOverflowEngine()
    service = TextEmbeddingService(
        TextEmbeddingConfig(checkpoint=checkpoint, context_size=128),
        model_factory=lambda _config: engine,
    )

    with pytest.raises(ValueError, match="fenêtre configurée est de 128 jetons"):
        service.create_embedding(
            model="text-embedding-bge-m3",
            input_value="texte trop long",
        )

    assert engine.called is False
