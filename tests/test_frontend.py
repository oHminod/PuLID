from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

from frontend.server import build_server, frontend_directory


class _FormFieldParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.fields: dict[str, dict[str, str | None]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in {"input", "select", "textarea"}:
            return
        attributes = dict(attrs)
        name = attributes.get("name")
        if name:
            self.fields[name] = attributes


def test_frontend_directory_is_resolved_from_server_module() -> None:
    directory = frontend_directory()
    html = (directory / "index.html").read_text(encoding="utf-8")

    assert directory == Path(__file__).parents[1] / "frontend"
    assert (directory / "index.html").is_file()
    assert (directory / "storage.js").is_file()
    assert html.index('src="storage.js"') < html.index('src="app.js"')


def test_frontend_marks_every_backend_required_field_as_required() -> None:
    parser = _FormFieldParser()
    parser.feed((frontend_directory() / "index.html").read_text(encoding="utf-8"))

    for name in ("reference", "character", "prompt", "model"):
        assert "required" in parser.fields[name]


def test_frontend_client_sends_every_generation_parameter() -> None:
    source = (frontend_directory() / "app.js").read_text(encoding="utf-8")
    direct_fields = {
        "reference",
        "character",
        "prompt",
        "negative_prompt",
        "clip_skip_2",
        "model",
        "method",
        "sigmas",
    }

    assert all(f'form.append("{name}"' in source for name in direct_fields)
    assert all(f"{name}:" in source for name in ("cfg", "steps", "strength", "seed"))
    assert "form.append(name, value)" in source


def test_frontend_persists_settings_reference_and_last_result_locally() -> None:
    app_source = (frontend_directory() / "app.js").read_text(encoding="utf-8")
    storage_source = (frontend_directory() / "storage.js").read_text(encoding="utf-8")

    assert "localStorageBackend?.setItem" in storage_source
    assert 'const LAST_REFERENCE_KEY = "last-reference"' in storage_source
    assert 'const LAST_RESULT_KEY = "last-result"' in storage_source
    assert "indexedDbBackend.open" in storage_source
    assert "saveReference" in app_source
    assert "restoreReference" in app_source
    assert "saveLastResult" in app_source
    assert "restoreLastResult" in app_source


def test_clearing_only_reference_preserves_settings_and_result() -> None:
    app_source = (frontend_directory() / "app.js").read_text(encoding="utf-8")
    storage_source = (frontend_directory() / "storage.js").read_text(encoding="utf-8")

    clear_reference_block = storage_source.split("function clearReference()", 1)[1].split(
        "}", 1
    )[0]

    assert "storage.clearReference()" in app_source
    assert "deleteArtifact(LAST_REFERENCE_KEY)" in clear_reference_block
    assert "SETTINGS_KEY" not in clear_reference_block
    assert "LAST_RESULT_KEY" not in clear_reference_block


def test_build_server_configures_static_root_and_backend(monkeypatch) -> None:
    captured = {}

    class FakeHttpServer:
        def __init__(self, address, handler) -> None:
            captured["address"] = address
            captured["handler"] = handler

    monkeypatch.setattr("frontend.server.ThreadingHTTPServer", FakeHttpServer)

    server = build_server(port=8888, backend_url="http://127.0.0.1:12693")

    assert isinstance(server, FakeHttpServer)
    assert captured["address"] == ("127.0.0.1", 8888)
    assert captured["handler"].keywords["directory"] == str(frontend_directory())
    assert captured["handler"].keywords["backend"].geturl() == (
        "http://127.0.0.1:12693"
    )
