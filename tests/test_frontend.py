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

    assert directory == Path(__file__).parents[1] / "frontend"
    assert (directory / "index.html").is_file()


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
