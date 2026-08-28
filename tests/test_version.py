from __future__ import annotations

from importlib import metadata

import pulid_app


def test_version_is_read_from_distribution_metadata(monkeypatch) -> None:
    requested_distributions: list[str] = []

    def resolve_version(distribution: str) -> str:
        requested_distributions.append(distribution)
        return "9.8.7"

    monkeypatch.setattr(pulid_app.metadata, "version", resolve_version)

    assert pulid_app._resolve_version() == "9.8.7"
    assert requested_distributions == ["pulid-app"]


def test_version_has_explicit_fallback_when_distribution_is_missing(monkeypatch) -> None:
    def missing_distribution(distribution: str) -> str:
        raise metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(pulid_app.metadata, "version", missing_distribution)

    assert pulid_app._resolve_version() == "0+unknown"
