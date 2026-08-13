"""Catégorisation commune des tests du projet."""

from __future__ import annotations

import pytest


EXPLICIT_CATEGORIES = frozenset({"integration", "slow", "gpu"})


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Classe automatiquement les tests isolés existants dans ``unit``."""

    for item in items:
        marker_names = {marker.name for marker in item.iter_markers()}
        if marker_names.isdisjoint(EXPLICIT_CATEGORIES):
            item.add_marker(pytest.mark.unit)
