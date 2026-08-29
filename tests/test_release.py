from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
import tomllib

from scripts.build_release import MODEL_SUFFIXES, build_release


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_release_archive_is_deterministic_installable_source_without_local_data(
    tmp_path: Path,
) -> None:
    first_archive, first_checksums = build_release(PROJECT_ROOT, tmp_path / "first")
    second_archive, second_checksums = build_release(PROJECT_ROOT, tmp_path / "second")

    assert first_archive.read_bytes() == second_archive.read_bytes()
    assert first_checksums.read_text(encoding="ascii") == second_checksums.read_text(
        encoding="ascii"
    )

    expected_digest = hashlib.sha256(first_archive.read_bytes()).hexdigest()
    assert first_checksums.read_text(encoding="ascii") == (
        f"{expected_digest}  {first_archive.name}\n"
    )

    pyproject = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    version = pyproject["project"]["version"]
    prefix = f"pulid-{version}"
    assert first_archive.name == f"{prefix}.tar.gz"

    with tarfile.open(first_archive, "r:gz") as archive:
        members = archive.getmembers()
        names = {member.name for member in members}
        metadata = json.load(archive.extractfile(f"{prefix}/release-metadata.json"))

    assert metadata == {
        "apiContractVersion": "1.0.0",
        "archiveFormatVersion": 1,
        "component": "pulid",
        "version": version,
    }
    assert f"{prefix}/pyproject.toml" in names
    assert f"{prefix}/RELEASE.md" in names
    assert f"{prefix}/install_macos.sh" in names
    assert f"{prefix}/install_windows.bat" in names
    assert f"{prefix}/install_production_macos.sh" in names
    assert f"{prefix}/install_production_windows.bat" in names
    assert not any("/.git/" in name or "/.venv/" in name for name in names)
    assert not any("/tests/" in name or "/scripts/test_" in name for name in names)
    assert not any(name.endswith("config/local.yaml") for name in names)
    assert not any(name.endswith("inputs/noemie.webp") for name in names)
    assert not any(name.endswith(".DS_Store") for name in names)
    assert not any(Path(name).suffix.casefold() in MODEL_SUFFIXES for name in names)
    assert all(
        member.uid == 0 and member.gid == 0 and member.mtime == 0
        for member in members
    )
