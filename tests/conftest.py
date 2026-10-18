"""Shared fixtures.

Unit tests use small synthetic inputs and never touch the network. The
regression tests (marked `slow`) re-run whole studies into a temporary
directory and compare the result with the published files in outputs/.
Studies that need the downloaded data are skipped when data/ is empty.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from betai import common, s4_house_limits, s8_synthesis

PUBLISHED = common.OUT


def has_football_data() -> bool:
    return (common.DATA / "football").exists()


def has_french_data() -> bool:
    return (common.DATA / "french").exists()


needs_football = pytest.mark.skipif(
    not has_football_data(), reason="football data not downloaded (run ./run_all.sh once)"
)
needs_french = pytest.mark.skipif(not has_french_data(), reason="French data not downloaded (run ./run_all.sh once)")


@pytest.fixture
def isolated_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every study's output directory to a temporary one."""
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setattr(common, "OUT", out)
    monkeypatch.setattr(common, "FIG", out / "figures")
    # Modules that read other studies' JSON imported OUT by name.
    monkeypatch.setattr(s4_house_limits, "OUT", out)
    monkeypatch.setattr(s8_synthesis, "OUT", out)
    return out


def copy_published(names: list[str], dest: Path) -> None:
    for name in names:
        shutil.copy(PUBLISHED / f"{name}.json", dest / f"{name}.json")


def assert_same_as_published(name: str, out: Path) -> None:
    for ext in ("json", "md"):
        new = (out / f"{name}.{ext}").read_text(encoding="utf-8")
        old = (PUBLISHED / f"{name}.{ext}").read_text(encoding="utf-8")
        assert new == old, f"{name}.{ext} differs from the published version"
