"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def font_dir(tmp_path: Path) -> Path:
    """A directory for synthetic fonts, created lazily by the builders."""
    return tmp_path / "fonts"
