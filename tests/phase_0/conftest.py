"""Phase 0 scaffolding tests.

These tests enforce architectural invariants only: ADR state, directory
presence, naming conventions, and compatibility facade documentation.

They intentionally do not import the legacy `src/` application so they can
run in isolation on a minimal toolchain (no pandas/numpy required).
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    # tests/phase_0/conftest.py -> repo root is two parents up.
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def adr_dir(repo_root: Path) -> Path:
    return repo_root / "docs" / "adr"


@pytest.fixture(scope="session")
def architecture_dir(repo_root: Path) -> Path:
    return repo_root / "docs" / "architecture"
