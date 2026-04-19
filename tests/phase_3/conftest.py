"""Phase 3 shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for domain in (
    "shared_lib",
    "data_platform",
    "backtest_engine",
    "alpha_research",
    "ai_agents",
    "trading_system",
):
    candidate = REPO_ROOT / domain / "src"
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT
