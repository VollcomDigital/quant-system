"""Phase 1 shared fixtures.

These tests exercise the new `shared_lib.*` packages directly. They do not
import `src/` code, consistent with the Phase 0 compatibility facade rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make every domain package importable via its canonical prefix during tests.
# ADR-0001 says domain packages use a `src/<domain>/` layout; we add each
# `<domain>/src` to sys.path so `import shared_lib` resolves.
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
