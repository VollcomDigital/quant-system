"""Phase 0 Task 2 - Monorepo skeleton directory invariants.

Verifies that the top-level monorepo directories, package layouts, and
minimal marker files exist so Phase 1 extraction work can begin against
stable package roots.

These tests are intentionally structural and filesystem-only. They do not
import or execute any domain code, which keeps them runnable on a minimal
toolchain and on CI runners that have not yet installed Phase 2+ deps.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Top-level domains required by docs/architecture/phase-0-scaffold.md and
# ADR-0001 "Decision".
# ---------------------------------------------------------------------------


PYTHON_DOMAINS = (
    "shared_lib",
    "data_platform",
    "backtest_engine",
    "alpha_research",
    "ai_agents",
    "trading_system",
)


TOP_LEVEL_DIRS = (
    *PYTHON_DOMAINS,
    "web_control_plane",
    "infrastructure",
)


# ---------------------------------------------------------------------------
# Edge case 1: every declared top-level domain must exist as a directory.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("top", TOP_LEVEL_DIRS)
def test_top_level_directory_exists(repo_root: Path, top: str) -> None:
    path = repo_root / top
    assert path.is_dir(), f"Top-level monorepo directory missing: {top}/"


# ---------------------------------------------------------------------------
# Edge case 2: Python domains must use a `src/<domain>/` layout with an
# `__init__.py` so absolute imports like `shared_lib.contracts` resolve.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain", PYTHON_DOMAINS)
def test_python_domain_has_src_layout(repo_root: Path, domain: str) -> None:
    init = repo_root / domain / "src" / domain / "__init__.py"
    assert init.is_file(), (
        f"{domain} must expose a src-layout package root at {init.relative_to(repo_root)}"
    )


# ---------------------------------------------------------------------------
# Edge case 3: Python domains must own a package-local pyproject.toml
# (ADR-0001 Decision).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain", PYTHON_DOMAINS)
def test_python_domain_has_local_manifest(repo_root: Path, domain: str) -> None:
    manifest = repo_root / domain / "pyproject.toml"
    assert manifest.is_file(), (
        f"{domain} must ship a package-local pyproject.toml (ADR-0001)"
    )


# ---------------------------------------------------------------------------
# Edge case 4: every Python domain must carry a tests/ package root so
# package-scoped pytest discovery works from Phase 1 onward.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain", PYTHON_DOMAINS)
def test_python_domain_has_tests_dir(repo_root: Path, domain: str) -> None:
    tests_dir = repo_root / domain / "tests"
    assert tests_dir.is_dir(), f"{domain} must ship a tests/ directory"


# ---------------------------------------------------------------------------
# Edge case 5: trading_system must host a native/ subtree so HFT code is
# physically separated from the Python workspace (ADR-0001 + ADR-0003).
# ---------------------------------------------------------------------------


def test_trading_system_native_tree_exists(repo_root: Path) -> None:
    native = repo_root / "trading_system" / "native"
    assert native.is_dir(), "trading_system/native/ must exist for HFT code"
    for leaf in ("hft_engine/core", "hft_engine/network", "hft_engine/fast_inference", "shared"):
        assert (native / leaf).is_dir(), (
            f"trading_system/native/{leaf}/ must exist"
        )


# ---------------------------------------------------------------------------
# Edge case 6: shared_lib must pre-carve the Phase 1 sub-packages so the
# Phase 0 -> Phase 1 handoff does not require extra restructuring.
# ---------------------------------------------------------------------------


SHARED_LIB_SUBPACKAGES = (
    "contracts",
    "logging",
    "math_utils",
    "risk",
    "transport",
)


@pytest.mark.parametrize("sub", SHARED_LIB_SUBPACKAGES)
def test_shared_lib_has_phase_1_subpackages(repo_root: Path, sub: str) -> None:
    init = repo_root / "shared_lib" / "src" / "shared_lib" / sub / "__init__.py"
    assert init.is_file(), (
        f"shared_lib.{sub} must exist as an importable sub-package"
    )


# ---------------------------------------------------------------------------
# Edge case 7: legacy `src/` must be preserved (compatibility facade).
# ---------------------------------------------------------------------------


def test_legacy_src_is_preserved(repo_root: Path) -> None:
    legacy_main = repo_root / "src" / "main.py"
    assert legacy_main.is_file(), (
        "Legacy src/main.py must be preserved as the compatibility CLI facade"
    )


# ---------------------------------------------------------------------------
# Edge case 8: infrastructure/ must contain the Phase 9 sub-directories as
# empty markers so IaC layout is pre-declared.
# ---------------------------------------------------------------------------


def test_infrastructure_subdirs_exist(repo_root: Path) -> None:
    infra = repo_root / "infrastructure"
    for sub in ("terraform", "kubernetes", "runbooks"):
        assert (infra / sub).is_dir(), f"infrastructure/{sub}/ must exist"


# ---------------------------------------------------------------------------
# Edge case 9: web_control_plane/ must declare its backend/frontend split
# (Phase 5 and Phase 10) so the boundary is pre-declared.
# ---------------------------------------------------------------------------


def test_web_control_plane_has_backend_and_frontend(repo_root: Path) -> None:
    wcp = repo_root / "web_control_plane"
    assert (wcp / "backend").is_dir(), "web_control_plane/backend/ must exist"
    assert (wcp / "frontend").is_dir(), "web_control_plane/frontend/ must exist"
