"""Phase 0 Task 7 - CLI compatibility facade invariants.

Phase 0 must not break the existing CLI. This test locks in three things:

1. `src/main.py` still exists as the current user-facing CLI.
2. The compatibility strategy is documented in writing so future phases
   know when (and how) the legacy module may be retired.
3. No new domain package silently imports `src.*` - the compatibility
   facade is the only legal bridge to legacy code.
"""

from __future__ import annotations

import re
from pathlib import Path

STRATEGY_DOC = Path("docs") / "architecture" / "cli-compatibility-facade.md"


REQUIRED_SECTIONS = (
    "## Purpose",
    "## Current CLI Surface",
    "## Migration Strategy",
    "## Allowed Legacy Imports",
    "## Retirement Plan",
    "## Enforcement",
)


PYTHON_DOMAINS = (
    "shared_lib",
    "data_platform",
    "backtest_engine",
    "alpha_research",
    "ai_agents",
    "trading_system",
)


SRC_IMPORT_PATTERN = re.compile(
    r"^\s*(?:from|import)\s+src(?:\.|\s)", re.MULTILINE
)


# ---------------------------------------------------------------------------
# Edge case 1: legacy CLI entrypoint must still be present.
# ---------------------------------------------------------------------------


def test_legacy_cli_entrypoint_still_exists(repo_root: Path) -> None:
    main = repo_root / "src" / "main.py"
    assert main.is_file(), "src/main.py must remain on disk as the compatibility CLI"


# ---------------------------------------------------------------------------
# Edge case 2: pyproject.toml must still expose the src package so the
# CLI can be imported by its legacy path during migration.
# ---------------------------------------------------------------------------


def test_root_pyproject_still_includes_src_package(repo_root: Path) -> None:
    manifest = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'include = "src"' in manifest, (
        "Root pyproject.toml must retain the `src` package include until Phase 10 cutover"
    )


# ---------------------------------------------------------------------------
# Edge case 3: the compatibility strategy must be documented.
# ---------------------------------------------------------------------------


def test_compatibility_strategy_document_exists(repo_root: Path) -> None:
    doc = repo_root / STRATEGY_DOC
    assert doc.is_file(), f"Compatibility strategy doc missing at {STRATEGY_DOC}"


def test_compatibility_strategy_has_required_sections(repo_root: Path) -> None:
    doc = (repo_root / STRATEGY_DOC).read_text(encoding="utf-8")
    missing = [s for s in REQUIRED_SECTIONS if s not in doc]
    assert not missing, f"Compatibility doc missing sections: {missing}"


# ---------------------------------------------------------------------------
# Edge case 4: the doc must state a retirement trigger tied to Phase 10.
# ---------------------------------------------------------------------------


def test_compatibility_doc_ties_retirement_to_phase_10(repo_root: Path) -> None:
    doc = (repo_root / STRATEGY_DOC).read_text(encoding="utf-8").lower()
    assert "phase 10" in doc, (
        "Compatibility doc must tie retirement to Phase 10 cutover"
    )


# ---------------------------------------------------------------------------
# Edge case 5: no domain package may import `src.*`. Only `src/` itself
# and the test suite may reference the legacy prefix.
# ---------------------------------------------------------------------------


def test_no_domain_package_imports_src(repo_root: Path) -> None:
    offenders: list[str] = []
    for domain in PYTHON_DOMAINS:
        root = repo_root / domain / "src" / domain
        if not root.is_dir():
            continue
        for py in root.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            if SRC_IMPORT_PATTERN.search(text):
                offenders.append(str(py.relative_to(repo_root)))
    assert not offenders, (
        f"Domain packages must not import `src.*` directly: {offenders}"
    )


# ---------------------------------------------------------------------------
# Edge case 6: the legacy CLI file must still be runnable as a module target.
# We do a static check only - import-level verification would require the
# full legacy toolchain.
# ---------------------------------------------------------------------------


def test_legacy_cli_declares_entrypoint(repo_root: Path) -> None:
    main = (repo_root / "src" / "main.py").read_text(encoding="utf-8")
    assert "__main__" in main or "def main" in main or "app = " in main, (
        "src/main.py must continue to expose a CLI entrypoint"
    )
