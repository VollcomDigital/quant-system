"""Phase 0 Exit Criteria - aggregate gate.

Every Phase 0 exit criterion from `tasks/todo.md` must be verifiable from
the file system and the design package. This file collapses the criteria
into a single explicit check so the Phase 0 -> Phase 1 transition cannot
regress later.
"""

from __future__ import annotations

import re
from pathlib import Path

DELIVERABLES_DOCS = (
    Path("docs") / "architecture" / "phase-0-scaffold.md",
    Path("docs") / "architecture" / "package-and-import-conventions.md",
    Path("docs") / "architecture" / "service-communication-standards.md",
    Path("docs") / "architecture" / "hft-latency-boundary.md",
    Path("docs") / "architecture" / "cli-compatibility-facade.md",
)


TOP_LEVEL_DIRS = (
    "shared_lib",
    "data_platform",
    "backtest_engine",
    "alpha_research",
    "ai_agents",
    "trading_system",
    "web_control_plane",
    "infrastructure",
)


# ---------------------------------------------------------------------------
# Exit criterion 1: Top-level monorepo directories created or approved for
# creation.
# ---------------------------------------------------------------------------


def test_exit_top_level_directories_exist(repo_root: Path) -> None:
    missing = [d for d in TOP_LEVEL_DIRS if not (repo_root / d).is_dir()]
    assert not missing, f"Phase 0 exit: missing top-level directories: {missing}"


# ---------------------------------------------------------------------------
# Exit criterion 2: Package/import naming conventions fixed in writing.
# ---------------------------------------------------------------------------


def test_exit_naming_conventions_documented(repo_root: Path) -> None:
    doc = repo_root / "docs" / "architecture" / "package-and-import-conventions.md"
    assert doc.is_file(), "Naming conventions doc must exist"
    text = doc.read_text(encoding="utf-8")
    assert "## Import Convention" in text and "## Forbidden Patterns" in text


# ---------------------------------------------------------------------------
# Exit criterion 3: Current CLI compatibility facade strategy documented.
# ---------------------------------------------------------------------------


def test_exit_cli_compatibility_documented(repo_root: Path) -> None:
    doc = repo_root / "docs" / "architecture" / "cli-compatibility-facade.md"
    assert doc.is_file(), "CLI compatibility doc must exist"
    text = doc.read_text(encoding="utf-8").lower()
    assert "phase 10" in text, "CLI compatibility doc must tie retirement to Phase 10"


# ---------------------------------------------------------------------------
# Exit criterion 4: ADR-0001 approved so Phase 1 extraction can begin.
# ---------------------------------------------------------------------------


def test_exit_adr_0001_accepted(repo_root: Path) -> None:
    adr = (repo_root / "docs" / "adr" / "0001-monorepo-workspace-and-package-boundaries.md").read_text(
        encoding="utf-8"
    )
    m = re.search(r"^[-*]\s*Status\s*:\s*Accepted", adr, re.MULTILINE)
    assert m is not None, "Phase 0 cannot exit without ADR-0001 Accepted"


# ---------------------------------------------------------------------------
# Exit criterion 5: all Phase 0 design-package artifacts exist.
# ---------------------------------------------------------------------------


def test_exit_design_package_complete(repo_root: Path) -> None:
    missing = [str(p) for p in DELIVERABLES_DOCS if not (repo_root / p).is_file()]
    assert not missing, f"Phase 0 design package missing: {missing}"


# ---------------------------------------------------------------------------
# Exit criterion 6: no Phase 1 extraction has started. shared_lib must be
# a scaffold only: no module should carry runtime implementation yet.
# We enforce this weakly by checking every .py under shared_lib/src is
# either empty or only contains comments/docstrings/`from __future__`.
# ---------------------------------------------------------------------------


def test_exit_shared_lib_is_scaffold_only(repo_root: Path) -> None:
    root = repo_root / "shared_lib" / "src" / "shared_lib"
    non_empty: list[str] = []
    for py in root.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        # Strip comments, docstrings, and `from __future__` imports; anything
        # left is real runtime code, which is not allowed in Phase 0.
        stripped = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
        stripped = re.sub(r"'''.*?'''", "", stripped, flags=re.DOTALL)
        stripped = re.sub(r"#.*", "", stripped)
        stripped = re.sub(r"from\s+__future__\s+import.*", "", stripped)
        stripped = stripped.strip()
        if stripped:
            non_empty.append(str(py.relative_to(repo_root)))
    assert not non_empty, (
        f"Phase 0 exit: shared_lib must be scaffold-only; found content in: {non_empty}"
    )
