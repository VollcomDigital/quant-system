"""Phase 3 Task 7 - notebook governance invariants.

Notebook rules (roadmap):

- Notebook code and production factor code live in separate packages.
- Notebooks under `alpha_research/notebooks/` must never be imported by
  any other domain package.
- The governance document explicitly documents the notebook execution
  + cleanup rules (how outputs are handled in CI, how secrets are
  scrubbed, what graduates a cell to `factor_library`).
"""

from __future__ import annotations

import re
from pathlib import Path

DOC_PATH = Path("docs") / "architecture" / "notebook-governance.md"


REQUIRED_SECTIONS = (
    "## Purpose",
    "## Notebook Execution Rules",
    "## Output and Cleanup Rules",
    "## Notebook to Factor Library Promotion",
    "## Forbidden Patterns",
    "## Enforcement",
)


REQUIRED_CONCEPTS = (
    "strip output",
    "secrets",
    "factor_library",
    "promotion",
    "ci",
    "adr",
)


def _read(repo_root: Path) -> str:
    path = repo_root / DOC_PATH
    assert path.is_file(), f"Notebook governance doc missing at {DOC_PATH}"
    return path.read_text(encoding="utf-8")


def test_doc_has_required_sections(repo_root: Path) -> None:
    text = _read(repo_root)
    missing = [s for s in REQUIRED_SECTIONS if s not in text]
    assert not missing, f"Notebook governance doc missing sections: {missing}"


def test_doc_addresses_required_concepts(repo_root: Path) -> None:
    text = _read(repo_root).lower()
    missing = [c for c in REQUIRED_CONCEPTS if c not in text]
    assert not missing, f"Notebook governance doc missing concepts: {missing}"


def test_doc_forbids_importing_notebooks(repo_root: Path) -> None:
    text = _read(repo_root).lower()
    assert "must not import" in text or "forbidden" in text, (
        "Notebook governance doc must explicitly forbid importing notebooks"
    )


# ---------------------------------------------------------------------------
# Static check: no domain package may import anything under
# `alpha_research/notebooks/`.
# ---------------------------------------------------------------------------


NOTEBOOK_IMPORT_PATTERN = re.compile(
    r"^\s*(?:from|import)\s+alpha_research\.notebooks\b", re.MULTILINE
)


def test_no_domain_package_imports_notebooks(repo_root: Path) -> None:
    offenders: list[str] = []
    for domain in (
        "shared_lib",
        "data_platform",
        "alpha_research",
        "backtest_engine",
        "ai_agents",
        "trading_system",
    ):
        root = repo_root / domain / "src"
        if not root.is_dir():
            continue
        for py in root.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            if NOTEBOOK_IMPORT_PATTERN.search(text):
                offenders.append(str(py.relative_to(repo_root)))
    assert not offenders, (
        f"Domain code must not import alpha_research.notebooks: {offenders}"
    )
