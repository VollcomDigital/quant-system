"""Phase 4 Task 9 - authenticated web shell scope doc.

Phase 4 evolves the legacy dashboard into an authenticated initial web
shell that exposes managed run browsing, comparison, report access, and
provenance — *without* introducing direct execution controls. Execution
controls arrive in Phase 6/Phase 10 only after OMS/EMS/RMS land.
"""

from __future__ import annotations

from pathlib import Path

DOC_PATH = Path("docs") / "architecture" / "web-shell-phase-4.md"


REQUIRED_SECTIONS = (
    "## Purpose",
    "## In Scope for Phase 4",
    "## Explicitly Out of Scope",
    "## Authentication Requirements",
    "## Routes",
    "## Enforcement",
)


REQUIRED_CONCEPTS = (
    "run",
    "tear sheet",
    "provenance",
    "authentication",
    "read-only",
    "no execution",
    "phase 6",
)


def _read(repo_root: Path) -> str:
    path = repo_root / DOC_PATH
    assert path.is_file(), f"Web shell doc missing at {DOC_PATH}"
    return path.read_text(encoding="utf-8")


def test_doc_has_required_sections(repo_root: Path) -> None:
    text = _read(repo_root)
    missing = [s for s in REQUIRED_SECTIONS if s not in text]
    assert not missing, f"Web shell doc missing sections: {missing}"


def test_doc_addresses_required_concepts(repo_root: Path) -> None:
    text = _read(repo_root).lower()
    missing = [c for c in REQUIRED_CONCEPTS if c not in text]
    assert not missing, f"Web shell doc missing concepts: {missing}"


def test_doc_explicitly_forbids_execution_controls(repo_root: Path) -> None:
    text = _read(repo_root).lower()
    assert "no execution" in text, (
        "Web shell doc must explicitly state there are no execution controls in Phase 4"
    )
