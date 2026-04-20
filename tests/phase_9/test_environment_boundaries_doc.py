"""Phase 9 Task 6 - environment boundaries + promotion doc."""

from __future__ import annotations

from pathlib import Path

DOC_PATH = Path("docs") / "architecture" / "environment-boundaries-phase-9.md"


REQUIRED_SECTIONS = (
    "## Purpose",
    "## Environment Ladder",
    "## Local",
    "## Research / Dev",
    "## Paper",
    "## Production / Live",
    "## Promotion Gates",
    "## Data Boundaries",
    "## Enforcement",
)


REQUIRED_CONCEPTS = (
    "local",
    "research",
    "paper",
    "production",
    "kms",
    "paper-trading",
    "recovery workflow",
    "ib gateway",
    "promotion",
)


def _read(repo_root: Path) -> str:
    path = repo_root / DOC_PATH
    assert path.is_file(), f"doc missing at {DOC_PATH}"
    return path.read_text(encoding="utf-8")


def test_doc_has_required_sections(repo_root: Path) -> None:
    text = _read(repo_root)
    missing = [s for s in REQUIRED_SECTIONS if s not in text]
    assert not missing, f"doc missing sections: {missing}"


def test_doc_addresses_required_concepts(repo_root: Path) -> None:
    text = _read(repo_root).lower()
    missing = [c for c in REQUIRED_CONCEPTS if c not in text]
    assert not missing, f"doc missing concepts: {missing}"


def test_doc_separates_credentials_per_environment(repo_root: Path) -> None:
    text = _read(repo_root).lower()
    assert "credentials" in text
    assert "must not" in text or "never share" in text or "separate" in text


def test_doc_documents_ib_gateway_restart_automation(repo_root: Path) -> None:
    text = _read(repo_root).lower()
    assert "ib gateway" in text or "ibkr" in text
    assert "restart" in text
