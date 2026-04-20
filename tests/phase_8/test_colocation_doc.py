"""Phase 8 Task 5 - co-location + bare-metal deployment requirements."""

from __future__ import annotations

from pathlib import Path

DOC_PATH = Path("docs") / "architecture" / "hft-colocation-phase-8.md"


REQUIRED_SECTIONS = (
    "## Purpose",
    "## Co-Location Requirements",
    "## Bare-Metal Hardware Baseline",
    "## Network Fabric",
    "## Kernel and OS Tuning",
    "## Deployment Boundary vs Cloud Runtime",
    "## Enforcement",
)


REQUIRED_CONCEPTS = (
    "co-located",
    "bare-metal",
    "solarflare",
    "cpu pinning",
    "hugepages",
    "ptp",
    "iommu",
    "no python",
    "kubernetes",  # must say kubernetes is NOT the HFT runtime
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


def test_doc_references_adr_0003(repo_root: Path) -> None:
    assert "ADR-0003" in _read(repo_root)


def test_doc_explicitly_separates_hft_from_cloud(repo_root: Path) -> None:
    text = _read(repo_root).lower()
    assert "not on kubernetes" in text or "not in kubernetes" in text or (
        "kubernetes" in text and "not" in text
    )
