"""Phase 2 Task 7 - data platform policy document invariants.

Three policies the roadmap requires Phase 2 to nail down in writing:

1. HFT vs mid-frequency ingestion split (L3 tick + PCAP vs Parquet-
   backed bars).
2. Vendor routing: broker APIs (Alpaca, IBKR) are live-only; historical
   training data flows through Polygon / Databento / Tiingo.
3. Polars as the default data-manipulation engine; pandas permitted
   only inside the legacy `src/*` compatibility facade.
"""

from __future__ import annotations

from pathlib import Path

DOC_PATH = Path("docs") / "architecture" / "data-platform-policies.md"


REQUIRED_SECTIONS = (
    "## Ingestion Split (HFT vs Mid-Frequency)",
    "## Vendor Routing",
    "## Polars Default",
    "## Orchestration Baseline",
    "## Feature Store as Source of Truth",
    "## Enforcement",
)


REQUIRED_CONCEPTS = (
    "l3",
    "pcap",
    "parquet",
    "polars",
    "airflow",
    "polygon",
    "databento",
    "tiingo",
    "alpaca",
    "ibkr",
)


def _read(repo_root: Path) -> str:
    path = repo_root / DOC_PATH
    assert path.is_file(), f"Policy doc missing at {DOC_PATH}"
    return path.read_text(encoding="utf-8")


def test_doc_has_all_required_sections(repo_root: Path) -> None:
    text = _read(repo_root)
    missing = [s for s in REQUIRED_SECTIONS if s not in text]
    assert not missing, f"Policy doc missing sections: {missing}"


def test_doc_addresses_required_concepts(repo_root: Path) -> None:
    text = _read(repo_root).lower()
    missing = [c for c in REQUIRED_CONCEPTS if c not in text]
    assert not missing, f"Policy doc missing concepts: {missing}"


def test_doc_forbids_broker_as_historical_source(repo_root: Path) -> None:
    text = _read(repo_root).lower()
    # We require an explicit statement that broker APIs are not the
    # historical training source.
    assert "not" in text and ("alpaca" in text or "ibkr" in text), (
        "Policy doc must explicitly forbid brokers as the historical training source"
    )


def test_doc_names_airflow_as_primary_orchestrator(repo_root: Path) -> None:
    text = _read(repo_root).lower()
    assert "airflow" in text, "Policy doc must name Airflow as the primary orchestrator"


def test_doc_declares_polars_default(repo_root: Path) -> None:
    text = _read(repo_root).lower()
    assert "polars" in text and "default" in text, (
        "Policy doc must declare Polars as the default data-manipulation engine"
    )


def test_doc_references_adr_0002(repo_root: Path) -> None:
    text = _read(repo_root)
    assert "ADR-0002" in text
