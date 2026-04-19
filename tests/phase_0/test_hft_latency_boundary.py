"""Phase 0 Task 6 - Python/HFT latency boundary documentation.

ADR-0003 accepts the two-speed architecture. This document turns that
decision into an auditable boundary:

- where Python is allowed
- where Python is explicitly forbidden
- what latency budget moves a strategy from mid-frequency to HFT
- which handoff transport crosses the line

The tests enforce that the document stays concrete (numbers, budgets, and
transports) instead of degenerating into a stub.
"""

from __future__ import annotations

import re
from pathlib import Path

DOC_PATH = Path("docs") / "architecture" / "hft-latency-boundary.md"


REQUIRED_SECTIONS = (
    "## Purpose",
    "## Python-Allowed Zones",
    "## Python-Forbidden Zones",
    "## Latency Budgets",
    "## Handoff Contract",
    "## Escalation Rules",
    "## Enforcement",
)


REQUIRED_CONCEPTS = (
    "tick-to-trade",
    "co-located",
    "rust",
    "onnx",
    "shared memory",
    "kernel bypass",
)


def _read(repo_root: Path) -> str:
    path = repo_root / DOC_PATH
    assert path.is_file(), f"HFT latency boundary doc missing at {DOC_PATH}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Edge case 1: every required section must exist.
# ---------------------------------------------------------------------------


def test_doc_has_all_required_sections(repo_root: Path) -> None:
    text = _read(repo_root)
    missing = [s for s in REQUIRED_SECTIONS if s not in text]
    assert not missing, f"HFT boundary doc missing sections: {missing}"


# ---------------------------------------------------------------------------
# Edge case 2: the boundary must reference ADR-0003 so the rule is traceable.
# ---------------------------------------------------------------------------


def test_doc_references_adr_0003(repo_root: Path) -> None:
    assert "ADR-0003" in _read(repo_root), "Doc must cite ADR-0003"


# ---------------------------------------------------------------------------
# Edge case 3: required low-level concepts must appear so the boundary is
# operationally meaningful.
# ---------------------------------------------------------------------------


def test_doc_addresses_required_concepts(repo_root: Path) -> None:
    text = _read(repo_root).lower()
    missing = [c for c in REQUIRED_CONCEPTS if c.lower() not in text]
    assert not missing, f"HFT boundary doc missing concepts: {missing}"


# ---------------------------------------------------------------------------
# Edge case 4: latency budgets must be numeric and include at least one
# microsecond-scale figure. We look for patterns like "500 us", "100 µs",
# or "10 microseconds".
# ---------------------------------------------------------------------------


MICROSECOND_PATTERN = re.compile(
    r"\b\d+\s?(?:us|µs|microsecond|microseconds)\b", re.IGNORECASE
)


def test_doc_declares_microsecond_budget(repo_root: Path) -> None:
    text = _read(repo_root)
    assert MICROSECOND_PATTERN.search(text), (
        "HFT boundary doc must declare at least one microsecond-scale budget"
    )


# ---------------------------------------------------------------------------
# Edge case 5: explicit enumeration of Python-forbidden paths - the doc must
# name at least these: order entry, market data decoding, HFT inference.
# ---------------------------------------------------------------------------


FORBIDDEN_PATHS = ("order entry", "market data decoding", "inference")


def test_doc_enumerates_forbidden_paths(repo_root: Path) -> None:
    text = _read(repo_root).lower()
    missing = [p for p in FORBIDDEN_PATHS if p not in text]
    assert not missing, (
        f"HFT boundary doc must enumerate forbidden Python paths; missing: {missing}"
    )


# ---------------------------------------------------------------------------
# Edge case 6: escalation must describe how a strategy graduates from
# mid-frequency to HFT. We require the words "escalation" and a reference
# to compiled artifacts (ONNX / C++ / TensorRT).
# ---------------------------------------------------------------------------


def test_doc_describes_escalation_path(repo_root: Path) -> None:
    text = _read(repo_root).lower()
    assert "escalation" in text, "Doc must document escalation from mid-freq to HFT"
    assert any(t in text for t in ("onnx", "tensorrt", "c++")), (
        "Doc must reference the compiled-artifact requirement for HFT eligibility"
    )
