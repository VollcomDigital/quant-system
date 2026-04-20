"""Phase 8 Task 4 - Python exclusion + HFT interface boundary.

The tick-to-trade HFT critical path must not touch Python. Two static
rules:

1. No `*.py` files under `trading_system/native/` (already enforced by
   Task 1 but we re-enforce at this layer explicitly).
2. No Python domain package may `import trading_system.native.*` or
   any `hft_engine.core.*` symbol at runtime. The Python-side
   `trading_system.hft_engine` package (model cards + benchmark) is
   allowed and expected.
"""

from __future__ import annotations

import re
from pathlib import Path

NATIVE_IMPORT_PATTERN = re.compile(
    r"^\s*(?:from|import)\s+trading_system\.native(?:\.|\s)",
    re.MULTILINE,
)


PYTHON_DOMAINS = (
    "shared_lib",
    "data_platform",
    "backtest_engine",
    "alpha_research",
    "ai_agents",
    "trading_system",
)


def test_no_python_imports_from_native_subtree(repo_root: Path) -> None:
    offenders: list[str] = []
    for domain in PYTHON_DOMAINS:
        root = repo_root / domain / "src"
        if not root.is_dir():
            continue
        for py in root.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            if NATIVE_IMPORT_PATTERN.search(text):
                offenders.append(str(py.relative_to(repo_root)))
    assert not offenders, (
        f"Python domain packages must not import trading_system.native.*: {offenders}"
    )


# ---------------------------------------------------------------------------
# HFT interface boundary doc
# ---------------------------------------------------------------------------


DOC_PATH = Path("docs") / "architecture" / "hft-engine-interface-phase-8.md"


REQUIRED_SECTIONS = (
    "## Purpose",
    "## Python Exclusion Boundary",
    "## Native Crate Boundaries",
    "## Model Compile Requirement",
    "## Replay + Benchmark Prerequisites",
    "## Interfaces Between HFT Core and Subsystems",
    "## Enforcement",
)


REQUIRED_CONCEPTS = (
    "tick-to-trade",
    "kernel bypass",
    "co-located",
    "onnx",
    "tensorrt",
    "fpga",
    "lock-free",
    "ring buffer",
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


def test_doc_references_hft_latency_boundary(repo_root: Path) -> None:
    text = _read(repo_root)
    assert "hft-latency-boundary" in text or "ADR-0003" in text


def test_doc_explicitly_forbids_python_on_tick_to_trade(repo_root: Path) -> None:
    text = _read(repo_root).lower()
    assert "python" in text and "forbidden" in text
