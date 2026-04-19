"""Phase 0 Task 1 - Monorepo package strategy decision.

Arrange-Act-Assert tests verifying ADR-0001 records an Accepted packaging
strategy with explicit, unambiguous content. These act as an executable
architecture gate: ADR-0001 cannot regress back to `Proposed` or lose its
normative content without breaking CI.
"""

from __future__ import annotations

import re
from pathlib import Path

ADR_FILENAME = "0001-monorepo-workspace-and-package-boundaries.md"


def _read(adr_dir: Path) -> str:
    path = adr_dir / ADR_FILENAME
    assert path.exists(), f"ADR-0001 missing at {path}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Edge case 1: status must be Accepted, not Proposed/Deprecated/blank.
# ---------------------------------------------------------------------------


def test_adr_0001_status_is_accepted(adr_dir: Path) -> None:
    text = _read(adr_dir)
    match = re.search(r"^[-*]\s*Status\s*:\s*(\S+)", text, re.MULTILINE | re.IGNORECASE)
    assert match is not None, "ADR-0001 is missing a Status field"
    status = match.group(1).strip().lower()
    assert status == "accepted", (
        f"ADR-0001 status must be 'Accepted' for Phase 0 exit, got '{status}'"
    )


# ---------------------------------------------------------------------------
# Edge case 2: the decision must be binding - a `## Decision` section with
# a concrete chosen option, not a list of open options or a `## Proposed
# Direction` stub.
# ---------------------------------------------------------------------------


def test_adr_0001_has_binding_decision_section(adr_dir: Path) -> None:
    text = _read(adr_dir)
    assert re.search(r"^##\s+Decision\s*$", text, re.MULTILINE), (
        "ADR-0001 must contain a binding '## Decision' section, "
        "not just '## Proposed Direction' or '## Decision Drivers'"
    )


# ---------------------------------------------------------------------------
# Edge case 3: the decision must explicitly name the chosen model so readers
# do not have to infer intent. We require two things in the Decision body:
#   (a) a root/workspace-style manifest, and
#   (b) package-local `pyproject.toml` files per domain package.
# ---------------------------------------------------------------------------


def _decision_body(text: str) -> str:
    # Grab everything under the exact "## Decision" heading (not "## Decision
    # Drivers") up to the next top-level `## ` heading.
    match = re.search(
        r"^##\s+Decision\s*$\s*(.*?)(?=^##\s+\S|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, "Decision section must exist before parsing its body"
    return match.group(1).lower()


def test_adr_0001_decision_specifies_root_workspace(adr_dir: Path) -> None:
    body = _decision_body(_read(adr_dir))
    assert "workspace" in body or "root" in body, (
        "ADR-0001 Decision must describe the root/workspace manifest model"
    )


def test_adr_0001_decision_specifies_package_local_pyproject(adr_dir: Path) -> None:
    body = _decision_body(_read(adr_dir))
    # Either exact token or a tolerant phrasing that still names the manifest.
    assert "pyproject.toml" in body or "package-local manifest" in body, (
        "ADR-0001 Decision must reference package-local pyproject.toml manifests"
    )


# ---------------------------------------------------------------------------
# Edge case 4: the six Python domains declared in
# docs/architecture/phase-0-scaffold.md must all appear in the Decision body
# so there is no ambiguity about which packages are in scope.
# ---------------------------------------------------------------------------


REQUIRED_DOMAINS = (
    "shared_lib",
    "data_platform",
    "backtest_engine",
    "alpha_research",
    "ai_agents",
    "trading_system",
)


def test_adr_0001_decision_enumerates_all_python_domains(adr_dir: Path) -> None:
    body = _decision_body(_read(adr_dir))
    missing = [d for d in REQUIRED_DOMAINS if d not in body]
    assert not missing, (
        f"ADR-0001 Decision must enumerate all Python domains; missing: {missing}"
    )


# ---------------------------------------------------------------------------
# Edge case 5: native HFT placement must be explicit, otherwise the two-speed
# architecture is not actually constrained by ADR-0001.
# ---------------------------------------------------------------------------


def test_adr_0001_decision_addresses_native_hft_placement(adr_dir: Path) -> None:
    body = _decision_body(_read(adr_dir))
    assert "native" in body and (
        "hft" in body or "trading_system/native" in body
    ), "ADR-0001 Decision must place native HFT code explicitly"


# ---------------------------------------------------------------------------
# Edge case 6: legacy `src/` compatibility must be preserved by the decision.
# ---------------------------------------------------------------------------


def test_adr_0001_decision_preserves_src_compatibility(adr_dir: Path) -> None:
    body = _decision_body(_read(adr_dir))
    assert "src/" in body or "src.main" in body or "compatibility" in body, (
        "ADR-0001 Decision must record the `src/` compatibility stance"
    )
