"""Phase 0 Task 4 - ADR status and implementation-owner invariants.

Per the roadmap delivery rule:
  "No major implementation phase should start without the relevant ADR
   status set to `accepted` or `proposed with explicit implementation owner`."

These tests lock each ADR into a Phase-0-approved state so later phases
cannot start without either an Accepted ADR or a Proposed ADR that names an
implementation owner.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# ADR id -> (required status set, required implementation owner phrase)
ADR_REQUIREMENTS: dict[str, tuple[set[str], bool]] = {
    "0001-monorepo-workspace-and-package-boundaries.md": ({"accepted"}, True),
    "0002-data-platform-orchestration-and-immutability.md": ({"accepted", "proposed"}, True),
    "0003-two-speed-execution-runtime-boundaries.md": ({"accepted"}, True),
    "0004-agent-permissions-and-control-plane.md": ({"accepted", "proposed"}, True),
    "0005-tradfi-and-web3-gateway-architecture.md": ({"accepted", "proposed"}, True),
    "0006-execution-signing-custody-and-kill-switches.md": ({"accepted", "proposed"}, True),
}


def _load(adr_dir: Path, filename: str) -> str:
    path = adr_dir / filename
    assert path.is_file(), f"ADR missing: {filename}"
    return path.read_text(encoding="utf-8")


def _status(text: str) -> str:
    match = re.search(r"^[-*]\s*Status\s*:\s*([A-Za-z]+)", text, re.MULTILINE)
    assert match is not None, "ADR is missing Status field"
    return match.group(1).strip().lower()


# ---------------------------------------------------------------------------
# Edge case 1: every ADR must be at one of the whitelisted statuses.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename,spec", list(ADR_REQUIREMENTS.items()))
def test_adr_status_is_whitelisted(
    adr_dir: Path, filename: str, spec: tuple[set[str], bool]
) -> None:
    allowed, _requires_owner = spec
    text = _load(adr_dir, filename)
    status = _status(text)
    assert status in allowed, (
        f"{filename} status '{status}' not in allowed set {sorted(allowed)}"
    )


# ---------------------------------------------------------------------------
# Edge case 2: every Proposed ADR must name an explicit implementation owner
# so no phase can start against a faceless stub.
# ---------------------------------------------------------------------------


OWNER_PATTERN = re.compile(
    r"^[-*]\s*(?:Implementation\s+Owner|Owners?)\s*:\s*(.+)$",
    re.MULTILINE,
)


@pytest.mark.parametrize("filename,spec", list(ADR_REQUIREMENTS.items()))
def test_adr_declares_implementation_owner(
    adr_dir: Path, filename: str, spec: tuple[set[str], bool]
) -> None:
    _allowed, requires_owner = spec
    if not requires_owner:
        return
    text = _load(adr_dir, filename)
    match = OWNER_PATTERN.search(text)
    assert match is not None, (
        f"{filename} must declare 'Owners' or 'Implementation Owner'"
    )
    owners = match.group(1).strip()
    assert owners and owners.lower() not in {"tbd", "todo", "unassigned"}, (
        f"{filename} implementation owner cannot be TBD/TODO/unassigned"
    )


# ---------------------------------------------------------------------------
# Edge case 3: proposed ADRs for later phases must explicitly reference
# their target phase so the reviewer can map them onto the roadmap.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", list(ADR_REQUIREMENTS.keys()))
def test_adr_has_target_phase(adr_dir: Path, filename: str) -> None:
    text = _load(adr_dir, filename)
    assert re.search(r"^[-*]\s*Target\s+phase\s*:", text, re.MULTILINE | re.IGNORECASE), (
        f"{filename} must declare a 'Target phase' field"
    )


# ---------------------------------------------------------------------------
# Edge case 4: ADR-0003 must be Accepted because Phase 0 exit depends on
# two-speed separation being a binding decision.
# ---------------------------------------------------------------------------


def test_adr_0003_is_accepted(adr_dir: Path) -> None:
    text = _load(adr_dir, "0003-two-speed-execution-runtime-boundaries.md")
    assert _status(text) == "accepted", (
        "ADR-0003 must be Accepted for Phase 0 exit"
    )


# ---------------------------------------------------------------------------
# Edge case 5: every Accepted ADR must carry a binding `## Decision` section,
# not a `## Proposed Direction` stub.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename,spec", list(ADR_REQUIREMENTS.items()))
def test_accepted_adr_has_decision_section(
    adr_dir: Path, filename: str, spec: tuple[set[str], bool]
) -> None:
    text = _load(adr_dir, filename)
    if _status(text) != "accepted":
        return
    assert re.search(r"^##\s+Decision\s*$", text, re.MULTILINE), (
        f"Accepted ADR {filename} must contain a '## Decision' section"
    )
