"""Phase 7 Task 7 - IBKR operational playbook + gateway-architecture doc.

The roadmap requires IBKR-specific operational workflows (scheduled
daily restart, pre-restart trading halt, re-auth + reconnect, OMS
reconciliation after reconnect). Phase 7 codifies the playbook so the
Phase 9 IB Gateway container deployment can implement it.
"""

from __future__ import annotations

from pathlib import Path

DOC_PATH = Path("docs") / "architecture" / "gateway-operations-phase-7.md"


REQUIRED_SECTIONS = (
    "## Purpose",
    "## TradFi vs Web3 Gateway Split",
    "## IBKR Daily Restart Playbook",
    "## Reconnect and Reconciliation",
    "## DeFi Kill Controls",
    "## No Browser Credentials",
    "## Enforcement",
)


REQUIRED_CONCEPTS = (
    "ib gateway",
    "ibc",
    "reqglobalcancel",
    "alpaca",
    "alchemy",
    "infura",
    "kms",
    "pause()",
    "approve",
    "denylist",
    "reconcile",
)


def _read(repo_root: Path) -> str:
    path = repo_root / DOC_PATH
    assert path.is_file(), f"Doc missing at {DOC_PATH}"
    return path.read_text(encoding="utf-8")


def test_doc_has_required_sections(repo_root: Path) -> None:
    text = _read(repo_root)
    missing = [s for s in REQUIRED_SECTIONS if s not in text]
    assert not missing, f"Doc missing sections: {missing}"


def test_doc_addresses_required_concepts(repo_root: Path) -> None:
    text = _read(repo_root).lower()
    missing = [c for c in REQUIRED_CONCEPTS if c not in text]
    assert not missing, f"Doc missing concepts: {missing}"


def test_doc_explicitly_forbids_keys_in_browser(repo_root: Path) -> None:
    text = _read(repo_root).lower()
    assert "browser" in text and (
        "private keys" in text and "never" in text
    )
