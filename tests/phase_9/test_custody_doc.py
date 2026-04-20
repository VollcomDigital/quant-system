"""Phase 9 Task 5 - custody + treasury + multisig doc."""

from __future__ import annotations

from pathlib import Path

DOC_PATH = Path("docs") / "architecture" / "custody-treasury-phase-9.md"


REQUIRED_SECTIONS = (
    "## Purpose",
    "## Signer Permission Tiers",
    "## Treasury Custody Model",
    "## Human Multisig Requirements",
    "## Withdrawal Policy",
    "## Nitro Enclave / Enclave-Adjacent Signing",
    "## Enforcement",
)


REQUIRED_CONCEPTS = (
    "aws kms",
    "vault",
    "safe",
    "fireblocks",
    "fordefi",
    "mpc",
    "multisig",
    "withdrawal",
    "nitro enclave",
    "bot",
    "exchange transfer",
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


def test_doc_explicitly_forbids_unrestricted_bot_withdrawals(repo_root: Path) -> None:
    text = _read(repo_root).lower()
    assert "bot" in text and "withdrawal" in text and (
        "forbidden" in text or "cannot" in text or "must not" in text
    )


def test_doc_references_adr_0006(repo_root: Path) -> None:
    assert "ADR-0006" in _read(repo_root)
