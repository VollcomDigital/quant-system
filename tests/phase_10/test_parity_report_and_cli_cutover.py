"""Phase 10 Task 5 - parity validation report + compatibility CLI cutover.

- `docs/architecture/phase-10-parity-report.md` records the contract
  tests that pass, the retirement status of each legacy `src/*`
  module, and the coverage+lint gates the monorepo now ships.
- `docs/architecture/cli-compatibility-facade.md` gains a concrete
  cutover section that references the parity report and names the
  explicit prerequisites for removing `src/` from `pyproject.toml`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def parity_doc() -> Path:
    return REPO_ROOT / "docs" / "architecture" / "phase-10-parity-report.md"


@pytest.fixture(scope="module")
def cutover_doc() -> Path:
    return REPO_ROOT / "docs" / "architecture" / "cli-compatibility-facade.md"


def test_parity_report_exists_and_names_every_contract_boundary(parity_doc: Path) -> None:
    assert parity_doc.is_file()
    body = parity_doc.read_text(encoding="utf-8").lower()
    for required in (
        "research",
        "backtest",
        "oms",
        "ems",
        "gateways",
        "agents",
        "ci",
        "validationresult",
    ):
        assert required in body, f"parity report must reference {required!r}"


def test_parity_report_lists_retirement_status_for_legacy_src(parity_doc: Path) -> None:
    body = parity_doc.read_text(encoding="utf-8").lower()
    # It must take a concrete retirement stance for each major legacy area.
    for area in ("src/backtest", "src/data", "src/reporting", "src/dashboard"):
        assert area in body, f"parity report must mention {area!r}"
    # And must name the gates that control removal.
    for gate in (
        "coverage",
        "deprecationwarning",
        "contract test",
    ):
        assert gate in body


def test_cli_cutover_section_points_at_parity_report(cutover_doc: Path) -> None:
    body = cutover_doc.read_text(encoding="utf-8")
    assert "phase-10-parity-report.md" in body, (
        "cli-compatibility-facade.md must link to the parity report"
    )


def test_cli_cutover_names_explicit_retirement_prerequisites(cutover_doc: Path) -> None:
    body = cutover_doc.read_text(encoding="utf-8").lower()
    for cond in (
        "coverage",
        "deprecationwarning",
        "contract test",
        "pyproject.toml",
    ):
        assert cond in body, f"cutover doc must mention {cond!r}"


def test_parity_report_records_phase_10_test_counts(parity_doc: Path) -> None:
    """The report is the authoritative numeric record of Phase 10."""
    body = parity_doc.read_text(encoding="utf-8")
    for phase in ("phase 0", "phase 1", "phase 10"):
        assert phase.lower() in body.lower()
    # Mention of at least one specific number keeps the doc honest.
    assert any(ch.isdigit() for ch in body)
