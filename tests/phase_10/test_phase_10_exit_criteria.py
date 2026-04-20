"""Phase 10 Exit-Criteria verification.

Binding checks that the repository shape, contract tests, and
documentation match what `tasks/todo.md` promised for Phase 10 Exit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def phase_10_tests_dir() -> Path:
    return REPO_ROOT / "tests" / "phase_10"


def test_every_contract_boundary_has_a_test_module(phase_10_tests_dir: Path) -> None:
    required = {
        "test_research_to_backtest_contract.py",
        "test_backtest_to_oms_ems_contract.py",
        "test_oms_ems_to_gateways_contract.py",
        "test_agents_to_ci_telemetry_contract.py",
        "test_parity_report_and_cli_cutover.py",
    }
    present = {p.name for p in phase_10_tests_dir.glob("test_*.py")}
    missing = required - present
    assert not missing, f"Phase 10 is missing contract tests: {sorted(missing)}"


def test_legacy_src_retained_behind_compat_facade() -> None:
    """Phase 10 exit criterion: legacy `src/*` either removed or
    retained behind the compatibility facade; the facade doc owns
    this decision."""
    facade = REPO_ROOT / "docs" / "architecture" / "cli-compatibility-facade.md"
    assert facade.is_file()
    body = facade.read_text(encoding="utf-8").lower()
    assert "retirement" in body
    assert "phase-10-parity-report.md" in body


def test_web_control_plane_is_primary_operator_interface() -> None:
    """Exit criterion: the web control plane is the primary operator
    surface; it must exist and expose the key operator APIs."""
    backend = REPO_ROOT / "web_control_plane" / "backend" / "src" / "web_control_plane" / "backend"
    assert backend.is_dir()
    # At minimum: an API tree with an execution-surface module.
    api = backend / "api"
    assert api.is_dir()
    # An execution surface must exist (per Phase 6 design).
    assert (api / "execution.py").is_file()


def test_repo_can_be_reasoned_about_by_domain_package() -> None:
    """Exit criterion: the repository is reasoned about by package/domain."""
    expected_domains = (
        "shared_lib",
        "data_platform",
        "alpha_research",
        "backtest_engine",
        "ai_agents",
        "trading_system",
        "web_control_plane",
        "infrastructure",
    )
    for pkg in expected_domains:
        assert (REPO_ROOT / pkg).is_dir(), f"missing domain package: {pkg}"


def test_parity_validation_report_exists() -> None:
    """Exit criterion: parity validation report exists for data,
    backtests, reporting, and execution-control paths."""
    report = REPO_ROOT / "docs" / "architecture" / "phase-10-parity-report.md"
    assert report.is_file()
    body = report.read_text(encoding="utf-8").lower()
    for path in ("backtest", "data", "reporting", "execution"):
        assert path in body, f"parity report must address {path!r}"
