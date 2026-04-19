"""Phase 3 Task 6 - alpha_research.promotion gates.

Promotion is staged:

  notebook candidate -> factor library candidate -> validated -> promoted
  (live-eligible)

Each transition is guarded by a `PromotionGate` that runs a fixed set
of checks. Gates cannot be skipped; the result of any failing check
blocks the transition and is surfaced as a
`shared_lib.contracts.ValidationResult`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest


def _metadata(status: str = "candidate"):
    from alpha_research.factor_library import FactorMetadata

    return FactorMetadata(
        factor_id="mom_12_1",
        version="v1",
        description="12-1 momentum",
        source_dependencies=("bars",),
        stationarity_assumption="mean-reverting monthly",
        universe_coverage="us_equities",
        leakage_review="reviewed: uses only t-1 bar",
        validation_status=status,  # type: ignore[arg-type]
    )


def _model_record(stage: str = "staging"):
    from alpha_research.ml_models.registry import ModelRecord

    return ModelRecord(
        model_id="kronos-v0",
        version="v1",
        training_run_id="run-1",
        trained_at=datetime(2026, 4, 19, tzinfo=UTC),
        training_data_snapshot_id="snap-abc",
        metrics={"sharpe": Decimal("1.2")},
        artifact_path="s3://m/v1.onnx",
        lifecycle_stage=stage,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Factor promotion gates
# ---------------------------------------------------------------------------


def test_factor_gate_candidate_to_validated_requires_backtest() -> None:
    from alpha_research.promotion import (
        FactorPromotionRequest,
        promote_factor,
    )

    req = FactorPromotionRequest(
        metadata=_metadata("candidate"),
        target_status="validated",
        coverage_pct=Decimal("0.9"),
        oos_sharpe=None,
        leakage_check_passed=True,
    )
    result = promote_factor(req)
    assert result.passed is False
    assert "oos" in (result.reason or "").lower()


def test_factor_gate_requires_leakage_check() -> None:
    from alpha_research.promotion import (
        FactorPromotionRequest,
        promote_factor,
    )

    req = FactorPromotionRequest(
        metadata=_metadata("candidate"),
        target_status="validated",
        coverage_pct=Decimal("0.9"),
        oos_sharpe=Decimal("1.0"),
        leakage_check_passed=False,  # <- the blocker
    )
    result = promote_factor(req)
    assert result.passed is False
    assert "leakage" in (result.reason or "").lower()


def test_factor_gate_promotes_when_all_checks_pass() -> None:
    from alpha_research.promotion import (
        FactorPromotionRequest,
        promote_factor,
    )

    req = FactorPromotionRequest(
        metadata=_metadata("candidate"),
        target_status="validated",
        coverage_pct=Decimal("0.9"),
        oos_sharpe=Decimal("1.0"),
        leakage_check_passed=True,
    )
    result = promote_factor(req)
    assert result.passed is True


def test_factor_gate_refuses_skip_to_promoted() -> None:
    from alpha_research.promotion import (
        FactorPromotionRequest,
        promote_factor,
    )

    req = FactorPromotionRequest(
        metadata=_metadata("candidate"),
        target_status="promoted",  # skipping validated
        coverage_pct=Decimal("0.99"),
        oos_sharpe=Decimal("2.0"),
        leakage_check_passed=True,
    )
    result = promote_factor(req)
    assert result.passed is False
    assert "skip" in (result.reason or "").lower()


def test_factor_gate_refuses_low_coverage() -> None:
    from alpha_research.promotion import (
        FactorPromotionRequest,
        promote_factor,
    )

    req = FactorPromotionRequest(
        metadata=_metadata("candidate"),
        target_status="validated",
        coverage_pct=Decimal("0.5"),  # below threshold
        oos_sharpe=Decimal("1.0"),
        leakage_check_passed=True,
    )
    result = promote_factor(req)
    assert result.passed is False
    assert "coverage" in (result.reason or "").lower()


# ---------------------------------------------------------------------------
# Model promotion gates
# ---------------------------------------------------------------------------


def test_model_gate_staging_to_production_requires_validation() -> None:
    from alpha_research.promotion import (
        ModelPromotionRequest,
        promote_model,
    )

    req = ModelPromotionRequest(
        record=_model_record("staging"),
        target_stage="production",
        validation_passed=False,
        drift_check_passed=True,
        approval_id=None,
    )
    result = promote_model(req)
    assert result.passed is False
    assert "validation" in (result.reason or "").lower()


def test_model_gate_requires_drift_check() -> None:
    from alpha_research.promotion import (
        ModelPromotionRequest,
        promote_model,
    )

    req = ModelPromotionRequest(
        record=_model_record("staging"),
        target_stage="production",
        validation_passed=True,
        drift_check_passed=False,
        approval_id="app-1",
    )
    result = promote_model(req)
    assert result.passed is False


def test_model_gate_requires_approval_for_production() -> None:
    from alpha_research.promotion import (
        ModelPromotionRequest,
        promote_model,
    )

    req = ModelPromotionRequest(
        record=_model_record("staging"),
        target_stage="production",
        validation_passed=True,
        drift_check_passed=True,
        approval_id=None,
    )
    result = promote_model(req)
    assert result.passed is False
    assert "approval" in (result.reason or "").lower()


def test_model_gate_promotes_when_all_checks_pass() -> None:
    from alpha_research.promotion import (
        ModelPromotionRequest,
        promote_model,
    )

    req = ModelPromotionRequest(
        record=_model_record("staging"),
        target_stage="production",
        validation_passed=True,
        drift_check_passed=True,
        approval_id="app-1",
    )
    result = promote_model(req)
    assert result.passed is True


def test_model_gate_rejects_input_requests() -> None:
    # Defensive: request with negative Sharpe should raise at construction.
    from alpha_research.promotion import FactorPromotionRequest

    with pytest.raises(ValueError):
        FactorPromotionRequest(
            metadata=_metadata("candidate"),
            target_status="validated",
            coverage_pct=Decimal("-0.1"),
            oos_sharpe=Decimal("1.0"),
            leakage_check_passed=True,
        )
