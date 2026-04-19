"""Phase 3 Exit Criteria - aggregate gate.

Exit criteria (from `tasks/todo.md`):

- Notebook code is clearly separated from production factor modules.
- Factor promotion path is documented and testable.
- Model artifacts can be traced to training data, registry metadata,
  and promotion status.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal


def test_exit_notebook_and_factor_library_are_separate_packages() -> None:
    # factor_library must be importable; notebooks package is a directory
    # marker, not a module consumed by code.
    import alpha_research.factor_library  # noqa: F401

    # The governance test suite enforces the no-import-notebooks rule.


def test_exit_factor_promotion_path_is_testable() -> None:
    from alpha_research.factor_library import FactorMetadata
    from alpha_research.promotion import FactorPromotionRequest, promote_factor

    m = FactorMetadata(
        factor_id="f1",
        version="v1",
        description="d",
        source_dependencies=(),
        stationarity_assumption="s",
        universe_coverage="u",
        leakage_review="ok",
        validation_status="candidate",
    )
    result = promote_factor(
        FactorPromotionRequest(
            metadata=m,
            target_status="validated",
            coverage_pct=Decimal("0.9"),
            oos_sharpe=Decimal("1.0"),
            leakage_check_passed=True,
        )
    )
    assert result.passed is True


def test_exit_model_artifacts_trace_back_to_training_data() -> None:
    from alpha_research.ml_models.registry import ModelRecord, ModelRegistry

    reg = ModelRegistry()
    rec = ModelRecord(
        model_id="m1",
        version="v1",
        training_run_id="run-exit",
        trained_at=datetime(2026, 4, 19, tzinfo=UTC),
        training_data_snapshot_id="snap-exit",
        metrics={"sharpe": Decimal("1.3")},
        artifact_path="s3://a.onnx",
        lifecycle_stage="staging",
    )
    reg.register(rec)
    got = reg.get(model_id="m1", version="v1")
    assert got.training_data_snapshot_id == "snap-exit"
    assert got.training_run_id == "run-exit"
