"""Phase 3 Task 4 - alpha_research.ml_models.registry contract.

Model registry must bind deployed model weights to specific training
runs and dates (roadmap: MLflow or W&B). Phase 3 builds the contract;
an MLflow backend is a Phase 9 infrastructure concern.

Contract:

- A `ModelRecord` has `model_id`, `version`, `training_run_id`,
  `trained_at`, `training_data_snapshot_id`, `metrics`, `artifact_path`,
  and `lifecycle_stage`.
- A `ModelRegistry` is append-only per `(model_id, version)`.
- Transitions are explicit: Staging → Production → Archived, and no
  transition can skip a stage.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest


def _record(**overrides):
    from alpha_research.ml_models.registry import ModelRecord

    defaults = {
        "model_id": "kronos-v0",
        "version": "v1",
        "training_run_id": "run-1",
        "trained_at": datetime(2026, 4, 19, tzinfo=UTC),
        "training_data_snapshot_id": "snap-abc",
        "metrics": {"sharpe": Decimal("1.1")},
        "artifact_path": "s3://models/kronos-v0/v1.onnx",
        "lifecycle_stage": "staging",
    }
    defaults.update(overrides)
    return ModelRecord(**defaults)


# ---------------------------------------------------------------------------
# Record validation
# ---------------------------------------------------------------------------


def test_model_record_accepts_valid_data() -> None:
    rec = _record()
    assert rec.lifecycle_stage == "staging"


def test_model_record_requires_training_run_id() -> None:
    with pytest.raises(ValueError):
        _record(training_run_id="")


def test_model_record_requires_training_data_snapshot() -> None:
    with pytest.raises(ValueError):
        _record(training_data_snapshot_id="")


def test_model_record_rejects_invalid_stage() -> None:
    with pytest.raises(ValueError):
        _record(lifecycle_stage="heaven")


def test_model_record_rejects_naive_trained_at() -> None:
    with pytest.raises(ValueError):
        _record(trained_at=datetime(2026, 4, 19))


# ---------------------------------------------------------------------------
# Registry immutability + transitions
# ---------------------------------------------------------------------------


def test_registry_rejects_duplicate_version() -> None:
    from alpha_research.ml_models.registry import ModelRegistry

    reg = ModelRegistry()
    reg.register(_record())
    with pytest.raises(ValueError, match="already"):
        reg.register(_record())


def test_registry_transition_staging_to_production() -> None:
    from alpha_research.ml_models.registry import ModelRegistry

    reg = ModelRegistry()
    reg.register(_record())
    rec = reg.transition(model_id="kronos-v0", version="v1", to="production")
    assert rec.lifecycle_stage == "production"


def test_registry_transition_refuses_stage_skip() -> None:
    from alpha_research.ml_models.registry import ModelRegistry

    reg = ModelRegistry()
    reg.register(_record())
    with pytest.raises(ValueError, match="skip"):
        reg.transition(model_id="kronos-v0", version="v1", to="archived")


def test_registry_transition_refuses_reverse_move() -> None:
    from alpha_research.ml_models.registry import ModelRegistry

    reg = ModelRegistry()
    reg.register(_record(lifecycle_stage="production"))
    with pytest.raises(ValueError, match="reverse"):
        reg.transition(model_id="kronos-v0", version="v1", to="staging")


def test_registry_get_returns_latest_mutation() -> None:
    from alpha_research.ml_models.registry import ModelRegistry

    reg = ModelRegistry()
    reg.register(_record())
    reg.transition(model_id="kronos-v0", version="v1", to="production")
    rec = reg.get(model_id="kronos-v0", version="v1")
    assert rec.lifecycle_stage == "production"


def test_registry_binds_snapshot_id_to_record() -> None:
    from alpha_research.ml_models.registry import ModelRegistry

    reg = ModelRegistry()
    reg.register(_record(training_data_snapshot_id="snap-XYZ"))
    rec = reg.get(model_id="kronos-v0", version="v1")
    assert rec.training_data_snapshot_id == "snap-XYZ"


# ---------------------------------------------------------------------------
# Production uniqueness: only one record per model_id in production.
# ---------------------------------------------------------------------------


def test_registry_prevents_multiple_production_versions_for_same_model() -> None:
    from alpha_research.ml_models.registry import ModelRegistry

    reg = ModelRegistry()
    reg.register(_record(version="v1"))
    reg.register(_record(version="v2"))
    reg.transition(model_id="kronos-v0", version="v1", to="production")
    with pytest.raises(ValueError, match="production"):
        reg.transition(model_id="kronos-v0", version="v2", to="production")
