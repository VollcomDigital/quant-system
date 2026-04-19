"""Promotion gates.

Two flows are staged:

1. Factor: `candidate` -> `validated` -> `promoted` -> `retired`.
2. Model:  `staging`   -> `production` -> `archived`.

Both return a `shared_lib.contracts.ValidationResult` so the Phase 5
approval workflow and the Phase 5 code_reviewer agent can surface
failures uniformly. Skipping stages is forbidden; each transition has
its own gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal

from shared_lib.contracts import ValidationResult

from alpha_research.factor_library import FactorMetadata
from alpha_research.ml_models.registry import ModelRecord

__all__ = [
    "FactorPromotionRequest",
    "ModelPromotionRequest",
    "promote_factor",
    "promote_model",
]


FactorTargetStatus = Literal["validated", "promoted", "retired"]
ModelTargetStage = Literal["production", "archived"]

_FACTOR_ORDER = {"candidate": 0, "validated": 1, "promoted": 2, "retired": 3}
_MODEL_ORDER = {"staging": 0, "production": 1, "archived": 2}

COVERAGE_THRESHOLD = Decimal("0.8")
MIN_OOS_SHARPE = Decimal("0.5")


@dataclass(frozen=True, slots=True)
class FactorPromotionRequest:
    metadata: FactorMetadata
    target_status: FactorTargetStatus
    coverage_pct: Decimal
    oos_sharpe: Decimal | None
    leakage_check_passed: bool

    def __post_init__(self) -> None:
        if self.coverage_pct < Decimal("0") or self.coverage_pct > Decimal("1"):
            raise ValueError("coverage_pct must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class ModelPromotionRequest:
    record: ModelRecord
    target_stage: ModelTargetStage
    validation_passed: bool
    drift_check_passed: bool
    approval_id: str | None


def _vr(check_id: str, target: str, passed: bool, reason: str | None) -> ValidationResult:
    return ValidationResult(
        check_id=check_id,
        target=target,
        passed=passed,
        reason=reason,
        evaluated_at=datetime.now(tz=UTC),
    )


def promote_factor(req: FactorPromotionRequest) -> ValidationResult:
    target = f"factor:{req.metadata.factor_id}@{req.metadata.version}"
    current = req.metadata.validation_status
    if current not in _FACTOR_ORDER or req.target_status not in _FACTOR_ORDER:
        return _vr("factor_promotion", target, False, "invalid status values")

    if _FACTOR_ORDER[req.target_status] - _FACTOR_ORDER[current] != 1:
        return _vr(
            "factor_promotion",
            target,
            False,
            f"cannot skip stages: {current} -> {req.target_status}",
        )

    if not req.leakage_check_passed:
        return _vr(
            "factor_promotion",
            target,
            False,
            "leakage_check_passed=False blocks promotion",
        )

    if req.coverage_pct < COVERAGE_THRESHOLD:
        return _vr(
            "factor_promotion",
            target,
            False,
            f"coverage_pct={req.coverage_pct} below threshold {COVERAGE_THRESHOLD}",
        )

    if req.target_status in {"validated", "promoted"}:
        if req.oos_sharpe is None:
            return _vr(
                "factor_promotion",
                target,
                False,
                "missing oos_sharpe evidence",
            )
        if req.oos_sharpe < MIN_OOS_SHARPE:
            return _vr(
                "factor_promotion",
                target,
                False,
                f"oos_sharpe={req.oos_sharpe} below minimum {MIN_OOS_SHARPE}",
            )

    return _vr("factor_promotion", target, True, None)


def promote_model(req: ModelPromotionRequest) -> ValidationResult:
    target = f"model:{req.record.model_id}@{req.record.version}"
    current = req.record.lifecycle_stage

    if _MODEL_ORDER[req.target_stage] - _MODEL_ORDER[current] != 1:
        return _vr(
            "model_promotion",
            target,
            False,
            f"cannot skip stages: {current} -> {req.target_stage}",
        )

    if not req.validation_passed:
        return _vr(
            "model_promotion",
            target,
            False,
            "validation_passed=False blocks promotion",
        )
    if not req.drift_check_passed:
        return _vr(
            "model_promotion",
            target,
            False,
            "drift_check_passed=False blocks promotion",
        )
    if req.target_stage == "production" and not req.approval_id:
        return _vr(
            "model_promotion",
            target,
            False,
            "production promotion requires approval_id",
        )

    return _vr("model_promotion", target, True, None)
