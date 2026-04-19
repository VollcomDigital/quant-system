"""Model registry contract.

MLflow/W&B-style lifecycle; Phase 3 ships the contract, Phase 9 wires
a backend. Every record binds model weights to a specific training run,
training-data snapshot, and evaluation metrics. Transitions are
explicit and append-only: Staging → Production → Archived.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from decimal import Decimal
from typing import Literal

__all__ = [
    "LifecycleStage",
    "ModelRecord",
    "ModelRegistry",
]


LifecycleStage = Literal["staging", "production", "archived"]
_STAGE_ORDER: dict[str, int] = {"staging": 0, "production": 1, "archived": 2}


@dataclass(frozen=True, slots=True)
class ModelRecord:
    model_id: str
    version: str
    training_run_id: str
    trained_at: datetime
    training_data_snapshot_id: str
    metrics: dict[str, Decimal]
    artifact_path: str
    lifecycle_stage: LifecycleStage

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if not self.version:
            raise ValueError("version must be non-empty")
        if not self.training_run_id:
            raise ValueError("training_run_id must be non-empty")
        if not self.training_data_snapshot_id:
            raise ValueError(
                "training_data_snapshot_id must be non-empty so every "
                "model binds to a reproducible dataset snapshot"
            )
        if self.trained_at.tzinfo is None:
            raise ValueError("trained_at must be tz-aware")
        if self.lifecycle_stage not in _STAGE_ORDER:
            raise ValueError(
                f"invalid lifecycle_stage: {self.lifecycle_stage!r}"
            )


@dataclass
class ModelRegistry:
    _records: dict[tuple[str, str], ModelRecord] = field(
        default_factory=dict, init=False, repr=False
    )

    def register(self, record: ModelRecord) -> None:
        key = (record.model_id, record.version)
        if key in self._records:
            raise ValueError(f"model {key} already registered")
        self._records[key] = record

    def get(self, *, model_id: str, version: str) -> ModelRecord:
        try:
            return self._records[(model_id, version)]
        except KeyError as exc:
            raise LookupError(
                f"no model registered for {(model_id, version)!r}"
            ) from exc

    def transition(
        self, *, model_id: str, version: str, to: LifecycleStage
    ) -> ModelRecord:
        current = self.get(model_id=model_id, version=version)
        from_stage = _STAGE_ORDER[current.lifecycle_stage]
        if to not in _STAGE_ORDER:
            raise ValueError(f"invalid target stage: {to!r}")
        to_stage = _STAGE_ORDER[to]
        if to_stage < from_stage:
            raise ValueError(
                f"reverse transition {current.lifecycle_stage} -> {to} is forbidden"
            )
        if to_stage - from_stage > 1:
            raise ValueError(
                f"cannot skip stages: {current.lifecycle_stage} -> {to}"
            )
        if to == "production":
            for (mid, _), rec in self._records.items():
                if mid == model_id and rec.lifecycle_stage == "production":
                    raise ValueError(
                        f"another version of {model_id!r} is already in production"
                    )
        new_record = replace(current, lifecycle_stage=to)
        self._records[(model_id, version)] = new_record
        return new_record
