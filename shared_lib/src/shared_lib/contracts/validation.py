"""Validation result + anomaly event schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from shared_lib.contracts._base import Schema, aware_datetime_validator

Severity = Literal["info", "low", "medium", "high", "critical"]


class ValidationResult(Schema):
    check_id: str = Field(min_length=1)
    target: str = Field(min_length=1)
    passed: bool
    reason: str | None = None
    evaluated_at: datetime

    _ts = aware_datetime_validator("evaluated_at")

    @model_validator(mode="after")
    def _fail_requires_reason(self) -> ValidationResult:
        if not self.passed and not self.reason:
            raise ValueError("failed validations must include a reason")
        return self


class AnomalyEvent(Schema):
    anomaly_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    severity: Severity
    summary: str
    detected_at: datetime
    details: dict[str, str] = Field(default_factory=dict)

    _ts = aware_datetime_validator("detected_at")
