"""Approval request/decision schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from shared_lib.contracts._base import Schema, aware_datetime_validator

ApprovalSubject = Literal[
    "factor_promotion",
    "model_promotion",
    "strategy_activation",
    "treasury_transfer",
    "kill_switch_reset",
]
DecisionOutcome = Literal["approved", "rejected", "deferred"]


class ApprovalRequest(Schema):
    approval_id: str = Field(min_length=1)
    subject: ApprovalSubject
    target_id: str = Field(min_length=1)
    requested_by: str = Field(min_length=1)
    requested_at: datetime
    context: dict[str, str] = Field(default_factory=dict)

    _ts = aware_datetime_validator("requested_at")


class ApprovalDecision(Schema):
    approval_id: str = Field(min_length=1)
    decision: DecisionOutcome
    decided_by: str = Field(min_length=1)
    decided_at: datetime
    notes: str = ""

    _ts = aware_datetime_validator("decided_at")
