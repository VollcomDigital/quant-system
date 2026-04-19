"""Audit event + operator action schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from shared_lib.contracts._base import Schema, aware_datetime_validator


class AuditEvent(Schema):
    event_id: str = Field(min_length=1)
    actor: str = Field(min_length=1)
    action: str = Field(min_length=1)
    target: str = Field(min_length=1)
    at: datetime
    trace_id: str | None = None
    details: dict[str, str] = Field(default_factory=dict)

    _ts = aware_datetime_validator("at")
