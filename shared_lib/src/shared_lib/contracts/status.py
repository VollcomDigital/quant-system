"""UI-facing execution and health status payloads."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from shared_lib.contracts._base import Schema, aware_datetime_validator

ServiceState = Literal["starting", "running", "degraded", "halted", "stopped"]


class ExecutionStatus(Schema):
    service: str = Field(min_length=1)
    state: ServiceState
    last_heartbeat: datetime
    open_orders: int = Field(ge=0)
    pending_fills: int = Field(ge=0)

    _ts = aware_datetime_validator("last_heartbeat")


class HealthStatus(Schema):
    service: str = Field(min_length=1)
    ok: bool
    checks: dict[str, bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ok_matches_checks(self) -> HealthStatus:
        expected = all(self.checks.values()) if self.checks else True
        if self.ok != expected:
            raise ValueError(
                f"`ok` ({self.ok}) must equal all(checks.values()) ({expected})"
            )
        return self
