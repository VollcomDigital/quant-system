"""Run metadata and job-status payloads."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import Field

from shared_lib.contracts._base import Schema, aware_datetime_validator

RunKind = Literal["backtest", "ingestion", "training", "agent_run", "live_deploy"]
JobState = Literal["pending", "running", "succeeded", "failed", "cancelled"]


class RunMetadata(Schema):
    run_id: str = Field(min_length=1)
    kind: RunKind
    started_at: datetime
    git_sha: str = Field(min_length=1)

    _ts = aware_datetime_validator("started_at")


class JobStatus(Schema):
    run_id: str = Field(min_length=1)
    state: JobState
    progress: Decimal = Field(ge=0, le=1)
