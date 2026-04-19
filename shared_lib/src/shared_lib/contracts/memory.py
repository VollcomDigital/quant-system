"""Research memory records (MemPalace-inspired)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from shared_lib.contracts._base import Schema, aware_datetime_validator

MemoryKind = Literal[
    "factor_hypothesis",
    "experiment_rationale",
    "rejected_idea",
    "dataset_caveat",
    "review_finding",
]


class ResearchMemoryRecord(Schema):
    record_id: str = Field(min_length=1)
    kind: MemoryKind
    title: str = Field(min_length=1)
    body: str
    tags: tuple[str, ...] = ()
    created_at: datetime

    _ts = aware_datetime_validator("created_at")
