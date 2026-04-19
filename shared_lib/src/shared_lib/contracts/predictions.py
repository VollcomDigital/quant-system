"""Prediction artifact schemas."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import Field

from shared_lib.contracts._base import Schema, aware_datetime_validator


class PredictionArtifact(Schema):
    """A single model prediction persisted by the data platform."""

    model_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    horizon: str = Field(min_length=1)
    generated_at: datetime
    value: Decimal
    confidence: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))

    _ts = aware_datetime_validator("generated_at")
