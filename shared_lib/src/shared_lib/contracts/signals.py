"""Trade signal schemas."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import Field

from shared_lib.contracts._base import Schema, aware_datetime_validator

Direction = Literal["long", "short", "flat"]


class TradeSignal(Schema):
    signal_id: str = Field(min_length=1)
    strategy_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    direction: Direction
    strength: Decimal = Field(ge=0, le=1)
    generated_at: datetime

    _ts = aware_datetime_validator("generated_at")
