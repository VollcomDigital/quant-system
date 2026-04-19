"""Market-data bar schema."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import Field, model_validator

from shared_lib.contracts._base import Schema, aware_datetime_validator


class Bar(Schema):
    symbol: str = Field(min_length=1)
    interval: str = Field(min_length=1)
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal = Field(ge=0)

    _ts = aware_datetime_validator("timestamp")

    @model_validator(mode="after")
    def _ohlc_invariants(self) -> Bar:
        if self.high < self.low:
            raise ValueError("high must be >= low")
        if self.high < self.open or self.high < self.close:
            raise ValueError("high must be >= open and close")
        if self.low > self.open or self.low > self.close:
            raise ValueError("low must be <= open and close")
        return self
