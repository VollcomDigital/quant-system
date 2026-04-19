"""Factor frame schemas."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import Field

from shared_lib.contracts._base import Schema, aware_datetime_validator


class FactorRecord(Schema):
    factor_id: str = Field(min_length=1)
    as_of: datetime
    symbol: str = Field(min_length=1)
    value: Decimal
    version: str = Field(min_length=1)

    _ts = aware_datetime_validator("as_of")
