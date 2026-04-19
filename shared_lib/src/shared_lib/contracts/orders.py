"""Order / fill / position schemas."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import Field

from shared_lib.contracts._base import Schema, aware_datetime_validator

OrderSide = Literal["buy", "sell"]
TimeInForce = Literal["day", "gtc", "ioc", "fok"]


class Order(Schema):
    order_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    side: OrderSide
    quantity: Decimal = Field(gt=0)
    limit_price: Decimal | None = None
    time_in_force: TimeInForce
    placed_at: datetime

    _ts = aware_datetime_validator("placed_at")


class Fill(Schema):
    fill_id: str = Field(min_length=1)
    order_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    side: OrderSide
    quantity: Decimal = Field(gt=0)
    price: Decimal = Field(gt=0)
    fee: Decimal = Field(ge=0)
    currency: str = Field(min_length=1)
    filled_at: datetime

    _ts = aware_datetime_validator("filled_at")


class Position(Schema):
    symbol: str = Field(min_length=1)
    quantity: Decimal
    avg_price: Decimal = Field(ge=0)
    currency: str = Field(min_length=1)
    as_of: datetime

    _ts = aware_datetime_validator("as_of")
