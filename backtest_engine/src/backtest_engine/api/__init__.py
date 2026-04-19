"""Engine API boundary.

`OrderPayload` is the wire shape that both the backtest simulator and
the Phase 6 `trading_system` OMS consume. A backtest can record the
sequence of payloads it emits and replay the exact bytes against the
OMS/gateway contracts in paper and live modes.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import Field
from shared_lib.contracts import Order
from shared_lib.contracts._base import Schema, aware_datetime_validator

__all__ = [
    "OrderPayload",
    "payload_to_order",
    "record_payloads",
    "replay_payloads",
]


OrderSide = Literal["buy", "sell"]
TimeInForce = Literal["day", "gtc", "ioc", "fok"]


class OrderPayload(Schema):
    idempotency_key: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    side: OrderSide
    quantity: Decimal = Field(gt=0)
    limit_price: Decimal | None = None
    time_in_force: TimeInForce
    placed_at: datetime

    _ts = aware_datetime_validator("placed_at")


def payload_to_order(payload: OrderPayload, *, order_id: str) -> Order:
    return Order(
        order_id=order_id,
        idempotency_key=payload.idempotency_key,
        symbol=payload.symbol,
        side=payload.side,
        quantity=payload.quantity,
        limit_price=payload.limit_price,
        time_in_force=payload.time_in_force,
        placed_at=payload.placed_at,
    )


def record_payloads(payloads: Sequence[OrderPayload]) -> bytes:
    """Record payloads as newline-delimited JSON (NDJSON)."""
    lines = [p.model_dump_json() for p in payloads]
    return ("\n".join(lines) + "\n").encode("utf-8")


def replay_payloads(buffer: bytes) -> list[OrderPayload]:
    """Replay payloads from NDJSON bytes.

    Raises `ValueError` on any malformed line so replay is fail-closed.
    """
    text = buffer.decode("utf-8")
    out: list[OrderPayload] = []
    for idx, raw in enumerate(text.splitlines()):
        if not raw.strip():
            continue
        try:
            json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"corrupt replay at line {idx}: {exc}") from exc
        out.append(OrderPayload.model_validate_json(raw))
    return out
