"""Execution Management System.

Turns parent orders into child orders via pluggable slicers; routes
children to gateway-compatible payloads. Smart-routing decisions (venue
selection) plug in at `OrderRouter`. Actual broker/exchange submission
lives behind Phase 7 gateway adapters.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from backtest_engine.api import OrderPayload
from shared_lib.contracts import Order

__all__ = ["EMS", "EqualSliceSchedule", "OrderRouter", "Slicer"]


@runtime_checkable
class Slicer(Protocol):
    def slice_order(self, parent: Order) -> Sequence[Order]:
        ...


@dataclass(frozen=True, slots=True)
class EqualSliceSchedule:
    num_slices: int

    def __post_init__(self) -> None:
        if self.num_slices <= 0:
            raise ValueError("num_slices must be > 0")

    def slice_order(self, parent: Order) -> Sequence[Order]:
        n = self.num_slices
        base = parent.quantity // n
        children: list[Order] = []
        for i in range(n):
            qty = base if i < n - 1 else parent.quantity - base * (n - 1)
            children.append(
                Order(
                    order_id=f"{parent.order_id}-{i+1:03d}",
                    idempotency_key=f"{parent.idempotency_key}-{i+1:03d}",
                    symbol=parent.symbol,
                    side=parent.side,
                    quantity=qty,
                    limit_price=parent.limit_price,
                    time_in_force=parent.time_in_force,
                    placed_at=parent.placed_at,
                )
            )
        return tuple(children)


class EMS:
    def schedule(self, parent: Order, *, slicer: Slicer) -> Sequence[Order]:
        return slicer.slice_order(parent)


class OrderRouter:
    """Projects child orders onto the shared `OrderPayload` wire shape."""

    def to_payload(self, order: Order) -> OrderPayload:
        return OrderPayload(
            idempotency_key=order.idempotency_key,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            limit_price=order.limit_price,
            time_in_force=order.time_in_force,
            placed_at=order.placed_at,
        )
