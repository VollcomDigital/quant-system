"""Shared gateway protocols + SimulatedGateway reference.

ADR-0005 splits gateways into TradFi and Web3 paradigms; both honour the
same protocol so the Phase 6 EMS can talk to either through one seam.
The Phase 7 SimulatedGateway is the paper-trading reference impl that
the test harnesses + PanicPlaybook integration use.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from backtest_engine.api import OrderPayload
from shared_lib.contracts import HealthStatus

__all__ = [
    "Gateway",
    "GatewayOrder",
    "OrderAck",
    "SimulatedGateway",
]


@dataclass(frozen=True, slots=True)
class OrderAck:
    order_id: str
    accepted: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class GatewayOrder:
    order_id: str
    payload: OrderPayload


@runtime_checkable
class Gateway(Protocol):
    def submit(self, payload: OrderPayload) -> OrderAck: ...
    def cancel(self, order_id: str) -> OrderAck: ...
    def cancel_all(self) -> tuple[str, ...]: ...
    def open_orders(self) -> Iterator[GatewayOrder]: ...
    def health(self) -> HealthStatus: ...


@dataclass
class SimulatedGateway:
    """Paper-trading reference gateway."""

    _orders: dict[str, GatewayOrder] = field(
        default_factory=dict, init=False, repr=False
    )
    _idem: set[str] = field(default_factory=set, init=False, repr=False)
    _connected: bool = field(default=True, init=False, repr=False)

    def submit(self, payload: OrderPayload) -> OrderAck:
        if payload.idempotency_key in self._idem:
            raise ValueError(
                f"duplicate idempotency_key: {payload.idempotency_key!r}"
            )
        # In the simulator the order_id is derived from the idempotency key
        # tail when the caller doesn't dedupe; here we re-use it directly.
        order_id = payload.idempotency_key.removeprefix("idem-")
        self._orders[order_id] = GatewayOrder(order_id=order_id, payload=payload)
        self._idem.add(payload.idempotency_key)
        return OrderAck(order_id=order_id, accepted=True)

    def cancel(self, order_id: str) -> OrderAck:
        if order_id not in self._orders:
            raise LookupError(f"unknown order: {order_id!r}")
        self._orders.pop(order_id)
        return OrderAck(order_id=order_id, accepted=True)

    def cancel_all(self) -> tuple[str, ...]:
        ids = tuple(self._orders.keys())
        self._orders.clear()
        return ids

    def open_orders(self) -> Iterator[GatewayOrder]:
        yield from self._orders.values()

    def simulate_disconnect(self) -> None:
        self._connected = False

    def health(self) -> HealthStatus:
        checks = {"connected": self._connected, "queue_open": True}
        return HealthStatus(
            service="shared_gateways.simulated",
            ok=all(checks.values()),
            checks=checks,
        )
