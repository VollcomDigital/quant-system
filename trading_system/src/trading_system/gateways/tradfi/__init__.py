"""TradFi gateway adapters.

Each adapter wraps a vendor `BrokerClient` Protocol so concrete vendor
SDKs (alpaca-py, ib_insync) plug in behind a clean seam. Phase 7 ships
the contract + an in-memory `FakeBrokerClient` for tests; the real
integrations land alongside KMS/Vault credential wiring in Phase 9.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from backtest_engine.api import OrderPayload
from shared_lib.contracts import HealthStatus

from trading_system.shared_gateways import GatewayOrder, OrderAck

__all__ = [
    "AlpacaGateway",
    "BrokerClient",
    "FakeBrokerClient",
    "IBKRGateway",
]


@runtime_checkable
class BrokerClient(Protocol):
    """Vendor-shim protocol every TradFi adapter consumes."""

    connected: bool

    def call(self, *, op: str, body: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class FakeBrokerClient:
    """In-memory `BrokerClient` for tests + paper-trading parity."""

    connected: bool = True
    calls: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def call(self, *, op: str, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"op": op, "body": body})
        return {"ok": True, "order_id": body.get("order_id")}


def _payload_to_alpaca(payload: OrderPayload, *, order_id: str) -> dict[str, Any]:
    return {
        "order_id": order_id,
        "symbol": payload.symbol,
        "side": payload.side,
        "qty": str(payload.quantity),
        "limit_price": str(payload.limit_price) if payload.limit_price else None,
        "time_in_force": payload.time_in_force,
        "client_order_id": payload.idempotency_key,
    }


def _payload_to_ibkr(payload: OrderPayload, *, order_id: str) -> dict[str, Any]:
    return {
        "order_id": order_id,
        "symbol": payload.symbol,
        "action": "BUY" if payload.side == "buy" else "SELL",
        "totalQuantity": str(payload.quantity),
        "lmtPrice": str(payload.limit_price) if payload.limit_price else None,
        "tif": payload.time_in_force.upper(),
    }


@dataclass
class _BaseTradFiGateway:
    """Common scaffolding shared by Alpaca + IBKR."""

    client: BrokerClient
    _orders: dict[str, GatewayOrder] = field(default_factory=dict, init=False, repr=False)
    _idem: set[str] = field(default_factory=set, init=False, repr=False)

    def open_orders(self) -> Iterator[GatewayOrder]:
        yield from self._orders.values()

    def health(self) -> HealthStatus:
        connected = bool(getattr(self.client, "connected", False))
        checks = {"vendor_connected": connected}
        return HealthStatus(
            service=self.__class__.__name__,
            ok=all(checks.values()),
            checks=checks,
        )


@dataclass
class AlpacaGateway(_BaseTradFiGateway):
    def submit(self, payload: OrderPayload) -> OrderAck:
        if payload.idempotency_key in self._idem:
            raise ValueError(
                f"duplicate idempotency_key: {payload.idempotency_key!r}"
            )
        order_id = payload.idempotency_key.removeprefix("idem-")
        self.client.call(op="submit", body=_payload_to_alpaca(payload, order_id=order_id))
        self._orders[order_id] = GatewayOrder(order_id=order_id, payload=payload)
        self._idem.add(payload.idempotency_key)
        return OrderAck(order_id=order_id, accepted=True)

    def cancel(self, order_id: str) -> OrderAck:
        if order_id not in self._orders:
            raise LookupError(f"unknown order: {order_id!r}")
        self.client.call(op="cancel", body={"order_id": order_id})
        self._orders.pop(order_id)
        return OrderAck(order_id=order_id, accepted=True)

    def cancel_all(self) -> tuple[str, ...]:
        ids = tuple(self._orders.keys())
        self.client.call(op="cancel_all", body={})
        self._orders.clear()
        return ids


@dataclass
class IBKRGateway(_BaseTradFiGateway):
    def submit(self, payload: OrderPayload) -> OrderAck:
        if payload.idempotency_key in self._idem:
            raise ValueError(
                f"duplicate idempotency_key: {payload.idempotency_key!r}"
            )
        order_id = payload.idempotency_key.removeprefix("idem-")
        self.client.call(op="submit", body=_payload_to_ibkr(payload, order_id=order_id))
        self._orders[order_id] = GatewayOrder(order_id=order_id, payload=payload)
        self._idem.add(payload.idempotency_key)
        return OrderAck(order_id=order_id, accepted=True)

    def cancel(self, order_id: str) -> OrderAck:
        if order_id not in self._orders:
            raise LookupError(f"unknown order: {order_id!r}")
        self.client.call(op="cancelOrder", body={"order_id": order_id})
        self._orders.pop(order_id)
        return OrderAck(order_id=order_id, accepted=True)

    def cancel_all(self) -> tuple[str, ...]:
        ids = tuple(self._orders.keys())
        self.client.call(op="reqGlobalCancel", body={})
        self._orders.clear()
        return ids
