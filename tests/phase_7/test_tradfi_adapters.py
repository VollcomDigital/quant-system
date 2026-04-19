"""Phase 7 Task 2 - TradFi gateway adapter contracts (Alpaca + IBKR).

Adapters wrap a vendor `BrokerClient` Protocol so concrete vendor SDKs
land behind a clean seam. Each adapter:

- Constructs vendor-shaped requests from a Phase 4 `OrderPayload`.
- Translates vendor responses into `OrderAck`.
- Translates vendor disconnects into `HealthStatus(ok=False)`.
- Records the vendor's order id alongside the local `order_id`.
- Has a `cancel_all` flow that maps to `reqGlobalCancel` (IBKR) or
  `DELETE /v2/orders` (Alpaca) at the contract level.

Phase 7 ships the adapter contracts + an in-memory fake `BrokerClient`.
Real vendor SDKs (alpaca-py, ib_insync) wire in Phase 9 with credentials
behind KMS / Vault.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest


def _payload(order_id: str = "o-1", side: str = "buy"):
    from backtest_engine.api import OrderPayload

    return OrderPayload(
        idempotency_key=f"idem-{order_id}",
        symbol="AAPL",
        side=side,  # type: ignore[arg-type]
        quantity=Decimal("10"),
        limit_price=Decimal("100"),
        time_in_force="day",
        placed_at=datetime(2026, 4, 19, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# Alpaca adapter
# ---------------------------------------------------------------------------


def test_alpaca_adapter_translates_payload_to_vendor_request() -> None:
    from trading_system.gateways.tradfi import AlpacaGateway, FakeBrokerClient

    client = FakeBrokerClient()
    gw = AlpacaGateway(client=client)
    ack = gw.submit(_payload())
    assert ack.accepted is True
    # Vendor sees side="buy", qty=10, limit_price=100; the adapter is the
    # only place that vendor-specific keys exist.
    assert client.calls[0]["op"] == "submit"
    assert client.calls[0]["body"]["side"] == "buy"
    assert client.calls[0]["body"]["qty"] == "10"


def test_alpaca_cancel_all_maps_to_delete_orders() -> None:
    from trading_system.gateways.tradfi import AlpacaGateway, FakeBrokerClient

    client = FakeBrokerClient()
    gw = AlpacaGateway(client=client)
    gw.submit(_payload(order_id="o-1"))
    gw.submit(_payload(order_id="o-2"))
    cancelled = gw.cancel_all()
    assert set(cancelled) == {"o-1", "o-2"}
    # The vendor shim records the cancel-all op explicitly.
    assert any(c["op"] == "cancel_all" for c in client.calls)


def test_alpaca_health_reports_vendor_disconnect() -> None:
    from trading_system.gateways.tradfi import AlpacaGateway, FakeBrokerClient

    client = FakeBrokerClient(connected=False)
    gw = AlpacaGateway(client=client)
    h = gw.health()
    assert h.ok is False


# ---------------------------------------------------------------------------
# IBKR adapter
# ---------------------------------------------------------------------------


def test_ibkr_adapter_translates_payload_to_vendor_request() -> None:
    from trading_system.gateways.tradfi import FakeBrokerClient, IBKRGateway

    client = FakeBrokerClient()
    gw = IBKRGateway(client=client)
    gw.submit(_payload())
    assert client.calls[0]["op"] == "submit"
    # IBKR uses lmtPrice + totalQuantity; the adapter is the only place
    # with that vocabulary.
    assert client.calls[0]["body"]["lmtPrice"] == "100"
    assert client.calls[0]["body"]["totalQuantity"] == "10"


def test_ibkr_cancel_all_maps_to_req_global_cancel() -> None:
    from trading_system.gateways.tradfi import FakeBrokerClient, IBKRGateway

    client = FakeBrokerClient()
    gw = IBKRGateway(client=client)
    gw.submit(_payload(order_id="o-1"))
    cancelled = gw.cancel_all()
    assert set(cancelled) == {"o-1"}
    assert any(c["op"] == "reqGlobalCancel" for c in client.calls)


def test_ibkr_adapter_refuses_unknown_cancel() -> None:
    from trading_system.gateways.tradfi import FakeBrokerClient, IBKRGateway

    gw = IBKRGateway(client=FakeBrokerClient())
    with pytest.raises(LookupError):
        gw.cancel("ghost")


# ---------------------------------------------------------------------------
# Adapter base contract: idempotency duplicates rejected at the boundary.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory_path",
    ["trading_system.gateways.tradfi.AlpacaGateway",
     "trading_system.gateways.tradfi.IBKRGateway"],
)
def test_adapter_refuses_duplicate_idempotency_key(factory_path: str) -> None:
    import importlib

    from trading_system.gateways.tradfi import FakeBrokerClient

    module_name, _, class_name = factory_path.rpartition(".")
    cls = getattr(importlib.import_module(module_name), class_name)
    gw = cls(client=FakeBrokerClient())
    gw.submit(_payload(order_id="o-1"))
    with pytest.raises(ValueError, match="idempotency"):
        gw.submit(_payload(order_id="o-1"))
