"""Phase 7 Task 1 - shared_gateways base + SimulatedGateway.

Every gateway exposes the same surface:

- `submit(OrderPayload)` -> gateway-side `OrderAck`.
- `cancel(order_id)` -> ack.
- `cancel_all()` -> set of cancelled order_ids (used by Phase 6
  PanicPlaybook).
- `health()` -> structured `GatewayHealth`.
- `replay()` (for the paper-trading SimulatedGateway).

The Gateway protocol is the seam between the Phase 6 EMS and the
Phase 7 broker/Web3 adapters.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest


def _payload(order_id: str = "o-1", side: str = "buy", qty: str = "10"):
    from backtest_engine.api import OrderPayload

    return OrderPayload(
        idempotency_key=f"idem-{order_id}",
        symbol="AAPL",
        side=side,  # type: ignore[arg-type]
        quantity=Decimal(qty),
        limit_price=Decimal("100"),
        time_in_force="day",
        placed_at=datetime(2026, 4, 19, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------


def test_simulated_gateway_submit_returns_acknowledged_ack() -> None:
    from trading_system.shared_gateways import SimulatedGateway

    gw = SimulatedGateway()
    ack = gw.submit(_payload())
    assert ack.accepted is True
    assert ack.order_id == "o-1" or ack.order_id  # gateway may rewrite


def test_simulated_gateway_refuses_duplicate_idempotency_key() -> None:
    from trading_system.shared_gateways import SimulatedGateway

    gw = SimulatedGateway()
    gw.submit(_payload(order_id="o-1"))
    with pytest.raises(ValueError, match="idempotency"):
        gw.submit(_payload(order_id="o-1"))


def test_simulated_gateway_tracks_open_orders() -> None:
    from trading_system.shared_gateways import SimulatedGateway

    gw = SimulatedGateway()
    gw.submit(_payload(order_id="o-1"))
    gw.submit(_payload(order_id="o-2"))
    assert {o.order_id for o in gw.open_orders()} == {"o-1", "o-2"}


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


def test_simulated_gateway_cancel_removes_order() -> None:
    from trading_system.shared_gateways import SimulatedGateway

    gw = SimulatedGateway()
    gw.submit(_payload(order_id="o-1"))
    gw.cancel("o-1")
    assert {o.order_id for o in gw.open_orders()} == set()


def test_simulated_gateway_cancel_all_returns_cancelled_ids() -> None:
    from trading_system.shared_gateways import SimulatedGateway

    gw = SimulatedGateway()
    gw.submit(_payload(order_id="o-1"))
    gw.submit(_payload(order_id="o-2"))
    cancelled = gw.cancel_all()
    assert set(cancelled) == {"o-1", "o-2"}
    assert list(gw.open_orders()) == []


def test_cancel_unknown_order_raises() -> None:
    from trading_system.shared_gateways import SimulatedGateway

    gw = SimulatedGateway()
    with pytest.raises(LookupError):
        gw.cancel("ghost")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_gateway_health_default_is_running() -> None:
    from trading_system.shared_gateways import SimulatedGateway

    gw = SimulatedGateway()
    h = gw.health()
    assert h.ok is True
    assert h.checks  # at least one check declared


def test_gateway_health_marks_degraded_after_simulated_disconnect() -> None:
    from trading_system.shared_gateways import SimulatedGateway

    gw = SimulatedGateway()
    gw.simulate_disconnect()
    h = gw.health()
    assert h.ok is False


# ---------------------------------------------------------------------------
# Cancel-all callable for Phase 6 PanicPlaybook integration.
# ---------------------------------------------------------------------------


def test_panic_playbook_uses_simulated_gateway_cancel_all() -> None:
    from datetime import UTC, datetime

    from trading_system.kill_switch import KillSwitch, PanicPlaybook
    from trading_system.shared_gateways import SimulatedGateway

    gw = SimulatedGateway()
    gw.submit(_payload(order_id="o-1"))
    gw.submit(_payload(order_id="o-2"))

    ks = KillSwitch()
    pb = PanicPlaybook(kill_switch=ks)
    result = pb.execute(
        reason="phase 7 wiring test",
        actor="operator",
        at=datetime(2026, 4, 19, tzinfo=UTC),
        cancel_all_orders=gw.cancel_all,
    )
    assert set(result.cancelled_orders) == {"o-1", "o-2"}
    assert ks.trading_halted is True
