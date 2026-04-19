"""Phase 4 Task 8 - Engine API boundary + exact order-payload replay.

The backtest engine and the live `trading_system` must share the same
`OrderPayload` shape so a simulated fill can be replayed byte-for-byte
against the gateway contract in paper/live modes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

# ---------------------------------------------------------------------------
# OrderPayload: the shared wire shape.
# ---------------------------------------------------------------------------


def test_order_payload_round_trip_is_byte_stable() -> None:
    from backtest_engine.api import OrderPayload

    p = OrderPayload(
        idempotency_key="idem-1",
        symbol="AAPL",
        side="buy",
        quantity=Decimal("100"),
        limit_price=Decimal("170.50"),
        time_in_force="day",
        placed_at=datetime(2026, 4, 19, 14, tzinfo=UTC),
    )
    json_str = p.model_dump_json()
    assert OrderPayload.model_validate_json(json_str) == p


def test_order_payload_equivalent_to_order_contract() -> None:
    """The engine payload must be convertible to the shared Order contract so
    the same data can be replayed through Phase 6 OMS."""
    from backtest_engine.api import OrderPayload, payload_to_order

    p = OrderPayload(
        idempotency_key="idem-1",
        symbol="AAPL",
        side="buy",
        quantity=Decimal("100"),
        limit_price=Decimal("170.50"),
        time_in_force="day",
        placed_at=datetime(2026, 4, 19, 14, tzinfo=UTC),
    )
    order = payload_to_order(p, order_id="o-9000")
    assert order.order_id == "o-9000"
    assert order.idempotency_key == "idem-1"
    assert order.side == "buy"


def test_order_payload_rejects_zero_quantity() -> None:
    from backtest_engine.api import OrderPayload

    with pytest.raises(ValueError):
        OrderPayload(
            idempotency_key="x",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("0"),
            limit_price=None,
            time_in_force="day",
            placed_at=datetime(2026, 4, 19, tzinfo=UTC),
        )


def test_order_payload_rejects_naive_placed_at() -> None:
    from backtest_engine.api import OrderPayload

    with pytest.raises(ValueError):
        OrderPayload(
            idempotency_key="x",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("1"),
            limit_price=None,
            time_in_force="day",
            placed_at=datetime(2026, 4, 19),
        )


# ---------------------------------------------------------------------------
# Replay: a recorded sequence of order payloads must reproduce identically.
# ---------------------------------------------------------------------------


def test_payload_replay_is_byte_stable() -> None:
    from backtest_engine.api import OrderPayload, record_payloads, replay_payloads

    originals = [
        OrderPayload(
            idempotency_key=f"idem-{i}",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("10"),
            limit_price=Decimal("100"),
            time_in_force="day",
            placed_at=datetime(2026, 4, 19, 14, i, tzinfo=UTC),
        )
        for i in range(3)
    ]
    buf = record_payloads(originals)
    replayed = replay_payloads(buf)
    assert replayed == originals


def test_replay_refuses_corrupted_line() -> None:
    from backtest_engine.api import replay_payloads

    with pytest.raises(ValueError):
        replay_payloads(b"{not_json}\n")
