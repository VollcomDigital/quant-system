"""Phase 6 Task 1 - trading_system.oms.

- `OMS` tracks order state (new/acknowledged/partially_filled/filled/
  cancelled/rejected); transitions are explicit and unidirectional.
- `apply_fill` updates the internal `PositionBook` via Phase 4's
  `Portfolio` so cash and positions are Decimal-safe end-to-end.
- `reconcile` compares an authoritative broker snapshot with the local
  OMS and returns a structured diff.
- Duplicate orders (same idempotency_key) are refused at submission.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest


def _order(order_id: str = "o-1", idem: str = "i-1", side: str = "buy", qty: str = "10"):
    from shared_lib.contracts import Order

    return Order(
        order_id=order_id,
        idempotency_key=idem,
        symbol="AAPL",
        side=side,  # type: ignore[arg-type]
        quantity=Decimal(qty),
        limit_price=Decimal("100"),
        time_in_force="day",
        placed_at=datetime(2026, 4, 19, tzinfo=UTC),
    )


def _fill(order_id: str = "o-1", fill_id: str = "f-1", qty: str = "10", price: str = "100"):
    from shared_lib.contracts import Fill

    return Fill(
        fill_id=fill_id,
        order_id=order_id,
        symbol="AAPL",
        side="buy",
        quantity=Decimal(qty),
        price=Decimal(price),
        fee=Decimal("0"),
        currency="USD",
        filled_at=datetime(2026, 4, 19, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# Submit + duplicate idempotency
# ---------------------------------------------------------------------------


def test_oms_submit_new_order_is_acknowledged() -> None:
    from shared_lib.math_utils import Money
    from trading_system.oms import OMS

    oms = OMS(starting_cash=Money("10000", "USD"))
    oms.submit(_order())
    state = oms.get_state("o-1")
    assert state == "acknowledged"


def test_oms_refuses_duplicate_idempotency_key() -> None:
    from shared_lib.math_utils import Money
    from trading_system.oms import OMS

    oms = OMS(starting_cash=Money("10000", "USD"))
    oms.submit(_order(idem="i-1"))
    with pytest.raises(ValueError, match="idempotency"):
        oms.submit(_order(order_id="o-2", idem="i-1"))


def test_oms_rejects_unknown_state_transition() -> None:
    from shared_lib.math_utils import Money
    from trading_system.oms import OMS

    oms = OMS(starting_cash=Money("10000", "USD"))
    oms.submit(_order())
    # Cannot cancel after filled.
    oms.apply_fill(_fill(qty="10"))
    with pytest.raises(ValueError, match="transition"):
        oms.cancel("o-1")


# ---------------------------------------------------------------------------
# Fills and partial fills
# ---------------------------------------------------------------------------


def test_partial_fill_keeps_order_partially_filled() -> None:
    from shared_lib.math_utils import Money
    from trading_system.oms import OMS

    oms = OMS(starting_cash=Money("10000", "USD"))
    oms.submit(_order(qty="10"))
    oms.apply_fill(_fill(qty="4"))
    assert oms.get_state("o-1") == "partially_filled"
    oms.apply_fill(_fill(fill_id="f-2", qty="6"))
    assert oms.get_state("o-1") == "filled"


def test_fill_on_unknown_order_raises() -> None:
    from shared_lib.math_utils import Money
    from trading_system.oms import OMS

    oms = OMS(starting_cash=Money("10000", "USD"))
    with pytest.raises(LookupError):
        oms.apply_fill(_fill(order_id="ghost"))


def test_fill_exceeding_quantity_raises() -> None:
    from shared_lib.math_utils import Money
    from trading_system.oms import OMS

    oms = OMS(starting_cash=Money("10000", "USD"))
    oms.submit(_order(qty="5"))
    with pytest.raises(ValueError, match="exceeds"):
        oms.apply_fill(_fill(qty="10"))


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


def test_cancel_transitions_to_cancelled() -> None:
    from shared_lib.math_utils import Money
    from trading_system.oms import OMS

    oms = OMS(starting_cash=Money("10000", "USD"))
    oms.submit(_order())
    oms.cancel("o-1")
    assert oms.get_state("o-1") == "cancelled"


def test_cancel_twice_raises() -> None:
    from shared_lib.math_utils import Money
    from trading_system.oms import OMS

    oms = OMS(starting_cash=Money("10000", "USD"))
    oms.submit(_order())
    oms.cancel("o-1")
    with pytest.raises(ValueError):
        oms.cancel("o-1")


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def test_reconcile_returns_no_diff_when_in_sync() -> None:
    from shared_lib.math_utils import Money
    from trading_system.oms import OMS

    oms = OMS(starting_cash=Money("10000", "USD"))
    oms.submit(_order(qty="10"))
    oms.apply_fill(_fill(qty="10"))
    diff = oms.reconcile(broker_positions={"AAPL": Decimal("10")})
    assert diff.in_sync is True
    assert diff.missing_at_broker == {}
    assert diff.missing_at_local == {}


def test_reconcile_detects_missing_at_broker() -> None:
    from shared_lib.math_utils import Money
    from trading_system.oms import OMS

    oms = OMS(starting_cash=Money("10000", "USD"))
    oms.submit(_order(qty="10"))
    oms.apply_fill(_fill(qty="10"))
    diff = oms.reconcile(broker_positions={})
    assert diff.in_sync is False
    assert diff.missing_at_broker == {"AAPL": Decimal("10")}


def test_reconcile_detects_missing_at_local() -> None:
    from shared_lib.math_utils import Money
    from trading_system.oms import OMS

    oms = OMS(starting_cash=Money("10000", "USD"))
    diff = oms.reconcile(broker_positions={"MSFT": Decimal("5")})
    assert diff.in_sync is False
    assert diff.missing_at_local == {"MSFT": Decimal("5")}


def test_reconcile_required_before_new_order_placement() -> None:
    from shared_lib.math_utils import Money
    from trading_system.oms import OMS

    oms = OMS(starting_cash=Money("10000", "USD"), require_reconciliation=True)
    # Without a successful reconcile, new orders must be refused.
    with pytest.raises(RuntimeError, match="reconciliation"):
        oms.submit(_order())
    # After reconcile, order submission proceeds.
    oms.reconcile(broker_positions={})
    oms.submit(_order())
    assert oms.get_state("o-1") == "acknowledged"
