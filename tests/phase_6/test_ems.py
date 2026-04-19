"""Phase 6 Task 3 - trading_system.ems execution scheduler + slicing.

- `EMS.schedule(parent_order, slicer)` returns a sequence of child
  orders that sum to the parent quantity.
- `Slicer` protocol: take `(parent_order, now)` -> list of child orders.
- `EqualSliceSchedule(num_slices)` splits qty evenly; the last slice
  absorbs the remainder.
- `OrderRouter.route(child_order, gateway)` returns a payload the
  gateway can submit; the EMS never calls the gateway directly (the
  OMS/gateway integration owns submission in Phase 7).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest


def _parent(qty: str = "1000"):
    from shared_lib.contracts import Order

    return Order(
        order_id="parent-1",
        idempotency_key="idem-parent-1",
        symbol="AAPL",
        side="buy",
        quantity=Decimal(qty),
        limit_price=Decimal("170"),
        time_in_force="day",
        placed_at=datetime(2026, 4, 19, 14, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# EqualSliceSchedule
# ---------------------------------------------------------------------------


def test_equal_slice_schedule_splits_evenly() -> None:
    from trading_system.ems import EqualSliceSchedule

    slicer = EqualSliceSchedule(num_slices=4)
    slices = slicer.slice_order(_parent(qty="100"))
    assert len(slices) == 4
    assert [s.quantity for s in slices] == [Decimal("25")] * 4


def test_equal_slice_schedule_absorbs_remainder_into_last() -> None:
    from trading_system.ems import EqualSliceSchedule

    slicer = EqualSliceSchedule(num_slices=3)
    slices = slicer.slice_order(_parent(qty="100"))
    # 33 + 33 + 34
    assert sum((s.quantity for s in slices), start=Decimal("0")) == Decimal("100")
    assert slices[-1].quantity == Decimal("34")


def test_equal_slice_schedule_rejects_non_positive_slices() -> None:
    from trading_system.ems import EqualSliceSchedule

    with pytest.raises(ValueError):
        EqualSliceSchedule(num_slices=0)


def test_equal_slice_schedule_child_order_ids_are_unique() -> None:
    from trading_system.ems import EqualSliceSchedule

    slicer = EqualSliceSchedule(num_slices=3)
    slices = slicer.slice_order(_parent(qty="30"))
    ids = {s.order_id for s in slices}
    assert len(ids) == 3
    idem = {s.idempotency_key for s in slices}
    assert len(idem) == 3


# ---------------------------------------------------------------------------
# EMS
# ---------------------------------------------------------------------------


def test_ems_schedule_produces_children_summing_to_parent() -> None:
    from trading_system.ems import EMS, EqualSliceSchedule

    ems = EMS()
    children = ems.schedule(_parent(qty="90"), slicer=EqualSliceSchedule(num_slices=3))
    total = sum((c.quantity for c in children), start=Decimal("0"))
    assert total == Decimal("90")


def test_ems_schedule_preserves_side_and_symbol_on_children() -> None:
    from trading_system.ems import EMS, EqualSliceSchedule

    ems = EMS()
    children = ems.schedule(_parent(), slicer=EqualSliceSchedule(num_slices=2))
    assert all(c.symbol == "AAPL" for c in children)
    assert all(c.side == "buy" for c in children)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_router_projects_child_order_to_payload() -> None:
    from trading_system.ems import OrderRouter

    router = OrderRouter()
    child = _parent(qty="10")
    payload = router.to_payload(child)
    # Uses the Phase 4 backtest_engine.api.OrderPayload contract.
    assert payload.idempotency_key == child.idempotency_key
    assert payload.quantity == child.quantity
    assert payload.symbol == child.symbol
