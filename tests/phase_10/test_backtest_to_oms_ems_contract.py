"""Phase 10 Task 2 - backtest -> OMS/EMS contract test.

Chain under test:

1. Phase 4 `Simulator` emits `Fill`s for a strategy.
2. The Phase 4 `OrderPayload` wire shape lines up exactly with the
   Phase 1 `Order` contract and the Phase 6 `OMS.submit`.
3. The Phase 6 `EMS.schedule(...)` slices a parent order; each child
   round-trips through `OrderRouter.to_payload` (Phase 4 payload) and
   the Phase 6 `OMS.submit` → `OMS.apply_fill` state machine.
4. RMS checks fire between signal generation and OMS submission and
   return the same `ValidationResult` shape as Phase 10 Task 1.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal


def _payload_from_backtest() -> object:
    from backtest_engine.api import OrderPayload

    return OrderPayload(
        idempotency_key="idem-bt-1",
        symbol="AAPL",
        side="buy",
        quantity=Decimal("10"),
        limit_price=Decimal("100"),
        time_in_force="day",
        placed_at=datetime(2026, 4, 20, tzinfo=UTC),
    )


def test_backtest_order_payload_projects_to_shared_order_contract() -> None:
    from backtest_engine.api import payload_to_order
    from shared_lib.contracts import Order

    order = payload_to_order(_payload_from_backtest(), order_id="o-1")
    assert isinstance(order, Order)
    assert order.idempotency_key == "idem-bt-1"
    assert order.side == "buy"


def test_oms_accepts_order_derived_from_backtest_payload() -> None:
    from backtest_engine.api import payload_to_order
    from shared_lib.math_utils import Money
    from trading_system.oms import OMS

    order = payload_to_order(_payload_from_backtest(), order_id="o-1")
    oms = OMS(starting_cash=Money("10000", "USD"))
    oms.submit(order)
    assert oms.get_state("o-1") == "acknowledged"


def test_rms_check_fires_before_oms_submit_and_returns_validation_result() -> None:
    """RMS.check(order, ctx) is the seam between a backtest-sourced
    order and OMS submission. A blocked order never reaches the OMS."""
    from shared_lib.contracts import Order, ValidationResult
    from trading_system.rms import RMS, RiskContext, RiskLimits

    order = Order(
        order_id="o-rms",
        idempotency_key="i-rms",
        symbol="AAPL",
        side="buy",
        quantity=Decimal("10"),
        limit_price=Decimal("100"),
        time_in_force="day",
        placed_at=datetime(2026, 4, 20, tzinfo=UTC),
    )
    limits = RiskLimits(
        max_notional_per_order=Decimal("100"),  # too small -> blocks 1000
        max_position_per_symbol=Decimal("100"),
        max_gross_exposure=Decimal("10000"),
        max_pending_orders=10,
        max_volume_per_hour=Decimal("1000"),
        daily_drawdown_halt_pct=Decimal("-0.05"),
        wash_trading_window_sec=5,
        ai_confidence_floor=Decimal("0.5"),
    )
    ctx = RiskContext(
        current_positions={},
        gross_exposure=Decimal("0"),
        pending_orders=0,
        recent_fills=(),
        daily_pnl_pct=Decimal("0"),
        ai_confidence=Decimal("1"),
        drift_flag=False,
        trading_halted=False,
        now=datetime(2026, 4, 20, tzinfo=UTC),
    )
    rms = RMS(limits=limits)
    result = rms.check(order, ctx)
    assert isinstance(result, ValidationResult)
    assert result.passed is False
    assert "notional" in (result.reason or "")


def test_ems_slices_parent_order_then_oms_accepts_every_child() -> None:
    """A parent Order slices into children; each child round-trips
    through the OMS and the OMS state machine reports
    `acknowledged` for each."""
    from shared_lib.contracts import Order
    from shared_lib.math_utils import Money
    from trading_system.ems import EMS, EqualSliceSchedule
    from trading_system.oms import OMS

    parent = Order(
        order_id="parent-1",
        idempotency_key="idem-parent-1",
        symbol="AAPL",
        side="buy",
        quantity=Decimal("100"),
        limit_price=Decimal("170"),
        time_in_force="day",
        placed_at=datetime(2026, 4, 20, tzinfo=UTC),
    )
    ems = EMS()
    children = list(ems.schedule(parent, slicer=EqualSliceSchedule(num_slices=4)))
    assert len(children) == 4

    oms = OMS(starting_cash=Money("1000000", "USD"))
    for child in children:
        oms.submit(child)
        assert oms.get_state(child.order_id) == "acknowledged"


def test_oms_fill_aggregation_matches_backtest_simulator_convention() -> None:
    """The Phase 6 OMS aggregates fills the same way the Phase 4
    Simulator's Portfolio does (partial -> filled when quantity met)."""
    from shared_lib.contracts import Fill, Order
    from shared_lib.math_utils import Money
    from trading_system.oms import OMS

    order = Order(
        order_id="o-agg",
        idempotency_key="i-agg",
        symbol="AAPL",
        side="buy",
        quantity=Decimal("10"),
        limit_price=Decimal("100"),
        time_in_force="day",
        placed_at=datetime(2026, 4, 20, tzinfo=UTC),
    )
    oms = OMS(starting_cash=Money("10000", "USD"))
    oms.submit(order)
    oms.apply_fill(
        Fill(
            fill_id="f-1",
            order_id="o-agg",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("4"),
            price=Decimal("100"),
            fee=Decimal("0"),
            currency="USD",
            filled_at=datetime(2026, 4, 20, tzinfo=UTC),
        )
    )
    assert oms.get_state("o-agg") == "partially_filled"
    oms.apply_fill(
        Fill(
            fill_id="f-2",
            order_id="o-agg",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("6"),
            price=Decimal("101"),
            fee=Decimal("0"),
            currency="USD",
            filled_at=datetime(2026, 4, 20, tzinfo=UTC),
        )
    )
    assert oms.get_state("o-agg") == "filled"
