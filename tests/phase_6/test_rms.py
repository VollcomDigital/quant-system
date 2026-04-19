"""Phase 6 Task 2 - trading_system.rms deterministic pre-trade risk engine.

`RMS.check(order, context)` returns a `ValidationResult`. The context
carries the live state the RMS needs: current positions, daily PnL,
order-rate window, AI confidence, drift flag, and the global halt
flag. All checks are deterministic - no probabilistic layer.

Checks Phase 6 must enforce:

- max notional per order (fat-finger)
- max position per symbol
- max gross exposure
- max pending order count
- max volume per hour
- daily drawdown kill-switch
- wash-trading (opposite-side in <N seconds)
- AI-side: confidence floor, drift halt, model circuit breaker
- global `TRADING_HALTED` flag
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest


def _order(qty: str = "10", side: str = "buy", price: str = "100"):
    from shared_lib.contracts import Order

    return Order(
        order_id="o-1",
        idempotency_key="i-1",
        symbol="AAPL",
        side=side,  # type: ignore[arg-type]
        quantity=Decimal(qty),
        limit_price=Decimal(price),
        time_in_force="day",
        placed_at=datetime(2026, 4, 19, 14, tzinfo=UTC),
    )


def _ctx(**kwargs):
    from trading_system.rms import RiskContext

    defaults = dict(
        current_positions={"AAPL": Decimal("0")},
        gross_exposure=Decimal("0"),
        pending_orders=0,
        recent_fills=(),
        daily_pnl_pct=Decimal("0"),
        ai_confidence=Decimal("1"),
        drift_flag=False,
        trading_halted=False,
        now=datetime(2026, 4, 19, 14, tzinfo=UTC),
    )
    defaults.update(kwargs)
    return RiskContext(**defaults)


def _limits(**kwargs):
    from trading_system.rms import RiskLimits

    defaults = dict(
        max_notional_per_order=Decimal("5000"),
        max_position_per_symbol=Decimal("1000"),
        max_gross_exposure=Decimal("100000"),
        max_pending_orders=5,
        max_volume_per_hour=Decimal("10000"),
        daily_drawdown_halt_pct=Decimal("-0.05"),
        wash_trading_window_sec=5,
        ai_confidence_floor=Decimal("0.6"),
    )
    defaults.update(kwargs)
    return RiskLimits(**defaults)


# ---------------------------------------------------------------------------
# Fat-finger notional
# ---------------------------------------------------------------------------


def test_fat_finger_notional_blocks_oversized_order() -> None:
    from trading_system.rms import RMS

    # Order notional = 10 * 100 = 1000. Limit = 500 -> blocks.
    rms = RMS(limits=_limits(max_notional_per_order=Decimal("500")))
    result = rms.check(_order(qty="10", price="100"), _ctx())
    assert result.passed is False
    assert "notional" in (result.reason or "")


def test_fat_finger_notional_passes_within_cap() -> None:
    from trading_system.rms import RMS

    rms = RMS(limits=_limits(max_notional_per_order=Decimal("2000")))
    assert rms.check(_order(qty="10", price="100"), _ctx()).passed is True


# ---------------------------------------------------------------------------
# Position caps
# ---------------------------------------------------------------------------


def test_position_cap_blocks_when_projected_exceeds_limit() -> None:
    from trading_system.rms import RMS

    rms = RMS(limits=_limits(max_position_per_symbol=Decimal("15")))
    ctx = _ctx(current_positions={"AAPL": Decimal("10")})
    # Buy 10 more -> projected 20 > 15.
    result = rms.check(_order(qty="10"), ctx)
    assert result.passed is False
    assert "position" in (result.reason or "")


def test_sell_reduces_position_and_does_not_hit_cap() -> None:
    from trading_system.rms import RMS

    rms = RMS(limits=_limits(max_position_per_symbol=Decimal("15")))
    ctx = _ctx(current_positions={"AAPL": Decimal("10")})
    assert rms.check(_order(qty="5", side="sell"), ctx).passed is True


# ---------------------------------------------------------------------------
# Pending-order cap
# ---------------------------------------------------------------------------


def test_pending_orders_cap_blocks_at_limit() -> None:
    from trading_system.rms import RMS

    rms = RMS(limits=_limits(max_pending_orders=3))
    result = rms.check(_order(), _ctx(pending_orders=3))
    assert result.passed is False
    assert "pending" in (result.reason or "")


# ---------------------------------------------------------------------------
# Daily drawdown halt
# ---------------------------------------------------------------------------


def test_daily_drawdown_halt_blocks_new_orders() -> None:
    from trading_system.rms import RMS

    rms = RMS(limits=_limits(daily_drawdown_halt_pct=Decimal("-0.05")))
    result = rms.check(_order(), _ctx(daily_pnl_pct=Decimal("-0.06")))
    assert result.passed is False
    assert "drawdown" in (result.reason or "")


# ---------------------------------------------------------------------------
# Wash trading
# ---------------------------------------------------------------------------


def test_wash_trading_blocks_opposite_side_within_window() -> None:
    from trading_system.rms import RMS

    now = datetime(2026, 4, 19, 14, tzinfo=UTC)
    from shared_lib.contracts import Fill

    recent = (
        Fill(
            fill_id="f-1",
            order_id="o-prior",
            symbol="AAPL",
            side="sell",
            quantity=Decimal("1"),
            price=Decimal("100"),
            fee=Decimal("0"),
            currency="USD",
            filled_at=now - timedelta(seconds=2),
        ),
    )
    rms = RMS(limits=_limits(wash_trading_window_sec=5))
    # Buy after recent sell within 5s -> wash trading.
    result = rms.check(_order(side="buy"), _ctx(recent_fills=recent, now=now))
    assert result.passed is False
    assert "wash" in (result.reason or "")


def test_wash_trading_allows_outside_window() -> None:
    from shared_lib.contracts import Fill
    from trading_system.rms import RMS

    now = datetime(2026, 4, 19, 14, tzinfo=UTC)
    recent = (
        Fill(
            fill_id="f-1",
            order_id="o-prior",
            symbol="AAPL",
            side="sell",
            quantity=Decimal("1"),
            price=Decimal("100"),
            fee=Decimal("0"),
            currency="USD",
            filled_at=now - timedelta(seconds=60),
        ),
    )
    rms = RMS(limits=_limits(wash_trading_window_sec=5))
    assert rms.check(_order(side="buy"), _ctx(recent_fills=recent, now=now)).passed is True


# ---------------------------------------------------------------------------
# Volume per hour
# ---------------------------------------------------------------------------


def test_volume_per_hour_blocks_when_over_cap() -> None:
    from shared_lib.contracts import Fill
    from trading_system.rms import RMS

    now = datetime(2026, 4, 19, 14, tzinfo=UTC)
    recent = tuple(
        Fill(
            fill_id=f"f-{i}",
            order_id=f"o-{i}",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("2000"),
            price=Decimal("100"),
            fee=Decimal("0"),
            currency="USD",
            filled_at=now - timedelta(minutes=10 * i),
        )
        for i in range(5)
    )
    # Total ~10000 units already; cap = 10000; new order of 100 units pushes over.
    rms = RMS(limits=_limits(max_volume_per_hour=Decimal("10000")))
    result = rms.check(_order(qty="100"), _ctx(recent_fills=recent, now=now))
    assert result.passed is False


# ---------------------------------------------------------------------------
# AI-specific halts
# ---------------------------------------------------------------------------


def test_ai_confidence_below_floor_blocks() -> None:
    from trading_system.rms import RMS

    rms = RMS(limits=_limits(ai_confidence_floor=Decimal("0.6")))
    assert (
        rms.check(_order(), _ctx(ai_confidence=Decimal("0.4"))).passed is False
    )


def test_drift_flag_blocks() -> None:
    from trading_system.rms import RMS

    rms = RMS(limits=_limits())
    assert rms.check(_order(), _ctx(drift_flag=True)).passed is False


def test_trading_halted_blocks_all_orders() -> None:
    from trading_system.rms import RMS

    rms = RMS(limits=_limits())
    assert rms.check(_order(), _ctx(trading_halted=True)).passed is False


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_passes_all_checks() -> None:
    from trading_system.rms import RMS

    rms = RMS(limits=_limits())
    assert rms.check(_order(), _ctx()).passed is True


# ---------------------------------------------------------------------------
# Defensive: limits refuse bad construction.
# ---------------------------------------------------------------------------


def test_risk_limits_rejects_negative_cap() -> None:
    from trading_system.rms import RiskLimits

    with pytest.raises(ValueError):
        RiskLimits(
            max_notional_per_order=Decimal("-1"),
            max_position_per_symbol=Decimal("1"),
            max_gross_exposure=Decimal("1"),
            max_pending_orders=1,
            max_volume_per_hour=Decimal("1"),
            daily_drawdown_halt_pct=Decimal("-0.01"),
            wash_trading_window_sec=1,
            ai_confidence_floor=Decimal("0.5"),
        )
