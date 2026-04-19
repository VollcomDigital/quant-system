"""Phase 6 Exit Criteria aggregate gate.

- OMS and EMS are separated by responsibility.
- RMS controls enforced between signal generation and gateway execution.
- Automated panic-button workflow defined for TradFi and DeFi paths.
- Mid-frequency execution contracts work without HFT-only assumptions.
"""

from __future__ import annotations


def test_exit_oms_and_ems_are_separate_packages() -> None:
    import trading_system.ems as ems
    import trading_system.oms as oms

    assert oms.__name__ != ems.__name__
    assert hasattr(oms, "OMS")
    assert hasattr(ems, "EMS")


def test_exit_rms_refuses_order_when_trading_halted() -> None:
    from datetime import UTC, datetime
    from decimal import Decimal

    from shared_lib.contracts import Order
    from trading_system.rms import RMS, RiskContext, RiskLimits

    order = Order(
        order_id="o",
        idempotency_key="i",
        symbol="X",
        side="buy",
        quantity=Decimal("1"),
        limit_price=Decimal("1"),
        time_in_force="day",
        placed_at=datetime(2026, 4, 19, tzinfo=UTC),
    )
    limits = RiskLimits(
        max_notional_per_order=Decimal("1000"),
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
        trading_halted=True,
        now=datetime(2026, 4, 19, tzinfo=UTC),
    )
    assert RMS(limits=limits).check(order, ctx).passed is False


def test_exit_panic_playbook_is_available() -> None:
    from trading_system.kill_switch import KillSwitch, PanicPlaybook  # noqa: F401


def test_exit_contracts_have_no_hft_assumptions() -> None:
    # The Phase 6 mid-frequency execution contracts must not import any
    # trading_system/native/* symbols (those are Phase 8).
    import trading_system.ems
    import trading_system.oms
    import trading_system.rms

    for module in (trading_system.oms, trading_system.ems, trading_system.rms):
        for attr in dir(module):
            val = getattr(module, attr, None)
            name = getattr(val, "__module__", "") or ""
            assert "trading_system.native" not in name
