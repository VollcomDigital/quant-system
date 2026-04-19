"""Phase 4 Task 6 - Analytics / tear-sheet metrics.

Wraps `shared_lib.math_utils` into a single `tear_sheet(equity_curve)`
summary so every backtest emits the same structured result payload.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

# ---------------------------------------------------------------------------
# Tear sheet
# ---------------------------------------------------------------------------


def test_tear_sheet_from_equity_curve_matches_math_utils() -> None:
    from backtest_engine.analytics import tear_sheet

    equity = [
        Decimal("10000"),
        Decimal("10100"),
        Decimal("10050"),
        Decimal("10200"),
    ]
    ts = tear_sheet(equity_curve=equity, periods_per_year=252)
    assert ts.total_return > Decimal("0")
    assert ts.max_drawdown <= Decimal("0")
    assert ts.num_periods == 3


def test_tear_sheet_rejects_single_point_curve() -> None:
    from backtest_engine.analytics import tear_sheet

    with pytest.raises(ValueError):
        tear_sheet(equity_curve=[Decimal("10000")], periods_per_year=252)


def test_tear_sheet_total_return_is_correct() -> None:
    from backtest_engine.analytics import tear_sheet

    equity = [Decimal("100"), Decimal("110")]
    ts = tear_sheet(equity_curve=equity, periods_per_year=252)
    assert ts.total_return == Decimal("0.10")


def test_tear_sheet_max_drawdown_is_worst_peak_to_trough() -> None:
    from backtest_engine.analytics import tear_sheet

    # 100 -> 120 -> 90 -> 110. Max DD from 120 to 90 = -0.25.
    equity = [Decimal("100"), Decimal("120"), Decimal("90"), Decimal("110")]
    ts = tear_sheet(equity_curve=equity, periods_per_year=252)
    assert abs(ts.max_drawdown - Decimal("-0.25")) < Decimal("0.0001")


# ---------------------------------------------------------------------------
# Equity curve from fills
# ---------------------------------------------------------------------------


def test_equity_curve_from_fills_is_monotonic_in_time() -> None:
    from datetime import UTC, datetime, timedelta

    from backtest_engine.analytics import equity_curve_from_fills
    from shared_lib.contracts import Fill
    from shared_lib.math_utils import Money

    start = datetime(2026, 4, 1, tzinfo=UTC)
    fills = [
        Fill(
            fill_id=f"f{i}",
            order_id=f"o{i}",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("1"),
            price=Decimal(f"{100 + i}"),
            fee=Decimal("0"),
            currency="USD",
            filled_at=start + timedelta(days=i),
        )
        for i in range(3)
    ]
    curve = equity_curve_from_fills(
        starting_cash=Money("10000", "USD"),
        fills=fills,
        price_by_timestamp=lambda ts: Decimal("100") + Decimal((ts - start).days),
    )
    timestamps = [pt[0] for pt in curve]
    assert timestamps == sorted(timestamps)
