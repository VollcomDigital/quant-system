"""Phase 4 Task 2 - Portfolio / account state.

Contract:
- All cash is `shared_lib.math_utils.Money` (rejects floats).
- Positions track quantity + average price per symbol.
- Applying a `Fill` updates cash, position, and realized PnL
  deterministically.
- `mark_to_market(price_map)` computes unrealized PnL without mutating.
- Short positions allowed; quantity can be negative.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest


def _fill(**kw):
    from shared_lib.contracts import Fill

    defaults = dict(
        fill_id="f-1",
        order_id="o-1",
        symbol="AAPL",
        side="buy",
        quantity=Decimal("10"),
        price=Decimal("100"),
        fee=Decimal("0.5"),
        currency="USD",
        filled_at=datetime(2026, 4, 1, tzinfo=UTC),
    )
    defaults.update(kw)
    return Fill(**defaults)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_portfolio_starts_with_given_cash() -> None:
    from backtest_engine.simulator.portfolio import Portfolio
    from shared_lib.math_utils import Money

    p = Portfolio(starting_cash=Money("10000", "USD"))
    assert p.cash == Money("10000", "USD")
    assert p.realized_pnl == Money("0", "USD")


def test_portfolio_rejects_mismatched_fill_currency() -> None:
    from backtest_engine.simulator.portfolio import Portfolio
    from shared_lib.math_utils import Money

    p = Portfolio(starting_cash=Money("10000", "USD"))
    with pytest.raises(ValueError, match="currency"):
        p.apply_fill(_fill(currency="EUR"))


# ---------------------------------------------------------------------------
# Buy / sell flow
# ---------------------------------------------------------------------------


def test_buy_decreases_cash_and_opens_long() -> None:
    from backtest_engine.simulator.portfolio import Portfolio
    from shared_lib.math_utils import Money

    p = Portfolio(starting_cash=Money("10000", "USD"))
    p.apply_fill(_fill(side="buy", quantity=Decimal("10"), price=Decimal("100"), fee=Decimal("0.5")))
    # 10 * 100 + 0.5 fee = 1000.5
    assert p.cash == Money("8999.5", "USD")
    assert p.position("AAPL").quantity == Decimal("10")
    assert p.position("AAPL").avg_price == Decimal("100")


def test_sell_after_buy_realizes_pnl() -> None:
    from backtest_engine.simulator.portfolio import Portfolio
    from shared_lib.math_utils import Money

    p = Portfolio(starting_cash=Money("10000", "USD"))
    p.apply_fill(_fill(side="buy", quantity=Decimal("10"), price=Decimal("100"), fee=Decimal("0")))
    # Full close at 110: realized = 10 * (110 - 100) = 100
    p.apply_fill(
        _fill(
            fill_id="f-2",
            order_id="o-2",
            side="sell",
            quantity=Decimal("10"),
            price=Decimal("110"),
            fee=Decimal("0"),
        )
    )
    assert p.position("AAPL").quantity == Decimal("0")
    assert p.realized_pnl == Money("100", "USD")


def test_short_sell_opens_short_position() -> None:
    from backtest_engine.simulator.portfolio import Portfolio
    from shared_lib.math_utils import Money

    p = Portfolio(starting_cash=Money("10000", "USD"))
    # Open short: sell 10 @ 100.
    p.apply_fill(_fill(side="sell", quantity=Decimal("10"), price=Decimal("100"), fee=Decimal("0")))
    assert p.position("AAPL").quantity == Decimal("-10")
    # Shorts credit cash (ignoring margin complexities for Phase 4).
    assert p.cash == Money("11000", "USD")


def test_partial_close_splits_realized_and_remaining() -> None:
    from backtest_engine.simulator.portfolio import Portfolio
    from shared_lib.math_utils import Money

    p = Portfolio(starting_cash=Money("10000", "USD"))
    p.apply_fill(_fill(side="buy", quantity=Decimal("10"), price=Decimal("100"), fee=Decimal("0")))
    # Sell 4 @ 105 -> realized = 4 * (105-100) = 20, remaining qty = 6, avg = 100
    p.apply_fill(
        _fill(
            fill_id="f-2",
            order_id="o-2",
            side="sell",
            quantity=Decimal("4"),
            price=Decimal("105"),
            fee=Decimal("0"),
        )
    )
    assert p.realized_pnl == Money("20", "USD")
    assert p.position("AAPL").quantity == Decimal("6")
    assert p.position("AAPL").avg_price == Decimal("100")


def test_adding_to_position_recomputes_weighted_avg_price() -> None:
    from backtest_engine.simulator.portfolio import Portfolio
    from shared_lib.math_utils import Money

    p = Portfolio(starting_cash=Money("10000", "USD"))
    p.apply_fill(_fill(side="buy", quantity=Decimal("10"), price=Decimal("100"), fee=Decimal("0")))
    p.apply_fill(
        _fill(
            fill_id="f-2",
            order_id="o-2",
            side="buy",
            quantity=Decimal("10"),
            price=Decimal("120"),
            fee=Decimal("0"),
        )
    )
    # (10*100 + 10*120) / 20 = 110
    assert p.position("AAPL").avg_price == Decimal("110")
    assert p.position("AAPL").quantity == Decimal("20")


# ---------------------------------------------------------------------------
# Mark to market
# ---------------------------------------------------------------------------


def test_mark_to_market_computes_unrealized() -> None:
    from backtest_engine.simulator.portfolio import Portfolio
    from shared_lib.math_utils import Money

    p = Portfolio(starting_cash=Money("10000", "USD"))
    p.apply_fill(_fill(side="buy", quantity=Decimal("10"), price=Decimal("100"), fee=Decimal("0")))
    mtm = p.mark_to_market({"AAPL": Decimal("115")})
    # Unrealized = 10 * (115 - 100) = 150
    assert mtm.unrealized_pnl == Money("150", "USD")
    assert mtm.equity == Money("10150", "USD")


def test_mark_to_market_rejects_missing_price() -> None:
    from backtest_engine.simulator.portfolio import Portfolio
    from shared_lib.math_utils import Money

    p = Portfolio(starting_cash=Money("10000", "USD"))
    p.apply_fill(_fill(side="buy", quantity=Decimal("1")))
    with pytest.raises(KeyError, match="AAPL"):
        p.mark_to_market({})


def test_mark_to_market_on_empty_portfolio() -> None:
    from backtest_engine.simulator.portfolio import Portfolio
    from shared_lib.math_utils import Money

    p = Portfolio(starting_cash=Money("10000", "USD"))
    mtm = p.mark_to_market({})
    assert mtm.unrealized_pnl == Money("0", "USD")
    assert mtm.equity == Money("10000", "USD")


# ---------------------------------------------------------------------------
# Defensive: fees are always non-negative and subtracted from cash.
# ---------------------------------------------------------------------------


def test_fees_debit_cash() -> None:
    from backtest_engine.simulator.portfolio import Portfolio
    from shared_lib.math_utils import Money

    p = Portfolio(starting_cash=Money("10000", "USD"))
    p.apply_fill(_fill(side="buy", quantity=Decimal("1"), price=Decimal("100"), fee=Decimal("2")))
    assert p.cash == Money("9898", "USD")
