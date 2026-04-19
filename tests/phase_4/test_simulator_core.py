"""Phase 4 Task 4 - Simulator core (end-to-end bars -> fills -> equity).

Composes:
- EventLoop (drives bars through strategy)
- Portfolio (account state)
- Market mechanics (produces realistic fills)

The simulator emits fills that the portfolio applies, and yields a
`BacktestRun` summary with final equity, fill history, and signal
history.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal


def _bar(ts: datetime, symbol: str = "AAPL", close: str = "100"):
    from shared_lib.contracts import Bar

    return Bar(
        symbol=symbol,
        interval="1d",
        timestamp=ts,
        open=Decimal(close),
        high=Decimal(close),
        low=Decimal(close),
        close=Decimal(close),
        volume=Decimal("100000"),
    )


# ---------------------------------------------------------------------------
# Buy-and-hold strategy runs end-to-end.
# ---------------------------------------------------------------------------


def test_buy_and_hold_produces_expected_equity() -> None:
    from backtest_engine.market_mechanics import FixedBpsSlippage, PercentageFee
    from backtest_engine.simulator import Simulator
    from shared_lib.math_utils import Money

    class _BuyAndHold:
        def on_bar(self, bar, context):
            if context.run_id and len(list(context.history())) == 1:
                return [context.make_signal(direction="long", strength=Decimal("1"))]
            return []

    start = datetime(2026, 4, 1, tzinfo=UTC)
    bars = [_bar(start + timedelta(days=i), close=str(100 + i)) for i in range(5)]

    sim = Simulator(
        strategy=_BuyAndHold(),
        starting_cash=Money("10000", "USD"),
        slippage=FixedBpsSlippage(bps=Decimal("0")),
        fee=PercentageFee(rate=Decimal("0")),
        trade_size=Decimal("10"),  # 10 shares per signal
    )
    result = sim.run(run_id="bh-1", bars=bars)
    # Buy 10 on the first bar at 100 (next-bar execution would be t=1 price=101;
    # Phase 4 uses t=0 fill at bar close to keep this minimal and test-first).
    # Sim semantics (Phase 4): signal at t_i is filled at bar(t_i).close.
    # Cash = 10000 - 10*100 = 9000. Market value at t=4: 10 * 104 = 1040.
    # Equity = 9000 + 1040 = 10040.
    assert result.final_equity == Money("10040", "USD")
    assert result.fill_count == 1
    assert result.signal_count == 1
