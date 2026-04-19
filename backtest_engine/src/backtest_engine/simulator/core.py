"""Simulator core - composes EventLoop + Portfolio + Market mechanics.

Phase 4 simulator semantics (intentionally minimal and explicit):

- A strategy emits `TradeSignal`s on each bar.
- Each signal is filled at the current bar's close (plus slippage and
  impact overrides; minus spread for sells). Fees are applied per fill.
- Fills flow into the Portfolio. Final equity is computed against the
  last bar's close price per held symbol.
- Time-in-force is `day`; Phase 6 extends this via OMS/EMS semantics.

Phase 4 does NOT try to be the live OMS; its job is to deliver
reproducible fills and equity curves for research.
"""

from __future__ import annotations

import secrets
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal

from shared_lib.contracts import Bar, Fill, TradeSignal
from shared_lib.math_utils import Money

from backtest_engine.market_mechanics import (
    FixedBpsSlippage,
    PercentageFee,
)
from backtest_engine.simulator import EventLoop, Strategy
from backtest_engine.simulator.portfolio import Portfolio

__all__ = [
    "BacktestRun",
    "Simulator",
]


@dataclass(frozen=True, slots=True)
class BacktestRun:
    run_id: str
    final_equity: Money
    realized_pnl: Money
    unrealized_pnl: Money
    fill_count: int
    signal_count: int
    fills: tuple[Fill, ...]


class Simulator:
    """Compose everything."""

    def __init__(
        self,
        *,
        strategy: Strategy,
        starting_cash: Money,
        slippage: FixedBpsSlippage,
        fee: PercentageFee,
        trade_size: Decimal,
    ) -> None:
        if trade_size <= 0:
            raise ValueError("trade_size must be > 0")
        self._strategy = strategy
        self._starting_cash = starting_cash
        self._slippage = slippage
        self._fee = fee
        self._trade_size = trade_size

    def _signal_to_fill(
        self, *, bar: Bar, signal: TradeSignal, fill_seq: int
    ) -> Fill:
        side = "buy" if signal.direction == "long" else "sell"
        price = self._slippage.apply(side=side, mid=bar.close)
        notional = price * self._trade_size
        fee = self._fee.compute(notional=notional)
        return Fill(
            fill_id=f"fill-{signal.signal_id}-{fill_seq:04d}",
            order_id=f"ord-{signal.signal_id}",
            symbol=signal.symbol,
            side=side,  # type: ignore[arg-type]
            quantity=self._trade_size,
            price=price,
            fee=fee,
            currency=self._starting_cash.currency,
            filled_at=bar.timestamp,
        )

    def run(self, *, run_id: str, bars: Iterable[Bar]) -> BacktestRun:
        bars_list = list(bars)
        loop = EventLoop(
            strategy=self._strategy,
            run_id=run_id,
            strategy_id=f"bt-{secrets.token_hex(4)}",
        )
        loop_result = loop.run(bars_list)

        portfolio = Portfolio(starting_cash=self._starting_cash)
        fills: list[Fill] = []
        price_by_signal_ts: dict = {b.timestamp: b for b in bars_list}
        fill_seq = 0
        for signal in loop_result.signals:
            if signal.direction == "flat":
                continue
            fill_seq += 1
            bar = price_by_signal_ts[signal.generated_at]
            fill = self._signal_to_fill(bar=bar, signal=signal, fill_seq=fill_seq)
            portfolio.apply_fill(fill)
            fills.append(fill)

        last_prices = {}
        for b in bars_list:
            last_prices[b.symbol] = b.close
        mtm = portfolio.mark_to_market(last_prices) if bars_list else None

        return BacktestRun(
            run_id=run_id,
            final_equity=mtm.equity if mtm else self._starting_cash,
            realized_pnl=portfolio.realized_pnl,
            unrealized_pnl=mtm.unrealized_pnl if mtm else Money("0", self._starting_cash.currency),
            fill_count=len(fills),
            signal_count=loop_result.signal_count,
            fills=tuple(fills),
        )
