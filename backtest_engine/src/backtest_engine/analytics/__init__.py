"""Analytics / tear-sheet metrics.

Single `tear_sheet(equity_curve, periods_per_year)` wraps the Phase 1
`shared_lib.math_utils` primitives so every backtest emits the same
structured result.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import numpy as np
from shared_lib.contracts import Fill
from shared_lib.math_utils import (
    Money,
    cagr,
    max_drawdown,
    sharpe_ratio,
    simple_returns,
    volatility,
)

__all__ = [
    "TearSheet",
    "equity_curve_from_fills",
    "tear_sheet",
]


@dataclass(frozen=True, slots=True)
class TearSheet:
    num_periods: int
    total_return: Decimal
    cagr: Decimal
    volatility: Decimal
    sharpe: Decimal
    max_drawdown: Decimal


def tear_sheet(
    *,
    equity_curve: Sequence[Decimal],
    periods_per_year: int,
) -> TearSheet:
    if len(equity_curve) < 2:
        raise ValueError("tear_sheet requires at least 2 equity points")

    prices = np.array([float(x) for x in equity_curve], dtype=np.float64)
    returns = simple_returns(prices)

    total = prices[-1] / prices[0] - 1.0
    return TearSheet(
        num_periods=len(returns),
        total_return=Decimal(str(total)).quantize(Decimal("0.00000001")),
        cagr=Decimal(str(cagr(returns, periods_per_year=periods_per_year))),
        volatility=Decimal(
            str(volatility(returns, periods_per_year=periods_per_year))
        ),
        sharpe=Decimal(
            str(
                sharpe_ratio(
                    returns, risk_free=0.0, periods_per_year=periods_per_year
                )
            )
        ),
        max_drawdown=Decimal(str(max_drawdown(returns))).quantize(
            Decimal("0.00000001")
        ),
    )


def equity_curve_from_fills(
    *,
    starting_cash: Money,
    fills: Sequence[Fill],
    price_by_timestamp: Callable[[datetime], Decimal],
) -> list[tuple[datetime, Decimal]]:
    """Produce a `(timestamp, equity)` curve by applying fills in order."""
    cash = starting_cash.amount
    positions: dict[str, Decimal] = {}
    sorted_fills = sorted(fills, key=lambda f: f.filled_at)
    curve: list[tuple[datetime, Decimal]] = []
    for f in sorted_fills:
        qty = f.quantity if f.side == "buy" else -f.quantity
        notional = f.price * abs(qty)
        if f.side == "buy":
            cash -= notional
        else:
            cash += notional
        cash -= f.fee
        positions[f.symbol] = positions.get(f.symbol, Decimal("0")) + qty

        market_value = Decimal("0")
        mark = price_by_timestamp(f.filled_at)
        for _symbol, held in positions.items():
            if held != 0:
                market_value += held * mark
        curve.append((f.filled_at, cash + market_value))
    return curve
