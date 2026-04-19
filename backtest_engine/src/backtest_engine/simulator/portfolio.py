"""Portfolio / account state for the backtest simulator.

All cash is `shared_lib.math_utils.Money` (Decimal-backed, refuses
float). Positions track quantity + volume-weighted average price.
Realized PnL is computed only when a position is (partially) closed.
Shorts are permitted: quantity can be negative; opening a short credits
cash at fill price.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from shared_lib.contracts import Fill
from shared_lib.math_utils import Money

__all__ = [
    "MarkToMarket",
    "Portfolio",
    "PositionState",
]


@dataclass
class PositionState:
    symbol: str
    quantity: Decimal = Decimal("0")
    avg_price: Decimal = Decimal("0")


@dataclass(frozen=True, slots=True)
class MarkToMarket:
    cash: Money
    unrealized_pnl: Money
    realized_pnl: Money
    equity: Money


class Portfolio:
    def __init__(self, *, starting_cash: Money) -> None:
        self._cash = starting_cash
        self._realized = Money("0", starting_cash.currency)
        self._positions: dict[str, PositionState] = {}
        self._currency = starting_cash.currency

    @property
    def cash(self) -> Money:
        return self._cash

    @property
    def realized_pnl(self) -> Money:
        return self._realized

    def position(self, symbol: str) -> PositionState:
        return self._positions.setdefault(symbol, PositionState(symbol=symbol))

    def _spend(self, amount: Money) -> None:
        self._cash = self._cash - amount

    def _earn(self, amount: Money) -> None:
        self._cash = self._cash + amount

    def apply_fill(self, fill: Fill) -> None:
        if fill.currency != self._currency:
            raise ValueError(
                f"fill currency {fill.currency!r} != portfolio currency {self._currency!r}"
            )
        signed_qty = fill.quantity if fill.side == "buy" else -fill.quantity
        price = fill.price
        pos = self.position(fill.symbol)

        if pos.quantity == 0 or (pos.quantity * signed_qty) > 0:
            # Opening or adding in the same direction.
            new_qty = pos.quantity + signed_qty
            if new_qty == 0:
                pos.avg_price = Decimal("0")
            else:
                pos.avg_price = (
                    pos.avg_price * abs(pos.quantity)
                    + price * abs(signed_qty)
                ) / abs(new_qty)
            pos.quantity = new_qty
            notional = Money(price * abs(signed_qty), self._currency)
            if fill.side == "buy":
                self._spend(notional)
            else:
                self._earn(notional)
        else:
            # Reducing / flipping.
            close_qty = min(abs(pos.quantity), abs(signed_qty))
            direction = Decimal("1") if pos.quantity > 0 else Decimal("-1")
            realized_per_unit = (price - pos.avg_price) * direction
            self._realized = self._realized + Money(
                realized_per_unit * close_qty, self._currency
            )

            # Cash impact of the full fill (treat as simple BUY/SELL cash move).
            notional = Money(price * abs(signed_qty), self._currency)
            if fill.side == "buy":
                self._spend(notional)
            else:
                self._earn(notional)

            # Update quantity; average price survives until fully closed,
            # then resets to the new fill's price if the position flipped.
            remaining = pos.quantity + signed_qty
            if remaining == 0:
                pos.quantity = Decimal("0")
                pos.avg_price = Decimal("0")
            elif (pos.quantity > 0 and remaining < 0) or (pos.quantity < 0 and remaining > 0):
                # Flipped direction: the new avg price is the fill price for
                # the flipped residual.
                pos.quantity = remaining
                pos.avg_price = price
            else:
                pos.quantity = remaining

        self._spend(Money(fill.fee, self._currency))

    def mark_to_market(self, price_map: dict[str, Decimal]) -> MarkToMarket:
        unrealized = Decimal("0")
        position_market_value = Decimal("0")
        for symbol, pos in self._positions.items():
            if pos.quantity == 0:
                continue
            if symbol not in price_map:
                raise KeyError(f"no mark-to-market price for {symbol!r}")
            price = price_map[symbol]
            unrealized += pos.quantity * (price - pos.avg_price)
            position_market_value += pos.quantity * price
        unrealized_money = Money(unrealized, self._currency)
        # Equity follows the canonical definition: cash + current market
        # value of all open positions. This decouples equity from the
        # cost-basis arithmetic used to compute unrealized PnL.
        equity = self._cash + Money(position_market_value, self._currency)
        return MarkToMarket(
            cash=self._cash,
            unrealized_pnl=unrealized_money,
            realized_pnl=self._realized,
            equity=equity,
        )
