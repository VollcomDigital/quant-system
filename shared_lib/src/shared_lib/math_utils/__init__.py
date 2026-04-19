"""Vectorized math + Decimal-safe money helpers.

Two contracts in one module:

- Array math (`simple_returns`, `log_returns`, `cumulative_returns`,
  `volatility`, `sharpe_ratio`, `max_drawdown`, `sortino_ratio`, `cagr`)
  backed by NumPy. No pandas dependency. Zero-division and NaN inputs are
  handled deterministically.
- `Money` - a Decimal-backed value type for ledger logic. Rejects `float`
  at every surface so rounding bugs cannot leak into balances, fills, or
  fees.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "Money",
    "cagr",
    "cumulative_returns",
    "log_returns",
    "max_drawdown",
    "sharpe_ratio",
    "simple_returns",
    "sortino_ratio",
    "volatility",
]


ArrayF = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


def _as_float_array(values: ArrayLike) -> ArrayF:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("1-D array required")
    return arr


def simple_returns(prices: ArrayLike) -> ArrayF:
    """Return simple returns `(p_t / p_{t-1}) - 1`. Zero priors -> NaN."""
    arr = _as_float_array(prices)
    if arr.size < 2:
        raise ValueError("simple_returns requires at least 2 observations")
    prior = arr[:-1]
    out = np.full(arr.size - 1, np.nan, dtype=np.float64)
    nz = prior != 0.0
    out[nz] = arr[1:][nz] / prior[nz] - 1.0
    return out


def log_returns(prices: ArrayLike) -> ArrayF:
    """Natural-log returns."""
    arr = _as_float_array(prices)
    if arr.size < 2:
        raise ValueError("log_returns requires at least 2 observations")
    prior = arr[:-1]
    out = np.full(arr.size - 1, np.nan, dtype=np.float64)
    nz = (prior > 0.0) & (arr[1:] > 0.0)
    out[nz] = np.log(arr[1:][nz] / prior[nz])
    return out


def cumulative_returns(returns: ArrayLike) -> ArrayF:
    """Compound returns: `cumprod(1 + r) - 1`."""
    arr = _as_float_array(returns)
    if arr.size == 0:
        raise ValueError("cumulative_returns requires at least 1 observation")
    return np.cumprod(1.0 + arr) - 1.0


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------


def volatility(returns: ArrayLike, *, periods_per_year: int) -> float:
    arr = _as_float_array(returns)
    if arr.size < 2:
        return 0.0
    return float(np.std(arr, ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: ArrayLike,
    *,
    risk_free: float,
    periods_per_year: int,
) -> float:
    arr = _as_float_array(returns)
    if arr.size < 2:
        return 0.0
    excess_period = arr - (risk_free / periods_per_year)
    vol = float(np.std(excess_period, ddof=1))
    if vol == 0.0:
        return 0.0
    return float(np.mean(excess_period) / vol * np.sqrt(periods_per_year))


def max_drawdown(returns: ArrayLike) -> float:
    """Return the most-negative peak-to-trough equity drawdown as a fraction."""
    arr = _as_float_array(returns)
    if arr.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + arr)
    running_peak = np.maximum.accumulate(equity)
    drawdown = equity / running_peak - 1.0
    return float(drawdown.min())


def sortino_ratio(
    returns: ArrayLike,
    *,
    risk_free: float,
    periods_per_year: int,
) -> float:
    arr = _as_float_array(returns)
    if arr.size < 2:
        return 0.0
    excess = arr - (risk_free / periods_per_year)
    downside = excess[excess < 0.0]
    if downside.size == 0:
        return 0.0
    downside_std = float(np.sqrt(np.mean(downside**2)))
    if downside_std == 0.0:
        return 0.0
    return float(np.mean(excess) / downside_std * np.sqrt(periods_per_year))


def cagr(returns: ArrayLike, *, periods_per_year: int) -> float:
    arr = _as_float_array(returns)
    if arr.size == 0:
        raise ValueError("cagr requires at least 1 observation")
    total = float(np.prod(1.0 + arr))
    years = arr.size / periods_per_year
    if total <= 0.0 or years <= 0.0:
        return -1.0
    return total ** (1.0 / years) - 1.0


# ---------------------------------------------------------------------------
# Money
# ---------------------------------------------------------------------------


_DecimalInput = Decimal | int | str


@dataclass(frozen=True, slots=True)
class Money:
    """Decimal-backed money value object. Floats are rejected at every surface."""

    amount: Decimal
    currency: str

    def __init__(self, amount: _DecimalInput, currency: str) -> None:
        if isinstance(amount, bool):  # bool is an int subclass; refuse it
            raise TypeError("Money cannot be constructed from bool")
        if isinstance(amount, float):
            raise TypeError(
                "Money cannot be constructed from float; use Decimal or str"
            )
        if not currency or not isinstance(currency, str):
            raise ValueError("Money requires a non-empty currency code")
        object.__setattr__(self, "amount", Decimal(amount))
        object.__setattr__(self, "currency", currency.upper())

    # Arithmetic ------------------------------------------------------------

    def _check_currency(self, other: Money) -> None:
        if self.currency != other.currency:
            raise ValueError(
                f"currency mismatch: {self.currency} vs {other.currency}"
            )

    def __add__(self, other: Money) -> Money:
        if not isinstance(other, Money):
            return NotImplemented
        self._check_currency(other)
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: Money) -> Money:
        if not isinstance(other, Money):
            return NotImplemented
        self._check_currency(other)
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, factor: Decimal) -> Money:
        if isinstance(factor, bool):
            raise TypeError("Money * bool is forbidden")
        if isinstance(factor, float):
            raise TypeError("Money * float is forbidden; use Decimal")
        if not isinstance(factor, (Decimal, int)):
            return NotImplemented
        return Money(self.amount * Decimal(factor), self.currency)

    def __truediv__(self, other: Money) -> Decimal:
        if not isinstance(other, Money):
            return NotImplemented
        self._check_currency(other)
        if other.amount == 0:
            raise ZeroDivisionError("division by zero Money amount")
        return self.amount / other.amount

    # Quantization ----------------------------------------------------------

    def quantize(self, exp: Decimal) -> Money:
        """Return a new Money with the amount quantized to `exp` (banker's rounding)."""
        return Money(self.amount.quantize(exp, rounding=ROUND_HALF_EVEN), self.currency)
