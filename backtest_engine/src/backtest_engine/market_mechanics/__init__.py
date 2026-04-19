"""Market mechanics models.

Small, composable units: slippage / fee / spread / impact / latency.
All take `Decimal` for price-side math. `Decimal` is used instead of
`float` so backtest numbers are reproducible across Python versions
and compatible with the Decimal-safe Money ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

__all__ = [
    "FixedBpsSlippage",
    "FixedLatency",
    "HalfSpreadModel",
    "PerShareFee",
    "PercentageFee",
    "SquareRootImpact",
]


_BPS = Decimal("10000")


def _sqrt(value: Decimal) -> Decimal:
    if value < Decimal("0"):
        raise ValueError("sqrt of negative")
    if value == 0:
        return Decimal("0")
    x = value
    prev = Decimal("0")
    for _ in range(40):
        x = (x + value / x) / 2
        if abs(x - prev) < Decimal("1e-20"):
            break
        prev = x
    return x


@dataclass(frozen=True, slots=True)
class FixedBpsSlippage:
    bps: Decimal

    def __post_init__(self) -> None:
        if self.bps < 0:
            raise ValueError("bps must be >= 0")

    def apply(self, *, side: str, mid: Decimal) -> Decimal:
        adj = mid * self.bps / _BPS
        return mid + adj if side == "buy" else mid - adj


@dataclass(frozen=True, slots=True)
class HalfSpreadModel:
    spread_bps: Decimal

    def __post_init__(self) -> None:
        if self.spread_bps < 0:
            raise ValueError("spread_bps must be >= 0")

    def apply(self, *, side: str, mid: Decimal) -> Decimal:
        half = mid * self.spread_bps / _BPS / 2
        return mid + half if side == "buy" else mid - half


@dataclass(frozen=True, slots=True)
class PercentageFee:
    rate: Decimal
    minimum: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        if self.rate < 0:
            raise ValueError("rate must be >= 0")
        if self.minimum < 0:
            raise ValueError("minimum must be >= 0")

    def compute(self, *, notional: Decimal) -> Decimal:
        raw = notional * self.rate
        return max(raw, self.minimum)


@dataclass(frozen=True, slots=True)
class PerShareFee:
    per_share: Decimal

    def __post_init__(self) -> None:
        if self.per_share < 0:
            raise ValueError("per_share must be >= 0")

    def compute_from_quantity(self, *, quantity: Decimal) -> Decimal:
        return self.per_share * abs(quantity)


@dataclass(frozen=True, slots=True)
class SquareRootImpact:
    k: Decimal

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError("k must be >= 0")

    def impact_bps(self, *, quantity: Decimal, volume: Decimal) -> Decimal:
        if volume <= 0:
            raise ValueError("volume must be > 0")
        if quantity == 0:
            return Decimal("0")
        participation = abs(quantity) / volume
        return self.k * _sqrt(participation) * _BPS


@dataclass(frozen=True, slots=True)
class FixedLatency:
    delay: timedelta

    def __post_init__(self) -> None:
        if self.delay.total_seconds() < 0:
            raise ValueError("delay must be >= 0")

    def delay_fill(self, ts: datetime) -> datetime:
        return ts + self.delay
