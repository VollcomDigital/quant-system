"""Look-ahead / leakage guards.

These run in the backtest engine (and can also be reused by the
factor-promotion pipeline in Phase 3's `promote_factor`). Every guard
is a small function that either returns normally or raises
`ValueError`. No side effects.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime
from typing import TypeVar

from shared_lib.contracts import FactorRecord, Fill, TradeSignal

T = TypeVar("T")

__all__ = [
    "ensure_factor_precedes_signal",
    "ensure_signal_precedes_fill",
    "stable_replay_order",
]


T = TypeVar("T")


def ensure_signal_precedes_fill(*, signal: TradeSignal, fill: Fill) -> None:
    if fill.filled_at < signal.generated_at:
        raise ValueError(
            f"leakage: signal must precede fill "
            f"(signal={signal.generated_at.isoformat()}, "
            f"fill={fill.filled_at.isoformat()})"
        )


def ensure_factor_precedes_signal(
    *, factor: FactorRecord, signal: TradeSignal
) -> None:
    if factor.as_of > signal.generated_at:
        raise ValueError(
            f"leakage: factor as_of must precede signal generated_at "
            f"(factor={factor.as_of.isoformat()}, "
            f"signal={signal.generated_at.isoformat()})"
        )


def stable_replay_order(  # noqa: UP047 - TypeVar keeps Py 3.11 compat
    events: Iterable[T],
    *,
    key: Callable[[T], datetime],
) -> list[T]:
    """Return `events` sorted by `key` timestamp, preserving insertion
    order on ties. Refuses naive timestamps."""
    indexed = list(enumerate(events))
    for idx, event in indexed:
        ts = key(event)
        if ts.tzinfo is None:
            raise ValueError(
                f"event #{idx}: timestamp must be tz-aware (naive rejected)"
            )
    indexed.sort(key=lambda pair: (key(pair[1]), pair[0]))
    return [event for _, event in indexed]
