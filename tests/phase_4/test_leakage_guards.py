"""Phase 4 Task 5 - Look-ahead / leakage guards.

Three guards:

1. `ensure_signal_precedes_fill(signal, fill)` - fill.filled_at must be
   >= signal.generated_at.
2. `ensure_factor_as_of_precedes_signal(factor, signal)` - any
   FactorRecord feeding the signal must have `as_of <= signal.generated_at`.
3. `ensure_replay_order(events)` - replay must be stable and monotonic
   by timestamp + insertion index.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest


def _signal(ts: datetime):
    from shared_lib.contracts import TradeSignal

    return TradeSignal(
        signal_id="s-1",
        strategy_id="strat",
        symbol="AAPL",
        direction="long",
        strength=Decimal("0.5"),
        generated_at=ts,
    )


def _fill(ts: datetime):
    from shared_lib.contracts import Fill

    return Fill(
        fill_id="f-1",
        order_id="o-1",
        symbol="AAPL",
        side="buy",
        quantity=Decimal("10"),
        price=Decimal("100"),
        fee=Decimal("0"),
        currency="USD",
        filled_at=ts,
    )


def _factor(ts: datetime):
    from shared_lib.contracts import FactorRecord

    return FactorRecord(
        factor_id="mom",
        as_of=ts,
        symbol="AAPL",
        value=Decimal("0.1"),
        version="v1",
    )


# ---------------------------------------------------------------------------
# Signal -> Fill
# ---------------------------------------------------------------------------


def test_signal_must_precede_fill() -> None:
    from backtest_engine.leakage import ensure_signal_precedes_fill

    t = datetime(2026, 4, 1, tzinfo=UTC)
    sig = _signal(t)
    fill = _fill(t - timedelta(seconds=1))
    with pytest.raises(ValueError, match="precede"):
        ensure_signal_precedes_fill(signal=sig, fill=fill)


def test_signal_equal_timestamp_is_allowed() -> None:
    from backtest_engine.leakage import ensure_signal_precedes_fill

    t = datetime(2026, 4, 1, tzinfo=UTC)
    ensure_signal_precedes_fill(signal=_signal(t), fill=_fill(t))


def test_signal_before_fill_is_allowed() -> None:
    from backtest_engine.leakage import ensure_signal_precedes_fill

    t = datetime(2026, 4, 1, tzinfo=UTC)
    ensure_signal_precedes_fill(signal=_signal(t), fill=_fill(t + timedelta(seconds=1)))


# ---------------------------------------------------------------------------
# Factor -> Signal
# ---------------------------------------------------------------------------


def test_factor_future_as_of_is_rejected() -> None:
    from backtest_engine.leakage import ensure_factor_precedes_signal

    t = datetime(2026, 4, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="precede"):
        ensure_factor_precedes_signal(factor=_factor(t + timedelta(hours=1)), signal=_signal(t))


def test_factor_as_of_equal_signal_is_allowed() -> None:
    from backtest_engine.leakage import ensure_factor_precedes_signal

    t = datetime(2026, 4, 1, tzinfo=UTC)
    ensure_factor_precedes_signal(factor=_factor(t), signal=_signal(t))


# ---------------------------------------------------------------------------
# Replay ordering
# ---------------------------------------------------------------------------


def test_replay_order_is_monotonic_and_stable() -> None:
    from backtest_engine.leakage import stable_replay_order

    t = datetime(2026, 4, 1, tzinfo=UTC)
    events = [
        ("evt1", t + timedelta(seconds=2)),
        ("evt2", t),
        ("evt3", t + timedelta(seconds=1)),
        ("evt4", t),  # same timestamp as evt2; must keep insertion order
    ]
    ordered = stable_replay_order(events, key=lambda e: e[1])
    names = [e[0] for e in ordered]
    assert names == ["evt2", "evt4", "evt3", "evt1"]


def test_replay_order_rejects_naive_timestamp() -> None:
    from backtest_engine.leakage import stable_replay_order

    events = [("a", datetime(2026, 4, 1))]
    with pytest.raises(ValueError, match="naive"):
        stable_replay_order(events, key=lambda e: e[1])
