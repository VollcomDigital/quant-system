"""Simulator core: strategy adapter + event-loop scheduler.

Design rule: the simulator is custom-code-first. A `Strategy` is a
duck-typed object with `on_bar(bar, context)`. The `EventLoop` drives
bars in strict chronological order and gives the strategy a
`StrategyContext` that only exposes data available at or before the
current bar timestamp (look-ahead prevention).
"""

from __future__ import annotations

import secrets
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol, runtime_checkable

from shared_lib.contracts import Bar, TradeSignal

__all__ = [
    "EventLoop",
    "RunResult",
    "Simulator",
    "Strategy",
    "StrategyContext",
]


def __getattr__(name: str):
    # Lazy-import Simulator to avoid a circular import at module load.
    if name == "Simulator":
        from backtest_engine.simulator.core import Simulator as _S

        return _S
    if name == "BacktestRun":
        from backtest_engine.simulator.core import BacktestRun as _B

        return _B
    raise AttributeError(name)


@runtime_checkable
class Strategy(Protocol):
    def on_bar(
        self, bar: Bar, context: StrategyContext
    ) -> Iterable[TradeSignal]:
        ...


@dataclass
class StrategyContext:
    run_id: str
    strategy_id: str
    _history: list[Bar]
    _current_bar: Bar | None = None
    _signal_seq: int = 0

    def history(self) -> Iterator[Bar]:
        yield from self._history

    def make_signal(self, *, direction: str, strength: Decimal) -> TradeSignal:
        if self._current_bar is None:
            raise RuntimeError(
                "StrategyContext.make_signal called outside of on_bar"
            )
        self._signal_seq += 1
        return TradeSignal(
            signal_id=f"sig-{self.run_id}-{self._signal_seq:06d}",
            strategy_id=self.strategy_id,
            symbol=self._current_bar.symbol,
            direction=direction,  # type: ignore[arg-type]
            strength=strength,
            generated_at=self._current_bar.timestamp,
        )


@dataclass(frozen=True, slots=True)
class RunResult:
    run_id: str
    strategy_id: str
    bar_count: int
    signal_count: int
    signals: tuple[TradeSignal, ...]


class EventLoop:
    """Drives a strategy over an ordered bar stream.

    Rejects out-of-order bars; attaches the `run_id` / `strategy_id` to
    the context so downstream audit + telemetry can thread them.
    """

    def __init__(
        self,
        *,
        strategy: Strategy,
        run_id: str,
        strategy_id: str | None = None,
    ) -> None:
        if not run_id:
            raise ValueError("run_id must be non-empty")
        self._strategy = strategy
        self._run_id = run_id
        self._strategy_id = strategy_id or f"strat-{secrets.token_hex(4)}"

    def run(self, bars: Iterable[Bar]) -> RunResult:
        history: list[Bar] = []
        ctx = StrategyContext(
            run_id=self._run_id,
            strategy_id=self._strategy_id,
            _history=history,
        )
        last_ts = None
        signals: list[TradeSignal] = []
        for bar in bars:
            if last_ts is not None and bar.timestamp < last_ts:
                raise ValueError(
                    f"bars out of order: {bar.timestamp} after {last_ts}"
                )
            last_ts = bar.timestamp
            history.append(bar)
            ctx._current_bar = bar
            emitted = list(self._strategy.on_bar(bar, ctx))
            signals.extend(emitted)

        return RunResult(
            run_id=self._run_id,
            strategy_id=self._strategy_id,
            bar_count=len(history),
            signal_count=len(signals),
            signals=tuple(signals),
        )
