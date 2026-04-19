"""RL research lane.

Inspired by FinRL patterns: environments expose `metadata()` / `reset()` /
`step()`, training is explicitly split into train/test/trade windows, and
continuous action spaces are bounded (Layer 1 of the ADR-0004
kill-switch architecture).

RL is a *research lane*, not the default strategy framework. Live
trading of RL policies goes through the same OMS/EMS/RMS gates as any
other strategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from shared_lib.contracts import RLEnvironmentMetadata

__all__ = [
    "BoundedActionSpace",
    "RLEnvironment",
    "TrainTestTradeSplit",
    "bounded_action_space",
]


class RLEnvironment(ABC):
    """Minimum RL environment surface.

    - `metadata()` declares shapes + bounds as a
      `shared_lib.contracts.RLEnvironmentMetadata`.
    - `reset(seed)` must be deterministic when a seed is given.
    - `step(action)` returns `(observation, reward, done)`.
    """

    @abstractmethod
    def metadata(self) -> RLEnvironmentMetadata:
        ...

    @abstractmethod
    def reset(self, *, seed: int | None = None) -> Any:
        ...

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool]:
        ...


@dataclass(frozen=True, slots=True)
class BoundedActionSpace:
    low: Decimal
    high: Decimal

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(
                f"BoundedActionSpace: low={self.low} > high={self.high}"
            )

    def validate(self, action: Decimal) -> None:
        if action < self.low or action > self.high:
            raise ValueError(
                f"action {action} outside [{self.low}, {self.high}]"
            )


def bounded_action_space(*, low: Decimal, high: Decimal) -> BoundedActionSpace:
    return BoundedActionSpace(low=low, high=high)


@dataclass(frozen=True, slots=True)
class TrainTestTradeSplit:
    train: tuple[datetime, datetime]
    test: tuple[datetime, datetime]
    trade: tuple[datetime, datetime]

    def __post_init__(self) -> None:
        for name, span in (("train", self.train), ("test", self.test), ("trade", self.trade)):
            s, e = span
            if s.tzinfo is None or e.tzinfo is None:
                raise ValueError(f"{name} window must be tz-aware")
            if e <= s:
                raise ValueError(f"{name} window: end must be > start")

        # Enforce chronological ordering: train < test < trade, and no
        # overlaps (end of one equals start of next is OK).
        if self.test[0] < self.train[1]:
            raise ValueError("train/test overlap (or non-chronological)")
        if self.trade[0] < self.test[1]:
            raise ValueError("test/trade overlap (or non-chronological)")
        if self.train[0] >= self.test[0] or self.test[0] >= self.trade[0]:
            raise ValueError("windows must be chronological: train < test < trade")
