from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    name: str

    @abstractmethod
    def param_grid(self) -> dict[str, list]:
        """Return the parameter grid to search. Keys map to lists of candidate values."""

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
        """Return entries and exits boolean Series aligned to df index."""

    def to_tradingview_pine(self, params: dict) -> str | None:
        """Optional: return Pine v5 code snippet implementing the strategy with given params."""
        return None
