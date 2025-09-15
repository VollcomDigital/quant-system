from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class DataSource(ABC):
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def fetch(self, symbol: str, timeframe: str, only_cached: bool = False) -> pd.DataFrame:
        """Return OHLCV DataFrame indexed by UTC datetime, columns: [Open, High, Low, Close, Volume].

        If `only_cached` is True, must return cached data or raise an error if missing.
        """
