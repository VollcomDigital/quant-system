from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.core.data_manager import UnifiedDataManager


def test_probe_and_set_order(monkeypatch):
    dm = UnifiedDataManager()

    # Create two fake sources with differing coverage
    class FakeSource:
        def __init__(self, name, rows, start):
            self.config = SimpleNamespace(name=name, priority=1, asset_types=["stocks"])
            self._rows = rows
            self._start = start

        def fetch_data(self, symbol, start_date, end_date, interval, **kwargs):
            if self._rows == 0:
                return None
            idx = pd.date_range(self._start, periods=self._rows, freq="D")
            return pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": 1}, index=idx)

    dm.sources = {
        "yahoo": FakeSource("yahoo", rows=10, start="2020-01-01"),
        "alt": FakeSource("alt", rows=20, start="2019-01-01"),
    }

    ordered = dm.probe_and_set_order(
        "stocks", ["AAPL", "MSFT"], interval="1d", sample_size=2
    )
    assert ordered[0] == "alt"  # more rows and earlier start
