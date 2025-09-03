from __future__ import annotations

import pandas as pd

from src.core.data_manager import UnifiedDataManager


def _df(dates, val):
    idx = pd.to_datetime(dates)
    return pd.DataFrame({"open": val, "high": val, "low": val, "close": val}, index=idx)


def test_split_cache_merge(monkeypatch):
    dm = UnifiedDataManager()

    # Legacy fast-path must return None to exercise split layer merge
    calls = {"legacy": 0}

    def fake_get_data(
        symbol, start_date, end_date, interval, source=None, data_type=None
    ):
        calls["legacy"] += 1
        if data_type == "full":
            return _df(["2023-01-01", "2023-01-10"], 1)
        if data_type == "recent":
            return _df(["2023-01-08", "2023-01-15"], 2)
        return None

    monkeypatch.setattr(dm.cache_manager, "get_data", fake_get_data)

    df = dm.get_data("TLT", "2023-01-01", "2023-01-20", "1d", use_cache=True)
    assert df is not None
    assert not df.empty
    # Last day should be 2023-01-15 given our recent overlay
    assert df.index[-1].date().isoformat() == "2023-01-15"
    # Overlap region should reflect recent overlay value (2) where both provide data
    assert df.loc[pd.Timestamp("2023-01-08"), "close"] == 2
