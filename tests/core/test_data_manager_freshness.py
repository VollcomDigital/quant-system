from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd

from src.core.data_manager import UnifiedDataManager


def test_freshness_warning_for_daily(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    dm = UnifiedDataManager()

    # Fake source returning a stale last bar (two business days ago)
    class FakeSource:
        def __init__(self):
            self.config = SimpleNamespace(
                name="yahoo_finance", priority=1, asset_types=["stocks"]
            )

        def fetch_data(self, symbol, start_date, end_date, interval, **kwargs):
            idx = pd.date_range("2023-01-01", periods=10, freq="D")
            return pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": 1}, index=idx)

    # Route only to our fake source
    monkeypatch.setattr(dm, "_get_sources_for_asset_type", lambda at: [FakeSource()])

    # Force fetch path (skip cache) so freshness check executes
    df = dm.get_data(
        "AAPL", "2000-01-01", "2100-01-01", "1d", use_cache=False, asset_type="stocks"
    )
    assert df is not None
    assert not df.empty
    # Assert warning logged
    assert any("seems stale" in rec.message for rec in caplog.records)
