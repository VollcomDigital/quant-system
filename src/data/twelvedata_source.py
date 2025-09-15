from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from ..utils.http import create_retry_session
from .base import DataSource
from .cache import ParquetCache
from .ratelimiter import RateLimiter
from .symbol_mapper import map_symbol


class TwelveDataSource(DataSource):
    """TwelveData source (focused on FX intraday).

    Env: TWELVEDATA_API_KEY
    Timeframes supported: 1m,5m,15m,30m,1h,2h,4h,1d (subject to plan limits)
    """

    def __init__(self, cache_dir: Path):
        super().__init__(cache_dir)
        self.api_key = os.environ.get("TWELVEDATA_API_KEY")
        if not self.api_key:
            raise OSError("TWELVEDATA_API_KEY env var is required")
        self.cache = ParquetCache(cache_dir)
        self._limiter = RateLimiter(min_interval=0.25)

    def _map_tf(self, tf: str) -> str:
        tf = tf.lower()
        m = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1h",
            "2h": "2h",
            "4h": "4h",
            "1d": "1day",
        }
        if tf in m:
            return m[tf]
        raise ValueError(f"Unsupported timeframe for TwelveData: {tf}")

    def fetch(self, symbol: str, timeframe: str, only_cached: bool = False) -> pd.DataFrame:
        tf = timeframe.lower()
        cached = self.cache.load("twelvedata", symbol, tf)
        if cached is not None and len(cached) > 0:
            return cached
        if only_cached:
            raise RuntimeError(f"Cache miss for {symbol} {tf} (twelvedata) with only_cached=True")

        interval = self._map_tf(tf)
        sym_fetch = map_symbol("twelvedata", symbol)  # e.g., EURUSD -> EUR/USD
        # TwelveData time_series returns JSON with 'values' and 'datetime' fields
        # We'll fetch a broad range using 'start_date'; TD also supports 'outputsize'
        start = (datetime.now(timezone.utc) - timedelta(days=365)).date().isoformat()
        params = {
            "symbol": sym_fetch,
            "interval": interval,
            "start_date": start,
            "apikey": self.api_key,
            "format": "JSON",
            "order": "ASC",
            "dp": 8,
        }

        session = create_retry_session()
        self._limiter.acquire()
        url = "https://api.twelvedata.com/time_series"
        resp = session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json() or {}
        values = data.get("values") or []
        rows = []
        for r in values:
            ts = pd.to_datetime(r.get("datetime"), utc=True)
            if not isinstance(ts, pd.Timestamp):
                continue
            rows.append(
                [
                    ts,
                    float(r.get("open", 0.0)),
                    float(r.get("high", 0.0)),
                    float(r.get("low", 0.0)),
                    float(r.get("close", 0.0)),
                    float(r.get("volume", 0.0)),
                ]
            )
        if not rows:
            raise RuntimeError(f"No data from TwelveData for {symbol} {tf}")
        df = pd.DataFrame(
            rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"]
        ).set_index("Date")
        df.index = df.index.tz_convert(None)
        df = df.sort_index()
        self.cache.save("twelvedata", symbol, tf, df)
        return df
