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


class TiingoSource(DataSource):
    """Template for Tiingo data source.

    Configure env var TIINGO_API_KEY.
    Implement fetch logic for IEX/Tiingo endpoints as needed.
    """

    def __init__(self, cache_dir: Path):
        super().__init__(cache_dir)
        self.api_key = os.environ.get("TIINGO_API_KEY")
        if not self.api_key:
            raise OSError("TIINGO_API_KEY env var is required")
        self.cache = ParquetCache(cache_dir)
        self._limiter = RateLimiter(min_interval=0.25)

    def fetch(self, symbol: str, timeframe: str, only_cached: bool = False) -> pd.DataFrame:
        tf = timeframe.lower()
        cached = self.cache.load("tiingo", symbol, tf)
        if cached is not None and len(cached) > 0:
            return cached
        if only_cached:
            raise RuntimeError(f"Cache miss for {symbol} {tf} (tiingo) with only_cached=True")

        session = create_retry_session()
        headers = {"Content-Type": "application/json"}
        params_base = {"token": self.api_key}

        rows = []
        sym_fetch = map_symbol("tiingo", symbol)

        if tf.endswith("d"):
            # Daily: /tiingo/daily/{ticker}/prices
            url = f"https://api.tiingo.com/tiingo/daily/{sym_fetch}/prices"
            params = params_base | {"startDate": "1990-01-01"}
            self._limiter.acquire()
            resp = session.get(url, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json() or []
            for r in data:
                ts = pd.to_datetime(r["date"], utc=True)
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
        else:
            # Intraday via IEX endpoint: /iex/{ticker}/prices with resampleFreq
            def map_tf(tf: str) -> str:
                if tf.endswith("m"):
                    return f"{int(tf[:-1])}min"
                if tf.endswith("h"):
                    return f"{int(tf[:-1]) * 60}min"
                raise ValueError(f"Unsupported intraday timeframe for Tiingo IEX: {tf}")

            resample = map_tf(tf)
            start = datetime(2010, 1, 1, tzinfo=timezone.utc)
            end = datetime.now(timezone.utc)
            while start < end:
                chunk_end = min(start + timedelta(days=30), end)
                url = f"https://api.tiingo.com/iex/{sym_fetch}/prices"
                params = params_base | {
                    "startDate": start.date().isoformat(),
                    "endDate": chunk_end.date().isoformat(),
                    "resampleFreq": resample,
                }
                self._limiter.acquire()
                resp = session.get(url, params=params, headers=headers, timeout=30)
                resp.raise_for_status()
                data = resp.json() or []
                for r in data:
                    ts = pd.to_datetime(r["date"], utc=True)
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
                start = chunk_end + timedelta(days=1)

        if not rows:
            raise RuntimeError(f"No data from Tiingo for {symbol} {tf}")

        df = pd.DataFrame(
            rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"]
        ).set_index("Date")
        df.index = df.index.tz_convert(None)
        df = df.sort_index()
        self.cache.save("tiingo", symbol, tf, df)
        return df
