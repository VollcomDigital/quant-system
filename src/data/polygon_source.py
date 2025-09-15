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


class PolygonSource(DataSource):
    """Template for Polygon.io data source.

    Configure env var POLYGON_API_KEY.
    Implement fetch logic for desired endpoint(s), e.g., aggregates.
    """

    def __init__(self, cache_dir: Path):
        super().__init__(cache_dir)
        self.api_key = os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            raise OSError("POLYGON_API_KEY env var is required")
        self.cache = ParquetCache(cache_dir)
        self._limiter = RateLimiter(min_interval=0.25)

    def _map_tf(self, tf: str) -> tuple[int, str]:
        tf = tf.lower()
        if tf.endswith("m"):
            return int(tf[:-1]), "minute"
        if tf.endswith("h"):
            return int(tf[:-1]), "hour"
        if tf.endswith("d"):
            return int(tf[:-1]), "day"
        raise ValueError(f"Unsupported timeframe for Polygon: {tf}")

    def fetch(self, symbol: str, timeframe: str, only_cached: bool = False) -> pd.DataFrame:
        tf = timeframe.lower()
        cached = self.cache.load("polygon", symbol, tf)
        if cached is not None and len(cached) > 0:
            return cached
        if only_cached:
            raise RuntimeError(f"Cache miss for {symbol} {tf} (polygon) with only_cached=True")

        mult, span = self._map_tf(tf)
        # Fetch in yearly chunks to respect response size limits
        start = datetime(1990, 1, 1, tzinfo=timezone.utc)
        end = datetime.now(timezone.utc)

        rows = []
        session = create_retry_session()
        sym_fetch = map_symbol("polygon", symbol)
        while start < end:
            self._limiter.acquire()
            chunk_end = min(start + timedelta(days=365 * 2), end)
            url = (
                f"https://api.polygon.io/v2/aggs/ticker/{sym_fetch}/range/{mult}/{span}/"
                f"{start.date()}"  # from
                f"/{chunk_end.date()}"  # to
            )
            params = {
                "adjusted": "true",
                "sort": "asc",
                "limit": 50000,
                "apiKey": self.api_key,
            }
            resp = session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", []) or []
            for r in results:
                ts = pd.to_datetime(r["t"], unit="ms", utc=True)
                rows.append(
                    [
                        ts,
                        float(r.get("o", 0.0)),
                        float(r.get("h", 0.0)),
                        float(r.get("l", 0.0)),
                        float(r.get("c", 0.0)),
                        float(r.get("v", 0.0)),
                    ]
                )
            start = chunk_end + timedelta(days=1)

        if not rows:
            raise RuntimeError(f"No data from Polygon for {symbol} {tf}")

        df = pd.DataFrame(
            rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"]
        ).set_index("Date")
        df.index = df.index.tz_convert(None)
        df = df.sort_index()
        self.cache.save("polygon", symbol, tf, df)
        return df
