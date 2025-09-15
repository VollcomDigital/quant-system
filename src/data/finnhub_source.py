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


class FinnhubSource(DataSource):
    """Finnhub.io data source (focused on FX intraday).

    Env:
      - FINNHUB_API_KEY

    Timeframes supported (no resampling): 1m, 5m, 15m, 30m, 1h, 4h, 1d
    """

    def __init__(self, cache_dir: Path):
        super().__init__(cache_dir)
        self.api_key = os.environ.get("FINNHUB_API_KEY")
        if not self.api_key:
            raise OSError("FINNHUB_API_KEY env var is required")
        self.cache = ParquetCache(cache_dir)
        self._limiter = RateLimiter(min_interval=0.25)

    def _map_tf(self, tf: str) -> str:
        tf = tf.lower()
        if tf in ("1m", "5m", "15m", "30m"):
            return tf[:-1]  # 1,5,15,30
        if tf == "1h":
            return "60"
        if tf == "4h":
            return "240"
        if tf == "1d":
            return "D"
        raise ValueError(f"Unsupported timeframe for Finnhub: {tf}")

    def _is_fx(self, symbol: str) -> bool:
        s = symbol.replace("/", "").upper()
        return len(s) == 6 and s.isalpha()

    def fetch(self, symbol: str, timeframe: str, only_cached: bool = False) -> pd.DataFrame:
        tf = timeframe.lower()
        cached = self.cache.load("finnhub", symbol, tf)
        if cached is not None and len(cached) > 0:
            return cached
        if only_cached:
            raise RuntimeError(f"Cache miss for {symbol} {tf} (finnhub) with only_cached=True")

        mapped = map_symbol("finnhub", symbol)
        res = self._map_tf(tf)

        # Finnhub candles require a from/to Unix time range
        end = datetime.now(timezone.utc)
        # Pull a generous lookback window without pagination; adjust by tf
        lookback_days = 365 * 5 if tf == "1d" else 365
        start = end - timedelta(days=lookback_days)

        session = create_retry_session()
        self._limiter.acquire()

        # FX route
        if self._is_fx(symbol):
            url = "https://finnhub.io/api/v1/forex/candle"
            params = {
                "symbol": mapped,  # e.g., OANDA:EUR_USD
                "resolution": res,
                "from": int(start.timestamp()),
                "to": int(end.timestamp()),
                "token": self.api_key,
            }
        else:
            # Equities/ETFs (if used): /stock/candle
            url = "https://finnhub.io/api/v1/stock/candle"
            params = {
                "symbol": mapped,
                "resolution": res,
                "from": int(start.timestamp()),
                "to": int(end.timestamp()),
                "token": self.api_key,
            }

        resp = session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data or data.get("s") != "ok":
            raise RuntimeError(f"No data from Finnhub for {symbol} {tf}")
        # Finnhub returns arrays: t, o, h, low, c, v
        rows = []
        for t, o, h, low, c, v in zip(
            data.get("t", []),
            data.get("o", []),
            data.get("h", []),
            data.get("l", []),
            data.get("c", []),
            data.get("v", []),
            strict=False,
        ):
            ts = pd.to_datetime(t, unit="s", utc=True)
            rows.append([ts, float(o), float(h), float(low), float(c), float(v)])

        if not rows:
            raise RuntimeError(f"No data from Finnhub for {symbol} {tf}")

        df = pd.DataFrame(
            rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"]
        ).set_index("Date")
        df.index = df.index.tz_convert(None)
        df = df.sort_index()
        self.cache.save("finnhub", symbol, tf, df)
        return df
