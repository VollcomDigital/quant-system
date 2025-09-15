from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from ..utils.http import create_retry_session
from .base import DataSource
from .cache import ParquetCache
from .ratelimiter import RateLimiter


class AlphaVantageSource(DataSource):
    """Alpha Vantage (daily focus, no resampling).

    Env: ALPHAVANTAGE_API_KEY
    Supports: 1d daily for FX and US equities/ETFs. Intraday not recommended due to limits.
    """

    def __init__(self, cache_dir: Path):
        super().__init__(cache_dir)
        self.api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            raise OSError("ALPHAVANTAGE_API_KEY env var is required")
        self.cache = ParquetCache(cache_dir)
        self._limiter = RateLimiter(min_interval=12.0)  # respect 5 req/min free tier

    def fetch(self, symbol: str, timeframe: str, only_cached: bool = False) -> pd.DataFrame:
        tf = timeframe.lower()
        if tf != "1d":
            raise ValueError("AlphaVantageSource supports only 1d without resampling")
        cached = self.cache.load("alphavantage", symbol, tf)
        if cached is not None and len(cached) > 0:
            return cached
        if only_cached:
            raise RuntimeError(f"Cache miss for {symbol} {tf} (alphavantage) with only_cached=True")

        s = symbol.replace("/", "").upper()
        session = create_retry_session()
        self._limiter.acquire()

        # FX or Equity detection
        if len(s) == 6 and s.isalpha():
            # FX_DAILY
            from_sym, to_sym = s[:3], s[3:]
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "FX_DAILY",
                "from_symbol": from_sym,
                "to_symbol": to_sym,
                "apikey": self.api_key,
                "outputsize": "full",
                "datatype": "json",
            }
            resp = session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json() or {}
            series = data.get("Time Series FX (Daily)", {})
            rows = []
            for k in sorted(series.keys()):
                r = series[k]
                ts = pd.to_datetime(k, utc=True)
                rows.append(
                    [
                        ts,
                        float(r.get("1. open", 0.0)),
                        float(r.get("2. high", 0.0)),
                        float(r.get("3. low", 0.0)),
                        float(r.get("4. close", 0.0)),
                        0.0,
                    ]
                )
        else:
            # TIME_SERIES_DAILY_ADJUSTED for equities/ETFs
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": symbol.upper(),
                "outputsize": "full",
                "apikey": self.api_key,
                "datatype": "json",
            }
            resp = session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json() or {}
            series = data.get("Time Series (Daily)", {})
            rows = []
            for k in sorted(series.keys()):
                r = series[k]
                ts = pd.to_datetime(k, utc=True)
                rows.append(
                    [
                        ts,
                        float(r.get("1. open", 0.0)),
                        float(r.get("2. high", 0.0)),
                        float(r.get("3. low", 0.0)),
                        float(r.get("4. close", 0.0)),
                        float(r.get("6. volume", 0.0)),
                    ]
                )

        if not rows:
            raise RuntimeError(f"No data from AlphaVantage for {symbol} {tf}")
        df = pd.DataFrame(
            rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"]
        ).set_index("Date")
        df.index = df.index.tz_convert(None)
        df = df.sort_index()
        self.cache.save("alphavantage", symbol, tf, df)
        return df
