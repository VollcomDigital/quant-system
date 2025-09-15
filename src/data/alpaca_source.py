from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from ..utils.http import create_retry_session
from .base import DataSource
from .cache import ParquetCache
from .ratelimiter import RateLimiter
from .symbol_mapper import map_symbol


class AlpacaSource(DataSource):
    """Template for Alpaca Market Data API.

    Configure env vars ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY.
    Implement fetch logic for bars endpoint as needed.
    """

    def __init__(self, cache_dir: Path):
        super().__init__(cache_dir)
        self.api_key = os.environ.get("ALPACA_API_KEY_ID")
        self.api_secret = os.environ.get("ALPACA_API_SECRET_KEY")
        if not (self.api_key and self.api_secret):
            raise OSError("ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY env vars are required")
        self.cache = ParquetCache(cache_dir)
        self._limiter = RateLimiter(min_interval=0.25)

    def _map_tf(self, tf: str) -> str:
        tf = tf.lower()
        if tf.endswith("m"):
            return f"{int(tf[:-1])}Min"
        if tf.endswith("h"):
            return f"{int(tf[:-1])}Hour"
        if tf.endswith("d"):
            return "1Day"
        raise ValueError(f"Unsupported timeframe for Alpaca: {tf}")

    def _is_crypto(self, sym: str) -> bool:
        s = sym.upper()
        return "/" in s or s.endswith("USD") or s.endswith("USDT")

    def _map_crypto_symbol(self, sym: str) -> str:
        s = sym.upper().replace("USDT", "USD")
        if "/" not in s:
            # e.g., BTCUSD -> BTC/USD
            if s.endswith("USD"):
                return f"{s[:-3]}/USD"
        return s

    def fetch(self, symbol: str, timeframe: str, only_cached: bool = False) -> pd.DataFrame:
        tf = timeframe.lower()
        cached = self.cache.load("alpaca", symbol, tf)
        if cached is not None and len(cached) > 0:
            return cached
        if only_cached:
            raise RuntimeError(f"Cache miss for {symbol} {tf} (alpaca) with only_cached=True")

        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }
        rows = []
        session = create_retry_session()
        page_token = None

        sym_fetch = map_symbol("alpaca", symbol)
        if self._is_crypto(sym_fetch):
            # Crypto markets
            mapped = self._map_crypto_symbol(sym_fetch)
            url = "https://data.alpaca.markets/v1beta3/crypto/us/bars"
            params = {
                "symbols": mapped,
                "timeframe": self._map_tf(tf),
                "limit": 10000,
                "start": "2015-01-01T00:00:00Z",
            }
            while True:
                self._limiter.acquire()
                p = params.copy()
                if page_token:
                    p["page_token"] = page_token
                resp = session.get(url, params=p, headers=headers, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                bars = (data.get("bars") or {}).get(mapped, [])
                for b in bars:
                    ts = pd.to_datetime(b["t"], utc=True)
                    rows.append(
                        [
                            ts,
                            float(b.get("o", 0.0)),
                            float(b.get("h", 0.0)),
                            float(b.get("l", 0.0)),
                            float(b.get("c", 0.0)),
                            float(b.get("v", 0.0)),
                        ]
                    )
                page_token = data.get("next_page_token")
                if not page_token:
                    break
        else:
            # Stocks/ETFs
            url = "https://data.alpaca.markets/v2/stocks/bars"
            params = {
                "symbols": sym_fetch,
                "timeframe": self._map_tf(tf),
                "limit": 10000,
                "adjustment": "raw",
                "feed": "sip",
                "start": "1990-01-01T00:00:00Z",
            }
            while True:
                self._limiter.acquire()
                p = params.copy()
                if page_token:
                    p["page_token"] = page_token
                resp = session.get(url, params=p, headers=headers, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                bars = (data.get("bars") or {}).get(sym_fetch, [])
                for b in bars:
                    ts = pd.to_datetime(b["t"], utc=True)
                    rows.append(
                        [
                            ts,
                            float(b.get("o", 0.0)),
                            float(b.get("h", 0.0)),
                            float(b.get("l", 0.0)),
                            float(b.get("c", 0.0)),
                            float(b.get("v", 0.0)),
                        ]
                    )
                page_token = data.get("next_page_token")
                if not page_token:
                    break

        if not rows:
            raise RuntimeError(f"No data from Alpaca for {symbol} {tf}")

        df = pd.DataFrame(
            rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"]
        ).set_index("Date")
        df.index = df.index.tz_convert(None)
        df = df.sort_index()
        self.cache.save("alpaca", symbol, tf, df)
        return df
