from __future__ import annotations

import time
from pathlib import Path

import ccxt
import pandas as pd

from .base import DataSource
from .cache import ParquetCache
from .ratelimiter import RateLimiter
from .symbol_mapper import map_symbol

CCXT_TF_MAP: dict[str, str] = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "12h": "12h",
    "1d": "1d",
    "1w": "1w",
    "1M": "1M",
}


class CCXTSource(DataSource):
    def __init__(self, exchange: str, cache_dir: Path):
        super().__init__(cache_dir)
        self.exchange_name = exchange
        self.exchange = getattr(ccxt, exchange)({"enableRateLimit": True})
        self.cache = ParquetCache(cache_dir)
        # Extra inter-call limiter (in addition to ccxt's internal rate limit)
        self._limiter = RateLimiter(min_interval=self.exchange.rateLimit / 1000.0)

    def fetch(self, symbol: str, timeframe: str, only_cached: bool = False) -> pd.DataFrame:
        tf = timeframe
        cached = self.cache.load(self.exchange_name, symbol, tf)
        if cached is not None and len(cached) > 0:
            return cached
        if only_cached:
            raise RuntimeError(
                f"Cache miss for {symbol} {tf} ({self.exchange_name}) with only_cached=True"
            )

        if tf not in CCXT_TF_MAP:
            raise ValueError(f"Unsupported timeframe for ccxt: {tf}")

        # Map common variants to CCXT's expected format (e.g., BTCUSDT -> BTC/USDT)
        sym_fetch = map_symbol(self.exchange_name, symbol)

        ohlcv = []
        limit = 1000
        since = None
        backoff = 1.0
        max_backoff = 60.0
        while True:
            self._limiter.acquire()
            try:
                batch = self.exchange.fetch_ohlcv(sym_fetch, timeframe=tf, since=since, limit=limit)
            except Exception as e:
                # Handle ccxt-specific throttle/availability errors with backoff
                import ccxt

                if isinstance(
                    e,
                    (
                        ccxt.RateLimitExceeded,
                        ccxt.DDoSProtection,
                        ccxt.ExchangeNotAvailable,
                        ccxt.NetworkError,
                    ),
                ):
                    time.sleep(backoff)
                    backoff = min(max_backoff, backoff * 2)
                    continue
                raise
            backoff = 1.0
            if not batch:
                break
            ohlcv.extend(batch)
            if len(batch) < limit:
                break
            since = batch[-1][0] + 1
            # extra safety sleep to respect exchange rate limits
            time.sleep(self.exchange.rateLimit / 1000.0)

        if not ohlcv:
            raise RuntimeError(f"No data for {symbol} {tf} on {self.exchange_name}")

        df = pd.DataFrame(ohlcv, columns=["Date", "Open", "High", "Low", "Close", "Volume"])  # type: ignore
        df["Date"] = pd.to_datetime(df["Date"], unit="ms")
        df = df.set_index("Date").sort_index()

        self.cache.save(self.exchange_name, symbol, tf, df)
        return df
