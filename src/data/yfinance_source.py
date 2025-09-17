from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf

from .base import DataSource
from .cache import ParquetCache
from .ratelimiter import RateLimiter
from .symbol_mapper import map_symbol

YFINANCE_TF_MAP: dict[str, str] = {
    "1m": "1m",
    "2m": "2m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "60m": "60m",
    "90m": "90m",
    "1h": "60m",
    "1d": "1d",
    "1w": "1wk",
    "1wk": "1wk",
    "1mo": "1mo",
}


class YFinanceSource(DataSource):
    def __init__(self, cache_dir: Path):
        super().__init__(cache_dir)
        self.cache = ParquetCache(cache_dir)
        # Space requests to avoid Yahoo rate limits
        self._limiter = RateLimiter(min_interval=1.0)
        # Optional yfinance-cache integration
        self._yfc = None
        try:
            from yfinance_cache import YFCache  # type: ignore

            # Use a dedicated cache dir inside data cache
            self._yfc = YFCache(cache_dir=str(Path(cache_dir) / "yfinance-http"))
        except Exception:
            self._yfc = None
        # Yahoo's chart endpoint is more tolerant when an explicit UA is sent.
        self._http_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
            )
        }

    def _direct_chart_download(self, symbol: str, interval: str) -> pd.DataFrame | None:
        """Fallback downloader that hits Yahoo's chart API directly.

        Helpful for FX pairs where yfinance sometimes fails with timezone errors.
        """

        # Yahoo restricts how far back intraday intervals can go.
        interval = interval.lower()
        if interval in {"1m", "2m"}:
            data_range = "7d"
        elif interval in {"5m", "15m", "30m", "90m"}:
            data_range = "60d"
        elif interval in {"60m", "1h"}:
            data_range = "730d"
        else:
            data_range = "max"

        url = "https://query2.finance.yahoo.com/v8/finance/chart/" + symbol
        query = urlencode({"interval": interval, "range": data_range})
        req = Request(url + "?" + query, headers=self._http_headers)
        try:
            with urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

        chart = payload.get("chart") or {}
        results = chart.get("result") or []
        if not results:
            return None
        result = results[0]
        timestamps = result.get("timestamp") or []
        indicators = result.get("indicators") or {}
        quotes = (indicators.get("quote") or [{}])[0]
        if not timestamps or not quotes:
            return None

        def _series(key: str) -> list[Any]:
            vals = quotes.get(key)
            if vals is None:
                return [None] * len(timestamps)
            return vals

        frame = pd.DataFrame(
            {
                "Open": _series("open"),
                "High": _series("high"),
                "Low": _series("low"),
                "Close": _series("close"),
                "Volume": _series("volume"),
            },
            index=pd.to_datetime(timestamps, unit="s", utc=True),
        )
        frame.index = frame.index.tz_convert(None)
        frame = frame.dropna(subset=["Close"])
        frame = frame.apply(pd.to_numeric, errors="coerce")
        frame = frame.dropna(subset=["Close"])
        if frame.empty:
            return None
        return frame

    def fetch(self, symbol: str, timeframe: str, only_cached: bool = False) -> pd.DataFrame:
        tf = timeframe.lower()
        cached = self.cache.load("yfinance", symbol, tf)
        if cached is not None and len(cached) > 0:
            return cached
        if only_cached:
            raise RuntimeError(f"Cache miss for {symbol} {tf} (yfinance) with only_cached=True")

        if tf not in YFINANCE_TF_MAP:
            raise ValueError(f"Unsupported timeframe for yfinance: {tf}")

        interval = YFINANCE_TF_MAP[tf]
        self._limiter.acquire()
        sym_fetch = map_symbol("yfinance", symbol)

        # Retry/backoff wrapper for yfinance with multiple fallbacks
        def try_download() -> pd.DataFrame:
            # 1) yfinance-cache (if available)
            if self._yfc is not None:
                try:
                    ticker = self._yfc.ticker.Ticker(sym_fetch)  # type: ignore[attr-defined]
                    df_yfc = ticker.history(period="max", interval=interval, auto_adjust=False)
                    if df_yfc is not None and not df_yfc.empty:
                        return df_yfc
                except Exception:
                    pass
            # 2) yf.download
            try:
                df_dl = yf.download(
                    sym_fetch, period="max", interval=interval, auto_adjust=False, progress=False
                )
                if df_dl is not None and not df_dl.empty:
                    return df_dl
            except Exception:
                pass
            # 3) Ticker().history direct (often more robust for futures like ZW=F)
            t = yf.Ticker(sym_fetch)
            return t.history(period="max", interval=interval, auto_adjust=False)

        def run_with_optional_cache_disabled() -> pd.DataFrame | None:
            try:
                import requests_cache  # type: ignore

                with requests_cache.disabled():
                    backoff = 1.0
                    max_backoff = 30.0
                    for _ in range(5):
                        try:
                            return try_download()
                        except Exception:
                            time.sleep(backoff)
                            backoff = min(max_backoff, backoff * 2)
                    return try_download()
            except Exception:
                backoff = 1.0
                max_backoff = 30.0
                for _ in range(5):
                    try:
                        return try_download()
                    except Exception:
                        time.sleep(backoff)
                        backoff = min(max_backoff, backoff * 2)
                return try_download()

        df = run_with_optional_cache_disabled()
        if df is None or df.empty:
            df = self._direct_chart_download(sym_fetch, interval)
            if df is None or df.empty:
                raise RuntimeError(f"No data for {symbol} {tf}")
        df = df.rename(columns={c: c.split()[0] for c in df.columns})
        df.index = df.index.tz_localize(None)

        self.cache.save("yfinance", symbol, tf, df)
        return df
