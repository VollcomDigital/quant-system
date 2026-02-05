from __future__ import annotations

from pathlib import Path

from src.data.cache import ParquetCache


def test_parquet_cache_missing_returns_none(tmp_path: Path):
    cache = ParquetCache(tmp_path)
    assert cache.load("yfinance", "MSFT", "1d") is None
