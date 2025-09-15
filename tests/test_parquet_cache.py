import os
from pathlib import Path

import pytest

# Skip this module entirely in lightweight pre-commit environments
if os.environ.get("SKIP_PANDAS_TESTS") == "1":  # pragma: no cover
    pytest.skip("skipping pandas-dependent test in pre-commit", allow_module_level=True)

import pandas as pd

from src.data.cache import ParquetCache


def test_parquet_cache_roundtrip(tmp_path: Path):
    cache = ParquetCache(tmp_path)
    df = pd.DataFrame(
        {
            "Open": [1, 2, 3],
            "High": [1, 2, 3],
            "Low": [1, 2, 3],
            "Close": [1, 2, 3],
            "Volume": [10, 20, 30],
        },
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
    )
    cache.save("yfinance", "AAPL", "1d", df)
    got = cache.load("yfinance", "AAPL", "1d")
    assert got is not None
    assert len(got) == 3
    assert list(got.columns) == ["Open", "High", "Low", "Close", "Volume"]


pytestmark = []
