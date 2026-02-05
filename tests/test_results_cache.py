from __future__ import annotations

from pathlib import Path

from src.backtest.results_cache import ResultsCache


def test_results_cache_set_get_and_list(tmp_path: Path):
    cache = ResultsCache(tmp_path)
    cache.set(
        collection="demo",
        symbol="AAPL",
        timeframe="1d",
        strategy="strat",
        params={"x": 1},
        metric_name="sharpe",
        metric_value=1.25,
        stats={"sharpe": 1.25, "trades": 2},
        data_fingerprint="fp-1",
        fees=0.001,
        slippage=0.002,
        run_id="run-1",
    )

    hit = cache.get(
        collection="demo",
        symbol="AAPL",
        timeframe="1d",
        strategy="strat",
        params={"x": 1},
        metric_name="sharpe",
        data_fingerprint="fp-1",
        fees=0.001,
        slippage=0.002,
        run_id="run-1",
    )
    assert hit is not None
    assert hit["metric_value"] == 1.25
    assert hit["stats"]["trades"] == 2

    miss = cache.get(
        collection="demo",
        symbol="AAPL",
        timeframe="1d",
        strategy="strat",
        params={"x": 2},
        metric_name="sharpe",
        data_fingerprint="fp-1",
        fees=0.001,
        slippage=0.002,
        run_id="run-1",
    )
    assert miss is None

    rows = cache.list_by_run("run-1")
    assert len(rows) == 1
    assert rows[0]["params"]["x"] == 1
    assert rows[0]["metric_value"] == 1.25
