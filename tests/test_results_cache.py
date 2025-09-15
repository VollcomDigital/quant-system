from pathlib import Path

from src.backtest.results_cache import ResultsCache


def test_results_cache_set_get(tmp_path: Path):
    cache = ResultsCache(tmp_path)
    key = dict(a=1)
    stats = {"sharpe": 1.23, "profit": 0.12}
    cache.set(
        collection="test",
        symbol="SYMB",
        timeframe="1d",
        strategy="strat",
        params=key,
        metric_name="sharpe",
        metric_value=1.23,
        stats=stats,
        data_fingerprint="10:2020-01-01T00:00:00:100.0",
        fees=0.001,
        slippage=0.001,
        run_id="run-1",
    )
    got = cache.get(
        collection="test",
        symbol="SYMB",
        timeframe="1d",
        strategy="strat",
        params=key,
        metric_name="sharpe",
        data_fingerprint="10:2020-01-01T00:00:00:100.0",
        fees=0.001,
        slippage=0.001,
    )
    assert got is not None
    assert got["metric_value"] == 1.23
    assert got["stats"]["profit"] == 0.12

    rows = cache.list_by_run("run-1")
    assert len(rows) == 1
    assert rows[0]["strategy"] == "strat"
