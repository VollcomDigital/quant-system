from __future__ import annotations

from pathlib import Path

import pytest
import sqlite3

from src.backtest.results_cache import ENGINE_VERSION, ResultsCache, ResultsCacheRecord


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
    assert hit["metric_value"] == pytest.approx(1.25)
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
    assert rows[0]["metric_value"] == pytest.approx(1.25)


def test_results_cache_keeps_distinct_mode_entries(tmp_path: Path):
    cache = ResultsCache(tmp_path)
    common = {
        "collection": "demo",
        "symbol": "AAPL",
        "timeframe": "1d",
        "strategy": "strat",
        "params": {"x": 1},
        "metric_name": "sharpe",
        "data_fingerprint": "fp-1",
        "fees": 0.001,
        "slippage": 0.002,
        "run_id": "run-1",
    }
    cache.set(
        **common,
        metric_value=1.25,
        stats={"sharpe": 1.25, "mode": "backtest"},
        evaluation_mode="backtest",
        mode_config_hash="",
    )
    cache.set(
        **common,
        metric_value=0.75,
        stats={"sharpe": 0.75, "mode": "walk_forward"},
        evaluation_mode="walk_forward",
        mode_config_hash="wf-hash",
    )

    backtest_hit = cache.get(
        **common,
        evaluation_mode="backtest",
        mode_config_hash="",
    )
    walk_forward_hit = cache.get(
        **common,
        evaluation_mode="walk_forward",
        mode_config_hash="wf-hash",
    )

    assert backtest_hit is not None
    assert walk_forward_hit is not None
    assert backtest_hit["metric_value"] == pytest.approx(1.25)
    assert walk_forward_hit["metric_value"] == pytest.approx(0.75)
    assert backtest_hit["stats"]["mode"] == "backtest"
    assert walk_forward_hit["stats"]["mode"] == "walk_forward"


def test_results_cache_normalizes_evaluation_mode_for_set_and_get(tmp_path: Path):
    cache = ResultsCache(tmp_path)
    common = {
        "collection": "demo",
        "symbol": "AAPL",
        "timeframe": "1d",
        "strategy": "strat",
        "params": {"x": 1},
        "metric_name": "sharpe",
        "data_fingerprint": "fp-1",
        "fees": 0.001,
        "slippage": 0.002,
        "run_id": "run-1",
        "mode_config_hash": "",
    }
    cache.set(
        **common,
        metric_value=1.25,
        stats={"sharpe": 1.25, "mode": "backtest"},
        evaluation_mode=" BackTest ",
    )

    hit = cache.get(
        **common,
        evaluation_mode="BACKTEST",
    )

    assert hit is not None
    assert hit["metric_value"] == pytest.approx(1.25)


def test_results_cache_recovers_when_only_results_legacy_exists(tmp_path: Path):
    db_path = tmp_path / "results.sqlite"
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            """
            CREATE TABLE results_legacy (
                collection TEXT,
                symbol TEXT,
                timeframe TEXT,
                strategy TEXT,
                params_json TEXT,
                metric_name TEXT,
                metric_value REAL,
                stats_json TEXT,
                data_fingerprint TEXT,
                fees REAL,
                slippage REAL,
                run_id TEXT,
                evaluation_mode TEXT,
                mode_config_hash TEXT,
                engine_version TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(
                    collection,
                    symbol,
                    timeframe,
                    strategy,
                    params_json,
                    metric_name,
                    data_fingerprint,
                    fees,
                    slippage,
                    engine_version
                )
            )
            """
        )
        con.execute(
            """
            INSERT INTO results_legacy VALUES (
                'demo','AAPL','1d','strat','{"x": 1}','sharpe',1.0,'{"sharpe":1.0}',
                'fp',0.0,0.0,'run-1','backtest','',?,CURRENT_TIMESTAMP
            )
            """,
            (ENGINE_VERSION,),
        )
        con.commit()
    finally:
        con.close()

    cache = ResultsCache(tmp_path)
    hit = cache.get(
        collection="demo",
        symbol="AAPL",
        timeframe="1d",
        strategy="strat",
        params={"x": 1},
        metric_name="sharpe",
        data_fingerprint="fp",
        fees=0.0,
        slippage=0.0,
        evaluation_mode="backtest",
        mode_config_hash="",
    )
    assert hit is not None
    assert hit["metric_value"] == pytest.approx(1.0)


def test_results_cache_normalizes_evaluation_mode_when_setting_via_record(tmp_path: Path):
    cache = ResultsCache(tmp_path)
    record = ResultsCacheRecord(
        collection="demo",
        symbol="AAPL",
        timeframe="1d",
        strategy="strat",
        params={"x": 1},
        metric_name="sharpe",
        metric_value=1.25,
        stats={"sharpe": 1.25},
        data_fingerprint="fp-1",
        fees=0.001,
        slippage=0.002,
        run_id="run-1",
        evaluation_mode=" BackTest ",
        mode_config_hash="",
    )
    cache.set(record=record)

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
        evaluation_mode="backtest",
        mode_config_hash="",
    )
    assert hit is not None
    assert hit["metric_value"] == pytest.approx(1.25)
