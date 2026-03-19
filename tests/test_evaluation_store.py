from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from src.backtest.evaluation.contracts import EvaluationModeConfig, ResultRecord
from src.backtest.evaluation.store import EvaluationCache, ResultStore


def test_evaluation_cache_uses_mode_hash(tmp_path: Path):
    cache = EvaluationCache(tmp_path)
    mode_a = EvaluationModeConfig(mode="backtest", payload={"window": 100})
    mode_b = EvaluationModeConfig(mode="backtest", payload={"window": 200})
    hash_a = cache.hash_mode_config(mode_a)
    hash_b = cache.hash_mode_config(mode_b)
    assert hash_a != hash_b

    common = {
        "collection": "c",
        "symbol": "AAPL",
        "timeframe": "1d",
        "strategy": "s",
        "params": {"x": 1},
        "metric_name": "sharpe",
        "data_fingerprint": "fp",
        "fees": 0.0,
        "slippage": 0.0,
        "evaluation_mode": "backtest",
    }

    cache.set(
        **common,
        mode_config_hash=hash_a,
        metric_value=1.0,
        stats={"sharpe": 1.0},
    )
    cache.set(
        **common,
        mode_config_hash=hash_b,
        metric_value=2.0,
        stats={"sharpe": 2.0},
    )

    hit_a = cache.get(**common, mode_config_hash=hash_a)
    hit_b = cache.get(**common, mode_config_hash=hash_b)
    assert hit_a is not None and hit_b is not None
    assert hit_a["metric_value"] == pytest.approx(1.0)
    assert hit_b["metric_value"] == pytest.approx(2.0)


def test_result_store_round_trip(tmp_path: Path):
    store = ResultStore(tmp_path)
    record = ResultRecord(
        run_id="run-1",
        evaluation_mode="backtest",
        collection="c",
        symbol="AAPL",
        timeframe="1d",
        source="yfinance",
        strategy="s",
        params={"x": 1},
        metric_name="sharpe",
        metric_value=1.5,
        stats={"sharpe": 1.5},
        data_fingerprint="fp",
        fees=0.0,
        slippage=0.0,
        mode_config_hash="abc",
    )
    store.insert(record)

    rows = store.list_by_run("run-1")
    assert len(rows) == 1
    row = rows[0]
    assert row["evaluation_mode"] == "backtest"
    assert row["metric"] == "sharpe"
    assert row["metric_value"] == pytest.approx(1.5)
    assert row["params"] == {"x": 1}


def test_result_store_insert_is_idempotent_per_record_identity(tmp_path: Path):
    store = ResultStore(tmp_path)
    base = {
        "run_id": "run-1",
        "evaluation_mode": "backtest",
        "collection": "c",
        "symbol": "AAPL",
        "timeframe": "1d",
        "source": "yfinance",
        "strategy": "s",
        "params": {"x": 1},
        "metric_name": "sharpe",
        "data_fingerprint": "fp",
        "fees": 0.0,
        "slippage": 0.0,
        "mode_config_hash": "abc",
    }

    store.insert(ResultRecord(metric_value=1.5, stats={"sharpe": 1.5}, **base))
    store.insert(ResultRecord(metric_value=2.0, stats={"sharpe": 2.0}, **base))

    rows = store.list_by_run("run-1")
    assert len(rows) == 1
    assert rows[0]["metric_value"] == pytest.approx(2.0)


def test_result_store_repairs_malformed_identity_index(tmp_path: Path):
    store = ResultStore(tmp_path)
    con = sqlite3.connect(store.db_path)
    try:
        con.execute("DROP INDEX IF EXISTS idx_result_records_identity")
        con.execute(
            "CREATE INDEX idx_result_records_identity ON result_records(run_id, collection)"
        )
        con.commit()
    finally:
        con.close()

    repaired = ResultStore(tmp_path)
    con = sqlite3.connect(repaired.db_path)
    try:
        index_rows = con.execute("PRAGMA index_list('result_records')").fetchall()
        identity_rows = [row for row in index_rows if row[1] == "idx_result_records_identity"]
        assert len(identity_rows) == 1
        assert int(identity_rows[0][2]) == 1
        index_info = con.execute("PRAGMA index_info('idx_result_records_identity')").fetchall()
        index_columns = [row[2] for row in sorted(index_info, key=lambda row: int(row[0]))]
        assert index_columns == list(ResultStore._IDENTITY_COLUMNS)
    finally:
        con.close()
