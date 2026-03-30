from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pytest

from src.backtest.evaluation.contracts import EvaluationModeConfig, ResultRecord
from src.backtest.evaluation.store import EvaluationCache, EvaluationCacheRecord, ResultStore


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
        "validation_config_hash": "v1",
    }

    cache.set(
        record=EvaluationCacheRecord.from_mapping(
            {
                **common,
                "mode_config_hash": hash_a,
                "metric_value": 1.0,
                "stats": {"sharpe": 1.0},
            }
        )
    )
    cache.set(
        record=EvaluationCacheRecord.from_mapping(
            {
                **common,
                "mode_config_hash": hash_b,
                "metric_value": 2.0,
                "stats": {"sharpe": 2.0},
            }
        )
    )

    hit_a = cache.get(**common, mode_config_hash=hash_a)
    hit_b = cache.get(**common, mode_config_hash=hash_b)
    assert hit_a is not None and hit_b is not None
    assert hit_a["metric_value"] == pytest.approx(1.0)
    assert hit_b["metric_value"] == pytest.approx(2.0)


def test_evaluation_cache_uses_validation_hash(tmp_path: Path):
    cache = EvaluationCache(tmp_path)
    mode = EvaluationModeConfig(mode="backtest", payload={})
    mode_hash = cache.hash_mode_config(mode)
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
        "mode_config_hash": mode_hash,
    }
    cache.set(
        record=EvaluationCacheRecord.from_mapping(
            {
                **common,
                "validation_config_hash": "v1",
                "metric_value": 1.0,
                "stats": {"sharpe": 1.0},
            }
        )
    )
    cache.set(
        record=EvaluationCacheRecord.from_mapping(
            {
                **common,
                "validation_config_hash": "v2",
                "metric_value": 2.0,
                "stats": {"sharpe": 2.0},
            }
        )
    )

    hit_v1 = cache.get(**common, validation_config_hash="v1")
    hit_v2 = cache.get(**common, validation_config_hash="v2")
    assert hit_v1 is not None and hit_v2 is not None
    assert hit_v1["metric_value"] == pytest.approx(1.0)
    assert hit_v2["metric_value"] == pytest.approx(2.0)


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


def test_result_store_repair_deduplicates_with_last_write_wins(tmp_path: Path):
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

    con = sqlite3.connect(store.db_path)
    try:
        con.execute("DROP INDEX IF EXISTS idx_result_records_identity")
        con.execute(
            "CREATE INDEX idx_result_records_identity ON result_records(run_id, collection)"
        )
        params_json = json.dumps(base["params"], sort_keys=True)
        first_stats = json.dumps({"sharpe": 1.5}, sort_keys=True)
        second_stats = json.dumps({"sharpe": 2.0}, sort_keys=True)
        con.execute(
            """
            INSERT INTO result_records
            (
                run_id, evaluation_mode, collection, symbol, timeframe, source, strategy,
                params_json, metric_name, metric_value, stats_json, data_fingerprint, fees,
                slippage, mode_config_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                base["run_id"],
                base["evaluation_mode"],
                base["collection"],
                base["symbol"],
                base["timeframe"],
                base["source"],
                base["strategy"],
                params_json,
                base["metric_name"],
                1.5,
                first_stats,
                base["data_fingerprint"],
                base["fees"],
                base["slippage"],
                base["mode_config_hash"],
            ),
        )
        con.execute(
            """
            INSERT INTO result_records
            (
                run_id, evaluation_mode, collection, symbol, timeframe, source, strategy,
                params_json, metric_name, metric_value, stats_json, data_fingerprint, fees,
                slippage, mode_config_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                base["run_id"],
                base["evaluation_mode"],
                base["collection"],
                base["symbol"],
                base["timeframe"],
                base["source"],
                base["strategy"],
                params_json,
                base["metric_name"],
                2.0,
                second_stats,
                base["data_fingerprint"],
                base["fees"],
                base["slippage"],
                base["mode_config_hash"],
            ),
        )
        con.commit()
    finally:
        con.close()

    repaired = ResultStore(tmp_path)
    rows = repaired.list_by_run("run-1")
    assert len(rows) == 1
    assert rows[0]["metric_value"] == pytest.approx(2.0)


def test_result_store_run_metadata_round_trip(tmp_path: Path):
    store = ResultStore(tmp_path)
    store.upsert_run_metadata(
        run_id="run-1",
        evaluation_mode="backtest",
        mode_config_hash="abc",
        validation_profile={
            "collections": [],
        },
        active_gates=["data_quality.min_required_bars", "data_quality.min_data_points"],
        inactive_gates=["optimization.feasibility"],
    )

    row = store.get_run_metadata("run-1")
    assert row is not None
    assert row["run_id"] == "run-1"
    assert row["evaluation_mode"] == "backtest"
    assert row["mode_config_hash"] == "abc"
    assert row["active_gates"] == [
        "data_quality.min_required_bars",
        "data_quality.min_data_points",
    ]
    assert row["inactive_gates"] == ["optimization.feasibility"]
    assert row["validation_profile"]["collections"] == []
