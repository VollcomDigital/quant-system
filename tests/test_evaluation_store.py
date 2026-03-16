from __future__ import annotations

from pathlib import Path

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
        "validation_config_hash": "v1",
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
        **common,
        validation_config_hash="v1",
        metric_value=1.0,
        stats={"sharpe": 1.0},
    )
    cache.set(
        **common,
        validation_config_hash="v2",
        metric_value=2.0,
        stats={"sharpe": 2.0},
    )

    hit_v1 = cache.get(**common, validation_config_hash="v1")
    hit_v2 = cache.get(**common, validation_config_hash="v2")
    assert hit_v1 is not None and hit_v2 is not None
    assert hit_v1["metric_value"] == pytest.approx(1.0)
    assert hit_v2["metric_value"] == pytest.approx(2.0)


def test_result_store_run_metadata_round_trip(tmp_path: Path):
    store = ResultStore(tmp_path)
    store.upsert_run_metadata(
        run_id="run-1",
        evaluation_mode="backtest",
        mode_config_hash="abc",
        validation_profile={
            "global": {"data_quality": {"on_fail": "skip_job"}},
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
    assert row["validation_profile"]["global"]["data_quality"]["on_fail"] == "skip_job"
