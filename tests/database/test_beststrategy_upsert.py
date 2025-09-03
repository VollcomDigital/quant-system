from __future__ import annotations

from unittest.mock import patch

from src.core import direct_backtest
from src.database import unified_models


def setup_module(module):
    # Ensure sqlite tables exist for tests
    unified_models.create_tables()


def test_beststrategy_upsert_from_run(tmp_path):
    """
    Integration-style unit test that verifies run_strategy_comparison persists
    BacktestResult rows and then upserts a canonical BestStrategy row based on
    the configured target metric (sortino_ratio).

    Approach:
    - Create a run via create_run_from_manifest to obtain run_id.
    - Monkeypatch run_direct_backtest to return deterministic results for two strategies.
    - Call run_strategy_comparison with persistence_context containing run_id and target_metric.
    - Assert a BestStrategy row exists for the tested symbol/timeframe and matches the best metric.
    """
    # Create a run manifest and insert run row
    manifest = {
        "plan": {
            "plan_hash": "test-plan-hash-beststrategy",
            "actor": "test",
            "action": "backtest",
            "collection": "test_collection",
            "strategies": ["adx", "macd"],
            "intervals": ["1d"],
            "metric": "sortino_ratio",
        }
    }
    run = unified_models.create_run_from_manifest(manifest)
    assert run is not None
    run_id = run.run_id

    symbol = "TEST"
    start = "2020-01-01"
    end = "2020-12-31"
    timeframe = "1d"

    # Prepare deterministic fake results for two strategies
    def fake_run_direct_backtest(
        symbol_arg,
        strategy_name,
        start_date,
        end_date,
        timeframe_arg,
        initial_capital,
        persistence_context=None,
    ):
        # strategy 'macd' is better (higher sortino)
        if strategy_name == "adx":
            metrics = {"sortino_ratio": 0.5, "num_trades": 1}
        else:
            metrics = {"sortino_ratio": 2.0, "num_trades": 2}

        # Simulate persistence side-effect similar to _persist_result_to_db so the later
        # ranking/finalization code finds BacktestResult rows in the DB.
        try:
            sess = unified_models.Session()
            br = unified_models.BacktestResult(
                run_id=(persistence_context or {}).get("run_id"),
                symbol=symbol_arg,
                strategy=strategy_name,
                interval=timeframe_arg,
                start_at_utc=start_date,
                end_at_utc=end_date,
                rank_in_symbol=None,
                metrics=metrics,
                engine_ctx={"summary": "ok"},
                trades_raw=None,
                error=None,
            )
            sess.add(br)
            sess.flush()
            sess.commit()
        except Exception:
            try:
                sess.rollback()
            except Exception:
                pass
        finally:
            try:
                sess.close()
            except Exception:
                pass

        return {
            "symbol": symbol_arg,
            "strategy": strategy_name,
            "timeframe": timeframe_arg,
            "error": None,
            "metrics": metrics,
            "trades": None,
            "backtest_object": None,
            "bt_results": {"summary": "ok"},
            "start_date": start_date,
            "end_date": end_date,
        }

    # Patch the run_direct_backtest used by run_strategy_comparison
    with patch(
        "src.core.direct_backtest.run_direct_backtest",
        side_effect=fake_run_direct_backtest,
    ):
        out = direct_backtest.run_strategy_comparison(
            symbol,
            ["adx", "macd"],
            start,
            end,
            timeframe,
            initial_capital=10000.0,
            persistence_context={"run_id": run_id, "target_metric": "sortino_ratio"},
        )

    # Validate output contains best_strategy with macd
    assert out["best_strategy"] is not None
    assert (
        out["best_strategy"]["strategy"] == "macd"
        or out["best_strategy"]["strategy"] == "MACD"
        or out["best_strategy"]["strategy"].lower() == "macd"
    )

    # Now verify BestStrategy upsert exists in unified_models
    sess = unified_models.Session()
    try:
        bs = (
            sess.query(unified_models.BestStrategy)
            .filter_by(symbol=symbol, timeframe=timeframe)
            .one_or_none()
        )
        assert bs is not None, "BestStrategy was not upserted into the DB"
        assert bs.strategy.lower() == "macd"
        # Check sortino value was recorded (numeric-ish)
        try:
            val = float(bs.sortino_ratio)
            assert val >= 2.0
        except Exception:
            # If stored as JSON/text, still ensure the string contains '2'
            assert "2" in str(bs.sortino_ratio)
    finally:
        sess.close()
