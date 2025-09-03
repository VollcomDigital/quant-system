"""
Direct Backtesting Library Integration
Direct backtesting using the backtesting library.

This file was extended to support optional persistence of backtest results into the
project database via the lightweight unified_models helper (src.database.unified_models).
Pass an optional persistence_context (dict) to run_direct_backtest / run_strategy_comparison
to enable DB writes. Persistence is best-effort: guarded imports and safe upsert logic.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from backtesting import Backtest

from .backtest_engine import create_backtesting_strategy_adapter
from .data_manager import UnifiedDataManager
from .strategy import StrategyFactory


# Local utilities used when persisting
def _persist_result_to_db(
    result: Dict[str, Any], persistence_context: Dict[str, Any]
) -> None:
    """
    Persist a single backtest result into the DB using src.database.unified_models.
    This function is best-effort and will not raise on failures (only logs).
    persistence_context must include at least:
      - run_id (str)
      - target_metric (str) optional
      - session_factory or rely on unified_models.Session
    Added debug logging to help diagnose missing metrics persistence.
    """
    logger = logging.getLogger(__name__)
    logger.debug(
        "Called _persist_result_to_db for symbol=%s strategy=%s",
        result.get("symbol"),
        result.get("strategy"),
    )
    logger.debug(
        "Persistence context keys: %s",
        list((persistence_context or {}).keys())
        if persistence_context is not None
        else None,
    )

    try:
        from src.database import unified_models  # type: ignore[import-not-found]
        from src.utils.trades_parser import (
            parse_trades_from_string,  # type: ignore[import-not-found]
        )
    except Exception:
        logger.debug(
            "Persistence not available (unified_models or trades parser missing)"
        )
        return

    try:
        sess = unified_models.Session()
    except Exception as e:
        logger.exception("Failed to create unified_models.Session(): %s", e)
        return
    try:
        run_id = persistence_context.get("run_id")
        # If run_id is missing or falsy, avoid attempting DB writes which will violate NOT NULL constraints.
        if not run_id:
            logging.getLogger(__name__).debug(
                "Persistence context provided but run_id is missing; skipping DB persistence for %s",
                result.get("symbol"),
            )
            return

        # Check for existing BacktestResult (idempotency)
        existing = (
            sess.query(unified_models.BacktestResult)
            .filter(
                unified_models.BacktestResult.run_id == run_id,
                unified_models.BacktestResult.symbol == result.get("symbol"),
                unified_models.BacktestResult.strategy == result.get("strategy"),
                unified_models.BacktestResult.interval == result.get("timeframe"),
            )
            .one_or_none()
        )

        # Prepare payload
        metrics = result.get("metrics") or {}
        # Try to convert native backtesting stats to a plain dict unconditionally
        raw_stats = result.get("bt_results")
        engine_ctx = None
        try:
            if raw_stats is not None:
                engine_ctx = (
                    raw_stats if isinstance(raw_stats, dict) else dict(raw_stats)
                )
        except Exception:
            engine_ctx = None

        # If metrics were not provided, derive a few canonical ones from engine_ctx
        # so downstream ranking (target_metric) has values to work with.
        if not metrics and engine_ctx and isinstance(engine_ctx, dict):
            try:

                def _as_float(v):
                    try:
                        return float(v)
                    except Exception:
                        return None

                # Backtesting.py common keys
                sortino = engine_ctx.get("Sortino Ratio")
                calmar = engine_ctx.get("Calmar Ratio")
                sharpe = engine_ctx.get("Sharpe Ratio")
                total_ret = engine_ctx.get("Return [%]")
                max_dd = engine_ctx.get("Max. Drawdown [%]") or engine_ctx.get(
                    "Max Drawdown [%]"
                )
                num_trades = engine_ctx.get("# Trades")

                derived = {}
                if sortino is not None:
                    derived["sortino_ratio"] = _as_float(sortino)
                    derived["Sortino_Ratio"] = derived["sortino_ratio"]
                if calmar is not None:
                    derived["calmar_ratio"] = _as_float(calmar)
                    derived["Calmar_Ratio"] = derived["calmar_ratio"]
                if sharpe is not None:
                    derived["sharpe_ratio"] = _as_float(sharpe)
                    derived["Sharpe_Ratio"] = derived["sharpe_ratio"]
                if total_ret is not None:
                    derived["total_return"] = _as_float(total_ret)
                    derived["Total_Return"] = derived["total_return"]
                if max_dd is not None:
                    derived["max_drawdown"] = _as_float(max_dd)
                    derived["Max_Drawdown"] = derived["max_drawdown"]
                if num_trades is not None:
                    # leave as float to keep consistent handling downstream
                    derived["num_trades"] = _as_float(num_trades)

                metrics = derived
            except Exception:
                # Best-effort only; leave metrics as-is if derivation fails
                pass

        # Sanitize JSON-like payloads: replace NaN/Inf with None and convert numpy/pandas objects.
        def _sanitize_jsonable(obj):
            try:
                import math
            except Exception:
                math = None
            try:
                import pandas as _pd  # type: ignore[import-not-found]
            except Exception:
                _pd = None
            try:
                import numpy as _np  # type: ignore[import-not-found]
            except Exception:
                _np = None

            # Pandas DataFrame/Series first: convert then recurse to sanitize nested values
            try:
                if _pd is not None and isinstance(obj, _pd.DataFrame):
                    recs = obj.to_dict(orient="records")
                    return _sanitize_jsonable(recs)
                if _pd is not None and isinstance(obj, _pd.Series):
                    return _sanitize_jsonable(obj.to_dict())
            except Exception:
                pass

            # Primitive safe types
            if obj is None:
                return None
            if isinstance(obj, (str, bool, int)):
                return obj
            # Floats: guard against NaN / Inf which are invalid in JSONB
            if isinstance(obj, float):
                try:
                    if math is not None and (math.isnan(obj) or math.isinf(obj)):
                        return None
                except Exception:
                    return None
                return obj
            # Numpy scalars
            try:
                if _np is not None and isinstance(obj, _np.generic):
                    return _sanitize_jsonable(obj.item())
            except Exception:
                pass
            # Dicts and lists: recurse
            if isinstance(obj, dict):
                out = {}
                for k, v in obj.items():
                    try:
                        out[k] = _sanitize_jsonable(v)
                    except Exception:
                        out[k] = None
                return out
            if isinstance(obj, (list, tuple)):
                return [_sanitize_jsonable(v) for v in obj]
            # Fallback: try to coerce to string safely
            try:
                return str(obj)
            except Exception:
                return None

        # Apply sanitization before persisting into JSONB columns
        try:
            metrics = _sanitize_jsonable(metrics)
        except Exception:
            metrics = {}
        try:
            engine_ctx = _sanitize_jsonable(engine_ctx)
        except Exception:
            engine_ctx = None

        trades_raw = None
        trades_obj = result.get("trades")
        if trades_obj is not None:
            try:
                if isinstance(trades_obj, pd.DataFrame):
                    trades_raw = trades_obj.to_csv(index=False)
                else:
                    # If it's a list/dict or other, try json
                    import json as _json  # local import

                    trades_raw = _json.dumps(trades_obj)
            except Exception:
                trades_raw = str(trades_obj)

        # Attach equity curve into engine_ctx for reporting if available
        try:
            eq = result.get("equity_curve")
            if eq is not None:
                if engine_ctx is None:
                    engine_ctx = {}
                engine_ctx["_equity_curve"] = _sanitize_jsonable(eq)
                # Re-sanitize engine_ctx to ensure no NaN/Inf slipped in
                engine_ctx = _sanitize_jsonable(engine_ctx)
        except Exception:
            pass

        start_at = None
        end_at = None
        # Try to infer start/end from engine context or trades/data if present
        if "start_date" in result and "end_date" in result:
            try:
                import dateutil.parser as _parser  # type: ignore[import-not-found]

                start_at = _parser.parse(result["start_date"])
                end_at = _parser.parse(result["end_date"])
            except Exception:
                start_at = None
                end_at = None

        if existing:
            # Update existing row (idempotent upsert behavior)
            existing.metrics = metrics
            existing.engine_ctx = engine_ctx
            existing.trades_raw = trades_raw
            existing.error = result.get("error")
            if start_at is not None:
                existing.start_at_utc = start_at
            if end_at is not None:
                existing.end_at_utc = end_at
            sess.add(existing)
            sess.flush()
            result_id = existing.result_id
        else:
            br = unified_models.BacktestResult(
                run_id=run_id,
                symbol=result.get("symbol"),
                strategy=result.get("strategy"),
                interval=result.get("timeframe"),
                start_at_utc=start_at,
                end_at_utc=end_at,
                rank_in_symbol=None,
                metrics=metrics,
                engine_ctx=engine_ctx,
                trades_raw=trades_raw,
                error=result.get("error"),
            )
            sess.add(br)
            sess.flush()
            result_id = br.result_id

        # Persist trades normalized rows if possible
        if trades_raw:
            try:
                # Ensure new optional columns exist (best-effort, safe if already present)
                try:
                    unified_models.create_tables()
                except Exception:
                    pass
                parsed_trades = parse_trades_from_string(trades_raw)
                # Cleanup existing trades for this result (to keep idempotent)
                sess.query(unified_models.Trade).filter(
                    unified_models.Trade.result_id == result_id
                ).delete()
                for t in parsed_trades:
                    # Try to parse entry/exit timestamps if available
                    def _parse_dt(val):
                        try:
                            if val is None:
                                return None
                            import dateutil.parser as _parser  # type: ignore[import-not-found]

                            return _parser.parse(str(val))
                        except Exception:
                            return None

                    tr = unified_models.Trade(
                        result_id=result_id,
                        trade_index=int(t.get("trade_index", 0)),
                        entry_time=_parse_dt(
                            t.get("entry_time")
                            or t.get("EntryTime")
                            or t.get("entry time")
                        ),
                        exit_time=_parse_dt(
                            t.get("exit_time")
                            or t.get("ExitTime")
                            or t.get("exit time")
                        ),
                        size=str(t.get("size")) if t.get("size") is not None else None,
                        entry_bar=int(t.get("entry_bar"))
                        if t.get("entry_bar") is not None
                        else None,
                        exit_bar=int(t.get("exit_bar"))
                        if t.get("exit_bar") is not None
                        else None,
                        entry_price=str(t.get("entry_price"))
                        if t.get("entry_price") is not None
                        else None,
                        exit_price=str(t.get("exit_price"))
                        if t.get("exit_price") is not None
                        else None,
                        pnl=str(t.get("pnl")) if t.get("pnl") is not None else None,
                        duration=str(t.get("duration"))
                        if t.get("duration") is not None
                        else None,
                        tag=str(t.get("tag")) if t.get("tag") is not None else None,
                        entry_signals=str(t.get("entry_signals"))
                        if t.get("entry_signals") is not None
                        else None,
                        exit_signals=str(t.get("exit_signals"))
                        if t.get("exit_signals") is not None
                        else None,
                    )
                    sess.add(tr)
                sess.flush()
            except Exception:
                logging.getLogger(__name__).exception(
                    "Failed to persist trades for result %s", result.get("symbol")
                )

        sess.commit()
    except Exception:
        sess.rollback()
        logging.getLogger(__name__).exception(
            "Failed to persist backtest result for %s", result.get("symbol")
        )
    finally:
        sess.close()


def finalize_persistence_for_run(run_id: str, target_metric: Optional[str]) -> None:
    """
    Finalize DB persistence for a run: compute per-symbol ranks by target metric,
    upsert SymbolAggregate summaries and canonical BestStrategy rows.

    This is a best-effort helper and will log/continue on failures.
    """
    if not run_id or not target_metric:
        logging.getLogger(__name__).debug(
            "finalize_persistence_for_run skipped (missing run_id or target_metric)"
        )
        return

    def _is_higher_better(metric_name: str) -> bool:
        mn = (metric_name or "").lower()
        if "drawdown" in mn or "max_drawdown" in mn or "mdd" in mn:
            return False
        return True

    sess = None
    try:
        from src.database import unified_models  # type: ignore[import-not-found]

        sess = unified_models.Session()

        # Get distinct symbols for run
        symbols = (
            sess.query(unified_models.BacktestResult.symbol)
            .filter(unified_models.BacktestResult.run_id == run_id)
            .distinct()
            .all()
        )
        symbols = [s[0] for s in symbols]

        for symbol in symbols:
            rows = (
                sess.query(unified_models.BacktestResult)
                .filter(
                    unified_models.BacktestResult.run_id == run_id,
                    unified_models.BacktestResult.symbol == symbol,
                )
                .all()
            )

            entries = []
            higher_better = _is_higher_better(target_metric)
            for r in rows:
                mval = None
                try:
                    if r.metrics and isinstance(r.metrics, dict):
                        raw = r.metrics.get(target_metric)
                        mval = None if raw is None else float(raw)
                except Exception as exc:
                    logging.getLogger(__name__).debug(
                        "Failed to parse metric %s: %s", target_metric, exc
                    )
                sort_key = (
                    (float("-inf") if higher_better else float("inf"))
                    if mval is None
                    else mval
                )
                entries.append((sort_key, mval is None, r))

            entries.sort(key=lambda x: x[0], reverse=higher_better)

            for idx, (_sk, _is_null, row) in enumerate(entries):
                row.rank_in_symbol = idx + 1
                sess.add(row)

            if entries:
                best_row = entries[0][2]
                topn = []
                for e in entries[:3]:
                    r = e[2]
                    topn.append(
                        {
                            "strategy": r.strategy,
                            "interval": r.interval,
                            "rank": r.rank_in_symbol,
                            "metric": None
                            if r.metrics is None
                            else r.metrics.get(target_metric),
                        }
                    )
                # Upsert SymbolAggregate
                existing_agg = (
                    sess.query(unified_models.SymbolAggregate)
                    .filter(
                        unified_models.SymbolAggregate.run_id == run_id,
                        unified_models.SymbolAggregate.symbol == symbol,
                        unified_models.SymbolAggregate.best_by == target_metric,
                    )
                    .one_or_none()
                )
                summary = {"top": topn}
                if existing_agg:
                    existing_agg.best_result = best_row.result_id
                    existing_agg.summary = summary
                    sess.add(existing_agg)
                else:
                    agg = unified_models.SymbolAggregate(
                        run_id=run_id,
                        symbol=symbol,
                        best_by=target_metric,
                        best_result=best_row.result_id,
                        summary=summary,
                    )
                    sess.add(agg)

                # Upsert BestStrategy
                try:
                    bs_existing = (
                        sess.query(unified_models.BestStrategy)
                        .filter(
                            unified_models.BestStrategy.symbol == symbol,
                            unified_models.BestStrategy.timeframe == best_row.interval,
                        )
                        .one_or_none()
                    )

                    def _num(mdict, key):
                        try:
                            if mdict and isinstance(mdict, dict):
                                v = mdict.get(key)
                                return float(v) if v is not None else None
                        except Exception:
                            return None
                        return None

                    sortino_val = _num(best_row.metrics, "sortino_ratio") or _num(
                        best_row.metrics, "Sortino_Ratio"
                    )
                    calmar_val = _num(best_row.metrics, "calmar_ratio") or _num(
                        best_row.metrics, "Calmar_Ratio"
                    )
                    sharpe_val = _num(best_row.metrics, "sharpe_ratio") or _num(
                        best_row.metrics, "Sharpe_Ratio"
                    )
                    total_return_val = _num(best_row.metrics, "total_return") or _num(
                        best_row.metrics, "Total_Return"
                    )
                    max_dd_val = _num(best_row.metrics, "max_drawdown") or _num(
                        best_row.metrics, "Max_Drawdown"
                    )

                    from datetime import datetime as _dt

                    if bs_existing:
                        bs_existing.strategy = best_row.strategy
                        bs_existing.sortino_ratio = sortino_val
                        bs_existing.calmar_ratio = calmar_val
                        bs_existing.sharpe_ratio = sharpe_val
                        bs_existing.total_return = total_return_val
                        bs_existing.max_drawdown = max_dd_val
                        bs_existing.backtest_result_id = getattr(
                            best_row, "result_id", None
                        )
                        bs_existing.updated_at = _dt.utcnow()
                        sess.add(bs_existing)
                    else:
                        bs = unified_models.BestStrategy(
                            symbol=symbol,
                            timeframe=best_row.interval,
                            strategy=best_row.strategy,
                            sortino_ratio=sortino_val,
                            calmar_ratio=calmar_val,
                            sharpe_ratio=sharpe_val,
                            total_return=total_return_val,
                            max_drawdown=max_dd_val,
                            backtest_result_id=getattr(best_row, "result_id", None),
                            updated_at=_dt.utcnow(),
                        )
                        sess.add(bs)
                except Exception:
                    logging.getLogger(__name__).exception(
                        "Failed to upsert BestStrategy for %s", symbol
                    )

        sess.commit()
    except Exception:
        try:
            if sess:
                sess.rollback()
        except Exception:
            pass
        logging.getLogger(__name__).exception(
            "Failed to finalize ranks/aggregates for run %s", run_id
        )
    finally:
        try:
            if sess:
                sess.close()
        except Exception:
            pass


def run_direct_backtest(
    symbol: str,
    strategy_name: str,
    start_date: str,
    end_date: str,
    timeframe: str = "1d",
    initial_capital: float = 10000.0,
    commission: float = 0.001,
    period: Optional[str] = None,
    use_cache: bool = True,
    persistence_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run backtest using backtesting library directly.
    Returns ground truth results without wrapper complexity.

    If persistence_context is provided (dict), the function will attempt to persist
    the result into the DB via src.database.unified_models.
    """
    logger = logging.getLogger(__name__)

    try:
        # Get data
        data_manager = UnifiedDataManager()
        # If 'period' is provided, data sources like Yahoo will prefer it over start/end.
        data = data_manager.get_data(
            symbol,
            start_date,
            end_date,
            timeframe,
            use_cache=use_cache,
            period=period,
            period_mode=period,
        )

        if data is None or data.empty:
            res = {
                "symbol": symbol,
                "strategy": strategy_name,
                "timeframe": timeframe,
                "error": "No data available",
                "metrics": {},
                "trades": None,
                "backtest_object": None,
            }
            # Attempt to persist even no-data case
            if persistence_context:
                _persist_result_to_db(res, persistence_context)
            return res

        # Prepare data for backtesting library
        bt_data = data.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )[["Open", "High", "Low", "Close", "Volume"]]

        # Convert to UTC then remove timezone for backtesting library compatibility
        if bt_data.index.tz is None:
            bt_data.index = bt_data.index.tz_localize("UTC")
        else:
            bt_data.index = bt_data.index.tz_convert("UTC")
        bt_data.index = bt_data.index.tz_localize(None)

        # Create strategy
        strategy = StrategyFactory.create_strategy(strategy_name)
        StrategyClass = create_backtesting_strategy_adapter(strategy)

        # Run backtest with backtesting library
        bt = Backtest(
            bt_data,
            StrategyClass,
            cash=initial_capital,
            commission=commission,
            finalize_trades=True,  # Ensure all trades are captured
        )

        # Run and keep native stats object from backtesting library
        result = bt.run()

        # Extract trades if available
        trades = None
        if hasattr(result, "_trades") and not result._trades.empty:
            trades = result._trades.copy()

        # Extract equity curve if available
        equity_curve = None
        try:
            if hasattr(result, "_equity_curve") and result._equity_curve is not None:
                equity_curve = result._equity_curve.copy()
        except Exception:
            equity_curve = None

        ret = {
            "symbol": symbol,
            "strategy": strategy_name,
            "timeframe": timeframe,
            "error": None,
            # Do not extract custom metrics; return native stats instead
            "metrics": None,
            "trades": trades,
            "equity_curve": equity_curve,
            "backtest_object": bt,  # Include for plotting
            "bt_results": result,  # Native stats/series from backtesting library
            "start_date": start_date,
            "end_date": end_date,
        }

        # Persist if requested
        if persistence_context:
            try:
                _persist_result_to_db(ret, persistence_context)
            except Exception:
                logger.exception(
                    "Failed to persist result for %s/%s", symbol, strategy_name
                )

        return ret

    except Exception as e:
        logger.error("Direct backtest failed for %s/%s: %s", symbol, strategy_name, e)
        res = {
            "symbol": symbol,
            "strategy": strategy_name,
            "timeframe": timeframe,
            "error": str(e),
            "metrics": {},
            "trades": None,
            "backtest_object": None,
        }
        if persistence_context:
            _persist_result_to_db(res, persistence_context)
        return res


def run_strategy_comparison(
    symbol: str,
    strategies: List[str],
    start_date: str,
    end_date: str,
    timeframe: str = "1d",
    initial_capital: float = 10000.0,
    persistence_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compare multiple strategies for a symbol using backtesting library.
    Returns complete analysis with rankings and plot data.

    If persistence_context is provided, each individual strategy result will be persisted.
    """
    logger = logging.getLogger(__name__)
    logger.info(
        "Running strategy comparison for %s: %d strategies", symbol, len(strategies)
    )

    results = []
    best_result = None
    best_sortino = -999.0

    for strategy_name in strategies:
        result = run_direct_backtest(
            symbol,
            strategy_name,
            start_date,
            end_date,
            timeframe,
            initial_capital,
            persistence_context=persistence_context,
        )

        results.append(result)

        # Track best strategy by Sortino: prefer native bt_results, fallback to metrics['sortino_ratio']
        if not result["error"]:
            try:
                native = result.get("bt_results") or {}
                sortino = native.get("Sortino Ratio", None)
                if sortino is None:
                    # Fallback to normalized metrics key when native field absent
                    sortino = (result.get("metrics") or {}).get("sortino_ratio")
                sortino_val = float("nan") if sortino is None else float(sortino)
            except Exception as exc:
                logging.getLogger(__name__).debug("Failed to parse Sortino: %s", exc)
                sortino_val = float("nan")

            # Treat NaN as very poor
            if sortino_val == sortino_val and sortino_val > best_sortino:
                best_sortino = sortino_val
                best_result = result

    # Sort by native Sortino Ratio
    def _sort_key(res: Dict[str, Any]) -> float:
        try:
            native = res.get("bt_results") or {}
            v = native.get("Sortino Ratio", None)
            if v is None:
                v = (res.get("metrics") or {}).get("sortino_ratio")
            val = float(v) if v is not None else float("nan")
            # push NaN to the end by returning -inf when NaN
            return val if val == val else float("-inf")
        except Exception:
            return float("-inf")

    results.sort(key=_sort_key, reverse=True)

    # Add rankings
    for i, result in enumerate(results):
        result["rank"] = i + 1

    out = {
        "symbol": symbol,
        "timeframe": timeframe,
        "results": results,
        "best_strategy": best_result,
        "total_strategies": len(strategies),
        "successful_strategies": len(
            [
                r
                for r in results
                if not r["error"]
                and (lambda _n: (float(_n) if _n is not None else 0.0) > 0.0)(
                    (r.get("bt_results") or {}).get("# Trades", None)
                )
            ]
        ),
        "date_range": f"{start_date} to {end_date}",
    }

    # If persistence context contains a run_id and target_metric, finalize ranking/aggregates
    try:
        run_id = persistence_context.get("run_id") if persistence_context else None
        target_metric = (
            persistence_context.get("target_metric") if persistence_context else None
        )
        finalize_persistence_for_run(run_id, target_metric)
    except Exception:
        logging.getLogger(__name__).debug(
            "No persistence_context provided or failed to finalize ranks/aggregates"
        )

    # Safety net: directly upsert BestStrategy from in-memory best_result when possible.
    # This covers environments where DB state wasn't fully populated yet by finalize.
    try:
        if persistence_context and best_result and best_result.get("strategy"):
            from src.database import unified_models  # type: ignore[import-not-found]

            sess = unified_models.Session()
            try:
                bs_existing = (
                    sess.query(unified_models.BestStrategy)
                    .filter(
                        unified_models.BestStrategy.symbol == symbol,
                        unified_models.BestStrategy.timeframe == timeframe,
                    )
                    .one_or_none()
                )

                m = best_result.get("metrics") or {}

                def _num(d, k):
                    try:
                        if d and isinstance(d, dict):
                            v = d.get(k)
                            return float(v) if v is not None else None
                    except Exception:
                        return None
                    return None

                sortino_val = _num(m, "sortino_ratio") or _num(m, "Sortino_Ratio")
                calmar_val = _num(m, "calmar_ratio") or _num(m, "Calmar_Ratio")
                sharpe_val = _num(m, "sharpe_ratio") or _num(m, "Sharpe_Ratio")
                total_return_val = _num(m, "total_return") or _num(m, "Total_Return")
                max_dd_val = _num(m, "max_drawdown") or _num(m, "Max_Drawdown")

                from datetime import datetime as _dt

                if bs_existing:
                    bs_existing.strategy = best_result.get("strategy")
                    bs_existing.sortino_ratio = sortino_val
                    bs_existing.calmar_ratio = calmar_val
                    bs_existing.sharpe_ratio = sharpe_val
                    bs_existing.total_return = total_return_val
                    bs_existing.max_drawdown = max_dd_val
                    bs_existing.updated_at = _dt.utcnow()
                    sess.add(bs_existing)
                else:
                    bs = unified_models.BestStrategy(
                        symbol=symbol,
                        timeframe=timeframe,
                        strategy=best_result.get("strategy"),
                        sortino_ratio=sortino_val,
                        calmar_ratio=calmar_val,
                        sharpe_ratio=sharpe_val,
                        total_return=total_return_val,
                        max_drawdown=max_dd_val,
                        updated_at=_dt.utcnow(),
                    )
                    sess.add(bs)
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
    except Exception:
        logging.getLogger(__name__).debug("BestStrategy safety upsert skipped")

    return out
