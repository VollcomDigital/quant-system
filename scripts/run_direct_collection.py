#!/usr/bin/env python3
"""
Run direct backtests for a portfolio JSON across multiple intervals using the
backtesting library as the single source of truth.

This version persists all results to the project database (BestStrategy,
BacktestResult, Trades, etc.) so downstream reports and comparisons use the DB
instead of JSON. For convenience and offline inspection it still writes a
summary JSON to `exports/direct_portfolio_comparison.json`, but that file is
not intended to be a data source.

Usage:
  python3 scripts/run_direct_collection.py \
    --portfolio config/collections/bonds.json \
    --intervals "1m 5m 15m 1h 4h 1d 1wk" \
    [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] [--period max|1y|...]

Requirements:
  - The database must be reachable (e.g., via docker-compose). The script will
    attempt to create a Run row and then persist each backtest result under that
    run_id. At the end it finalizes rankings and upserts BestStrategy.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from src.core.direct_backtest import finalize_persistence_for_run, run_direct_backtest
from src.core.external_strategy_loader import get_strategy_loader
from src.core.strategy import StrategyFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("direct-portfolio")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--portfolio", required=True, help="Path to portfolio JSON")
    p.add_argument(
        "--intervals",
        required=True,
        help="Space-separated list of intervals (e.g. '1min 5min 15min 1h 4h 1d 1wk')",
    )
    p.add_argument("--start-date", help="Optional start date (YYYY-MM-DD)")
    p.add_argument("--end-date", help="Optional end date (YYYY-MM-DD)")
    p.add_argument(
        "--period",
        default=None,
        help="Optional provider period token (e.g. 'max', '1y'). When set, overrides start/end",
    )
    p.add_argument("--initial-capital", type=float, default=10000.0)
    return p.parse_args()


def main():
    args = parse_args()
    portfolio_path = Path(args.portfolio)
    if not portfolio_path.exists():
        logger.error("Portfolio file not found: %s", portfolio_path)
        return

    with portfolio_path.open() as f:
        portfolio_data = json.load(f)

    # Get first portfolio
    portfolio_name = list(portfolio_data.keys())[0]
    portfolio = portfolio_data[portfolio_name]
    symbols = portfolio.get("symbols", [])
    if not symbols:
        logger.error("No symbols found in portfolio")
        return

    # Ensure external strategies loader is initialized.
    # Prefer using the host-side quant-strategies path if it exists; otherwise
    # fall back to the default loader behavior which prefers the container-mounted
    # external_strategies directory.
    project_root = Path(__file__).resolve().parent.parent
    quant_path = project_root / "quant-strategies" / "algorithms" / "python"
    try:
        if quant_path.exists():
            # Host-side quant-strategies available (development); point loader there.
            get_strategy_loader(str(quant_path))
        else:
            # No host mount for quant-strategies in container — let loader pick defaults
            # (it will prefer /app/external_strategies if present).
            get_strategy_loader()
    except Exception:
        logger.warning("Could not initialize external strategy loader")

    # Determine strategies to test (prefer all available)
    all_strats = StrategyFactory.list_strategies().get("all", [])
    strategies = all_strats if all_strats else ["rsi", "macd", "bollinger_bands"]

    intervals = args.intervals.split()
    start_date = args.start_date or "1970-01-01"
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    period = args.period  # if provided, data manager will prefer this over start/end
    # Prefer initial capital from portfolio config when present
    initial_capital = portfolio.get("initial_capital", args.initial_capital)

    results = {}

    # Prepare a minimal manifest/run so that persistence_context can associate all
    # backtests under a single run_id. We keep this local to avoid importing the
    # full unified CLI; the DB helper provides a simple create_run_from_manifest.
    run_id = None
    target_metric = "sortino_ratio"
    try:
        import hashlib

        from src.database import unified_models as um  # type: ignore[import-not-found]

        # Minimal plan for hashing + traceability
        plan = {
            "action": "direct",
            "symbols": symbols,
            "strategies": strategies,
            "intervals": intervals,
            "period_mode": str(period or "max"),
            "start": start_date,
            "end": end_date,
            "initial_capital": float(initial_capital),
            "commission": float(portfolio.get("commission", 0.001)),
            "metric": target_metric,
        }
        plan_hash = hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        manifest = {
            "plan": {**plan, "plan_hash": plan_hash},
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

        # Ensure tables exist (best-effort; safe if already created)
        try:
            um.create_tables()
        except Exception:
            pass

        run_obj = um.create_run_from_manifest(manifest)
        run_id = getattr(run_obj, "run_id", None)
    except Exception:
        logger.warning(
            "Database persistence unavailable; continuing without DB (JSON will still be written)"
        )

    def _sanitize_jsonable(obj):
        """Best-effort conversion of stats to JSON-safe primitives."""
        try:
            import math

            import numpy as _np  # type: ignore[import-not-found]
            import pandas as _pd  # type: ignore[import-not-found]
        except Exception:
            math = None
            _np = None
            _pd = None

        # Pandas Series/DataFrame
        try:
            if _pd is not None and isinstance(obj, _pd.Series):
                return {k: _sanitize_jsonable(v) for k, v in obj.to_dict().items()}
            if _pd is not None and isinstance(obj, _pd.DataFrame):
                return obj.to_dict(orient="records")
        except Exception:
            pass

        # Pandas Timestamp / datetime-like
        try:
            import datetime as _dt  # type: ignore[import-not-found]

            if _pd is not None and isinstance(obj, _pd.Timestamp):
                return obj.isoformat()
            if isinstance(obj, (_dt.datetime, _dt.date)):
                return obj.isoformat()
        except Exception:
            pass

        # Numpy scalars/arrays
        if _np is not None and isinstance(obj, _np.generic):
            try:
                return _sanitize_jsonable(obj.item())
            except Exception:
                pass
        if _np is not None and isinstance(obj, _np.ndarray):
            try:
                return [_sanitize_jsonable(v) for v in obj.tolist()]
            except Exception:
                pass

        # Primitives
        if obj is None or isinstance(obj, (str, bool, int)):
            return obj
        if isinstance(obj, float):
            try:
                if math and (math.isnan(obj) or math.isinf(obj)):
                    return None
            except Exception:
                return None
            return obj

        # Collections
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                try:
                    out[str(k)] = _sanitize_jsonable(v)
                except Exception:
                    out[str(k)] = None
            return out
        if isinstance(obj, (list, tuple)):
            return [_sanitize_jsonable(v) for v in obj]

        # Fallback
        try:
            return str(obj)
        except Exception:
            return None

    total = len(symbols) * len(strategies) * len(intervals)
    counter = 0

    for interval in intervals:
        for symbol in symbols:
            for strategy in strategies:
                counter += 1
                logger.info(
                    "[%d/%d] Running direct backtest %s %s @ %s",
                    counter,
                    total,
                    symbol,
                    strategy,
                    interval,
                )
                try:
                    # If DB is available, pass persistence_context so direct_backtest
                    # will persist BacktestResult and Trades to the database.
                    persistence_context = (
                        {"run_id": run_id, "target_metric": target_metric}
                        if run_id
                        else None
                    )
                    result = run_direct_backtest(
                        symbol=symbol,
                        strategy_name=strategy,
                        start_date=start_date,
                        end_date=end_date,
                        timeframe=interval,
                        initial_capital=initial_capital,
                        commission=portfolio.get("commission", 0.001),
                        period=period,
                        persistence_context=persistence_context,
                    )
                    # Collect native stats from backtesting library
                    stats = _sanitize_jsonable(result.get("bt_results"))
                    results.setdefault(symbol, {}).setdefault(interval, []).append(
                        {
                            "strategy": strategy,
                            "stats": stats,
                            "error": result.get("error"),
                        }
                    )
                except Exception as e:
                    logger.error(
                        "Direct backtest failed for %s %s %s: %s",
                        symbol,
                        strategy,
                        interval,
                        e,
                    )
                    results.setdefault(symbol, {}).setdefault(interval, []).append(
                        {"strategy": strategy, "stats": {}, "error": str(e)}
                    )

    # Finalize DB ranks/best strategies once all results are persisted.
    try:
        if run_id:
            finalize_persistence_for_run(run_id, target_metric)
            logger.info(
                "Finalized DB aggregates for run_id=%s (metric=%s)",
                run_id,
                target_metric,
            )
    except Exception:
        logger.exception("Failed to finalize DB aggregates for run_id=%s", run_id)

    output_file = Path("exports") / "direct_portfolio_comparison.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Saved direct portfolio stats to %s", output_file)

    # Optionally generate a DB-backed HTML report for this portfolio using the same reporter
    try:
        from src.reporting.collection_report import DetailedPortfolioReporter

        portfolio_name = portfolio.get("name") or portfolio_path.stem
        reporter = DetailedPortfolioReporter()
        report_path = reporter.generate_comprehensive_report(
            {"name": portfolio_name, "symbols": symbols},
            start_date=start_date,
            end_date=end_date,
            strategies=["best"],
            timeframes=intervals,
        )
        logger.info("Generated HTML report (DB-backed) at %s", report_path)
    except Exception as e:
        logger.warning("Could not generate HTML report: %s", e)


if __name__ == "__main__":
    main()
