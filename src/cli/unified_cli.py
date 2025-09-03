#!/usr/bin/env python3
"""
Unified CLI entrypoint: src.cli.unified_cli

This module implements the `collection` subcommand which builds a deterministic
plan (plan_hash), writes a manifest, supports --dry-run, and delegates work to
the project's backtest engine and DB/persistence layers if available.

This is intentionally conservative: it validates inputs, expands strategies and
intervals where possible, and provides clear hooks for the engine and DB code.
All optional integrations are guarded to avoid import-time failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Constants
DEFAULT_METRIC = "sortino_ratio"
SUPPORTED_INTERVALS = ["1m", "5m", "15m", "1h", "1d", "1wk", "1mo", "3mo"]
INTRADAY_MAX_DAYS = 60
ONE_MINUTE_MAX_DAYS = 7

log = logging.getLogger("unified_cli")


# Install a global excepthook that will log uncaught exceptions with a full traceback.
# This is useful when running inside Docker where stderr/telnet output may be suppressed.
def _unified_excepthook(exc_type, exc_value, tb):
    import traceback as _traceback

    try:
        log.exception("Uncaught exception", exc_info=(exc_type, exc_value, tb))
    except Exception:
        # If logging fails for any reason, still print the traceback to stderr.
        _traceback.print_exception(exc_type, exc_value, tb)
    else:
        _traceback.print_exception(exc_type, exc_value, tb)


import sys as _sys

_sys.excepthook = _unified_excepthook


def _setup_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format="%(asctime)s %(levelname)s %(message)s")


def resolve_collection_path(collection_arg: str) -> Path:
    p = Path(collection_arg)
    if p.exists():
        return p.resolve()
    # try config/collections/<collection_arg>.json
    base = Path("config") / "collections"
    # Aliases for curated defaults
    alias_map = {
        # Curated defaults
        "bonds": "bonds_core",
        "commodities": "commodities_core",
        "crypto": "crypto_liquid",
        "forex": "forex_majors",
        "indices": "indices_global_core",
        # Convenience aliases
        "tech_growth": "stocks_us_growth_core",
        "us_mega": "stocks_us_mega_core",
        "value": "stocks_us_value_core",
        "quality": "stocks_us_quality_core",
        "minvol": "stocks_us_minvol_core",
        "global_factors": "stocks_global_factor_core",
    }
    key = alias_map.get(collection_arg, collection_arg)
    candidates = [
        base / f"{key}.json",
        base / "default" / f"{key}.json",
        base / "custom" / f"{key}.json",
    ]
    for alt in candidates:
        if alt.exists():
            return alt.resolve()
    raise FileNotFoundError(f"Collection file not found: {collection_arg}")


def compute_plan_hash(plan: Dict[str, Any]) -> str:
    # Deterministic serialization: sort keys
    payload = json.dumps(
        plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_collection_symbols(collection_path: Path) -> List[str]:
    """
    Load symbols from a collection JSON.

    Supported formats:
      - Plain list: ["AAPL", "MSFT", ...]
      - Dict with top-level "symbols" (or "assets"/"symbols_list"):
          {"symbols": ["AAPL", ...], ...}
      - Named collection object (common in config/collections/*.json):
          {"bonds": {"symbols": [...], "name": "...", ...}}
      - Dict of multiple named collections: returns symbols for the first matching
        collection that contains a 'symbols' list (best-effort).
    """
    try:
        with collection_path.open() as f:
            data = json.load(f)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read collection file {collection_path}: {exc}"
        ) from exc

    # If the file itself is a plain list of symbols
    if isinstance(data, list):
        return [str(s).upper() for s in data]

    # If the file is a dict, try common keys first
    if isinstance(data, dict):
        # Direct keys that point to a symbols list
        for key in ("symbols", "assets", "symbols_list"):
            if key in data and isinstance(data[key], list):
                return [str(s).upper() for s in data[key]]

        # If the file wraps one or more named collections (e.g., {"bonds": {...}})
        # find the first value that itself contains a 'symbols' list
        for val in data.values():
            if isinstance(val, dict):
                for key in ("symbols", "assets", "symbols_list"):
                    if key in val and isinstance(val[key], list):
                        return [str(s).upper() for s in val[key]]

    raise RuntimeError(
        f"Collection JSON at {collection_path} missing 'symbols' list or unsupported format"
    )


def expand_strategies(strategies_arg: str) -> List[str]:
    # strategies_arg can be comma-separated or 'all'
    parts = [p.strip() for p in strategies_arg.split(",") if p.strip()]
    if len(parts) == 1 and parts[0].lower() == "all":
        # Prefer explicit environment variable or container-mounted path when running inside Docker.
        # This avoids trying to read host paths from within the container.
        try:
            import os

            candidates = []
            env_path = os.getenv("STRATEGIES_PATH")
            if env_path:
                candidates.append(env_path)

            # Common container mount used in docker-compose
            candidates.append("/app/external_strategies")

            # Host-local fallback (works when running on host)
            candidates.append(str(Path("quant-strategies").resolve()))
            candidates.append(str(Path("external_strategies").resolve()))

            from src.core.external_strategy_loader import get_strategy_loader
            from src.core.strategy import StrategyFactory

            strategies = []
            for cand in candidates:
                try:
                    if not cand:
                        continue
                    p = Path(cand)
                    if not p.exists():
                        continue
                    loader = get_strategy_loader(str(cand))
                    try:
                        strategies = StrategyFactory.list_strategies(loader=loader)
                        if isinstance(strategies, dict):
                            strategies = (
                                strategies.get("all")
                                or strategies.get("external")
                                or []
                            )
                    except Exception:
                        strategies = []
                    # If we found any, return them (deduplicated & sorted)
                    if strategies:
                        return sorted(set(strategies))
                    # If loader supports listing candidates without importing, try that
                    try:
                        candidates_list = loader.list_strategy_candidates()
                        if candidates_list:
                            return sorted(set(candidates_list))
                    except Exception:
                        pass
                except Exception as exc:
                    # try next candidate, but log for diagnostics
                    log.debug("Strategy discovery failed for %s: %s", cand, exc)
                    continue

            # Last fallback: try the local algorithms/python dir if present
            alt_dir = Path("quant-strategies") / "algorithms" / "python"
            if alt_dir.exists():
                cand = [p.stem for p in alt_dir.glob("*.py") if p.is_file()]
                if cand:
                    return sorted(set(cand))

            # If nothing found, proceed with an empty list (safe default for dry-run/tests)
            log.warning(
                "Could not expand 'all' strategies: no strategy repository found; proceeding with none"
            )
            return []
        except Exception as exc:
            log.warning(
                "Could not expand 'all' strategies: %s; proceeding with none", exc
            )
            return []

    # explicit list
    expanded: List[str] = []
    for part in parts:
        expanded.extend([s.strip() for s in part.split("+") if s.strip()])
    return sorted(set(expanded))


def expand_intervals(interval_arg: str) -> List[str]:
    parts = [p.strip() for p in interval_arg.split(",") if p.strip()]
    if len(parts) == 1 and parts[0].lower() == "all":
        return SUPPORTED_INTERVALS.copy()
    # validate
    invalid = [p for p in parts if p not in SUPPORTED_INTERVALS]
    if invalid:
        raise RuntimeError(
            f"Unknown intervals requested: {invalid}. Supported: {SUPPORTED_INTERVALS}"
        )
    return parts


def clamp_interval_period(
    interval: str, start: Optional[str], end: Optional[str], period_mode: str
) -> Dict[str, Optional[str]]:
    """
    Enforce provider constraints:
      - 1m allowed only for last ONE_MINUTE_MAX_DAYS days
      - <1d intraday intervals allowed only for last INTRADAY_MAX_DAYS days
    Returns dict with possibly modified 'start'/'end' and 'period_mode' (may remain 'max')
    """
    # This function returns the passed args unchanged by default. Real clamping requires querying provider
    # for available date ranges; here we provide warnings and leave exact clamping to data manager.
    if interval == "1m":
        # warn user if period_mode == 'max'
        if period_mode == "max":
            log.warning(
                "Interval '1m' may be limited to the last %d days by the data provider",
                ONE_MINUTE_MAX_DAYS,
            )
    elif interval in ("5m", "15m", "1h"):
        if period_mode == "max":
            log.warning(
                "Intraday interval '%s' may be limited to the last %d days by the data provider",
                interval,
                INTRADAY_MAX_DAYS,
            )
    return {"start": start, "end": end, "period_mode": period_mode}


def write_manifest(outdir: Path, manifest: Dict[str, Any]) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, ensure_ascii=False)
    return manifest_path


def try_get_git_sha(path: Path) -> Optional[str]:
    # Try to read git sha for the given path if it's a git repo
    git_exe = shutil.which("git")
    if git_exe is None:
        return None
    if not (path / ".git").exists():
        return None
    try:
        import subprocess

        out = subprocess.check_output(
            [git_exe, "-C", str(path.resolve()), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def persist_run_row_placeholder(manifest: Dict[str, Any]) -> None:
    # Hook: try to persist the initial run row to DB using unified_models if available.
    try:
        from src.database import unified_models

        # unified_models should expose create_run_from_manifest(manifest) or similar.
        if hasattr(unified_models, "create_run_from_manifest"):
            unified_models.create_run_from_manifest(manifest)
            log.info(
                "Persisted run row to DB via unified_models.create_run_from_manifest"
            )
        else:
            log.debug(
                "unified_models module found but create_run_from_manifest not present"
            )
    except Exception:
        log.debug(
            "DB persistence not available (unified_models missing or failed). Continuing without DB."
        )


def run_plan(manifest: Dict[str, Any], outdir: Path, dry_run: bool = False) -> int:
    """
    Execute the resolved plan.

    This implementation delegates to src.core.direct_backtest.UnifiedBacktestEngine.run if available.
    If unavailable, it will write a placeholder summary and return 0 on success.
    """
    if dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
        return 0

    # Persist a run row (best-effort)
    persist_run_row_placeholder(manifest)

    # If action is 'direct', use the direct backtester with DB persistence
    try:
        plan_action = manifest.get("plan", {}).get("action")
    except Exception:
        plan_action = None

    if plan_action == "direct":
        try:
            from src.core.data_manager import UnifiedDataManager
            from src.core.direct_backtest import (
                finalize_persistence_for_run,
                run_direct_backtest,
            )
        except Exception:
            log.exception("Direct backtester not available")
            return 12

        plan = manifest.get("plan", {})
        symbols = plan.get("symbols", [])
        strategies = plan.get("strategies", [])
        intervals = plan.get("intervals", ["1d"])  # usually one
        period_mode = plan.get("period_mode", "max")
        start = plan.get("start") or ""
        end = plan.get("end") or ""
        initial_capital = plan.get("initial_capital", 10000)
        commission = plan.get("commission", 0.001)
        target_metric = plan.get("metric", DEFAULT_METRIC)
        plan_hash = plan.get("plan_hash")

        # Initialize external strategies loader when a path is available (container-safe)
        try:
            from src.core.external_strategy_loader import get_strategy_loader

            spath = plan.get("strategies_path")
            if spath:
                get_strategy_loader(str(spath))
        except Exception:
            # best-effort; loader may already be initialized elsewhere
            pass

        # Ensure a run row exists
        run_id = None
        try:
            from src.database import unified_models

            run_obj = None
            if hasattr(unified_models, "ensure_run_for_manifest"):
                run_obj = unified_models.ensure_run_for_manifest(manifest)
            else:
                run_obj = unified_models.create_run_from_manifest(manifest)
            run_id = getattr(run_obj, "run_id", None)
        except Exception:
            run_id = None

        persistence_context = (
            {"run_id": run_id, "target_metric": target_metric, "plan_hash": plan_hash}
            if run_id
            else None
        )

        # Optional: probe sources for best coverage and set ordering overrides
        try:
            dm_probe = UnifiedDataManager()
            # Detect asset type from the first symbol; fall back to 'stocks'
            asset_type_probe = "stocks"
            try:
                if symbols:
                    asset_type_probe = dm_probe._detect_asset_type(symbols[0])
            except Exception:
                pass
            sample_syms = symbols[: min(5, len(symbols))]
            if sample_syms:
                ordered = dm_probe.probe_and_set_order(
                    asset_type_probe,
                    sample_syms,
                    interval=intervals[0] if intervals else "1d",
                )
                if ordered:
                    log.info(
                        "Source order override for %s: %s", asset_type_probe, ordered
                    )
        except Exception:
            log.debug("Coverage probe failed; continuing with default ordering")

        for interval in intervals:
            for symbol in symbols:
                for strat in strategies:
                    try:
                        _ = run_direct_backtest(
                            symbol=symbol,
                            strategy_name=strat,
                            start_date=start,
                            end_date=end,
                            timeframe=interval,
                            initial_capital=float(initial_capital),
                            commission=float(commission),
                            period=(period_mode if period_mode else None),
                            use_cache=bool(plan.get("use_cache", True)),
                            persistence_context=persistence_context,
                        )
                    except Exception:
                        log.exception(
                            "Direct backtest failed for %s %s %s",
                            symbol,
                            strat,
                            interval,
                        )
                        continue

        # Finalize DB ranks/best strategy
        try:
            if persistence_context:
                finalize_persistence_for_run(
                    persistence_context.get("run_id"), target_metric
                )
        except Exception:
            log.exception(
                "Finalization failed for run %s",
                (persistence_context or {}).get("run_id"),
            )

        return 0

    # Delegate to engine if available (use the unified backtest engine implementation)
    try:
        from src.core.backtest_engine import UnifiedBacktestEngine

        # The Backtest Engine class expects different init args; instantiate and run batch if available.
        engine = UnifiedBacktestEngine()
        # If engine exposes a run() method accepting manifest/outdir, prefer that; otherwise, run a batch run.
        if hasattr(engine, "run"):
            try:
                res = engine.run(manifest=manifest, outdir=outdir)  # type: ignore[attr-defined]
                log.info(
                    "Engine run finished with result: %s",
                    getattr(res, "status", "unknown"),
                )
                # Best-effort: if engine returned a summary dict, persist it to the outdir
                try:
                    import json as _json  # local import

                    summary_path = Path(outdir) / "engine_run_summary.json"
                    if isinstance(res, dict):
                        try:
                            summary_path.parent.mkdir(parents=True, exist_ok=True)
                            with summary_path.open("w", encoding="utf-8") as fh:
                                _json.dump(
                                    res,
                                    fh,
                                    indent=2,
                                    sort_keys=True,
                                    ensure_ascii=False,
                                )
                            log.info("Wrote engine summary to %s", summary_path)
                        except Exception:
                            log.exception(
                                "Failed to write engine summary to %s", summary_path
                            )
                except Exception:
                    log.debug(
                        "Engine returned non-dict or failed to write summary (continuing)"
                    )
                return 0
            except Exception:
                # fall back to batch behavior below
                pass

        # Fall back: attempt to run batch backtests using run_batch_backtests if manifest is compatible
        try:
            plan = manifest.get("plan", {})
            config_kwargs = {
                "symbols": plan.get("symbols", []),
                "strategies": plan.get("strategies", []),
                "start_date": plan.get("start"),
                "end_date": plan.get("end"),
                "initial_capital": plan.get("initial_capital", 10000),
                "interval": plan.get("intervals", ["1d"])[0]
                if plan.get("intervals")
                else "1d",
                "max_workers": plan.get("max_workers", 4),
            }
            # Use BacktestConfig dataclass if available
            try:
                from src.core.backtest_engine import (
                    BacktestConfig,  # type: ignore[import-not-found]
                )

                cfg = BacktestConfig(**config_kwargs)
                results = engine.run_batch_backtests(cfg)
                log.info(
                    "Engine run_batch_backtests finished with %d results", len(results)
                )
                return 0
            except Exception:
                log.debug(
                    "Could not construct BacktestConfig; skipping engine batch run"
                )
        except Exception:
            log.debug("Engine fallback path failed")
        # If we reach here, engine couldn't be driven programmatically
        raise RuntimeError("Engine found but could not be executed with manifest")
    except Exception as exc:
        log.exception("Backtest engine not available or failed: %s", exc)

    # Fallback: write a minimal summaries JSON
    summary = {
        "manifest": manifest,
        "status": "fallback_no_engine",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    fallback_path = outdir / "run_summary_fallback.json"
    with fallback_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, ensure_ascii=False)
    log.warning("Wrote fallback summary to %s", fallback_path)
    return 0


def _run_requested_exports(
    resolved_plan: Dict[str, Any], collection_path: Path, symbols: List[str]
) -> None:
    """Run requested exports (report, csv, ai, tradingview) best-effort.

    - Avoids hard DB connectivity failures; individual exporters handle fallbacks.
    - CSV exporter falls back to unified_models or quarterly reports when needed.
    - AI recommendations fall back to unified_models when primary DB is unavailable.
    """
    exports_val = resolved_plan.get("exports", "") or ""
    try:
        exports_list = [
            e.strip().lower() for e in str(exports_val).split(",") if e.strip()
        ]
    except Exception:
        exports_list = []
    if not exports_list:
        return

    log = logging.getLogger("unified_cli")

    # Prepare portfolio context
    portfolio_name = collection_path.stem
    try:
        import json as _json

        with collection_path.open() as _fh:
            _cdata = _json.load(_fh)
        if isinstance(_cdata, dict):
            if isinstance(_cdata.get("name"), str):
                portfolio_name = _cdata.get("name") or portfolio_name
            else:
                first = next(iter(_cdata.values())) if _cdata else None
                if isinstance(first, dict) and isinstance(first.get("name"), str):
                    portfolio_name = first.get("name") or portfolio_name
    except Exception:
        pass

    portfolio_config = {"name": portfolio_name, "symbols": sorted(symbols)}

    do_report = ("report" in exports_list) or ("all" in exports_list)
    do_csv = ("csv" in exports_list) or ("all" in exports_list)
    do_tradingview = ("tradingview" in exports_list) or ("all" in exports_list)
    do_ai = ("ai" in exports_list) or ("all" in exports_list)

    # Determine quarter/year/interval context
    y_now = datetime.utcnow().year
    m = datetime.utcnow().month
    q_now = (m - 1) // 3 + 1
    quarter = f"Q{q_now}"
    year = str(y_now)
    try:
        _intervals = list(resolved_plan.get("intervals") or [])
        # Single-file export policy:
        # - Use '1d' for filenames when present, else first interval
        # - If multiple intervals were requested, do not filter by interval in exporters (pass None)
        interval_for_filename = (
            "1d" if "1d" in _intervals else (_intervals[0] if _intervals else "1d")
        )
        multiple_intervals = len(_intervals) > 1
        interval_filter = None if multiple_intervals else interval_for_filename
    except Exception:
        interval_for_filename = "1d"
        interval_filter = interval_for_filename

    # Report
    if do_report:
        try:
            from src.reporting.collection_report import DetailedPortfolioReporter

            reporter = DetailedPortfolioReporter()
            start_date = resolved_plan.get("start") or ""
            end_date = resolved_plan.get("end") or ""
            try:
                report_path = reporter.generate_comprehensive_report(
                    portfolio_config,
                    start_date or datetime.utcnow().strftime("%Y-%m-%d"),
                    end_date or datetime.utcnow().strftime("%Y-%m-%d"),
                    resolved_plan.get("strategies", []),
                    timeframes=[interval_for_filename]
                    if interval_for_filename
                    else None,
                    filename_interval=(
                        "multi"
                        if (len(resolved_plan.get("intervals") or []) > 1)
                        else interval_for_filename
                    ),
                )
            except TypeError:
                # Backward-compat: reporter without filename_interval arg
                report_path = reporter.generate_comprehensive_report(
                    portfolio_config,
                    start_date or datetime.utcnow().strftime("%Y-%m-%d"),
                    end_date or datetime.utcnow().strftime("%Y-%m-%d"),
                    resolved_plan.get("strategies", []),
                    timeframes=[interval_for_filename]
                    if interval_for_filename
                    else None,
                )
            log.info("Generated HTML report at %s", report_path)
        except Exception:
            log.exception("DetailedPortfolioReporter failed (continuing)")

    # CSV (DB-backed with fallback to unified_models or quarterly reports)
    if do_csv:
        try:
            from src.utils.csv_exporter import RawDataCSVExporter

            csv_exporter = RawDataCSVExporter()
            # Prefer calendar from plan start/end if present
            try:
                if resolved_plan.get("start"):
                    sd = datetime.fromisoformat(resolved_plan.get("start"))
                else:
                    sd = datetime.utcnow()
            except Exception:
                sd = datetime.utcnow()
            quarter = f"Q{((sd.month - 1) // 3) + 1}"
            year = str(sd.year)

            csv_files = csv_exporter.export_from_database_primary(
                quarter,
                year,
                output_filename=None,
                export_format="best-strategies",
                portfolio_name=portfolio_config.get("name") or "",
                portfolio_path=str(collection_path),
                interval=interval_filter,
            )
            if not csv_files:
                csv_files = csv_exporter.export_from_quarterly_reports(
                    quarter,
                    year,
                    export_format="best-strategies",
                    collection_name=portfolio_config.get("name"),
                    interval=interval_filter,
                )
            log.info("Generated CSV exports: %s", csv_files)
        except Exception:
            log.exception("CSV export failed (continuing)")

    # AI recommendations (DB primary with unified_models fallback inside class)
    if do_ai:
        try:
            from src.ai.investment_recommendations import AIInvestmentRecommendations
            from src.database.db_connection import get_db_session

            db_sess = None
            try:
                db_sess = get_db_session()
            except Exception:
                db_sess = None
            ai = AIInvestmentRecommendations(db_session=db_sess)
            _rec, ai_html_path = ai.generate_portfolio_recommendations(
                portfolio_config_path=str(collection_path),
                risk_tolerance="moderate",
                min_confidence=0.6,
                max_assets=10,
                quarter=f"{quarter}_{year}",
                timeframe=interval_for_filename,  # concrete for trading params
                filename_interval=(
                    "multi"
                    if (len(resolved_plan.get("intervals") or []) > 1)
                    else interval_for_filename
                ),
                generate_html=True,
            )
            log.info("Generated AI recommendations at %s", ai_html_path)
        except Exception:
            log.exception("AI recommendations export failed (continuing)")

    # TradingView alerts
    if do_tradingview:
        try:
            from src.utils.tv_alert_exporter import TradingViewAlertExporter

            tv_exporter = TradingViewAlertExporter(reports_dir="exports/reports")
            alerts = tv_exporter.export_alerts(
                output_file=None,
                collection_filter=portfolio_config.get("name"),
                interval=interval_filter,
                symbols=portfolio_config.get("symbols") or [],
            )
            log.info("Generated TradingView alerts for %d assets", len(alerts))
        except Exception:
            log.exception("TradingView alerts export failed (continuing)")


def handle_collection_run(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="unified_cli collection",
        description="Run unified backtests for a collection",
    )
    parser.add_argument(
        "collection",
        help="Path to collection JSON file or collection key under config/collections",
    )
    parser.add_argument(
        "--action",
        default="direct",
        choices=[
            "backtest",
            "direct",
            "optimization",
            "export",
            "report",
            "tradingview",
        ],
        help="Action to perform",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help=f"Primary metric used for ranking (default: {DEFAULT_METRIC})",
    )
    parser.add_argument(
        "--strategies",
        default="all",
        help="Comma-separated strategies or 'all' (default: all)",
    )
    period_group = parser.add_mutually_exclusive_group()
    period_group.add_argument(
        "--period",
        default="max",
        help="Named period token e.g. 1d, 1mo, 1y, ytd, max (default: max)",
    )
    period_group.add_argument("--start", help="ISO start date YYYY-MM-DD")
    parser.add_argument(
        "--end", help="ISO end date YYYY-MM-DD (required when --start is given)"
    )
    parser.add_argument(
        "--interval",
        default="all",
        help="Comma-separated intervals or 'all' (default: all)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass cache reads for data (fetch fresh)",
    )
    parser.add_argument(
        "--fresh", action="store_true", help="Alias for --no-cache (fetch fresh data)"
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Danger: drop and recreate DB tables before running",
    )
    parser.add_argument(
        "--exports",
        default="",
        help="Comma-separated export types to run (csv,report,tradingview,ai,all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not perform side effects; print manifest and exit",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory for artifacts (default: artifacts/run_<timestamp>)",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--config", default=None, help="Path to config file (optional)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force run even if plan_hash already succeeded",
    )
    parser.add_argument(
        "--max-workers", type=int, default=4, help="Concurrency for backtests"
    )
    args = parser.parse_args(argv)

    _setup_logging(args.log_level)

    try:
        collection_path = resolve_collection_path(args.collection)
    except Exception as exc:
        log.exception("Failed to resolve collection: %s", exc)
        return 2

    try:
        symbols = load_collection_symbols(collection_path)
    except Exception as exc:
        log.exception("Failed to load symbols from collection: %s", exc)
        return 3

    try:
        strategies = expand_strategies(args.strategies)
        # Filter out filesystem artifacts or invalid candidates (e.g., __pycache__)
        try:
            strategies = [
                s
                for s in strategies
                if not (isinstance(s, str) and s.strip().startswith("__"))
            ]
        except Exception:
            # Defensive: if filtering fails, keep original list
            pass
    except Exception as exc:
        log.exception("Failed to resolve strategies: %s", exc)
        return 4

    try:
        intervals = expand_intervals(args.interval)
    except Exception as exc:
        log.exception("Failed to resolve intervals: %s", exc)
        return 5

    # Basic validation for start/end
    if args.start and not args.end:
        log.error("--end is required when --start is provided")
        return 6

    # compute outdir
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) if args.outdir else Path("artifacts") / f"run_{ts}"
    outdir = outdir.resolve()

    # Collect git SHAs (best-effort)
    app_sha = try_get_git_sha(Path())
    strat_sha = try_get_git_sha(Path("quant-strategies"))

    # Build plan manifest
    resolved_plan = {
        "actor": "cli",
        "action": args.action,
        "collection": str(collection_path),
        "symbols": sorted(symbols),
        "strategies": sorted(strategies),
        "intervals": sorted(intervals),
        "metric": args.metric,
        "period_mode": args.period if args.start is None else "start_end",
        "start": args.start,
        "end": args.end,
        "exports": args.exports,
        "dry_run": bool(args.dry_run),
        "use_cache": not (args.no_cache or args.fresh),
        "max_workers": int(args.max_workers),
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "git_sha_app": app_sha,
        "git_sha_strat": strat_sha,
    }

    # Try to read initial_capital and commission from the collection file
    try:
        import json as _json

        with collection_path.open() as _fh:
            _data = _json.load(_fh)
        # Direct keys
        ic = None
        comm = None
        if isinstance(_data, dict):
            if "initial_capital" in _data:
                ic = _data.get("initial_capital")
            if "commission" in _data:
                comm = _data.get("commission")
            # Named collection wrapper
            if (ic is None or comm is None) and _data:
                try:
                    first = next(iter(_data.values()))
                    if isinstance(first, dict):
                        ic = first.get("initial_capital", ic)
                        comm = first.get("commission", comm)
                except Exception:
                    pass
        if ic is not None:
            resolved_plan["initial_capital"] = float(ic)
        if comm is not None:
            resolved_plan["commission"] = float(comm)
    except Exception:
        # Ignore; defaults will be applied downstream
        pass

    # Apply interval constraints (best-effort warning only)
    for interval in resolved_plan["intervals"]:
        _ = clamp_interval_period(
            interval,
            resolved_plan.get("start"),
            resolved_plan.get("end"),
            resolved_plan["period_mode"],
        )

    # Add strategies_path so worker processes can initialize the external strategy loader.
    # Prefer an explicit environment variable (STRATEGIES_PATH) when set, then check
    # the common container-mounted path (/app/external_strategies), then fall back to
    # a local `quant-strategies` checkout or `external_strategies` directory.
    # This ensures the CLI works both on the host and inside docker-compose containers.
    try:
        import os

        env_strat = os.getenv("STRATEGIES_PATH")
        if env_strat:
            resolved_plan["strategies_path"] = env_strat
        else:
            # Common mount inside the container used by docker-compose
            container_path = Path("/app/external_strategies")
            if container_path.exists():
                resolved_plan["strategies_path"] = str(container_path)
            else:
                # Host fallback: prefer local checkout 'quant-strategies'
                strat_path = Path("quant-strategies").resolve()
                if strat_path.exists():
                    resolved_plan["strategies_path"] = str(strat_path)
                else:
                    ext = Path("external_strategies")
                    resolved_plan["strategies_path"] = (
                        str(ext.resolve()) if ext.exists() else None
                    )
    except Exception:
        resolved_plan["strategies_path"] = None

    plan_hash = compute_plan_hash(resolved_plan)
    resolved_plan["plan_hash"] = plan_hash

    manifest = {
        "plan": resolved_plan,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    manifest_path = write_manifest(outdir, manifest)
    log.info("Wrote run manifest to %s", manifest_path)

    # Optional: reset DB (dangerous)
    if args.reset_db and not args.dry_run:
        try:
            from src.database import unified_models  # type: ignore[import-not-found]

            unified_models.drop_tables()
            unified_models.create_tables()
            log.warning(
                "Database tables dropped and recreated as requested (--reset-db)"
            )
        except Exception:
            log.exception("Failed to reset database tables")
            return 9

    # Dry-run behavior: print manifest and optionally generate exports, then exit
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
        _run_requested_exports(resolved_plan, collection_path, symbols)
        return 0

    # Idempotency: check DB for existing plan_hash if DB available
    if not args.force:
        try:
            from src.database import unified_models

            if hasattr(unified_models, "find_run_by_plan_hash"):
                existing = unified_models.find_run_by_plan_hash(plan_hash)
                if existing and getattr(existing, "status", None) == "succeeded":
                    log.info(
                        "A succeeded run with the same plan_hash already exists. Use --force to re-run."
                    )
                    return 0
        except Exception:
            log.debug("Could not query DB for existing plan_hash; continuing")

    # Execute the plan
    rc = run_plan(manifest, outdir, dry_run=args.dry_run)

    # Best-effort: persist artifact pointers (manifest, engine summary, fallback summary) into unified_models.RunArtifact
    try:
        from src.database import unified_models  # type: ignore[import-not-found]

        run = None
        try:
            if hasattr(unified_models, "find_run_by_plan_hash"):
                run = unified_models.find_run_by_plan_hash(plan_hash)
            else:
                # fallback: query by plan_hash manually
                sess_tmp = unified_models.Session()
                try:
                    run = (
                        sess_tmp.query(unified_models.Run)
                        .filter(unified_models.Run.plan_hash == plan_hash)
                        .one_or_none()
                    )
                finally:
                    try:
                        sess_tmp.close()
                    except Exception:
                        pass
        except Exception:
            log.exception("Failed to locate run for plan_hash %s", plan_hash)
            run = None

        if run:
            sess = unified_models.Session()
            try:
                artifact_candidates = [
                    ("manifest", manifest_path),
                    ("engine_summary", outdir / "engine_run_summary.json"),
                    ("run_summary_fallback", outdir / "run_summary_fallback.json"),
                ]
                added = 0
                for atype, p in artifact_candidates:
                    try:
                        # Only persist existing artifact files (handle Path objects)
                        p_path = Path(p)
                        if p_path.exists():
                            ra = unified_models.RunArtifact(
                                run_id=getattr(run, "run_id", None),
                                artifact_type=atype,
                                path_or_uri=str(p_path),
                                meta=None,
                            )
                            sess.add(ra)
                            added += 1
                        else:
                            log.debug("Artifact file not present, skipping: %s", p_path)
                    except Exception:
                        log.exception("Failed to add RunArtifact entry for %s", p)
                if added:
                    sess.commit()
                    # Log number of artifacts added for visibility
                    try:
                        cnt = (
                            sess.query(unified_models.RunArtifact)
                            .filter(
                                unified_models.RunArtifact.run_id
                                == getattr(run, "run_id", None)
                            )
                            .count()
                        )
                        log.info(
                            "Persisted %d run artifact pointers to DB for run %s",
                            cnt,
                            getattr(run, "run_id", None),
                        )
                    except Exception:
                        log.info(
                            "Persisted run artifact pointers to DB for run %s",
                            getattr(run, "run_id", None),
                        )
                else:
                    sess.rollback()
                    log.debug(
                        "No artifact files found to persist for run %s",
                        getattr(run, "run_id", None),
                    )
            except Exception:
                try:
                    sess.rollback()
                except Exception:
                    pass
                log.exception(
                    "Failed to persist run artifacts to DB for run %s",
                    getattr(run, "run_id", None),
                )
            finally:
                try:
                    sess.close()
                except Exception:
                    pass
    except Exception:
        log.debug(
            "Unified models not available for run_artifact persistence (continuing)"
        )

    # Post-run: also run exports when requested (best-effort)
    try:
        if rc == 0 and (resolved_plan.get("exports") or ""):
            _run_requested_exports(resolved_plan, collection_path, symbols)
    except Exception:
        log.exception("Exports failed after run (continuing)")

    return rc


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entrypoint compatible with direct module and top-level dispatch.

    Behavior:
      - If called with 'collection' as a subcommand (e.g. 'collection ...'),
        delegate to handle_collection_run with the args after 'collection'.
      - If called as part of a larger CLI where other args appear before 'collection',
        locate 'collection' in argv and delegate the remainder to handle_collection_run.
      - If no args are supplied, print a minimal help summary.
    """
    if argv is None:
        argv = sys.argv[1:]

    # If no arguments, show basic help
    if not argv:
        parser = argparse.ArgumentParser(
            prog="unified_cli", description="Unified Quant CLI"
        )
        parser.add_argument(
            "collection",
            nargs="?",
            help="Run against a collection (see subcommand 'collection')",
        )
        parser.print_help()
        return 1

    # Locate 'collection' subcommand anywhere in argv
    try:
        idx = int(argv.index("collection"))
    except ValueError:
        idx = -1

    if idx >= 0:
        # Pass everything after the 'collection' token to the dedicated handler
        return handle_collection_run(argv[idx + 1 :])

    # No recognized subcommand found
    print("Unknown command. Supported: collection")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
