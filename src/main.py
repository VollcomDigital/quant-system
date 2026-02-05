import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import typer

from .backtest.runner import BacktestRunner
from .config import load_config
from .dashboard.server import create_app
from .data.yfinance_source import YFinanceSource
from .reporting.all_csv_export import AllCSVExporter
from .reporting.csv_export import CSVExporter
from .reporting.dashboard import (
    DashboardReporter,
    build_dashboard_payload,
    collect_runs_manifest,
)
from .reporting.health import HealthReporter
from .reporting.html import HTMLReporter
from .reporting.manifest import refresh_manifest
from .reporting.markdown import MarkdownReporter
from .reporting.notifications import notify_all
from .reporting.tradingview import TradingViewExporter
from .strategies.registry import discover_external_strategies
from .utils.json_utils import safe_json_dumps
from .utils.symbols import DiscoverOptions, discover_ccxt_symbols

app = typer.Typer(add_completion=False, no_args_is_help=True)

DATA_SOURCES = {
    "yfinance": YFinanceSource,
}


@app.command()
def run(
    config: str = typer.Option("config/example.yaml", help="Path to YAML config"),
    output_dir: str = typer.Option(None, help="Reports root directory (default: reports/<run_id>)"),
    strategies_path: str = typer.Option(
        None, help="Path to external strategies repo (overrides env STRATEGIES_PATH)"
    ),
    only_cached: bool = typer.Option(False, help="Use only cached Parquet data; do not fetch"),
    top_n: int = typer.Option(3, help="Top-N per symbol for CSV/HTML reports"),
    inline_css: bool = typer.Option(False, help="Inline minimal CSS for offline HTML report"),
):
    # Load .env if present
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception as exc:
        logging.getLogger("quant.main").debug("dotenv load failed", exc_info=exc)
    # Basic logging
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    logger = logging.getLogger("quant.main")

    # Cache HTTP where possible to reduce provider calls
    try:
        import requests_cache

        requests_cache.install_cache("http_cache", expire_after=43200)  # 12 hours
    except Exception as exc:
        logger.debug("requests_cache unavailable", exc_info=exc)

    cfg = load_config(config)
    env_cache = os.environ.get("DATA_CACHE_DIR")
    if env_cache:
        cfg.cache_dir = env_cache

    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_id = os.environ.get("RUN_ID", ts)
    reports_root = Path(output_dir) if output_dir else Path("reports")
    base_out = reports_root if reports_root.name == run_id else reports_root / run_id
    base_out.mkdir(parents=True, exist_ok=True)

    strategies_root = (
        Path(strategies_path)
        if strategies_path
        else Path(os.environ.get("STRATEGIES_PATH", "/ext/strategies"))
    )

    start_ts = datetime.now(UTC)
    runner = BacktestRunner(cfg, strategies_root=strategies_root, run_id=run_id)
    if not getattr(runner, "external_index", {}):
        typer.secho(
            (
                f"No strategies discovered under {strategies_root}.\n"
                "- Ensure STRATEGIES_PATH points to the container path (e.g., /ext/strategies).\n"
                "- Or pass --strategies-path /ext/strategies.\n"
                "- Verify your strategy classes subclass BaseStrategy and import without errors.\n"
            ),
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    results = runner.run_all(only_cached=only_cached)
    if not results:
        tips = [
            "No backtest results produced.",
            f"- Strategies discovered: {len(getattr(runner, 'external_index', {}))}",
            f"- Collections: {len(cfg.collections)}, Timeframes: {len(cfg.timeframes)}",
            "Possible causes:",
            "  • Using --only-cached but no Parquet data is cached (warm the cache or disable).",
            "  • Unsupported timeframe for the selected data source (adjust config).",
            "  • Strategy generated invalid/no signals for all parameter sets (check logic/params).",
            "  • Data providers returned no data (rate limits/network/API keys).",
        ]
        typer.secho("\n".join(tips), fg=typer.colors.RED)
        raise typer.Exit(code=2)

    notifications_cfg = getattr(cfg, "notifications", None)
    notification_events = notify_all(results, notifications_cfg, run_id)
    for event in notification_events:
        if event.get("sent"):
            typer.echo(
                f"notification: {event['channel']} sent for {event.get('symbol', '')} {event.get('metric')} {event.get('value', '')}"
            )
        else:
            reason = event.get("reason", "skipped")
            typer.echo(
                f"notification: {event['channel']} {reason} for {event.get('symbol', '')} {event.get('metric')}"
            )

    end_ts = datetime.now(UTC)

    # Exports
    CSVExporter(base_out).export(results)
    # Consolidated CSVs from results cache
    try:
        AllCSVExporter(base_out, runner.results_cache, run_id, top_n=top_n).export(results)
    except Exception as exc:
        logger.warning("all_results export failed", exc_info=exc)
    MarkdownReporter(base_out).export(results)
    TradingViewExporter(base_out).export(results)
    # HTML report with Tailwind (dark mode)
    try:
        HTMLReporter(
            base_out, runner.results_cache, run_id, top_n=top_n, inline_css=inline_css
        ).export(results)
    except Exception as exc:
        logger.warning("HTML report export failed", exc_info=exc)

    duration = (end_ts - start_ts).total_seconds()
    dashboard_payload = build_dashboard_payload(runner.results_cache, run_id, results)
    dashboard_payload.update(
        {
            "metric": cfg.metric,
            "results_count": len(results),
            "started_at": start_ts.isoformat() + "Z",
            "finished_at": end_ts.isoformat() + "Z",
            "duration_sec": duration,
        }
    )
    runs_manifest = collect_runs_manifest(
        base_out.parent,
        run_id,
        dashboard_payload.get("summary"),
        {
            "metric": cfg.metric,
            "results_count": len(results),
            "started_at": start_ts.isoformat() + "Z",
            "duration_sec": duration,
        },
    )
    dashboard_payload["runs"] = runs_manifest
    manifest_status: list[dict[str, object]] = []
    try:
        manifest_status = refresh_manifest(
            Path(base_out).parent, base_out, runner.results_cache, dashboard_payload
        )
    except Exception as exc:
        logger.warning("manifest refresh failed", exc_info=exc)
        manifest_status = []

    warning_lines: list[str] = []
    for status in manifest_status:
        message = status.get("message")
        if message:
            warning_lines.append(f"{status.get('run_id')}: {message}")

    try:
        DashboardReporter(base_out).export(dashboard_payload)
    except Exception as exc:
        logger.warning("dashboard export failed", exc_info=exc)

    # Emit run summary JSON
    try:
        summary = {
            "started_at": start_ts.isoformat() + "Z",
            "finished_at": end_ts.isoformat() + "Z",
            "duration_sec": duration,
            "metric": cfg.metric,
            "results_count": len(results),
            "metrics": getattr(runner, "metrics", {}),
            "failures_count": len(getattr(runner, "failures", [])),
            "failures": getattr(runner, "failures", []),
            "dashboard": dashboard_payload.get("summary", {}),
            "runs": runs_manifest,
            "manifest_refresh": manifest_status,
            "notifications": notification_events,
        }
        (base_out / "summary.json").write_text(safe_json_dumps(summary, indent=2))
        if manifest_status:
            (base_out / "manifest_status.json").write_text(
                safe_json_dumps(manifest_status, indent=2)
            )
        if notification_events:
            (base_out / "notifications.json").write_text(
                safe_json_dumps(notification_events, indent=2)
            )
    except Exception as exc:
        logger.warning("summary emit failed", exc_info=exc)

    # Emit Prometheus-style metrics
    try:
        m = []
        duration = (end_ts - start_ts).total_seconds()
        rm = getattr(runner, "metrics", {})

        def line(k, v):
            m.append(f"quant_{k} {v}")

        line("run_duration_seconds", duration)
        line("results_count", len(results))
        for k in (
            "result_cache_hits",
            "result_cache_misses",
            "param_evals",
            "symbols_tested",
            "strategies_count",
        ):
            if k in rm:
                line(k, rm[k])
        (base_out / "metrics.prom").write_text("\n".join(m) + "\n")
    except Exception as exc:
        logger.warning("metrics emit failed", exc_info=exc)

    typer.echo(f"Done. Reports in: {base_out}")
    for line in warning_lines:
        typer.echo(f"warning: {line}")

    # Health report
    try:
        HealthReporter(base_out).export(getattr(runner, "failures", []))
    except Exception as exc:
        logger.warning("health report export failed", exc_info=exc)


@app.command()
def list_strategies(
    strategies_path: str = typer.Option(
        None, help="Path to external strategies repo (overrides env STRATEGIES_PATH)"
    ),
):
    strategies_root = (
        Path(strategies_path)
        if strategies_path
        else Path(os.environ.get("STRATEGIES_PATH", "/ext/strategies"))
    )
    index = discover_external_strategies(strategies_root)
    if not index:
        typer.echo(f"No strategies found under {strategies_root}")
        raise typer.Exit(code=1)
    typer.echo(f"Found {len(index)} strategies:")
    for name in sorted(index.keys()):
        typer.echo(f"- {name}")


@app.command()
def manifest_status(
    reports_dir: str = typer.Option("reports", help="Reports root directory"),
    run_id: str = typer.Option(None, help="Specific run id"),
    latest: bool = typer.Option(False, help="Use most recent run"),
):
    import json

    root = Path(reports_dir)
    if not root.exists():
        typer.secho(f"Reports directory not found: {root}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if run_id and latest:
        typer.secho("Use either --run-id or --latest, not both", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    candidate_dirs = [p for p in root.iterdir() if p.is_dir()]
    run_id_value = run_id
    if latest:
        run_id_value = None

    if run_id_value:
        run_dir = root / run_id_value
        if not run_dir.exists():
            typer.secho(f"Run directory not found: {run_dir}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
    else:
        if not candidate_dirs:
            typer.secho("No run directories found", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)
        run_dir = max(candidate_dirs, key=lambda p: p.name)

    status_path = run_dir / "manifest_status.json"
    if status_path.exists():
        try:
            statuses = json.loads(status_path.read_text())
        except Exception as exc:
            typer.secho(f"Unable to read manifest status: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc
    else:
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            typer.secho(
                f"No manifest_status.json or summary.json found under {run_dir}",
                fg=typer.colors.YELLOW,
            )
            raise typer.Exit(code=1)
        try:
            summary = json.loads(summary_path.read_text())
            statuses = summary.get("manifest_refresh") or []
        except Exception as exc:
            typer.secho(f"Unable to read summary.json: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc

    if not statuses:
        typer.secho(f"No manifest actions recorded for {run_dir.name}", fg=typer.colors.YELLOW)
        raise typer.Exit(code=0)

    typer.echo(f"Manifest status for run {run_dir.name}:")
    for item in statuses:
        run = item.get("run_id", run_dir.name)
        status = item.get("status", "unknown")
        message = item.get("message") or ""
        source = item.get("source")
        line = f"- {run}: {status}"
        if source:
            line += f" (source={source})"
        if message:
            line += f" — {message}"
        typer.echo(line)


@app.command()
def rebuild_dashboard(
    reports_dir: str = typer.Option("reports", help="Reports root directory"),
    run_id: str = typer.Option(None, help="Specific run id"),
    latest: bool = typer.Option(False, help="Use most recent run"),
    results_cache_dir: str = typer.Option(".cache/results", help="Results cache directory"),
):
    import json

    from .backtest.results_cache import ResultsCache
    from .reporting.dashboard import (
        METRIC_KEYS,
        DashboardReporter,
        _build_summary,
        _extract_highlights,
        collect_runs_manifest,
    )
    from .reporting.manifest import _build_payload_from_summary, _rows_from_csv

    root = Path(reports_dir)
    if not root.exists():
        typer.secho(f"Reports directory not found: {root}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if run_id and latest:
        typer.secho("Use either --run-id or --latest, not both", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    candidate_dirs = [p for p in root.iterdir() if p.is_dir()]
    run_id_value = run_id
    if latest:
        run_id_value = None

    if run_id_value:
        run_dir = root / run_id_value
        if not run_dir.exists():
            typer.secho(f"Run directory not found: {run_dir}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
    else:
        if not candidate_dirs:
            typer.secho("No run directories found", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)
        run_dir = max(candidate_dirs, key=lambda p: p.name)

    summary_path = run_dir / "summary.json"
    summary: dict[str, Any] | None = None
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            summary.setdefault("base_dir", str(run_dir))
        except Exception as exc:
            typer.secho(f"Unable to read summary.json: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc

    cache = ResultsCache(Path(results_cache_dir))
    payload: dict[str, Any] | None = None
    if summary:
        payload = _build_payload_from_summary(summary, run_dir.name, cache)

    rows: list[dict[str, Any]] = []
    if payload is None:
        rows = cache.list_by_run(run_dir.name)
        if not rows:
            results_csv = run_dir / "all_results.csv"
            if results_csv.exists():
                rows = _rows_from_csv(results_csv)
        if not rows:
            typer.secho(
                "Unable to rebuild dashboard; no cached rows or all_results.csv found.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        summary_data = _build_summary(rows)
        payload = {
            "run_id": run_dir.name,
            "rows": rows,
            "summary": summary_data,
            "available_metrics": list(summary_data.get("metrics", {}).keys()) or list(METRIC_KEYS),
            "highlights": _extract_highlights(summary_data),
        }

    if summary:
        payload.setdefault("metric", summary.get("metric"))
        payload.setdefault(
            "results_count", summary.get("results_count", len(payload.get("rows", [])))
        )
        payload.setdefault("started_at", summary.get("started_at"))
        payload.setdefault("finished_at", summary.get("finished_at"))
        payload.setdefault("duration_sec", summary.get("duration_sec"))
    else:
        payload.setdefault("results_count", len(rows))

    runs_manifest = collect_runs_manifest(
        run_dir.parent,
        run_dir.name,
        payload.get("summary"),
        {
            "metric": payload.get("metric"),
            "results_count": payload.get("results_count"),
            "started_at": payload.get("started_at"),
            "duration_sec": payload.get("duration_sec"),
        },
    )
    payload["runs"] = runs_manifest

    DashboardReporter(run_dir).export(payload)
    typer.echo(f"Rebuilt dashboard for run {run_dir.name}")


@app.command()
def dashboard(
    reports_dir: str = typer.Option("reports", help="Reports root directory"),
    host: str = typer.Option("127.0.0.1", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to bind"),
):
    import uvicorn

    app_fastapi = create_app(Path(reports_dir))
    uvicorn.run(app_fastapi, host=host, port=port, log_level="info")


@app.command()
def discover_symbols(
    exchanges: list[str] = typer.Option(
        ["binance"], "--exchange", "-e", help="CCXT exchange id(s); repeat flag"
    ),
    quote: str = typer.Option("USDT", help="Quote currency filter (e.g., USDT, USD)"),
    top_n: int = typer.Option(50, help="Top N symbols after merging exchanges"),
    max_per_exchange: int = typer.Option(
        None, help="Limit symbols fetched from each exchange before merging"
    ),
    min_volume: float = typer.Option(0.0, help="Minimum 24h volume to include"),
    exclude_symbol: list[str] = typer.Option(
        [],
        "--exclude-symbol",
        help="Symbol(s) to omit from the final universe",
    ),
    exclude_pattern: list[str] = typer.Option(
        [],
        "--exclude-pattern",
        help="fnmatch-style pattern(s) to omit (e.g., *UP/USDT)",
    ),
    extra_symbol: list[str] = typer.Option(
        [],
        "--extra-symbol",
        help="Additional symbols to append even if filters remove them",
    ),
    name: str = typer.Option("crypto_discovered", help="Collection name to embed in YAML"),
    output: str = typer.Option(None, help="Path to write YAML (default: print to stdout)"),
    annotate: bool = typer.Option(False, help="Include exchange/volume metadata in generated YAML"),
):
    import fnmatch

    import yaml

    if not exchanges:
        raise typer.BadParameter("Provide at least one --exchange")

    per_exchange_limit = max_per_exchange or top_n
    all_entries: list[tuple[str, str, float]] = []
    for exch in exchanges:
        opts = DiscoverOptions(
            exchange=exch, quote=quote, top_n=per_exchange_limit, min_volume=min_volume
        )
        pairs = discover_ccxt_symbols(opts)
        all_entries.extend((exch, symbol, float(volume)) for symbol, volume in pairs)

    if not all_entries:
        raise typer.BadParameter("No symbols discovered")

    def should_exclude(symbol: str) -> bool:
        if symbol in exclude_symbol:
            return True
        return any(fnmatch.fnmatch(symbol, pattern) for pattern in exclude_pattern)

    deduped: dict[str, dict[str, object]] = {}
    for exch, symbol, volume in all_entries:
        if should_exclude(symbol):
            continue
        current = deduped.get(symbol)
        if current is None or volume > float(current["volume"]):
            deduped[symbol] = {"symbol": symbol, "volume": float(volume), "exchange": exch}

    merged = sorted(deduped.values(), key=lambda item: float(item["volume"]), reverse=True)
    top_entries = merged[:top_n]

    existing_symbols = {entry["symbol"] for entry in top_entries}
    for extra in extra_symbol:
        if extra not in existing_symbols:
            top_entries.append({"symbol": extra, "volume": 0.0, "exchange": "manual"})
            existing_symbols.add(extra)

    symbols = [entry["symbol"] for entry in top_entries]
    sources = sorted({entry["exchange"] for entry in top_entries}) or exchanges
    cfg = {
        "metric": "sortino",
        "engine": "pybroker",
        "asset_workers": 4,
        "param_workers": 2,
        "max_fetch_concurrency": 2,
        "cache_dir": ".cache/data",
        "collections": [
            {
                "name": name,
                "source": "/".join(sources),
                "exchange": sources[0] if sources else exchanges[0],
                "quote": quote,
                "fees": 0.0006,
                "slippage": 0.0005,
                "symbols": symbols,
            }
        ],
        "timeframes": ["1d", "4h", "1h"],
        "strategies": [],
    }
    if annotate:
        cfg["liquidity"] = top_entries

    text = yaml.safe_dump(cfg, sort_keys=False)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text)
        typer.echo(f"Wrote: {output}")
    else:
        typer.echo(text)


@app.command()
def ingest_data(
    source: str = typer.Option("yfinance", help="Registered data source id"),
    symbols: list[str] = typer.Argument(..., help="One or more symbols to ingest"),
    timeframes: list[str] = typer.Option(
        ["1d"], "--timeframe", "-t", help="Timeframe(s) to refresh"
    ),
    cache_dir: str = typer.Option(".cache/data", help="Target cache directory"),
    only_cached: bool = typer.Option(
        False, help="If true, skip fetch when cache already populated (raises on miss)"
    ),
):
    cls = DATA_SOURCES.get(source)
    if cls is None:
        raise typer.BadParameter(f"Unknown data source: {source}")
    if not symbols:
        raise typer.BadParameter("Provide at least one symbol")
    if not timeframes:
        raise typer.BadParameter("Provide at least one timeframe")

    data_source = cls(Path(cache_dir))
    successes = 0
    failures: list[str] = []
    for symbol in symbols:
        for timeframe in timeframes:
            typer.echo(f"ingest: fetching {symbol} {timeframe} via {source}")
            try:
                data_source.fetch(symbol, timeframe, only_cached=only_cached)
                successes += 1
            except Exception as exc:
                failures.append(f"{symbol} {timeframe}: {exc}")
                typer.secho(
                    f"failed ingest for {symbol} {timeframe}: {exc}",
                    fg=typer.colors.RED,
                    err=True,
                )

    typer.echo(f"ingest: completed {successes} fetches")
    if failures:
        typer.echo("ingest: failures detected:")
        for msg in failures:
            typer.echo(f" - {msg}")
        raise typer.Exit(code=1)


@app.command()
def fundamentals(
    symbol: str = typer.Argument(..., help="Ticker symbol (e.g., AAPL)"),
    cache_dir: str = typer.Option(".cache/data", help="Data cache directory"),
    output: str = typer.Option(None, help="Write fundamentals to file"),
    format: str = typer.Option("json", "--format", "-f", help="json or yaml"),
):
    import json

    import yaml

    fmt = format.lower()
    if fmt not in {"json", "yaml"}:
        raise typer.BadParameter("format must be json or yaml")

    src = YFinanceSource(Path(cache_dir))
    data = src.fetch_fundamentals(symbol)

    if fmt == "json":
        text = json.dumps(data, indent=2, default=float)
    else:
        text = yaml.safe_dump(data, sort_keys=False)

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text)
        typer.echo(f"Wrote fundamentals to {output}")
    else:
        typer.echo(text)


@app.command()
def package_run(
    run_id: str = typer.Argument(..., help="Run identifier"),
    reports_dir: str = typer.Option("reports", help="Reports root directory"),
    output: str = typer.Option(None, help="Output archive path (default: reports/<run_id>.zip)"),
):
    import shutil

    run_dir = Path(reports_dir) / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        typer.secho(f"Run directory not found: {run_dir}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    out_path = Path(output) if output else Path(reports_dir) / f"{run_id}.zip"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_name = out_path.with_suffix("")
    archive = shutil.make_archive(str(base_name), "zip", run_dir)
    typer.echo(f"Packaged run to {archive}")


@app.command()
def clean_cache(
    cache_dir: str = typer.Option(".cache/data", help="Primary data cache directory"),
    results_cache_dir: str = typer.Option(
        ".cache/results", help="Results cache directory for aggregated outputs"
    ),
    include_results: bool = typer.Option(True, help="Also prune the results cache"),
    max_age_days: int = typer.Option(30, min=1, help="Delete files older than this many days"),
    dry_run: bool = typer.Option(False, help="List deletions without removing files"),
):
    """Permanently delete stale cache files beyond the retention window."""

    now = datetime.now(UTC)
    threshold = now - timedelta(days=max_age_days)
    targets: list[tuple[str, Path]] = [("data", Path(cache_dir))]
    if include_results:
        targets.append(("results", Path(results_cache_dir)))

    removed_files = 0
    removed_bytes = 0

    for label, root in targets:
        if not root.exists():
            typer.echo(f"cache-clean: {label} cache not found at {root}, skipping")
            continue

        candidate_files = [file for file in root.rglob("*") if file.is_file()]
        if not candidate_files:
            typer.echo(f"cache-clean: {label} cache at {root} has no files")
            continue

        typer.echo(f"cache-clean: scanning {len(candidate_files)} files under {root}")

        for file_path in candidate_files:
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime, UTC)
            if mtime > threshold:
                continue

            age_days = (now - mtime).days
            file_size = file_path.stat().st_size

            if dry_run:
                typer.echo(f"DRY-RUN delete {file_path} (age={age_days}d)")
                continue

            try:
                file_path.unlink()
                removed_files += 1
                removed_bytes += file_size
            except Exception as exc:
                typer.echo(f"cache-clean: failed to remove {file_path}: {exc}", err=True)

        if not dry_run:
            # Remove emptied directories bottom-up.
            for dir_path in sorted({p.parent for p in candidate_files}, reverse=True):
                try:
                    if dir_path.exists() and not any(dir_path.iterdir()):
                        dir_path.rmdir()
                except Exception:
                    pass

    if dry_run:
        typer.echo("cache-clean: dry run complete")
    elif removed_files:
        freed_mb = removed_bytes / (1024 * 1024) if removed_bytes else 0.0
        typer.echo(f"cache-clean: removed {removed_files} files totalling {freed_mb:.2f} MiB")
    else:
        typer.echo("cache-clean: no files exceeded retention threshold")


if __name__ == "__main__":
    app()
