from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import typer

from .backtest.runner import BacktestRunner
from .config import load_config
from .reporting.all_csv_export import AllCSVExporter
from .reporting.csv_export import CSVExporter
from .reporting.health import HealthReporter
from .reporting.html import HTMLReporter
from .reporting.markdown import MarkdownReporter
from .reporting.tradingview import TradingViewExporter

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    config: str = typer.Option("config/example.yaml", help="Path to YAML config"),
    output_dir: str | None = typer.Option(
        None, help="Reports output dir (default: reports/<timestamp>)"
    ),
    strategies_path: str | None = typer.Option(
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
    except Exception:
        pass
    # Basic logging
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

    # Cache HTTP where possible to reduce provider calls
    try:
        import requests_cache

        requests_cache.install_cache("http_cache", expire_after=43200)  # 12 hours
    except Exception:
        pass

    cfg = load_config(config)
    env_cache = os.environ.get("DATA_CACHE_DIR")
    if env_cache:
        cfg.cache_dir = env_cache

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    base_out = Path(output_dir) if output_dir else Path("reports") / ts
    base_out.mkdir(parents=True, exist_ok=True)

    strategies_root = (
        Path(strategies_path)
        if strategies_path
        else Path(os.environ.get("STRATEGIES_PATH", "/ext/strategies"))
    )

    start_ts = datetime.utcnow()
    run_id = os.environ.get("RUN_ID", ts)
    runner = BacktestRunner(cfg, strategies_root=strategies_root, run_id=run_id)
    if not getattr(runner, "external_index", {}):
        typer.secho(
            (
                f"No strategies discovered under {strategies_root}.\n"
                "- Ensure STRATEGIES_PATH points to the container path (e.g., /ext/strategies).\n"
                "- Or pass --strategies-path /ext/strategies.\n"
                "- Verify your strategy classes subclass BaseStrategy and import without errors.\n"
            ),
            err=True,
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
        typer.secho("\n".join(tips), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2)

    end_ts = datetime.utcnow()

    # Exports
    CSVExporter(base_out).export(results)
    # Consolidated CSVs from results cache
    try:
        AllCSVExporter(base_out, runner.results_cache, run_id, top_n=top_n).export(results)
    except Exception:
        pass
    MarkdownReporter(base_out).export(results)
    TradingViewExporter(base_out).export(results)
    # HTML report with Tailwind (dark mode)
    try:
        HTMLReporter(
            base_out, runner.results_cache, run_id, top_n=top_n, inline_css=inline_css
        ).export(results)
    except Exception:
        pass

    # Emit run summary JSON
    try:
        import json

        summary = {
            "started_at": start_ts.isoformat() + "Z",
            "finished_at": end_ts.isoformat() + "Z",
            "duration_sec": (end_ts - start_ts).total_seconds(),
            "metric": cfg.metric,
            "results_count": len(results),
            "metrics": getattr(runner, "metrics", {}),
            "failures_count": len(getattr(runner, "failures", [])),
            "failures": getattr(runner, "failures", []),
        }
        (base_out / "summary.json").write_text(json.dumps(summary, indent=2))
    except Exception:
        pass

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
    except Exception:
        pass

    typer.echo(f"Done. Reports in: {base_out}")

    # Health report
    try:
        HealthReporter(base_out).export(getattr(runner, "failures", []))
    except Exception:
        pass


@app.command()
def list_strategies(
    strategies_path: str | None = typer.Option(
        None, help="Path to external strategies repo (overrides env STRATEGIES_PATH)"
    ),
):
    from .strategies.registry import discover_external_strategies

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
def discover_symbols(
    exchange: str = typer.Option("binance", help="CCXT exchange id (e.g., binance, bybit)"),
    quote: str = typer.Option("USDT", help="Quote currency filter (e.g., USDT, USD)"),
    top_n: int = typer.Option(50, help="Top N symbols by 24h volume"),
    min_volume: float = typer.Option(0.0, help="Minimum 24h volume to include"),
    name: str = typer.Option("crypto_discovered", help="Collection name to embed in YAML"),
    output: str | None = typer.Option(None, help="Path to write YAML (default: print to stdout)"),
):
    import yaml

    from .utils.symbols import DiscoverOptions, discover_ccxt_symbols

    opts = DiscoverOptions(exchange=exchange, quote=quote, top_n=top_n, min_volume=min_volume)
    pairs = discover_ccxt_symbols(opts)
    symbols = [s for s, _ in pairs]
    cfg = {
        "metric": "sortino",
        "engine": "vectorbt",
        "asset_workers": 4,
        "param_workers": 2,
        "max_fetch_concurrency": 2,
        "cache_dir": ".cache/data",
        "collections": [
            {
                "name": name,
                "source": exchange,
                "exchange": exchange,
                "quote": quote,
                "fees": 0.0006,
                "slippage": 0.0005,
                "symbols": symbols,
            }
        ],
        "timeframes": ["1d", "4h", "1h"],
        "strategies": [],
    }
    text = yaml.safe_dump(cfg, sort_keys=False)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text)
        typer.echo(f"Wrote: {output}")
    else:
        typer.echo(text)


if __name__ == "__main__":
    app()
