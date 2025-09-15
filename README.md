# Quant System (Dockerized)

## Overview

This repository provides a Docker-based, cache-aware backtesting system to systematically evaluate multiple strategies across multiple assets, timeframes, and the full available history. It discovers strategies from an external repo and produces:

- Markdown report (best combination per asset/strategy/timeframe)
- TradingView alert export (markdown)
- CSV summary of best results

## Key Features

- Pluggable data sources (free and premium-friendly) with local Parquet caching
- Strategy discovery from external repo via a clean BaseStrategy interface
- Batch runs across collections (e.g., crypto, forex, bonds, stocks)
- Parameter grid search with best-by metric (Sharpe, Sortino, or Profit)
- Dockerized runtime for reproducibility
- Results cache (SQLite) to resume and skip already-computed grids
- Structured logging and timing metrics per data fetch and grid search

## Requirements

- Docker and docker-compose
- Poetry (for local non-Docker runs)
- Python 3.9 or 3.10 (vectorbt requires <3.11)
- External strategies repo mounted at runtime (defaults to /Users/manuelheck/Documents/Websites/Private/quant/quant-strategies/algorithms/python)
- Optional: pre-commit for local linting hooks

## Project Structure

- src/main.py: CLI entrypoint (Typer)
- src/config.py: Loads and validates YAML config
- src/data/: Data source interfaces and caching helpers
- src/strategies/: Base strategy interface and external loader
- src/backtest/: Runner, metric computation, and results cache (resume)
- src/utils/telemetry.py: Structured logging utilities and timed context
- src/reporting/: Markdown, CSV, TradingView exporters
- config/example.yaml: Example configuration
- config/collections/: Per-collection configs (crypto, bonds, commodities, indices)

## Quick Start

1) Configure your run in config/example.yaml (collections, timeframes, metrics, strategies, params).
2) Ensure your strategies repo contains classes deriving BaseStrategy (see src/strategies/base.py and the example).
3) Check discovered strategies:

   docker-compose run --rm app bash -lc "poetry run python -m src.main list-strategies --strategies-path /ext/strategies"

4) Run via docker-compose (Poetry):

```bash
docker-compose run --rm app bash -lc "poetry run python -m src.main run --config config/collections/crypto_majors.yaml"
# or bonds/commodities/indices individually
docker-compose run --rm app bash -lc "poetry run python -m src.main run --config config/collections/bonds_majors.yaml"
# generate reports with top-5 per symbol and offline HTML
docker-compose run --rm app bash -lc "poetry run python -m src.main run --config config/collections/crypto_majors.yaml --top-n 5 --inline-css"
```

## Make Targets

- `make build` / `make build-nc`: build image (no-cache).
- `make sh`: open a shell in the container.
- `make list-strategies`: verify external strategies are discovered.
- `make run-bonds` / `make run-crypto` / `make run-commodities` / `make run-indices` / `make run-forex`: run a collection.
- `make discover-crypto EXCHANGE=binance QUOTE=USDT TOP=100 OUT=config/collections/crypto_top100.yaml NAME=crypto_top100`: generate a crypto universe config.
- `make lock` / `make lock-update`: create or update `poetry.lock` inside the container for reproducible builds.

## Outputs

- reports/`{timestamp}`/summary.csv: CSV of best combinations (one per symbol)
- reports/`{timestamp}`/all_results.csv: CSV of all parameter evaluations (consolidated)
- reports/`{timestamp}`/top3.csv: Top-N (default 3) per symbol
- reports/`{timestamp}`/report.md: Markdown report (top combos and metrics)
- reports/`{timestamp}`/tradingview.md: TradingView alert export (per best combo)
- reports/`{timestamp}`/summary.json: Run summary (timings, counters)
- reports/`{timestamp}`/metrics.prom: Prometheus-style metrics textfile

## Notes

- Data caching uses Parquet files under .cache/data; HTTP cached for 12h. yfinance also integrates yfinance-cache when available.
- Free data: yfinance for equities/ETFs/futures; crypto via ccxt with exchange set (e.g., binance, bybit). Calls are rate-limited to avoid throttling.
- Premium data templates: Polygon, Tiingo, Alpaca under src/data/*. Provide API keys via env vars and implement fetch.
- Additional sources: Finnhub (fx/equities intraday), Twelve Data (fx/equities intraday), Alpha Vantage (daily fallback).

### Symbol Mapping

Use provider‑agnostic symbols in config; a mapper translates per provider:

- Futures: use roots like `GC`, `CL`, `SI`, `ZW`, `ZC`, `ZS`, ...
  - yfinance: mapped to `GC=F`, `CL=F`, etc.
  - polygon/tiingo/alpaca: Yahoo decorations removed.
- Indices: you can use `SPX`, `NDX`, `DJI`, `RUT`, `VIX`.
  - yfinance: mapped to `^GSPC`, `^NDX`, `^DJI`, `^RUT`, `^VIX`.
- Share classes: prefer dot form in config, e.g., `BRK.B`.
  - yfinance: mapped to `BRK-B`; others strip back to dot.
- Forex: `EURUSD` or `EUR/USD` in config.
  - yfinance: mapped to `EURUSD=X`; others use raw pair.
- Crypto: `BTCUSD`, `BTC/USDT`, or `BTCUSDT` in config.
  - yfinance: mapped to `BTC-USD`; ccxt uses the slash form.

If you see a log line with a Yahoo‑decorated symbol (e.g., `ZW=F`) under yfinance, it usually means your config already uses the decorated form. Prefer the canonical form (`ZW`) in config so mapping can adapt automatically.

### Providers Overview

- Tiingo: stable daily/intraday for US equities/ETFs. Recommended for bonds, commodities ETFs, and index ETF proxies. Timeframes: 1d and selected intraday (no resampling).
- yfinance: broad free coverage for indices and weekly bars. Recommended for index levels (e.g., SPX), forex daily/hourly. Timeframes: native only.
- CCXT: crypto OHLCV from exchanges (e.g., Binance). Timeframes: exchange-supported only.
- Polygon/Alpaca: robust intraday equities data at scale (paid). Use when you need minute bars with SLAs.
- Finnhub: equities/FX/crypto intraday + fundamentals/news (paid). Good for FX intraday. Env: FINNHUB_API_KEY.
- Twelve Data: FX/equities intraday (paid/free). Good as primary/backup for FX intraday. Env: TWELVEDATA_API_KEY.
- Alpha Vantage: daily fallback for equities/FX (free). Not ideal for heavy intraday. Env: ALPHAVANTAGE_API_KEY.

See new collection examples under `config/collections/` for FX intraday via Finnhub and Twelve Data.

- Results cache: SQLite under .cache/results to resume and skip recomputation per param-set. Cache invalidates automatically when data changes (based on fingerprint).
- Concurrency: set `asset_workers`, `param_workers`, and `max_fetch_concurrency` to control parallelization.
- Per-collection configs live under `config/collections/`. Extend symbol lists to be as comprehensive as desired (majors/minors).
- Strategy selection: all discovered strategies are tested by default; `strategies:` only overrides parameter grids by name.

## CI & Scheduling

- Linting via Ruff in `.github/workflows/ci.yml` on push/PR.
- Daily scheduled backtest via `.github/workflows/daily-backtest.yml` (05:00 UTC). To use your strategies repo in CI:
  - Add secrets `STRATEGIES_REPO` (e.g., org/repo) and `GH_TOKEN` with read access.
  - The workflow checks out both repos and runs `poetry run python -m src.main run --config config/example.yaml --strategies-path strategies`.
  - Security: Gitleaks runs on PRs and `main`.
  - CodeQL: Uses GitHub’s Default setup (enable under Security → Code scanning). No custom workflow is required.

## Governance

- Branch protection and required status checks recommendations are in `GOVERNANCE.md`.
- CODEOWNERS is set under `.github/CODEOWNERS`.

## Symbol Discovery (Crypto)

- Build a universe of top volume pairs via ccxt and emit a config file:

  docker-compose run --rm app bash -lc "poetry run python -m src.main discover-symbols --exchange binance --quote USDT --top-n 100 --name crypto_top100 --output config/collections/crypto_top100.yaml"

## Environment Variables (.env)

- Copy `.env.example` to `.env` and fill keys: `POLYGON_API_KEY`, `TIINGO_API_KEY`, `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`, `FINNHUB_API_KEY`, `TWELVEDATA_API_KEY`, `ALPHAVANTAGE_API_KEY`.
- `docker-compose` loads `.env` automatically; the app also loads `.env` at startup.
- Override cache and strategies path via `DATA_CACHE_DIR` and `STRATEGIES_PATH`.
- For docker-compose host mount, set `HOST_STRATEGIES_PATH` to your local strategies repo; if unset, it falls back to `./external-strategies`.
- Provider keys for scheduled runs can be set as repository secrets and are exported in `.github/workflows/daily-backtest.yml`.

## Git Ignore

- `.gitignore` excludes local caches, reports, virtualenvs, and `.env`.

## Strategy Selection

- The runner discovers all strategies under your external repo and tests all of them by default.
- If you provide `strategies:` in config, their `params` act as overrides for the discovered strategies with matching names; nothing is filtered by collection.

## New CLI Options and Outputs

- `--only-cached`: avoid API calls and use cached Parquet data only; errors on cache miss.
- Emits `summary.json` (run summary + counts) and `metrics.prom` (Prometheus-style gauges) alongside CSV/Markdown exports in `reports/<timestamp>/`.

## Pre-commit Hooks

- Install and enable locally:

```bash
  pip install pre-commit
  pre-commit install
```

- Run hooks on all files once:

```bash
  pre-commit run --all-files
```

- Hooks: Ruff lint and format, YAML checks, whitespace fixes.

## Backtesting Engine & Optimization

- Uses vectorbt to execute and grid-search parameters with resume via SQLite.
- The `backtesting` library is also available; we can enable it as an alternative engine with a strategy adapter if you prefer its built-in optimizer.

Strategy Interface (External)

- Derive from BaseStrategy and implement (in your external repo only):
  - name: str
  - param_grid(self) -> dict[str, list]
  - generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]
  - optional: to_tradingview_pine(self, params: dict) -> str

Note: This repo does not contain strategies; it loads them from your external repo. If none are found, the run will fail.
