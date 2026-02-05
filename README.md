# Quant System

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
- Python 3.12 or 3.13
- External strategies repo mounted at runtime
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
- config/collections/: Per-collection configs (stocks, bonds, crypto, commodities)

## Quick Start

1) Configure your run in config/example.yaml (collections, timeframes, metrics, strategies, params).
2) Ensure your strategies repo contains classes deriving BaseStrategy (see src/strategies/base.py and the example).
3) Check discovered strategies:

   docker-compose run --rm app bash -lc "poetry run python -m src.main list-strategies --strategies-path /ext/strategies"

4) Run via docker-compose (Poetry):

```bash
docker-compose run --rm app bash -lc "poetry run python -m src.main run --config config/collections/crypto.yaml"
# or stocks/bonds/commodities individually
docker-compose run --rm app bash -lc "poetry run python -m src.main run --config config/collections/bonds_global.yaml"
# generate reports with top-5 per symbol and offline HTML
docker-compose run --rm app bash -lc "poetry run python -m src.main run --config config/collections/crypto.yaml --top-n 5 --inline-css"
```

## Make Targets

- `make build` / `make build-nc`: build image (no-cache).
- `make sh`: open a shell in the container.
- `make list-strategies`: verify external strategies are discovered.
- `make run-stocks-dividend` / `make run-stocks-large-cap-value` / `make run-stocks-large-cap-growth` / `make run-stocks-mid-cap` / `make run-stocks-small-cap` / `make run-stocks-international` / `make run-stocks-emerging` / `make run-bonds-global` / `make run-bonds-high-yield` / `make run-bonds-corporate` / `make run-bonds-municipal` / `make run-bonds-tips` / `make run-bonds-us-treasuries` / `make run-crypto` / `make run-commodities`: run a collection.
- `make discover-crypto EXCHANGE=binance QUOTE=USDT TOP=100 OUT=config/collections/crypto_top100.yaml NAME=crypto_top100`: generate a crypto universe config.
- `make lock` / `make lock-update`: create or update `poetry.lock` inside the container for reproducible builds.
- `make manifest-status`: inspect the most recent run’s dashboard refresh actions (`--latest`).
- `poetry run quant-system dashboard --reports-dir reports`: launch a lightweight FastAPI dashboard (defaults to `127.0.0.1:8000`).

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

## Support & Requests

- Report bugs via: https://vollcom-digital.atlassian.net/jira/software/c/form/014e93fe-22c4-4211-a93e-1803d8788ab3
- Request new features via: https://vollcom-digital.atlassian.net/jira/software/c/form/35d2a798-bc62-4bbf-aed5-c6149bc3a5a5
- The future roadmap is available to internal team members only: https://vollcom-digital.atlassian.net/wiki/spaces/VD/folder/3096117251?atlOrigin=eyJpIjoiN2FmODYwYWQxNWI5NDUyOWJiODU3ZDBkNmYxZGI4Y2IiLCJwIjoiYyJ9

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
- `discover-symbols`: fetch top symbols from one or more CCXT exchanges, merge by quote, and emit a YAML stub. Supports multiple `--exchange` flags, `--max-per-exchange` before merging, exclusions via `--exclude-symbol/--exclude-pattern`, manual additions with `--extra-symbol`, and `--annotate` to include volume/exchange metadata.
- `ingest-data`: on-demand cache refresher. Accepts a `--source` (currently `yfinance`), target symbols, and `--timeframe` flags to pull fresh OHLCV data into `.cache/data`.
- `fundamentals`: snapshot yfinance fundamentals (info, splits, dividends, financial statements) for a symbol in JSON or YAML format.
- `manifest-status`: inspect dashboard refresh actions (supports `--latest`). Example:

  ```bash
  poetry run quant-system manifest-status --reports-dir reports --run-id 20240101-000000
  # or simply
  poetry run quant-system manifest-status --reports-dir reports --latest
  ```

  The command falls back to `summary.json` when `manifest_status.json` is missing so older runs still surface CSV fallbacks or missing summaries.

- `clean-cache`: prune stale files from data/results caches. Defaults to removing items older than 30 days while keeping recent entries. Use `--dry-run` to preview deletions. Example:

  ```bash
  poetry run quant-system clean-cache --cache-dir .cache/data --results-cache-dir .cache/results --max-age-days 14
  ```

  To preview without deleting anything:

  ```bash
  poetry run quant-system clean-cache --cache-dir .cache/data --dry-run
  ```

### Dashboard

Spin up a lightweight FastAPI dashboard to browse runs and open existing HTML reports:

```bash
poetry run quant-system dashboard --reports-dir reports --host 0.0.0.0 --port 8000
```

- `/` renders a Tailwind-powered index of runs with links to `report.html`.
- `/<run_id>` shows summary metrics, manifest/notification tables, and download links.
- `/api/runs` provides JSON summaries (run id, metrics, timestamps) for automation.

### Notifications

Configure Slack alerts in your run config:

```yaml
notifications:
  slack:
    webhook_url: ${SLACK_WEBHOOK_URL}
    metric: sharpe
    threshold: 2.0
```

During a run, the top result for the configured metric is evaluated; if it meets the threshold the system posts a summary message to the webhook. Results below the threshold are skipped and reported in the CLI output.

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

- Uses PyBroker to execute and grid-search parameters with resume via SQLite.
- Supports Optuna-based Bayesian parameter search via `param_search: optuna` and `param_trials`.

Strategy Interface (External)

- Derive from BaseStrategy and implement (in your external repo only):
  - name: str
  - param_grid(self) -> dict[str, list]
  - generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]
  - optional: to_tradingview_pine(self, params: dict) -> str

Note: This repo does not contain strategies; it loads them from your external repo. If none are found, the run will fail.
