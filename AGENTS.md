# AGENTS.md

## Project overview
This repo is a Dockerized, cache-aware backtesting system for running multiple strategies across asset collections (crypto, stocks, bonds, commodities, FX). The CLI lives in `src/main.py` and loads YAML collection configs under `config/collections/`.

## How to run

- Docker (recommended):
  - `docker-compose run --rm app bash -lc "poetry run python -m src.main run --config config/collections/crypto.yaml"`
- Local (Poetry):
  - `poetry run python -m src.main run --config config/collections/crypto.yaml`
- List discovered strategies:
  - Local: `poetry run python -m src.main list-strategies` (uses `.env` / `STRATEGIES_PATH` on the host)
  - Docker: `docker-compose run --rm app bash -lc "poetry run python -m src.main list-strategies --strategies-path /ext/strategies"`

## Configuration conventions

- Collections live in `config/collections/*.yaml`.
- Prefer provider-agnostic symbols (e.g., `GC`, `CL`, `SPX`, `EURUSD`); mappers decorate per provider (yfinance, ccxt, etc.).
- If you see decorated symbols like `ZW=F` in logs, it usually means the config used the decorated form; prefer canonical symbols in configs.

## Reports and caches

- Outputs go to `reports/<timestamp>/` and include `summary.csv`, `all_results.csv`, `report.md`, `tradingview.md`, `summary.json`, and `metrics.prom`.
- Caches live under `.cache/` (data and results). Use `quant-system clean-cache` to prune.
- The dashboard uses `summary.json`; older runs without it may log missing-summary warnings.
- On Windows, writers for Unicode-heavy exports (e.g. `tradingview.md`) must use UTF-8 explicitly; relying on the default locale encoding can raise `UnicodeEncodeError`.

## External strategies repo

- Strategies are discovered from an external repo. Set `STRATEGIES_PATH` to the strategies root **on the machine running the CLI** (absolute host path for local Poetry). Inside the app container use `/ext/strategies`; a container-only path on the host (e.g. Windows resolving `/ext/...`) will not discover modules.
- `HOST_STRATEGIES_PATH` in docker-compose only defines the bind mount into the container; it does not set `STRATEGIES_PATH` for Poetry on the host.
- All discovered strategies run by default; `strategies:` in YAML only overrides parameter grids.

## Environment

- API keys live in `.env` (copy from `.env.example`). Do not commit secrets.
- The CLI loads `.env` once at startup for all Typer commands (not only `run`), so variables like `STRATEGIES_PATH` apply to `list-strategies` and other subcommands without shell exports.

## Useful CLI commands

- `poetry run quant-system dashboard --reports-dir reports`
- `poetry run quant-system manifest-status --reports-dir reports --latest`
- `poetry run quant-system ingest-data --source yfinance --symbols GC,CL --timeframe 1d`
- `poetry run quant-system clean-cache --cache-dir .cache/data --dry-run`
