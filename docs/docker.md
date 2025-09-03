# Docker Guide

Guide for running this repository with Docker Compose. This reflects the current compose file and unified CLI.

## Services

From `docker-compose.yml`:

- `postgres` — PostgreSQL 15 (persisted via `postgres-data` volume, exposed on host `5433`).
- `pgadmin` — pgAdmin UI (exposed on host `5050`).
- `quant` — Application container (mounts source, strategies, cache, exports, logs, config, artifacts).

## Quick Start

```bash
# 1) Copy env and edit keys
cp .env.example .env

# 2) Start DB + pgAdmin (may pull images on first run)
docker compose up -d postgres pgadmin

# 3) Build the app image
docker compose build quant

# 4) Show CLI help
docker compose run --rm quant python -m src.cli.unified_cli --help
```

## Preferred Run: Bonds, 1d, Max, All Strategies

```bash
docker compose run --rm \
  -e STRATEGIES_PATH=/app/external_strategies \
  quant python -m src.cli.unified_cli collection bonds \
  --action direct \
  --interval 1d \
  --period max \
  --strategies all \
  --exports all \
  --log-level INFO
```

## Dry Run (Plan Only) + Exports

```bash
docker compose run --rm \
  -e STRATEGIES_PATH=/app/external_strategies \
  quant python -m src.cli.unified_cli collection bonds \
  --interval 1d --period max --strategies all \
  --dry-run --exports all --log-level DEBUG

Exports written under `exports/`:
- CSV → `exports/csv/<Year>/<Quarter>/...`
- Reports → `exports/reports/<Year>/<Quarter>/...`
- TV alerts → `exports/tv_alerts/<Year>/<Quarter>/...`
- AI recos (md/html/csv) → `exports/ai_reco/<Year>/<Quarter>/...`
```

## Interactive Shell

```bash
docker compose run --rm quant bash
```

## Ports, Mounts, and Volumes

- Ports: Postgres `5433→5432`, pgAdmin `5050→80`
- Mounts (repo → container):
  - `./cache` → `/app/cache`
  - `./exports` → `/app/exports`
  - `./logs` → `/app/logs`
  - `./config` → `/app/config:ro`
  - `./src` → `/app/src:ro`
  - `./artifacts` → `/app/artifacts`
  - `./quant-strategies/algorithms/python` → `/app/external_strategies:ro`
- Volume: `postgres-data` → `/var/lib/postgresql/data`

## Environment

- In-container DB: `DATABASE_URL=postgresql://quantuser:quantpass@postgres:5432/quant_system`
- Optional API keys: `ALPHA_VANTAGE_API_KEY`, `TWELVE_DATA_API_KEY`, `POLYGON_API_KEY`, `TIINGO_API_KEY`, `FINNHUB_API_KEY`, `BYBIT_API_KEY`, `BYBIT_API_SECRET`, `BYBIT_TESTNET`
- Optional LLMs: `OPENAI_API_KEY`, `OPENAI_MODEL`, `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`

## pgAdmin

- Open `http://localhost:5050`
- Credentials from `.env` (`PGADMIN_DEFAULT_EMAIL`, `PGADMIN_DEFAULT_PASSWORD`)
- Register server: host `postgres`, port `5432`, DB `quant_system`, user `quantuser`

## Troubleshooting

- Use singular subcommand `collection` (not `collections`).
- Ensure strategies are mounted and set `STRATEGIES_PATH=/app/external_strategies` when running.
- For timeouts/long runs, start with `--dry-run`, then narrow strategies/symbols or set `--max-workers` appropriately.
- See `docs/pgadmin-and-performance.md` for SQL queries, performance tuning, and psql connection strings.
