# Quant System

A unified, Dockerized quantitative backtesting and reporting system. Run cross‑strategy comparisons for asset collections (e.g., bonds) and persist results to PostgreSQL with exportable artifacts.

## 🚀 Quick Start

### Docker Setup

```bash
# Clone repository
git clone <repository-url>
cd quant-system

# Start PostgreSQL and pgAdmin
docker compose up -d postgres pgadmin

# Build the app image (uses DOCKERFILE)
docker compose build quant

# Show CLI help
docker compose run --rm quant python -m src.cli.unified_cli --help

# Interactive shell inside the app container
docker compose run --rm quant bash
```

## 📈 Usage

See also: docs/pgadmin-and-performance.md for DB inspection and performance tips.

The unified CLI currently exposes a single subcommand: `collection`.

### Run Bonds (1d interval, max period, all strategies)

Use the collection key (`bonds`) or the JSON file path. The `direct` action runs the backtests and writes results to the DB. Add `--exports all` to generate CSV/HTML/TV/AI artifacts when possible.

```bash
# Using the collection key (recommended)
docker compose run --rm \
  -e STRATEGIES_PATH=/app/external_strategies \
  quant python -m src.cli.unified_cli collection bonds \
  --action direct \
  --interval 1d \
  --period max \
  --strategies all \
  --exports all \
  --log-level INFO

# Using the JSON file
docker compose run --rm \
  -e STRATEGIES_PATH=/app/external_strategies \
  quant python -m src.cli.unified_cli collection config/collections/bonds.json \
  --action direct \
  --interval 1d \
  --period max \
  --strategies all \
  --exports all \
  --log-level INFO
```

Notes

- Default metric is `sortino_ratio`.
- Strategies are mounted at `/app/external_strategies` via `docker-compose.yml`; `STRATEGIES_PATH` makes discovery explicit.
- Artifacts are written under `artifacts/run_*`. DB tables used include `runs`, `backtest_results`, `best_strategies`, and `run_artifacts`.
- pgAdmin is available at `http://localhost:5050` (defaults configured via `.env`/`.env.example`).

### Dry Run (plan only + optional exports)

```bash
docker compose run --rm \
  -e STRATEGIES_PATH=/app/external_strategies \
  quant python -m src.cli.unified_cli collection bonds \
  --interval 1d --period max --strategies all \
  --dry-run --exports all --log-level DEBUG
```

### Other Actions

The `collection` subcommand supports these `--action` values: `backtest`, `direct`, `optimization`, `export`, `report`, `tradingview`. In most workflows, use `--action direct` and optionally `--exports`.

## 🔧 Configuration

### Environment Variables (.env)

```bash
# PostgreSQL (inside the container, use the service name 'postgres')
DATABASE_URL=postgresql://quantuser:quantpass@postgres:5432/quant_system

# Optional data providers
ALPHA_VANTAGE_API_KEY=your_key
TWELVE_DATA_API_KEY=your_key
POLYGON_API_KEY=your_key
TIINGO_API_KEY=your_key
FINNHUB_API_KEY=your_key
BYBIT_API_KEY=your_key
BYBIT_API_SECRET=your_secret
BYBIT_TESTNET=false

# Optional LLMs
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o
ANTHROPIC_API_KEY=your_key
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

Host access tips

- Postgres is published on `localhost:5433` (mapped to container `5432`).
- pgAdmin runs at `http://localhost:5050` (see `.env` for credentials).

### Collections

Collections live under `config/collections/` and are split into:

- `default/` (curated, liquid, fast to iterate)
- `custom/` (your own research sets)

Default examples:

- Bonds: `default/bonds_core.json` (liquid bond ETFs), `default/bonds.json` (broader set)
- Commodities: `default/commodities_core.json` (gold/silver/energy/agriculture/broad)
- Crypto: `default/crypto_liquid.json` (top market-cap, USDT pairs)
- Forex: `default/forex_majors.json` (majors and key crosses; Yahoo Finance format `=X`)
- Indices: `default/indices_global_core.json` (SPY/QQQ/DIA/IWM/EFA/EEM/EWJ/FXI etc.)
- Stocks: `default/stocks_us_mega_core.json`, `default/stocks_us_growth_core.json`
  - Factors: `default/stocks_us_value_core.json`, `default/stocks_us_quality_core.json`, `default/stocks_us_minvol_core.json`
  - Global factors: `default/stocks_global_factor_core.json`

Custom examples (research-driven):

- `custom/stocks_traderfox_dax.json`
- `custom/stocks_traderfox_european.json`
- `custom/stocks_traderfox_us_financials.json`
- `custom/stocks_traderfox_us_healthcare.json`
- `custom/stocks_traderfox_us_tech.json`

You can reference any collection by key without the folder prefix (resolver searches `default/` and `custom/`). For example, `bonds_core` resolves `config/collections/default/bonds_core.json`.

## 🧪 Testing

```bash
# Run tests in Docker
docker compose run --rm quant pytest
```

## 📊 Exports & Reporting

Artifacts and exports are written under `artifacts/run_*` and `exports/`. When running with `--action direct` or `--dry-run`, pass `--exports csv,report,tradingview,ai` or `--exports all`.

```bash
# Produce exports from DB for bonds without re-running backtests
docker compose run --rm quant \
  python -m src.cli.unified_cli collection bonds --dry-run --exports all
```

Output locations and unified naming (`{Collection}_Collection_{Year}_{Quarter}_{Interval}`):
- CSV: `exports/csv/{Year}/{Quarter}/{Collection}_Collection_{Year}_{Quarter}_{Interval}.csv`
- HTML reports: `exports/reports/{Year}/{Quarter}/{Collection}_Collection_{Year}_{Quarter}_{Interval}.html`
- TradingView alerts (Markdown): `exports/tv_alerts/{Year}/{Quarter}/{Collection}_Collection_{Year}_{Quarter}_{Interval}.md`
- AI recommendations:
  - Markdown: `exports/ai_reco/{Year}/{Quarter}/{Collection}_Collection_{Year}_{Quarter}_{Interval}.md`
  - HTML (dark Tailwind): same path with `.html` and a Download CSV link

Notes:
- Exporters are DB-backed (read best strategies); no HTML scraping.
- With multiple intervals in plan, filenames prefer `1d`. Pass `--interval 1d` to constrain both content and filenames.

## 🗄️ Data & Cache

- Split caching: the system maintains two layers for market data.
  - Full snapshot: stored when requesting provider periods like `--period max` (long TTL).
  - Recent overlay: normal runs cache the last ~90 days (short TTL).
  - Reads merge both, prefer recent on overlap, and auto‑extend when a request exceeds cached range.
- Fresh fetch: add `--no-cache` (alias: `--fresh`) to bypass cache reads and fetch from the provider. The result still writes through to cache.
- Coverage probe: before backtests, the CLI samples a few symbols with `period=max` and prefers the source with the most rows and earliest start for this run.

### Prefetching Collections (avoid rate limits)

Use the prefetch script to refresh data on a schedule (e.g., nightly recent overlay and weekly full snapshot):

```bash
# Full history snapshot (bonds)
docker compose run --rm quant \
  python scripts/prefetch_collection.py bonds --mode full --interval 1d

# Recent overlay (last 90 days)
docker compose run --rm quant \
  python scripts/prefetch_collection.py bonds --mode recent --interval 1d --recent-days 90
```

Example cron (runs at 01:30 local time):

```
30 1 * * * cd /path/to/quant-system && docker compose run --rm quant \
  python scripts/prefetch_collection.py bonds --mode recent --interval 1d --recent-days 90 >/dev/null 2>&1
```

### Optional Redis Overlay (advanced)

- For higher throughput, you can use Redis for the “recent” layer and keep full snapshots on disk.
- Pros: very fast hot reads, simple TTL eviction. Cons: extra service; volatile if not persisted.
- Suggested setup: run Redis via compose, store recent overlay (last 90 days) with TTL ~24–48h; keep full history on disk (gzip).
- Current repo ships with file‑based caching; Redis is an optional enhancement and can be added without breaking existing flows.

## 📚 Further Docs

- docs/pgadmin-and-performance.md — pgAdmin queries and performance tips
- docs/data-sources.md — supported providers and configuration
- docs/development.md — local dev, testing, and repo layout
- docs/docker.md — Docker specifics and mounts
- docs/features.md — feature overview and roadmap
- docs/cli-guide.md — CLI details and examples

## 🛠️ Troubleshooting

- Command name: use `docker compose` (or legacy `docker-compose`) consistently.
- Subcommand: it is `collection` (singular), not `collections`.
- Strategy discovery: ensure strategies are mounted at `/app/external_strategies` and set `STRATEGIES_PATH=/app/external_strategies` when running.
- Database URL: inside containers use `postgres:5432` (`DATABASE_URL=postgresql://quantuser:quantpass@postgres:5432/quant_system`). On the host, Postgres is published at `localhost:5433`.
- Initialize tables: if tables are missing, run:
  `docker compose run --rm quant python -c "from src.database.unified_models import create_tables; create_tables()"`
- Long runs/timeouts: backtests can take minutes to hours depending on strategies and symbols. Prefer `--log-level INFO` or `DEBUG` to monitor progress. Use `--dry-run` to validate plans quickly. Extra tips in docs/pgadmin-and-performance.md.
- Permissions/cache: ensure `cache/`, `exports/`, `logs/`, and `artifacts/` exist and are writable on the host (compose mounts them into the container).
- API limits: some data sources rate-limit; providing API keys in `.env` can reduce throttling.

## ⚠️ Disclaimer

This project is for educational and research purposes only. It does not constitute financial advice. Use at your own risk and always perform your own due diligence before making investment decisions.
