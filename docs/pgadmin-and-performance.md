# DB Inspection (pgAdmin) and Run Performance Tips

## pgAdmin: Connect and Inspect

- Login: open http://localhost:5050 and use `PGADMIN_DEFAULT_EMAIL` / `PGADMIN_DEFAULT_PASSWORD` from `.env`.
- Register server (first time):
  - Name: `quant-local`
  - Hostname/address: `postgres`
  - Port: `5432`
  - Maintenance DB: `quant_system`
  - Username: `quantuser`
  - Password: `quantpass`

### Handy Queries

- Recent runs (most recent first):
```sql
SELECT run_id, started_at_utc, action, collection_ref,
       strategies_mode, intervals_mode, target_metric, period_mode,
       status, plan_hash
FROM runs
ORDER BY started_at_utc DESC
LIMIT 50;
```

- Find a run by plan_hash:
```sql
SELECT *
FROM runs
WHERE plan_hash = '<paste-plan-hash>';
```

- Count backtest results per run:
```sql
SELECT run_id, COUNT(*) AS results
FROM backtest_results
GROUP BY run_id
ORDER BY results DESC;
```

- Best strategies for 1d timeframe (top by Sortino):
```sql
SELECT symbol, timeframe, strategy,
       COALESCE(sortino_ratio::float, 0) AS sortino_ratio,
       COALESCE(total_return::float, 0) AS total_return,
       COALESCE(max_drawdown::float, 0) AS max_drawdown,
       updated_at
FROM best_strategies
WHERE timeframe = '1d'
ORDER BY sortino_ratio DESC
LIMIT 50;
```

- Latest results for a symbol (e.g., TLT):
```sql
SELECT symbol, strategy, interval, start_at_utc, end_at_utc, metrics, engine_ctx
FROM backtest_results
WHERE symbol = 'TLT'
ORDER BY end_at_utc DESC NULLS LAST
LIMIT 5;
```

## Speeding Up Runs

- Limit strategies: pass `--strategies RSI,BollingerBands,Breakout` instead of `all`.
- Limit symbols: create a small collection JSON (3–5 symbols) for iteration.
- Fix interval: keep `--interval 1d` during development.
- Concurrency: use `--max-workers 4` (or higher if CPU allows). Monitor memory.
- Validate plan first: add `--dry-run` to print the manifest before running.
- Re-run same plan: use `--force` if you need to execute a plan with the same `plan_hash` again.
- Data/API constraints: provide API keys in `.env` to reduce throttling and widen history where providers allow.

## Paths & Artifacts

- Artifacts: `artifacts/run_<timestamp>/` (manifest, summaries, exports if enabled).
- Exports: `exports/` (CSV, HTML reports, TradingView), organized by quarter in some flows.

## Connections (psql)

- Inside container: `DATABASE_URL=postgresql://quantuser:quantpass@postgres:5432/quant_system`
- From host: `psql postgresql://quantuser:quantpass@localhost:5433/quant_system`
