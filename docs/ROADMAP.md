# Future Roadmap

## Overview

This document outlines potential enhancements to evolve the system toward a production-grade, research and execution platform.

## Data & Ingestion

- API integrations: complete Polygon, Tiingo, Alpaca implementations with retries, paging, and symbol discovery.
- Exchange symbol discovery and liquidity filters (e.g., Bybit/Binance with volume thresholds) and universe builders.
- Corporate actions / splits / dividends normalization and corporate events feed.
- Multiple caching tiers: HTTP cache, Parquet cache, feature cache; retention policies and compaction.

## Backtesting & Optimization

- Alternative engines: adapter for backtesting.py with built-in optimizer; modular engine interface.
- Parameter search: Bayesian optimization (Optuna), random search, hyperband; early stopping.
- Walk-forward analysis and nested CV; rolling re-optimization; regime-aware parameter sets.
- Transaction cost models: tiered fees, borrow rates, shorting constraints, per-venue slippage and market impact.

## Risk & Analytics

- Advanced metrics: Calmar, Omega, Tail ratio, Pain index; probabilistic drawdown forecasts.
- Risk decomposition: factor models, sector/asset class exposure; Kelly sizing and volatility targeting.
- Scenario analysis: stress testing against historical shocks; Monte Carlo path sampling.

## Execution & Live

- API server: REST/GraphQL to query results, trigger runs, retrieve artifacts, and push signals.
- Streaming signal topics (Kafka/NATS) and order routing adapters (IBKR, CCXT exchanges).
- Scheduling: Airflow/Prefect orchestration; historical + daily incremental pipelines.
- TradingAgents integration: [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) for advanced agentic RL strategies.

## Reporting & UX

- Rich HTML reports: interactive charts (Plotly/Altair), equity curves, drawdown charts, and trade logs.
- Dashboard: lightweight UI (FastAPI + HTMX/Tailwind) to browse runs, compare strategies, download exports.
- Notifications: email/Slack alerts when new best models/params surpass thresholds.

## Infra & Quality

- Distributed compute: Ray/Dask for large param grids and asset universes.
- Cloud object storage for caches and artifacts; retention lifecycles.
- Secrets management with Vault/SOPS; per-environment configs.
- Test coverage >80% with unit + integration tests; golden files for reports.
- Pre-commit hooks for code, markdown, and schema validation; nightly CI e2e runs.
