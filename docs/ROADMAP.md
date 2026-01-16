# Roadmap

This roadmap highlights recently delivered capabilities and the next wave of work required to evolve the platform into a production-grade research and execution stack.

## Recently Delivered

- ✅ Core data connectors for YFinance, Polygon, Tiingo, Alpaca, Finnhub, TwelveData, AlphaVantage and CCXT, including caching and retry-aware HTTP clients.
- ✅ Migration to PyBroker for portfolio simulations, execution modelling, and parameter searches (grid + Optuna).
- ✅ Advanced run publications: HTML/Markdown/CSV exports, Prometheus metrics, manifest status reporting, Slack notifications, and FastAPI dashboard.
- ✅ CLI tooling improvements such as `manifest-status`, `package-run`, and the new `clean-cache` command for automated cache retention.
- ✅ Multi-exchange symbol discovery enhancements (merged results, liquidity annotations) to streamline universe construction.

## Next Focus Areas

### Data & Ingestion

- Harden provider adapters with richer observability (structured logging, rate-limit telemetry, retry metrics) — instrumentation landed, next step is capturing regression fixtures for each provider.
- Expand universe discovery: liquidity-screened symbol lists, configurable exclusions, and composite universes across equity/FX/crypto.
- Extend fundamentals coverage beyond yfinance (additional providers, macro data) now that yfinance splits/dividends/fundamentals snapshots are available.
- Incremental ingestion pipeline: CLI `ingest-data` covers manual refresh; extend to scheduled batches and provider-specific delta fetches.

### Backtesting & Optimisation

- Introduce walk-forward / rolling cross-validation pipelines with automatic parameter re-tuning.
- Enhance cost and execution modelling: tiered commissions, borrow fees, short-availability constraints, venue-specific slippage/impact curves.
- Add stochastic or bandit-style search strategies (random search, Hyperband) on top of the current grid and Optuna integrations.
- Evaluate optional event-driven engines (e.g., Freqtrade or vectorised alternatives) for crypto-specific workflows.

### Risk & Analytics

- Deliver factor and sector exposure breakdowns, volatility targeting, and Kelly sizing utilities on top of existing summary metrics.
- Build scenario analysis modules: stress tests against historical shocks, Monte Carlo path sampling, and drawdown probability dashboards.
- Surface richer post-run analytics (e.g., distribution charts, attribution) directly inside the dashboard UI and exported reports.

### Execution & Live Readiness

- Provide a REST/GraphQL API for triggering runs, querying results, and orchestrating backtests from external schedulers.
- Add streaming signal/output topics (Kafka/NATS) and order-routing adapters (IBKR, CCXT) for semi-automated/live deployment.
- Integrate with workflow schedulers (Airflow/Prefect) to support daily incremental updates and historical refresh jobs.
- Explore TradingAgents (RL-based strategies) interoperability for agentic portfolio logic.

### Reporting & UX

- Extend the FastAPI dashboard with interactive filtering, run comparisons, and embedded visualisations (Plotly/Altair) without leaving the app.
- Implement drill-downs for collection/strategy level stats, notification history, and cache health snapshots.
- Offer templated report bundles (PDF export, notebook summaries) for stakeholder-ready presentations.

### Developer Experience & Infrastructure

- Scale-out workloads via Ray/Dask for large asset universes and parameter grids.
- Support cloud storage backends (S3/GCS/Azure) for caches and reports with retention lifecycle policies.
- Integrate secrets management (Vault/SOPS) and environment-specific configuration overlays.
- Maintain >80% test coverage with targeted integration tests, golden report fixtures, and nightly CI runs.
- Continue strengthening automation: linting/formatting, schema validation, and smoke-test workflows in pre-commit and CI.
