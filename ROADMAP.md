# Competitor Landscape Matrix

## Tier-1 benchmark matrix

| Benchmark | What they are best at | Flagship features | Why it matters for this repo | Gap in the current codebase |
| --- | --- | --- | --- | --- |
| **Palantir Foundry** | Operational data platforms | Ontology/object model, workflow lineage, governed actions, semantic control plane, audited operations | Shows how quant research becomes an operational system instead of a folder of reports | The repo has run artifacts and manifests, but no semantic layer over symbols, strategies, providers, or promotion decisions |
| **Snowflake** | Reproducible analytical infrastructure | Time Travel, zero-copy cloning, secure data sharing, Snowpark, cost governance | Sets the bar for reproducible experiments, data sharing, and low-friction collaboration across teams | The repo has Parquet plus SQLite caches, but no experiment cloning, shared data product layer, or cost attribution per run |
| **SigNoz** | Unified observability | OpenTelemetry-native traces, logs, metrics, exception correlation, alerting, dashboards | Proves that platform reliability must be observable end to end, not only logged after the fact | The repo writes structured logs and `metrics.prom`, but has no distributed tracing, live alerting, or SLO-style runtime telemetry |
| **NautilusTrader** | Institutional-grade trading engine | Rust core, deterministic event-driven backtesting, nanosecond resolution, research-to-live parity, `ParquetDataCatalog` | Defines the benchmark for execution realism and a single strategy path from research to live deployment | The repo is strong in batch PyBroker backtests, but has no event-driven engine, no live execution path, and no OMS/RMS abstraction |
| **Tauric Research / TradingAgents** | Autonomous quant research | Multi-agent analyst swarm, bullish/bearish debate, trader agent, portfolio/risk agent, model-provider abstraction | Shows how research can become a self-improving closed loop instead of a manually curated grid search | The repo can execute configured searches, but it cannot generate hypotheses, critique results, or maintain an institutional research memory |

## Master feature inventory from the market

### Data and control plane

- Semantic object model over providers, symbols, strategies, runs, and approvals.
- Reproducible data snapshots with point-in-time recovery and cloning.
- Secure sharing of curated datasets, features, and approved research artifacts.
- Provider health scoring, automated failover, and data confidence ranking.

### Research and backtesting

- Event-driven simulation with research-to-live parity.
- Walk-forward validation, promotion gates, and experiment registry.
- Portfolio construction, risk budgeting, and regime-aware allocation.
- Rich financial libraries for calendars, pricing, optimization, and alternative datasets from the `awesome-quant` ecosystem.

### Execution and operations

- OMS/RMS separation, reconciliation, kill-switches, and policy-based approvals.
- Execution-quality telemetry feeding model selection and de-weighting.
- Real-time incident detection, remediation playbooks, and audit trails.

### Observability and platform engineering

- Unified logs, metrics, traces, and alerts across the full research-to-execution path.
- Cost and capacity attribution per run, provider, and strategy family.
- Multi-user governance, dashboard access controls, and release hardening.

### Autonomous AI and self-improvement

- Research agents that propose, test, challenge, and document new hypotheses.
- Alpha-decay monitors that re-rank or quarantine degrading strategies.
- Self-healing data pipelines and anomaly remediators with policy constraints.
- Experiment memory that learns which ideas survive across regimes and why.

## Cross-market white space

The largest white space across the market is not another backtester; it is a
**governed autonomous research-to-execution loop** that combines enterprise
lineage, event-driven trading realism, and self-improving agentic research in
one platform.

1. **Policy-constrained autonomous quant ops**: most platforms are either
   enterprise-governed or autonomous, but rarely both, leaving room for an
   explainable agent layer that can act without breaking risk policy.
2. **Self-healing multi-provider data fabric for research**: quant teams still
   lose alpha to brittle provider pipelines, delayed backfills, and silent
   schema drift.
3. **Continuous alpha lifecycle management**: the market lacks a strong default
   for measuring alpha decay, de-weighting weak signals, and promoting only
   robust replacements.
4. **Experiment memory with portfolio context**: most systems store results,
   but few learn which strategies work in which regimes and with which peers.
5. **Unified semantic operating model for research and execution**: there is
   still room for a quant-specific ontology that links thesis, data, model,
   backtest, approval, execution, and post-trade outcomes.

# Autonomous AI Vision

## Self-improvement matrix

| Capability | Learning loop | Repo starting point | Missing system | Guardrails | ROI / alpha justification |
| --- | --- | --- | --- | --- | --- |
| **Self-healing data pipeline** | Learn provider reliability, detect stale or incomplete bars, and automatically reroute to the best source | Multi-provider sources plus Parquet cache in `src/data/*` and `src/data/cache.py` | Provider scoring, schema drift checks, confidence metadata, failover orchestration | Human-visible confidence score, source allow-list, immutable raw cache | Fewer bad datasets means fewer false positives and less alpha leakage from silent data quality failures. |
| **Continuous alpha-decay monitor** | Track live or paper outcomes versus backtest expectations and quarantine degrading strategies | Batch metrics and run summaries already exist in `src/backtest/metrics.py` and `summary.json` | Rolling drift monitor, alert thresholds, automatic de-weighting policy | Only reduce exposure automatically; never increase without approval | The fastest way to improve Sharpe is to remove decaying strategies before they compound losses. |
| **Autonomous anomaly remediation** | Detect abnormal fetch latency, cache miss spikes, or execution issues and trigger playbooks | Structured logging in `src/utils/telemetry.py` and run artifacts under `reports/` | Incident classifier, remediation policies, alert routing, acknowledgement workflow | Action catalog must be deterministic and reversible | Shorter incident duration directly preserves signal freshness and execution quality. |
| **Tauric-style research swarm** | Generate hypotheses, debate long and short cases, and propose experiment specs | External strategy loading in `src/strategies/registry.py` provides a pluggable research entry point | Agent orchestration, thesis memory, evaluator agents, experiment writer | Research agents stay in paper mode and cannot route orders | Automating idea generation expands the search frontier without linearly scaling researcher headcount. |
| **Execution-quality feedback optimizer** | Learn which venues, slippage settings, and order tactics preserve edge | Fee and slippage knobs exist in `src/backtest/runner.py` | Fill analytics, venue telemetry, adverse-selection monitor, policy updates | RMS owns limits; optimizer can only tune within approved ranges | Better execution converts existing gross alpha into realized alpha with no new model risk. |
| **Regime-aware allocator** | Reweight strategies as volatility, liquidity, and correlation structures change | Collection-level batch runs exist, but no portfolio layer | Regime detector, capital allocator, covariance and liquidity models | Exposure caps and drawdown budgets override all model suggestions | Matching strategy mix to regime improves capital efficiency and reduces hidden concentration. |
| **Experiment memory and promotion engine** | Learn which parameter sets survive out of sample and across regimes | Results cache in `src/backtest/results_cache.py` stores prior evaluations | Experiment registry, walk-forward scoring, promotion states, rollback history | Promotion must require evidence from paper or forward validation | Institutional memory prevents the team from repeatedly paying research cost for already-failed ideas. |
| **Policy-constrained RMS/OMS tuner** | Adapt risk thresholds and routing rules based on realized volatility and fill outcomes | No OMS/RMS exists today | Order state machine, risk policy engine, post-trade learner | Hard kill-switch, approval workflow, and immutable policy audit log | Adaptive controls reduce avoidable slippage and over-sizing while keeping risk inside predefined envelopes. |

# Prioritized Roadmap

## Codebase gap mapping

| Current capability | Evidence in the repo | Gap versus benchmarks and `awesome-quant` | Strategic impact |
| --- | --- | --- | --- |
| **Config-driven batch backtesting** | `src/main.py`, `src/backtest/runner.py` | Strong baseline, but still batch-oriented and not event-driven like NautilusTrader | Good for research throughput, insufficient for research-to-live parity |
| **Multi-provider data ingestion plus Parquet cache** | `src/data/*`, `src/data/cache.py` | No provider scoring, failover, unified metadata catalog, or complete CLI parity across all sources | Data quality remains a hidden operational risk |
| **Optuna and grid search** | `src/config.py`, `src/backtest/runner.py` | No walk-forward validation, purged CV, experiment registry, or promotion gates from the `awesome-quant` research stack | Optimization can overfit without a formal promotion process |
| **Results cache and reporting** | `src/backtest/results_cache.py`, `src/reporting/*` | No semantic lineage layer like Palantir or reproducibility features like Snowflake cloning | Insights are stored, but institutional memory is weak |
| **FastAPI dashboard** | `src/dashboard/server.py` | No auth, RBAC, team workflow, or audit model | Useful internally, risky for broader operational use |
| **Structured logging and metrics file** | `src/utils/telemetry.py`, `metrics.prom` | No OpenTelemetry, trace correlation, or real-time alerts like SigNoz | Troubleshooting remains manual and reactive |
| **Notifications** | `src/reporting/notifications.py` | Slack only, with no workflow routing, escalation, or execution-aware event taxonomy | The system cannot serve as an operational cockpit |
| **External strategy discovery** | `src/strategies/base.py`, `src/strategies/registry.py` | No Tauric-style research agents, thesis memory, or model-governance layer | The platform executes ideas but does not help create or rank them |
| **No OMS, RMS, paper/live split, or portfolio engine** | Missing across `src/` | This is the single largest gap versus institutional quant platforms and open-source execution engines | The repo is a research platform, not yet a deployable trading operating system |
| **Limited `awesome-quant` surface area** | Current focus is PyBroker, data connectors, dashboarding | Missing exchange calendars, portfolio optimization, pricing stack, richer time-series tooling, and vectorized research lanes | The product is narrower than the broader quant ecosystem it could leverage |

## Must Have

| Feature | Priority | Why it is missing today | ROI / alpha justification |
| --- | --- | --- | --- |
| **Paper/live isolation plus OMS/RMS core** | **Very High** | There is no execution runtime, no order state machine, and no risk-first boundary between research and trading | This is the minimum required to deploy capital safely without turning research mistakes into live losses. |
| **NautilusTrader-aligned execution abstraction** | **Very High** | The platform is built around PyBroker batch simulations and lacks event-driven research-to-live parity | A unified execution abstraction reduces model-to-production drift and protects realized PnL from implementation mismatches. |
| **Walk-forward validation and promotion gates** | **Very High** | Optuna and grid search exist, but there is no formal out-of-sample promotion workflow | Better promotion discipline raises the hit rate of strategies that survive first contact with the market. |
| **Portfolio optimizer and regime-aware allocator** | **High** | Results are still evaluated mostly per strategy or symbol, not as a correlated capital book | Portfolio-level capital allocation usually improves risk-adjusted returns faster than adding more raw strategies. |
| **Execution-quality feedback loop** | **High** | Fees and slippage are static inputs, not learned from actual paper or live outcomes | Capturing and minimizing slippage converts existing forecast edge into realized alpha at low research cost. |
| **OpenTelemetry observability baseline** | **High** | `src/utils/telemetry.py` emits logs only, and `metrics.prom` is an offline artifact | Faster root-cause analysis cuts downtime, protects data freshness, and prevents silent performance degradation. |
| **Dashboard auth, audit, and experiment registry** | **High** | The dashboard exposes run information but has no operator controls or approval model | Governance increases trust, which is required before teams can rely on the platform for production decisions. |
| **Unified data control plane with provider failover** | **High** | Multi-source ingestion exists, but source trust, failover, and point-in-time confidence do not | Better data quality improves every downstream strategy without requiring new alpha models. |

## Nice to Have

| Feature | Priority | Why it is missing today | ROI / alpha justification |
| --- | --- | --- | --- |
| **Alpha-decay monitor with auto de-weighting** | **Medium** | The repo computes backtest metrics but does not manage strategy half-life after deployment | Removing decaying strategies early protects portfolio Sharpe more reliably than constantly adding new ones. |
| **Monte Carlo and scenario engine** | **Medium** | There is no explicit stress framework for returns, liquidity, latency, or spread shocks | Stress testing lowers tail risk and improves capital allocation discipline before losses become real. |
| **Vectorized research lane from the `awesome-quant` stack** | **Medium** | Research still centers on PyBroker workflows rather than a fast exploratory sweep engine | Faster exploration increases the number of viable ideas tested per unit of engineering time. |
| **Exchange calendars, business-day logic, and market-microstructure utilities** | **Medium** | The platform fetches bars, but it lacks a strong exchange-calendar and session-control layer | Better session alignment reduces subtle backtest errors that can distort edge estimates. |
| **Alternative data and feature store layer** | **Medium** | The repo supports prices well, but fundamentals, news, sentiment, and custom factors are not unified | More diverse orthogonal signals create a wider alpha surface than price-only strategies can provide. |
| **Multi-channel operational notifications** | **Low** | `src/reporting/notifications.py` only supports Slack-style alerts today | Better delivery and escalation ensures humans react quickly when the platform detects alpha or operational risk. |
| **Research notebook to thesis registry bridge** | **Low** | Insights remain scattered across reports and external strategy repos | Capturing thesis-to-result lineage compounds team learning and reduces duplicated research effort. |

## Optional

| Feature | Priority | Why it is missing today | ROI / alpha justification |
| --- | --- | --- | --- |
| **Tauric-style research-agent swarm** | **Low** | There is no agentic orchestration for technical, news, sentiment, and fundamentals debate | Autonomous idea generation can expand the alpha frontier once validation and governance are already strong. |
| **Self-healing anomaly remediation agent** | **Low** | The platform can report errors, but it cannot yet diagnose or correct them | Recovering faster from incidents preserves opportunity set and reduces operator load. |
| **Options, rates, and pricing-library expansion from `awesome-quant`** | **Lowest** | The current stack is bar-based and not yet a general pricing laboratory | Broader instrument coverage opens new strategy classes and diversifies the alpha mix. |
| **Smart order routing and multi-venue execution** | **Lowest** | There is no venue-aware order-routing layer because live execution is not yet present | Smart routing matters once the platform already trades enough size for venue choice to move realized alpha. |
| **Enterprise ontology and decision graph** | **Lowest** | The repo stores outputs, but not a full semantic graph linking ideas, approvals, trades, and outcomes | This adds durable institutional memory and explainability, but only after execution and governance are mature. |
| **Meta-learning allocator** | **Lowest** | There is no portfolio layer yet, so an allocator-of-allocators would be premature | Meta-learning becomes valuable only after the platform has a rich set of validated strategies and portfolio telemetry. |

## Strict execution order

1. Build the **trust boundary** first: paper/live isolation, OMS/RMS, auth, and audit.
2. Add **research discipline** second: walk-forward validation, promotion gates, experiment registry, and data control plane.
3. Add **capital intelligence** third: portfolio optimization, regime detection, and execution-quality feedback.
4. Add **platform reliability** fourth: OpenTelemetry, live alerting, self-healing data reliability, and scenario testing.
5. Add **autonomous research and self-improvement** fifth: alpha-decay agents, research swarms, and policy-constrained adaptive tuning.

## Recommended next implementation tranche

If the goal is to maximize institutional readiness with the fewest moving parts,
the next tranche should be:

1. **OMS/RMS plus paper/live separation**
2. **Walk-forward validation and promotion gates**
3. **OpenTelemetry observability baseline**
4. **Portfolio optimizer plus regime-aware allocator**
5. **Provider failover and data confidence layer**

That sequence converts the repository from a capable backtest runner into a
credible quant operating system with a clear path toward NautilusTrader-style
execution realism and Tauric-style autonomous research on top.

