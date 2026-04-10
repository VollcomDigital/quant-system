# Quant Monorepo — Phased Implementation Plan

## Status Legend

- [ ] Not started
- [~] In progress
- [x] Complete

---

## Current State Assessment

The codebase (`quant-system`) is a **backtesting-only** system with:

- **Data layer:** 8 provider connectors (yfinance, ccxt, alpaca, etc.) with Parquet caching
- **Backtest engine:** PyBroker-backed runner with Optuna param search
- **Reporting:** CSV, HTML, MD, TradingView, FastAPI dashboard
- **Strategies:** External discovery via `STRATEGIES_PATH`
- **Infra:** Docker + docker-compose, GitHub Actions CI, no Terraform/K8s
- **No OMS/EMS/gateways, no ML pipeline, no AI agents, no feature store, no HFT engine**

The target is a full **quant-monorepo** with 7 top-level modules and a dedicated
`trading_system/` subtree containing HFT, mid-freq, and shared gateway engines.

---

## Phase 0 — Scaffolding & Shared Foundations

**Goal:** Establish monorepo layout, shared library, and project-wide conventions
without breaking existing functionality.

### 0.1 Monorepo directory scaffold

- [ ] Create all top-level directories with `__init__.py` / `README.md`:
  ```
  quant-monorepo/
  ├── ai_agents/{alpha_researcher,code_reviewer,risk_monitor}/
  ├── data_platform/{connectors,pipelines,feature_store}/
  ├── alpha_research/{notebooks,factor_library,ml_models}/
  ├── backtest_engine/{simulator,market_mechanics,analytics}/
  ├── trading_system/{oms,ems,gateways}/
  ├── infrastructure/{terraform,kubernetes}/
  └── shared_lib/{math_utils,logging}/
  ```
- [ ] Create `trading_system/` sub-scaffold:
  ```
  trading_system/
  ├── hft_engine/{core,fpga,network,fast_inference}/
  ├── mid_freq_engine/{model_serving,portfolio_optimizer,execution_algos}/
  └── shared_gateways/{fix_engine,binary_protocols}/
  ```

### 0.2 `shared_lib/` — Core utilities

- [ ] Migrate existing `src/utils/` into `shared_lib/` as canonical location
- [ ] `shared_lib/math_utils/` — Polars/NumPy wrappers, Decimal-safe price arithmetic
- [ ] `shared_lib/logging/` — Structured JSON logger with OTel trace/span propagation
- [ ] Ensure backward-compatible re-export from `src/utils/` → `shared_lib/`

### 0.3 Project infrastructure updates

- [ ] Root `pyproject.toml` workspace or per-module build configs
- [ ] Root `Makefile` with module-scoped targets
- [ ] `.pre-commit-config.yaml` updated for new paths
- [ ] `ruff.toml` includes new modules
- [ ] Each module gets its own `README.md` with purpose + interface contract

---

## Phase 1 — Data Platform (Foundation Layer)

**Goal:** Extract and harden the data layer into a standalone module with
orchestration, lineage, and a feature store stub.

### 1.1 `data_platform/connectors/`

- [ ] Migrate `src/data/*_source.py` → `data_platform/connectors/`
- [ ] Add abstract `BaseConnector` protocol (async-ready, typed)
- [ ] Add FIX drop file connector stub
- [ ] Add generic web scraper connector skeleton (respx/httpx)
- [ ] Unified rate limiter + circuit breaker (from existing `ratelimiter.py`)
- [ ] Symbol mapper as shared service

### 1.2 `data_platform/pipelines/`

- [ ] Airflow/Prefect DAG skeletons for daily data ingestion
- [ ] Task: fetch → validate → store (Parquet/object-store)
- [ ] Task: data quality checks (Pandera schemas)
- [ ] Pipeline config YAML convention (source, symbols, schedule, destination)

### 1.3 `data_platform/feature_store/`

- [ ] Feature definition schema (Pydantic models)
- [ ] Offline feature materialization (Polars lazy → Parquet)
- [ ] Feature registry (YAML manifest of all defined features)
- [ ] Feast/Hopsworks integration stub (adapter pattern)

---

## Phase 2 — Backtest Engine (Upgrade Existing)

**Goal:** Promote the current `src/backtest/` into a first-class engine module
with event-driven simulation, pluggable market mechanics, and analytics.

### 2.1 `backtest_engine/simulator/`

- [ ] Extract `src/backtest/runner.py` logic → `backtest_engine/simulator/`
- [ ] Add event-driven simulation core (OrderEvent, FillEvent, BarEvent)
- [ ] Strategy adapter interface (wraps existing `BaseStrategy`)
- [ ] Deterministic replay mode (seeded randomness for fill simulation)

### 2.2 `backtest_engine/market_mechanics/`

- [ ] Slippage models: fixed, proportional, volume-impact, Almgren-Chriss
- [ ] Fee models: tiered maker/taker, funding rates (crypto perps)
- [ ] Market impact model interface + simple square-root implementation
- [ ] Config-driven model selection (YAML)

### 2.3 `backtest_engine/analytics/`

- [ ] Migrate `src/backtest/metrics.py` → `analytics/`
- [ ] QuantStats / pyfolio tear-sheet generation
- [ ] Drawdown decomposition, rolling Sharpe, regime-conditional stats
- [ ] Benchmark comparison module
- [ ] Exporters: PDF, HTML, JSON for dashboard consumption

---

## Phase 3 — Alpha Research

**Goal:** Structured research environment with a factor library and ML training
pipelines.

### 3.1 `alpha_research/notebooks/`

- [ ] Jupyter/Quarto notebook templates (signal research, regime analysis)
- [ ] Notebook linting rules (no secrets, no hardcoded paths)
- [ ] `nbstripout` pre-commit hook to prevent output commits

### 3.2 `alpha_research/factor_library/`

- [ ] Factor base class: `compute(df) → Series`, metadata, IS/OOS split
- [ ] Turnover and decay analysis utilities
- [ ] Cross-validation framework (purged k-fold, walk-forward)
- [ ] Factor correlation / orthogonalization tools
- [ ] Registry: YAML catalog of validated factors with stats

### 3.3 `alpha_research/ml_models/`

- [ ] Training pipeline scaffolding (PyTorch, XGBoost)
- [ ] Ray Train / Ray Tune integration for distributed HPO
- [ ] Model registry (MLflow or simple versioned artifacts)
- [ ] Feature importance / SHAP analysis module
- [ ] Anti-lookahead-bias guardrails (strict train/val/test chronological split)

---

## Phase 4 — Trading System (Core Execution)

**Goal:** Build the OMS, EMS, and exchange gateways that take a signal and
produce fills.

### 4.1 `trading_system/oms/`

- [ ] Portfolio state manager (positions, cash, margin, PnL)
- [ ] Order lifecycle FSM (New → Pending → PartialFill → Filled → Cancelled)
- [ ] Risk gate: pre-trade checks (exposure, drawdown, concentration)
- [ ] Kill switch: global drawdown halts all trading
- [ ] Audit log: immutable order/fill journal (append-only Parquet or DB)

### 4.2 `trading_system/ems/`

- [ ] Execution algo base class + TWAP implementation
- [ ] VWAP implementation
- [ ] Smart order router (split across venues by liquidity)
- [ ] Slippage/spread sanity check before order submission
- [ ] Cooldown / rate-limit enforcement between orders

### 4.3 `trading_system/gateways/`

- [ ] Gateway protocol (abstract: connect, submit, cancel, stream)
- [ ] CCXT gateway (crypto exchanges)
- [ ] REST/WebSocket gateway skeleton for equities brokers
- [ ] FIX protocol gateway stub (QuickFIX adapter pattern)
- [ ] Paper-trade gateway (same feed, writes to local DB)

---

## Phase 5 — Trading System / HFT & Mid-Freq Engines

**Goal:** Specialized execution engines for different latency regimes.

### 5.1 `trading_system/hft_engine/core/`

- [ ] Lock-free ring buffer (C++ or Rust, Python bindings via PyO3/pybind11)
- [ ] SPSC queue for order flow
- [ ] Nanosecond timestamping utilities
- [ ] Memory-mapped shared state for cross-process communication

### 5.2 `trading_system/hft_engine/fpga/`

- [ ] Directory scaffold + documentation for Verilog/VHDL modules
- [ ] Build instructions for Xilinx/Intel toolchains
- [ ] Stub market data parser (ITCH protocol in HDL pseudocode)

### 5.3 `trading_system/hft_engine/network/`

- [ ] Kernel-bypass networking documentation (DPDK / OpenOnload)
- [ ] Raw socket receiver skeleton (C/Rust)
- [ ] Multicast join utilities for market data feeds

### 5.4 `trading_system/hft_engine/fast_inference/`

- [ ] ONNX Runtime C++ wrapper for signal models
- [ ] TensorRT optimization pipeline stub
- [ ] Batch inference vs. single-tick inference interface

### 5.5 `trading_system/mid_freq_engine/model_serving/`

- [ ] Triton Inference Server config generator
- [ ] gRPC client for model predictions
- [ ] Model versioning and A/B routing

### 5.6 `trading_system/mid_freq_engine/portfolio_optimizer/`

- [ ] Mean-variance optimizer (cvxpy)
- [ ] Risk parity, max Sharpe, min variance objectives
- [ ] Constraint framework (sector, turnover, position limits)
- [ ] Rebalance scheduler

### 5.7 `trading_system/mid_freq_engine/execution_algos/`

- [ ] TWAP/VWAP algo (reuse from EMS, adapted for mid-freq)
- [ ] IS (Implementation Shortfall) benchmark tracker
- [ ] Participation-rate algo

### 5.8 `trading_system/shared_gateways/fix_engine/`

- [ ] QuickFIX/J or quickfix-rs adapter
- [ ] FIX 4.2/4.4/5.0 session management
- [ ] NewOrderSingle / ExecutionReport message builders

### 5.9 `trading_system/shared_gateways/binary_protocols/`

- [ ] ITCH 5.0 market data parser (Rust)
- [ ] OUCH order entry protocol builder (Rust)
- [ ] Shared C-struct definitions for zero-copy parsing

---

## Phase 6 — AI Agents

**Goal:** Autonomous workers for research, code review, and risk monitoring.

### 6.1 `ai_agents/alpha_researcher/`

- [ ] ArXiv paper scraper + summarizer (LLM-driven)
- [ ] Factor idea generator from paper abstracts
- [ ] Auto-backtest launcher for generated ideas
- [ ] Human-in-the-loop approval workflow

### 6.2 `ai_agents/code_reviewer/`

- [ ] GitHub PR webhook listener
- [ ] Look-ahead bias detector (AST analysis of data access patterns)
- [ ] Strategy code quality checker (complexity, test coverage)
- [ ] Auto-comment on PR with findings

### 6.3 `ai_agents/risk_monitor/`

- [ ] Live anomaly log consumer (structured log parser)
- [ ] Regime detection integration (from alpha_research)
- [ ] Alert escalation: Slack / PagerDuty webhook
- [ ] Auto-reduce position sizing when anomalies spike

---

## Phase 7 — Infrastructure as Code

**Goal:** Production-grade deployment with Terraform and Kubernetes.

### 7.1 `infrastructure/terraform/`

- [ ] AWS / GCP modules for compute (EC2/GCE), networking, storage (S3/GCS)
- [ ] Database provisioning (TimescaleDB / ClickHouse for tick data)
- [ ] Secrets management (AWS Secrets Manager / GCP Secret Manager)
- [ ] State backend (S3 + DynamoDB locking / GCS)

### 7.2 `infrastructure/kubernetes/`

- [ ] Helm charts or Kustomize overlays for each module
- [ ] Argo CD application manifests for GitOps
- [ ] HPA (Horizontal Pod Autoscaler) configs for model serving
- [ ] Network policies, pod security standards
- [ ] Monitoring stack: Prometheus, Grafana, Loki, Tempo

---

## Phase 8 — Integration & Hardening

**Goal:** End-to-end wiring, testing, observability, and production readiness.

### 8.1 Cross-module integration

- [ ] Signal → OMS → EMS → Gateway pipeline (paper mode)
- [ ] Feature store → ML model → signal generation pipeline
- [ ] Backtest → walk-forward → paper → live promotion workflow
- [ ] Dashboard: unified view across backtest, paper, live

### 8.2 Observability

- [ ] OpenTelemetry instrumentation across all Python modules
- [ ] Grafana dashboards: strategy PnL, system health, latency
- [ ] Structured logging with trace propagation (shared_lib/logging)
- [ ] Alerting rules (Prometheus → AlertManager → Slack)

### 8.3 Testing & CI hardening

- [ ] Integration test suite per module boundary
- [ ] Contract tests between modules (Pact or schema-based)
- [ ] CI matrix: unit → integration → e2e (paper trade)
- [ ] Coverage gates per module (80% minimum)
- [ ] Security scanning: CodeQL, Trivy, Dependabot

### 8.4 Documentation

- [ ] Architecture decision records (ADRs) for major choices
- [ ] API contracts (OpenAPI for REST, protobuf for gRPC)
- [ ] Runbook for live trading operations
- [ ] Onboarding guide for new quant developers

---

## Dependency Graph (Critical Path)

```
Phase 0 (Scaffold + shared_lib)
    │
    ├──→ Phase 1 (Data Platform)
    │        │
    │        ├──→ Phase 2 (Backtest Engine) ──→ Phase 3 (Alpha Research)
    │        │                                        │
    │        └──→ Phase 4 (Trading System Core) ◄─────┘
    │                     │
    │                     └──→ Phase 5 (HFT & Mid-Freq)
    │
    ├──→ Phase 6 (AI Agents) — can start after Phase 1 + Phase 3
    │
    └──→ Phase 7 (Infrastructure) — can start after Phase 0
              │
              └──→ Phase 8 (Integration) — requires all above
```

### Parallelization opportunities

- **Phase 7** (IaC) is independent and can proceed in parallel with Phases 1–5
- **Phase 6** (AI Agents) can start once Phase 1 (data) and Phase 3 (research) deliver interfaces
- **Phase 5** (HFT/Mid-Freq) can start once Phase 4 (core trading) defines its protocols
- **Phase 2** and **Phase 4** can overlap — they share the market mechanics models

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing backtest functionality during migration | HIGH | Phase 0 backward-compat re-exports; integration tests first |
| Scope creep in HFT/FPGA (Phase 5) | MEDIUM | Keep as stubs/docs until real hardware available |
| AI agent hallucination in live risk monitoring | HIGH | Human-in-the-loop mandatory; agent only suggests, never executes |
| Feature store schema drift | MEDIUM | Pandera schemas + versioned registry |
| C++/Rust FFI complexity for HFT core | MEDIUM | Start with pure-Python prototypes, optimize later |
