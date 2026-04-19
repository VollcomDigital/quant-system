# Quant Monorepo Migration Plan

## Objective

Evolve the current single-package backtesting application into a staged monorepo that can support:

- research and factor discovery
- data ingestion and feature management
- a reusable backtest engine
- autonomous AI agents
- a web control plane for management, reporting, approvals, and execution oversight
- a production trading system split between mid-frequency and HFT concerns
- infrastructure and deployment automation

## Architectural Pillars to Preserve

The target implementation should explicitly preserve these design rules throughout all phases:

- AI agents are treated as scoped programmatic workers, not ad hoc LLM wrappers.
- The data platform is immutable, versioned, and reproducible.
- The backtest engine remains custom-code-first for market-structure realism.
- The trading system is polyglot, with Python for research and native code for latency-critical execution.
- CI/CD and MLOps are first-class guardrails, especially for factor promotion and model deployment.
- TradFi broker execution and Web3 protocol execution are treated as separate gateway paradigms with different state, security, and failure models.

## Execution Security and Custody Principles

The target implementation should explicitly enforce these execution-security constraints:

- Private keys must never live in source control, application code, or plain environment variables.
- Programmatic on-chain signing should default to cloud HSM/KMS-backed signing flows.
- Treasury capital should not sit in a bot-controlled hot EOA when a smart-contract vault or Safe-based control plane is available.
- The default crypto operating model should separate:
  - execution signer permissions
  - treasury custody
  - human withdrawal approval
- MPC custodians such as Fireblocks or Fordefi should be treated as optional institutional deployment tiers, not the only supported design.
- AI-driven execution must be bounded by hard kill-switches, model-drift controls, and human-escalation paths.

## Multi-Layer Circuit Breaker Principle

The target implementation should explicitly enforce a multi-layered Kill Switch architecture so catastrophic AI behavior is intercepted before, during, and after execution:

- Layer 1: AI/model guardrails before order formulation
- Layer 2: deterministic pre-trade risk engine before execution
- Layer 3: automated panic-button workflow during execution incidents
- Layer 4: DeFi-specific pause and allowance-revocation controls for on-chain risk
- Layer 5: out-of-band infrastructure isolation that can revoke runtime permissions even if the main system is unresponsive

## Workflow Orchestration Principle

The default workflow orchestrator for this stack should be Apache Airflow because it best fits:

- mixed batch ETL and dependency-heavy DAGs
- scheduled TradFi and on-chain ingestion workloads
- explicit operational visibility for a multi-team platform
- long-running backfills and replay jobs

Prefect and Dagster can still be evaluated for specific subdomains, but the implementation plan should assume Airflow as the primary orchestration baseline unless an ADR formally replaces it.

## Two-Speed Architecture Principle

The monorepo should keep a unified research and data foundation while enforcing two separate execution paths:

- a fast path for HFT, with deterministic native execution and co-located infrastructure
- a smart path for mid-frequency strategies, with cloud-native model serving and execution scheduling

This means:

- `data_platform/`, `alpha_research/`, `backtest_engine/`, and `shared_lib/` remain shared
- `trading_system/hft_engine/` and `trading_system/mid_freq_engine/` diverge by latency, runtime, and infrastructure
- `trading_system/shared_gateways/` provides the shared market connectivity contracts used by both paths

## Current-to-Target Mapping

The current repository already contains the seeds of several target domains:

- `src/data/*` -> `data_platform/connectors`
- `src/backtest/*` -> `backtest_engine/simulator` and `backtest_engine/analytics`
- `src/dashboard/*` -> `web_control_plane/backend` initial compatibility shell
- `src/reporting/*` -> `backtest_engine/analytics` and shared reporting interfaces
- `src/utils/telemetry.py` -> `shared_lib/logging`
- `src/config.py` -> shared contracts/config packages
- `src/strategies/*` -> `alpha_research/factor_library` bootstrap interfaces

The largest missing capabilities are:

- a true monorepo package layout with internal contracts
- a data platform with orchestration and feature-store abstractions
- research training and model-serving pipelines
- a first-class web application for operator workflows and approvals
- OMS/EMS separation for live trading
- agent runtime and review/risk agents
- low-latency gateway/HFT foundations
- infrastructure-as-code and platform deployment boundaries

## Upstream-Inspired Capability Adoption Strategy

The roadmap should incorporate ideas from the following repositories without allowing any of them to become the architectural spine of the platform:

- `shiyu-coder/Kronos`
- `MemPalace/mempalace`
- `HKUDS/Vibe-Trading`
- `rtk-ai/rtk`
- `NVIDIA-AI-Blueprints/quantitative-portfolio-optimization`
- `AI4Finance-Foundation/FinRL`
- `666ghj/MiroFish`

### Adoption Model

Use a **hybrid adoption model** with a strong bias toward **pattern-level adoption first** and **selective integration only at clean adapter boundaries**.

This means:

- internal packages own core contracts, orchestration, control planes, and risk boundaries
- upstream repositories are treated as architecture donors, bounded providers, or external operator tools
- direct integration is allowed only where the upstream capability is already naturally artifact- or service-shaped
- execution, custody, OMS, RMS, and treasury boundaries remain internal and deterministic
- the web application is an operator surface and must not bypass backend enforcement for approvals, OMS, RMS, gateways, or custody controls

### Repository Classification

#### `HKUDS/Vibe-Trading`

- primary architecture donor
- pattern-level adoption only
- borrow:
  - loader registries and fallback patterns
  - composite backtest orchestration concepts
  - robustness validation and research-to-report automation
- avoid:
  - transplanting its full agent stack as the core runtime

#### `shiyu-coder/Kronos`

- bounded model provider
- selective integration only through an internal forecasting adapter
- use for:
  - cacheable prediction features from OHLCV windows
  - batch forecasting across multiple assets
- land behind:
  - `alpha_research/ml_models/providers/`
  - `shared_lib` prediction contracts
  - `data_platform/feature_store/` persistence

#### `MemPalace/mempalace`

- research-memory subsystem donor
- pattern adoption first, optional backend integration later
- use for:
  - factor hypotheses
  - experiment rationale
  - failed trial history
  - retrieval context for researcher and reviewer agents

#### `NVIDIA-AI-Blueprints/quantitative-portfolio-optimization`

- portfolio optimization donor with optional accelerator backend
- pattern adoption first, optional GPU backend later
- use for:
  - Mean-CVaR and constrained allocation workflows
  - portfolio construction after alpha generation and before execution

#### `AI4Finance-Foundation/FinRL`

- RL research-pattern donor
- pattern-level adoption only
- use for:
  - RL environment design
  - train/test/trade workflow patterns
  - benchmark tasks for RL-specific strategy tracks

#### `rtk-ai/rtk`

- operator and agent ergonomics tool
- external tooling only
- use for:
  - compact CI/test/backtest output
  - failure summarization for human and agent review
- do not integrate into runtime or execution paths

#### `666ghj/MiroFish`

- scenario-simulation idea donor
- concept mining only
- use for:
  - late-stage narrative stress testing and what-if simulations
- do not integrate into core backtesting truth

### Cross-Phase Capability Priorities

The upstream-inspired roadmap should be layered onto the existing phases in three waves:

#### Wave 1 - Research Operating System

Primary phases: Phase 1 through Phase 4

- strengthen `shared_lib`, `data_platform`, `alpha_research`, and `backtest_engine`
- add contracts for:
  - predictions
  - optimizer requests/responses
  - research memory records
  - RL environment metadata
- add prediction-provider architecture for Kronos-like forecasting backends
- add RL research boundaries inspired by FinRL
- add richer validation and statistical robustness checks inspired by Vibe-Trading

#### Wave 2 - Agentic Research and Portfolio Intelligence

Primary phases: Phase 5 and parts of Phase 3 / Phase 6

- add research-memory abstractions influenced by MemPalace
- add memory-aware researcher and reviewer agents
- add a portfolio optimizer service influenced by NVIDIA quantitative portfolio optimization
- expose Kronos-like forecasting as an optional bounded model provider

#### Wave 3 - Production Trading Hardening

Primary phases: Phase 6 through Phase 10

- connect validated signals, forecasts, and optimized allocations into OMS/EMS/RMS boundaries
- keep execution deterministic and internal
- add late-stage scenario simulation inspired by MiroFish as optional stress tooling
- adopt RTK-style operator tooling around CI, Docker, and incident review workflows

### Non-Negotiable Guardrails

- adapters live at the edge and contracts live at the center
- upstream projects must not define internal schemas for orders, fills, positions, or risk events
- forecasting, optimization, memory, RL, and simulation remain optional modules
- agents may propose, validate, and summarize, but may not bypass OMS, RMS, KMS/HSM, approval gates, or treasury controls
- the web control plane may launch, monitor, approve, pause, reconcile, and halt workflows, but every mutating action must resolve through authenticated backend control APIs
- the browser must never hold broker credentials, exchange credentials, private keys, or signing authority
- portfolio optimization is a distinct stage after alpha generation and before execution
- scenario simulation is an enhancement to stress testing, not a replacement for replay- and market-mechanics-based backtesting
- if licensing or dependency drag becomes material, revert to pattern adoption and avoid direct integration

## Execution-Grade Design Package

The migration plan should be implemented alongside a lightweight design package so early implementation work is constrained by written decisions instead of informal assumptions.

### Design Artifacts

- ADR index: `docs/adr/README.md`
- ADR-0001: `docs/adr/0001-monorepo-workspace-and-package-boundaries.md`
- ADR-0002: `docs/adr/0002-data-platform-orchestration-and-immutability.md`
- ADR-0003: `docs/adr/0003-two-speed-execution-runtime-boundaries.md`
- ADR-0004: `docs/adr/0004-agent-permissions-and-control-plane.md`
- ADR-0005: `docs/adr/0005-tradfi-and-web3-gateway-architecture.md`
- ADR-0006: `docs/adr/0006-execution-signing-custody-and-kill-switches.md`
- Phase 0 scaffold: `docs/architecture/phase-0-scaffold.md`

### Delivery Rule

No major implementation phase should start without:

- the relevant ADR status set to `accepted` or `proposed with explicit implementation owner`
- explicit entry criteria met
- explicit exit criteria recorded and verified

## Phase 0 - Architecture Baseline and Repo Restructure

### Goal

Create the monorepo skeleton without breaking the existing CLI or backtesting flows.

### Tasks

- [x] Define the monorepo package strategy:
  - single root `pyproject.toml` with workspace-style internal packages, or
  - poly-package layout with package-local manifests and shared build tooling
  - **Decision:** Hybrid root workspace + package-local manifests per domain;
    native code isolated under `trading_system/native/`. Recorded in
    ADR-0001 (Accepted). Guarded by `tests/phase_0/test_adr_0001_package_strategy.py`.
- [x] Create the top-level directory structure:
  - `ai_agents/`
  - `data_platform/`
  - `alpha_research/`
  - `backtest_engine/`
  - `web_control_plane/`
  - `trading_system/`
  - `infrastructure/`
  - `shared_lib/`
  - Skeleton guarded by `tests/phase_0/test_monorepo_skeleton.py`
    (34 structural invariants, all green).
- [x] Introduce internal package naming and import conventions. Codified in
  `docs/architecture/package-and-import-conventions.md`. Guarded by
  `tests/phase_0/test_import_naming_conventions.py`.
- [x] Add architecture decision records for:
  - package boundaries (ADR-0001 Accepted)
  - Python vs Rust/C++ ownership (ADR-0003 Accepted)
  - runtime orchestration model (ADR-0002 Proposed, owner: Data Platform)
  - shared contracts and schemas (ADR-0001 + Phase 1 work)
  - two-speed execution separation and ownership boundaries (ADR-0003 Accepted)
  - shared research foundation vs divergent execution pathways (ADR-0003 Accepted)
  - agents (ADR-0004), gateways (ADR-0005), custody/kill-switches (ADR-0006)
    all Proposed with explicit Implementation Owners.
  - Guarded by `tests/phase_0/test_adr_status_and_owners.py`.
- [x] Define service-to-service communication standards for:
  - gRPC request/response services
  - Kafka or ZeroMQ event streaming
  - synchronous vs asynchronous control-plane traffic
  - Codified in `docs/architecture/service-communication-standards.md` with
    a binding Transport Matrix. Guarded by
    `tests/phase_0/test_service_communication_standards.py`.
- [x] Document the latency boundary where Python is removed from the HFT critical path.
  Codified in `docs/architecture/hft-latency-boundary.md` with a normative
  latency-budget table and explicit Python-forbidden zones. Guarded by
  `tests/phase_0/test_hft_latency_boundary.py`.
- [x] Preserve the current CLI as a compatibility facade until downstream
  modules are extracted. Strategy documented in
  `docs/architecture/cli-compatibility-facade.md`. Guarded by
  `tests/phase_0/test_cli_compatibility_facade.py` (includes static import
  rule: no domain package may import `src.*`).

### Deliverables

- [x] Monorepo skeleton committed
- [x] Internal package naming standard documented
- [x] Compatibility strategy for existing `src.main` documented

### Entry Criteria

- [x] ADR-0001 reviewed (Accepted)
- [x] ADR-0003 reviewed (Accepted)
- [x] `docs/architecture/phase-0-scaffold.md` agreed as the initial package skeleton
- [x] Existing branch and release compatibility constraints documented
  (`docs/architecture/cli-compatibility-facade.md`)

### Exit Criteria

- [x] Top-level monorepo directories created or approved for creation
- [x] Package/import naming conventions fixed in writing
- [x] Current CLI compatibility facade strategy documented
- [x] No Phase 1 extraction starts without an approved package-boundary direction

## Phase 1 - Shared Contracts, Telemetry, and Core Utilities

### Goal

Extract the common primitives used by all future modules before any service split.

### Tasks

- [x] Move logging/telemetry concerns from `src/utils/telemetry.py` into `shared_lib/logging/`.
- [x] Replace ad hoc logging with structured JSON contracts that include:
  - `trace_id`
  - `span_id`
  - service/module name
  - execution stage
  - `shared_lib.logging` (97% coverage, 16 AAA tests). Supports
    `configure_logging`, `log_event`, `bind_trace` (contextvars), and
    `time_block`. Redacts secrets; serialises Decimal/datetime.
- [x] Add OpenTelemetry bootstrap code and conventions for:
  - batch jobs
  - API services
  - agent execution traces
  - live trading execution paths
  - `shared_lib.telemetry` (100% coverage, 14 tests). `bootstrap(service, profile)`
    with four profiles (batch/api/agent/live); `start_span` binds
    trace_id/span_id onto `shared_lib.logging` contextvars; OTel optional
    dep (fallback tracer used when OTel not installed); `record_exception`
    emits ERROR log correlated to the active trace.
- [x] Create `shared_lib/math_utils/` for:
  - vectorized return math
  - stable risk metrics
  - Decimal-safe money helpers for ledger logic
  - migration path away from Pandas-heavy routines toward Polars/Numpy
  - `shared_lib.math_utils` (96% coverage, 39 tests). NumPy-only (no
    pandas dep). `Money` rejects `float` at every surface; currency
    mismatch raises; int factors allowed; bankers-rounded quantize.
- [x] Standardize RPC and event contracts shared between Python research services and Rust/C++ execution services.
  `shared_lib.transport` (98% coverage, 12 tests). `RpcEnvelope`
  requires idempotency_key + future deadline; `EventEnvelope` enforces
  topic naming convention; `dlq_topic` derives DLQ names;
  `redact_payload_for_logging` strips secrets before emitting envelopes
  to logs.
- [x] Define shared schema modules for:
  - market data bars
  - factor frames
  - prediction artifacts
  - portfolio optimization requests/responses
  - research memory records
  - RL environment metadata
  - run metadata and job status payloads
  - approval requests and approval decisions
  - audit events and operator actions
  - UI-facing execution and health status payloads
  - trade signals
  - orders/fills/positions
  - validation results
  - anomaly events
- [x] Introduce strict validation with Pydantic/Pandera for cross-module data contracts.
  `shared_lib.contracts` (98% coverage, 28 tests). All 15 contract
  modules use `frozen=True`/`extra="forbid"`; timestamps must be
  tz-aware; OHLC invariants, weights-sum-to-1, confidence 0..1,
  fail-requires-reason, HealthStatus.ok-matches-checks all enforced
  at construction time.

### Deliverables

- [x] `shared_lib/logging` (97% cov)
- [x] `shared_lib/math_utils` (96% cov)
- [x] shared domain schemas and validation contracts (98% cov)
- [x] OTel instrumentation baseline (100% cov, 14 tests)

### Entry Criteria

- [x] Phase 0 exit criteria satisfied
- [x] ADR-0001 accepted or implementation-ready (Accepted)
- [x] Shared schema ownership agreed across research, backtest, and execution modules

### Exit Criteria

- [x] Shared schemas are importable without `src/*` coupling
- [x] Telemetry baseline is reusable across batch jobs, APIs, agents, and execution paths
- [x] Decimal-safe money primitives exist before OMS work begins

## Phase 2 - Data Platform Extraction

### Goal

Turn the current datasource layer into a reusable ingestion and feature foundation.

### Tasks

- [ ] Extract `src/data/base.py` into `data_platform/connectors/` contract interfaces.
- [ ] Migrate existing providers into connector modules:
  - `yfinance`
  - `ccxt`
  - `alpaca`
  - `polygon`
  - `tiingo`
  - `finnhub`
  - `twelvedata`
  - `alphavantage`
- [ ] Separate connector concerns into:
  - auth/config
  - rate limiting
  - retrieval
  - normalization
  - cache policy
- [ ] Standardize the analytical storage layer around Apache Parquet.
- [ ] Make Polars the default data-manipulation engine for large immutable datasets.
- [ ] Convert cache logic into reusable ingestion storage policies with dataset versioning and immutable snapshots.
- [ ] Add prediction artifact persistence rules for model-generated features so forecasts can be versioned and reused exactly like factors.
- [ ] Add orchestration package under `data_platform/pipelines/`:
  - DAG definitions
  - backfill jobs
  - incremental updates
  - validation and retry handling
- [ ] Standardize on Apache Airflow as the primary workflow orchestrator for ingestion, backfills, feature refresh, and cross-source dependency management.
- [ ] Evaluate Prefect and Dagster only as secondary candidates through ADRs for narrower workflows or developer-experience tradeoffs.
- [ ] Define dbt transformation layers for reproducible downstream analytics and feature definitions.
- [ ] Define feature-store write/read interfaces under `data_platform/feature_store/`.
- [ ] Make the feature store the source of truth guaranteeing that research, backtests, and live trading use the exact same data logic and identical feature definitions.
- [ ] Model factor definitions as versioned, reproducible assets.
- [ ] Split data ingestion into two paths:
  - L3 tick and packet-capture ingestion for HFT
  - aggregated bars, fundamentals, and alternative data for mid-frequency research
- [ ] Define HFT ingestion requirements:
  - custom native parsers for binary market feeds and PCAP replay
  - order-book reconstruction from raw packets
  - storage policy for very large binary datasets
- [ ] Define mid-frequency ingestion requirements:
  - Parquet-backed bars and factors
  - alternative data enrichment
  - query patterns optimized for Polars
- [ ] Treat IBKR and Alpaca strictly as live execution and live market-connectivity providers, not as the primary historical training-data source.
- [ ] Add dedicated historical/vendor data integrations for training and backtesting:
  - Polygon.io
  - Databento
  - Tiingo
- [ ] Add on-chain indexing and decoding layers for Mid-Frequency AI research:
  - The Graph subgraph ingestion for aggregated protocol events
  - custom ETL that decodes raw EVM logs into tabular Parquet datasets
  - protocol-normalized schemas for swaps, lending, liquidity, and vault events
- [ ] Add data quality checks:
  - schema validity
  - continuity checks
  - duplicate/missing bar checks
  - source freshness
  - survivorship and symbol mapping audits

### Deliverables

- [ ] reusable connector layer
- [ ] orchestration DAG baseline
- [ ] feature-store contracts and storage abstraction
- [ ] dataset validation jobs

### Entry Criteria

- [ ] Phase 1 exit criteria satisfied
- [ ] ADR-0002 reviewed and implementation-ready
- [ ] Orchestrator baseline fixed to Apache Airflow unless superseded by ADR

### Exit Criteria

- [ ] Connector interfaces are separated from cache/storage concerns
- [ ] Airflow DAG structure exists for backfill and refresh workflows
- [ ] Feature definitions are versioned and reusable across research, backtesting, and live systems
- [ ] Vendor historical data path is documented separately from live broker connectivity

## Phase 3 - Alpha Research Workspace

### Goal

Separate ephemeral research from production factor logic.

### Tasks

- [ ] Create `alpha_research/notebooks/` with notebook execution and cleanup rules.
- [ ] Extract reusable strategy/factor logic into `alpha_research/factor_library/`.
- [ ] Distinguish clearly between:
  - exploratory notebook code
  - validated factors ready for backtest usage
  - live-approved factors eligible for OMS/RMS integration
- [ ] Build factor metadata contracts:
  - description
  - source dependencies
  - stationarity assumptions
  - universe coverage
  - leakage risk review
  - validation status
- [ ] Add forecasting-provider interfaces under `alpha_research/ml_models/providers/` so Kronos-like models can be integrated without dictating internal architecture.
- [ ] Create `alpha_research/ml_models/` for:
  - feature generation pipelines
  - training/evaluation loops
  - model registry metadata
  - hyperparameter tuning jobs
  - experiment tracking
- [ ] Create an RL-specific research lane under `alpha_research/ml_models/rl/` using FinRL-inspired environment and benchmark patterns without making RL the default strategy framework.
- [ ] Standardize model registry workflows with MLflow or Weights & Biases so deployed model weights can be tied back to specific training runs and dates.
- [ ] Add walk-forward and cross-validation pipelines that can be reused by the backtest engine.
- [ ] Define promotion gates from notebook -> factor library -> model artifact.

### Deliverables

- [ ] notebook governance pattern
- [ ] production factor library
- [ ] ML training pipeline structure
- [ ] promotion rules for factors/models

### Entry Criteria

- [ ] Phase 1 and Phase 2 exit criteria satisfied
- [ ] Feature-store read path available or stubbed with stable contracts
- [ ] Model registry approach documented in the active design package

### Exit Criteria

- [ ] Notebook code is clearly separated from production factor modules
- [ ] Factor promotion path is documented and testable
- [ ] Model artifacts can be traced to training data, registry metadata, and promotion status

## Phase 4 - Backtest Engine Modularization

### Goal

Convert the current runner into a reusable engine with explicit simulator, mechanics, and analytics boundaries.

### Tasks

- [ ] Extract `src/backtest/runner.py` into `backtest_engine/simulator/`.
- [ ] Split the current runner into focused modules:
  - event loop / scheduler
  - strategy adapter interface
  - portfolio/account state
  - fill simulation
  - result persistence
- [ ] Keep the simulator custom-code-first instead of adopting a generic OSS backtester as the core execution truth.
- [ ] Extract metrics/reporting responsibilities into `backtest_engine/analytics/`.
- [ ] Create `backtest_engine/market_mechanics/` for:
  - slippage models
  - fee models
  - spread models
  - market impact approximations
  - latency modeling
- [ ] Add market-structure realism modules for:
  - queue-position modeling in the limit order book
  - multi-leg execution constraints
  - microstructure-aware fill rules
  - exact order payload replay against gateway contracts
- [ ] Ensure no look-ahead bias by enforcing:
  - signal timestamp contracts
  - feature availability windows
  - execution timestamp separation
  - reproducible replay ordering
- [ ] Add scenario and stress replay support.
- [ ] Add statistical robustness and post-backtest validation modules inspired by Vibe-Trading, including walk-forward, confidence intervals, and stability checks where they provide real signal quality value.
- [ ] Add an engine API boundary so research factors and live execution systems can consume the same contracts.
- [ ] Ensure the simulator can emit and replay the exact API payloads and order payloads that `trading_system` will submit in paper and live modes.
- [ ] Evolve the current FastAPI dashboard into an authenticated initial web shell that exposes managed run browsing, run comparison, report access, and job/provenance views without introducing direct execution controls yet.

### Deliverables

- [ ] modular simulator core
- [ ] isolated market mechanics package
- [ ] analytics/tear-sheet package
- [ ] validation guardrails against leakage

### Entry Criteria

- [ ] Phase 1 exit criteria satisfied
- [ ] Phase 2 data contracts available
- [ ] Phase 3 factor contracts available for engine consumption

### Exit Criteria

- [ ] Simulator, analytics, and market-mechanics boundaries are separate packages or modules
- [ ] Exact API/order payload replay path is defined against gateway contracts
- [ ] Look-ahead and leakage validation runs as part of normal engine validation

## Phase 5 - AI Agents Foundation

### Goal

Introduce agent runtime and the first three target agents on top of stable contracts.

### Tasks

- [ ] Create `ai_agents/` runtime primitives:
  - agent registry
  - job queue/trigger model
  - prompt/version registry
  - tool interfaces
  - trace and audit logging
- [ ] Add a research-memory abstraction influenced by MemPalace so agents can retrieve prior experiment rationale, rejected ideas, and dataset caveats during controlled workflows.
- [ ] Evaluate and select a multi-agent orchestration framework baseline:
  - LangChain
  - AutoGen
  - CrewAI
- [ ] Implement scoped permissions so agents only receive the minimum API and file access needed for each workflow.
- [ ] Create restricted programmatic interfaces from agents into:
  - `alpha_research/factor_library`
  - `backtest_engine`
  - reporting/notification sinks
- [ ] Implement `ai_agents/alpha_researcher/`:
  - ArXiv ingestion
  - paper summarization
  - factor hypothesis generation
  - citation/provenance storage
  - promotion into research backlog, not direct production use
  - optional factor stub generation followed by controlled backtest execution
  - summary delivery to Slack or equivalent research channels
- [ ] Implement `ai_agents/code_reviewer/`:
  - static review hooks in CI
  - look-ahead bias heuristics
  - data leakage pattern checks
  - missing validation/test detection
- [ ] Implement `ai_agents/risk_monitor/`:
  - anomaly intake from live logs/metrics
  - interpretation rules for execution/risk alerts
  - escalation routing and kill-switch recommendations
- [ ] Add specialized hybrid-fund agents:
  - HFT Latency Agent for parsing native execution telemetry and detecting latency regressions
  - Mid-Freq Allocation Agent for adjusting portfolio risk posture from macro or alternative-data signals
- [ ] Add memory-aware researcher and reviewer workflows that can retrieve prior findings without giving agents unrestricted filesystem or production access.
- [ ] Add stack-specific resilience and routing agents:
  - On-Chain Routing Agent for gas-aware and liquidity-aware DEX routing decisions
  - Gateway Health Agent for broker, gateway, and container health monitoring plus reconnection workflows
- [ ] Add AI failure controls:
  - hallucinate detection gates before order generation
  - model drift monitoring hooks
  - Confidence Thresholding defaults to `Do Not Trade` or `Reduce Exposure`
  - Bounded Output Action Spaces so models cannot request unconstrained buying power
  - panic-sell / anomalous-allocation Kill Switch escalation rules
- [ ] Add human approval boundaries for all agents.
- [ ] Instrument agent runs with OTel GenAI conventions.
- [ ] Create `web_control_plane/backend/` and `web_control_plane/frontend/` as the long-term home for the control-plane web application while keeping the existing dashboard surface as a compatibility shell during migration.
- [ ] Add research and approval console workflows for:
  - backtest run review
  - factor/model promotion review
  - agent finding triage
  - approval queues with dataset, model, and validation provenance
- [ ] Ensure the web control plane uses authenticated backend APIs and audit logging for every approval or workflow mutation.

### Deliverables

- [ ] agent execution framework
- [ ] alpha researcher prototype
- [ ] PR/code review agent in CI
- [ ] live risk monitor prototype

### Entry Criteria

- [ ] Phase 1 shared telemetry and contracts are available
- [ ] ADR-0004 reviewed and implementation-ready
- [ ] Restricted tool/API surface defined for each agent role

### Exit Criteria

- [ ] Agent permissions are scoped by workflow and target system
- [ ] Agent traces and audit logs are emitted with shared telemetry conventions
- [ ] No agent has direct unrestricted access to OMS, KMS, or treasury systems

## Phase 6 - Trading System Mid-Frequency Layer

### Goal

Implement a production-grade trading system for minute-to-week horizons before introducing HFT complexity.

### Tasks

- [ ] Create `trading_system/oms/` for:
  - positions
  - orders
  - executions
  - account balances
  - reconciliation
- [ ] Create `trading_system/ems/` for:
  - execution scheduling
  - slicing policies
  - smart routing hooks
  - broker/exchange execution adapters
- [ ] Add RMS controls between signal generation and order submission:
  - max exposure
  - capital allocation
  - per-symbol limits
  - daily drawdown kill switch
  - pending-order caps
- [ ] Add deterministic pre-trade risk-engine controls:
  - Fat-Finger Checks on notional size and unit size
  - max volume per hour rules
  - Wash Trading Prevention and opposing-flow blocking
  - `TRADING_HALTED = True` style hard stop flags for intraday drawdown breaches
- [ ] Add AI-specific RMS controls:
  - model confidence thresholds
  - drift-triggered trading halts
  - panic-selling detection
  - portfolio-level circuit breakers
- [ ] Require state reconciliation before new order placement:
  - broker positions vs local OMS
  - exchange balances vs local ledger
  - on-chain balances/allowances vs execution assumptions
- [ ] Add mandatory stop-loss / take-profit / time-exit registration policies.
- [ ] Create `trading_system/gateways/` with clean interfaces first, even if low-latency implementations come later.
- [ ] Split the target sub-tree:
  - `mid_freq_engine/model_serving/`
  - `mid_freq_engine/portfolio_optimizer/`
  - `mid_freq_engine/execution_algos/`
- [ ] Introduce model-serving integration contracts for Triton/gRPC.
- [ ] Expose heavy prediction services as cloud-native microservices callable over gRPC.
- [ ] Introduce optimizer interfaces for convex position sizing and risk budgeting.
- [ ] Add a portfolio-construction stage inspired by NVIDIA quantitative portfolio optimization so validated signals can be transformed into bounded allocation decisions before OMS submission.
- [ ] Define Python-to-native transport patterns for signal handoff:
  - gRPC for low-latency request/response model calls
  - Kafka or ZeroMQ for streaming signals and execution events
- [ ] Ensure large block decisions are handed to execution algorithms that can TWAP/VWAP orders over time to minimize market impact.
- [ ] Define automated panic-button execution playbooks:
  - halt new AI signal intake into OMS
  - TradFi global cancel flows such as `reqGlobalCancel()` and Alpaca `DELETE /v2/orders`
  - DeFi stop-signing behavior for pending or future transactions
  - Flatten or Delta-Hedge logic for emergency containment
- [ ] Add web control plane execution-oversight views for paper and live trading that can:
  - display OMS, EMS, RMS, and reconciliation status
  - pause strategies
  - halt new signal intake
  - trigger bounded reconciliation and panic-button workflows
  - review alerts and incidents
- [ ] Explicitly forbid raw browser-driven trade entry or any web path that bypasses OMS, EMS, RMS, gateway, or approval policies.

### Deliverables

- [ ] OMS baseline
- [ ] EMS baseline
- [ ] RMS controls and kill-switches
- [ ] mid-frequency execution engine structure

### Entry Criteria

- [ ] Phase 1 and Phase 4 exit criteria satisfied
- [ ] ADR-0003 and ADR-0006 reviewed and implementation-ready
- [ ] Shared order/fill/position schemas stabilized

### Exit Criteria

- [ ] OMS and EMS are separated by responsibility
- [ ] RMS controls are enforced between signal generation and gateway execution
- [ ] Automated panic-button workflow is defined for TradFi and DeFi paths
- [ ] Mid-frequency execution contracts work without introducing HFT-only assumptions

## Phase 7 - Shared Market Connectivity

### Goal

Create reusable connectivity layers that can be shared by both mid-frequency and HFT stacks.

### Tasks

- [ ] Split gateway architecture into two paradigms:
  - TradFi gateways for broker-style order routing
  - Web3/DeFi gateways for transaction construction, signing, and broadcast
- [ ] Add TradFi gateway modules for:
  - Alpaca REST/WebSocket execution and streaming
  - IBKR execution through a locally hosted IB Gateway container
- [ ] Containerize IB Gateway with IBC/IB Controller support.
- [ ] Evaluate broker integration libraries such as `ib_insync` for the Python control plane while preserving a cleaner long-term native or FIX abstraction.
- [ ] Add IBKR operational workflows for:
  - scheduled daily restart automation
  - safe pre-restart trading halt
  - re-authentication and reconnect
  - OMS reconciliation after reconnect
- [ ] Add Web3 gateway modules for:
  - Alchemy or Infura-backed RPC access
  - EVM transaction construction and broadcast
  - protocol adapter interfaces for DEXs, lending venues, and routers
- [ ] Add a version-controlled ABI registry for on-chain protocol integration.
- [ ] Define chain-execution flows for:
  - transaction simulation
  - signing request generation
  - gas estimation
  - broadcast and confirmation handling
- [ ] Add DeFi-specific kill-switch controls:
  - Pausable contract or Safe-module `pause()` flows
  - automated ERC-20 allowance revocation back to zero
  - protocol-denylist enforcement when exploits or flash-loan attacks are detected
- [ ] Add `trading_system/shared_gateways/fix_engine/`.
- [ ] Add `trading_system/shared_gateways/binary_protocols/`.
- [ ] Separate protocol parsing from order state logic.
- [ ] Define test harnesses for:
  - message replay
  - sequence recovery
  - gap handling
  - heartbeat/session resets
- [ ] Add simulated gateways for paper trading parity.

### Deliverables

- [ ] FIX connectivity foundation
- [ ] binary protocol adapter contracts
- [ ] replayable paper-trading gateway layer

### Entry Criteria

- [ ] Phase 6 OMS/EMS schemas and state model are available
- [ ] ADR-0005 reviewed and implementation-ready

### Exit Criteria

- [ ] TradFi and Web3 gateway abstractions are separate and explicit
- [ ] Replay and reconciliation workflows exist for gateway failures
- [ ] Paper-trading parity is possible without live credential access

## Phase 8 - HFT Engine Foundations

### Goal

Only after OMS/EMS/RMS and shared gateways are stable, add the low-latency HFT-specific stack.

### Tasks

- [ ] Create `trading_system/hft_engine/core/` in Rust or C++ for:
  - lock-free queues
  - ring buffers
  - memory layout guarantees
  - deterministic event dispatch
- [ ] Enforce that Python is excluded from the tick-to-trade HFT critical path.
- [ ] Create `trading_system/hft_engine/network/` for Kernel bypass and NIC-specific integrations.
- [ ] Create `trading_system/hft_engine/fast_inference/` for ONNX/TensorRT wrappers.
- [ ] Create `trading_system/hft_engine/fpga/` only after software-path baselines exist.
- [ ] Define HFT inference constraints so research models must be compiled down to ONNX, C++, or FPGA-compatible logic before live eligibility.
- [ ] Add co-location and bare-metal deployment requirements to the HFT runtime specification.
- [ ] Define a strict interface between the HFT core and:
  - risk controls
  - market data decoding
  - order entry
  - model inference
- [ ] Build replay and latency benchmark harnesses before any live deployment path.

### Deliverables

- [ ] native low-latency core
- [ ] market-data/order-entry network layer
- [ ] inference wrapper interfaces
- [ ] benchmark and replay harnesses

### Entry Criteria

- [ ] Phase 7 exit criteria satisfied
- [ ] Two-speed boundary is enforced in code/package ownership
- [ ] Native toolchain ownership (Rust/C++/FPGA) assigned

### Exit Criteria

- [ ] Python is excluded from the HFT critical path by design
- [ ] Benchmark/replay harnesses exist before any live-path consideration
- [ ] HFT runtime contracts integrate with shared risk and gateway boundaries without bypassing them

## Phase 9 - Infrastructure and Platform Delivery

### Goal

Add deployment, orchestration, and environment separation once service boundaries are stable.

### Tasks

- [ ] Create `infrastructure/terraform/` for foundational cloud resources:
  - compute
  - queues
  - object storage
  - secrets/config stores
  - observability plumbing
- [ ] Add execution-security infrastructure for:
  - AWS KMS-backed signing
  - HashiCorp Vault-backed secrets and broker credentials
  - optional Nitro Enclave or enclave-adjacent signing isolation
- [ ] Add custody integration decision records for:
  - AWS KMS direct signing
  - Safe smart-contract treasury controls
  - MPC custodian integration such as Fireblocks or Fordefi
- [ ] Add separate infrastructure tracks:
  - bare-metal or co-located provisioning patterns for HFT
  - cloud-native Kubernetes infrastructure for mid-frequency and research workloads
- [ ] Create `infrastructure/kubernetes/` for service deployment patterns.
- [ ] Add deployment overlays for:
  - research workloads
  - ingestion pipelines
  - API/model-serving workloads
  - agent runners
  - live trading control-plane services
- [ ] Add deployment patterns for the `web_control_plane` frontend and backend, including auth, session management, and operator-safe routing to backend control APIs.
- [ ] Add GPU-backed model-serving deployment patterns for mid-frequency inference services.
- [ ] Add CI/CD workflows for:
  - package-level lint/test/build
  - container scanning
  - IaC scanning
  - code review agent integration
  - staged deployment promotion
- [ ] Add optional RTK-style operator tooling or wrappers for compact CI, Docker, and backtest failure summaries to reduce review noise for humans and agents.
- [ ] Add factor-library promotion gates so every PR from a human or agent triggers:
  - unit tests
  - leakage and look-ahead checks
  - out-of-sample backtest validation
- [ ] Add model deployment guardrails with registry-backed approvals using MLflow or Weights & Biases metadata.
- [ ] Add treasury and signing guardrails so:
  - bots cannot initiate unrestricted withdrawals
  - treasury assets can remain in a Safe or equivalent smart-contract vault
  - human multisig approval is required for exchange transfers, fiat off-ramps, or large treasury moves
- [ ] Add automated recovery workflows for gateway/container failures, especially IB Gateway restart and reconnect handling.
- [ ] Add out-of-band hard-kill infrastructure paths:
  - independent AWS Lambda or equivalent control-plane isolation function
  - IAM role revocation for compromised trading instances
  - immediate severing of broker, KMS, and cloud execution permissions
- [ ] Add environment boundaries:
  - local
  - research/dev
  - paper trading
  - production/live

### Deliverables

- [ ] Terraform baseline
- [ ] Kubernetes deployment skeleton
- [ ] CI/CD per package/service group
- [ ] environment promotion model

### Entry Criteria

- [ ] ADR-0002, ADR-0005, and ADR-0006 reviewed and implementation-ready
- [ ] Service boundaries from Phases 2 through 8 are stable enough to deploy independently

### Exit Criteria

- [ ] Deployment patterns exist for research, data, agents, model serving, and trading control-plane services
- [ ] Signing/custody model is documented in deployable infrastructure terms
- [ ] Out-of-band hard-kill path is documented and testable

## Phase 10 - Cutover and Decomposition of Current App

### Goal

Retire the legacy single-package structure after all critical modules are stable.

### Tasks

- [ ] Replace direct `src/*` imports with package-local imports from the monorepo modules.
- [ ] Keep a compatibility CLI during transition.
- [ ] Retire the legacy standalone dashboard by absorbing its responsibilities into `web_control_plane`, leaving only compatibility routes where needed.
- [ ] Port existing tests into module-scoped suites.
- [ ] Add contract tests across:
  - research -> backtest engine
  - backtest engine -> OMS/EMS
  - OMS/EMS -> gateways
  - agents -> CI/live telemetry
- [ ] Decommission legacy folders once:
  - parity is proven
  - coverage remains above gate
  - packaging and deploy flows are stable

### Deliverables

- [ ] compatibility CLI preserved during migration
- [ ] legacy module retirement checklist
- [ ] parity validation report

### Entry Criteria

- [ ] Phases 1 through 9 provide stable replacement boundaries for legacy `src/*` functionality
- [ ] Migration test coverage and contract tests exist for all critical flows

### Exit Criteria

- [ ] Legacy `src/*` dependencies are either removed or intentionally retained behind compatibility wrappers
- [ ] Parity validation is documented for data, backtests, reporting, and execution-control paths
- [ ] The web control plane is the primary operator interface for managed runs, approvals, reporting, and bounded execution oversight
- [ ] The repository can be reasoned about by package/domain instead of legacy module inheritance

## Recommended Execution Order

1. Phase 0 - repo/package architecture
2. Phase 1 - shared contracts and telemetry
3. Phase 2 - data platform extraction
4. Phase 3 - alpha research workspace
5. Phase 4 - backtest engine modularization
6. Phase 5 - AI agents
7. Phase 6 - mid-frequency OMS/EMS/RMS
8. Phase 7 - shared gateways
9. Phase 8 - HFT foundations
10. Phase 9 - infrastructure rollout
11. Phase 10 - final cutover

## Sequencing Notes

- HFT should not be the first build target. The OMS/EMS/RMS contracts must exist first.
- FPGA work should remain a late-stage optimization track, not an early repository concern.
- The current application is still the fastest path to bootstrap `data_platform`, `backtest_engine`, and `shared_lib`.
- The first implementation milestone should be a non-breaking monorepo skeleton plus shared contracts, not a full rewrite.
- The shared foundation should stay unified, but execution and infrastructure must split into HFT fast-path and mid-frequency smart-path concerns as early architecture decisions.
- Phase work should not be marked complete without satisfying both the technical tasks and the explicit exit criteria for that phase.
