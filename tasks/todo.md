# Quant Monorepo Migration Plan

## Objective

Evolve the current single-package backtesting application into a staged monorepo that can support:

- research and factor discovery
- data ingestion and feature management
- a reusable backtest engine
- autonomous AI agents
- a production trading system split between mid-frequency and HFT concerns
- infrastructure and deployment automation

## Architectural Pillars to Preserve

The target implementation should explicitly preserve these design rules throughout all phases:

- AI agents are treated as scoped programmatic workers, not ad hoc LLM wrappers.
- The data platform is immutable, versioned, and reproducible.
- The backtest engine remains custom-code-first for market-structure realism.
- The trading system is polyglot, with Python for research and native code for latency-critical execution.
- CI/CD and MLOps are first-class guardrails, especially for factor promotion and model deployment.

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
- `src/reporting/*` -> `backtest_engine/analytics` and shared reporting interfaces
- `src/utils/telemetry.py` -> `shared_lib/logging`
- `src/config.py` -> shared contracts/config packages
- `src/strategies/*` -> `alpha_research/factor_library` bootstrap interfaces

The largest missing capabilities are:

- a true monorepo package layout with internal contracts
- a data platform with orchestration and feature-store abstractions
- research training and model-serving pipelines
- OMS/EMS separation for live trading
- agent runtime and review/risk agents
- low-latency gateway/HFT foundations
- infrastructure-as-code and platform deployment boundaries

## Phase 0 - Architecture Baseline and Repo Restructure

### Goal

Create the monorepo skeleton without breaking the existing CLI or backtesting flows.

### Tasks

- [ ] Define the monorepo package strategy:
  - single root `pyproject.toml` with workspace-style internal packages, or
  - poly-package layout with package-local manifests and shared build tooling
- [ ] Create the top-level directory structure:
  - `ai_agents/`
  - `data_platform/`
  - `alpha_research/`
  - `backtest_engine/`
  - `trading_system/`
  - `infrastructure/`
  - `shared_lib/`
- [ ] Introduce internal package naming and import conventions.
- [ ] Add architecture decision records for:
  - package boundaries
  - Python vs Rust/C++ ownership
  - runtime orchestration model
  - shared contracts and schemas
  - two-speed execution separation and ownership boundaries
  - shared research foundation vs divergent execution pathways
- [ ] Define service-to-service communication standards for:
  - gRPC request/response services
  - Kafka or ZeroMQ event streaming
  - synchronous vs asynchronous control-plane traffic
- [ ] Document the latency boundary where Python is removed from the HFT critical path.
- [ ] Preserve the current CLI as a compatibility facade until downstream modules are extracted.

### Deliverables

- [ ] Monorepo skeleton committed
- [ ] Internal package naming standard documented
- [ ] Compatibility strategy for existing `src.main` documented

## Phase 1 - Shared Contracts, Telemetry, and Core Utilities

### Goal

Extract the common primitives used by all future modules before any service split.

### Tasks

- [ ] Move logging/telemetry concerns from `src/utils/telemetry.py` into `shared_lib/logging/`.
- [ ] Replace ad hoc logging with structured JSON contracts that include:
  - `trace_id`
  - `span_id`
  - service/module name
  - execution stage
- [ ] Add OpenTelemetry bootstrap code and conventions for:
  - batch jobs
  - API services
  - agent execution traces
  - live trading execution paths
- [ ] Create `shared_lib/math_utils/` for:
  - vectorized return math
  - stable risk metrics
  - Decimal-safe money helpers for ledger logic
  - migration path away from Pandas-heavy routines toward Polars/Numpy
- [ ] Standardize RPC and event contracts shared between Python research services and Rust/C++ execution services.
- [ ] Define shared schema modules for:
  - market data bars
  - factor frames
  - trade signals
  - orders/fills/positions
  - validation results
  - anomaly events
- [ ] Introduce strict validation with Pydantic/Pandera for cross-module data contracts.

### Deliverables

- [ ] `shared_lib/logging`
- [ ] `shared_lib/math_utils`
- [ ] shared domain schemas and validation contracts
- [ ] OTel instrumentation baseline

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
- [ ] Add orchestration package under `data_platform/pipelines/`:
  - DAG definitions
  - backfill jobs
  - incremental updates
  - validation and retry handling
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
- [ ] Create `alpha_research/ml_models/` for:
  - feature generation pipelines
  - training/evaluation loops
  - model registry metadata
  - hyperparameter tuning jobs
  - experiment tracking
- [ ] Standardize model registry workflows with MLflow or Weights & Biases so deployed model weights can be tied back to specific training runs and dates.
- [ ] Add walk-forward and cross-validation pipelines that can be reused by the backtest engine.
- [ ] Define promotion gates from notebook -> factor library -> model artifact.

### Deliverables

- [ ] notebook governance pattern
- [ ] production factor library
- [ ] ML training pipeline structure
- [ ] promotion rules for factors/models

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
- [ ] Add an engine API boundary so research factors and live execution systems can consume the same contracts.
- [ ] Ensure the simulator can emit and replay the exact API payloads and order payloads that `trading_system` will submit in paper and live modes.

### Deliverables

- [ ] modular simulator core
- [ ] isolated market mechanics package
- [ ] analytics/tear-sheet package
- [ ] validation guardrails against leakage

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
- [ ] Add human approval boundaries for all agents.
- [ ] Instrument agent runs with OTel GenAI conventions.

### Deliverables

- [ ] agent execution framework
- [ ] alpha researcher prototype
- [ ] PR/code review agent in CI
- [ ] live risk monitor prototype

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
- [ ] Add mandatory stop-loss / take-profit / time-exit registration policies.
- [ ] Create `trading_system/gateways/` with clean interfaces first, even if low-latency implementations come later.
- [ ] Split the target sub-tree:
  - `mid_freq_engine/model_serving/`
  - `mid_freq_engine/portfolio_optimizer/`
  - `mid_freq_engine/execution_algos/`
- [ ] Introduce model-serving integration contracts for Triton/gRPC.
- [ ] Expose heavy prediction services as cloud-native microservices callable over gRPC.
- [ ] Introduce optimizer interfaces for convex position sizing and risk budgeting.
- [ ] Define Python-to-native transport patterns for signal handoff:
  - gRPC for low-latency request/response model calls
  - Kafka or ZeroMQ for streaming signals and execution events
- [ ] Ensure large block decisions are handed to execution algorithms that can TWAP/VWAP orders over time to minimize market impact.

### Deliverables

- [ ] OMS baseline
- [ ] EMS baseline
- [ ] RMS controls and kill-switches
- [ ] mid-frequency execution engine structure

## Phase 7 - Shared Market Connectivity

### Goal

Create reusable connectivity layers that can be shared by both mid-frequency and HFT stacks.

### Tasks

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
- [ ] Add GPU-backed model-serving deployment patterns for mid-frequency inference services.
- [ ] Add CI/CD workflows for:
  - package-level lint/test/build
  - container scanning
  - IaC scanning
  - code review agent integration
  - staged deployment promotion
- [ ] Add factor-library promotion gates so every PR from a human or agent triggers:
  - unit tests
  - leakage and look-ahead checks
  - out-of-sample backtest validation
- [ ] Add model deployment guardrails with registry-backed approvals using MLflow or Weights & Biases metadata.
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

## Phase 10 - Cutover and Decomposition of Current App

### Goal

Retire the legacy single-package structure after all critical modules are stable.

### Tasks

- [ ] Replace direct `src/*` imports with package-local imports from the monorepo modules.
- [ ] Keep a compatibility CLI during transition.
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

## Recommended Execution Order

1. Phase 0 - repo/package architecture
2. Phase 1 - shared contracts and telemetry
3. Phase 2 - data platform extraction
4. Phase 4 - backtest engine modularization
5. Phase 3 - alpha research workspace
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
