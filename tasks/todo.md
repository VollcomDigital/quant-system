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

### Entry Criteria

- [ ] ADR-0001 reviewed
- [ ] ADR-0003 reviewed
- [ ] `docs/architecture/phase-0-scaffold.md` agreed as the initial package skeleton
- [ ] Existing branch and release compatibility constraints documented

### Exit Criteria

- [ ] Top-level monorepo directories created or approved for creation
- [ ] Package/import naming conventions fixed in writing
- [ ] Current CLI compatibility facade strategy documented
- [ ] No Phase 1 extraction starts without an approved package-boundary direction

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

### Entry Criteria

- [ ] Phase 0 exit criteria satisfied
- [ ] ADR-0001 accepted or implementation-ready
- [ ] Shared schema ownership agreed across research, backtest, and execution modules

### Exit Criteria

- [ ] Shared schemas are importable without `src/*` coupling
- [ ] Telemetry baseline is reusable across batch jobs, APIs, agents, and execution paths
- [ ] Decimal-safe money primitives exist before OMS work begins

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
- [ ] Add an engine API boundary so research factors and live execution systems can consume the same contracts.
- [ ] Ensure the simulator can emit and replay the exact API payloads and order payloads that `trading_system` will submit in paper and live modes.

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
- [ ] Define Python-to-native transport patterns for signal handoff:
  - gRPC for low-latency request/response model calls
  - Kafka or ZeroMQ for streaming signals and execution events
- [ ] Ensure large block decisions are handed to execution algorithms that can TWAP/VWAP orders over time to minimize market impact.
- [ ] Define automated panic-button execution playbooks:
  - halt new AI signal intake into OMS
  - TradFi global cancel flows such as `reqGlobalCancel()` and Alpaca `DELETE /v2/orders`
  - DeFi stop-signing behavior for pending or future transactions
  - Flatten or Delta-Hedge logic for emergency containment

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
