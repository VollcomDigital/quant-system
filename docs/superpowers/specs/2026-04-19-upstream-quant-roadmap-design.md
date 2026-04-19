# Upstream Quant Platform Adoption Design

## Status

Approved by user for specification write-up.

## Objective

Define a master roadmap for incorporating the most valuable ideas from these upstream repositories into this codebase without letting any of them become the architectural spine:

- `shiyu-coder/Kronos`
- `MemPalace/mempalace`
- `HKUDS/Vibe-Trading`
- `rtk-ai/rtk`
- `NVIDIA-AI-Blueprints/quantitative-portfolio-optimization`
- `AI4Finance-Foundation/FinRL`
- `666ghj/MiroFish`

The target outcome is a research-first quant platform with production-grade control boundaries that fit the existing monorepo migration plan in `tasks/todo.md`, `docs/architecture/phase-0-scaffold.md`, and ADRs `0001` through `0006`.

## Design Summary

The platform should use a **hybrid upstream adoption model** with a strong bias toward **pattern-level adoption first** and **selective integration only at clean adapter boundaries**.

That means:

- internal packages own core contracts, orchestration, control planes, and risk boundaries
- upstream repositories are treated as architecture donors, bounded providers, or external operator tools
- direct integration is allowed only where the upstream capability is already naturally artifact- or service-shaped
- execution, custody, OMS, RMS, and treasury boundaries remain internal and deterministic

## Context

The current repository is a Dockerized, cache-aware backtesting system with a Typer CLI in `src/main.py`, a pluggable datasource layer under `src/data/`, strategy discovery under `src/strategies/`, runner-centric backtesting under `src/backtest/`, and reporting under `src/reporting/`.

The repo already contains a target architecture direction:

- `shared_lib/`
- `data_platform/`
- `alpha_research/`
- `backtest_engine/`
- `ai_agents/`
- `trading_system/`
- `infrastructure/`

This design extends that direction rather than replacing it.

## Design Principles

### 1. Contracts at the center, adapters at the edge

Every upstream-influenced capability must enter through an internal interface owned by this repository. No upstream repo should be allowed to define the internal schema for:

- market data bars
- feature frames
- factor metadata
- model predictions
- optimization requests
- backtest results
- order, fill, or position state
- risk events

### 2. Research-first, but execution-safe

The roadmap should prioritize research throughput and model experimentation first, while preserving production constraints from day one:

- no agent may bypass OMS, RMS, or gateway controls
- no external project may own execution logic
- no forecasting or optimization component may directly formulate live orders
- model outputs must be transformed into bounded internal signals before they can reach backtests or execution systems

### 3. Shared data lineage across the lifecycle

Research, backtesting, paper trading, and live decisioning must consume the same versioned datasets and feature definitions. Any upstream-inspired subsystem must fit into the shared data platform and feature-store design already planned in Phase 2 and Phase 3.

### 4. Optional capabilities stay optional

Forecasting, RL, portfolio optimization, memory, and simulation should all be introduced as optional modules or service backends. The base platform must remain usable without them.

### 5. License and dependency risk are first-class constraints

If a repo creates significant licensing friction, forces an incompatible runtime model, or introduces excessive dependency drag, prefer pattern adoption over direct integration.

## Upstream Repository Classification

### `HKUDS/Vibe-Trading`

**Classification:** primary architecture donor  
**Adoption mode:** pattern-level adoption

This repository has the strongest overlap with the target platform. Its most relevant value is not the exact code, but the way it combines:

- agent workflows
- loader registries
- multi-engine backtesting
- research-to-report pipelines
- validation and statistical post-processing

Recommended borrow targets:

- registry-based data loader and fallback patterns
- composite backtest orchestration concepts
- post-backtest validation modules such as walk-forward and robustness checks
- agent packaging patterns for research and review workflows

Recommended avoidance:

- do not make its full agent architecture the core runtime
- do not inherit its directory structure or control-plane decisions without translation into this repo's ADRs

### `shiyu-coder/Kronos`

**Classification:** bounded model provider  
**Adoption mode:** selective integration through an internal adapter

Kronos is best treated as an optional forecasting backend. The platform should not absorb its repository structure. Instead, it should add a forecasting provider interface inside `alpha_research/ml_models` and expose Kronos through that interface.

Recommended role:

- generate cacheable forward-looking prediction features from OHLCV windows
- support offline batch inference for many symbols
- feed predictions into backtests as versioned features or factor inputs

Recommended internal boundary:

- `alpha_research/ml_models/providers/kronos_adapter.py`
- `shared_lib/contracts/predictions.py`
- `data_platform/feature_store/` for prediction persistence and reuse

### `MemPalace/mempalace`

**Classification:** research memory subsystem donor  
**Adoption mode:** hybrid, with pattern adoption first

MemPalace offers a strong model for structured memory and retrieval. Its best fit is not generic chat history, but research continuity.

Recommended role:

- store factor hypotheses
- track experiment rationale and decision history
- retain model notes, failed trials, and dataset caveats
- retrieve context for researcher and reviewer agents

Recommended internal boundary:

- `ai_agents/runtime/memory/`
- `shared_lib/contracts/research_memory.py`
- optional external backend adapter if direct integration proves clean and low-risk

### `NVIDIA-AI-Blueprints/quantitative-portfolio-optimization`

**Classification:** optimization-pattern donor with optional accelerator backend  
**Adoption mode:** pattern-level adoption with optional backend integration

This repository is most useful for portfolio construction, not signal generation or execution.

Recommended role:

- add a portfolio optimization stage after alpha generation
- support Mean-CVaR and constrained optimization workflows
- expose GPU acceleration as an optional backend, not a core runtime assumption

Recommended internal boundary:

- `alpha_research/ml_models/portfolio_optimizer/`
- `trading_system/mid_freq_engine/portfolio_optimizer/`
- `shared_lib/contracts/portfolio_optimization.py`

### `AI4Finance-Foundation/FinRL`

**Classification:** RL research-pattern donor  
**Adoption mode:** pattern-level adoption only

FinRL is valuable as a source of RL environment design and benchmark workflow ideas. It should not become the base framework of the platform.

Recommended role:

- define RL-friendly train/test/trade lifecycle patterns
- inspire environment interfaces for strategy research
- provide benchmark tasks for RL-specific strategy tracks

Recommended internal boundary:

- `alpha_research/ml_models/rl/`
- `shared_lib/contracts/rl_env.py`

### `rtk-ai/rtk`

**Classification:** operator and agent ergonomics tool  
**Adoption mode:** selective external tooling use

RTK is not a trading subsystem. Its value is reducing token and operator noise in logs, test output, Docker output, and CI traces.

Recommended role:

- compress backtest failure output
- compact CI and test logs for agent review
- reduce context waste during platform operations

Recommended boundary:

- external tooling in developer workflows
- optional wrappers around CI and operator commands
- no integration into platform runtime or execution code

### `666ghj/MiroFish`

**Classification:** scenario-simulation idea donor  
**Adoption mode:** concept mining only

MiroFish is the weakest direct fit. The useful idea is event- and narrative-driven simulation rather than any direct code integration.

Recommended role:

- future scenario generation for stress testing
- what-if simulation for extreme regime narratives
- report generation around simulated shock scenarios

Recommended boundary:

- late-stage optional research subsystem
- separate from core backtesting truth
- no direct integration into core runtime

## Target Capability Map

The roadmap should add these capability layers.

### 1. Forecasting and predictive feature layer

Influenced by Kronos.

This layer produces versioned prediction artifacts that behave like features. Predictions should be:

- generated in batch
- cached by model version, dataset version, symbol, timeframe, and horizon
- reusable across backtests and portfolio construction
- visible in reports and experiment metadata

### 2. Research memory and retrieval layer

Influenced by MemPalace.

This layer provides continuity for:

- factor and model hypotheses
- why a strategy was approved or rejected
- historical experiment summaries
- dataset anomalies and source-specific quirks
- agent retrieval context for follow-up analysis

### 3. Composite research and validation layer

Influenced primarily by Vibe-Trading.

This layer upgrades the current backtest system with:

- richer loader and provider registries
- stronger validation and statistical robustness checks
- research-to-report automation
- more explicit research orchestration patterns for agents

### 4. Portfolio construction and risk allocation layer

Influenced by NVIDIA QPO.

This layer sits after alpha generation and before execution. It transforms candidate signals into allocation decisions using constrained optimizers and risk budgets.

### 5. RL strategy track

Influenced by FinRL.

This layer creates a dedicated RL research lane that can coexist with rule-based and forecast-based strategies without becoming the default strategy framework.

### 6. Scenario simulation and narrative stress testing

Influenced by MiroFish.

This layer is intentionally late. It should enrich stress testing and research storytelling without changing the core replay- and market-mechanics-based backtesting truth.

### 7. Agent and operator ergonomics

Influenced by RTK.

This layer improves how humans and agents inspect outputs, especially in CI, Docker, and long-running research workflows.

## Phased Master Roadmap

The roadmap should be layered on top of the existing migration phases rather than replacing them.

### Wave 1: Research Operating System

**Primary phases:** Phase 1 through Phase 4  
**Goal:** make research, data, and backtests strong enough to host upstream-inspired capabilities cleanly

Deliveries:

1. strengthen `shared_lib` contracts for features, predictions, optimizers, and memory
2. extract the data platform and feature-store design so external model outputs can be versioned and reused
3. modularize the backtest engine enough to accept:
   - classical factors
   - predictive model outputs
   - RL policies
4. add validation and robustness reporting patterns inspired by Vibe-Trading
5. define prediction-provider interfaces that can host Kronos later
6. define RL environment interfaces inspired by FinRL

Success condition:

The platform can evaluate rules, factors, and model-generated features through the same backtest and reporting stack.

### Wave 2: Agentic Research and Portfolio Intelligence

**Primary phases:** Phase 5 and parts of Phase 3 / Phase 6  
**Goal:** build a memory-aware research and optimization layer on top of stable data and backtest contracts

Deliveries:

1. implement a research memory abstraction influenced by MemPalace
2. build scoped researcher, reviewer, and risk-monitor agents using the ADR-0004 mediated control plane
3. add a portfolio optimizer service influenced by NVIDIA QPO
4. add a model-provider adapter for Kronos as an optional forecasting backend
5. add RL research modules and benchmark workflows inspired by FinRL

Success condition:

Agents can propose experiments, retrieve prior context, trigger controlled backtests, and summarize results without ever touching execution-critical systems directly.

### Wave 3: Production Trading Hardening

**Primary phases:** Phase 6 through Phase 10  
**Goal:** connect the improved research system into OMS, RMS, and gateway boundaries without allowing upstream-driven design drift

Deliveries:

1. route optimized allocations and validated signals through the internal OMS/EMS/RMS pipeline
2. keep forecasting and optimization outputs bounded as inputs to deterministic execution layers
3. expose research memory and scenario outputs to risk and control-plane workflows only where useful
4. add scenario-simulation experiments inspired by MiroFish as optional stress inputs
5. adopt RTK-style operator tooling around CI, Docker, and incident review workflows

Success condition:

The platform gains richer upstream-inspired intelligence without losing execution safety, reproducibility, or ownership of the control plane.

## File and Package Design Impact

This roadmap implies the following package additions or extensions.

### `shared_lib/contracts`

Add canonical contracts for:

- predictions
- portfolio optimization requests and responses
- research memory records
- RL environment metadata
- validation findings

### `data_platform/feature_store`

Add support for:

- model-generated feature persistence
- versioned prediction snapshots
- factor and prediction provenance

### `alpha_research`

Extend with:

- `factor_library/`
- `ml_models/providers/`
- `ml_models/portfolio_optimizer/`
- `ml_models/rl/`
- promotion metadata for experimental vs validated capabilities

### `backtest_engine`

Extend with:

- prediction-aware strategy adapters
- optimizer-aware pre-allocation stages
- richer statistical validation modules
- replay-safe contracts for model and optimizer outputs

### `ai_agents/runtime`

Extend with:

- memory abstraction and retrieval interfaces
- controlled experiment launch tools
- report synthesis tools
- validation summary tools

### `trading_system/mid_freq_engine`

Extend with:

- portfolio optimizer consumption
- bounded model-signal ingestion
- deterministic transformation from research outputs into OMS-ready intents

## Data Flow Design

The intended flow is:

1. raw and normalized market data is ingested through `data_platform/connectors`
2. canonical features are stored and versioned through `data_platform/feature_store`
3. optional model providers such as Kronos generate prediction artifacts from those features
4. `alpha_research` promotes selected factors, predictions, and policies into validated research assets
5. `backtest_engine` evaluates those assets using the same market-mechanics and analytics stack
6. optional portfolio optimization transforms validated candidate exposures into allocations
7. only approved, bounded outputs move toward `trading_system`
8. OMS, EMS, RMS, and gateways remain deterministic and internal

## Error Handling and Failure Policy

### Upstream adapter failures

If an upstream-backed provider fails:

- the failure must be isolated to the adapter boundary
- the platform must surface deterministic error metadata
- cached historical results must remain readable
- execution systems must not block unless the failure affects an active dependency explicitly selected for a workflow

### Agent failures

If an agent cannot retrieve memory, run a backtest, or summarize a result:

- the failure must be logged with the shared telemetry conventions
- no agent may silently substitute invented outputs
- the workflow should degrade to a human-readable failure summary

### Model and optimizer failures

If a forecast or optimizer result is missing, stale, or outside contract bounds:

- the system must mark the artifact invalid for promotion
- backtests may fail fast or skip the run depending on workflow configuration
- live trading flows must default to no-trade or reduced-exposure states

## Testing Strategy

The implementation plan derived from this spec should include:

- contract tests for prediction, optimizer, and memory schemas
- adapter tests for each bounded external provider
- deterministic replay tests proving model outputs do not bypass feature lineage
- regression tests for backtest parity when new provider types are introduced
- validation tests for portfolio optimization safety bounds
- agent tests that mock all external calls and verify approval boundaries

No direct upstream integration should be considered complete without tests that prove:

- internal contracts are stable
- the adapter failure mode is explicit
- the platform remains usable without the optional upstream capability enabled

## Non-Goals

This roadmap does not propose:

- making any upstream repository the main framework of the platform
- replacing the internal backtest engine with a generic OSS engine
- allowing agents to own execution decisions end-to-end
- introducing narrative simulation ahead of core research and execution hardening
- forcing GPU, RL, or memory infrastructure into every deployment tier

## Key Decisions

1. Use a **hybrid adoption model** with **pattern-first bias**
2. Treat **Vibe-Trading** as the main architecture donor
3. Treat **Kronos** as an optional bounded forecasting provider
4. Treat **MemPalace** as inspiration for a research-memory subsystem
5. Treat **NVIDIA QPO** as the basis for an optimizer layer, not a platform core
6. Treat **FinRL** as an RL design donor only
7. Treat **RTK** as operator tooling, not platform runtime
8. Treat **MiroFish** as a late-stage scenario-simulation idea source only
9. Prioritize **research operating system first**, then **agentic intelligence**, then **production hardening**

## Approval Gate for Planning

This design is sufficient to support a detailed implementation plan. The next planning stage should convert this design into a bite-sized execution plan that:

- maps the work onto the existing migration phases
- identifies the exact files and packages to create or modify
- sequences optional capabilities behind stable contracts
- preserves the existing CLI and compatibility facade while the new packages are introduced
