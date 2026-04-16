# Phase 0 Scaffold Recommendation

## Purpose

This document turns Phase 0 of `tasks/todo.md` into a concrete bootstrap scaffold that can be implemented without rewriting the current application on day one.

## Design Goals

- preserve the current `src/` application while new packages are introduced
- create clean module boundaries for the target monorepo
- keep Python on the shared, research, orchestration, and control-plane paths
- reserve Rust/C++ for latency-critical execution and feed-handling paths
- support a staged migration rather than a flag day rewrite

## Recommended Top-Level Layout

```text
quant-monorepo/
├── ai_agents/
│   ├── pyproject.toml
│   ├── src/ai_agents/
│   │   ├── runtime/
│   │   ├── alpha_researcher/
│   │   ├── code_reviewer/
│   │   └── risk_monitor/
│   └── tests/
├── alpha_research/
│   ├── pyproject.toml
│   ├── notebooks/
│   ├── src/alpha_research/
│   │   ├── factor_library/
│   │   ├── ml_models/
│   │   └── promotion/
│   └── tests/
├── backtest_engine/
│   ├── pyproject.toml
│   ├── src/backtest_engine/
│   │   ├── simulator/
│   │   ├── market_mechanics/
│   │   └── analytics/
│   └── tests/
├── data_platform/
│   ├── pyproject.toml
│   ├── dags/
│   ├── dbt/
│   ├── src/data_platform/
│   │   ├── connectors/
│   │   ├── pipelines/
│   │   ├── feature_store/
│   │   ├── indexing/
│   │   └── schemas/
│   └── tests/
├── docs/
│   ├── adr/
│   └── architecture/
├── infrastructure/
│   ├── terraform/
│   ├── kubernetes/
│   └── runbooks/
├── shared_lib/
│   ├── pyproject.toml
│   ├── src/shared_lib/
│   │   ├── contracts/
│   │   ├── logging/
│   │   ├── math_utils/
│   │   ├── risk/
│   │   └── transport/
│   └── tests/
├── trading_system/
│   ├── pyproject.toml
│   ├── src/trading_system/
│   │   ├── oms/
│   │   ├── ems/
│   │   ├── gateways/
│   │   ├── shared_gateways/
│   │   └── mid_freq_engine/
│   ├── native/
│   │   ├── hft_engine/
│   │   │   ├── core/
│   │   │   ├── network/
│   │   │   ├── fast_inference/
│   │   │   └── fpga/
│   │   └── shared/
│   └── tests/
├── src/
│   └── ... existing application retained during migration
├── tasks/
│   └── todo.md
└── pyproject.toml
```

## Packaging Recommendation

## Preferred approach

Use a root Python workspace with package-local `pyproject.toml` files for the major Python domains:

- `shared_lib`
- `data_platform`
- `backtest_engine`
- `alpha_research`
- `ai_agents`
- `trading_system`

This gives:

- isolated dependency management
- faster package-local lint and test runs
- a natural migration path from `src/`
- cleaner ownership boundaries per subsystem

## Native code placement

Keep native HFT code under:

- `trading_system/native/hft_engine/...`

This avoids forcing Rust/C++ build complexity into every Python package while still preserving a coherent monorepo layout.

## Import and Naming Convention

Use absolute package imports only:

- `shared_lib.contracts.*`
- `data_platform.connectors.*`
- `backtest_engine.simulator.*`
- `alpha_research.factor_library.*`
- `ai_agents.runtime.*`
- `trading_system.oms.*`

Do not create cross-package imports through ad hoc relative traversal.

## Phase 0 Bootstrap Sequence

1. Create the package directories and empty package roots.
2. Add package-local manifests with minimal dependencies only.
3. Add placeholder `__init__.py` and test packages.
4. Add shared contract stubs in `shared_lib/contracts/`.
5. Keep `src/main.py` as the compatibility CLI.
6. Add thin compatibility adapters that call into new packages only when functionality is migrated.

## Minimal Initial Package Responsibilities

### `shared_lib`

- contracts
- logging
- math helpers
- risk primitives
- transport envelopes

### `data_platform`

- datasource contracts
- ingestion DAG definitions
- dbt transformations
- feature-store interfaces
- on-chain indexing interfaces

### `backtest_engine`

- event-driven simulator
- market mechanics
- analytics and validation

### `alpha_research`

- notebook governance
- factor library
- model training pipelines
- promotion metadata

### `ai_agents`

- agent runtime
- permission model
- control-plane adapters

### `trading_system`

- OMS
- EMS
- gateway control-plane adapters
- mid-frequency model serving integration
- native-code boundary for HFT

## What stays in `src/` initially

Retain the current application as:

- the user-facing CLI
- the operational baseline
- the migration reference implementation

Nothing in Phase 0 should require deleting or moving large portions of `src/`.

## Phase 0 Non-Goals

- no full code extraction yet
- no HFT runtime implementation
- no production Airflow deployment
- no custody/signing implementation
- no complete CI split per package yet

## Definition of Ready for Phase 1

Phase 0 is considered ready for Phase 1 when:

- the package roots exist
- ADR ownership is established
- import naming is fixed
- compatibility strategy is documented
- shared contracts package location is agreed
- the team can start extracting `shared_lib` without revisiting repo structure
