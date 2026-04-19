# Third-Party Integration Plan

## Purpose

This document maps seven external open-source projects onto the existing
phased monorepo plan in `tasks/todo.md` and the package skeleton in
`docs/architecture/phase-0-scaffold.md`. Each integration is scoped to a
specific target package, a specific phase, and a specific ADR-gated entry
point so that adoption does not bypass the governance rules already defined
for this repository.

The governing principle is **vendor-in, not build-on-top-of**: each
integration must either land behind an internal adapter under
`shared_lib/`, `alpha_research/`, `ai_agents/`, or `backtest_engine/`, or it
must stay outside the runtime surface of the trading system entirely.
Nothing here is allowed to create a direct dependency from the
`trading_system/` packages onto an unvetted external repository.

## Candidate Projects

1. [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos) — foundation model for K-line / OHLCV forecasting (MIT, PyTorch, HF weights).
2. [MemPalace/mempalace](https://github.com/MemPalace/mempalace) — local-first verbatim agent memory with a temporal knowledge graph (MIT, Python, MCP server).
3. [HKUDS/Vibe-Trading](https://github.com/HKUDS/Vibe-Trading) — multi-agent natural-language trading workspace and backtest engine bundle (MIT, Python + React).
4. [rtk-ai/rtk](https://github.com/rtk-ai/rtk) — Rust CLI proxy that compresses developer-tool output before it reaches a coding agent (MIT, Rust).
5. [NVIDIA-AI-Blueprints/quantitative-portfolio-optimization](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization) — GPU-accelerated Mean-CVaR blueprint (Apache-2.0, CUDA / cuOpt).
6. [AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL) — deep reinforcement learning framework for finance (MIT, PyTorch + Gym).
7. [666ghj/MiroFish](https://github.com/666ghj/MiroFish) — multi-agent "swarm intelligence" simulation engine over OASIS + Zep (license unclear, Docker Compose app).

## Integration Governance Rules

The following rules apply to every integration listed below and are the
acceptance bar for any PR that introduces one of these projects. These
rules are the subject of ADR-0007 (see `docs/adr/README.md`).

- Every external project enters the repository behind an adapter module
  under `shared_lib/`, `alpha_research/`, or `ai_agents/`. The rest of the
  monorepo imports the adapter, not the upstream package.
- Direct imports of upstream packages from any module under
  `trading_system/` are forbidden. Signals and artifacts must cross the
  boundary through the shared contracts defined in Phase 1.
- Every integration declares its license, upstream commit or tag that was
  vendored or depended on, and the review date in the adapter module
  docstring and in this document.
- Any integration whose license is unclear or whose upstream has not been
  verified cannot be installed as a runtime dependency. It may only be
  prototyped in `alpha_research/notebooks/` with explicit ADR approval.
- GPU-only or hardware-gated integrations are exposed behind Poetry
  optional extras and excluded from the default `docker-compose` runtime.
- Any integration that ships its own agent loop, model, or execution
  engine must be wrapped so that kill-switch, RMS, and custody rules from
  ADR-0004 and ADR-0006 still apply before any live order path is touched.

## Integration Fit Matrix

| Project | Primary target package | Phase entry point | Installation mode | License | Status flag |
| --- | --- | --- | --- | --- | --- |
| Kronos | `alpha_research/ml_models/forecasters/kronos/` | Phase 3 | Poetry extra `kronos` (torch + HF Hub) | MIT | accept with adapter |
| MemPalace | `ai_agents/runtime/memory/mempalace/` | Phase 5 | Poetry extra `agent-memory` (pip) | MIT | accept with adapter |
| Vibe-Trading | cherry-picked into `ai_agents/` and `alpha_research/` | Phase 5 after Phase 3/4 stable | selective code vendor, no runtime dep | MIT | cherry-pick only |
| rtk-ai | repo-wide developer tooling | Phase 9 (dev-env track) | external binary, not a monorepo dep | MIT | dev-env only, no runtime dep |
| NVIDIA QPO | `alpha_research/optimizers/gpu_cvar/` and `backtest_engine/analytics/optimizers/` | Phase 3 research, Phase 4 integration | Poetry extra `gpu-cvar` + CUDA image | Apache-2.0 | opt-in GPU track |
| FinRL | `alpha_research/ml_models/rl/finrl/` | Phase 3 | Poetry extra `finrl-rl` | MIT (trademark restrictions) | legacy-aware adapter |
| MiroFish | research sandbox only | blocked pending ADR | no runtime dep until license verified | unclear | blocked |

## Per-Project Integration Plans

Every integration plan below lists:

- target package path
- entry criteria relative to the existing phased plan
- task breakdown
- exit criteria
- risk flags

### 1. Kronos (OHLCV forecasting foundation model)

- Target package: `alpha_research/ml_models/forecasters/kronos/`.
- Entry criteria: Phase 1 exit satisfied (shared schemas for bars),
  Phase 2 feature-store read path available or stubbed, Phase 3 entry
  criteria met.
- Tasks:
  - Define `KronosForecasterAdapter` against a shared `BarForecaster`
    interface in `alpha_research/ml_models/forecasters/base.py`.
  - Load model weights from Hugging Face Hub and cache under a
    configurable `MODELS_CACHE_DIR` to stay compatible with the existing
    `.cache/` discipline.
  - Route input bars through `shared_lib/math_utils/` normalization so the
    adapter never consumes raw DataFrames with unchecked schema.
  - Enforce the model context window limit in the adapter and expose it
    as adapter metadata so the backtest engine can refuse inputs that
    exceed 512 bars for the small and base checkpoints.
  - Add a `param_grid`-compatible wrapper so Kronos-generated signals can
    flow into `backtest_engine/simulator/` without bypassing grid or
    Optuna-driven search.
- Exit criteria:
  - adapter importable from `alpha_research/` without any `src/*` import
  - model weights pulled through a reproducible cache contract, not ad
    hoc HF downloads
  - inference gated behind an optional extra, not installed by default
- Risk flags:
  - GPU strongly recommended; the adapter must stay CPU-runnable for
    research and CI smoke tests
  - short 512-bar context limit is easy to silently violate; the adapter
    must fail fast instead of truncating
  - large model checkpoint is closed-weight; do not depend on it

### 2. MemPalace (agent memory backend)

- Target package: `ai_agents/runtime/memory/mempalace/`.
- Entry criteria: Phase 1 shared telemetry contracts are available,
  Phase 5 entry criteria met, ADR-0004 accepted or implementation-ready.
- Tasks:
  - Wrap the MemPalace Python client behind an `AgentMemoryBackend`
    interface under `ai_agents/runtime/memory/base.py`.
  - Treat MemPalace only as an optional backend; the default
    `AgentMemoryBackend` must be a local in-process stub so no agent
    requires an external install for CI.
  - Keep the temporal knowledge graph strictly scoped to agent traces,
    not to execution, OMS, or custody systems.
  - Disable or gate the MCP server surface so agent-to-agent memory
    cannot be exposed to external tools without explicit configuration.
  - Never persist secrets, private keys, or broker credentials into the
    memory backend; enforce this through a schema allow-list.
- Exit criteria:
  - the memory interface has at least one in-process backend and one
    MemPalace-backed backend, selectable by configuration
  - ADR-0004 permissions map agent roles to memory scopes explicitly
  - agent traces stay within `ai_agents/` and never reach
    `trading_system/` module state
- Risk flags:
  - impostor domains reported upstream; installs must be pinned to the
    official GitHub or PyPI package
  - pluggable vector backend pulls heavy dependencies; keep behind the
    `agent-memory` extra

### 3. Vibe-Trading (multi-agent workspace)

- Target behavior: **cherry-pick only**. Vibe-Trading ships an opinionated
  runtime that overlaps heavily with `ai_agents/`, `backtest_engine/`, and
  `alpha_research/`. Adopting it wholesale would collapse three planned
  packages into one upstream framework.
- Entry criteria: Phase 3 factor library contracts stable, Phase 4
  backtest engine modularized, Phase 5 agent runtime primitives shipped.
- Tasks:
  - Identify which upstream modules are actually wanted:
    - selected finance skills that map cleanly to factors
    - swarm presets that can be re-expressed as agent role graphs
    - exporters for Pine Script, TDX, MQL5 that can live in a small
      adapter under `backtest_engine/analytics/exports/`
  - Vendor the chosen files with explicit upstream attribution and the
    upstream commit hash recorded in each file header.
  - Do not introduce `vibe-trading-ai` as a runtime Poetry dependency.
  - Never import Vibe-Trading's backtest engines from
    `backtest_engine/`; if a specific engine is needed, rewrite it
    against the existing simulator contracts.
- Exit criteria:
  - cherry-picked modules compile and test without pulling the upstream
    package
  - upstream attribution, license, and commit are recorded in every
    vendored file
  - no part of `trading_system/` imports anything originating from
    Vibe-Trading
- Risk flags:
  - heavy LLM-provider surface; do not inherit it
  - A-share and Chinese-market bias in many skills; curate carefully
  - upstream scope collision is the main risk, not technical difficulty

### 4. rtk-ai (developer tooling)

- Target behavior: **developer-environment tool only**. `rtk` belongs in
  the developer ergonomics layer, not in any monorepo package.
- Entry criteria: none, since it does not enter the Python runtime. May
  be adopted at any time as a developer-environment decision under
  Phase 9's dev-tooling track.
- Tasks:
  - Document installation instructions in `DEVELOPMENT.md` as optional.
  - Add no Poetry dependency, no Docker image layer, and no CI step.
  - If adopted for CI, gate it behind an explicit workflow step that is
    allowed to fall back to raw command output when `rtk` is missing.
- Exit criteria:
  - the repo builds and tests pass with and without `rtk` installed
  - no workflow silently depends on `rtk` being present
- Risk flags:
  - pure developer productivity tool; do not architect around it
  - Windows native mode degrades to a CLAUDE.md injection shim, not a
    real command wrapper

### 5. NVIDIA Quantitative Portfolio Optimization blueprint

- Target packages:
  - `alpha_research/optimizers/gpu_cvar/` for the Mean-CVaR solver
    adapter
  - `backtest_engine/analytics/optimizers/cvar_adapter.py` for invoking
    the solver from the simulator
- Entry criteria: Phase 3 entry criteria met for research; integration
  with Phase 4 is only allowed once the backtest engine has a stable
  optimizer plug-in contract.
- Tasks:
  - Wrap the CVaR solver as a `PortfolioOptimizer` implementation in
    `alpha_research/optimizers/base.py`.
  - Expose the adapter behind a Poetry extra `gpu-cvar` so CPU-only
    developer and CI paths stay unaffected.
  - Provide a reference Docker image under
    `docker/quant-gpu.Dockerfile` that builds on top of the NVIDIA
    PyTorch container. Keep the default `docker-compose` runtime CPU-only.
  - Refactor the upstream notebook logic into a pure-function module:
    scenarios generated from a Polars/Pandas bar frame, solver invoked
    through a typed configuration object, results returned as a
    portfolio-weight frame with the shared Phase 1 schema.
  - Provide a CPU fallback `PortfolioOptimizer` (for example convex-ish
    baseline or simple risk parity) so backtests remain runnable without
    a GPU.
- Exit criteria:
  - CPU fallback exists and is used by default in CI
  - GPU path runs on a dedicated container image, not the default app
    container
  - solver invocations pass through the shared contracts for portfolio
    weights, not raw DataFrames
- Risk flags:
  - recommended hardware is H100 class; no assumption that contributors
    or CI have it
  - `cuOpt` licensing must be reviewed before the GPU path is enabled in
    any automated deployment
  - notebook structure is not a library; adapter work is non-trivial

### 6. FinRL (financial deep reinforcement learning)

- Target package: `alpha_research/ml_models/rl/finrl/`.
- Entry criteria: Phase 3 entry criteria met.
- Tasks:
  - Wrap FinRL data processors behind the data-platform connector
    contracts from Phase 2 so FinRL does not bypass the feature store or
    cache policy.
  - Expose FinRL environments as `alpha_research` training artifacts,
    not as part of the live trading path. RL policies must be promoted
    through the same factor and model promotion gates as any other
    `alpha_research` artifact.
  - Explicitly forbid FinRL's built-in broker integrations
    (Alpaca and Binance glue) from being invoked by the trading system.
    `trading_system/` must keep its own OMS/EMS/RMS pipeline per Phase 6.
  - Record FinRL's legacy status in the adapter docstring and plan a
    follow-up ADR for whether to migrate to FinRL-X before any policy is
    promoted to paper trading.
- Exit criteria:
  - FinRL policies can be trained and evaluated offline against shared
    schemas
  - no FinRL code path touches the live or paper trading execution path
  - upstream trademark restrictions on the "FinRL" name are honored in
    docs and packaging
- Risk flags:
  - upstream is in legacy maintenance; plan for migration to FinRL-X
  - Gym and Stable-Baselines3 versions drift quickly; pin them in the
    `finrl-rl` extra
  - RL policies are easy to over-fit; they must go through the existing
    leakage, out-of-sample, and walk-forward gates

### 7. MiroFish (swarm intelligence simulation)

- Target behavior: **blocked pending license review**. Until the upstream
  license is confirmed compatible with the repository's policy, MiroFish
  cannot enter the runtime.
- Entry criteria:
  - upstream LICENSE file confirmed and recorded in
    `docs/adr/0007-third-party-integration-governance.md`
  - Phase 5 agent runtime primitives shipped
  - Zep Cloud dependency replaced or made optional with a local stub
    before any runtime adoption
- Tasks when unblocked:
  - Extract the GraphRAG and scenario-simulation primitives only; do not
    adopt the full Docker Compose application.
  - Wrap the scenario engine behind an `ai_agents/scenarios/` adapter
    with an explicit rate and cost limiter (LLM call budget per run).
  - Never use MiroFish output to modify OMS state directly; scenario
    outputs are analytical context for researchers and the
    `risk_monitor` agent, not execution signals.
- Exit criteria:
  - license confirmed and recorded
  - scenario simulator runs with budgeted LLM calls and deterministic
    seed where possible
  - no direct coupling to `trading_system/` state
- Risk flags:
  - license is currently unverified; this is a hard blocker
  - upstream is LLM-call heavy and expensive to run at scale
  - Zep Cloud is a paid external service; must be optional

## Execution Order Relative to Existing Phases

The integrations slot into the existing phased plan without altering the
high-level order:

- Phase 3 (Alpha Research): Kronos adapter, FinRL adapter, NVIDIA QPO
  research notebooks. Each lands as a separate PR behind its own
  optional extra.
- Phase 4 (Backtest Engine): NVIDIA QPO solver plugged into the
  portfolio-optimizer contract, Vibe-Trading exporter cherry-picks into
  `backtest_engine/analytics/exports/`.
- Phase 5 (AI Agents): MemPalace as an optional memory backend,
  Vibe-Trading swarm-role cherry-picks into agent role graphs, MiroFish
  only if license review unblocks.
- Phase 9 (Infrastructure and developer tooling): `rtk` documented as an
  optional developer tool; GPU Docker image added for the QPO extra.

No integration is permitted to run ahead of its phase's entry criteria,
and no integration is permitted to bypass the shared contracts, kill
switches, or custody rules defined in ADR-0004 and ADR-0006.

## Open Questions

- Confirm MiroFish upstream license before any further planning work.
- Decide whether FinRL-X supersedes FinRL for this repository before any
  live promotion path is designed.
- Decide whether NVIDIA QPO's CPU fallback should be risk parity, a
  small convex solver, or a shared baseline optimizer.
- Decide whether Vibe-Trading's Pine Script and TDX exporters are worth
  vendoring given that the current `reporting/` layer already emits a
  TradingView export.
