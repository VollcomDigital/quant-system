# ADR-0007: Third-Party Integration Governance

## Status

Proposed.

## Context

The monorepo migration plan in `tasks/todo.md` introduces several target
packages (`alpha_research/`, `ai_agents/`, `backtest_engine/`,
`trading_system/`, `shared_lib/`) with strict boundary and kill-switch
rules defined by ADR-0001 through ADR-0006.

A set of external open-source projects is being considered for adoption
in Phases 3 through 9:

- shiyu-coder/Kronos (OHLCV forecasting foundation model)
- MemPalace/mempalace (agent memory backend)
- HKUDS/Vibe-Trading (multi-agent trading workspace)
- rtk-ai/rtk (developer CLI output compressor)
- NVIDIA-AI-Blueprints/quantitative-portfolio-optimization (GPU Mean-CVaR)
- AI4Finance-Foundation/FinRL (finance DRL framework)
- 666ghj/MiroFish (multi-agent scenario simulation)

Without a governance decision, there is a real risk that these projects
are imported directly into `trading_system/`, `backtest_engine/`, or
`src/`, bypassing the shared contracts, telemetry, custody, and kill-
switch rules that the earlier ADRs exist to enforce.

This ADR records the governance decision for how external projects enter
the monorepo.

## Decision

Third-party projects enter the monorepo through adapter modules only.

- Every external project is wrapped behind an adapter inside the
  appropriate internal package:
  - model repos (Kronos, FinRL) under
    `alpha_research/ml_models/forecasters/` or
    `alpha_research/ml_models/rl/`
  - agent infrastructure (MemPalace) under `ai_agents/runtime/memory/`
  - portfolio optimizers (NVIDIA QPO) under
    `alpha_research/optimizers/` with a thin invocation adapter under
    `backtest_engine/analytics/optimizers/`
  - scenario or agent simulators (MiroFish) under
    `ai_agents/scenarios/` only after license review
  - upstream bundles that overlap the target architecture (Vibe-Trading)
    are cherry-picked at file granularity with attribution, never
    installed as a runtime dependency
  - developer tools (rtk) stay outside Python packaging entirely and
    are documented as optional developer-environment conveniences
- No module under `trading_system/` may import an external package
  directly. Execution-critical modules import only the adapter contract
  from `shared_lib/` or `alpha_research/` and never the upstream package.
- All external projects are installed through Poetry optional extras,
  never through the base dependency set, unless the project is a pure
  developer tool that is not part of the runtime.
- GPU-gated integrations are shipped through a separate Docker image and
  are excluded from the default `docker-compose` runtime.
- Every adapter module records, in its docstring and in
  `docs/architecture/third-party-integration-plan.md`:
  - upstream repository URL
  - upstream commit or tag that was reviewed or vendored
  - upstream license
  - review date
- Any external project whose license cannot be verified is blocked from
  runtime adoption. Prototyping is permitted only under
  `alpha_research/notebooks/` and only with explicit ADR approval.
- ADR-0004 (agent permissions) and ADR-0006 (custody, signing, and kill
  switches) continue to apply to any integration that ships agent loops,
  model inference paths, or execution logic. No integration is allowed
  to bypass kill-switch, RMS, or custody rules.

## Consequences

- Adding a new external project is a reviewable ADR-linked change, not
  an opportunistic `pip install`.
- The shared schemas and telemetry conventions from Phase 1 stay
  authoritative; external projects are forced to translate at the
  adapter boundary.
- Some adoption cost is paid up front for each integration (adapter
  module, optional extra, license review). That cost is accepted as the
  price of keeping execution boundaries intact.
- Projects that cannot reasonably be wrapped (Vibe-Trading's bundled
  runtime, MiroFish's Docker Compose app) either degrade to cherry-picks
  or stay out of the repository until their scope can be cleanly split.

## Alternatives Considered

- Importing external projects directly as top-level Poetry dependencies.
  Rejected because it collapses the boundary between research,
  execution, and agent runtime that ADR-0001 and ADR-0003 define.
- Forking each external project into internal repositories. Rejected as
  premature; most integrations benefit from tracking upstream, and only
  Vibe-Trading and MiroFish warrant file-level vendoring today.
- Treating each integration as a per-PR ad hoc decision. Rejected
  because it recreates the governance gap that this ADR exists to close.

## Related ADRs

- ADR-0001: Monorepo Workspace and Package Boundaries
- ADR-0003: Two-Speed Execution Runtime Boundaries
- ADR-0004: Agent Permissions and Control Plane
- ADR-0006: Execution Signing, Custody, and Kill Switches

## References

- `docs/architecture/third-party-integration-plan.md`
- `tasks/todo.md`
