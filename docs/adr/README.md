# Architecture Decision Records

## Purpose

These ADRs capture the highest-risk architectural decisions required to turn the
current single-package backtesting application into the target quant monorepo.

They are intentionally lightweight stubs for Phase 0. Each ADR should be filled
in before the corresponding implementation phase starts and should be updated
when a material architectural choice changes.

## ADR Workflow

1. Create or refine the ADR before implementation starts.
2. Record the chosen option and why competing options were rejected.
3. Link the ADR from `tasks/todo.md` and any PR that implements it.
4. Update status as the decision matures:
   - `Proposed`
   - `Accepted`
   - `Superseded`
   - `Deprecated`

## Stub Set

- `0001-monorepo-workspace-and-package-boundaries.md`
- `0002-data-platform-orchestration-and-immutability.md`
- `0003-two-speed-execution-runtime-boundaries.md`
- `0004-agent-permissions-and-control-plane.md`
- `0005-tradfi-and-web3-gateway-architecture.md`
- `0006-custody-signing-and-kill-switch-architecture.md`
