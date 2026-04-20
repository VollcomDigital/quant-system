# CLI Compatibility Facade

- Status: Accepted (Phase 0)
- Source ADRs: ADR-0001 (package boundaries)
- Applies to: `src/main.py`, root `pyproject.toml`, and every domain package
  that eventually absorbs a legacy `src/*` responsibility.

## Purpose

Phase 0 introduces the monorepo skeleton without breaking the existing
application. That promise depends on a clear compatibility facade: the
legacy CLI must keep working for users while new domain packages grow in
parallel. This document defines the facade, the rules that keep it honest,
and the retirement plan that closes the loop in Phase 10.

## Current CLI Surface

The user-facing CLI lives at `src/main.py`. Its public surface includes:

- the Typer/Click-based command tree exposed by `src.main`;
- the current `src/backtest/runner.py`-backed backtest workflow;
- the current `src/dashboard/` FastAPI entrypoint;
- the current data-source routing through `src/data/*`;
- the current reporting pipeline under `src/reporting/*`.

This surface is frozen for Phase 0. No Phase 0 change deletes, renames, or
restructures these modules. The legacy CLI remains the default way to run
backtests, start the dashboard, and export reports.

## Migration Strategy

New domain packages (`shared_lib`, `data_platform`, ...) grow in parallel
with `src/`. When a responsibility moves to a new package, the migration
follows this staged model:

1. **Greenfield implementation** — the new package ships its own
   implementation under its canonical import path (e.g.
   `shared_lib.logging`).
2. **Compatibility adapter** — a thin shim inside `src/` is rewritten to
   import from the new package. The legacy module name stays stable for
   callers; only the implementation moves.
3. **Parity verification** — contract tests demonstrate equivalence
   between the new and legacy code paths. Coverage does not drop.
4. **Deprecation notice** — the legacy module emits a
   `DeprecationWarning` pointing to the new import path.
5. **Retirement** — only in Phase 10 is the legacy module removed.

During stages 1–4 the user-facing CLI behaviour does not change.

## Allowed Legacy Imports

- Files physically under `src/` may import other `src.*` modules (no
  restriction; this is the legacy package).
- Compatibility adapters inside `src/` may import from new domain
  packages (e.g. `from shared_lib.logging import get_logger`) and
  re-export the symbols that legacy callers expect.
- **No domain package (`shared_lib`, `data_platform`, `backtest_engine`,
  `alpha_research`, `ai_agents`, `trading_system`) may import `src.*`.**
  This is enforced by `tests/phase_0/test_cli_compatibility_facade.py`.
- Tests under `./tests/` may import `src.*` because they exercise the
  legacy surface. Tests under `tests/phase_0/` must not.

Any PR that needs a new shim must name it in the PR description and keep
it inside `src/`.

## Retirement Plan

Retirement is scheduled for Phase 10, not earlier. A legacy module is
eligible for deletion only when **all** of the following hold:

1. A new package fully owns the functionality.
2. Contract tests between research → backtest, backtest → OMS/EMS, and
   OMS/EMS → gateways demonstrate parity.
3. Coverage for the new implementation meets or exceeds the legacy
   coverage baseline.
4. A one-release deprecation window has elapsed with
   `DeprecationWarning` in place.
5. The Phase 10 parity validation report records the retirement.

The root `pyproject.toml` continues to include `src` as a package until
Phase 10 cutover. Removing that include is a Phase 10 action gated by the
`## Retirement Plan` section of the Phase 10 todo.

## Phase 10 Cutover

The binding cutover plan — the concrete removal schedule for every
legacy `src/*` module — lives in
[`phase-10-parity-report.md`](./phase-10-parity-report.md). That
document is the authoritative record of:

- which legacy module is at which retirement stage;
- the coverage deltas between the legacy implementation and the new
  domain package;
- the contract tests under `tests/phase_10/` that prove parity;
- the `DeprecationWarning` rollout schedule.

The root `pyproject.toml` keeps `src` in its packages list until the
parity report lists every legacy module at stage 5 (removed). Any PR
that touches `pyproject.toml`'s packages entry must also update the
parity report.

## Enforcement

- `tests/phase_0/test_cli_compatibility_facade.py` verifies:
  - the legacy CLI entrypoint still exists on disk;
  - the root `pyproject.toml` still ships the `src` package;
  - no domain package imports `src.*`.
- The Phase 5 `code_reviewer` agent treats a missing compatibility shim
  (i.e. a legacy import broken by a refactor) as a blocking review
  finding.
- Pre-commit runs the Phase 0 invariant suite so drift is caught before
  push.
- Any change to this document requires an ADR update (ADR-0001 or a
  superseder).
