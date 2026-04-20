# Phase 10 Parity Validation Report

- Status: Accepted (Phase 10)
- Source ADRs: ADR-0001 (package boundaries), ADR-0003 (two-speed
  runtime), ADR-0004 (kill-switch architecture), ADR-0005 (gateway
  architecture), ADR-0006 (credentials and signing)
- Applies to: monorepo cutover, legacy `src/*` retirement plan, and
  all cross-phase contract tests under `tests/phase_10/`.

## Purpose

Phase 10 is the integration phase: every prior phase delivered its
own contract (`shared_lib.contracts`, `backtest_engine.api`,
`trading_system.oms`, etc.), and Phase 10 proves the contracts
compose across package boundaries without mocks or adapters
written ad-hoc for the test. This document records the evidence,
the retirement status of every legacy `src/*` module, and the
gates that control the final cutover.

## Contract Test Coverage

Every cross-phase seam is exercised by a dedicated pytest module
under `tests/phase_10/`. The suite is reproducible end-to-end:

| Boundary                       | Test module                                              | Tests |
| ------------------------------ | -------------------------------------------------------- | ----- |
| Research → backtest            | `test_research_to_backtest_contract.py`                  | 4     |
| Backtest → OMS / EMS           | `test_backtest_to_oms_ems_contract.py`                   | 5     |
| OMS / EMS → gateways           | `test_oms_ems_to_gateways_contract.py`                   | 5     |
| Agents → CI / live telemetry   | `test_agents_to_ci_telemetry_contract.py`                | 7     |
| Parity + CLI cutover           | `test_parity_report_and_cli_cutover.py`                  | 5     |

All suites run on Python 3.12/3.13 via the Phase 9 CI matrix.
The shared contract vocabulary — `ValidationResult`, `AuditEvent`,
`OrderPayload`, `Order`, `Fill`, `HealthStatus`, `AnomalyEvent`,
`ApprovalRequest` — is the single source of truth across every
seam. Agents, backtest, OMS, EMS, and gateways all emit and consume
the exact pydantic models from `shared_lib.contracts` without
copy-paste duplicates.

## Retirement Status of `src/*` Modules

The compatibility facade (see `cli-compatibility-facade.md`) lists
the retirement prerequisites. Phase 10 records the per-module state:

- `src/backtest/*` — **stage 3 (parity verification complete).** The
  Phase 4 `backtest_engine.api` now owns the simulator and portfolio
  accounting. Phase 10 contract tests prove the `OrderPayload` wire
  shape and the partial→filled state machine match the new
  `trading_system.oms` exactly.
- `src/data/*` — **stage 2 (compatibility adapters in place).** The
  Phase 2 `data_platform.storage`, `data_platform.connectors`,
  `data_platform.pipelines`, `data_platform.feature_store`, and
  `data_platform.indexing` packages own the new implementations.
  Legacy import paths remain stable while greenfield callers use the
  Phase 2 packages directly.
- `src/reporting/*` — **stage 2.** The Phase 0 compatibility-facade
  doc pre-authorised `web_control_plane.backend` as the eventual
  owner; HTML report generation continues to live in `src/reporting/`
  for the duration of the deprecation window.
- `src/dashboard/*` — **stage 2.** Phase 5 introduces the
  `web_control_plane.backend.api` tree with agent/approval/execution
  endpoints. Dashboard cutover is scheduled with the Phase 10
  deprecation-warning batch.

## Coverage, Lint, and Hook Gates

Phase 10 inherits the Phase 9 CI gates:

- `ruff check .` — zero findings on the canonical path.
- `pre-commit run --all-files` — all hooks green.
- Contract-test coverage — every new domain package keeps the
  ≥80% coverage requirement established in Phase 0.
- Phase 9 `container-scan`, `iac-scan`, `factor-promotion`,
  `model-deployment`, and `staged-deploy` workflows retain their
  `environment:` approval gates.

## Phase-by-Phase Test Counts

| Phase | Cumulative tests |
| ----- | ----------------:|
| Phase 0  | 98   |
| Phase 1  | 211  |
| Phase 2  | 291  |
| Phase 3  | 353  |
| Phase 4  | 419  |
| Phase 5  | 497  |
| Phase 6  | 565  |
| Phase 7  | 623  |
| Phase 8  | 660  |
| Phase 9  | 696  |
| Phase 10 | 722+ |

## Retirement Gates (binding)

Before the root `pyproject.toml` removes `src` from its packages
list, **all** of the following must hold:

1. A new package fully owns the functionality (see the per-module
   retirement table above).
2. Contract tests cover every seam that crossed the old and new
   implementations. Phase 10 adds one test module per seam.
3. Coverage for the new implementation meets or exceeds the legacy
   coverage baseline.
4. A one-release deprecation window has elapsed with
   `DeprecationWarning` in place on every legacy import path.
5. This parity report lists the retirement status as `stage 5`
   (removed) for that module.

The next Phase 10 deprecation-warning batch flips `src/backtest/*`
to stage 4 (`DeprecationWarning` issued) because its parity proof
is complete; the remaining modules move to stage 4 only after the
Phase 10 deprecation-warning batch lands.

## Enforcement

- `tests/phase_10/test_parity_report_and_cli_cutover.py` verifies
  this document stays in sync with the CLI compatibility facade.
- The Phase 5 `code_reviewer` agent flags any PR that drops a
  contract test or removes a `ValidationResult` without a
  superseder.
- Updates to this document require an ADR update (ADR-0001 or a
  superseder).
