# Web Shell Scope (Phase 4)

- Status: Accepted (Phase 4)
- Source ADRs: ADR-0004 (agent permissions / control plane)
- Owner: web_control_plane backend + frontend
- Target phases: Phase 4 (this doc), extended in Phase 5 / Phase 6.

## Purpose

Phase 4 turns the legacy FastAPI dashboard into an *authenticated
initial* web shell. The shell gives operators a read-only window into
managed backtest runs, their tear sheets, and their provenance. It is
the foundation for Phase 5's approval queues and Phase 6's execution
oversight, but it intentionally ships **no execution controls** in this
phase.

## In Scope for Phase 4

- **Run browsing** — list managed `RunMetadata` records by strategy,
  factor_id, and date range. Read through the Phase 1 contracts.
- **Run comparison** — side-by-side tear sheets (sharpe, max_drawdown,
  cagr, volatility) from `backtest_engine.analytics.tear_sheet`.
- **Report access** — links to the tear sheet payloads and equity
  curves persisted through `data_platform.storage.SnapshotIndex`.
- **Provenance views** — for each run: the git sha, ADR references,
  factor and model versions, training snapshot ids, and the
  recorded `OrderPayload` replay buffer (so reviewers can see the
  exact bytes the simulator emitted).

## Explicitly Out of Scope

- **No execution**. The web shell cannot place, cancel, halt, or modify
  orders in Phase 4. Any control-plane mutation of live trading is a
  Phase 6 feature, gated by OMS/EMS/RMS which do not exist yet.
- **No direct model promotion**. Factor and model promotion still goes
  through `alpha_research.promotion.promote_factor` /
  `promote_model` and the Phase 5 approval queue — the web shell can
  *display* the outcome but cannot *trigger* a promotion in Phase 4.
- **No broker credentials in the browser**. The browser never holds
  broker credentials, exchange credentials, private keys, or signing
  authority (Non-Negotiable Guardrail from the roadmap).
- **No direct writes** to Parquet snapshots, factor stores, or model
  registries. Writes happen on the backend through the Phase 2/3
  contracts.

## Authentication Requirements

- Every route requires an authenticated session. Anonymous access is
  refused at the load balancer / API gateway.
- The backend enforces RBAC tags: `viewer`, `approver`, `operator`.
  Phase 4 endpoints need only `viewer`.
- Sessions are short-lived (≤ 30 min idle, ≤ 8 h absolute). Refresh
  tokens are server-only; the browser holds the access token in a
  `HttpOnly`, `Secure`, `SameSite=Strict` cookie.
- Every mutating action — even in later phases — traces back to an
  authenticated user id. Phase 4 has no mutating actions, but the
  auth scaffold must be in place.

## Routes

All routes return JSON or HTML; none are mutating in Phase 4.

- `GET /v1/runs` — list `RunMetadata`; query params: `strategy_id`,
  `factor_id`, `from`, `to`, `limit`, `cursor`.
- `GET /v1/runs/{run_id}` — full detail.
- `GET /v1/runs/{run_id}/tear-sheet` — `TearSheet` payload.
- `GET /v1/runs/{run_id}/equity-curve` — `(timestamp, equity)` pairs.
- `GET /v1/runs/{run_id}/payload-replay` — raw `OrderPayload` NDJSON
  buffer (read-only for reviewers).
- `GET /v1/factors` / `GET /v1/factors/{factor_id}` — factor definitions
  + validation status; **read-only**.
- `GET /v1/models` / `GET /v1/models/{model_id}` — model registry
  records; **read-only**.
- `GET /v1/health` — liveness + build metadata.

## Enforcement

- Every route is an integration-tested authenticated endpoint; no
  unauthenticated path ships.
- The Phase 5 `code_reviewer` agent treats any PR that adds a mutating
  endpoint to the web shell before Phase 6 / Phase 10 as a blocking
  review finding.
- ADR-0004 owns the permission model for the web control plane; any
  change to the Phase 4 shell scope requires an ADR update.
- Changes to this document require an ADR update (ADR-0004 or a
  superseder) plus a matching update to
  `tests/phase_4/test_web_shell_scope.py`.
