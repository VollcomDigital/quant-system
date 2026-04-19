# Web Control Plane (Phase 6)

- Status: Accepted (Phase 6)
- Source ADRs: ADR-0004 (control plane), ADR-0006 (kill switch)
- Target phases: Phase 6, extended in Phase 7 / 9.

## Purpose

Phase 6 extends the control plane with **execution oversight** for
paper and live trading. Operators can see OMS / EMS / RMS state, halt
trading, reset the kill switch (after an approval), and run the panic
playbook. Every mutating call is bounded, authenticated, and audited.

This doc extends `web-shell-phase-4.md` (read-only runs) and
`web-control-plane-phase-5.md` (approvals).

## In Scope for Phase 6

- **Read-only status** (`GET /v1/execution/status`): OMS open orders,
  positions, daily PnL, reconciliation state, `trading_halted` flag.
- **Halt trading** (`POST /v1/execution/halt`): operator role +
  authenticated user + audit event.
- **Reset kill switch** (`POST /v1/execution/kill_switch/reset`):
  operator role + a referenced `approval_id` for
  `subject='kill_switch_reset'`.
- **Panic playbook** (`POST /v1/execution/panic`): triggers halt +
  cancels all open orders through the Phase 7 gateway layer; does not
  itself place orders.
- **Alerts and incidents** feed (`GET /v1/alerts`): lists `AnomalyEvent`
  + `ValidationResult` findings produced by risk monitor + RMS.

## Out of Scope

- **No raw browser-driven trade entry.** The execution API has no
  `/orders` POST. No `submit_order`, no `place_order`, no direct OMS
  mutation from the browser. This is enforced statically by
  `tests/phase_6/test_execution_oversight.py::test_execution_api_has_no_submit_order_endpoint`.
- **No broker credentials in the browser.** Credentials live behind
  KMS / Vault (ADR-0006). The backend never returns them; the
  frontend never holds them.
- **No direct KMS signing.** Phase 9 owns KMS wiring.

## Authentication and RBAC

- Every endpoint requires an authenticated session.
- Roles:
  - `viewer`   — read-only run browsing (Phase 4 surface) + status.
  - `approver` — approve / reject approval requests (Phase 5).
  - `operator` — halt trading, reset kill switch, trigger panic
    playbook. Phase 6 introduces this role.
- All Phase 4/5 session rules apply (short-lived, `HttpOnly` cookies,
  `SameSite=Strict`, mTLS between backend services).

## Read-Only Status Endpoints

- `GET /v1/execution/status` — `ExecutionStatusResponse`
  (trading_halted, open_orders, positions, daily_pnl_pct,
  last_heartbeat).
- `GET /v1/execution/orders` — paginated `Order` list from the OMS.
- `GET /v1/execution/fills` — paginated `Fill` list from the OMS.
- `GET /v1/execution/rms/checks` — latest per-order RMS
  `ValidationResult`s so operators can see which check blocked a
  rejected order.

## Bounded Mutating Endpoints

Every mutating endpoint:

1. Parses a pydantic request body.
2. Verifies the authenticated user exists.
3. Verifies the `operator` role.
4. Calls the Phase 6 handler which, in turn, calls `KillSwitch` or
   the gateway (Phase 7) — **never a direct broker API**.
5. Lets the `KillSwitch` / `ApprovalQueue` emit the audit event.

Defined handlers:

- `handle_halt_trading` — sets `TRADING_HALTED`; returns 204.
- `handle_reset_kill_switch` (Phase 6.1) — gated behind an approved
  `kill_switch_reset` approval; unblocks the flag.
- `handle_panic_playbook` (Phase 6.1) — runs `PanicPlaybook.execute`
  using a gateway cancel-all callback.

## Audit Requirements

- `KillSwitch.trigger` / `.reset` emit `AuditEvent`s with
  `action = kill_switch.triggered` / `kill_switch.reset`, `actor =
  system:<role>` or `user:<id>`, `target = trading_system`,
  `details = {"reason": ..., "approval_id": ...}`.
- Panic playbook execution emits a summary event with the set of
  cancelled order ids.
- Audit events are queryable at `GET /v1/audit?target=trading_system`.

## Enforcement

- `tests/phase_6/test_execution_oversight.py` locks the doc contract +
  the "no submit order" static invariant.
- The Phase 5 `code_reviewer` agent treats any PR that adds a trade-
  entry handler (`submit_order` / `place_order` / raw broker call)
  to `web_control_plane.backend.api` as a blocking review finding.
- Changes to this document require an ADR update (ADR-0004 or a
  superseder).
