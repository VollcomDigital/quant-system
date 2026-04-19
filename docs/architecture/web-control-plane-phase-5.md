# Web Control Plane (Phase 5)

- Status: Accepted (Phase 5)
- Source ADRs: ADR-0004 (agent permissions / control plane)
- Target phases: Phase 5 (this doc), extended in Phase 6 / 7 / 9.

## Purpose

Phase 5 adds the first *mutating* endpoints to the web control plane:
the approval console. Operators can submit and decide approvals for
factor promotion, model promotion, strategy activation, and
kill-switch resets. Every action is authenticated, audited, and
capability-bound.

This doc extends `web-shell-phase-4.md` which defines the read-only
routes that continue to ship in Phase 5.

## In Scope for Phase 5

- `POST /v1/approvals` — submit an approval request
  (`SubmitApprovalRequest`), typically from an agent or operator.
- `POST /v1/approvals/{approval_id}/decision` — record the human
  decision (`DecideApprovalRequest`). Approver role required.
- `GET /v1/approvals` — list pending approvals (paginated).
- `GET /v1/approvals/{approval_id}` — single approval detail with its
  decision history and AuditEvent trail.
- Agent finding triage views backed by `ValidationResult` payloads from
  `ai_agents.code_reviewer`, `ai_agents.guardrails`, and
  `data_platform.quality`.

## Out of Scope

- **No execution controls**. Halting, cancelling, and modifying orders
  still ships in Phase 6 alongside OMS/EMS/RMS.
- **No broker credentials in the browser**. The browser never holds
  broker credentials, exchange credentials, private keys, or signing
  authority. Backend endpoints never return raw credentials.
- **No direct KMS signing**. Any signing flow lives behind ADR-0006 and
  uses KMS/HSM; the web app is only a display + approval surface.

## Authentication and RBAC

- Every endpoint requires an authenticated session. Unauthenticated
  requests are refused with `PermissionError` / HTTP 401.
- Roles, in ascending power:
  - `viewer`   — can read runs, tear sheets, approvals, audit trails.
  - `approver` — can submit + decide approvals.
  - `operator` — Phase 6 only; can trigger bounded halts + panic
    workflows.
- Session tokens are short-lived (≤ 30 min idle, ≤ 8 h absolute) and
  delivered via `HttpOnly`, `Secure`, `SameSite=Strict` cookies.
- Service-to-service calls use mTLS per the Phase 0 transport matrix.

## Mutating Endpoints

Every mutating endpoint follows this template:

1. Validate the request body via pydantic
   (`SubmitApprovalRequest` / `DecideApprovalRequest`).
2. Verify the authenticated user exists (else refuse).
3. Verify the user's roles include the minimum required role for the
   endpoint.
4. Call the `ai_agents.approvals.ApprovalQueue` handler.
5. Let the queue emit the `AuditEvent`; the endpoint never bypasses
   audit.

## Audit Requirements

- Every state transition (`approval.submitted`, `approval.decided`)
  produces a `shared_lib.contracts.AuditEvent` stored in the same
  backend as the approvals.
- Audit events capture the authenticated user id (`actor =
  "user:<id>"`), the target (`target = "approval:<id>"`), and the
  action. An AuditEvent with an empty actor is a bug.
- The Phase 5 control plane exposes `GET /v1/audit?target=...` for
  reviewers and the risk monitor.

## Enforcement

- Every mutating route has an integration test that verifies (a) the
  unauthenticated path refuses, (b) the role-gated path refuses the
  wrong role, and (c) the happy path emits exactly one `AuditEvent`.
- `tests/phase_5/test_web_control_plane.py` locks in the contract-level
  invariants.
- The Phase 5 `code_reviewer` agent treats any PR that adds a mutating
  endpoint without an auth check or without an audit path as a
  blocking review finding.
- Changes to this document require an ADR update (ADR-0004 or a
  superseder).
