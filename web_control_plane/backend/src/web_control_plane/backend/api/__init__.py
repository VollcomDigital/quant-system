"""Web control plane API contracts + handlers.

Phase 5 ships typed request models and handler functions. A FastAPI
transport layer is a Phase 9 deliverable. Every mutating handler:

- Refuses unauthenticated requests (`authenticated_user is None`).
- Enforces RBAC (approver role required for decisions).
- Emits an `AuditEvent` via the `ApprovalQueue` it writes through.
- Never touches OMS/KMS/treasury directly.
"""

from __future__ import annotations

from datetime import datetime

from ai_agents.approvals import ApprovalQueue
from pydantic import Field
from shared_lib.contracts import ApprovalDecision, ApprovalRequest
from shared_lib.contracts._base import Schema, aware_datetime_validator

__all__ = [
    "DecideApprovalRequest",
    "SubmitApprovalRequest",
    "handle_decide_approval",
    "handle_submit_approval",
]


class SubmitApprovalRequest(Schema):
    approval_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    requested_by: str = Field(min_length=1)
    requested_at: datetime
    context: dict[str, str] = Field(default_factory=dict)

    _ts = aware_datetime_validator("requested_at")


class DecideApprovalRequest(Schema):
    approval_id: str = Field(min_length=1)
    decision: str = Field(min_length=1)
    decided_by: str = Field(min_length=1)
    decided_at: datetime
    notes: str = ""

    _ts = aware_datetime_validator("decided_at")


def handle_submit_approval(
    *,
    request: SubmitApprovalRequest,
    authenticated_user: str | None,
    approval_queue: ApprovalQueue,
) -> None:
    if authenticated_user is None:
        raise PermissionError("authentication required")
    approval_queue.submit(
        ApprovalRequest(
            approval_id=request.approval_id,
            subject=request.subject,  # type: ignore[arg-type]
            target_id=request.target_id,
            requested_by=request.requested_by,
            requested_at=request.requested_at,
            context=request.context,
        )
    )


def handle_decide_approval(
    *,
    request: DecideApprovalRequest,
    authenticated_user: str | None,
    user_roles: tuple[str, ...],
    approval_queue: ApprovalQueue,
) -> ApprovalDecision:
    if authenticated_user is None:
        raise PermissionError("authentication required")
    if "approver" not in user_roles:
        raise PermissionError("approver role required to decide approvals")
    return approval_queue.decide(
        approval_id=request.approval_id,
        decision=request.decision,
        decided_by=request.decided_by,
        notes=request.notes,
        decided_at=request.decided_at,
    )
