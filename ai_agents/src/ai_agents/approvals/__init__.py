"""Human approval workflow.

Wraps `shared_lib.contracts.ApprovalRequest` / `ApprovalDecision` with
an in-memory queue that emits `AuditEvent`s for every state change.
Phase 9 swaps the storage backend; the Phase 5 control plane consumes
this queue through authenticated REST.
"""

from __future__ import annotations

import secrets
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime

from shared_lib.contracts import ApprovalDecision, ApprovalRequest, AuditEvent

__all__ = ["ApprovalQueue"]


@dataclass
class ApprovalQueue:
    _requests: dict[str, ApprovalRequest] = field(
        default_factory=dict, init=False, repr=False
    )
    _decisions: dict[str, ApprovalDecision] = field(
        default_factory=dict, init=False, repr=False
    )
    _audit: list[AuditEvent] = field(default_factory=list, init=False, repr=False)

    def _audit_event(self, *, action: str, approval_id: str, actor: str, at: datetime) -> None:
        self._audit.append(
            AuditEvent(
                event_id=f"aud-{secrets.token_hex(8)}",
                actor=actor,
                action=action,
                target=f"approval:{approval_id}",
                at=at,
                trace_id=None,
                details={},
            )
        )

    def submit(self, request: ApprovalRequest) -> None:
        if request.approval_id in self._requests:
            raise ValueError(f"approval {request.approval_id!r} already submitted")
        self._requests[request.approval_id] = request
        self._audit_event(
            action="approval.submitted",
            approval_id=request.approval_id,
            actor=f"user:{request.requested_by}",
            at=request.requested_at,
        )

    def decide(
        self,
        *,
        approval_id: str,
        decision: str,
        decided_by: str,
        notes: str,
        decided_at: datetime,
    ) -> ApprovalDecision:
        if approval_id not in self._requests:
            raise LookupError(f"no approval for {approval_id!r}")
        if approval_id in self._decisions:
            raise ValueError(f"approval {approval_id!r} already decided")
        d = ApprovalDecision(
            approval_id=approval_id,
            decision=decision,  # type: ignore[arg-type]
            decided_by=decided_by,
            decided_at=decided_at,
            notes=notes,
        )
        self._decisions[approval_id] = d
        self._audit_event(
            action="approval.decided",
            approval_id=approval_id,
            actor=f"user:{decided_by}",
            at=decided_at,
        )
        return d

    def list_pending(self) -> Iterator[ApprovalRequest]:
        for rid, req in self._requests.items():
            if rid not in self._decisions:
                yield req

    def audit_log(self) -> list[AuditEvent]:
        return list(self._audit)
