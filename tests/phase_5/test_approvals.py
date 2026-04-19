"""Phase 5 Task 5 - ai_agents.approvals human approval workflow.

- `ApprovalQueue` wraps the Phase 1 ApprovalRequest/Decision contracts.
- Submitting an approval emits an AuditEvent.
- Deciding on an approval emits an AuditEvent.
- An approval can only be decided once.
- A decision's subject must match the original request's subject.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest


def _request(approval_id: str = "app-1", subject: str = "factor_promotion"):
    from shared_lib.contracts import ApprovalRequest

    return ApprovalRequest(
        approval_id=approval_id,
        subject=subject,  # type: ignore[arg-type]
        target_id="mom_12_1",
        requested_by="alice",
        requested_at=datetime(2026, 4, 19, tzinfo=UTC),
        context={"coverage_pct": "0.9"},
    )


# ---------------------------------------------------------------------------
# Submission + audit
# ---------------------------------------------------------------------------


def test_approval_queue_submit_emits_audit_event() -> None:
    from ai_agents.approvals import ApprovalQueue

    q = ApprovalQueue()
    q.submit(_request())
    audits = q.audit_log()
    assert len(audits) == 1
    assert audits[0].action == "approval.submitted"


def test_approval_queue_refuses_duplicate_approval_id() -> None:
    from ai_agents.approvals import ApprovalQueue

    q = ApprovalQueue()
    q.submit(_request())
    with pytest.raises(ValueError, match="already"):
        q.submit(_request())


def test_approval_queue_list_pending_filters_decided() -> None:
    from ai_agents.approvals import ApprovalQueue

    q = ApprovalQueue()
    q.submit(_request("a1"))
    q.submit(_request("a2"))
    q.decide(
        approval_id="a1",
        decision="approved",
        decided_by="bob",
        notes="LGTM",
        decided_at=datetime(2026, 4, 19, tzinfo=UTC),
    )
    pending = {r.approval_id for r in q.list_pending()}
    assert pending == {"a2"}


# ---------------------------------------------------------------------------
# Decision + audit
# ---------------------------------------------------------------------------


def test_decide_emits_audit_event_and_returns_decision() -> None:
    from ai_agents.approvals import ApprovalQueue

    q = ApprovalQueue()
    q.submit(_request())
    d = q.decide(
        approval_id="app-1",
        decision="approved",
        decided_by="bob",
        notes="x",
        decided_at=datetime(2026, 4, 19, tzinfo=UTC),
    )
    assert d.decision == "approved"
    actions = [a.action for a in q.audit_log()]
    assert actions == ["approval.submitted", "approval.decided"]


def test_decide_refuses_second_decision() -> None:
    from ai_agents.approvals import ApprovalQueue

    q = ApprovalQueue()
    q.submit(_request())
    q.decide(
        approval_id="app-1",
        decision="approved",
        decided_by="bob",
        notes="",
        decided_at=datetime(2026, 4, 19, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="already"):
        q.decide(
            approval_id="app-1",
            decision="rejected",
            decided_by="bob",
            notes="",
            decided_at=datetime(2026, 4, 19, tzinfo=UTC),
        )


def test_decide_unknown_approval_raises() -> None:
    from ai_agents.approvals import ApprovalQueue

    q = ApprovalQueue()
    with pytest.raises(LookupError):
        q.decide(
            approval_id="ghost",
            decision="approved",
            decided_by="bob",
            notes="",
            decided_at=datetime(2026, 4, 19, tzinfo=UTC),
        )
