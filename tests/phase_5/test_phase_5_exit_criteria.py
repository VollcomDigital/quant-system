"""Phase 5 Exit Criteria aggregate gate.

- Agent permissions scoped by workflow and target system.
- Agent traces + audit logs emitted with shared telemetry conventions.
- No agent has direct unrestricted access to OMS, KMS, or treasury.
"""

from __future__ import annotations


def test_exit_agent_permissions_never_allow_oms_kms_treasury() -> None:
    # Every agent prototype declares REQUIRED_PERMISSIONS.
    from ai_agents.alpha_researcher import AlphaResearcher
    from ai_agents.risk_monitor import RiskMonitor

    for agent_cls in (AlphaResearcher, RiskMonitor):
        for p in agent_cls.REQUIRED_PERMISSIONS:
            assert not p.startswith(("oms.", "kms.", "treasury.", "src."))


def test_exit_permissions_registry_refuses_forbidden_namespaces() -> None:
    import pytest
    from ai_agents.permissions import AgentPermissions

    for bad in ("oms.submit_order", "kms.sign", "treasury.transfer"):
        with pytest.raises(ValueError):
            AgentPermissions.from_strings((bad,))


def test_exit_approval_queue_emits_audit_events() -> None:
    from datetime import UTC, datetime

    from ai_agents.approvals import ApprovalQueue
    from shared_lib.contracts import ApprovalRequest

    q = ApprovalQueue()
    q.submit(
        ApprovalRequest(
            approval_id="exit-a",
            subject="factor_promotion",
            target_id="x",
            requested_by="alice",
            requested_at=datetime(2026, 4, 19, tzinfo=UTC),
            context={},
        )
    )
    q.decide(
        approval_id="exit-a",
        decision="approved",
        decided_by="bob",
        notes="",
        decided_at=datetime(2026, 4, 19, tzinfo=UTC),
    )
    actions = [a.action for a in q.audit_log()]
    assert actions == ["approval.submitted", "approval.decided"]


def test_exit_runtime_primitives_are_importable() -> None:
    from ai_agents.runtime import (  # noqa: F401
        AgentRegistry,
        JobQueue,
        PromptRegistry,
        Tool,
    )
