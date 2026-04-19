"""Phase 5 Task 6 - alpha_researcher agent prototype.

The researcher agent:
- Takes a factor hypothesis (input payload).
- Records it to the research-memory store as `factor_hypothesis`.
- Emits an `ApprovalRequest` for the research backlog, NOT direct
  production use.
- Carries scoped permissions: `factor_library.read`, `research_memory.write`,
  `approvals.submit`.
- Cannot access the forbidden namespaces (oms/kms/treasury/src).
"""

from __future__ import annotations

from datetime import UTC, datetime


def test_alpha_researcher_records_hypothesis_and_requests_approval() -> None:
    from ai_agents.alpha_researcher import AlphaResearcher
    from ai_agents.approvals import ApprovalQueue
    from ai_agents.memory import ResearchMemoryStore

    memory = ResearchMemoryStore()
    queue = ApprovalQueue()
    agent = AlphaResearcher(memory=memory, approvals=queue)

    result = agent.propose_factor(
        hypothesis_id="h-1",
        title="Momentum in small caps",
        body="12-1 momentum on the bottom quintile.",
        tags=("momentum", "small_cap"),
        requested_by="alice",
        requested_at=datetime(2026, 4, 19, tzinfo=UTC),
    )
    assert result.record_id == "h-1"
    assert memory.get("h-1").kind == "factor_hypothesis"

    pending = list(queue.list_pending())
    assert len(pending) == 1
    assert pending[0].subject == "factor_promotion"
    assert pending[0].target_id == "h-1"


def test_alpha_researcher_uses_only_scoped_permissions() -> None:
    from ai_agents.alpha_researcher import AlphaResearcher

    # The agent declares a permission list; the runtime guarantees none
    # of them are in the forbidden namespaces.
    assert "research_memory.write" in AlphaResearcher.REQUIRED_PERMISSIONS
    assert "approvals.submit" in AlphaResearcher.REQUIRED_PERMISSIONS
    # Never oms/kms/treasury.
    for p in AlphaResearcher.REQUIRED_PERMISSIONS:
        assert not p.startswith(("oms.", "kms.", "treasury.", "src."))


def test_alpha_researcher_propose_twice_raises() -> None:
    import pytest
    from ai_agents.alpha_researcher import AlphaResearcher
    from ai_agents.approvals import ApprovalQueue
    from ai_agents.memory import ResearchMemoryStore

    memory = ResearchMemoryStore()
    queue = ApprovalQueue()
    agent = AlphaResearcher(memory=memory, approvals=queue)
    agent.propose_factor(
        hypothesis_id="h-1",
        title="t",
        body="b",
        tags=(),
        requested_by="alice",
        requested_at=datetime(2026, 4, 19, tzinfo=UTC),
    )
    with pytest.raises(ValueError):
        agent.propose_factor(
            hypothesis_id="h-1",
            title="t",
            body="b",
            tags=(),
            requested_by="alice",
            requested_at=datetime(2026, 4, 19, tzinfo=UTC),
        )
