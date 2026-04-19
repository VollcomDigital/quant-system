"""Alpha Researcher agent prototype.

Bounded surface: takes a factor hypothesis, records it to the
research-memory store, and emits an `ApprovalRequest` for the research
backlog. Never writes directly to the factor library - promotion must
go through the Phase 3 `promote_factor` gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from shared_lib.contracts import ApprovalRequest, ResearchMemoryRecord

from ai_agents.approvals import ApprovalQueue
from ai_agents.memory import ResearchMemoryStore

__all__ = ["AlphaResearcher"]


@dataclass
class AlphaResearcher:
    memory: ResearchMemoryStore
    approvals: ApprovalQueue

    REQUIRED_PERMISSIONS: tuple[str, ...] = (
        "factor_library.read",
        "research_memory.write",
        "approvals.submit",
        "notifications.slack.post",
    )

    def propose_factor(
        self,
        *,
        hypothesis_id: str,
        title: str,
        body: str,
        tags: tuple[str, ...],
        requested_by: str,
        requested_at: datetime,
    ) -> ResearchMemoryRecord:
        record = ResearchMemoryRecord(
            record_id=hypothesis_id,
            kind="factor_hypothesis",
            title=title,
            body=body,
            tags=tags,
            created_at=requested_at,
        )
        self.memory.add(record)
        self.approvals.submit(
            ApprovalRequest(
                approval_id=f"approval-{hypothesis_id}",
                subject="factor_promotion",
                target_id=hypothesis_id,
                requested_by=requested_by,
                requested_at=requested_at,
                context={"title": title},
            )
        )
        return record
