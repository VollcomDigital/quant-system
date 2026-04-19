"""Risk monitor agent prototype.

Consumes `AnomalyEvent`s and recommends a kill-switch action. Cannot
submit orders, cannot call KMS, cannot transfer treasury.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from shared_lib.contracts import AnomalyEvent

from ai_agents.guardrails import escalate_panic

__all__ = ["Recommendation", "RiskMonitor"]


class Recommendation(Enum):
    NONE = "none"
    REDUCE = "reduce"
    HALT = "halt"
    KILL_SWITCH = "kill_switch"


_RULES: dict[str, Recommendation] = {
    "info": Recommendation.NONE,
    "low": Recommendation.NONE,
    "medium": Recommendation.REDUCE,
    "high": Recommendation.HALT,
    "critical": Recommendation.KILL_SWITCH,
}


@dataclass
class RiskMonitor:
    REQUIRED_PERMISSIONS: tuple[str, ...] = (
        "anomaly_events.read",
        "notifications.slack.post",
        "approvals.submit",
    )

    def evaluate(self, event: AnomalyEvent) -> Recommendation:
        return _RULES.get(event.severity, Recommendation.NONE)

    def escalate(
        self,
        *,
        source: str,
        summary: str,
        details: dict[str, str],
        detected_at: datetime,
    ) -> AnomalyEvent:
        return escalate_panic(
            source=source,
            summary=summary,
            severity="critical",
            details=details,
            detected_at=detected_at,
        )
