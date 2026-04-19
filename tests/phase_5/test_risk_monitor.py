"""Phase 5 Task 8 - risk_monitor agent prototype.

The risk monitor consumes `AnomalyEvent`s (from guardrails or from live
telemetry), evaluates them against a rule set, and either returns
`KillSwitchRecommendation.NONE` or emits a panic escalation. It cannot
directly submit orders or call KMS.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest


def _anomaly(severity: str = "high", source: str = "oms"):
    from shared_lib.contracts import AnomalyEvent

    return AnomalyEvent(
        anomaly_id="a-1",
        source=source,
        severity=severity,  # type: ignore[arg-type]
        summary="intraday drawdown breach",
        detected_at=datetime(2026, 4, 19, tzinfo=UTC),
        details={"drawdown": "0.06"},
    )


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


def test_low_severity_anomaly_returns_none() -> None:
    from ai_agents.risk_monitor import Recommendation, RiskMonitor

    monitor = RiskMonitor()
    rec = monitor.evaluate(_anomaly(severity="info"))
    assert rec is Recommendation.NONE


def test_medium_anomaly_recommends_reduce() -> None:
    from ai_agents.risk_monitor import Recommendation, RiskMonitor

    monitor = RiskMonitor()
    assert monitor.evaluate(_anomaly(severity="medium")) is Recommendation.REDUCE


def test_high_anomaly_recommends_halt() -> None:
    from ai_agents.risk_monitor import Recommendation, RiskMonitor

    monitor = RiskMonitor()
    assert monitor.evaluate(_anomaly(severity="high")) is Recommendation.HALT


def test_critical_anomaly_recommends_kill_switch() -> None:
    from ai_agents.risk_monitor import Recommendation, RiskMonitor

    monitor = RiskMonitor()
    assert monitor.evaluate(_anomaly(severity="critical")) is Recommendation.KILL_SWITCH


# ---------------------------------------------------------------------------
# Permissions: risk monitor must not hold any forbidden permission.
# ---------------------------------------------------------------------------


def test_risk_monitor_permissions_never_include_forbidden_surfaces() -> None:
    from ai_agents.risk_monitor import RiskMonitor

    for p in RiskMonitor.REQUIRED_PERMISSIONS:
        assert not p.startswith(("oms.", "kms.", "treasury.", "src."))


# ---------------------------------------------------------------------------
# Escalation returns a usable panic AnomalyEvent for downstream routing.
# ---------------------------------------------------------------------------


def test_escalate_produces_critical_anomaly() -> None:
    from ai_agents.risk_monitor import RiskMonitor

    monitor = RiskMonitor()
    ev = monitor.escalate(
        source="risk_monitor",
        summary="drawdown 6% in 10m",
        details={"drawdown": "0.06"},
        detected_at=datetime(2026, 4, 19, tzinfo=UTC),
    )
    assert ev.severity == "critical"
    assert ev.source == "risk_monitor"


# ---------------------------------------------------------------------------
# Defensive: recommendation enum is closed.
# ---------------------------------------------------------------------------


def test_recommendation_has_four_members() -> None:
    from ai_agents.risk_monitor import Recommendation

    names = {m.name for m in Recommendation}
    assert names == {"NONE", "REDUCE", "HALT", "KILL_SWITCH"}


def test_unknown_severity_is_rejected_via_pydantic() -> None:
    from shared_lib.contracts import AnomalyEvent

    with pytest.raises(ValueError):
        AnomalyEvent(
            anomaly_id="x",
            source="y",
            severity="doomsday",  # not in enum
            summary="",
            detected_at=datetime(2026, 4, 19, tzinfo=UTC),
            details={},
        )
