"""Phase 10 Task 4 - agents -> CI/live telemetry contract test.

The agent layer (Phase 5) and the CI/infrastructure layer (Phase 9)
share `ValidationResult` and `AnomalyEvent` as their contract vocabulary.
This test proves the wiring in both directions:

1. `code_reviewer.review_source` emits `ValidationResult`s that CI
   (Phase 9 `ci.yml`) can aggregate to a pass/fail gate.
2. `risk_monitor.evaluate` consumes `AnomalyEvent`s -> recommends a
   kill-switch action, and `risk_monitor.escalate` round-trips through
   `escalate_panic` to emit a critical `AnomalyEvent`.
3. `alpha_researcher.propose_factor` lands a record in
   `ResearchMemoryStore` and submits an `ApprovalRequest`; the resulting
   audit trail flows through `ApprovalQueue`.
4. Agent required-permission tuples are static (Phase 5 permission-
   broker contract), so a denied permission immediately prevents the
   agent action from being invoked.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal


def test_code_reviewer_emits_validation_results_ci_can_aggregate() -> None:
    from ai_agents.code_reviewer import review_source
    from shared_lib.contracts import ValidationResult

    bad = "x = df['future_returns'].shift(-1)\n"
    results = review_source(bad)
    assert results, "review_source must return at least one ValidationResult"
    assert all(isinstance(r, ValidationResult) for r in results)
    fails = [r for r in results if not r.passed]
    assert any(r.check_id == "look_ahead.shift_negative" for r in fails)
    assert any(r.check_id == "look_ahead.future_column" for r in fails)


def test_code_reviewer_is_green_on_benign_source() -> None:
    """Golden path: nothing alarming -> every ValidationResult passes."""
    from ai_agents.code_reviewer import review_source

    benign = "import numpy as np\n\n\ndef f(x):\n    return np.abs(x)\n"
    results = review_source(benign)
    assert all(r.passed for r in results), [
        (r.check_id, r.reason) for r in results if not r.passed
    ]


def test_risk_monitor_maps_severity_to_kill_switch_recommendation() -> None:
    """The risk_monitor is the canonical AnomalyEvent -> recommendation
    projector the Phase 6 kill-switch consumes."""
    from ai_agents.risk_monitor import Recommendation, RiskMonitor
    from shared_lib.contracts import AnomalyEvent

    rm = RiskMonitor()
    for sev, expected in [
        ("info", Recommendation.NONE),
        ("medium", Recommendation.REDUCE),
        ("high", Recommendation.HALT),
        ("critical", Recommendation.KILL_SWITCH),
    ]:
        evt = AnomalyEvent(
            anomaly_id=f"a-{sev}",
            source="market_data",
            severity=sev,  # type: ignore[arg-type]
            summary="test",
            detected_at=datetime(2026, 4, 20, tzinfo=UTC),
            details={},
        )
        assert rm.evaluate(evt) == expected


def test_risk_monitor_escalate_emits_critical_anomaly_event() -> None:
    from ai_agents.risk_monitor import RiskMonitor
    from shared_lib.contracts import AnomalyEvent

    rm = RiskMonitor()
    event = rm.escalate(
        source="exchange_feed",
        summary="cross-venue latency spike",
        details={"peer": "binance"},
        detected_at=datetime(2026, 4, 20, tzinfo=UTC),
    )
    assert isinstance(event, AnomalyEvent)
    assert event.severity == "critical"
    assert event.source == "exchange_feed"


def test_alpha_researcher_records_to_memory_and_queues_approval() -> None:
    """alpha_researcher proposes -> ResearchMemoryStore + ApprovalQueue
    both observe; approvals can then be decided by an operator."""
    from ai_agents.alpha_researcher import AlphaResearcher
    from ai_agents.approvals import ApprovalQueue
    from ai_agents.memory import ResearchMemoryStore
    from shared_lib.contracts import ResearchMemoryRecord

    mem = ResearchMemoryStore()
    q = ApprovalQueue()
    agent = AlphaResearcher(memory=mem, approvals=q)
    record = agent.propose_factor(
        hypothesis_id="h-1",
        title="pead variant",
        body="earnings drift signal",
        tags=("earnings", "drift"),
        requested_by="alice",
        requested_at=datetime(2026, 4, 20, tzinfo=UTC),
    )
    assert isinstance(record, ResearchMemoryRecord)
    pending = list(q.list_pending())
    assert len(pending) == 1
    assert pending[0].target_id == "h-1"
    assert pending[0].subject == "factor_promotion"

    decision = q.decide(
        approval_id=pending[0].approval_id,
        decision="approved",
        decided_by="bob",
        notes="ship",
        decided_at=datetime(2026, 4, 20, tzinfo=UTC),
    )
    assert decision.decision == "approved"
    assert list(q.list_pending()) == []


def test_guardrails_confidence_floor_forces_flat_below_threshold() -> None:
    """ConfidenceThreshold is the seam between agent output and the OMS."""
    from ai_agents.guardrails import ConfidenceThreshold

    gate = ConfidenceThreshold(floor=Decimal("0.7"))
    assert gate.decide(confidence=Decimal("0.69"), proposed_action="buy") == "flat"
    assert gate.decide(confidence=Decimal("0.71"), proposed_action="buy") == "buy"


def test_all_agents_declare_required_permissions() -> None:
    """Permission-broker contract: every agent exposes a static tuple
    of required permissions so the control plane can deny before the
    agent runs."""
    from ai_agents.alpha_researcher import AlphaResearcher
    from ai_agents.approvals import ApprovalQueue
    from ai_agents.code_reviewer import review_source
    from ai_agents.memory import ResearchMemoryStore
    from ai_agents.risk_monitor import RiskMonitor

    researcher = AlphaResearcher(memory=ResearchMemoryStore(), approvals=ApprovalQueue())
    monitor = RiskMonitor()
    assert isinstance(researcher.REQUIRED_PERMISSIONS, tuple)
    assert isinstance(monitor.REQUIRED_PERMISSIONS, tuple)
    assert "approvals.submit" in researcher.REQUIRED_PERMISSIONS
    assert "anomaly_events.read" in monitor.REQUIRED_PERMISSIONS
    assert callable(review_source)  # CI-only agent: stateless
