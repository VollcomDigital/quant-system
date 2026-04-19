"""Phase 5 Task 4 - ai_agents.guardrails.

Layer 1 of ADR-0004 kill-switch architecture lives here - guardrails
BEFORE order formulation. Every model output passes through:

- `ConfidenceThreshold` - below the floor the action defaults to
  `Do Not Trade` or `Reduce Exposure`.
- `BoundedOutputActionSpace` - model cannot request unbounded notional
  / leverage.
- `DriftDetector` - compares output distribution against a reference
  window; flags drift.
- `PanicEscalation` - emits an `AnomalyEvent` when catastrophic output
  is detected.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

# ---------------------------------------------------------------------------
# Confidence threshold
# ---------------------------------------------------------------------------


def test_confidence_threshold_allows_when_above() -> None:
    from ai_agents.guardrails import ConfidenceThreshold

    gate = ConfidenceThreshold(floor=Decimal("0.6"))
    action = gate.decide(confidence=Decimal("0.8"), proposed_action="long")
    assert action == "long"


def test_confidence_threshold_defaults_to_do_not_trade_when_below() -> None:
    from ai_agents.guardrails import ConfidenceThreshold

    gate = ConfidenceThreshold(floor=Decimal("0.6"))
    assert gate.decide(confidence=Decimal("0.4"), proposed_action="long") == "flat"


def test_confidence_threshold_rejects_out_of_range_floor() -> None:
    from ai_agents.guardrails import ConfidenceThreshold

    with pytest.raises(ValueError):
        ConfidenceThreshold(floor=Decimal("1.2"))


# ---------------------------------------------------------------------------
# Bounded output action space
# ---------------------------------------------------------------------------


def test_bounded_action_space_enforces_max_notional() -> None:
    from ai_agents.guardrails import BoundedOutputActionSpace

    gate = BoundedOutputActionSpace(
        max_notional_per_order=Decimal("100000"),
        max_leverage=Decimal("2"),
    )
    gate.validate(notional=Decimal("50000"), leverage=Decimal("1.5"))
    with pytest.raises(ValueError, match="notional"):
        gate.validate(notional=Decimal("200000"), leverage=Decimal("1"))


def test_bounded_action_space_enforces_leverage_cap() -> None:
    from ai_agents.guardrails import BoundedOutputActionSpace

    gate = BoundedOutputActionSpace(
        max_notional_per_order=Decimal("1000000"),
        max_leverage=Decimal("2"),
    )
    with pytest.raises(ValueError, match="leverage"):
        gate.validate(notional=Decimal("1000"), leverage=Decimal("3"))


def test_bounded_action_space_rejects_negative_bounds() -> None:
    from ai_agents.guardrails import BoundedOutputActionSpace

    with pytest.raises(ValueError):
        BoundedOutputActionSpace(
            max_notional_per_order=Decimal("-1"),
            max_leverage=Decimal("1"),
        )


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


def test_drift_detector_flags_distribution_shift() -> None:
    from ai_agents.guardrails import DriftDetector

    # Reference: values tightly around 0.
    ref = [Decimal("0.01"), Decimal("0.02"), Decimal("-0.01"), Decimal("0")]
    det = DriftDetector(reference_window=ref, max_abs_mean_shift=Decimal("0.1"))
    # Same distribution: no drift.
    result = det.evaluate([Decimal("0.015"), Decimal("0.005")])
    assert result.passed is True
    # Shifted distribution: drift.
    shifted = det.evaluate([Decimal("0.5"), Decimal("0.6")])
    assert shifted.passed is False


def test_drift_detector_rejects_empty_reference() -> None:
    from ai_agents.guardrails import DriftDetector

    with pytest.raises(ValueError):
        DriftDetector(reference_window=[], max_abs_mean_shift=Decimal("0.1"))


# ---------------------------------------------------------------------------
# Panic escalation
# ---------------------------------------------------------------------------


def test_panic_escalation_emits_anomaly_event() -> None:
    from ai_agents.guardrails import escalate_panic

    ev = escalate_panic(
        source="alpha_researcher",
        summary="proposed 20x leverage short",
        severity="critical",
        details={"notional": "1000000"},
        detected_at=datetime(2026, 4, 19, tzinfo=UTC),
    )
    assert ev.severity == "critical"
    assert ev.source == "alpha_researcher"


def test_panic_escalation_rejects_unknown_severity() -> None:
    from ai_agents.guardrails import escalate_panic

    with pytest.raises(ValueError):
        escalate_panic(
            source="x",
            summary="",
            severity="catastrophic",
            details={},
            detected_at=datetime(2026, 4, 19, tzinfo=UTC),
        )


# ---------------------------------------------------------------------------
# Hallucination detection - structured output must match declared schema.
# ---------------------------------------------------------------------------


def test_hallucination_check_rejects_unknown_fields() -> None:
    from ai_agents.guardrails import check_hallucination

    # Expected schema: {direction, symbol, confidence}.
    with pytest.raises(ValueError, match="unknown"):
        check_hallucination(
            output={"direction": "long", "symbol": "AAPL", "magic": 42},
            allowed_fields={"direction", "symbol", "confidence"},
        )


def test_hallucination_check_rejects_missing_required() -> None:
    from ai_agents.guardrails import check_hallucination

    with pytest.raises(ValueError, match="missing"):
        check_hallucination(
            output={"direction": "long"},
            allowed_fields={"direction", "symbol"},
            required_fields={"direction", "symbol"},
        )


def test_hallucination_check_accepts_valid_output() -> None:
    from ai_agents.guardrails import check_hallucination

    check_hallucination(
        output={"direction": "long", "symbol": "AAPL", "confidence": 0.7},
        allowed_fields={"direction", "symbol", "confidence"},
        required_fields={"direction", "symbol"},
    )
