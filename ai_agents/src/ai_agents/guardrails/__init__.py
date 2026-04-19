"""AI-failure guardrails (Layer 1 of the ADR-0004 kill-switch architecture).

Every agent-produced action passes through these gates before it can
reach order formulation. The gates are deterministic and cheap; they
are not model-based classifiers.

- `ConfidenceThreshold` - defaults to `flat` (do-not-trade) below the floor.
- `BoundedOutputActionSpace` - refuses unbounded notional / leverage.
- `DriftDetector` - flags distribution shift vs a reference window.
- `escalate_panic` - emits an `AnomalyEvent` for catastrophic behaviour.
- `check_hallucination` - validates structured output against a schema.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from shared_lib.contracts import AnomalyEvent, ValidationResult

__all__ = [
    "BoundedOutputActionSpace",
    "ConfidenceThreshold",
    "DriftDetector",
    "check_hallucination",
    "escalate_panic",
]


@dataclass(frozen=True, slots=True)
class ConfidenceThreshold:
    floor: Decimal

    def __post_init__(self) -> None:
        if not (Decimal("0") <= self.floor <= Decimal("1")):
            raise ValueError("floor must be in [0, 1]")

    def decide(
        self, *, confidence: Decimal, proposed_action: str
    ) -> str:
        if confidence < self.floor:
            return "flat"
        return proposed_action


@dataclass(frozen=True, slots=True)
class BoundedOutputActionSpace:
    max_notional_per_order: Decimal
    max_leverage: Decimal

    def __post_init__(self) -> None:
        if self.max_notional_per_order < 0:
            raise ValueError("max_notional_per_order must be >= 0")
        if self.max_leverage < 0:
            raise ValueError("max_leverage must be >= 0")

    def validate(self, *, notional: Decimal, leverage: Decimal) -> None:
        if notional > self.max_notional_per_order:
            raise ValueError(
                f"notional {notional} exceeds max {self.max_notional_per_order}"
            )
        if leverage > self.max_leverage:
            raise ValueError(
                f"leverage {leverage} exceeds max {self.max_leverage}"
            )


@dataclass
class DriftDetector:
    reference_window: list[Decimal]
    max_abs_mean_shift: Decimal

    def __post_init__(self) -> None:
        if not self.reference_window:
            raise ValueError("reference_window must be non-empty")

    def _mean(self, xs: Iterable[Decimal]) -> Decimal:
        values = list(xs)
        return sum(values, start=Decimal("0")) / Decimal(len(values))

    def evaluate(self, observations: list[Decimal]) -> ValidationResult:
        from datetime import UTC

        if not observations:
            return ValidationResult(
                check_id="drift",
                target="model_output",
                passed=True,
                reason=None,
                evaluated_at=datetime.now(tz=UTC),
            )
        shift = abs(self._mean(observations) - self._mean(self.reference_window))
        passed = shift <= self.max_abs_mean_shift
        reason = (
            None
            if passed
            else f"mean shift {shift} exceeds threshold {self.max_abs_mean_shift}"
        )
        return ValidationResult(
            check_id="drift",
            target="model_output",
            passed=passed,
            reason=reason,
            evaluated_at=datetime.now(tz=UTC),
        )


Severity = Literal["info", "low", "medium", "high", "critical"]


def escalate_panic(
    *,
    source: str,
    summary: str,
    severity: Severity,
    details: dict[str, str],
    detected_at: datetime,
) -> AnomalyEvent:
    return AnomalyEvent(
        anomaly_id=f"panic-{int(detected_at.timestamp())}",
        source=source,
        severity=severity,
        summary=summary,
        detected_at=detected_at,
        details=details,
    )


def check_hallucination(
    *,
    output: dict[str, Any],
    allowed_fields: set[str],
    required_fields: set[str] = frozenset(),
) -> None:
    unknown = set(output) - allowed_fields
    if unknown:
        raise ValueError(f"unknown fields in model output: {sorted(unknown)}")
    missing = required_fields - set(output)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
