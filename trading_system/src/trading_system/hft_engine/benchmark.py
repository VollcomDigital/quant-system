"""Latency benchmark harness.

Turns a sample of latency measurements (microseconds) into a
deterministic `LatencyReport` + gates it against a `LatencyBudget`.
Used by:

- The Phase 8 HFT replay harness before any live-path consideration.
- The Phase 5 HFT Latency Agent for telemetry consumption (Phase 8
  ships the contract; the agent class lands when native telemetry is
  emitting).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
from shared_lib.contracts import ValidationResult

__all__ = [
    "LatencyBudget",
    "LatencyReport",
    "enforce_budget",
    "summarise_latency",
]


@dataclass(frozen=True, slots=True)
class LatencyReport:
    n: int
    p50: float
    p95: float
    p99: float
    max: float


def summarise_latency(*, samples_us: Sequence[float]) -> LatencyReport:
    if not samples_us:
        raise ValueError("samples_us must be non-empty")
    arr = np.asarray(samples_us, dtype=np.float64)
    if (arr < 0).any():
        raise ValueError("samples_us must all be >= 0")
    # Use `method="linear"` (NumPy default) deterministically.
    p50, p95, p99 = np.quantile(arr, [0.5, 0.95, 0.99])
    return LatencyReport(
        n=int(arr.size),
        p50=float(p50),
        p95=float(p95),
        p99=float(p99),
        max=float(arr.max()),
    )


@dataclass(frozen=True, slots=True)
class LatencyBudget:
    p50_us: Decimal
    p95_us: Decimal
    p99_us: Decimal

    def __post_init__(self) -> None:
        for name, value in (
            ("p50_us", self.p50_us),
            ("p95_us", self.p95_us),
            ("p99_us", self.p99_us),
        ):
            if value <= Decimal("0"):
                raise ValueError(f"{name} must be > 0")
        if not (self.p50_us <= self.p95_us <= self.p99_us):
            raise ValueError(
                "LatencyBudget must be monotonic: p50 <= p95 <= p99"
            )


def enforce_budget(report: LatencyReport, *, budget: LatencyBudget) -> ValidationResult:
    now = datetime.now(tz=UTC)
    breaches: list[str] = []
    if Decimal(str(report.p50)) > budget.p50_us:
        breaches.append(f"p50 {report.p50}us > {budget.p50_us}us")
    if Decimal(str(report.p95)) > budget.p95_us:
        breaches.append(f"p95 {report.p95}us > {budget.p95_us}us")
    if Decimal(str(report.p99)) > budget.p99_us:
        breaches.append(f"p99 {report.p99}us > {budget.p99_us}us")
    if breaches:
        return ValidationResult(
            check_id="hft.latency_budget",
            target="hft_engine",
            passed=False,
            reason="; ".join(breaches),
            evaluated_at=now,
        )
    return ValidationResult(
        check_id="hft.latency_budget",
        target="hft_engine",
        passed=True,
        reason=None,
        evaluated_at=now,
    )
