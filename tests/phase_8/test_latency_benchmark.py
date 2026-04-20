"""Phase 8 Task 3 - latency benchmark harness.

The harness turns a raw sample of latency measurements into a
`LatencyReport` (p50/p95/p99/max) and gates it against a declared
budget. Pure Python, NumPy-assisted but never threading- or
timing-dependent so tests are deterministic.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

# ---------------------------------------------------------------------------
# Percentile summary
# ---------------------------------------------------------------------------


def test_latency_report_percentiles_are_monotonic() -> None:
    from trading_system.hft_engine.benchmark import summarise_latency

    samples_us = [float(i) for i in range(1, 101)]  # 1..100
    report = summarise_latency(samples_us=samples_us)
    assert report.p50 < report.p95 < report.p99 <= report.max


def test_latency_report_exact_percentiles_on_linear_sample() -> None:
    from trading_system.hft_engine.benchmark import summarise_latency

    samples_us = [float(i) for i in range(1, 101)]
    report = summarise_latency(samples_us=samples_us)
    # numpy.quantile uses linear interp: p50 = 50.5 for 1..100.
    assert 49.0 <= report.p50 <= 51.0
    assert 94.0 <= report.p95 <= 96.0
    assert 98.0 <= report.p99 <= 100.0
    assert report.n == 100


def test_latency_report_rejects_empty_sample() -> None:
    from trading_system.hft_engine.benchmark import summarise_latency

    with pytest.raises(ValueError):
        summarise_latency(samples_us=[])


def test_latency_report_rejects_negative_samples() -> None:
    from trading_system.hft_engine.benchmark import summarise_latency

    with pytest.raises(ValueError):
        summarise_latency(samples_us=[1.0, 2.0, -0.1])


# ---------------------------------------------------------------------------
# Budget gate
# ---------------------------------------------------------------------------


def test_budget_gate_accepts_report_within_budget() -> None:
    from trading_system.hft_engine.benchmark import (
        LatencyBudget,
        enforce_budget,
        summarise_latency,
    )

    report = summarise_latency(samples_us=[1.0, 2.0, 3.0, 4.0, 5.0])
    budget = LatencyBudget(
        p50_us=Decimal("5"),
        p95_us=Decimal("5"),
        p99_us=Decimal("5"),
    )
    result = enforce_budget(report, budget=budget)
    assert result.passed is True


def test_budget_gate_fails_when_p99_exceeds_budget() -> None:
    from trading_system.hft_engine.benchmark import (
        LatencyBudget,
        enforce_budget,
        summarise_latency,
    )

    # 90 samples at 1us + 10 samples at 50us: p99 lands around 50us.
    samples = [1.0] * 90 + [50.0] * 10
    report = summarise_latency(samples_us=samples)
    assert report.p99 > 5.0
    budget = LatencyBudget(
        p50_us=Decimal("5"),
        p95_us=Decimal("5"),
        p99_us=Decimal("5"),
    )
    result = enforce_budget(report, budget=budget)
    assert result.passed is False
    assert "p99" in (result.reason or "")


def test_budget_rejects_non_positive_limits() -> None:
    from trading_system.hft_engine.benchmark import LatencyBudget

    with pytest.raises(ValueError):
        LatencyBudget(
            p50_us=Decimal("-1"),
            p95_us=Decimal("5"),
            p99_us=Decimal("5"),
        )


def test_budget_rejects_inverted_limits() -> None:
    """Non-monotonic budgets are nonsensical (p50 > p95 can never pass)."""
    from trading_system.hft_engine.benchmark import LatencyBudget

    with pytest.raises(ValueError, match="monotonic"):
        LatencyBudget(
            p50_us=Decimal("10"),
            p95_us=Decimal("5"),
            p99_us=Decimal("20"),
        )


# ---------------------------------------------------------------------------
# Determinism: same input -> same report.
# ---------------------------------------------------------------------------


def test_summarise_latency_is_deterministic() -> None:
    from trading_system.hft_engine.benchmark import summarise_latency

    samples = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    a = summarise_latency(samples_us=samples)
    b = summarise_latency(samples_us=samples)
    assert a == b
