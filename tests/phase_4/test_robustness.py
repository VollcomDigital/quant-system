"""Phase 4 Task 7 - Statistical robustness.

Three small primitives:

- `bootstrap_confidence_interval(returns, metric, n_samples, ci)` -
  non-parametric CI around any scalar metric.
- `stability_across_windows(metric_per_window)` - ratio of max-min to
  mean; a high number flags unstable performance.
- `split_and_compare(full_returns, split_point, metric)` - compute the
  metric on the first vs second half and return both plus their delta
  so reviewers can see out-of-sample drift.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def test_bootstrap_ci_contains_the_sample_mean() -> None:
    from backtest_engine.robustness import bootstrap_confidence_interval

    rng = np.random.default_rng(0)
    data = rng.normal(0.01, 0.02, size=500)
    mean = float(np.mean(data))
    lo, hi = bootstrap_confidence_interval(
        data, metric=lambda x: float(np.mean(x)), n_samples=500, ci=0.95, seed=42
    )
    assert lo < mean < hi


def test_bootstrap_ci_rejects_empty() -> None:
    from backtest_engine.robustness import bootstrap_confidence_interval

    with pytest.raises(ValueError):
        bootstrap_confidence_interval(
            np.array([]), metric=lambda x: 0.0, n_samples=10, ci=0.9
        )


def test_bootstrap_ci_rejects_invalid_ci() -> None:
    from backtest_engine.robustness import bootstrap_confidence_interval

    with pytest.raises(ValueError):
        bootstrap_confidence_interval(
            np.array([0.0]), metric=lambda x: 0.0, n_samples=10, ci=1.5
        )


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


def test_stability_high_for_wide_range() -> None:
    from backtest_engine.robustness import stability_across_windows

    narrow = stability_across_windows([Decimal("1.0"), Decimal("1.1"), Decimal("0.95")])
    wide = stability_across_windows([Decimal("1.0"), Decimal("4.0"), Decimal("-3.0")])
    assert wide > narrow


def test_stability_zero_for_constant_series() -> None:
    from backtest_engine.robustness import stability_across_windows

    assert stability_across_windows([Decimal("1"), Decimal("1"), Decimal("1")]) == Decimal(
        "0"
    )


def test_stability_rejects_empty() -> None:
    from backtest_engine.robustness import stability_across_windows

    with pytest.raises(ValueError):
        stability_across_windows([])


# ---------------------------------------------------------------------------
# split_and_compare
# ---------------------------------------------------------------------------


def test_split_and_compare_detects_drift() -> None:
    from backtest_engine.robustness import split_and_compare

    rng = np.random.default_rng(0)
    first_half = rng.normal(0.01, 0.01, size=100)
    second_half = rng.normal(-0.01, 0.01, size=100)
    full = np.concatenate([first_half, second_half])
    first_mean, second_mean, delta = split_and_compare(
        full, split_point=100, metric=lambda x: float(np.mean(x))
    )
    assert first_mean > 0 > second_mean
    assert abs(delta - (second_mean - first_mean)) < 1e-12


def test_split_and_compare_rejects_bad_split() -> None:
    from backtest_engine.robustness import split_and_compare

    with pytest.raises(ValueError):
        split_and_compare(np.array([1.0, 2.0, 3.0]), split_point=0, metric=lambda x: 0.0)
    with pytest.raises(ValueError):
        split_and_compare(np.array([1.0, 2.0, 3.0]), split_point=5, metric=lambda x: 0.0)
