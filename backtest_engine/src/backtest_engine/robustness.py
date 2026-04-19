"""Statistical robustness primitives.

Three light-weight functions that turn the raw returns from a backtest
into signal-quality evidence:

- `bootstrap_confidence_interval` - non-parametric CI for any scalar
  metric computed from a sample.
- `stability_across_windows` - coefficient-of-variation-like ratio
  measuring performance stability across walk-forward windows.
- `split_and_compare` - first-half vs second-half drift detector.

Inspired by Vibe-Trading's robustness validation patterns without
adopting the upstream project as a dependency.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from decimal import Decimal

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "bootstrap_confidence_interval",
    "split_and_compare",
    "stability_across_windows",
]


def bootstrap_confidence_interval(
    data: ArrayLike,
    *,
    metric: Callable[[NDArray[np.float64]], float],
    n_samples: int,
    ci: float,
    seed: int | None = None,
) -> tuple[float, float]:
    """Return `(lower, upper)` for the metric under non-parametric bootstrap.

    - `ci` in (0, 1) - e.g. 0.95 for a 95% interval.
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("bootstrap_confidence_interval requires non-empty data")
    if not (0.0 < ci < 1.0):
        raise ValueError("ci must be in (0, 1)")
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    rng = np.random.default_rng(seed)
    samples = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        idx = rng.integers(0, arr.size, size=arr.size)
        samples[i] = metric(arr[idx])
    alpha = 1.0 - ci
    lower = float(np.quantile(samples, alpha / 2))
    upper = float(np.quantile(samples, 1.0 - alpha / 2))
    return lower, upper


def stability_across_windows(
    metric_per_window: Sequence[Decimal],
) -> Decimal:
    """Return (max - min) / |mean|, or 0 when the mean is 0.

    Higher values mean less stable. Constant series return 0.
    """
    if not metric_per_window:
        raise ValueError("stability_across_windows requires at least 1 window")
    hi = max(metric_per_window)
    lo = min(metric_per_window)
    if hi == lo:
        return Decimal("0")
    mean = sum(metric_per_window, start=Decimal("0")) / Decimal(len(metric_per_window))
    if mean == 0:
        return Decimal("0")
    return (hi - lo) / abs(mean)


def split_and_compare(
    data: ArrayLike,
    *,
    split_point: int,
    metric: Callable[[NDArray[np.float64]], float],
) -> tuple[float, float, float]:
    """Compute `metric` on `data[:split_point]` and `data[split_point:]`.

    Returns `(first, second, delta)` where `delta = second - first`.
    """
    arr = np.asarray(data, dtype=np.float64)
    if split_point <= 0 or split_point >= arr.size:
        raise ValueError(
            f"split_point {split_point} must be in (0, {arr.size})"
        )
    first = float(metric(arr[:split_point]))
    second = float(metric(arr[split_point:]))
    return first, second, second - first
