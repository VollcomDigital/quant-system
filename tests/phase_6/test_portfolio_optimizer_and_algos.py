"""Phase 6 Task 4 - portfolio optimizer + TWAP/VWAP execution algos.

- `solve_min_variance` - a tiny pure-Python min-variance solver over
  Phase 1 `OptimizerRequest` / `OptimizerResponse`. Real optimizers
  (NVIDIA cuQuant, cvxpy, ...) plug in behind the same contract.
- `twap_slice` - equal-time slicing returning `(slice_time, child_qty)`
  tuples.
- `vwap_slice` - participation-rate slicing weighted by a historical
  volume profile.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

# ---------------------------------------------------------------------------
# Min-variance solver
# ---------------------------------------------------------------------------


def test_min_variance_equal_weights_when_covariance_is_diagonal_equal() -> None:
    from shared_lib.contracts import OptimizerRequest
    from trading_system.mid_freq_engine.portfolio_optimizer import solve_min_variance

    req = OptimizerRequest(
        request_id="r1",
        universe=("AAPL", "MSFT"),
        objective="min_variance",
        gross_leverage=Decimal("1"),
        bounds={},
        risk_aversion=Decimal("1"),
    )
    # Equal variances, zero covariance -> equal weights.
    covariance = {("AAPL", "AAPL"): Decimal("0.01"),
                  ("MSFT", "MSFT"): Decimal("0.01"),
                  ("AAPL", "MSFT"): Decimal("0"),
                  ("MSFT", "AAPL"): Decimal("0")}
    resp = solve_min_variance(req, covariance=covariance)
    assert resp.weights["AAPL"] == Decimal("0.5")
    assert resp.weights["MSFT"] == Decimal("0.5")


def test_min_variance_weights_sum_to_1_given_bounds() -> None:
    from shared_lib.contracts import OptimizerRequest
    from trading_system.mid_freq_engine.portfolio_optimizer import solve_min_variance

    req = OptimizerRequest(
        request_id="r1",
        universe=("A", "B", "C"),
        objective="min_variance",
        gross_leverage=Decimal("1"),
        bounds={
            "A": (Decimal("0"), Decimal("1")),
            "B": (Decimal("0"), Decimal("1")),
            "C": (Decimal("0"), Decimal("1")),
        },
        risk_aversion=Decimal("1"),
    )
    # Lopsided variances: A=0.04, B=0.01, C=0.01; C should get more weight.
    cov = {
        ("A", "A"): Decimal("0.04"), ("B", "B"): Decimal("0.01"), ("C", "C"): Decimal("0.01"),
        ("A", "B"): Decimal("0"), ("A", "C"): Decimal("0"), ("B", "C"): Decimal("0"),
        ("B", "A"): Decimal("0"), ("C", "A"): Decimal("0"), ("C", "B"): Decimal("0"),
    }
    resp = solve_min_variance(req, covariance=cov)
    total = sum(resp.weights.values(), start=Decimal("0"))
    assert abs(total - Decimal("1")) <= Decimal("1e-6")
    assert resp.weights["A"] < resp.weights["B"]


def test_min_variance_rejects_missing_diagonal() -> None:
    from shared_lib.contracts import OptimizerRequest
    from trading_system.mid_freq_engine.portfolio_optimizer import solve_min_variance

    req = OptimizerRequest(
        request_id="r1",
        universe=("A", "B"),
        objective="min_variance",
        gross_leverage=Decimal("1"),
        bounds={},
        risk_aversion=Decimal("1"),
    )
    with pytest.raises(ValueError, match="diagonal"):
        solve_min_variance(req, covariance={("A", "A"): Decimal("0.01")})  # missing B


# ---------------------------------------------------------------------------
# TWAP
# ---------------------------------------------------------------------------


def test_twap_slice_produces_equal_time_gaps() -> None:
    from trading_system.mid_freq_engine.execution_algos import twap_slice

    start = datetime(2026, 4, 19, 14, tzinfo=UTC)
    end = datetime(2026, 4, 19, 15, tzinfo=UTC)
    slices = twap_slice(total_quantity=Decimal("60"), start=start, end=end, num_slices=3)
    assert len(slices) == 3
    assert slices[0][0] == start
    assert slices[1][0] == start + timedelta(minutes=20)
    assert slices[2][0] == start + timedelta(minutes=40)


def test_twap_slice_quantities_sum_to_total() -> None:
    from trading_system.mid_freq_engine.execution_algos import twap_slice

    slices = twap_slice(
        total_quantity=Decimal("100"),
        start=datetime(2026, 4, 19, 14, tzinfo=UTC),
        end=datetime(2026, 4, 19, 15, tzinfo=UTC),
        num_slices=7,
    )
    total = sum((q for _, q in slices), start=Decimal("0"))
    assert total == Decimal("100")


def test_twap_slice_rejects_zero_slices() -> None:
    from trading_system.mid_freq_engine.execution_algos import twap_slice

    with pytest.raises(ValueError):
        twap_slice(
            total_quantity=Decimal("10"),
            start=datetime(2026, 4, 19, 14, tzinfo=UTC),
            end=datetime(2026, 4, 19, 15, tzinfo=UTC),
            num_slices=0,
        )


def test_twap_slice_rejects_end_before_start() -> None:
    from trading_system.mid_freq_engine.execution_algos import twap_slice

    with pytest.raises(ValueError):
        twap_slice(
            total_quantity=Decimal("10"),
            start=datetime(2026, 4, 19, 15, tzinfo=UTC),
            end=datetime(2026, 4, 19, 14, tzinfo=UTC),
            num_slices=3,
        )


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------


def test_vwap_slice_weights_quantities_by_volume_profile() -> None:
    from trading_system.mid_freq_engine.execution_algos import vwap_slice

    # Higher volume in the middle bucket -> more quantity there.
    profile = (Decimal("1"), Decimal("3"), Decimal("1"))
    quantities = vwap_slice(total_quantity=Decimal("50"), volume_profile=profile)
    assert len(quantities) == 3
    assert quantities[1] > quantities[0]
    assert sum(quantities, start=Decimal("0")) == Decimal("50")


def test_vwap_slice_rejects_empty_profile() -> None:
    from trading_system.mid_freq_engine.execution_algos import vwap_slice

    with pytest.raises(ValueError):
        vwap_slice(total_quantity=Decimal("10"), volume_profile=())


def test_vwap_slice_rejects_zero_total_volume() -> None:
    from trading_system.mid_freq_engine.execution_algos import vwap_slice

    with pytest.raises(ValueError):
        vwap_slice(
            total_quantity=Decimal("10"),
            volume_profile=(Decimal("0"), Decimal("0")),
        )
