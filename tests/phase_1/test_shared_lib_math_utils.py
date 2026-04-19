"""Phase 1 Task 3 - shared_lib.math_utils invariants.

The module provides three groups of helpers:

1. Vectorized return math (simple and log returns, cumulative returns,
   CAGR) backed by NumPy. These must tolerate zero/NaN edges.
2. Stable risk metrics (volatility, Sharpe, max drawdown, Sortino) that
   work on any 1-D array-like without pandas.
3. Decimal-safe money helpers (`Money`, quantize, add, subtract, multiply,
   ratio) for ledger logic. Floats must be refused.
"""

from __future__ import annotations

import math
from decimal import Decimal

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


def test_simple_returns_basic_vector() -> None:
    from shared_lib.math_utils import simple_returns

    prices = np.array([100.0, 101.0, 99.99])
    r = simple_returns(prices)
    assert r.shape == (2,)
    np.testing.assert_allclose(r, [0.01, -0.01])


def test_simple_returns_handles_zero_price_without_inf() -> None:
    from shared_lib.math_utils import simple_returns

    prices = np.array([0.0, 1.0, 2.0])
    r = simple_returns(prices)
    # First diff would be inf; contract: replace inf with NaN.
    assert np.isnan(r[0])
    assert math.isclose(r[1], 1.0)


def test_simple_returns_rejects_empty() -> None:
    from shared_lib.math_utils import simple_returns

    with pytest.raises(ValueError, match="at least 2"):
        simple_returns(np.array([]))


def test_log_returns_is_mathematically_consistent() -> None:
    from shared_lib.math_utils import log_returns

    prices = np.array([100.0, 110.0, 121.0])
    r = log_returns(prices)
    np.testing.assert_allclose(r, np.log([1.1, 1.1]), rtol=1e-12)


def test_cumulative_returns_compounds() -> None:
    from shared_lib.math_utils import cumulative_returns

    r = np.array([0.1, -0.1, 0.2])
    cum = cumulative_returns(r)
    np.testing.assert_allclose(cum, [0.1, -0.01, 0.188], rtol=1e-12)


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------


def test_volatility_annualizes_correctly() -> None:
    from shared_lib.math_utils import volatility

    daily = np.random.default_rng(0).normal(0, 0.01, size=252)
    vol = volatility(daily, periods_per_year=252)
    # Should be close to 0.01 * sqrt(252).
    expected = 0.01 * math.sqrt(252)
    assert abs(vol - expected) < 0.02


def test_sharpe_ratio_zero_excess_is_zero() -> None:
    from shared_lib.math_utils import sharpe_ratio

    r = np.array([0.01, 0.01, 0.01, 0.01])
    # Constant returns have zero volatility -> contract: return 0.0 not inf.
    assert sharpe_ratio(r, risk_free=0.01 * 252, periods_per_year=252) == 0.0


def test_max_drawdown_monotone_up_is_zero() -> None:
    from shared_lib.math_utils import max_drawdown

    r = np.array([0.01, 0.01, 0.01, 0.01])
    assert max_drawdown(r) == 0.0


def test_max_drawdown_simple_dip() -> None:
    from shared_lib.math_utils import max_drawdown

    # Equity: 1.0 -> 1.2 -> 0.6 -> 1.0 -> drawdown from 1.2 to 0.6 = -50%.
    r = np.array([0.2, -0.5, 2.0 / 3.0])
    dd = max_drawdown(r)
    assert abs(dd - (-0.5)) < 1e-9


def test_sortino_handles_all_positive_returns() -> None:
    from shared_lib.math_utils import sortino_ratio

    r = np.array([0.01, 0.02, 0.03])
    # No downside -> contract: return 0.0, not inf.
    assert sortino_ratio(r, risk_free=0.0, periods_per_year=252) == 0.0


# ---------------------------------------------------------------------------
# Decimal-safe money
# ---------------------------------------------------------------------------


def test_money_rejects_float_input() -> None:
    from shared_lib.math_utils import Money

    with pytest.raises(TypeError, match="float"):
        Money(1.1, "USD")  # type: ignore[arg-type]


def test_money_equality_and_quantize() -> None:
    from shared_lib.math_utils import Money

    a = Money("1.23456", "USD")
    b = Money(Decimal("1.23456"), "USD")
    assert a == b
    assert a.quantize(Decimal("0.01")) == Money("1.23", "USD")


def test_money_add_same_currency() -> None:
    from shared_lib.math_utils import Money

    a = Money("10.00", "USD")
    b = Money("0.05", "USD")
    assert a + b == Money("10.05", "USD")


def test_money_add_rejects_mismatched_currency() -> None:
    from shared_lib.math_utils import Money

    with pytest.raises(ValueError, match="currency"):
        _ = Money("1", "USD") + Money("1", "EUR")


def test_money_multiply_by_decimal_keeps_currency() -> None:
    from shared_lib.math_utils import Money

    got = Money("100.00", "USD") * Decimal("0.25")
    assert got == Money("25.00", "USD")


def test_money_multiply_rejects_float_factor() -> None:
    from shared_lib.math_utils import Money

    with pytest.raises(TypeError):
        _ = Money("100", "USD") * 0.25  # type: ignore[operator]


def test_money_ratio_returns_decimal() -> None:
    from shared_lib.math_utils import Money

    ratio = Money("250", "USD") / Money("1000", "USD")
    assert ratio == Decimal("0.25")


def test_money_ratio_rejects_mismatched_currency() -> None:
    from shared_lib.math_utils import Money

    with pytest.raises(ValueError):
        _ = Money("1", "USD") / Money("1", "EUR")


# ---------------------------------------------------------------------------
# CAGR
# ---------------------------------------------------------------------------


def test_cagr_doubles_in_one_year() -> None:
    from shared_lib.math_utils import cagr

    # 252 daily returns that compound to 2x.
    factor = 2.0 ** (1.0 / 252)
    r = np.full(252, factor - 1.0)
    result = cagr(r, periods_per_year=252)
    assert abs(result - 1.0) < 1e-6


def test_cagr_rejects_empty() -> None:
    from shared_lib.math_utils import cagr

    with pytest.raises(ValueError):
        cagr(np.array([]), periods_per_year=252)


# ---------------------------------------------------------------------------
# Additional branch coverage
# ---------------------------------------------------------------------------


def test_as_float_array_rejects_non_1d() -> None:
    from shared_lib.math_utils import simple_returns

    with pytest.raises(ValueError, match="1-D"):
        simple_returns(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_log_returns_rejects_empty() -> None:
    from shared_lib.math_utils import log_returns

    with pytest.raises(ValueError):
        log_returns(np.array([1.0]))


def test_cumulative_returns_rejects_empty() -> None:
    from shared_lib.math_utils import cumulative_returns

    with pytest.raises(ValueError):
        cumulative_returns(np.array([]))


def test_volatility_single_obs_is_zero() -> None:
    from shared_lib.math_utils import volatility

    assert volatility(np.array([0.01]), periods_per_year=252) == 0.0


def test_sharpe_single_obs_is_zero() -> None:
    from shared_lib.math_utils import sharpe_ratio

    assert sharpe_ratio(np.array([0.01]), risk_free=0.0, periods_per_year=252) == 0.0


def test_max_drawdown_empty_is_zero() -> None:
    from shared_lib.math_utils import max_drawdown

    assert max_drawdown(np.array([])) == 0.0


def test_sortino_single_obs_is_zero() -> None:
    from shared_lib.math_utils import sortino_ratio

    assert sortino_ratio(np.array([0.01]), risk_free=0.0, periods_per_year=252) == 0.0


def test_cagr_total_loss_returns_minus_one() -> None:
    from shared_lib.math_utils import cagr

    # Losing 100% in a single period -> total becomes 0.
    assert cagr(np.array([-1.0]), periods_per_year=252) == -1.0


def test_money_rejects_bool() -> None:
    from shared_lib.math_utils import Money

    with pytest.raises(TypeError, match="bool"):
        Money(True, "USD")  # type: ignore[arg-type]


def test_money_rejects_empty_currency() -> None:
    from shared_lib.math_utils import Money

    with pytest.raises(ValueError):
        Money("1", "")


def test_money_subtract() -> None:
    from shared_lib.math_utils import Money

    assert Money("5", "USD") - Money("2", "USD") == Money("3", "USD")


def test_money_subtract_rejects_mismatched_currency() -> None:
    from shared_lib.math_utils import Money

    with pytest.raises(ValueError):
        _ = Money("5", "USD") - Money("2", "EUR")


def test_money_div_by_zero() -> None:
    from shared_lib.math_utils import Money

    with pytest.raises(ZeroDivisionError):
        _ = Money("1", "USD") / Money("0", "USD")


def test_money_mul_rejects_bool() -> None:
    from shared_lib.math_utils import Money

    with pytest.raises(TypeError):
        _ = Money("10", "USD") * True  # type: ignore[operator]


def test_money_mul_by_int() -> None:
    from shared_lib.math_utils import Money

    assert Money("10", "USD") * 3 == Money("30", "USD")


def test_money_add_returns_notimplemented_for_non_money() -> None:
    from shared_lib.math_utils import Money

    result = Money("1", "USD").__add__("not-money")  # type: ignore[arg-type]
    assert result is NotImplemented


def test_money_sub_returns_notimplemented_for_non_money() -> None:
    from shared_lib.math_utils import Money

    result = Money("1", "USD").__sub__(42)  # type: ignore[arg-type]
    assert result is NotImplemented


def test_money_div_returns_notimplemented_for_non_money() -> None:
    from shared_lib.math_utils import Money

    result = Money("1", "USD").__truediv__(2)  # type: ignore[arg-type]
    assert result is NotImplemented


def test_money_mul_returns_notimplemented_for_weird_type() -> None:
    from shared_lib.math_utils import Money

    result = Money("1", "USD").__mul__("nope")  # type: ignore[arg-type]
    assert result is NotImplemented
