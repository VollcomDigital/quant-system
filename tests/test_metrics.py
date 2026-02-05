import pandas as pd
import pytest


def _pandas_ok() -> bool:
    try:
        pd.Series([1.0, 2.0, 3.0]).sum()
    except TypeError:
        return False
    return True


if not _pandas_ok():  # pragma: no cover - guard for numpy reload under coverage
    pytest.skip(
        "NumPy reload detected under coverage; skipping pandas-heavy tests",
        allow_module_level=True,
    )

from src.backtest.metrics import omega_ratio, pain_index, tail_ratio


def test_omega_ratio_basic():
    returns = pd.Series([0.1, -0.05, 0.02])
    assert omega_ratio(returns) == pytest.approx(2.4, rel=1e-3)


def test_tail_ratio_quantiles():
    returns = pd.Series([0.1, -0.05, 0.02, 0.03, -0.04])
    assert tail_ratio(returns) == pytest.approx(1.7917, rel=1e-3)


def test_pain_index_from_equity():
    equity = pd.Series([100.0, 95.0, 105.0, 100.0])
    assert pain_index(equity) == pytest.approx(0.04881, rel=1e-3)
