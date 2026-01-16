from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_returns(returns: pd.Series) -> pd.Series:
    if returns.empty:
        return returns
    return returns.dropna()


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    returns = _clean_returns(returns)
    if returns.empty:
        return float("nan")
    er = returns.mean() * periods_per_year - risk_free_rate
    sd = returns.std(ddof=0) * np.sqrt(periods_per_year)
    if sd == 0 or np.isnan(sd):
        if np.isclose(er, 0.0):
            return 0.0
        return float("inf") if er > 0 else float("-inf")
    return float(er / sd)


def sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    returns = _clean_returns(returns)
    if returns.empty:
        return float("nan")
    downside = returns.copy()
    downside[downside > 0] = 0
    dd = downside.std(ddof=0) * np.sqrt(periods_per_year)
    er = returns.mean() * periods_per_year - risk_free_rate
    if dd == 0 or np.isnan(dd):
        if np.isclose(er, 0.0):
            return 0.0
        return float("inf") if er > 0 else float("-inf")
    return float(er / dd)


def total_return(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    returns = _clean_returns(returns)
    if returns.empty:
        return float("nan")
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()
    if np.isclose(losses, 0.0):
        if gains > 0:
            return float("inf")
        return float("nan")
    return float(gains / losses)


def tail_ratio(returns: pd.Series, upper_q: float = 0.95, lower_q: float = 0.05) -> float:
    returns = _clean_returns(returns)
    if returns.empty:
        return float("nan")
    try:
        upper = float(np.quantile(returns, upper_q))
        lower = float(np.quantile(returns, lower_q))
    except Exception:
        return float("nan")
    denom = abs(lower)
    if np.isclose(denom, 0.0):
        return float("inf") if abs(upper) > 0 else float("nan")
    return float(abs(upper) / denom)


def pain_index(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    equity = equity.dropna()
    if equity.empty:
        return float("nan")
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    negatives = drawdown[drawdown < 0]
    if negatives.empty:
        return 0.0
    pain = -negatives.mean()
    return float(pain)
