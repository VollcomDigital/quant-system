from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    er = returns.mean() * periods_per_year - risk_free_rate
    sd = returns.std(ddof=0) * np.sqrt(periods_per_year)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(er / sd)


def sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    downside = returns.copy()
    downside[downside > 0] = 0
    dd = downside.std(ddof=0) * np.sqrt(periods_per_year)
    er = returns.mean() * periods_per_year - risk_free_rate
    if dd == 0 or np.isnan(dd):
        return float("nan")
    return float(er / dd)


def total_return(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)
