"""Data Continuity Score computation (VD-4344).

The score quantifies how complete an OHLC time-series is relative to the
expected number of bars for its date range and timeframe.  A score of 1.0
indicates zero missing bars; 0.0 means no data at all.

Time Complexity: O(n) where n is the number of rows in the DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import SymbolContinuityReport

_TIMEFRAME_FREQ: dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1D",
    "1w": "1W",
}


def _infer_freq(timeframe: str) -> str | None:
    """Map a config timeframe string to a pandas frequency alias.

    Args:
        timeframe: Timeframe string from config (e.g. ``"1d"``, ``"4h"``).

    Returns:
        Pandas-compatible frequency alias, or ``None`` if unknown.
    """
    return _TIMEFRAME_FREQ.get(timeframe.lower())


def compute_continuity_score(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    *,
    trading_days_per_week: int = 5,
) -> SymbolContinuityReport:
    """Compute the Data Continuity Score for an OHLC DataFrame.

    The score is defined as::

        score = present_bars / expected_bars

    where ``expected_bars`` is derived from a regular frequency grid
    spanning ``[df.index.min(), df.index.max()]``.

    For daily (``1d``) timeframes the expected grid excludes weekends
    (business-day frequency) unless ``trading_days_per_week`` is set to 7
    (e.g. for crypto markets that trade 24/7).

    Args:
        df: OHLC DataFrame with a ``DatetimeIndex``.
        symbol: Canonical symbol identifier (for the report).
        timeframe: Bar interval string (e.g. ``"1d"``, ``"4h"``).
        trading_days_per_week: 5 for equity/FX, 7 for crypto.

    Returns:
        A ``SymbolContinuityReport`` with the computed score and gap stats.

    Raises:
        ValueError: If the DataFrame index is not a ``DatetimeIndex``.
    """
    if df.empty:
        return SymbolContinuityReport(
            symbol=symbol,
            timeframe=timeframe,
            total_expected_bars=0,
            total_present_bars=0,
            missing_bars=0,
            score=0.0,
            largest_gap_bars=0,
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            f"DataFrame index must be a DatetimeIndex, got {type(df.index).__name__}"
        )

    freq = _infer_freq(timeframe)
    if freq is None:
        return _fallback_score(df, symbol, timeframe)

    start, end = df.index.min(), df.index.max()

    if timeframe.lower() == "1d" and trading_days_per_week < 7:
        expected_index = pd.bdate_range(start=start, end=end, freq="B")
    else:
        expected_index = pd.date_range(start=start, end=end, freq=freq)

    total_expected = len(expected_index)
    total_present = len(df)

    if total_expected == 0:
        return SymbolContinuityReport(
            symbol=symbol,
            timeframe=timeframe,
            total_expected_bars=0,
            total_present_bars=total_present,
            missing_bars=0,
            score=1.0 if total_present > 0 else 0.0,
            largest_gap_bars=0,
        )

    missing = max(0, total_expected - total_present)
    score = min(1.0, total_present / total_expected) if total_expected > 0 else 0.0

    largest_gap = _largest_gap(df.index, freq, timeframe, trading_days_per_week)

    return SymbolContinuityReport(
        symbol=symbol,
        timeframe=timeframe,
        total_expected_bars=total_expected,
        total_present_bars=total_present,
        missing_bars=missing,
        score=round(score, 6),
        largest_gap_bars=largest_gap,
    )


def _largest_gap(
    index: pd.DatetimeIndex,
    freq: str,
    timeframe: str,
    trading_days_per_week: int,
) -> int:
    """Find the largest contiguous gap in the index measured in expected bars.

    Args:
        index: Sorted ``DatetimeIndex`` of the OHLC data.
        freq: Pandas frequency alias.
        timeframe: Original timeframe string.
        trading_days_per_week: 5 for equity, 7 for crypto.

    Returns:
        Number of missing bars in the longest gap.
    """
    if len(index) < 2:
        return 0

    sorted_idx = index.sort_values()
    deltas = np.diff(sorted_idx.values).astype("timedelta64[s]").astype(np.float64)

    if timeframe.lower() == "1d":
        expected_seconds = 86400.0
    elif timeframe.lower() == "1w":
        expected_seconds = 7 * 86400.0
    elif "h" in timeframe.lower():
        hours = int(timeframe.lower().replace("h", ""))
        expected_seconds = hours * 3600.0
    elif "m" in timeframe.lower():
        minutes = int(timeframe.lower().replace("m", "").replace("in", ""))
        expected_seconds = minutes * 60.0
    else:
        return 0

    gap_bars = (deltas / expected_seconds) - 1.0

    if timeframe.lower() == "1d" and trading_days_per_week < 7:
        gap_bars = np.where(gap_bars <= 2, 0, gap_bars - 2)

    gap_bars = np.maximum(gap_bars, 0)
    return int(np.max(gap_bars)) if len(gap_bars) > 0 else 0


def _fallback_score(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
) -> SymbolContinuityReport:
    """Produce a report when frequency inference fails.

    Args:
        df: OHLC DataFrame.
        symbol: Symbol identifier.
        timeframe: Timeframe string.

    Returns:
        A ``SymbolContinuityReport`` with score=1.0 and zero missing bars,
        since we cannot determine expected bars for an unknown frequency.
    """
    return SymbolContinuityReport(
        symbol=symbol,
        timeframe=timeframe,
        total_expected_bars=len(df),
        total_present_bars=len(df),
        missing_bars=0,
        score=1.0,
        largest_gap_bars=0,
    )
