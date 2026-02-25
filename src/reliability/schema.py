"""Pydantic models for reliability metadata (VD-4344).

Defines three layers:
  1. **CollectionReliability** – per-collection YAML flags (is_verified, min_data_points, …).
  2. **ReliabilityThresholds** – global or per-collection enforcement thresholds.
  3. **SymbolContinuityReport** – computed result of a data-continuity analysis for one symbol.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class CollectionReliability(BaseModel):
    """Per-collection reliability metadata stored in the collection YAML.

    These are *declaration* flags set by a human reviewer or an automated
    ingestion pipeline.  They travel with the collection definition and are
    read at config-load time.

    Attributes:
        is_verified: Manual approval flag.  ``False`` means the collection
            has not been reviewed and should be treated as unreliable.
        min_data_points: Minimum number of OHLC bars required *per symbol*
            before any backtest (including fixed-param evaluation) is
            attempted.  ``None`` falls back to the global default.
        last_updated: ISO-8601 timestamp of the last human review or
            automated quality check.
        min_continuity_score: Per-collection override for the minimum
            acceptable Data Continuity Score (0.0–1.0).  ``None`` falls
            back to the global ``ReliabilityThresholds.min_continuity_score``.
    """

    is_verified: bool = True
    min_data_points: int | None = None
    last_updated: datetime | None = None
    min_continuity_score: float | None = Field(default=None, ge=0.0, le=1.0)


class ReliabilityThresholds(BaseModel):
    """Global enforcement thresholds (configurable via YAML root ``reliability:``).

    These separate *metric computation* from *execution gating*: every
    threshold here can be overridden per collection via
    ``CollectionReliability`` fields.

    Attributes:
        min_data_points: Global floor for the minimum bar count.
        min_continuity_score: Minimum Data Continuity Score (0.0–1.0).
            The VD-3919 guardrail table sets the fail condition at >0.1 %
            missing bars, which corresponds to a score of 0.999.
        max_gap_percentage: Maximum tolerated percentage of missing bars
            expressed as a fraction (0.001 = 0.1 %).
        max_kurtosis: Maximum excess kurtosis of log-returns before a
            collection is flagged as unreliable.
        require_verified: When ``True``, collections with
            ``is_verified == False`` are skipped.
    """

    min_data_points: int = Field(default=2000, ge=1)
    min_continuity_score: float = Field(default=0.999, ge=0.0, le=1.0)
    max_gap_percentage: float = Field(default=0.001, ge=0.0, le=1.0)
    max_kurtosis: float = Field(default=10.0, gt=0.0)
    require_verified: bool = True


class SymbolContinuityReport(BaseModel):
    """Computed result of a data-continuity analysis for a single symbol.

    This is a *metric* — it describes the quality of the fetched OHLC data
    for one (symbol, timeframe) pair.  It is **not** a threshold; compare
    ``score`` against ``ReliabilityThresholds.min_continuity_score`` (or
    the per-collection override) to decide whether to gate execution.

    Attributes:
        symbol: Canonical symbol identifier.
        timeframe: Bar interval (e.g. ``"1d"``, ``"4h"``).
        total_expected_bars: Number of bars expected given the date range
            and timeframe.
        total_present_bars: Number of bars actually present in the dataset.
        missing_bars: Number of detected gaps.
        score: Data Continuity Score in [0.0, 1.0].
            ``1.0`` = perfect continuity; ``0.0`` = no data at all.
        largest_gap_bars: Size of the longest contiguous gap (in bars).
    """

    symbol: str
    timeframe: str
    total_expected_bars: int = Field(ge=0)
    total_present_bars: int = Field(ge=0)
    missing_bars: int = Field(ge=0)
    score: float = Field(ge=0.0, le=1.0)
    largest_gap_bars: int = Field(default=0, ge=0)
