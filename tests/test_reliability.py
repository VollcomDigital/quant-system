"""Tests for the reliability metadata module (VD-4344)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.config import load_config
from src.reliability.continuity import compute_continuity_score
from src.reliability.schema import (
    CollectionReliability,
    ReliabilityThresholds,
    SymbolContinuityReport,
)

# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestCollectionReliability:
    def test_defaults(self) -> None:
        r = CollectionReliability()
        assert r.is_verified is True
        assert r.min_data_points is None
        assert r.last_updated is None
        assert r.min_continuity_score is None

    def test_full_construction(self) -> None:
        r = CollectionReliability(
            is_verified=False,
            min_data_points=3000,
            last_updated=datetime(2026, 2, 25),
            min_continuity_score=0.995,
        )
        assert r.is_verified is False
        assert r.min_data_points == 3000
        assert r.last_updated == datetime(2026, 2, 25)
        assert r.min_continuity_score == 0.995

    def test_continuity_score_range_validation(self) -> None:
        with pytest.raises(ValidationError):
            CollectionReliability(min_continuity_score=1.5)
        with pytest.raises(ValidationError):
            CollectionReliability(min_continuity_score=-0.1)

    def test_model_validate_from_dict(self) -> None:
        raw = {
            "is_verified": False,
            "min_data_points": 1500,
            "last_updated": "2026-01-15T12:00:00",
        }
        r = CollectionReliability.model_validate(raw)
        assert r.is_verified is False
        assert r.min_data_points == 1500
        assert r.last_updated == datetime(2026, 1, 15, 12, 0, 0)


class TestReliabilityThresholds:
    def test_defaults(self) -> None:
        t = ReliabilityThresholds()
        assert t.min_data_points == 2000
        assert t.min_continuity_score == 0.999
        assert t.max_gap_percentage == 0.001
        assert t.max_kurtosis == 10.0
        assert t.require_verified is True

    def test_custom_values(self) -> None:
        t = ReliabilityThresholds(
            min_data_points=500,
            min_continuity_score=0.99,
            max_gap_percentage=0.01,
            max_kurtosis=5.0,
            require_verified=False,
        )
        assert t.min_data_points == 500
        assert t.min_continuity_score == 0.99
        assert t.max_gap_percentage == 0.01
        assert t.max_kurtosis == 5.0
        assert t.require_verified is False

    def test_min_data_points_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            ReliabilityThresholds(min_data_points=0)

    def test_max_kurtosis_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            ReliabilityThresholds(max_kurtosis=-1.0)


class TestSymbolContinuityReport:
    def test_basic_construction(self) -> None:
        r = SymbolContinuityReport(
            symbol="BTC/USDT",
            timeframe="1d",
            total_expected_bars=1000,
            total_present_bars=998,
            missing_bars=2,
            score=0.998,
            largest_gap_bars=1,
        )
        assert r.symbol == "BTC/USDT"
        assert r.score == 0.998
        assert r.missing_bars == 2

    def test_score_range_validation(self) -> None:
        with pytest.raises(ValidationError):
            SymbolContinuityReport(
                symbol="X",
                timeframe="1d",
                total_expected_bars=10,
                total_present_bars=10,
                missing_bars=0,
                score=1.5,
            )


# ---------------------------------------------------------------------------
# Continuity score tests
# ---------------------------------------------------------------------------


class TestComputeContinuityScore:
    def _make_daily_df(
        self, start: str, periods: int, *, drop_indices: list[int] | None = None
    ) -> pd.DataFrame:
        """Create a minimal OHLCV DataFrame with optional dropped bars."""
        idx = pd.bdate_range(start=start, periods=periods, freq="B")
        df = pd.DataFrame(
            {
                "open": np.random.default_rng(42).uniform(100, 200, len(idx)),
                "high": np.random.default_rng(42).uniform(200, 300, len(idx)),
                "low": np.random.default_rng(42).uniform(50, 100, len(idx)),
                "close": np.random.default_rng(42).uniform(100, 200, len(idx)),
                "volume": np.random.default_rng(42).integers(1000, 10000, len(idx)),
            },
            index=idx,
        )
        if drop_indices:
            df = df.drop(df.index[drop_indices])
        return df

    def test_perfect_continuity(self) -> None:
        df = self._make_daily_df("2024-01-01", 252)
        report = compute_continuity_score(df, "AAPL", "1d")
        assert report.score == 1.0
        assert report.missing_bars == 0

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame(
            columns=["open", "high", "low", "close"],
            index=pd.DatetimeIndex([], name="date"),
        )
        report = compute_continuity_score(df, "AAPL", "1d")
        assert report.score == 0.0
        assert report.total_expected_bars == 0
        assert report.total_present_bars == 0

    def test_missing_bars_lowers_score(self) -> None:
        df = self._make_daily_df("2024-01-01", 100, drop_indices=[10, 20, 30])
        report = compute_continuity_score(df, "AAPL", "1d")
        assert report.score < 1.0
        assert report.missing_bars > 0
        assert report.total_present_bars == 97

    def test_single_bar(self) -> None:
        idx = pd.DatetimeIndex([pd.Timestamp("2024-06-01")])
        df = pd.DataFrame(
            {"open": [100], "high": [110], "low": [90], "close": [105]},
            index=idx,
        )
        report = compute_continuity_score(df, "TEST", "1d")
        assert report.total_present_bars == 1
        assert report.score >= 1.0

    def test_crypto_7day_continuity(self) -> None:
        idx = pd.date_range(start="2024-01-01", periods=365, freq="1D")
        df = pd.DataFrame(
            {
                "open": np.ones(365),
                "high": np.ones(365),
                "low": np.ones(365),
                "close": np.ones(365),
            },
            index=idx,
        )
        report = compute_continuity_score(
            df, "BTC/USDT", "1d", trading_days_per_week=7
        )
        assert report.score == 1.0
        assert report.missing_bars == 0

    def test_hourly_continuity(self) -> None:
        idx = pd.date_range(start="2024-01-01", periods=100, freq="4h")
        df = pd.DataFrame(
            {
                "open": np.ones(100),
                "high": np.ones(100),
                "low": np.ones(100),
                "close": np.ones(100),
            },
            index=idx,
        )
        report = compute_continuity_score(
            df, "ETH/USDT", "4h", trading_days_per_week=7
        )
        assert report.score == 1.0

    def test_non_datetime_index_raises(self) -> None:
        df = pd.DataFrame(
            {"open": [1], "high": [2], "low": [0.5], "close": [1.5]},
            index=[0],
        )
        with pytest.raises(ValueError, match="DatetimeIndex"):
            compute_continuity_score(df, "X", "1d")

    def test_unknown_timeframe_returns_fallback(self) -> None:
        idx = pd.DatetimeIndex([pd.Timestamp("2024-01-01")])
        df = pd.DataFrame(
            {"open": [100], "high": [110], "low": [90], "close": [105]},
            index=idx,
        )
        report = compute_continuity_score(df, "X", "3d")
        assert report.score == 1.0
        assert report.missing_bars == 0

    def test_largest_gap_detection(self) -> None:
        idx = pd.date_range(start="2024-01-01", periods=30, freq="1D")
        drop = list(range(10, 20))
        df = pd.DataFrame(
            {
                "open": np.ones(30),
                "high": np.ones(30),
                "low": np.ones(30),
                "close": np.ones(30),
            },
            index=idx,
        ).drop(idx[drop])
        report = compute_continuity_score(
            df, "BTC/USDT", "1d", trading_days_per_week=7
        )
        assert report.largest_gap_bars >= 9
        assert report.missing_bars == 10


# ---------------------------------------------------------------------------
# Config integration tests
# ---------------------------------------------------------------------------


class TestConfigReliabilityParsing:
    def test_config_without_reliability_uses_defaults(self, tmp_path: Path) -> None:
        config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
"""
        path = tmp_path / "config.yaml"
        path.write_text(config_text)
        cfg = load_config(path)

        assert cfg.reliability.min_data_points == 2000
        assert cfg.reliability.min_continuity_score == 0.999
        assert cfg.reliability.require_verified is True

        col = cfg.collections[0]
        assert col.reliability.is_verified is True
        assert col.reliability.min_data_points is None

    def test_config_with_global_reliability(self, tmp_path: Path) -> None:
        config_text = """
reliability:
  min_data_points: 500
  min_continuity_score: 0.99
  require_verified: false
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
"""
        path = tmp_path / "config.yaml"
        path.write_text(config_text)
        cfg = load_config(path)

        assert cfg.reliability.min_data_points == 500
        assert cfg.reliability.min_continuity_score == 0.99
        assert cfg.reliability.require_verified is False

    def test_config_with_collection_reliability(self, tmp_path: Path) -> None:
        config_text = """
collections:
  - name: crypto
    source: binance
    symbols: ['BTC/USDT']
    reliability:
      is_verified: false
      min_data_points: 3000
      last_updated: "2026-02-25T00:00:00"
      min_continuity_score: 0.9999
timeframes: ['1d']
metric: sharpe
"""
        path = tmp_path / "config.yaml"
        path.write_text(config_text)
        cfg = load_config(path)

        col = cfg.collections[0]
        assert col.reliability.is_verified is False
        assert col.reliability.min_data_points == 3000
        assert col.reliability.min_continuity_score == 0.9999
        assert col.reliability.last_updated == datetime(2026, 2, 25)

    def test_config_mixed_collections(self, tmp_path: Path) -> None:
        config_text = """
reliability:
  min_data_points: 2000
  require_verified: true
collections:
  - name: stocks
    source: yfinance
    symbols: ['AAPL']
    reliability:
      is_verified: true
      min_data_points: 2500
  - name: bonds
    source: yfinance
    symbols: ['AGGH.L']
timeframes: ['1d']
metric: sharpe
"""
        path = tmp_path / "config.yaml"
        path.write_text(config_text)
        cfg = load_config(path)

        stocks = cfg.collections[0]
        assert stocks.reliability.is_verified is True
        assert stocks.reliability.min_data_points == 2500

        bonds = cfg.collections[1]
        assert bonds.reliability.is_verified is True
        assert bonds.reliability.min_data_points is None
