from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config


def test_load_config_allows_missing_strategies(tmp_path: Path):
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
    assert cfg.strategies == []


def test_load_config_reliability_thresholds(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
reliability_thresholds:
  min_data_points: 500
  min_continuity_score: 0.98
  on_fail: skip_evaluation
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.reliability_thresholds is not None
    assert cfg.reliability_thresholds.min_data_points == 500
    assert cfg.reliability_thresholds.min_continuity_score == pytest.approx(0.98)
    assert cfg.reliability_thresholds.on_fail == "skip_evaluation"


def test_load_config_reliability_thresholds_invalid_on_fail(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
reliability_thresholds:
  on_fail: abort_run
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_reliability_thresholds_invalid_min_continuity_score(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
reliability_thresholds:
  min_continuity_score: 1.2
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError):
        load_config(path)
