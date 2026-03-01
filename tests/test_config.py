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
  on_fail: skip_job
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.reliability_thresholds is not None
    assert cfg.reliability_thresholds.min_data_points == 500
    assert cfg.reliability_thresholds.min_continuity_score == pytest.approx(0.98)
    assert cfg.reliability_thresholds.on_fail == "skip_job"


@pytest.mark.parametrize(
    "on_fail_input",
    [
        "skip_job",
        "skip_optimization",
    ],
)
def test_load_config_reliability_thresholds_on_fail_values(tmp_path: Path, on_fail_input: str):
    config_text = f"""
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
reliability_thresholds:
  on_fail: {on_fail_input}
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.reliability_thresholds is not None
    assert cfg.reliability_thresholds.on_fail == on_fail_input


@pytest.mark.parametrize(
    "reliability_yaml",
    [
        "on_fail: abort_run",
        "min_continuity_score: 1.2",
    ],
    ids=["invalid_on_fail", "invalid_continuity_score"],
)
def test_load_config_reliability_thresholds_invalid_values(tmp_path: Path, reliability_yaml: str):
    config_text = f"""
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
reliability_thresholds:
  {reliability_yaml}
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError):
        load_config(path)
