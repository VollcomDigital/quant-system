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
    assert cfg.evaluation_mode == "backtest"


def test_load_config_accepts_evaluation_mode(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
evaluation_mode: walk_forward
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.evaluation_mode == "walk_forward"


def test_load_config_rejects_invalid_evaluation_mode(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
evaluation_mode: invalid_mode
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_reliability_thresholds(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
timeframes: ['1d']
metric: sharpe
validation:
  policy:
    on_fail: skip_job
  data_quality:
    min_data_points: 500
    min_continuity_score: 0.98
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.validation is not None
    assert cfg.validation.data_quality is not None
    assert cfg.validation.data_quality.min_data_points == 500
    assert cfg.validation.data_quality.min_continuity_score == pytest.approx(0.98)
    assert cfg.validation.policy is not None
    assert cfg.validation.policy.on_fail == "skip_job"


def test_load_config_collection_reliability_thresholds_override(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
    validation:
      policy:
        on_fail: skip_optimization
      data_quality:
        min_data_points: 250
        min_continuity_score: 0.95
timeframes: ['1d']
metric: sharpe
validation:
  policy:
    on_fail: skip_job
  data_quality:
    min_data_points: 500
    min_continuity_score: 0.98
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.validation is not None
    assert len(cfg.collections) == 1
    col = cfg.collections[0]
    assert col.validation is not None
    assert col.validation.data_quality is not None
    assert col.validation.data_quality.min_data_points == 250
    assert col.validation.data_quality.min_continuity_score == pytest.approx(0.95)
    assert col.validation.policy is not None
    assert col.validation.policy.on_fail == "skip_optimization"


@pytest.mark.parametrize(
    "on_fail_input",
    [
        "skip_job",
        "skip_optimization",
        "skip_collection",
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
validation:
  policy:
    on_fail: {on_fail_input}
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    cfg = load_config(path)
    assert cfg.validation is not None
    assert cfg.validation.policy is not None
    assert cfg.validation.policy.on_fail == on_fail_input


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
validation:
  data_quality:
    {reliability_yaml}
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_collection_reliability_thresholds_invalid_values(tmp_path: Path):
    config_text = """
collections:
  - name: test
    source: yfinance
    symbols: ['AAPL']
    validation:
      policy:
        on_fail: abort_run
timeframes: ['1d']
metric: sharpe
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)

    with pytest.raises(ValueError):
        load_config(path)
