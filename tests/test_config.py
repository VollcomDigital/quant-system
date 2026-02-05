from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config


def test_load_config_requires_strategies(tmp_path: Path):
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

    with pytest.raises(ValueError, match="Missing required `strategies`"):
        load_config(path)
