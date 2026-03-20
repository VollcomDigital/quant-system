from __future__ import annotations

from src.backtest.runner import BestResult
from src.reporting.tradingview import TradingViewExporter


def test_tradingview_export_writes_utf8_with_emoji(tmp_path) -> None:
    result = BestResult(
        collection="test",
        symbol="SPY",
        timeframe="1d",
        strategy="DummyStrategy",
        params={},
        metric_name="sortino",
        metric_value=1.0,
        stats={"sharpe": 1.0, "sortino": 1.0, "calmar": 1.0},
    )
    TradingViewExporter(tmp_path).export([result])
    text = (tmp_path / "tradingview.md").read_text(encoding="utf-8")
    assert "🚨" in text
    assert "SPY" in text
