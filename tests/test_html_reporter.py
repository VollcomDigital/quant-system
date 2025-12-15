from pathlib import Path
from types import SimpleNamespace

from src.reporting.html import HTMLReporter


class DummyCache:
    def __init__(self, rows):
        self._rows = rows

    def list_by_run(self, run_id: str):
        return self._rows


def test_html_reporter_generates_file(tmp_path: Path):
    # Prepare dummy rows (all results) and best results
    rows = [
        {
            "collection": "crypto",
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "strategy": "stratA",
            "metric": "sortino",
            "metric_value": 1.5,
            "params": {"x": 1},
            "stats": {
                "sharpe": 1.2,
                "sortino": 1.5,
                "omega": 1.8,
                "tail_ratio": 1.6,
                "profit": 0.2,
                "pain_index": 0.05,
                "trades": 10,
                "max_drawdown": -0.1,
                "equity_curve": [
                    {"ts": "2024-01-01T00:00:00", "value": 1.0},
                    {"ts": "2024-01-02T00:00:00", "value": 1.2},
                ],
                "drawdown_curve": [
                    {"ts": "2024-01-01T00:00:00", "value": 0.0},
                    {"ts": "2024-01-02T00:00:00", "value": -0.05},
                ],
                "trades_log": [
                    {
                        "Entry Timestamp": "2024-01-01T00:00:00",
                        "Exit Timestamp": "2024-01-01T12:00:00",
                        "Direction": "Long",
                        "Size": 1.0,
                        "PnL": 120.5,
                    }
                ],
            },
        }
    ]
    best = [
        SimpleNamespace(
            collection="crypto",
            symbol="BTC/USDT",
            timeframe="1d",
            strategy="stratA",
            params={"x": 1},
            metric_name="sortino",
            metric_value=1.5,
            stats={
                "sharpe": 1.2,
                "sortino": 1.5,
                "omega": 1.8,
                "tail_ratio": 1.6,
                "profit": 0.2,
                "pain_index": 0.05,
                "trades": 10,
                "max_drawdown": -0.1,
                "equity_curve": [
                    {"ts": "2024-01-01T00:00:00", "value": 1.0},
                    {"ts": "2024-01-02T00:00:00", "value": 1.2},
                ],
                "drawdown_curve": [
                    {"ts": "2024-01-01T00:00:00", "value": 0.0},
                    {"ts": "2024-01-02T00:00:00", "value": -0.05},
                ],
                "trades_log": [
                    {
                        "Entry Timestamp": "2024-01-01T00:00:00",
                        "Exit Timestamp": "2024-01-01T12:00:00",
                        "Direction": "Long",
                        "Size": 1.0,
                        "PnL": 120.5,
                    }
                ],
            },
        )
    ]
    cache = DummyCache(rows)
    HTMLReporter(tmp_path, cache, run_id="run-1", top_n=1, inline_css=True).export(best)
    html = (tmp_path / "report.html").read_text()
    assert "Backtest Report" in html
    assert "BTC/USDT" in html
    assert "Top Results (Overall)" in html
    assert "Best Per Strategy" in html
    assert "metric-scatter" in html
    assert "Equity & Drawdown Explorer" in html
    assert "detail-selector" in html
    assert "trade-table-container" in html
