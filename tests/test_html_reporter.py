from __future__ import annotations

import re
from pathlib import Path

from src.backtest.runner import BestResult
from src.reporting.html import HTMLReporter


class _DummyCache:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def list_by_run(self, run_id: str) -> list[dict]:
        return list(self._rows)


def _row(symbol: str, metric_value: float) -> dict:
    return {
        "collection": "demo",
        "symbol": symbol,
        "timeframe": "1d",
        "strategy": "stratA",
        "params": {"x": 1},
        "metric": "sharpe",
        "metric_value": metric_value,
        "stats": {
            "sharpe": metric_value,
            "sortino": metric_value + 0.1,
            "omega": 1.2,
            "tail_ratio": 1.1,
            "profit": 0.05,
            "pain_index": 0.02,
            "trades": 3,
            "max_drawdown": -0.1,
            "cagr": 0.12,
            "calmar": -1.2,
            "equity_curve": [
                {"ts": "2024-01-01T00:00:00", "value": 1.0},
                {"ts": "2024-01-02T00:00:00", "value": 1.05},
            ],
            "drawdown_curve": [
                {"ts": "2024-01-01T00:00:00", "value": 0.0},
                {"ts": "2024-01-02T00:00:00", "value": -0.02},
            ],
            "trades_log": [{"entry_time": "2024-01-01", "exit_time": "2024-01-02"}],
        },
    }


def test_html_reporter_exports_inline_css(tmp_path: Path):
    rows = [_row("AAA", 1.2), _row("BBB", 0.8)]
    reporter = HTMLReporter(tmp_path, _DummyCache(rows), run_id="run-1", top_n=2, inline_css=True)
    reporter.export([])

    output = (tmp_path / "report.html").read_text()
    assert "Backtest Report" in output
    assert "Metric vs. Sharpe" in output
    assert "Equity & Drawdown Explorer" in output
    assert "https://cdn.tailwindcss.com" not in output
    assert "https://cdn.plot.ly/plotly-2.32.0.min.js" not in output
    assert 'src="plotly.min.js"' in output
    assert (tmp_path / "plotly.min.js").exists()


def test_html_reporter_fallback_from_best_results(tmp_path: Path):
    best = [
        BestResult(
            collection="demo",
            symbol="ZZZ",
            timeframe="1d",
            strategy="stratZ",
            params={"x": 2},
            metric_name="sharpe",
            metric_value=0.5,
            stats=_row("ZZZ", 0.5)["stats"],
        )
    ]
    reporter = HTMLReporter(tmp_path, _DummyCache([]), run_id="run-2", top_n=1, inline_css=False)
    reporter.export(best)

    output = (tmp_path / "report.html").read_text()
    assert "Backtest Report" in output
    # SRI on a cross-origin <script> requires crossorigin="anonymous" (otherwise some browsers
    # fetch in no-cors mode and block integrity validation, breaking Tailwind styling).
    tailwind_match = re.search(r'<script[^>]+src="https://cdn\\.tailwindcss\\.com[^"]*"[^>]*>', output)
    assert tailwind_match, "Expected Tailwind CDN <script> tag in non-inline HTML report"
    assert 'integrity="' in tailwind_match.group(0)
    assert 'crossorigin="anonymous"' in tailwind_match.group(0)


def test_html_reporter_escapes_user_content(tmp_path: Path):
    # Symbols/strategy names can come from user configs and external strategy repos.
    # The HTML reporter should escape content to avoid XSS when opening report.html.
    xss = "<img src=x onerror=alert(1)>"
    rows = [_row(xss, 1.2)]
    reporter = HTMLReporter(tmp_path, _DummyCache(rows), run_id="run-xss", top_n=1, inline_css=False)
    reporter.export([])

    output = (tmp_path / "report.html").read_text()
    assert xss not in output
    assert "&lt;img src=x onerror=alert(1)&gt;" in output
