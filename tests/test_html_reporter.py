from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

from src.backtest.runner import BestResult
from src.reporting.html import HTMLReporter


class _ScriptTagParser(HTMLParser):
    """Collect <script> tag attributes in a single HTML document."""

    def __init__(self) -> None:
        super().__init__()
        self.script_attrs: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "script":
            return
        d: dict[str, str] = {}
        for k, v in attrs:
            if v is None:
                continue
            d[k] = v
        self.script_attrs.append(d)


def _find_script_attrs_by_src_prefix(html_text: str, src_prefix: str) -> dict[str, str] | None:
    """Return the first <script> attrs whose src starts with `src_prefix`."""

    parser = _ScriptTagParser()
    parser.feed(html_text)
    for attrs in parser.script_attrs:
        src = attrs.get("src", "")
        if src.startswith(src_prefix):
            return attrs
    return None


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
    tailwind_attrs = _find_script_attrs_by_src_prefix(output, "https://cdn.tailwindcss.com")
    assert tailwind_attrs, "Expected Tailwind CDN <script> tag in non-inline HTML report"
    assert "integrity" in tailwind_attrs
    assert tailwind_attrs.get("crossorigin") == "anonymous"


def test_html_reporter_escapes_user_content(tmp_path: Path):
    # Symbols/strategy names can come from user configs and external strategy repos.
    # The HTML reporter should escape content to avoid XSS when opening report.html.
    # Keep the payload simple to avoid triggering static analyzers on "regex-like" strings.
    xss = "<img src=x>"
    rows = [_row(xss, 1.2)]
    reporter = HTMLReporter(tmp_path, _DummyCache(rows), run_id="run-xss", top_n=1, inline_css=False)
    reporter.export([])

    output = (tmp_path / "report.html").read_text()
    assert xss not in output
    assert "&lt;img src=x&gt;" in output
