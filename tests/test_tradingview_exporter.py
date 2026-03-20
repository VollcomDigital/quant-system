from __future__ import annotations

from pathlib import Path

import pytest

from src.backtest.runner import BestResult
from src.reporting.tradingview import TradingViewExporter

_real_path_write_text = Path.write_text


def test_tradingview_export_writes_utf8_with_emoji(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Spy write_text: reading the file back can pass on UTF-8-default locales even if encoding is omitted.
    encodings_seen: list[str | None] = []

    def capture_write_text(
        self: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        if self.name == "tradingview.md":
            encodings_seen.append(encoding)
        return _real_path_write_text(
            self, data, encoding=encoding, errors=errors, newline=newline
        )

    monkeypatch.setattr(Path, "write_text", capture_write_text)

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
    assert encodings_seen == ["utf-8"]
    text = (tmp_path / "tradingview.md").read_text(encoding="utf-8")
    assert "🚨" in text
    assert "SPY" in text
