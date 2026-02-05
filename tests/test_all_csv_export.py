from __future__ import annotations

import csv
from pathlib import Path

from src.reporting.all_csv_export import AllCSVExporter


class _DummyCache:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def list_by_run(self, run_id: str) -> list[dict]:
        return list(self._rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_all_csv_export_includes_negative_metric(tmp_path: Path):
    rows = [
        {
            "collection": "test",
            "symbol": "AAA",
            "timeframe": "1d",
            "strategy": "stratA",
            "metric": "sharpe",
            "metric_value": -0.5,
            "params": {"x": 1},
            "stats": {
                "sharpe": -0.5,
                "sortino": -0.4,
                "omega": 0.8,
                "tail_ratio": 0.9,
                "profit": -0.1,
                "pain_index": 0.2,
                "trades": 2,
                "max_drawdown": -0.3,
            },
        },
        {
            "collection": "test",
            "symbol": "BBB",
            "timeframe": "1d",
            "strategy": "stratB",
            "metric": "sharpe",
            "metric_value": 0.2,
            "params": {"x": 2},
            "stats": {
                "sharpe": 0.2,
                "sortino": 0.1,
                "omega": 1.1,
                "tail_ratio": 1.2,
                "profit": 0.05,
                "pain_index": 0.1,
                "trades": 3,
                "max_drawdown": -0.1,
            },
        },
    ]

    exporter = AllCSVExporter(tmp_path, _DummyCache(rows), run_id="run-1", top_n=2)
    exporter.export([])

    all_rows = _read_rows(tmp_path / "all_results.csv")
    assert {r["symbol"] for r in all_rows} == {"AAA", "BBB"}

    top_rows = _read_rows(tmp_path / "top2.csv")
    assert {r["symbol"] for r in top_rows} == {"AAA", "BBB"}
