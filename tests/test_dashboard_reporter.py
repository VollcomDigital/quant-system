import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.reporting.dashboard import (
    DashboardReporter,
    build_dashboard_payload,
    collect_runs_manifest,
)


class DummyCache:
    def __init__(self, rows):
        self._rows = rows

    def list_by_run(self, run_id: str):
        return self._rows


def _sample_row():
    return {
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
            "max_drawdown": -0.1,
            "cagr": 0.12,
            "calmar": 1.1,
        },
    }


def test_dashboard_reporter_writes_files(tmp_path: Path):
    rows = [_sample_row()]
    payload = build_dashboard_payload(DummyCache(rows), run_id="run-1", best_results=None)
    payload["available_metrics"] = ["omega", "tail_ratio", "pain_index"]
    payload["runs"] = [
        {
            "run_id": "run-1",
            "path": "dashboard.json",
            "metric": "sortino",
            "results_count": 1,
            "started_at": "2024-01-01T00:00:00Z",
        }
    ]
    payload["metric"] = "sortino"
    payload["results_count"] = 1
    (tmp_path / "report.html").write_text("<html></html>")
    DashboardReporter(tmp_path).export(payload)

    html = (tmp_path / "dashboard.html").read_text()
    data = json.loads((tmp_path / "dashboard.json").read_text())

    assert "Backtest Dashboard" in html
    assert "run-selector" in html
    assert "x-metric" in html
    assert "histogram" in html
    assert "highlights-list" in html
    assert "downloads-list" in html
    assert "Run History &amp; Comparison" in html
    assert "compare-metric" in html
    assert "Omega" in html
    assert data["available_metrics"][0] == "omega"
    assert data["highlights"]["omega"]["value"] == pytest.approx(rows[0]["stats"]["omega"])
    assert data["downloads"] == ["report.html"]


def test_dashboard_payload_falls_back_to_best_results(tmp_path: Path):
    best = [
        SimpleNamespace(
            collection="crypto",
            symbol="ETH/USDT",
            timeframe="4h",
            strategy="stratB",
            params={"y": 2},
            metric_name="sharpe",
            metric_value=2.1,
            stats={
                "sharpe": 2.1,
                "sortino": 2.3,
                "omega": 1.9,
                "tail_ratio": 1.7,
                "profit": 0.25,
                "pain_index": 0.04,
                "max_drawdown": -0.08,
                "cagr": 0.18,
                "calmar": 2.0,
            },
        )
    ]
    payload = build_dashboard_payload(DummyCache([]), run_id="run-2", best_results=best)
    assert payload["rows"]
    row = payload["rows"][0]
    assert row["symbol"] == "ETH/USDT"
    assert row["stats"]["omega"] == pytest.approx(1.9)


def test_collect_runs_manifest(tmp_path: Path):
    reports_root = tmp_path
    current_summary = {"counts": {"results": 10}}
    current_meta = {
        "metric": "sharpe",
        "results_count": 10,
        "started_at": "2024-02-01T00:00:00Z",
    }

    previous = reports_root / "20240101-000000"
    previous.mkdir()
    (previous / "summary.json").write_text(
        json.dumps(
            {
                "metric": "sortino",
                "results_count": 5,
                "started_at": "2024-01-01T00:00:00Z",
                "dashboard": {"counts": {"results": 5}},
            }
        )
    )

    manifest = collect_runs_manifest(reports_root, "20240201-000000", current_summary, current_meta)
    run_ids = [entry["run_id"] for entry in manifest]
    assert "20240101-000000" in run_ids
    assert "20240201-000000" in run_ids
    previous_entry = next(entry for entry in manifest if entry["run_id"] == "20240101-000000")
    assert "highlights" in previous_entry
    current_entry = next(entry for entry in manifest if entry["run_id"] == "20240201-000000")
    assert current_entry["summary"] == current_summary
    assert current_entry["highlights"] == {}
