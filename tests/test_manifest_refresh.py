import json
from pathlib import Path

import pytest

from src.backtest.results_cache import ResultsCache
from src.reporting.dashboard import build_dashboard_payload
from src.reporting.manifest import refresh_manifest


class DummyCache(ResultsCache):
    def __init__(self, rows, cache_dir: Path):
        super().__init__(cache_dir)
        self._rows = rows

    def list_by_run(self, run_id: str):
        return self._rows


def test_refresh_manifest_creates_dashboard_for_legacy_run(tmp_path: Path):
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
                "max_drawdown": -0.1,
                "cagr": 0.12,
                "calmar": 1.1,
            },
        }
    ]
    cache = DummyCache(rows, tmp_path / "cache")

    current_dir = tmp_path / "20240201-000000"
    current_dir.mkdir()
    legacy_dir = tmp_path / "20240101-000000"
    legacy_dir.mkdir()

    current_payload = build_dashboard_payload(cache, "20240201-000000", rows)
    (current_dir / "dashboard.json").write_text(json.dumps(current_payload))
    summary = {
        "metric": "sortino",
        "results_count": 1,
        "started_at": "2024-01-01T00:00:00Z",
        "finished_at": "2024-01-01T01:00:00Z",
        "duration_sec": 3600,
    }
    (legacy_dir / "summary.json").write_text(json.dumps(summary))

    statuses = refresh_manifest(tmp_path, current_dir, cache, current_payload)

    legacy_dashboard = json.loads((legacy_dir / "dashboard.json").read_text())
    assert legacy_dashboard["run_id"] == "20240101-000000"
    assert "highlights" in legacy_dashboard
    assert legacy_dashboard["highlights"]["omega"]["value"] == pytest.approx(1.8)
    assert any(
        status["run_id"] == "20240101-000000" and status["status"] == "created"
        for status in statuses
    )


def test_refresh_manifest_csv_fallback_without_summary_json(tmp_path: Path):
    """Runs that only have all_results.csv (no summary.json) still get dashboard.json."""
    cache = DummyCache([], tmp_path / "cache")

    current_dir = tmp_path / "20240201-000000"
    current_dir.mkdir()
    legacy_dir = tmp_path / "20240101-000000"
    legacy_dir.mkdir()

    current_payload = build_dashboard_payload(cache, "20240201-000000", [])
    (current_dir / "dashboard.json").write_text(json.dumps(current_payload))

    (legacy_dir / "all_results.csv").write_text(
        "collection,symbol,timeframe,strategy,metric,metric_value,params,"
        "sharpe,sortino,omega,tail_ratio,profit,pain_index,trades,max_drawdown\n"
        "crypto,BTC,1d,S,sortino,1.5,{},1.0,1.5,1.8,1.6,0.2,0.05,3,-0.1\n",
        encoding="utf-8",
    )

    statuses = refresh_manifest(tmp_path, current_dir, cache, current_payload)

    legacy_dashboard = json.loads((legacy_dir / "dashboard.json").read_text(encoding="utf-8"))
    assert legacy_dashboard["run_id"] == "20240101-000000"
    assert legacy_dashboard["metric"] == "sortino"
    assert legacy_dashboard["source"] == "csv"
    assert any(
        s["run_id"] == "20240101-000000" and s["status"] == "created" for s in statuses
    )


def test_refresh_manifest_stub_csv_invalid_utf8_does_not_crash(tmp_path: Path):
    """Corrupt UTF-8 in all_results.csv must not abort manifest refresh."""
    cache = DummyCache([], tmp_path / "cache")
    current_dir = tmp_path / "20240201-000000"
    current_dir.mkdir()
    legacy_dir = tmp_path / "20240101-000000"
    legacy_dir.mkdir()
    current_payload = build_dashboard_payload(cache, "20240201-000000", [])
    (current_dir / "dashboard.json").write_text(json.dumps(current_payload))
    (legacy_dir / "all_results.csv").write_bytes(b"collection,metric\n\xff\xfe not utf-8,sortino\n")

    statuses = refresh_manifest(tmp_path, current_dir, cache, current_payload)

    assert any(
        s.get("run_id") == "20240101-000000" and s.get("status") == "missing_summary"
        for s in statuses
    )
