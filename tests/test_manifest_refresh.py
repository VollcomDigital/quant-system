import json
from pathlib import Path

import pytest

from src.backtest.results_cache import ResultsCache
from src.reporting.dashboard import build_dashboard_payload
from src.reporting.manifest import refresh_manifest


class DummyCache(ResultsCache):
    def __init__(self, rows):
        super().__init__(Path("/tmp"))
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
    cache = DummyCache(rows)

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
