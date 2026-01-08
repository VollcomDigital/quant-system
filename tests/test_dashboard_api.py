from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.dashboard.server import create_app


def _write_summary(run_dir: Path, **fields: object) -> None:
    data = {
        "metric": "sharpe",
        "results_count": 5,
        "started_at": "2024-01-01T00:00:00Z",
        "finished_at": "2024-01-01T01:00:00Z",
    }
    data.update(fields)
    (run_dir / "summary.json").write_text(json.dumps(data))


def test_list_runs(tmp_path: Path):
    latest = tmp_path / "20240102-000000"
    latest.mkdir()
    _write_summary(latest, metric="sortino", results_count=10)
    older = tmp_path / "20240101-000000"
    older.mkdir()
    _write_summary(older, metric="sharpe", results_count=3)

    app = create_app(tmp_path)
    client = TestClient(app)
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert data[0]["run_id"] == "20240102-000000"
    assert data[1]["run_id"] == "20240101-000000"


def test_run_detail_not_found(tmp_path: Path):
    app = create_app(tmp_path)
    client = TestClient(app)
    resp = client.get("/api/runs/missing")
    assert resp.status_code == 404


def test_run_detail_page_and_download(tmp_path: Path):
    run_dir = tmp_path / "20240103-000000"
    run_dir.mkdir()
    summary_path = run_dir / "summary.json"
    summary_data = {
        "metric": "profit",
        "results_count": 7,
        "started_at": "2024-01-03T00:00:00Z",
        "finished_at": "2024-01-03T03:00:00Z",
    }
    summary_path.write_text(json.dumps(summary_data))
    report_path = run_dir / "report.html"
    report_path.write_text("<html></html>")

    app = create_app(tmp_path)
    client = TestClient(app)

    detail = client.get("/run/20240103-000000")
    assert detail.status_code == 200
    assert "Run 20240103-000000" in detail.text
    download = client.get("/api/runs/20240103-000000/files/report.html")
    assert download.status_code == 200
    assert download.text == "<html></html>"


def test_compare_api_and_page(tmp_path: Path):
    run_a = tmp_path / "20240106-000000"
    run_a.mkdir()
    (run_a / "summary.json").write_text(json.dumps({"metric": "sharpe", "results_count": 5}))
    with open(run_a / "summary.csv", "w") as f:
        f.write("symbol,strategy,metric,metric_value\nBTC,StratA,sharpe,1.5\n")

    run_b = tmp_path / "20240107-000000"
    run_b.mkdir()
    (run_b / "summary.json").write_text(json.dumps({"metric": "sortino", "results_count": 3}))
    with open(run_b / "summary.csv", "w") as f:
        f.write("symbol,strategy,metric,metric_value\nETH,StratB,sortino,2.0\n")

    app = create_app(tmp_path)
    client = TestClient(app)
    api_resp = client.get("/api/compare", params={"runs": "20240106-000000,20240107-000000"})
    assert api_resp.status_code == 200
    data = api_resp.json()
    assert "20240107-000000" in data

    page_resp = client.get("/compare", params={"runs": "20240106-000000,20240107-000000"})
    assert page_resp.status_code == 200
    assert "Run Comparison" in page_resp.text
