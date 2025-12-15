from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import typer
from typer.testing import CliRunner

if "ccxt" not in sys.modules:
    sys.modules["ccxt"] = types.ModuleType("ccxt")

from src.main import manifest_status, package_run

app = typer.Typer()
app.command()(manifest_status)
app.command()(package_run)
runner = CliRunner()


def test_manifest_status_reads_manifest_file(tmp_path: Path):
    run_dir = tmp_path / "20240101-000000"
    run_dir.mkdir()
    payload = [
        {
            "run_id": "20240101-000000",
            "status": "created",
            "source": "csv",
            "message": "Dashboard regenerated using CSV fallback",
        }
    ]
    (run_dir / "manifest_status.json").write_text(json.dumps(payload))

    result = runner.invoke(
        app,
        [
            "manifest-status",
            "--reports-dir",
            str(tmp_path),
            "--run-id",
            "20240101-000000",
        ],
    )
    assert result.exit_code == 0
    assert "source=csv" in result.stdout
    assert "Dashboard regenerated using CSV fallback" in result.stdout


def test_manifest_status_falls_back_to_summary(tmp_path: Path):
    run_dir = tmp_path / "20240102-000000"
    run_dir.mkdir()
    summary = {
        "manifest_refresh": [
            {
                "run_id": "20240102-000000",
                "status": "created",
                "source": "cache",
                "message": "Dashboard regenerated from results cache",
            }
        ]
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))

    result = runner.invoke(
        app,
        [
            "manifest-status",
            "--reports-dir",
            str(tmp_path),
            "--run-id",
            "20240102-000000",
        ],
    )
    assert result.exit_code == 0
    assert "source=cache" in result.stdout
    assert "Dashboard regenerated from results cache" in result.stdout


def test_manifest_status_missing_artifacts(tmp_path: Path):
    run_dir = tmp_path / "20240103-000000"
    run_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "manifest-status",
            "--reports-dir",
            str(tmp_path),
            "--run-id",
            "20240103-000000",
        ],
    )
    assert result.exit_code == 1
    assert "No manifest_status.json or summary.json" in result.stdout


def test_manifest_status_latest(tmp_path: Path):
    newer = tmp_path / "20240105-000000"
    newer.mkdir()
    (newer / "manifest_status.json").write_text(
        json.dumps(
            [
                {
                    "run_id": "20240105-000000",
                    "status": "created",
                    "source": "cache",
                    "message": "Dashboard regenerated from results cache",
                }
            ]
        )
    )

    result = runner.invoke(
        app,
        [
            "manifest-status",
            "--reports-dir",
            str(tmp_path),
            "--latest",
        ],
    )
    assert result.exit_code == 0
    assert "20240105-000000" in result.stdout
