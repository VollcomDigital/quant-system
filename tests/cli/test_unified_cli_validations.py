from __future__ import annotations

import json

from src.cli.unified_cli import handle_collection_run


def _write_collection(tmp_path):
    base = tmp_path / "config" / "collections" / "default"
    base.mkdir(parents=True, exist_ok=True)
    (base / "bonds_core.json").write_text(
        json.dumps({"bonds_core": {"symbols": ["TLT"]}})
    )


def test_start_without_end_returns_6(tmp_path, monkeypatch):
    _write_collection(tmp_path)
    monkeypatch.chdir(tmp_path)
    rc = handle_collection_run(
        [
            "bonds_core",
            "--action",
            "direct",
            "--interval",
            "1d",
            # Intentionally provide --start without --end to trigger rc 6
            "--start",
            "2024-01-01",
        ]
    )
    assert rc == 6


def test_invalid_interval_returns_5(tmp_path, monkeypatch):
    _write_collection(tmp_path)
    monkeypatch.chdir(tmp_path)
    rc = handle_collection_run(
        [
            "bonds_core",
            "--action",
            "direct",
            "--interval",
            "not-an-interval",
            "--period",
            "max",
            "--dry-run",
        ]
    )
    assert rc == 5
