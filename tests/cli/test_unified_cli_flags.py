from __future__ import annotations

import json

from src.cli.unified_cli import handle_collection_run


def test_no_cache_flag_sets_use_cache_false(tmp_path, monkeypatch, capsys):
    # Create temp collection tree in a writable tmp directory
    base = tmp_path / "config" / "collections" / "default"
    base.mkdir(parents=True, exist_ok=True)
    (base / "bonds_core.json").write_text(
        json.dumps({"bonds_core": {"symbols": ["TLT"]}})
    )
    # Chdir so resolver finds temp config path
    monkeypatch.chdir(tmp_path)

    # Use --dry-run so manifest prints to stdout
    rc = handle_collection_run(
        [
            "bonds_core",
            "--action",
            "direct",
            "--interval",
            "1d",
            "--period",
            "max",
            "--no-cache",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    manifest = json.loads(out)
    assert manifest["plan"]["use_cache"] is False
