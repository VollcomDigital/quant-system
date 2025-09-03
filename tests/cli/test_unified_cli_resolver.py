from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cli.unified_cli import resolve_collection_path


def test_resolver_default_and_custom(tmp_path: Path, monkeypatch):
    base = tmp_path / "config" / "collections"
    (base / "default").mkdir(parents=True)
    (base / "custom").mkdir(parents=True)

    # Create default and custom jsons
    (base / "default" / "bonds_core.json").write_text(
        json.dumps({"bonds_core": {"symbols": ["TLT"]}})
    )
    (base / "custom" / "stocks_traderfox_us_tech.json").write_text(
        json.dumps({"stocks_traderfox_us_tech": {"symbols": ["AAPL"]}})
    )

    # Chdir so resolver finds our structure
    monkeypatch.chdir(tmp_path)

    # Alias mapping: bonds -> bonds_core
    p1 = resolve_collection_path("bonds")
    assert p1.name == "bonds_core.json"
    assert p1.parent.name == "default"

    # Custom key resolves
    p2 = resolve_collection_path("stocks_traderfox_us_tech")
    assert p2.name == "stocks_traderfox_us_tech.json"
    assert p2.parent.name == "custom"


def test_resolver_missing_raises(tmp_path: Path, monkeypatch):
    (tmp_path / "config" / "collections").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_collection_path("does_not_exist")
