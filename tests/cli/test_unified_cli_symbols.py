from __future__ import annotations

import json

from src.cli.unified_cli import load_collection_symbols


def test_load_symbols_plain_list(tmp_path):
    p = tmp_path / "list.json"
    p.write_text(json.dumps(["aapl", "msft"]))
    assert load_collection_symbols(p) == ["AAPL", "MSFT"]


def test_load_symbols_dict_keys(tmp_path):
    p = tmp_path / "dict.json"
    p.write_text(json.dumps({"symbols": ["AAPL"], "name": "x"}))
    assert load_collection_symbols(p) == ["AAPL"]


def test_load_symbols_nested_named(tmp_path):
    p = tmp_path / "nested.json"
    p.write_text(json.dumps({"bonds": {"symbols": ["TLT", "IEF"]}}))
    assert load_collection_symbols(p) == ["TLT", "IEF"]
