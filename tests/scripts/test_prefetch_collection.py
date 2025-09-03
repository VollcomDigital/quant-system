from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

_path = Path("scripts/prefetch_collection.py")
_spec = importlib.util.spec_from_file_location("prefetch_collection", _path)
assert _spec is not None
assert _spec.loader is not None
prefetch_mod = importlib.util.module_from_spec(_spec)
assert prefetch_mod is not None
_spec.loader.exec_module(prefetch_mod)
prefetch = prefetch_mod.prefetch


def test_prefetch_full(monkeypatch):
    fake_dm = MagicMock()
    monkeypatch.setattr(
        prefetch_mod, "UnifiedDataManager", MagicMock(return_value=fake_dm)
    )
    prefetch("bonds_core", mode="full", interval="1d", recent_days=90)
    dm = fake_dm
    # full: period='max', use_cache=False
    dm.get_batch_data.assert_called_once()
    args, kwargs = dm.get_batch_data.call_args
    assert kwargs.get("use_cache") is False
    assert kwargs.get("period") == "max"


def test_prefetch_recent(monkeypatch):
    fake_dm = MagicMock()
    monkeypatch.setattr(
        prefetch_mod, "UnifiedDataManager", MagicMock(return_value=fake_dm)
    )
    prefetch("bonds_core", mode="recent", interval="1d", recent_days=90)
    dm = fake_dm
    dm.get_batch_data.assert_called_once()
    args, kwargs = dm.get_batch_data.call_args
    assert kwargs.get("use_cache") is False
    # recent should not set provider period
    assert kwargs.get("period") is None
