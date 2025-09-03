from __future__ import annotations

import importlib
import json
import sys
from types import SimpleNamespace


def test_cli_probe_called(monkeypatch, tmp_path):
    # Create temp collection
    base = tmp_path / "config" / "collections" / "default"
    base.mkdir(parents=True)
    (base / "bonds_core.json").write_text(
        json.dumps({"bonds_core": {"symbols": ["TLT"]}})
    )
    monkeypatch.chdir(tmp_path)

    # Patch heavy dependencies in unified_cli
    mod = importlib.import_module("src.cli.unified_cli")

    # Fake UnifiedDataManager with probe flag
    class FakeDM:
        def __init__(self):
            self.called = False

        def probe_and_set_order(
            self, asset_type, symbols, interval="1d", sample_size=5
        ):
            self.called = True
            return ["yahoo_finance"]

    fake_dm_instance = FakeDM()
    # Inject a fake module so `from src.core.data_manager import UnifiedDataManager` returns our fake
    fake_dm_module = SimpleNamespace(UnifiedDataManager=lambda: fake_dm_instance)
    monkeypatch.setitem(sys.modules, "src.core.data_manager", fake_dm_module)

    # Patch direct backtest functions to no-op by injecting a fake module in sys.modules
    fake_direct_mod = SimpleNamespace(
        finalize_persistence_for_run=lambda *a, **k: None,
        run_direct_backtest=lambda **k: {},
    )
    monkeypatch.setitem(sys.modules, "src.core.direct_backtest", fake_direct_mod)

    # Patch unified_models ensure_run_for_manifest
    mod.unified_models = SimpleNamespace(  # type: ignore[attr-defined]
        ensure_run_for_manifest=lambda m: SimpleNamespace(run_id="test-run")
    )

    # Run without --dry-run so run_plan executes
    rc = mod.handle_collection_run(
        ["bonds_core", "--action", "direct", "--interval", "1d", "--period", "max"]
    )
    assert rc == 0
    assert fake_dm_instance.called is True
