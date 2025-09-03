from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pandas as pd


def _load_module():
    p = Path("scripts/data_health_report.py")
    spec = importlib.util.spec_from_file_location("data_health_report", p)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    assert mod is not None
    spec.loader.exec_module(mod)
    return mod


def test_health_report_outputs_csv(monkeypatch, tmp_path):
    mod = _load_module()

    # Build fake DF
    idx = pd.date_range("2023-01-01", periods=10, freq="D")
    df = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": 1}, index=idx)
    fake_dm = patch.object(mod, "UnifiedDataManager").start().return_value
    fake_dm.get_data.return_value = df

    # Create a simple collection file
    coll_dir = tmp_path / "config/collections/default"
    coll_dir.mkdir(parents=True, exist_ok=True)
    f = coll_dir / "bonds_core.json"
    f.write_text('{"bonds_core": {"symbols": ["TLT", "IEF"]}}')
    # Ensure resolver reads from tmp config
    import os as _os

    old_cwd = Path.cwd()
    _os.chdir(str(tmp_path))

    out_csv = tmp_path / "health.csv"
    rc = mod.main(
        ["bonds_core", "--interval", "1d", "--period", "max", "--out", str(out_csv)]
    )
    assert rc == 0
    assert out_csv.exists()
    rows = list(csv.DictReader(out_csv.open()))
    # two symbols
    assert len(rows) == 2
    patch.stopall()
    _os.chdir(str(old_cwd))
