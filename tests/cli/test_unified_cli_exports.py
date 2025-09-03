from __future__ import annotations

import json
import sys
from types import ModuleType

from src.cli.unified_cli import handle_collection_run


def test_exports_report_and_csv_dry_run(monkeypatch, tmp_path):
    # Create a minimal collection file the resolver can find
    base = tmp_path / "config" / "collections" / "default"
    base.mkdir(parents=True, exist_ok=True)
    (base / "bonds_core.json").write_text(
        json.dumps({"bonds_core": {"symbols": ["TLT", "IEF"]}})
    )
    monkeypatch.chdir(tmp_path)

    # Fake reporter module to capture call parameters without touching DB
    reporter_mod = ModuleType("src.reporting.collection_report")

    class FakeReporter:
        called = False
        last_kwargs = None

        def generate_comprehensive_report(
            self, portfolio_config, start_date, end_date, strategies, timeframes=None
        ):
            # Record call and verify required parameters are propagated
            FakeReporter.called = True
            FakeReporter.last_kwargs = {
                "portfolio_config": portfolio_config,
                "start": start_date,
                "end": end_date,
                "strategies": strategies,
                "timeframes": timeframes,
            }
            return "exports/reports/2025/Q3/Test.html"

    reporter_mod.DetailedPortfolioReporter = lambda: FakeReporter()
    monkeypatch.setitem(sys.modules, "src.reporting.collection_report", reporter_mod)

    # Fake CSV exporter to avoid DB access; capture interval passed in
    csv_mod = ModuleType("src.utils.csv_exporter")

    class FakeCSVExporter:
        last_instance = None

        def __init__(self, *a, **k):
            FakeCSVExporter.last_instance = self
            self.calls = []

        def export_from_database_primary(self, quarter, year, **kwargs):
            # Record call and return a dummy path
            self.calls.append((quarter, year, kwargs))
            return [f"exports/csv/{year}/{quarter}/dummy.csv"]

        def export_from_quarterly_reports(self, *a, **k):
            return []

    csv_mod.RawDataCSVExporter = FakeCSVExporter
    monkeypatch.setitem(sys.modules, "src.utils.csv_exporter", csv_mod)

    # Run CLI with dry-run so it executes export block only
    rc = handle_collection_run(
        [
            "bonds_core",
            "--action",
            "direct",
            "--interval",
            "1d",
            "--period",
            "max",
            "--exports",
            "report,csv",
            "--dry-run",
        ]
    )

    assert rc == 0
    # Reporter called with our interval propagated via timeframes
    assert FakeReporter.called is True
    assert FakeReporter.last_kwargs["timeframes"] == ["1d"]
    # CSV exporter was invoked and received interval in kwargs
    assert FakeCSVExporter.last_instance is not None
    calls = FakeCSVExporter.last_instance.calls
    assert calls
    assert calls[0][2].get("interval") == "1d"
