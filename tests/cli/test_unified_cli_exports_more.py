from __future__ import annotations

import json
import sys
from types import ModuleType

from src.cli.unified_cli import handle_collection_run


def test_exports_tradingview_and_ai_dry_run(monkeypatch, tmp_path):
    # Minimal collection
    base = tmp_path / "config" / "collections" / "default"
    base.mkdir(parents=True, exist_ok=True)
    (base / "bonds_core.json").write_text(
        json.dumps({"bonds_core": {"symbols": ["TLT", "IEF"], "name": "Bonds"}})
    )
    monkeypatch.chdir(tmp_path)

    # Fake TradingView exporter
    tv_mod = ModuleType("src.utils.tv_alert_exporter")

    class FakeTradingViewAlertExporter:
        last_init = None
        last_call = None

        def __init__(self, reports_dir: str):
            FakeTradingViewAlertExporter.last_init = {"reports_dir": reports_dir}

        def export_alerts(self, **kwargs):
            FakeTradingViewAlertExporter.last_call = kwargs
            return [{"symbol": s} for s in kwargs.get("symbols", [])]

    tv_mod.TradingViewAlertExporter = FakeTradingViewAlertExporter
    monkeypatch.setitem(sys.modules, "src.utils.tv_alert_exporter", tv_mod)

    # Fake AI recommendations + db connection
    ai_mod = ModuleType("src.ai.investment_recommendations")

    class FakeAI:
        last_init = None
        last_call = None

        def __init__(self, db_session=None):
            FakeAI.last_init = {"db_session": db_session}

        def generate_portfolio_recommendations(self, **kwargs):
            FakeAI.last_call = kwargs
            return {"ok": True}, "exports/ai_reco/fake.html"

    ai_mod.AIInvestmentRecommendations = FakeAI
    monkeypatch.setitem(sys.modules, "src.ai.investment_recommendations", ai_mod)

    db_mod = ModuleType("src.database.db_connection")

    def fake_get_db_session():
        return None

    db_mod.get_db_session = fake_get_db_session  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.database.db_connection", db_mod)

    # Run CLI with dry-run to trigger exports only
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
            "tradingview,ai",
            "--dry-run",
        ]
    )

    assert rc == 0
    # TradingView exporter initialized and called
    assert FakeTradingViewAlertExporter.last_init == {"reports_dir": "exports/reports"}
    assert FakeTradingViewAlertExporter.last_call is not None
    assert FakeTradingViewAlertExporter.last_call.get("interval") == "1d"
    assert FakeTradingViewAlertExporter.last_call.get("collection_filter") == "Bonds"
    assert FakeTradingViewAlertExporter.last_call.get("symbols") == ["IEF", "TLT"]

    # AI recommendations called with timeframe propagation
    assert FakeAI.last_call is not None
    assert FakeAI.last_call.get("timeframe") == "1d"
    # Quarter string present (e.g., Q3_2025); format not asserted strictly
    assert isinstance(FakeAI.last_call.get("quarter"), str)
    assert "_" in FakeAI.last_call.get("quarter")
