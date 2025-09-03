from __future__ import annotations

import types

from src.utils.csv_exporter import RawDataCSVExporter


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    # chainable API
    def filter(self, *a, **k):
        return self

    def all(self):
        return list(self._rows)


def test_db_primary_empty_falls_back_to_unified_models(monkeypatch, tmp_path):
    # Chdir for output paths
    monkeypatch.chdir(tmp_path)

    # Fake primary DB session returning empty
    class _PrimarySession:
        def query(self, *_):
            return _FakeQuery([])

        def close(self):
            pass

    def fake_get_db_session():
        return _PrimarySession()

    monkeypatch.setattr("src.utils.csv_exporter.get_db_session", fake_get_db_session)

    # Fake unified_models module with one BestStrategy row
    class Row:
        symbol = "AAPL"
        strategy = "BuyHold"
        timeframe = "1d"
        sortino_ratio = 1.1
        sharpe_ratio = 0.9
        calmar_ratio = 0.8
        total_return = 12.3
        max_drawdown = 5.0
        updated_at = None

    class _UnifiedSession:
        def query(self, *_):
            return _FakeQuery([Row()])

        def close(self):
            pass

    fake_um = types.SimpleNamespace(Session=lambda: _UnifiedSession(), BestStrategy=Row)
    monkeypatch.setitem(
        __import__("sys").modules, "src.database.unified_models", fake_um
    )

    exp = RawDataCSVExporter()
    files = exp.export_from_database_primary(
        quarter="Q3",
        year="2025",
        export_format="best-strategies",
        portfolio_name="Test",
        portfolio_path=None,
        interval="1d",
    )

    # Should produce a CSV via the fallback path
    assert files
    assert files[0].endswith(".csv")
