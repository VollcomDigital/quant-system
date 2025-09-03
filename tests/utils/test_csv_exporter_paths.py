from __future__ import annotations

from src.utils.csv_exporter import RawDataCSVExporter


def test_csv_exporter_default_output_dir():
    exp = RawDataCSVExporter()
    assert str(exp.output_dir).endswith("exports/csv")
