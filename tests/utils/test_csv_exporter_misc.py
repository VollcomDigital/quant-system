from __future__ import annotations

import csv
from pathlib import Path

from src.utils.csv_exporter import RawDataCSVExporter


def _write_sample_report_html(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    html = """
    <html>
      <body>
        <table>
          <tr>
            <th>Symbol</th><th>Strategy</th><th>Timeframe</th><th>Sortino_Ratio</th><th>Total_Return_Pct</th>
          </tr>
          <tr><td>BTCUSDT</td><td>Momentum</td><td>1d</td><td>1.6</td><td>45.0</td></tr>
          <tr><td>ETHUSDT</td><td>BuyAndHold</td><td>1d</td><td>1.1</td><td>30.0</td></tr>
        </table>
      </body>
    </html>
    """
    target.write_text(html, encoding="utf-8")


def test_get_available_columns_contains_core_fields():
    exp = RawDataCSVExporter()
    cols = exp.get_available_columns()
    # Core identifiers and metrics should be present
    assert "Symbol" in cols
    assert "Strategy" in cols
    assert "Timeframe" in cols
    assert "Sortino_Ratio" in cols
    assert "Total_Return_Pct" in cols


def test_export_from_quarterly_reports_full(tmp_path, monkeypatch):
    # Anchor repo CWD and create an example HTML report
    monkeypatch.chdir(tmp_path)
    report_path = Path("exports") / "reports" / "2025" / "Q2" / "Crypto.html"
    _write_sample_report_html(report_path)

    exp = RawDataCSVExporter()
    files = exp.export_from_quarterly_reports(
        quarter="Q2",
        year="2025",
        export_format="full",
        collection_name="Crypto",
        interval="1d",
    )

    assert len(files) == 1
    out = Path(files[0])
    assert out.exists()
    # CSV should contain added metadata columns and original metrics
    with out.open(newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        assert "Quarter" in cols
        assert "Year" in cols
        assert "Export_Date" in cols
        assert "Symbol" in cols
        assert "Strategy" in cols
        assert "Timeframe" in cols
        # At least two rows written from our HTML table
        rows = list(reader)
        assert len(rows) >= 2
