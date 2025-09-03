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
          <tr><td>AAPL</td><td>BuyAndHold</td><td>1d</td><td>1.2</td><td>25.0</td></tr>
          <tr><td>MSFT</td><td>MeanRevert</td><td>1d</td><td>0.8</td><td>18.0</td></tr>
        </table>
      </body>
    </html>
    """
    target.write_text(html, encoding="utf-8")


def test_export_from_quarterly_reports_best_strategies(tmp_path, monkeypatch):
    # Ensure exporter reads from repo-relative exports/reports
    monkeypatch.chdir(tmp_path)
    reports_root = Path("exports") / "reports" / "2025" / "Q3"
    _write_sample_report_html(reports_root / "Tech_Portfolio.html")

    exp = RawDataCSVExporter()
    files = exp.export_from_quarterly_reports(
        quarter="Q3",
        year="2025",
        export_format="best-strategies",
        collection_name="Tech Portfolio",
        interval="1d",
    )

    # One CSV file generated in standard location
    assert len(files) == 1
    out = Path(files[0])
    assert out.exists()
    assert str(out).endswith(
        "exports/csv/2025/Q3/Tech_Portfolio_Collection_2025_Q3_1d.csv"
    )

    # Validate basic columns present
    with out.open(newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        assert set(["Asset", "Best Strategy", "Resolution"]).issubset(cols)
