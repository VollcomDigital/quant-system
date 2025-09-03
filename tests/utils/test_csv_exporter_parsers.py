from __future__ import annotations

from bs4 import BeautifulSoup

from src.utils.csv_exporter import RawDataCSVExporter


def test_parse_table_row_maps_headers():
    exp = RawDataCSVExporter()
    headers = ["Symbol", "Strategy", "Timeframe", "Sortino Ratio", "Total Return %"]
    cells = ["AAPL", "BuyHold", "1d", "1.2", "34.5"]
    row = exp._parse_table_row(headers, cells)
    assert row is not None
    assert row["Symbol"] == "AAPL"
    assert row["Strategy"] == "BuyHold"
    assert row["Timeframe"] == "1d"
    assert row["Sortino_Ratio"] == 1.2
    assert row["Total_Return_Pct"] == 34.5


def test_parse_metric_card_extracts_data():
    exp = RawDataCSVExporter()
    # Use a symbol that matches the regex in _parse_metric_card (e.g., BTCUSDT)
    html = '<div class="metric">BTCUSDT: 12.3% Strategy: Momentum</div>'
    soup = BeautifulSoup(html, "html.parser")
    card = soup.find("div")
    data = exp._parse_metric_card(card)
    assert data is not None
    assert data["Symbol"] == "BTCUSDT"
    assert data["Strategy"] == "Momentum"
    assert data["Total_Return_Pct"] == 12.3
