"""Tests for TradingView Alert Exporter utility."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import mock_open, patch

from src.utils.tv_alert_exporter import TradingViewAlertExporter


class TestTradingViewAlertExporter:
    """Test cases for TradingViewAlertExporter class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        exporter = TradingViewAlertExporter()
        assert exporter.reports_dir == Path("exports/reports")

    def test_init_custom_reports_dir(self):
        """Test initialization with custom reports directory."""
        custom_dir = "custom/reports"
        exporter = TradingViewAlertExporter(reports_dir=custom_dir)
        assert exporter.reports_dir == Path(custom_dir)

    def test_get_quarter_from_date(self):
        """Test quarter calculation from date."""
        exporter = TradingViewAlertExporter()

        # Test Q1
        jan_date = datetime(2023, 1, 15)
        year, quarter = exporter.get_quarter_from_date(jan_date)
        assert year == 2023
        assert quarter == 1

        # Test Q2
        may_date = datetime(2023, 5, 10)
        year, quarter = exporter.get_quarter_from_date(may_date)
        assert year == 2023
        assert quarter == 2

    @patch("src.utils.tv_alert_exporter.Path.mkdir")
    def test_organize_output_path(self, mock_mkdir):
        """Test organized output path creation."""
        exporter = TradingViewAlertExporter()

        with patch("src.utils.tv_alert_exporter.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 6, 15)
            mock_datetime.timezone.utc = datetime.now().tzinfo

            output_path = exporter.organize_output_path("test_base")

            assert "2023" in str(output_path)
            assert "Q2" in str(output_path)
            mock_mkdir.assert_called_once()

    def test_extract_asset_data_basic(self):
        """Test basic asset data extraction."""
        exporter = TradingViewAlertExporter()

        html_content = """
        <html>
            <body>
                <table>
                    <tr><th>Symbol</th><th>Strategy</th><th>Timeframe</th></tr>
                    <tr><td>AAPL</td><td>BuyAndHold</td><td>1D</td></tr>
                </table>
            </body>
        </html>
        """

        assets = exporter.extract_asset_data(html_content)
        assert isinstance(assets, list)

    def test_generate_alert_message(self):
        """Test alert message generation."""
        exporter = TradingViewAlertExporter()

        asset_data = {
            "symbol": "AAPL",
            "strategy": "BuyAndHold",
            "timeframe": "1D",
            "metrics": {"Sharpe Ratio": "1.2", "Net Profit": "15%", "Win Rate": "65%"},
        }

        alert = exporter.generate_tradingview_alert(asset_data)

        assert isinstance(alert, str)
        assert "AAPL" in alert
        assert "BuyAndHold" in alert
        assert "1D" in alert

    @patch("builtins.open", new_callable=mock_open, read_data="<html></html>")
    @patch("src.utils.tv_alert_exporter.Path.exists")
    def test_process_html_file(self, mock_exists, mock_file):
        """Test HTML file processing."""
        mock_exists.return_value = True

        exporter = TradingViewAlertExporter()
        test_file = Path("test_report.html")

        with patch.object(exporter, "extract_asset_data", return_value=[]):
            result = exporter.process_html_file(test_file)

        assert isinstance(result, list)

    @patch("os.walk")
    def test_find_html_reports(self, mock_walk):
        """Test finding HTML report files."""
        mock_walk.return_value = [
            ("/exports/reports", [], ["report1.html", "report2.html", "data.json"])
        ]

        exporter = TradingViewAlertExporter()
        html_files = exporter.find_html_reports()

        assert len(html_files) == 2
        assert all(str(f).endswith(".html") for f in html_files)


class TestIntegration:
    """Integration tests for the complete workflow."""

    @patch("os.walk")
    @patch("builtins.open", new_callable=mock_open, read_data="<html></html>")
    def test_complete_workflow(self, mock_file, mock_walk):
        """Test complete workflow from finding files to processing."""
        mock_walk.return_value = [("/exports/reports", [], ["test_report.html"])]

        exporter = TradingViewAlertExporter()

        # Find HTML files
        html_files = exporter.find_html_reports()
        assert len(html_files) > 0

        # Process first file
        with patch.object(exporter, "extract_asset_data", return_value=[]):
            result = exporter.process_html_file(html_files[0])
            assert isinstance(result, list)


# Test comment
