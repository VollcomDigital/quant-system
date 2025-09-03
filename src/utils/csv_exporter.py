"""
Raw Data CSV Export Utility

Exports portfolio performance data with best strategies and timeframes to CSV format.
Based on the features.md specification and crypto_best_strategies.csv format.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from src.database.db_connection import get_db_session
from src.database.models import BestStrategy


class RawDataCSVExporter:
    """
    Export raw portfolio data with best strategies and performance metrics to CSV.

    Features:
    - CSV export with symbol, best strategy, best timeframe, and performance metrics
    - Bulk export for all assets from quarterly reports
    - Customizable column selection (Sharpe, Sortino, profit, drawdown)
    - Integration with existing quarterly report structure
    """

    def __init__(self, output_dir: str = "exports/csv"):
        # Default output directory aligned with repo: exports/csv
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir = Path("exports/reports")
        self.logger = logging.getLogger(__name__)

    def export_from_database_primary(
        self,
        quarter: str,
        year: str,
        output_filename: str | None = None,
        export_format: str = "full",
        portfolio_name: str = "all",
        portfolio_path: str | None = None,
        interval: str | None = None,
    ) -> list[str]:
        """
        Export data directly from database - primary data source for CSV exports.
        Filters by specific collection symbols if portfolio is provided.

        Args:
            quarter: Quarter (Q1, Q2, Q3, Q4)
            year: Year (YYYY)
            output_filename: Custom filename, auto-generated if None
            export_format: 'full', 'best-strategies', or 'quarterly'
            portfolio_name: Portfolio collection name for filtering
            portfolio_path: Path to portfolio config file for symbol filtering

        Returns:
            List of paths to exported CSV files
        """
        output_files = []

        try:
            db_session = get_db_session()

            # Load portfolio symbols for filtering if portfolio path provided
            portfolio_symbols = None
            if portfolio_path:
                try:
                    import json
                    from pathlib import Path

                    with Path(portfolio_path).open() as f:
                        portfolio_config = json.load(f)
                        # Get the first (and usually only) portfolio config
                        portfolio_key = list(portfolio_config.keys())[0]
                        portfolio_symbols = portfolio_config[portfolio_key].get(
                            "symbols", []
                        )
                        portfolio_name = portfolio_key  # Use actual collection name
                        self.logger.info(
                            "Filtering by %s symbols from %s collection",
                            len(portfolio_symbols),
                            portfolio_name,
                        )
                except Exception as e:
                    self.logger.warning("Could not load portfolio config: %s", e)

            # Query best strategies from database with optional symbol filtering.
            # Primary canonical table is backtests.best_strategies (models.BestStrategy).
            # If that is empty (e.g., legacy or different persistence layer), fall back
            # to the lightweight unified_models BestStrategy table (unified_models.BestStrategy).
            query = db_session.query(BestStrategy)

            if portfolio_symbols:
                query = query.filter(BestStrategy.symbol.in_(portfolio_symbols))
            # Filter by timeframe/interval if provided
            if "interval" in locals() and interval:
                try:
                    query = query.filter(BestStrategy.timeframe == interval)
                except Exception:
                    pass

            best_strategies = query.all()

            # Fallback to unified_models if no rows found in canonical backtests schema
            if not best_strategies:
                try:
                    from src.database import unified_models

                    sess2 = unified_models.Session()
                    try:
                        uq = sess2.query(unified_models.BestStrategy)
                        if portfolio_symbols:
                            uq = uq.filter(
                                unified_models.BestStrategy.symbol.in_(
                                    portfolio_symbols
                                )
                            )
                        if "interval" in locals() and interval:
                            try:
                                uq = uq.filter(
                                    unified_models.BestStrategy.timeframe == interval
                                )
                            except Exception:
                                pass
                        unified_rows = uq.all()
                        if unified_rows:
                            # Map unified_models rows into a structure compatible with the rest of this function.
                            # unified_models.BestStrategy has attributes with same names used below (symbol, timeframe, strategy, sortino_ratio, calmar_ratio, sharpe_ratio, total_return, max_drawdown, updated_at)
                            best_strategies = unified_rows
                            self.logger.info(
                                "Fell back to unified_models BestStrategy table (%d rows)",
                                len(best_strategies),
                            )
                    finally:
                        sess2.close()
                except Exception:
                    # If fallback fails, continue with empty list to trigger no-data path below
                    pass

            if not best_strategies:
                self.logger.warning(
                    "No strategies found in database for specified filters"
                )
                return output_files

            self.logger.info(
                "Found %s strategies in database for %s collection",
                len(best_strategies),
                portfolio_name,
            )

            # Convert to DataFrame
            data = []
            for strategy in best_strategies:
                data.append(
                    {
                        "Symbol": strategy.symbol,
                        "Strategy": strategy.strategy,
                        "Timeframe": strategy.timeframe,
                        "Sortino_Ratio": strategy.sortino_ratio or 0.0,
                        "Sharpe_Ratio": strategy.sharpe_ratio or 0.0,
                        "Calmar_Ratio": strategy.calmar_ratio or 0.0,
                        "Total_Return": strategy.total_return or 0.0,
                        "Max_Drawdown": strategy.max_drawdown or 0.0,
                        "Updated_At": strategy.updated_at.strftime("%Y-%m-%d %H:%M:%S")
                        if strategy.updated_at
                        else "",
                        "Quarter": quarter,
                        "Year": year,
                    }
                )

            df = pd.DataFrame(data)

            # Create output directory following standard naming convention
            csv_output_dir = self.output_dir / year / quarter
            csv_output_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename following naming convention
            # Prefer human-readable collection name from config when available
            display_name = portfolio_name or "All_Collections"
            if portfolio_path:
                try:
                    import json

                    with Path(portfolio_path).open() as f:
                        portfolio_config = json.load(f)
                        portfolio_key = list(portfolio_config.keys())[0]
                        display_name = (
                            portfolio_config[portfolio_key].get("name") or display_name
                        )
                except Exception:
                    pass

            # Sanitize and build unified base filename: <Name>_Collection_<Year>_<Quarter>_<Interval>
            sanitized = re.sub(r"\W+", "_", str(display_name)).strip("_")
            safe_interval = (interval or "multi").replace("/", "-")
            if output_filename:
                base_filename = output_filename.replace(".csv", "")
            else:
                base_filename = (
                    f"{sanitized}_Collection_{year}_{quarter}_{safe_interval}"
                )

            if export_format == "best-strategies":
                filename = f"{base_filename}.csv"
                # Keep only one row per symbol with highest Sortino ratio
                df = (
                    df.sort_values("Sortino_Ratio", ascending=False)
                    .groupby("Symbol")
                    .first()
                    .reset_index()
                )
                # Select and rename columns for best strategies format
                df = df[
                    ["Symbol", "Strategy", "Timeframe", "Sortino_Ratio", "Total_Return"]
                ].rename(
                    columns={
                        "Symbol": "Asset",
                        "Strategy": "Best_Strategy",
                        "Timeframe": "Best_Timeframe",
                        "Sortino_Ratio": "Sortino_Ratio",
                        "Total_Return": "Total_Return_Pct",
                    }
                )
            elif export_format == "quarterly":
                filename = f"{base_filename}.csv"
                # Create summary statistics
                summary_data = []
                for symbol in df["Symbol"].unique():
                    symbol_data = df[df["Symbol"] == symbol]
                    best_strategy = symbol_data.loc[
                        symbol_data["Sortino_Ratio"].idxmax()
                    ]
                    summary_data.append(
                        {
                            "Asset": symbol,
                            "Best_Strategy": best_strategy["Strategy"],
                            "Best_Timeframe": best_strategy["Timeframe"],
                            "Best_Sortino": best_strategy["Sortino_Ratio"],
                            "Strategies_Tested": len(symbol_data),
                            "Avg_Return": symbol_data["Total_Return"].mean(),
                            "Best_Return": symbol_data["Total_Return"].max(),
                        }
                    )
                df = pd.DataFrame(summary_data)
            else:  # full
                filename = f"{base_filename}.csv"
                # Keep all data with proper column names
                df = df.rename(
                    columns={
                        "Symbol": "Asset",
                        "Strategy": "Strategy_Name",
                        "Timeframe": "Time_Resolution",
                    }
                )

            output_file = csv_output_dir / filename

            # Export to CSV
            df.to_csv(output_file, index=False)
            output_files.append(str(output_file))

            self.logger.info("Exported %s records to %s", len(df), output_file)

            return output_files

        except Exception as e:
            # Attempt unified_models fallback even if primary DB session creation failed early
            try:
                from src.database import unified_models

                # Load portfolio symbols for filtering if portfolio_path is provided
                portfolio_symbols = None
                if "portfolio_path" in locals() and portfolio_path:
                    try:
                        import json

                        with Path(portfolio_path).open() as f:
                            portfolio_config = json.load(f)
                            portfolio_key = list(portfolio_config.keys())[0]
                            portfolio_symbols = portfolio_config[portfolio_key].get(
                                "symbols", []
                            )
                            portfolio_name = portfolio_key
                    except Exception:
                        pass

                sess2 = unified_models.Session()
                try:
                    uq = sess2.query(unified_models.BestStrategy)
                    if portfolio_symbols:
                        uq = uq.filter(
                            unified_models.BestStrategy.symbol.in_(portfolio_symbols)
                        )
                    if interval:
                        try:
                            uq = uq.filter(
                                unified_models.BestStrategy.timeframe == interval
                            )
                        except Exception:
                            pass
                    best_strategies = uq.all()
                finally:
                    try:
                        sess2.close()
                    except Exception:
                        pass

                if not best_strategies:
                    self.logger.error(
                        "Failed to export from database and unified_models had no rows: %s",
                        e,
                    )
                    return output_files

                # Build DataFrame from unified_models rows (same as above)
                data = []
                for strategy in best_strategies:
                    data.append(
                        {
                            "Symbol": strategy.symbol,
                            "Strategy": strategy.strategy,
                            "Timeframe": strategy.timeframe,
                            "Sortino_Ratio": strategy.sortino_ratio or 0.0,
                            "Sharpe_Ratio": strategy.sharpe_ratio or 0.0,
                            "Calmar_Ratio": strategy.calmar_ratio or 0.0,
                            "Total_Return": strategy.total_return or 0.0,
                            "Max_Drawdown": strategy.max_drawdown or 0.0,
                            "Updated_At": strategy.updated_at.strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                            if strategy.updated_at
                            else "",
                            "Quarter": quarter,
                            "Year": year,
                        }
                    )

                df = pd.DataFrame(data)
                csv_output_dir = self.output_dir / year / quarter
                csv_output_dir.mkdir(parents=True, exist_ok=True)

                display_name = portfolio_name or "All_Collections"
                if "portfolio_path" in locals() and portfolio_path:
                    try:
                        import json

                        with Path(portfolio_path).open() as f:
                            portfolio_config = json.load(f)
                            portfolio_key = list(portfolio_config.keys())[0]
                            display_name = (
                                portfolio_config[portfolio_key].get("name")
                                or display_name
                            )
                    except Exception:
                        pass

                sanitized = re.sub(r"\W+", "_", str(display_name)).strip("_")
                safe_interval = (interval or "multi").replace("/", "-")
                base_filename = (
                    f"{sanitized}_Collection_{year}_{quarter}_{safe_interval}"
                )

                if export_format == "best-strategies":
                    filename = f"{base_filename}.csv"
                    df = (
                        df.sort_values("Sortino_Ratio", ascending=False)
                        .groupby("Symbol")
                        .first()
                        .reset_index()
                    )
                    df = df[
                        [
                            "Symbol",
                            "Strategy",
                            "Timeframe",
                            "Sortino_Ratio",
                            "Total_Return",
                        ]
                    ].rename(
                        columns={
                            "Symbol": "Asset",
                            "Strategy": "Best_Strategy",
                            "Timeframe": "Best_Timeframe",
                            "Sortino_Ratio": "Sortino_Ratio",
                            "Total_Return": "Total_Return_Pct",
                        }
                    )
                elif export_format == "quarterly":
                    filename = f"{base_filename}.csv"
                    summary_data = []
                    for symbol in df["Symbol"].unique():
                        symbol_data = df[df["Symbol"] == symbol]
                        best_strategy = symbol_data.loc[
                            symbol_data["Sortino_Ratio"].idxmax()
                        ]
                        summary_data.append(
                            {
                                "Asset": symbol,
                                "Best_Strategy": best_strategy["Strategy"],
                                "Best_Timeframe": best_strategy["Timeframe"],
                                "Best_Sortino": best_strategy["Sortino_Ratio"],
                                "Strategies_Tested": len(symbol_data),
                                "Avg_Return": symbol_data["Total_Return"].mean(),
                                "Best_Return": symbol_data["Total_Return"].max(),
                            }
                        )
                    df = pd.DataFrame(summary_data)
                else:
                    filename = f"{base_filename}.csv"
                    df = df.rename(
                        columns={
                            "Symbol": "Asset",
                            "Strategy": "Strategy_Name",
                            "Timeframe": "Time_Resolution",
                        }
                    )

                output_file = csv_output_dir / filename
                df.to_csv(output_file, index=False)
                output_files.append(str(output_file))
                self.logger.info(
                    "Exported %s records to %s (unified_models)", len(df), output_file
                )
                return output_files
            except Exception as e2:
                self.logger.error(
                    "Failed CSV export fallback: %s (original: %s)", e2, e
                )
                return output_files
        finally:
            if "db_session" in locals():
                db_session.close()

    def export_from_quarterly_reports(
        self,
        quarter: str,
        year: str,
        output_filename: str | None = None,
        export_format: str = "full",
        collection_name: str | None = None,
        interval: str | None = None,
    ) -> list[str]:
        """
        Extract data from existing quarterly reports and export to CSV.
        Creates separate CSV files for each HTML report (e.g., Crypto_Portfolio_Q3_2025.csv).

        Args:
            quarter: Quarter (Q1, Q2, Q3, Q4)
            year: Year (YYYY)
            output_filename: Custom filename, auto-generated if None (used for single file export)
            export_format: 'full' or 'best-strategies'

        Returns:
            List of paths to exported CSV files
        """
        # Check if quarterly reports exist
        quarterly_reports_dir = self.reports_dir / year / quarter
        if not quarterly_reports_dir.exists():
            self.logger.warning("No quarterly reports found for %s %s", quarter, year)
            return []

        # Find HTML report files
        html_files = list(quarterly_reports_dir.glob("*.html"))
        if not html_files:
            self.logger.warning("No HTML reports found in %s", quarterly_reports_dir)
            return []

        self.logger.info(
            "Found %d HTML reports for %s %s", len(html_files), quarter, year
        )

        # Create quarterly directory structure under exports/csv
        quarterly_dir = self.output_dir / year / quarter
        quarterly_dir.mkdir(parents=True, exist_ok=True)

        exported_files = []

        # Process each HTML report separately
        for html_file in html_files:
            # Extract data from this specific report
            extracted_data = self._extract_data_from_html_report(html_file)

            if not extracted_data:
                self.logger.warning("No data extracted from %s", html_file.name)
                continue

            # Convert to DataFrame
            df = pd.DataFrame(extracted_data)

            # Build unified filename
            name_for_file = collection_name or html_file.stem
            sanitized = re.sub(r"\W+", "_", str(name_for_file)).strip("_")
            safe_interval = (interval or "multi").replace("/", "-")
            csv_filename = (
                output_filename
                if output_filename and len(html_files) == 1
                else f"{sanitized}_Collection_{year}_{quarter}_{safe_interval}.csv"
            )

            # Process based on format
            if export_format == "best-strategies":
                # Group by symbol and keep best performing strategy
                if "Symbol" in df.columns and "Sortino_Ratio" in df.columns:
                    df = (
                        df.sort_values("Sortino_Ratio", ascending=False)
                        .groupby("Symbol")
                        .first()
                        .reset_index()
                    )
                    df = df[["Symbol", "Strategy", "Timeframe"]].rename(
                        columns={
                            "Symbol": "Asset",
                            "Strategy": "Best Strategy",
                            "Timeframe": "Resolution",
                        }
                    )

            # Add quarterly metadata
            df["Quarter"] = quarter
            df["Year"] = year
            df["Export_Date"] = pd.Timestamp.now().strftime("%Y-%m-%d")

            # Sort by performance
            if "Sortino_Ratio" in df.columns:
                df = df.sort_values("Sortino_Ratio", ascending=False)
            elif "Total_Return_Pct" in df.columns:
                df = df.sort_values("Total_Return_Pct", ascending=False)

            # Export to quarterly directory
            output_path = quarterly_dir / csv_filename
            df.to_csv(output_path, index=False)

            exported_files.append(str(output_path))

            self.logger.info(
                "Exported %s data from %s to %s (%d rows)",
                export_format,
                html_file.name,
                output_path,
                len(df),
            )

        self.logger.info(
            "Exported %d CSV files from quarterly reports for %s %s",
            len(exported_files),
            quarter,
            year,
        )

        return exported_files

    def get_available_columns(self) -> list[str]:
        """Get list of all available columns for export."""
        return [
            "Symbol",
            "Strategy",
            "Timeframe",
            "Total_Return_Pct",
            "Sortino_Ratio",
            "Sharpe_Ratio",
            "Calmar_Ratio",
            "Max_Drawdown_Pct",
            "Win_Rate_Pct",
            "Profit_Factor",
            "Number_of_Trades",
            "Volatility_Pct",
            "Downside_Deviation",
            "Average_Win",
            "Average_Loss",
            "Longest_Win_Streak",
            "Longest_Loss_Streak",
            "Data_Points",
            "Backtest_Duration_Seconds",
        ]

    def _extract_data_from_html_report(self, html_file: Path) -> list[dict[str, Any]]:
        """Extract performance data from HTML report files."""
        try:
            with html_file.open("r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")

            extracted_data = []

            # Look for tables with performance data
            tables = soup.find_all("table")

            for table in tables:
                # Check if this is a performance metrics table
                headers = table.find("tr")
                if not headers:
                    continue

                header_cells = [
                    th.get_text(strip=True) for th in headers.find_all(["th", "td"])
                ]

                # Look for tables that contain symbol/strategy information
                if any(
                    keyword in " ".join(header_cells).lower()
                    for keyword in ["symbol", "strategy", "asset", "sortino", "sharpe"]
                ):
                    rows = table.find_all("tr")[1:]  # Skip header row

                    for row in rows:
                        cells = [
                            td.get_text(strip=True) for td in row.find_all(["td", "th"])
                        ]
                        if len(cells) < 2:
                            continue

                        # Try to extract data based on common patterns
                        row_data = self._parse_table_row(header_cells, cells)
                        if row_data:
                            extracted_data.append(row_data)

            # Also look for metric cards or divs with performance data
            metric_cards = soup.find_all(
                "div", class_=re.compile(r".*metric.*|.*card.*|.*performance.*", re.I)
            )
            for card in metric_cards:
                card_data = self._parse_metric_card(card)
                if card_data:
                    extracted_data.append(card_data)

            # Tailwind report structure fallback: section[id^='asset-'] with h2 and spans
            try:
                sections = soup.select("section[id^='asset-']")
                for sec in sections:
                    h2 = sec.find("h2")
                    symbol = h2.get_text(strip=True) if h2 else None
                    best_strategy = None
                    timeframe = None
                    for sp in sec.find_all("span"):
                        txt = sp.get_text(strip=True)
                        if txt.startswith("Best:") and best_strategy is None:
                            best_strategy = txt.replace("Best:", "").strip()
                        if "⏰" in txt and timeframe is None:
                            timeframe = txt.replace("⏰", "").strip()
                    if symbol and best_strategy:
                        extracted_data.append(
                            {
                                "Symbol": symbol,
                                "Strategy": best_strategy,
                                "Timeframe": timeframe or "1d",
                            }
                        )
            except Exception:
                pass

            self.logger.info(
                "Extracted %d data points from %s", len(extracted_data), html_file.name
            )
            return extracted_data

        except Exception as e:
            self.logger.error("Failed to parse HTML file %s: %s", html_file, e)
            return []

    def _parse_table_row(
        self, headers: list[str], cells: list[str]
    ) -> dict[str, Any] | None:
        """Parse a table row and extract relevant metrics."""
        if len(headers) != len(cells):
            return None

        row_data = {}

        # Map common header patterns to our standard columns
        header_mapping = {
            "symbol": "Symbol",
            "asset": "Symbol",
            "strategy": "Strategy",
            "timeframe": "Timeframe",
            "resolution": "Timeframe",
            "total_return": "Total_Return_Pct",
            "return": "Total_Return_Pct",
            "sortino": "Sortino_Ratio",
            "sharpe": "Sharpe_Ratio",
            "calmar": "Calmar_Ratio",
            "drawdown": "Max_Drawdown_Pct",
            "win_rate": "Win_Rate_Pct",
            "profit_factor": "Profit_Factor",
            "trades": "Number_of_Trades",
            "volatility": "Volatility_Pct",
        }

        for i, header in enumerate(headers):
            if i >= len(cells):
                break

            header_lower = (
                header.lower()
                .replace(" ", "_")
                .replace("%", "")
                .replace("(", "")
                .replace(")", "")
            )

            # Find matching column name
            mapped_column = None
            for pattern, column in header_mapping.items():
                if pattern in header_lower:
                    mapped_column = column
                    break

            if mapped_column and cells[i]:
                try:
                    # Try to convert numeric values
                    if mapped_column in [
                        "Total_Return_Pct",
                        "Sortino_Ratio",
                        "Sharpe_Ratio",
                        "Calmar_Ratio",
                        "Max_Drawdown_Pct",
                        "Win_Rate_Pct",
                        "Profit_Factor",
                        "Volatility_Pct",
                    ]:
                        # Remove % signs and other formatting
                        clean_value = re.sub(r"[%$,\s]", "", cells[i])
                        if clean_value and clean_value != "-":
                            row_data[mapped_column] = float(clean_value)
                    elif mapped_column == "Number_of_Trades":
                        clean_value = re.sub(r"[,\s]", "", cells[i])
                        if clean_value and clean_value.isdigit():
                            row_data[mapped_column] = int(clean_value)
                    else:
                        row_data[mapped_column] = cells[i]
                except (ValueError, TypeError):
                    row_data[mapped_column] = cells[i]

        # Only return if we have at least symbol or strategy
        if "Symbol" in row_data or "Strategy" in row_data:
            # Set defaults
            if "Timeframe" not in row_data:
                row_data["Timeframe"] = "1d"
            return row_data

        return None

    def _parse_metric_card(self, card) -> dict[str, Any] | None:
        """Parse metric cards for performance data."""
        # This would need to be customized based on the actual HTML structure
        # of the reports generated by the system
        text = card.get_text(strip=True)

        # Look for patterns like "BTCUSDT: 45.2%" or "Strategy: BuyAndHold"
        symbol_match = re.search(r"([A-Z0-9]+USDT?):?\s*([-+]?\d+\.?\d*%?)", text)
        strategy_match = re.search(r"Strategy:?\s*([A-Za-z\s]+)", text)

        if symbol_match or strategy_match:
            card_data = {}
            if symbol_match:
                card_data["Symbol"] = symbol_match.group(1)
                if len(symbol_match.groups()) > 1:
                    try:
                        value = float(symbol_match.group(2).replace("%", ""))
                        card_data["Total_Return_Pct"] = value
                    except ValueError:
                        pass

            if strategy_match:
                card_data["Strategy"] = strategy_match.group(1).strip()

            card_data["Timeframe"] = "1d"  # Default
            return card_data

        return None

    def _export_from_database(
        self,
        quarter: str,
        year: str,
        export_format: str = "full",
        interval: str | None = None,
    ) -> list[str]:
        """
        Export data directly from database when HTML reports have no data.

        Args:
            quarter: Quarter (Q1, Q2, Q3, Q4)
            year: Year (YYYY)
            export_format: Export format ('full', 'best-strategies', 'quarterly')

        Returns:
            List of generated CSV file paths
        """
        output_files = []

        try:
            db_session = get_db_session()

            # Query all best strategies from database
            q = db_session.query(BestStrategy)
            if "interval" in locals() and interval:
                try:
                    q = q.filter(BestStrategy.timeframe == interval)
                except Exception:
                    pass
            best_strategies = q.all()

            if not best_strategies:
                self.logger.warning("No strategies found in database")
                return output_files

            self.logger.info("Found %s strategies in database", len(best_strategies))

            # Convert to DataFrame
            data = []
            for strategy in best_strategies:
                data.append(
                    {
                        "Symbol": strategy.symbol,
                        "Strategy": strategy.strategy,
                        "Timeframe": strategy.timeframe,
                        "Sortino_Ratio": strategy.sortino_ratio,
                        "Sharpe_Ratio": strategy.sharpe_ratio,
                        "Calmar_Ratio": strategy.calmar_ratio,
                        "Total_Return": strategy.total_return,
                        "Max_Drawdown": strategy.max_drawdown,
                        "Updated_At": strategy.updated_at.strftime("%Y-%m-%d %H:%M:%S")
                        if strategy.updated_at
                        else "",
                    }
                )

            df = pd.DataFrame(data)

            # Create output directory
            csv_output_dir = self.output_dir / year / quarter
            csv_output_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename (fallback method) using unified convention
            safe_interval = (interval or "multi").replace("/", "-")
            filename = (
                f"All_Collections_Collection_{year}_{quarter}_{safe_interval}.csv"
            )
            if export_format == "best-strategies":
                # Keep only one row per symbol with highest Sortino ratio
                df = (
                    df.sort_values("Sortino_Ratio", ascending=False)
                    .groupby("Symbol")
                    .first()
                    .reset_index()
                )
            elif export_format == "quarterly":
                pass
            else:  # full
                pass

            output_file = csv_output_dir / filename

            # Export to CSV
            df.to_csv(output_file, index=False)
            output_files.append(str(output_file))

            self.logger.info("Exported %s records to %s", len(df), output_file)

            return output_files

        except Exception as e:
            self.logger.error("Failed to export from database: %s", e)
            return output_files
        finally:
            if "db_session" in locals():
                db_session.close()
