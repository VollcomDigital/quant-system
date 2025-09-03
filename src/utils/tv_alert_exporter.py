#!/usr/bin/env python3
"""
TradingView Alert Exporter

Extracts asset strategies and timeframes from HTML reports and generates
TradingView alert messages with appropriate placeholders.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from bs4 import BeautifulSoup

# DB models
try:
    from src.database.db_connection import (
        get_db_session,  # type: ignore[import-not-found]
    )
    from src.database.models import BestStrategy  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - guarded imports
    get_db_session = None  # type: ignore[assignment]
    BestStrategy = None  # type: ignore[assignment]


class TradingViewAlertExporter:
    def __init__(self, reports_dir: str = "exports/reports"):
        self.reports_dir = Path(reports_dir)

        # Check if old location exists and new location is empty for migration
        old_dir = Path("reports_output")
        if old_dir.exists() and not self.reports_dir.exists():
            print(f"⚠️  Found reports in old location: {old_dir}")
            print(
                f"💡 Consider running report organizer to migrate to: {self.reports_dir}"
            )
            self.reports_dir = old_dir

    def get_quarter_from_date(self, date: datetime) -> tuple[int, int]:
        """Get quarter and year from date."""
        quarter = (date.month - 1) // 3 + 1
        return date.year, quarter

    def organize_output_path(self, base_dir: str) -> Path:
        """Create organized output path based on current quarter/year."""
        now = datetime.now(timezone.utc)
        year, quarter = self.get_quarter_from_date(now)

        output_dir = Path(base_dir) / str(year) / f"Q{quarter}"
        output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir

    def _build_filename(
        self, collection_name: str, year: int, quarter: int, interval: str | None
    ) -> str:
        """Builds <Collectionname>_Collection_<Year>_<Quarter>_<Interval>.md"""
        sanitized = (
            collection_name.replace(" ", "_").replace("/", "_").strip("_")
            or "All_Collections"
        )
        interval_part = (interval or "multi").replace("/", "-")
        return f"{sanitized}_Collection_{year}_Q{quarter}_{interval_part}.md"

    def extract_asset_data(self, html_content: str) -> List[Dict]:
        """Extract asset information from HTML report"""
        soup = BeautifulSoup(html_content, "html.parser")
        assets: List[Dict] = []

        # New Tailwind report structure (DetailedPortfolioReporter): sections with id="asset-<SYMBOL>"
        section_nodes = soup.select("section[id^='asset-']")
        for sec in section_nodes:
            h2 = sec.find("h2")
            symbol = h2.get_text(strip=True) if h2 else None
            best_strategy = None
            timeframe = None
            # The header line contains two spans: "Best: <name>" and "⏰ <interval>"
            tag_spans = sec.find_all("span")
            for sp in tag_spans:
                txt = sp.get_text(strip=True)
                if txt.startswith("Best:") and best_strategy is None:
                    best_strategy = txt.replace("Best:", "").strip()
                if "⏰" in txt and timeframe is None:
                    timeframe = txt.replace("⏰", "").strip()
            if symbol and best_strategy and timeframe:
                assets.append(
                    {
                        "symbol": symbol,
                        "strategy": best_strategy,
                        "timeframe": timeframe,
                        "metrics": {},
                    }
                )

        if assets:
            return assets

        # Fallback legacy structure support (older HTML reports)
        legacy_assets: List[Dict] = []
        asset_sections = soup.find_all("div", class_="asset-section")
        for section in asset_sections:
            asset_title = section.find("h2", class_="asset-title")
            if not asset_title:
                continue
            symbol = asset_title.text.strip()
            strategy_badges = section.find_all("span", class_="strategy-badge")
            best_strategy = None
            timeframe = None
            for badge in strategy_badges:
                text = badge.text.strip()
                if text.startswith("Best:"):
                    best_strategy = text.replace("Best:", "").strip()
                elif "⏰" in text:
                    timeframe = text.replace("⏰", "").strip()
            metrics = {}
            metric_cards = section.find_all("div", class_="metric-card")
            for card in metric_cards:
                label_elem = card.find("div", class_="metric-label")
                value_elem = card.find("div", class_="metric-value")
                if label_elem and value_elem:
                    label = label_elem.text.strip()
                    value = value_elem.text.strip()
                    metrics[label] = value
            if symbol and best_strategy and timeframe:
                legacy_assets.append(
                    {
                        "symbol": symbol,
                        "strategy": best_strategy,
                        "timeframe": timeframe,
                        "metrics": metrics,
                    }
                )
        return legacy_assets

    def generate_tradingview_alert(self, asset_data: Dict) -> str:
        """Generate TradingView alert message for asset"""
        symbol = asset_data["symbol"]
        strategy = asset_data["strategy"]
        timeframe = asset_data["timeframe"]
        metrics = asset_data.get("metrics", {})

        # Get key metrics for context
        sharpe_ratio = metrics.get("Sharpe Ratio", "N/A")
        sortino_ratio = metrics.get("Sortino Ratio", "N/A")
        calmar_ratio = metrics.get("Calmar Ratio", "N/A")

        alert_message = f"""🚨 QUANT SIGNAL: {symbol} 📊
Strategy: {strategy}
Timeframe: {timeframe}
📈 Sharpe: {sharpe_ratio}
📊 Sortino: {sortino_ratio}
⚖️ Calmar: {calmar_ratio}

Price: {{{{close}}}}
Time: {{{{timenow}}}}
Action: {{{{strategy.order.action}}}}
Qty: {{{{strategy.order.contracts}}}}

#QuantTrading #{symbol} #{strategy.replace(" ", "")}"""

        return alert_message

    def process_html_file(self, file_path: Path) -> List[Dict]:
        """Process single HTML file and extract asset data"""
        try:
            with file_path.open(encoding="utf-8") as f:
                content = f.read()

            assets = self.extract_asset_data(content)

            # Add file metadata
            for asset in assets:
                asset["source_file"] = str(file_path)
                asset["report_name"] = file_path.stem

            return assets
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []

    def find_html_reports(self) -> List[Path]:
        """Find all HTML report files"""
        html_files = []
        for root, dirs, files in os.walk(self.reports_dir):
            for file in files:
                if file.endswith(".html"):
                    html_files.append(Path(root) / file)
        return html_files

    def export_alerts(
        self,
        output_file: Optional[str] = None,
        collection_filter: Optional[str] = None,
        interval: Optional[str] = None,
        symbols: Optional[List[str]] = None,
    ) -> Dict:
        """Export TradingView alerts using database BestStrategy data.

        - Filters by provided symbols when available (preferred).
        - If symbols is None, uses all BestStrategy rows.
        - Writes markdown under exports/tv_alerts/<Year>/Q<Quarter>/ with unified name.
        """
        all_alerts: Dict[str, List[Dict]] = {}

        # Query DB for best strategies
        rows = []
        sess = None
        try:
            if get_db_session is None or BestStrategy is None:
                raise RuntimeError("Database session/models unavailable for TV export")
            sess = get_db_session()
            q = sess.query(BestStrategy)
            if symbols:
                q = q.filter(BestStrategy.symbol.in_(symbols))
            # Optionally, prefer the provided interval if filtering is desired
            if interval:
                q = q.filter(BestStrategy.timeframe == interval)
            rows = q.all()
            # Fallback to unified_models if no rows found (similar to csv exporter)
            if not rows:
                try:
                    from src.database import (
                        unified_models as um,  # type: ignore[import-not-found]
                    )

                    usess = um.Session()
                    try:
                        uq = usess.query(um.BestStrategy)
                        if symbols:
                            uq = uq.filter(um.BestStrategy.symbol.in_(symbols))
                        if interval:
                            uq = uq.filter(um.BestStrategy.timeframe == interval)
                        rows = uq.all()
                    finally:
                        usess.close()
                except Exception:
                    rows = []
        finally:
            if sess is not None:
                try:
                    sess.close()
                except Exception:
                    pass

        # If interval specified but produced no rows, relax interval filter
        if interval and not rows:
            try:
                sess = get_db_session()
                q = sess.query(BestStrategy)
                if symbols:
                    q = q.filter(BestStrategy.symbol.in_(symbols))
                rows = q.all()
            except Exception:
                rows = rows
            finally:
                if sess is not None:
                    try:
                        sess.close()
                    except Exception:
                        pass

        # Build alerts either from DB rows or (fallback) parse existing HTML reports
        by_symbol: Dict[str, Dict] = {}

        if rows:
            # Primary: DB-backed BestStrategy rows
            for r in rows:
                sym = getattr(r, "symbol", None)
                if not sym:
                    continue
                entry = by_symbol.get(sym)
                sortino = float(getattr(r, "sortino_ratio", 0.0) or 0.0)
                if entry is None or sortino > entry.get("sortino_ratio", -1e9):
                    by_symbol[sym] = {
                        "symbol": sym,
                        "strategy": getattr(r, "strategy", ""),
                        "timeframe": getattr(r, "timeframe", interval or "1d"),
                        "metrics": {
                            "Sharpe Ratio": f"{float(getattr(r, 'sharpe_ratio', 0.0) or 0.0):.3f}",
                            "Sortino Ratio": f"{sortino:.3f}",
                            "Calmar Ratio": f"{float(getattr(r, 'calmar_ratio', 0.0) or 0.0):.3f}",
                        },
                    }
        else:
            # Fallback: parse HTML reports for assets and their best strategy/timeframe
            try:
                html_files = self.find_html_reports()
                for fp in html_files:
                    # Optionally filter by collection name in filename if provided
                    if collection_filter:
                        # normalize name part (spaces/parentheses -> underscores)
                        cname = collection_filter.replace(" ", "_").replace("/", "_")
                        if cname not in fp.name:
                            continue
                    assets = self.process_html_file(fp)
                    for asset in assets:
                        if symbols and asset.get("symbol") not in set(symbols):
                            continue
                        if interval and asset.get("timeframe") != interval:
                            continue
                        sym = asset.get("symbol")
                        strat = asset.get("strategy")
                        tf = asset.get("timeframe") or (interval or "1d")
                        if not sym or not strat:
                            continue
                        # Keep first (or allow override if needed by future metrics)
                        by_symbol.setdefault(
                            sym,
                            {
                                "symbol": sym,
                                "strategy": strat,
                                "timeframe": tf,
                                "metrics": {},
                            },
                        )
            except Exception:
                # Silent fallback; will result in header-only file as before
                pass

        for sym, asset in by_symbol.items():
            alert = self.generate_tradingview_alert(asset)
            if sym not in all_alerts:
                all_alerts[sym] = []
            all_alerts[sym].append({"alert_message": alert, "asset_data": asset})

        # Write to file if requested
        if output_file is not None or collection_filter is not None:
            organized_dir = self.organize_output_path("exports/tv_alerts")
            now = datetime.now(timezone.utc)
            year, q = self.get_quarter_from_date(now)
            collection_name = collection_filter or "All_Collections"
            if output_file and output_file not in ("tradingview_alerts.md",):
                filename = (
                    output_file if output_file.endswith(".md") else f"{output_file}.md"
                )
            else:
                filename = self._build_filename(collection_name, year, q, interval)
            output_path = organized_dir / filename

            with output_path.open("w", encoding="utf-8") as f:
                f.write("# TradingView Alert Messages\n\n")
                for symbol, alerts in all_alerts.items():
                    f.write(f"## {symbol}\n\n")
                    for i, alert_data in enumerate(alerts):
                        asset = alert_data["asset_data"]
                        f.write(
                            f"### Alert {i + 1} - {asset['strategy']} ({asset['timeframe']})\n"
                        )
                        f.write("```\n")
                        f.write(alert_data["alert_message"])
                        f.write("\n```\n\n")
                        f.write("---\n\n")

        return all_alerts


def main():
    parser = argparse.ArgumentParser(
        description="Export TradingView alerts from HTML reports"
    )
    parser.add_argument(
        "--reports-dir",
        default="exports/reports",
        help="Directory containing HTML reports",
    )
    parser.add_argument(
        "--output",
        default="tradingview_alerts.md",
        help="Output file for alerts (auto-organized by quarter/year if just filename)",
    )
    parser.add_argument("--symbol", help="Export alerts for specific symbol only")
    parser.add_argument(
        "--collection",
        help="Export alerts for specific collection/portfolio only (e.g., 'Commodities', 'Bonds')",
    )

    args = parser.parse_args()

    exporter = TradingViewAlertExporter(args.reports_dir)
    alerts = exporter.export_alerts(args.output, collection_filter=args.collection)

    print("\n📊 Export Summary:")
    print(f"Found {len(alerts)} assets with alerts")

    if args.symbol:
        if args.symbol in alerts:
            print(f"\n🎯 Alerts for {args.symbol}:")
            for alert_data in alerts[args.symbol]:
                print("\n" + "=" * 60)
                print(alert_data["alert_message"])
        else:
            print(f"❌ No alerts found for {args.symbol}")
    else:
        for symbol, symbol_alerts in alerts.items():
            print(f"  {symbol}: {len(symbol_alerts)} alert(s)")

    if args.output:
        print(f"\n✅ Alerts exported to: {args.output}")


if __name__ == "__main__":
    main()
