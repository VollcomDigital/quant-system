"""
AI Investment Recommendations - AI-powered analysis of backtest results
to recommend optimal asset allocation and investment decisions.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src.ai.llm_client import LLMClient
from src.ai.models import AssetRecommendation, PortfolioRecommendation
from src.database.models import AIRecommendation, BacktestResult, BestStrategy
from src.database.models import AssetRecommendation as DbAssetRecommendation
from src.reporting.ai_report_generator import AIReportGenerator


class AIInvestmentRecommendations:
    """
    AI-powered investment recommendation system that analyzes backtest results
    to provide optimal asset allocation and investment decisions.
    """

    def __init__(self, db_session: Session = None):
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)
        self.llm_client = LLMClient()

        # Risk tolerance levels
        self.risk_levels = {
            "conservative": {"max_drawdown": 0.10, "min_sortino": 1.0},
            "moderate": {"max_drawdown": 0.20, "min_sortino": 0.75},
            "aggressive": {"max_drawdown": 0.35, "min_sortino": 0.5},
        }

        # Scoring weights
        self.scoring_weights = {
            "sortino_ratio": 0.35,
            "calmar_ratio": 0.25,
            "profit_factor": 0.20,
            "max_drawdown": 0.10,
            "win_rate": 0.10,
        }

    @staticmethod
    def _ensure_python_type(val):
        """Convert any numpy type to Python native type."""
        if val is None:
            return None

        if isinstance(val, (np.floating, np.integer, np.bool_)):
            return val.item()  # Convert to Python native type
        if isinstance(val, np.ndarray):
            if val.size == 1:
                return val.item()
            return val.tolist()
        if hasattr(val, "item"):  # Other numpy scalars
            return val.item()
        if isinstance(val, (list, tuple)):
            return [
                AIInvestmentRecommendations._ensure_python_type(item) for item in val
            ]
        if isinstance(val, dict):
            return {
                k: AIInvestmentRecommendations._ensure_python_type(v)
                for k, v in val.items()
            }
        return val

    def generate_recommendations(
        self,
        risk_tolerance: str = "moderate",
        min_confidence: float = 0.7,
        max_assets: int = 10,
        quarter: Optional[str] = None,
        timeframe: str = "1h",
        portfolio_name: Optional[str] = None,
        portfolio_path: Optional[str] = None,
    ) -> PortfolioRecommendation:
        """
        Generate AI-powered investment recommendations based on backtest results.

        Args:
            risk_tolerance: Risk level (conservative, moderate, aggressive)
            min_confidence: Minimum confidence score for recommendations
            max_assets: Maximum number of assets to recommend
            quarter: Specific quarter to analyze (e.g., "Q3_2025")

        Returns:
            PortfolioRecommendation with AI analysis
        """
        self.logger.info(
            "Generating AI recommendations for %s risk profile", risk_tolerance
        )

        # Load portfolio config if provided to get collection name and symbols
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
                        "Using %s symbols from %s collection",
                        len(portfolio_symbols),
                        portfolio_name,
                    )
            except Exception as e:
                self.logger.warning("Could not load portfolio config: %s", e)

        # Load backtest results with optional symbol filtering
        backtest_data = self._load_backtest_results(quarter, portfolio_symbols)
        if not backtest_data:
            raise ValueError("No backtest results found")

        # Performance-based scoring
        scored_assets = self._calculate_performance_scores(backtest_data)

        # Risk-adjusted filtering
        filtered_assets = self._apply_risk_filters(scored_assets, risk_tolerance)

        # Portfolio correlation analysis
        correlation_data = self._analyze_correlations(filtered_assets)

        # Strategy-asset matching
        optimized_assets = self._optimize_strategy_asset_matching(filtered_assets)

        # Generate allocation suggestions
        allocations = self._suggest_allocations(
            optimized_assets, risk_tolerance, max_assets
        )

        # Red flag detection
        flagged_assets = self._detect_red_flags(allocations)

        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(
            flagged_assets, backtest_data
        )

        # Create asset recommendations for ALL assets (no filtering by confidence)
        recommendations = []
        for asset_data in confidence_scores:
            # Determine investment recommendation
            base_reasoning = asset_data.get(
                "reasoning",
                f"Sortino: {asset_data['sortino_ratio']:.2f}, Max DD: {asset_data['max_drawdown']:.1%}",
            )

            if asset_data["confidence"] >= 0.6 and asset_data["score"] >= 0.3:
                invest_decision = "INVEST_WITH_RISK_MANAGEMENT"
                reasoning = f"Moderate performance metrics suggest cautious investment. {base_reasoning}"
            elif asset_data["confidence"] >= 0.4 and asset_data["score"] >= 0.2:
                invest_decision = "CONSIDER_WITH_HIGH_CAUTION"
                reasoning = f"Below-average performance requires extreme caution. {base_reasoning}"
            else:
                invest_decision = "DO_NOT_INVEST"
                reasoning = (
                    f"Poor performance metrics indicate high risk. {base_reasoning}"
                )

            # Calculate trading parameters
            trading_params = self._calculate_trading_parameters(asset_data, timeframe)

            recommendation = AssetRecommendation(
                symbol=asset_data["symbol"],
                strategy=asset_data["strategy"],
                score=self._ensure_python_type(asset_data["score"]),
                confidence=self._ensure_python_type(asset_data["confidence"]),
                allocation_percentage=self._ensure_python_type(
                    asset_data["allocation"]
                ),  # Always show suggested allocation
                risk_level=self._classify_risk_level(asset_data),
                reasoning=reasoning,
                red_flags=[*asset_data.get("red_flags", []), invest_decision],
                sortino_ratio=asset_data["sortino_ratio"],
                calmar_ratio=asset_data["calmar_ratio"],
                max_drawdown=asset_data["max_drawdown"],
                win_rate=0.0,  # Not available in database
                profit_factor=1.0,  # Not available in database
                total_return=asset_data["total_return"],
                # Trading parameters
                trading_style=trading_params["trading_style"],
                timeframe=trading_params["timeframe"],
                risk_per_trade=trading_params["risk_per_trade"],
                stop_loss=trading_params["stop_loss_points"],
                take_profit=trading_params["take_profit_points"],
                position_size=trading_params["position_size_percent"],
            )
            recommendations.append(recommendation)

        # Generate AI analysis
        ai_analysis = self._generate_ai_analysis(
            recommendations, correlation_data, risk_tolerance
        )

        portfolio_rec = PortfolioRecommendation(
            recommendations=recommendations,
            total_score=self._ensure_python_type(
                np.mean([r.score for r in recommendations])
            ),
            risk_profile=risk_tolerance,
            diversification_score=self._ensure_python_type(
                correlation_data["diversification_score"]
            ),
            correlation_analysis=correlation_data["correlations"],
            overall_reasoning=ai_analysis["reasoning"],
            warnings=ai_analysis["warnings"],
            confidence=self._ensure_python_type(
                np.mean([r.confidence for r in recommendations])
            ),
        )

        # Save to database and exports
        self._save_to_database(portfolio_rec, quarter, portfolio_name)
        self._save_to_exports(
            recommendations, risk_tolerance, quarter, portfolio_name, timeframe
        )

        return portfolio_rec

    def _generate_portfolio_filtered_recommendations(
        self,
        symbols: list[str],
        risk_tolerance: str = "moderate",
        min_confidence: float = 0.6,
        max_assets: int = 10,
        quarter: Optional[str] = None,
        timeframe: str = "1h",
        portfolio_name: Optional[str] = None,
    ) -> PortfolioRecommendation:
        """Generate recommendations specifically for portfolio symbols only."""
        self.logger.info(
            "Generating AI recommendations for %s risk profile", risk_tolerance
        )

        # Load backtest results and filter by portfolio symbols immediately
        backtest_data = self._load_backtest_results(quarter, None)
        if not backtest_data:
            raise ValueError("No backtest results found")

        # Filter to only include portfolio symbols and get best strategy per asset
        portfolio_backtest_data = []
        symbol_best_strategies = {}

        # Group by symbol and find best strategy for each
        for asset in backtest_data:
            if asset["symbol"] in symbols:
                symbol = asset["symbol"]

                # Keep track of best strategy per symbol (highest sortino ratio)
                if symbol not in symbol_best_strategies or (
                    asset["sortino_ratio"]
                    > symbol_best_strategies[symbol]["sortino_ratio"]
                ):
                    symbol_best_strategies[symbol] = asset

        # Convert to list format (only best strategy per asset)
        portfolio_backtest_data = list(symbol_best_strategies.values())

        if not portfolio_backtest_data:
            # Return empty portfolio recommendation
            return PortfolioRecommendation(
                recommendations=[],
                total_score=0.0,
                risk_profile=risk_tolerance,
                diversification_score=0.0,
                correlation_analysis={},
                overall_reasoning="No backtested assets found in portfolio. Only assets with backtest or optimization data are analyzed.",
                warnings=["Portfolio contains no backtested assets"],
                confidence=0.0,
            )

        self.logger.info(
            "Found %d backtested assets from portfolio symbols",
            len(portfolio_backtest_data),
        )

        # Performance-based scoring (no filtering, just scoring)
        scored_assets = self._calculate_performance_scores(portfolio_backtest_data)

        # Skip most filtering - just use scored assets directly
        # Portfolio correlation analysis
        correlation_data = self._analyze_correlations(scored_assets)

        # Calculate equal allocation for portfolio assets (like typical bond portfolios)
        num_assets = len(scored_assets)
        if num_assets > 0:
            base_allocation = 100.0 / num_assets  # Equal weight allocation
            for asset in scored_assets:
                # Adjust allocation slightly based on performance score
                score_multiplier = 0.8 + (
                    asset["score"] * 0.4
                )  # 0.8x to 1.2x based on score
                asset["allocation"] = self._ensure_python_type(
                    min(20.0, max(2.0, base_allocation * score_multiplier))
                )

        # Use scored assets directly (no filtering)
        confidence_scores = self._calculate_confidence_scores(
            scored_assets, portfolio_backtest_data
        )

        # Create asset recommendations for ALL portfolio assets (no filtering by confidence)
        recommendations = []
        for asset_data in confidence_scores:
            # Determine investment recommendation
            base_reasoning = asset_data.get(
                "reasoning",
                f"Sortino: {asset_data['sortino_ratio']:.2f}, Max DD: {asset_data['max_drawdown']:.1%}",
            )

            if asset_data["confidence"] >= 0.6 and asset_data["score"] >= 0.3:
                invest_decision = "INVEST_WITH_RISK_MANAGEMENT"
                reasoning = f"Moderate performance metrics suggest cautious investment. {base_reasoning}"
            elif asset_data["confidence"] >= 0.4 and asset_data["score"] >= 0.2:
                invest_decision = "CONSIDER_WITH_HIGH_CAUTION"
                reasoning = f"Below-average performance requires extreme caution. {base_reasoning}"
            else:
                invest_decision = "DO_NOT_INVEST"
                reasoning = (
                    f"Poor performance metrics indicate high risk. {base_reasoning}"
                )

            # Calculate trading parameters
            trading_params = self._calculate_trading_parameters(asset_data, timeframe)

            recommendation = AssetRecommendation(
                symbol=asset_data["symbol"],
                strategy=asset_data["strategy"],
                score=self._ensure_python_type(asset_data["score"]),
                confidence=self._ensure_python_type(asset_data["confidence"]),
                allocation_percentage=self._ensure_python_type(
                    asset_data["allocation"]
                ),  # Always show suggested allocation
                risk_level=self._classify_risk_level(asset_data),
                reasoning=reasoning,
                red_flags=[*asset_data.get("red_flags", []), invest_decision],
                sortino_ratio=asset_data["sortino_ratio"],
                calmar_ratio=asset_data["calmar_ratio"],
                max_drawdown=asset_data["max_drawdown"],
                win_rate=0.0,  # Not available in database
                profit_factor=1.0,  # Not available in database
                total_return=asset_data["total_return"],
                # Trading parameters
                trading_style=trading_params["trading_style"],
                timeframe=trading_params["timeframe"],
                risk_per_trade=trading_params["risk_per_trade"],
                stop_loss=trading_params["stop_loss_points"],
                take_profit=trading_params["take_profit_points"],
                position_size=trading_params["position_size_percent"],
            )
            recommendations.append(recommendation)

        # Generate AI analysis
        ai_analysis = self._generate_ai_analysis(
            recommendations, correlation_data, risk_tolerance
        )

        portfolio_rec = PortfolioRecommendation(
            recommendations=recommendations,
            total_score=self._ensure_python_type(
                np.mean([r.score for r in recommendations])
            ),
            risk_profile=risk_tolerance,
            diversification_score=self._ensure_python_type(
                correlation_data["diversification_score"]
            ),
            correlation_analysis=correlation_data["correlations"],
            overall_reasoning=ai_analysis["reasoning"],
            warnings=ai_analysis["warnings"],
            confidence=self._ensure_python_type(
                np.mean([r.confidence for r in recommendations])
            ),
        )

        return portfolio_rec

    def generate_portfolio_recommendations(
        self,
        portfolio_config_path: str,
        risk_tolerance: str = "moderate",
        min_confidence: float = 0.6,
        max_assets: int = 10,
        quarter: Optional[str] = None,
        timeframe: str = "1h",
        filename_interval: Optional[str] = None,
        generate_html: bool = True,
    ) -> tuple[PortfolioRecommendation, str]:
        """Generate AI recommendations for a specific portfolio with HTML report."""
        import json
        from pathlib import Path

        # Load portfolio configuration
        portfolio_path = Path(portfolio_config_path)
        with portfolio_path.open() as f:
            portfolio_config = json.load(f)

        # Handle nested portfolio configuration
        if len(portfolio_config) == 1:
            # Single key, assume it's the portfolio config
            portfolio_key = list(portfolio_config.keys())[0]
            portfolio_data = portfolio_config[portfolio_key]
        else:
            # Direct configuration
            portfolio_data = portfolio_config

        portfolio_name = portfolio_data.get(
            "name", portfolio_path.stem.replace("_", " ").title()
        )
        symbols = portfolio_data.get("symbols", [])

        self.logger.info(
            "Generating AI recommendations for %s portfolio (%d symbols)",
            portfolio_name,
            len(symbols),
        )

        # Generate recommendations for only the portfolio symbols
        # (filter backtest data first before generating recommendations)
        portfolio_filtered_recommendations = (
            self._generate_portfolio_filtered_recommendations(
                symbols=symbols,
                risk_tolerance=risk_tolerance,
                min_confidence=min_confidence,
                max_assets=max_assets,
                quarter=quarter,
                timeframe=timeframe,
                portfolio_name=portfolio_name,
            )
        )

        portfolio_recommendations = portfolio_filtered_recommendations.recommendations

        self.logger.info(
            "Generated recommendations for %d backtested assets from %d portfolio symbols",
            len(portfolio_recommendations),
            len(symbols),
        )

        # Use the filtered portfolio recommendations
        filtered_portfolio = portfolio_filtered_recommendations

        # Save to markdown exports (skip database save due to model mismatch)
        self._save_to_exports(
            filtered_portfolio.recommendations,
            risk_tolerance,
            quarter,
            portfolio_name,
            filename_interval or timeframe,
        )

        # Try to save to database (may fail due to model mismatch)
        try:
            self._save_to_database(filtered_portfolio, quarter, portfolio_name)
        except Exception as e:
            self.logger.warning(
                "Database save failed: %s - continuing with markdown export", e
            )

        html_path = ""
        if generate_html:
            # Generate HTML report
            report_generator = AIReportGenerator()
            # Determine year/quarter parts from quarter token or now
            from datetime import datetime as _dt

            if quarter and "_" in (quarter or ""):
                quarter_part, year_part = quarter.split("_")
            else:
                now = _dt.now()
                quarter_part = quarter or f"Q{(now.month - 1) // 3 + 1}"
                year_part = str(now.year)
            html_path = report_generator.generate_html_report(
                recommendation=filtered_portfolio,
                portfolio_name=portfolio_name,
                year=year_part,
                quarter=quarter_part,
                interval=filename_interval or timeframe,
            )

        return filtered_portfolio, html_path

    def _load_backtest_results(
        self,
        quarter: Optional[str] = None,
        portfolio_symbols: Optional[list[str]] = None,
    ) -> list[dict]:
        """Load backtest results, preferring primary DB but falling back to unified_models."""
        results: list[dict] = []
        used_source = None

        # Try primary DB if a session is available
        if self.db_session is not None:
            try:
                results = self._load_from_database(quarter, portfolio_symbols)
                if results:
                    used_source = "primary_db"
            except Exception as e:
                self.logger.warning("Primary DB load failed: %s", e)

        # Fallback to unified_models BestStrategy if no primary data
        if not results:
            try:
                results = self._load_from_unified_models(quarter, portfolio_symbols)
                if results:
                    used_source = "unified_models"
            except Exception as e:
                self.logger.warning("Unified models load failed: %s", e)

        if not results:
            self.logger.warning(
                "No results found for AI recommendations after all fallbacks"
            )
            # Last-resort fallback: parse CSV exports produced in the quarterly folder
            try:
                csv_results = self._load_from_csv_exports(quarter, portfolio_symbols)
                if csv_results:
                    self.logger.info(
                        "Using CSV exports for AI recommendations (%d rows)",
                        len(csv_results),
                    )
                    return csv_results
            except Exception as e:
                self.logger.debug("CSV fallback failed: %s", e)
            return []

        self.logger.info(
            "Using %s data for AI recommendations (%d rows)", used_source, len(results)
        )
        return results

    def _load_from_unified_models(
        self,
        quarter: Optional[str] = None,
        portfolio_symbols: Optional[list[str]] = None,
    ) -> list[dict]:
        from datetime import datetime

        try:
            from src.database import unified_models as um
        except Exception:
            return []

        sess = um.Session()
        try:
            q = sess.query(um.BestStrategy)
            if portfolio_symbols:
                q = q.filter(um.BestStrategy.symbol.in_(portfolio_symbols))

            if quarter:
                year, qstr = quarter.split("_")
                qnum = int(qstr[1])
                start_month = (qnum - 1) * 3 + 1
                end_month = qnum * 3
                start_date = datetime(int(year), start_month, 1)
                end_date = (
                    datetime(int(year) + 1, 1, 1)
                    if qnum == 4
                    else datetime(int(year), end_month + 1, 1)
                )
                q = q.filter(
                    um.BestStrategy.updated_at >= start_date,
                    um.BestStrategy.updated_at < end_date,
                )

            q = q.order_by(um.BestStrategy.sortino_ratio.desc())
            rows = q.all()
            out = [
                {
                    "symbol": r.symbol,
                    "strategy": r.strategy,
                    "sortino_ratio": float(r.sortino_ratio or 0),
                    "calmar_ratio": float(r.calmar_ratio or 0),
                    "sharpe_ratio": float(r.sharpe_ratio or 0),
                    "total_return": float(r.total_return or 0),
                    "max_drawdown": float(r.max_drawdown or 0),
                    "created_at": r.updated_at.isoformat() if r.updated_at else None,
                }
                for r in rows
            ]
            return out
        finally:
            try:
                sess.close()
            except Exception:
                pass

    def _load_from_csv_exports(
        self,
        quarter: Optional[str] = None,
        portfolio_symbols: Optional[list[str]] = None,
    ) -> list[dict]:
        """Load best-per-asset rows from CSV exports under exports/csv/<year>/<quarter>."""
        from pathlib import Path

        # Determine year and quarter folder
        year_part = None
        quarter_part = None
        if quarter and "_" in quarter:
            q, y = quarter.split("_")
            quarter_part = q
            year_part = y
        else:
            from datetime import datetime as _dt

            now = _dt.utcnow()
            year_part = str(now.year)
            quarter_part = f"Q{((now.month - 1) // 3) + 1}"

        base = Path("exports/csv") / str(year_part) / str(quarter_part)
        if not base.exists():
            return []

        rows: list[dict] = []
        # Load all CSVs for the quarter; we'll filter to portfolio symbols
        for csv_path in base.glob("*.csv"):
            try:
                df = pd.read_csv(str(csv_path))
            except Exception as e:
                self.logger.debug("Failed reading CSV %s: %s", csv_path, e)
                continue
            # Normalize expected columns
            cols = {c.lower(): c for c in df.columns}
            # Prefer 'Asset' else 'Symbol'
            asset_col = cols.get("asset") or cols.get("symbol")
            strat_col = cols.get("best_strategy") or cols.get("strategy")
            tf_col = cols.get("best_timeframe") or cols.get("timeframe")
            if not asset_col or not strat_col:
                continue
            # Filter symbols if provided
            if portfolio_symbols:
                df = df[df[asset_col].isin(portfolio_symbols)]
            if df.empty:
                continue

            # Build numeric metrics with safe defaults
            def _num(colname: str, _cols=cols, _df=df) -> pd.Series:
                c = _cols.get(colname.lower())
                if not c or c not in _df.columns:
                    return pd.Series([np.nan] * len(_df))
                try:
                    return pd.to_numeric(_df[c], errors="coerce")
                except Exception:
                    return pd.Series([np.nan] * len(_df))

            srt = _num("Sortino_Ratio")
            cal = _num("Calmar_Ratio")
            shp = _num("Sharpe_Ratio")
            trn = _num("Total_Return_Pct")
            if trn.isna().all():
                trn = _num("Total_Return")
            mdd = _num("Max_Drawdown_Pct")
            if mdd.isna().all():
                mdd = _num("Max_Drawdown")

            tmp = pd.DataFrame(
                {
                    "symbol": df[asset_col].astype(str),
                    "strategy": df[strat_col].astype(str),
                    "timeframe": df[tf_col].astype(str) if tf_col else "1d",
                    "sortino_ratio": srt.fillna(0.0),
                    "calmar_ratio": cal.fillna(0.0),
                    "sharpe_ratio": shp.fillna(0.0),
                    "total_return": trn.fillna(0.0),
                    "max_drawdown": mdd.fillna(0.0),
                }
            )
            rows.extend(tmp.to_dict("records"))

        if not rows:
            return []

        # Reduce to best per symbol by Sortino
        by_symbol = {}
        for r in rows:
            sym = r.get("symbol")
            if not sym:
                continue
            if sym not in by_symbol or float(r.get("sortino_ratio") or 0) > float(
                by_symbol[sym].get("sortino_ratio") or 0
            ):
                by_symbol[sym] = r

        return list(by_symbol.values())

    def _load_from_database(
        self,
        quarter: Optional[str] = None,
        portfolio_symbols: Optional[list[str]] = None,
    ) -> list[dict]:
        """Load best strategies from database for faster and cleaner recommendations."""
        from datetime import datetime

        # Query best_strategies table directly - much more efficient
        query = self.db_session.query(BestStrategy)

        # Filter by portfolio symbols if provided
        if portfolio_symbols:
            query = query.filter(BestStrategy.symbol.in_(portfolio_symbols))
            self.logger.info(
                "Filtering by %s portfolio symbols", len(portfolio_symbols)
            )

        if quarter:
            # Filter by quarter if specified
            year, q = quarter.split("_")
            quarter_num = int(q[1])
            start_month = (quarter_num - 1) * 3 + 1
            end_month = quarter_num * 3

            start_date = datetime(int(year), start_month, 1)
            if quarter_num == 4:
                end_date = datetime(int(year) + 1, 1, 1)
            else:
                end_date = datetime(int(year), end_month + 1, 1)

            query = query.filter(
                BestStrategy.updated_at >= start_date,
                BestStrategy.updated_at < end_date,
            )

        # Order by primary metric (Sortino ratio) descending
        query = query.order_by(BestStrategy.sortino_ratio.desc())
        results = query.all()

        self.logger.info("Loaded %d best strategies from database", len(results))

        return [
            {
                "symbol": result.symbol,
                "strategy": result.strategy,
                "sortino_ratio": float(result.sortino_ratio or 0),
                "calmar_ratio": float(result.calmar_ratio or 0),
                "sharpe_ratio": float(result.sharpe_ratio or 0),
                "total_return": float(result.total_return or 0),
                "max_drawdown": float(result.max_drawdown or 0),
                "created_at": result.updated_at.isoformat()
                if result.updated_at
                else None,
            }
            for result in results
        ]

    def _load_from_reports(self, quarter: Optional[str] = None) -> list[dict]:
        """Load backtest results from HTML reports."""
        reports_dir = Path("exports/reports")

        if quarter:
            year, q = quarter.split("_")
            reports_path = reports_dir / year / q
        else:
            # Get latest quarter
            reports_path = reports_dir / "2025" / "Q3"

        if not reports_path.exists():
            self.logger.warning("Reports directory %s not found", reports_path)
            return []

        # Parse HTML reports to extract metrics
        return self._parse_html_reports(reports_path)

    def _parse_html_reports(self, reports_path: Path) -> list[dict]:
        """Parse HTML reports to extract backtest metrics."""

        from bs4 import BeautifulSoup

        parsed_data = []

        # Find HTML reports in the directory
        html_files = list(reports_path.glob("*.html"))

        for html_file in html_files:
            try:
                with Path(html_file).open(encoding="utf-8") as f:
                    content = f.read()

                soup = BeautifulSoup(content, "html.parser")

                # Find asset sections
                asset_sections = soup.find_all("div", class_="asset-section")

                for section in asset_sections:
                    # Extract asset symbol from the title
                    asset_title = section.find("h2", class_="asset-title")
                    if not asset_title:
                        continue

                    symbol = asset_title.text.strip()

                    # Extract best strategy from the badge
                    strategy_badge = section.find("span", class_="strategy-badge")
                    if not strategy_badge:
                        continue

                    # Parse "Best: Strategy Name"
                    strategy_text = strategy_badge.text.strip()
                    if strategy_text.startswith("Best: "):
                        strategy = strategy_text[6:].strip()  # Remove "Best: "
                    else:
                        continue

                    # Extract metrics from metric cards
                    metrics_data = {
                        "symbol": symbol,
                        "strategy": strategy,
                        "sortino_ratio": 0.0,
                        "calmar_ratio": 0.0,
                        "sharpe_ratio": 0.0,
                        "profit_factor": 0.0,
                        "max_drawdown": 0.0,
                        "volatility": 0.0,
                        "win_rate": 0.0,
                        "total_return": 0.0,
                        "num_trades": 0,
                        "created_at": "2025-08-14",
                        "initial_capital": 10000,
                        "final_value": 10000,
                    }

                    # Find metric cards and extract values
                    metric_cards = section.find_all("div", class_="metric-card")
                    for card in metric_cards:
                        label_elem = card.find("div", class_="metric-label")
                        value_elem = card.find("div", class_="metric-value")

                        if not label_elem or not value_elem:
                            continue

                        label = label_elem.text.strip().lower()
                        value_text = value_elem.text.strip()

                        # Parse metric values
                        try:
                            # Remove % and convert to float
                            if "%" in value_text:
                                value = float(value_text.replace("%", "")) / 100
                            else:
                                value = float(value_text)

                            # Map labels to our metric keys
                            if "sortino" in label:
                                metrics_data["sortino_ratio"] = value
                            elif "calmar" in label:
                                metrics_data["calmar_ratio"] = value
                            elif "sharpe" in label:
                                metrics_data["sharpe_ratio"] = value
                            elif "profit factor" in label:
                                metrics_data["profit_factor"] = value
                            elif "max drawdown" in label or "maximum drawdown" in label:
                                metrics_data["max_drawdown"] = value
                            elif "volatility" in label:
                                metrics_data["volatility"] = value
                            elif "win rate" in label:
                                metrics_data["win_rate"] = value
                            elif "total return" in label:
                                metrics_data["total_return"] = value
                        except ValueError:
                            continue

                    parsed_data.append(metrics_data)

            except Exception as e:
                self.logger.warning("Error parsing HTML report %s: %s", html_file, e)
                continue

        self.logger.info("Parsed %d asset metrics from HTML reports", len(parsed_data))
        return parsed_data

    def _calculate_performance_scores(self, backtest_data: list[dict]) -> list[dict]:
        """Calculate performance scores for each asset based on metrics."""
        scored_assets = []

        for asset in backtest_data:
            # Normalize metrics for scoring
            sortino_score = min(max(asset["sortino_ratio"], 0), 5) / 5.0
            calmar_score = min(max(asset["calmar_ratio"], 0), 5) / 5.0
            drawdown_score = max(0, 1 - abs(asset["max_drawdown"]) / 100)
            return_score = min(max(asset["total_return"] / 100, 0), 1.0)

            # Calculate weighted score using available metrics
            score = (
                sortino_score * 0.4  # Primary metric
                + calmar_score * 0.3  # Secondary metric
                + return_score * 0.2  # Return component
                + drawdown_score * 0.1  # Risk component
            )

            asset["score"] = self._ensure_python_type(score)
            scored_assets.append(asset)

        return sorted(scored_assets, key=lambda x: x["score"], reverse=True)

    def _apply_risk_filters(
        self, assets: list[dict], risk_tolerance: str
    ) -> list[dict]:
        """Filter assets based on risk tolerance."""
        risk_criteria = self.risk_levels[risk_tolerance]

        filtered = []
        for asset in assets:
            max_dd = abs(asset["max_drawdown"])
            sortino = asset["sortino_ratio"]

            if (
                max_dd <= risk_criteria["max_drawdown"]
                and sortino >= risk_criteria["min_sortino"]
            ):
                filtered.append(asset)

        return filtered

    def _analyze_correlations(self, assets: list[dict]) -> dict:
        """Analyze portfolio correlations for diversification."""
        # This would calculate actual correlations using price data
        # TODO: Implement actual correlation calculation using price data
        symbols = [asset["symbol"] for asset in assets]

        # Placeholder correlation matrix - to be implemented
        correlations = {}
        # Calculate a basic diversification score based on number of assets
        # More assets generally means better diversification
        diversification_score = min(0.9, 0.3 + (len(symbols) * 0.1))
        _ = symbols  # Unused for now

        return {
            "correlations": correlations,
            "diversification_score": diversification_score,
        }

    def _optimize_strategy_asset_matching(self, assets: list[dict]) -> list[dict]:
        """Find optimal strategy-asset combinations."""
        # Group by symbol and find best strategy for each
        symbol_strategies = {}

        for asset in assets:
            symbol = asset["symbol"]
            if symbol not in symbol_strategies:
                symbol_strategies[symbol] = []
            symbol_strategies[symbol].append(asset)

        # Select best strategy per symbol
        optimized = []
        for symbol, strategies in symbol_strategies.items():
            best_strategy = max(strategies, key=lambda x: x["score"])
            optimized.append(best_strategy)

        return optimized

    def _suggest_allocations(
        self, assets: list[dict], _: str, max_assets: int
    ) -> list[dict]:
        """Suggest portfolio allocations based on scores and risk."""
        # Take top assets
        top_assets = assets[:max_assets]

        if not top_assets:
            return []

        # Calculate allocations based on scores
        total_score = sum(asset["score"] for asset in top_assets)

        for asset in top_assets:
            if total_score > 0:
                allocation = (asset["score"] / total_score) * 100
            else:
                allocation = 100 / len(top_assets)

            asset["allocation"] = self._ensure_python_type(allocation)

        return top_assets

    def _detect_red_flags(self, assets: list[dict]) -> list[dict]:
        """Detect potential issues with recommended assets."""
        for asset in assets:
            red_flags = []

            # High drawdown warning
            if abs(asset["max_drawdown"]) > 0.3:
                red_flags.append("High maximum drawdown risk")

            # Low Sortino ratio
            if asset["sortino_ratio"] < 0.5:
                red_flags.append("Low risk-adjusted returns")

            # High drawdown (using max_drawdown as risk indicator)
            if abs(asset["max_drawdown"]) > 40:
                red_flags.append("High drawdown risk")

            # Low total return
            if asset["total_return"] < 5:
                red_flags.append("Low returns")

            # Poor risk-adjusted return (low Sharpe ratio)
            if asset["sharpe_ratio"] < 0.5:
                red_flags.append("Poor risk-adjusted returns")

            asset["red_flags"] = red_flags

        return assets

    def _calculate_confidence_scores(
        self, assets: list[dict], _: list[dict]
    ) -> list[dict]:
        """Calculate confidence scores based on data quality and consistency."""
        for asset in assets:
            confidence_factors = []

            # Performance stability (Sortino ratio) - use actual database metric
            sortino_factor = float(min(max(asset["sortino_ratio"], 0) / 2, 1.0))
            confidence_factors.append(sortino_factor)

            # Risk management (drawdown control) - use actual database metric
            drawdown_factor = float(max(0, 1 - abs(asset["max_drawdown"]) / 50))
            confidence_factors.append(drawdown_factor)

            # Return consistency (Calmar ratio) - use actual database metric
            calmar_factor = float(min(max(asset["calmar_ratio"], 0) / 2, 1.0))
            confidence_factors.append(calmar_factor)

            # Risk-adjusted performance (Sharpe ratio) - use actual database metric
            sharpe_factor = float(min(max(asset["sharpe_ratio"], 0) / 2, 1.0))
            confidence_factors.append(sharpe_factor)

            # Calculate weighted confidence and ensure it's a Python float
            asset["confidence"] = self._ensure_python_type(np.mean(confidence_factors))

        return assets

    def _classify_risk_level(self, asset_data: dict) -> str:
        """Classify asset risk level based on database metrics only."""
        max_dd = abs(asset_data["max_drawdown"])
        sortino_ratio = asset_data["sortino_ratio"]

        # Use drawdown and Sortino ratio for risk classification
        if max_dd <= 10 and sortino_ratio >= 1.0:
            return "Low"
        if max_dd <= 25 and sortino_ratio >= 0.5:
            return "Medium"
        return "High"

    def _calculate_trading_parameters(
        self, asset_data: dict, timeframe: str = "1h"
    ) -> dict:
        """Calculate trading parameters based on timeframe and asset characteristics."""

        # Determine trading style based on timeframe
        timeframe_minutes = self._timeframe_to_minutes(timeframe)
        is_scalping = timeframe_minutes < 60  # Less than 1 hour = scalping

        trading_style = "scalp" if is_scalping else "swing"

        # Get asset volatility for parameter adjustment
        volatility = asset_data.get("volatility", 0.02)  # Default 2% volatility
        max_drawdown = abs(asset_data.get("max_drawdown", 0.05))

        if is_scalping:
            # Scalping parameters (tighter, more frequent trades)
            base_risk = 0.5  # 0.5% base risk per trade for scalping
            base_sl_points = max(
                5, volatility * 1000
            )  # Minimum 5 points, volatility-adjusted
            base_tp_points = base_sl_points * 2  # 1:2 risk-reward for scalping
            position_size = 5.0  # Smaller position sizes for scalping

            # Adjust based on volatility
            if volatility > 0.05:  # High volatility assets
                base_risk *= 0.7  # Reduce risk
                base_sl_points *= 1.5
                base_tp_points *= 1.5
        else:
            # Swing trading parameters (wider, longer-term trades)
            base_risk = 2.0  # 2% base risk per trade for swing
            base_sl_points = max(
                20, volatility * 3000
            )  # Minimum 20 points, volatility-adjusted
            base_tp_points = base_sl_points * 3  # 1:3 risk-reward for swing
            position_size = 10.0  # Larger position sizes for swing

            # Adjust based on volatility and drawdown
            if volatility > 0.03:  # High volatility assets
                base_risk *= 0.8
                base_sl_points *= 1.2
                base_tp_points *= 1.2

            if max_drawdown > 0.2:  # High drawdown history
                base_risk *= 0.6
                position_size *= 0.8

        # Risk level adjustments
        risk_level = self._classify_risk_level(asset_data)
        if risk_level == "High":
            base_risk *= 0.5
            position_size *= 0.7
        elif risk_level == "Low":
            base_risk *= 1.2
            position_size *= 1.1

        return {
            "trading_style": trading_style,
            "timeframe": timeframe,
            "risk_per_trade": round(base_risk, 1),
            "stop_loss_points": round(base_sl_points, 0),
            "take_profit_points": round(base_tp_points, 0),
            "position_size_percent": round(position_size, 1),
        }

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        timeframe = timeframe.lower()

        if "m" in timeframe:
            return int(timeframe.replace("m", ""))
        if "h" in timeframe:
            return int(timeframe.replace("h", "")) * 60
        if "d" in timeframe:
            return int(timeframe.replace("d", "")) * 24 * 60
        if "w" in timeframe:
            return int(timeframe.replace("w", "")) * 7 * 24 * 60
        return 60  # Default to 1 hour

    def _generate_ai_analysis(
        self,
        recommendations: list[AssetRecommendation],
        correlation_data: dict,
        risk_tolerance: str,
    ) -> dict[str, Any]:
        """Generate AI-powered analysis and reasoning."""
        if not recommendations:
            return {
                "reasoning": "No backtested assets found in portfolio. Only assets with backtest or optimization data are analyzed.",
                "warnings": ["Portfolio contains no backtested assets"],
            }

        # Prepare data for AI analysis
        analysis_data = {
            "risk_tolerance": risk_tolerance,
            "num_recommendations": len(recommendations),
            "avg_sortino": np.mean([r.sortino_ratio for r in recommendations]),
            "avg_calmar": np.mean([r.calmar_ratio for r in recommendations]),
            "max_drawdown_range": [r.max_drawdown for r in recommendations],
            "diversification_score": correlation_data["diversification_score"],
            "total_allocation": sum(r.allocation_percentage for r in recommendations),
            "red_flags_count": sum(len(r.red_flags) for r in recommendations),
        }

        # Generate AI reasoning
        try:
            ai_response = self.llm_client.analyze_portfolio(
                analysis_data, recommendations
            )
            return {
                "reasoning": ai_response.get(
                    "reasoning", "Analysis completed successfully"
                ),
                "warnings": ai_response.get("warnings", []),
            }
        except Exception as e:
            self.logger.error("AI analysis failed: %s", e)
            return {
                "reasoning": f"Quantitative analysis complete. {len(recommendations)} assets recommended with average Sortino ratio of {analysis_data['avg_sortino']:.2f}",
                "warnings": [
                    "AI analysis unavailable - using quantitative metrics only"
                ],
            }

    def get_asset_comparison(
        self, symbols: list[str], strategy: Optional[str] = None
    ) -> pd.DataFrame:
        """Compare assets side by side with key metrics."""
        if self.db_session:
            from sqlalchemy import or_

            # Filter for results that contain any of the requested symbols
            symbol_filters = [BacktestResult.symbols.any(symbol) for symbol in symbols]
            query = self.db_session.query(BacktestResult).filter(or_(*symbol_filters))
            if strategy:
                query = query.filter(BacktestResult.strategy == strategy)

            results = query.all()

            comparison_data = []
            for result in results:
                comparison_data.append(
                    {
                        "Symbol": result.symbols[0] if result.symbols else "UNKNOWN",
                        "Strategy": result.strategy,
                        "Sortino Ratio": float(result.sortino_ratio or 0),
                        "Calmar Ratio": float(result.calmar_ratio or 0),
                        "Max Drawdown": float(result.max_drawdown or 0),
                        "Total Return": float(result.total_return or 0),
                        "Win Rate": float(result.win_rate or 0),
                        "Profit Factor": float(result.profit_factor or 0),
                    }
                )

            return pd.DataFrame(comparison_data)

        return pd.DataFrame()

    def explain_recommendation(self, symbol: str, strategy: str) -> dict[str, Any]:
        """Get detailed explanation for a specific recommendation."""
        # Load specific asset data
        asset_data = self._get_asset_data(symbol, strategy)

        if not asset_data:
            return {"error": "Asset data not found"}

        # Generate detailed AI explanation
        try:
            explanation = self.llm_client.explain_asset_recommendation(asset_data)
            return explanation
        except Exception as e:
            self.logger.error("Failed to generate explanation: %s", e)
            return {
                "summary": f"Asset {symbol} with {strategy} strategy shows Sortino ratio of {asset_data.get('sortino_ratio', 0):.2f}",
                "strengths": ["Quantitative metrics available"],
                "concerns": ["AI explanation unavailable"],
                "recommendation": "Review metrics manually",
            }

    def generate_practical_recommendations_from_html(
        self, html_report_path: str, risk_tolerance: str = "moderate"
    ) -> str:
        """Generate practical trading recommendations from HTML report using database data."""
        from pathlib import Path

        from bs4 import BeautifulSoup

        # Parse HTML to get collection assets
        html_path = Path(html_report_path)
        if not html_path.exists():
            raise ValueError(f"HTML report not found: {html_report_path}")

        with html_path.open(encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        # Extract asset symbols from HTML
        asset_sections = soup.find_all("div", class_="asset-section")
        symbols = []
        for section in asset_sections:
            title = section.find("h2", class_="asset-title")
            if title:
                symbols.append(title.get_text().strip())

        if not symbols:
            raise ValueError("No assets found in HTML report")

        # Get database data for these symbols
        if not self.db_session:
            raise ValueError("Database session required")

        from src.database.models import BestStrategy

        # Query database for actual performance data
        strategies = (
            self.db_session.query(BestStrategy)
            .filter(BestStrategy.symbol.in_(symbols))
            .order_by(BestStrategy.sortino_ratio.desc())
            .all()
        )

        if not strategies:
            raise ValueError("No strategy data found for collection assets")

        # Generate practical trading recommendations
        collection_name = html_path.stem.replace("_Q3_2025", "").replace("_", " ")

        recommendations = self._create_practical_trading_guide(
            strategies, collection_name, risk_tolerance
        )

        return recommendations

    def _create_practical_trading_guide(
        self, strategies: list, collection_name: str, risk_tolerance: str
    ) -> str:
        """Create practical trading guide with entry/exit rules."""

        # Sort by Sortino ratio and categorize by performance tiers (bond-appropriate thresholds)
        top_tier = [s for s in strategies if float(s.sortino_ratio) > 1.0]
        mid_tier = [s for s in strategies if 0.5 <= float(s.sortino_ratio) <= 1.0]

        guide = f"""# {collection_name} - Practical Trading Strategy Guide
**Model:** GPT-5-mini | **Generated:** Q3 2025
**Risk Profile:** {risk_tolerance.title()}
**Assets Analyzed:** {len(strategies)}

## 🎯 Actionable Investment Recommendations

### **🥇 TOP TIER STRATEGIES** (Sortino > 1.0)
"""

        for i, strategy in enumerate(top_tier[:5], 1):
            # Calculate practical levels based on historical performance
            take_profit = min(
                float(strategy.total_return) * 0.3, 10.0
            )  # 30% of total return, max 10%
            stop_loss = min(
                float(strategy.max_drawdown) * 0.5, 5.0
            )  # 50% of max drawdown, max 5%

            guide += f"""
#### {i}. **{strategy.symbol} - {strategy.strategy.upper()}** (Sortino: {strategy.sortino_ratio:.3f})
- **Entry Signal**: {self._get_entry_signal(strategy.strategy)}
- **Take Profit**: +{take_profit:.1f}% or technical reversal
- **Stop Loss**: -{stop_loss:.1f}% strict
- **Position Size**: {self._get_position_size(strategy.sortino_ratio, risk_tolerance)}% allocation
- **Max Drawdown Risk**: {strategy.max_drawdown:.1f}%
- **Historical Return**: {float(strategy.total_return):.1f}% over backtest period
"""

        # Individual recommendations for ALL assets
        guide += f"""
## 📋 **INDIVIDUAL ASSET RECOMMENDATIONS**
*Complete analysis of all {len(strategies)} assets in collection*

"""

        for strategy in strategies:
            sortino = float(strategy.sortino_ratio)
            total_return = float(strategy.total_return)
            max_dd = float(strategy.max_drawdown)

            # Calculate levels for each asset
            take_profit = min(total_return * 0.3, 10.0) if total_return > 0 else 3.0
            stop_loss = min(max_dd * 0.5, 5.0) if max_dd > 0 else 2.0

            # Determine recommendation level
            if sortino > 1.0:
                recommendation = "🟢 **BUY**"
                allocation = self._get_position_size(sortino, risk_tolerance)
            elif sortino > 0.5:
                recommendation = "🟡 **HOLD**"
                allocation = max(
                    self._get_position_size(sortino, risk_tolerance) - 5, 5
                )
            else:
                recommendation = "🔴 **AVOID**"
                allocation = 0

            guide += f"""
### **{strategy.symbol}** - {strategy.strategy.upper()}
- **Rating**: {recommendation} | **Sortino**: {sortino:.3f}
- **Entry**: {self._get_entry_signal(strategy.strategy)}
- **Take Profit**: +{take_profit:.1f}% | **Stop Loss**: -{stop_loss:.1f}%
- **Allocation**: {allocation}% | **Return**: {total_return:.1f}% | **Max DD**: {max_dd:.1f}%
"""

        guide += f"""
## 📊 **Portfolio Construction**

### **{risk_tolerance.title()} Risk Allocation:**
```
{self._create_allocation_table(top_tier, mid_tier, risk_tolerance)}
```

## 🚨 **Risk Management Rules**
- **Maximum single position**: {30 if risk_tolerance == "aggressive" else 25 if risk_tolerance == "moderate" else 20}%
- **Portfolio max drawdown**: {15 if risk_tolerance == "aggressive" else 10 if risk_tolerance == "moderate" else 6}%
- **Rebalance trigger**: 20% deviation from target weights
- **Emergency exit**: Strategy technical breakdown

## 🎯 **Entry/Exit Decision Framework**

### **Universal BUY Conditions:**
1. Technical signal confirmed (strategy-specific)
2. Risk-reward ratio > 2:1
3. No major economic events in next 48h
4. Portfolio correlation < 0.8

### **Universal SELL Conditions:**
1. Take profit target reached
2. Stop loss triggered
3. Technical momentum reversal
4. Risk management override

**This guide provides actionable trading rules based on {len(strategies)} backtested strategies using real database metrics.**
"""

        return guide

    def _get_entry_signal(self, strategy: str) -> str:
        """Get entry signal description for strategy."""
        strategy_lower = strategy.lower()
        if "bollinger" in strategy_lower:
            return "Price touches lower Bollinger Band"
        if "rsi" in strategy_lower:
            return "RSI < 30 (oversold)"
        if "macd" in strategy_lower:
            return "MACD bullish crossover"
        return "Buy signal confirmed"

    def _get_position_size(self, sortino_ratio: float, risk_tolerance: str) -> int:
        """Calculate position size based on performance and risk tolerance."""
        base_size = (
            20
            if risk_tolerance == "aggressive"
            else 15
            if risk_tolerance == "moderate"
            else 10
        )

        if sortino_ratio > 3.0:
            return min(base_size + 10, 30)
        if sortino_ratio > 2.0:
            return min(base_size + 5, 25)
        if sortino_ratio > 1.0:
            return base_size
        return max(base_size - 5, 5)

    def _create_allocation_table(
        self, top_tier: list, mid_tier: list, risk_tolerance: str
    ) -> str:
        """Create allocation percentage table."""
        total_allocation = (
            80
            if risk_tolerance == "aggressive"
            else 70
            if risk_tolerance == "moderate"
            else 60
        )

        if not top_tier:
            return "No suitable strategies found for allocation"

        # Distribute allocation among top performers
        top_3 = top_tier[:3]
        if len(top_3) == 1:
            allocations = [total_allocation]
        elif len(top_3) == 2:
            allocations = [total_allocation * 0.6, total_allocation * 0.4]
        else:
            allocations = [
                total_allocation * 0.4,
                total_allocation * 0.35,
                total_allocation * 0.25,
            ]

        table = ""
        for i, (strategy, alloc) in enumerate(zip(top_3, allocations)):
            table += f"{alloc:.0f}% {strategy.symbol} ({strategy.strategy.upper()}) - Sortino {strategy.sortino_ratio:.2f}\n"

        cash_reserve = 100 - total_allocation
        table += f"{cash_reserve}% Cash Reserve"

        return table

    def _get_asset_data(self, symbol: str, strategy: str) -> dict:
        """Get specific asset backtest data."""
        if self.db_session:
            # Use PostgreSQL-specific array operations to avoid ARRAY.contains() error
            from sqlalchemy import func

            result = (
                self.db_session.query(BacktestResult)
                .filter(
                    func.array_to_string(BacktestResult.symbols, ",").contains(symbol),
                    BacktestResult.strategy == strategy,
                )
                .first()
            )

            if result:
                return {
                    "symbol": symbol,
                    "strategy": strategy,
                    "sortino_ratio": float(result.sortino_ratio or 0),
                    "calmar_ratio": float(result.calmar_ratio or 0),
                    "max_drawdown": float(result.max_drawdown or 0),
                    "total_return": float(result.total_return or 0),
                    "win_rate": float(result.win_rate or 0),
                    "profit_factor": float(result.profit_factor or 0),
                    "volatility": float(result.volatility or 0),
                }

        return {}

    def _save_to_exports(
        self,
        recommendations: list[AssetRecommendation],
        risk_tolerance: str,
        quarter: str,
        portfolio_name: Optional[str] = None,
        interval: Optional[str] = None,
    ):
        """Save recommendations to exports/ai_reco using unified filename convention."""
        from datetime import datetime
        from pathlib import Path

        # Parse quarter and year or use current
        if quarter and "_" in quarter:
            # quarter might be like "Q3_2025"
            quarter_part, year_part = quarter.split("_")
        else:
            current_date = datetime.now()
            quarter_part = quarter or f"Q{(current_date.month - 1) // 3 + 1}"
            year_part = str(current_date.year)

        # Create organized exports directory
        exports_dir = Path("exports/ai_reco") / year_part / quarter_part
        exports_dir.mkdir(parents=True, exist_ok=True)

        # Build unified filename: <Collectionname>_Collection_<Year>_<Quarter>_<Interval>.md
        collection_name = portfolio_name or "All_Collections"
        sanitized = (
            collection_name.replace(" ", "_").replace("/", "_").strip("_")
            or "All_Collections"
        )
        safe_interval = (interval or "multi").replace("/", "-")
        filename = (
            f"{sanitized}_Collection_{year_part}_{quarter_part}_{safe_interval}.md"
        )

        # Generate markdown content
        markdown_content = self._generate_markdown_report(
            recommendations, risk_tolerance, quarter_part, year_part, collection_name
        )

        # Save to markdown file
        output_path = exports_dir / filename
        with output_path.open("w", encoding="utf-8") as f:
            f.write(markdown_content)

        # Also provide a CSV export for analysts
        try:
            import pandas as _pd

            rows = []
            for rec in recommendations:
                rows.append(
                    {
                        "Symbol": rec.symbol,
                        "Strategy": rec.strategy,
                        "Timeframe": rec.timeframe,
                        "Allocation_Pct": rec.allocation_percentage,
                        "Risk_Level": rec.risk_level,
                        "Confidence": rec.confidence,
                        "Sortino": rec.sortino_ratio,
                        "Calmar": rec.calmar_ratio,
                        "Max_Drawdown_Pct": rec.max_drawdown,
                        "Sharpe(approx)": rec.sharpe_ratio,
                        "Total_Return_Pct": rec.total_return,
                        "Trading_Style": rec.trading_style,
                        "Risk_Per_Trade_Pct": rec.risk_per_trade,
                        "Position_Size_Pct": rec.position_size,
                        "Stop_Loss_Points": rec.stop_loss,
                        "Take_Profit_Points": rec.take_profit,
                    }
                )
            df = _pd.DataFrame(rows)
            csv_filename = filename.replace(".md", ".csv")
            df.to_csv(exports_dir / csv_filename, index=False)
            self.logger.info(
                "AI recommendations CSV saved to %s", exports_dir / csv_filename
            )
        except Exception as _e:
            self.logger.debug("Could not write AI CSV export: %s", _e)

        self.logger.info("AI recommendations saved to %s", output_path)

    def _generate_markdown_report(
        self,
        recommendations: list[AssetRecommendation],
        risk_tolerance: str,
        quarter: str,
        year: str,
        collection_name: str,
    ) -> str:
        """Generate markdown report for AI recommendations."""
        from datetime import datetime

        # Header
        markdown = f"""# AI Investment Recommendations: {collection_name.title()} Collection

## Summary
- **Collection**: {collection_name.title()}
- **Quarter**: {quarter} {year}
- **Risk Tolerance**: {risk_tolerance.title()}
- **Total Recommendations**: {len(recommendations)}
- **Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

"""

        if not recommendations:
            markdown += """## No Recommendations Available

No backtested assets found in the portfolio. Only assets with backtest or optimization data are analyzed.

### Warnings:
- Portfolio contains no backtested assets

"""
            return markdown

        # Recommendations section
        markdown += "## Top Recommendations\n\n"

        for i, rec in enumerate(recommendations, 1):
            # Format risk level with appropriate emoji
            risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(
                rec.risk_level, "⚪"
            )

            markdown += f"""### {i}. {rec.symbol} - {rec.strategy}

**Allocation**: {rec.allocation_percentage:.1f}% | **Risk Level**: {risk_emoji} {rec.risk_level} | **Confidence**: {rec.confidence_score:.1f}%

#### Performance Metrics
| Metric | Value |
|--------|-------|
| Sortino Ratio | {rec.sortino_ratio:.3f} |
| Calmar Ratio | {rec.calmar_ratio:.3f} |
| Max Drawdown | {rec.max_drawdown:.2f}% |
| Win Rate | {rec.win_rate:.1f}% |
| Profit Factor | {rec.profit_factor:.2f} |

#### Analysis
{rec.reasoning}

"""
            if rec.red_flags:
                markdown += f"""#### ⚠️ Risk Factors
{chr(10).join(f"- {flag}" for flag in rec.red_flags)}

"""

            markdown += "---\n\n"

        # Footer
        markdown += f"""## Disclaimer

This analysis is for educational purposes only and should not be considered as financial advice.
Past performance does not guarantee future results. Always conduct your own research and consider
your risk tolerance before making investment decisions.

**Generated by**: Quant System AI Recommendations
**Report Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

        return markdown

    def _save_to_database(
        self,
        portfolio_rec: PortfolioRecommendation,
        quarter: str,
        portfolio_name: Optional[str] = None,
    ):
        """Save AI recommendations to PostgreSQL database using normalized structure."""
        if not self.db_session:
            self.logger.warning("No database session - skipping database save")
            return

        from datetime import datetime

        # Determine which LLM model was used
        llm_model = "unknown"
        if os.getenv("OPENAI_API_KEY"):
            llm_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        elif os.getenv("ANTHROPIC_API_KEY"):
            llm_model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

        # Parse quarter and year
        quarter_str = (
            quarter or f"Q{(datetime.now().month - 1) // 3 + 1}_{datetime.now().year}"
        )
        if "_" in quarter_str:
            q_part, year_part = quarter_str.split("_")
            year = int(year_part)
            quarter_only = q_part
        else:
            year = datetime.now().year
            quarter_only = quarter_str

        # Calculate portfolio-level metrics with type conversion
        total_return = self._ensure_python_type(
            sum(
                rec.allocation_percentage * rec.total_return
                for rec in portfolio_rec.recommendations
            )
            / 100
        )
        portfolio_risk = self._ensure_python_type(
            sum(
                rec.allocation_percentage * rec.max_drawdown
                for rec in portfolio_rec.recommendations
            )
            / 100
        )

        try:
            # Check if recommendation already exists (unique constraint check)
            existing_rec = (
                self.db_session.query(AIRecommendation)
                .filter_by(
                    portfolio_name=portfolio_name or "default",
                    quarter=quarter_only,
                    year=year,
                    risk_tolerance=portfolio_rec.risk_profile,
                )
                .first()
            )

            if existing_rec:
                # Update existing record
                existing_rec.total_score = self._ensure_python_type(
                    portfolio_rec.total_score
                )
                existing_rec.confidence = self._ensure_python_type(
                    portfolio_rec.confidence
                )
                existing_rec.diversification_score = self._ensure_python_type(
                    portfolio_rec.diversification_score
                )
                existing_rec.total_assets = len(portfolio_rec.recommendations)
                existing_rec.expected_return = total_return
                existing_rec.portfolio_risk = portfolio_risk
                existing_rec.overall_reasoning = portfolio_rec.overall_reasoning
                existing_rec.warnings = self._ensure_python_type(portfolio_rec.warnings)
                existing_rec.correlation_analysis = self._ensure_python_type(
                    portfolio_rec.correlation_analysis
                )
                existing_rec.llm_model = llm_model
                ai_rec = existing_rec
                self.logger.info("Updated existing AI recommendation record")
            else:
                # Create new AI recommendation record matching database schema
                ai_rec = AIRecommendation(
                    portfolio_name=portfolio_name or "default",
                    quarter=quarter_only,
                    year=year,
                    risk_tolerance=portfolio_rec.risk_profile,
                    total_score=self._ensure_python_type(portfolio_rec.total_score),
                    confidence=self._ensure_python_type(portfolio_rec.confidence),
                    diversification_score=self._ensure_python_type(
                        portfolio_rec.diversification_score
                    ),
                    total_assets=len(portfolio_rec.recommendations),
                    expected_return=total_return,
                    portfolio_risk=portfolio_risk,
                    overall_reasoning=portfolio_rec.overall_reasoning,
                    warnings=self._ensure_python_type(portfolio_rec.warnings),
                    correlation_analysis=self._ensure_python_type(
                        portfolio_rec.correlation_analysis
                    ),
                    llm_model=llm_model,
                )

                self.db_session.add(ai_rec)
                self.logger.info("Created new AI recommendation record")

            self.db_session.flush()  # Get/Update the ID

            # Create individual asset recommendation records using manual conversion
            for rec in portfolio_rec.recommendations:
                # Convert to plain dict to avoid dataclass numpy issues
                # Ultimate safety conversion - manually check each field
                def force_native_type(val):
                    """Forcefully convert to native Python type."""
                    if val is None:
                        return None
                    val_str = str(type(val))
                    if "numpy" in val_str:
                        return float(val)
                    return val

                # Create asset recommendation with only fields that exist in database model
                asset_rec = DbAssetRecommendation(
                    ai_recommendation_id=ai_rec.id,
                    symbol=rec.symbol,
                    recommendation_type=rec.recommendation_type,  # BUY/SELL/HOLD
                    confidence_score=force_native_type(rec.confidence),
                    reasoning=rec.reasoning,
                )
                self.db_session.add(asset_rec)

            self.db_session.commit()
            self.logger.info(
                "AI recommendations saved to database: %s_%s, %s",
                quarter_only,
                year,
                portfolio_rec.risk_profile,
            )

        except Exception as e:
            self.db_session.rollback()
            self.logger.error("Failed to save AI recommendations to database: %s", e)
            raise
