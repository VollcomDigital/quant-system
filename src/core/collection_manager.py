"""
Portfolio Manager - Handles portfolio comparison and investment prioritization.
Provides comprehensive portfolio analysis and investment recommendations.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np

from .backtest_engine import BacktestResult
from .result_analyzer import UnifiedResultAnalyzer

warnings.filterwarnings("ignore")


@dataclass
class PortfolioSummary:
    """Summary statistics for a portfolio."""

    name: str
    total_assets: int
    total_strategies: int
    best_performer: str
    worst_performer: str
    avg_return: float
    avg_sharpe: float
    max_drawdown: float
    risk_score: float
    return_score: float
    overall_score: float
    investment_priority: int
    recommended_allocation: float
    risk_category: str  # 'Conservative', 'Moderate', 'Aggressive'


@dataclass
class InvestmentRecommendation:
    """Investment recommendation for a portfolio."""

    portfolio_name: str
    priority_rank: int
    recommended_allocation_pct: float
    expected_annual_return: float
    expected_volatility: float
    max_drawdown_risk: float
    confidence_score: float
    risk_category: str
    investment_rationale: str
    key_strengths: list[str]
    key_risks: list[str]
    minimum_investment_period: str


class PortfolioManager:
    """
    Portfolio Manager for comparing portfolios and providing investment prioritization.
    Analyzes multiple portfolios and provides investment recommendations.
    """

    def __init__(self):
        self.result_analyzer = UnifiedResultAnalyzer()
        self.logger = logging.getLogger(__name__)

        # Risk scoring weights
        self.risk_weights = {
            "max_drawdown": 0.3,
            "volatility": 0.25,
            "var_95": 0.2,
            "sharpe_ratio": 0.15,  # Higher is better
            "sortino_ratio": 0.1,  # Higher is better
        }

        # Return scoring weights
        self.return_weights = {
            "total_return": 0.4,
            "annualized_return": 0.3,
            "sharpe_ratio": 0.2,
            "win_rate": 0.1,
        }

    def analyze_portfolios(
        self, portfolios: dict[str, list[BacktestResult]]
    ) -> dict[str, Any]:
        """
        Analyze multiple portfolios and generate comprehensive comparison.

        Args:
            portfolios: Dictionary mapping portfolio names to lists of BacktestResults

        Returns:
            Comprehensive portfolio analysis
        """
        self.logger.info("Analyzing %s portfolios...", len(portfolios))

        portfolio_summaries = {}
        detailed_analysis = {}

        # Analyze each portfolio
        for portfolio_name, results in portfolios.items():
            self.logger.info("Analyzing portfolio: %s", portfolio_name)

            # Calculate portfolio summary
            summary = self._calculate_portfolio_summary(portfolio_name, results)
            portfolio_summaries[portfolio_name] = summary

            # Calculate detailed metrics
            detailed_metrics = self._calculate_detailed_metrics(results)
            detailed_analysis[portfolio_name] = detailed_metrics

        # Rank portfolios and generate recommendations
        ranked_portfolios = self._rank_portfolios(portfolio_summaries)
        investment_recommendations = self._generate_investment_recommendations(
            ranked_portfolios, detailed_analysis
        )

        # Generate overall analysis
        return {
            "analysis_date": datetime.now().isoformat(),
            "portfolios_analyzed": len(portfolios),
            "portfolio_summaries": {
                name: asdict(summary) for name, summary in portfolio_summaries.items()
            },
            "detailed_analysis": detailed_analysis,
            "ranked_portfolios": ranked_portfolios,
            "investment_recommendations": [
                asdict(rec) for rec in investment_recommendations
            ],
            "market_analysis": self._generate_market_analysis(portfolio_summaries),
            "risk_analysis": self._generate_risk_analysis(portfolio_summaries),
            "diversification_analysis": self._analyze_diversification_opportunities(
                portfolios
            ),
        }

    def generate_investment_plan(
        self,
        total_capital: float,
        portfolios: dict[str, list[BacktestResult]],
        risk_tolerance: str = "moderate",
    ) -> dict[str, Any]:
        """
        Generate specific investment plan with capital allocation.

        Args:
            total_capital: Total capital to allocate
            portfolios: Portfolio analysis results
            risk_tolerance: 'conservative', 'moderate', 'aggressive'

        Returns:
            Detailed investment plan
        """
        self.logger.info(
            "Generating investment plan for $%.2f with %s risk tolerance",
            total_capital,
            risk_tolerance,
        )

        # Analyze portfolios
        analysis = self.analyze_portfolios(portfolios)
        recommendations = analysis["investment_recommendations"]

        # Filter recommendations based on risk tolerance
        suitable_recommendations = self._filter_by_risk_tolerance(
            recommendations, risk_tolerance
        )

        # Calculate allocations
        allocations = self._calculate_capital_allocations(
            suitable_recommendations, total_capital, risk_tolerance
        )

        # Generate implementation timeline
        implementation_plan = self._generate_implementation_plan(allocations)

        # Risk management plan
        risk_management = self._generate_risk_management_plan(allocations, analysis)

        return {
            "plan_date": datetime.now().isoformat(),
            "total_capital": total_capital,
            "risk_tolerance": risk_tolerance,
            "allocations": allocations,
            "implementation_plan": implementation_plan,
            "risk_management": risk_management,
            "expected_portfolio_metrics": self._calculate_expected_portfolio_metrics(
                allocations
            ),
            "monitoring_recommendations": self._generate_monitoring_recommendations(),
            "rebalancing_strategy": self._generate_rebalancing_strategy(allocations),
        }

    def _calculate_portfolio_summary(
        self, name: str, results: list[BacktestResult]
    ) -> PortfolioSummary:
        """Calculate summary statistics for a portfolio."""
        if not results:
            return PortfolioSummary(
                name=name,
                total_assets=0,
                total_strategies=0,
                best_performer="N/A",
                worst_performer="N/A",
                avg_return=0,
                avg_sharpe=0,
                max_drawdown=0,
                risk_score=0,
                return_score=0,
                overall_score=0,
                investment_priority=999,
                recommended_allocation=0,
                risk_category="Unknown",
            )

        # Filter successful results
        successful_results = [r for r in results if not r.error and r.metrics]

        if not successful_results:
            return PortfolioSummary(
                name=name,
                total_assets=len(results),
                total_strategies=0,
                best_performer="N/A",
                worst_performer="N/A",
                avg_return=0,
                avg_sharpe=0,
                max_drawdown=0,
                risk_score=0,
                return_score=0,
                overall_score=0,
                investment_priority=999,
                recommended_allocation=0,
                risk_category="High Risk",
            )

        # Extract metrics
        returns = [r.metrics.get("total_return", 0) for r in successful_results]
        sharpes = [r.metrics.get("sharpe_ratio", 0) for r in successful_results]
        drawdowns = [r.metrics.get("max_drawdown", 0) for r in successful_results]

        # Find best and worst performers
        best_idx = np.argmax(returns)
        worst_idx = np.argmin(returns)

        best_performer = f"{successful_results[best_idx].symbol}/{successful_results[best_idx].strategy}"
        worst_performer = f"{successful_results[worst_idx].symbol}/{successful_results[worst_idx].strategy}"

        # Calculate scores
        risk_score = self._calculate_risk_score(successful_results)
        return_score = self._calculate_return_score(successful_results)
        overall_score = (return_score * 0.6) + (
            risk_score * 0.4
        )  # Weight returns higher

        # Determine risk category
        risk_category = self._determine_risk_category(
            risk_score, np.mean(drawdowns), np.std(returns)
        )

        return PortfolioSummary(
            name=name,
            total_assets=len(set(r.symbol for r in results)),
            total_strategies=len(set(r.strategy for r in results)),
            best_performer=best_performer,
            worst_performer=worst_performer,
            avg_return=np.mean(returns),
            avg_sharpe=np.mean(sharpes),
            max_drawdown=np.mean(drawdowns),
            risk_score=risk_score,
            return_score=return_score,
            overall_score=overall_score,
            investment_priority=0,  # Will be set during ranking
            recommended_allocation=0,  # Will be calculated later
            risk_category=risk_category,
        )

    def _calculate_detailed_metrics(
        self, results: list[BacktestResult]
    ) -> dict[str, Any]:
        """Calculate detailed metrics for a portfolio."""
        successful_results = [r for r in results if not r.error and r.metrics]

        if not successful_results:
            return {}

        # Aggregate all metrics
        all_metrics = {}
        metric_names = set()
        for result in successful_results:
            metric_names.update(result.metrics.keys())

        for metric in metric_names:
            values = [
                r.metrics.get(metric, 0)
                for r in successful_results
                if metric in r.metrics
            ]
            if values:
                all_metrics[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values),
                    "count": len(values),
                }

        # Strategy analysis
        strategy_performance = {}
        for strategy in set(r.strategy for r in successful_results):
            strategy_results = [r for r in successful_results if r.strategy == strategy]
            strategy_returns = [
                r.metrics.get("total_return", 0) for r in strategy_results
            ]

            strategy_performance[strategy] = {
                "count": len(strategy_results),
                "avg_return": np.mean(strategy_returns),
                "success_rate": len([r for r in strategy_returns if r > 0])
                / len(strategy_returns)
                * 100,
                "best_return": np.max(strategy_returns),
                "worst_return": np.min(strategy_returns),
            }

        # Asset analysis
        asset_performance = {}
        for symbol in set(r.symbol for r in successful_results):
            symbol_results = [r for r in successful_results if r.symbol == symbol]
            symbol_returns = [r.metrics.get("total_return", 0) for r in symbol_results]

            asset_performance[symbol] = {
                "count": len(symbol_results),
                "avg_return": np.mean(symbol_returns),
                "consistency": (
                    1 - (np.std(symbol_returns) / np.mean(symbol_returns))
                    if np.mean(symbol_returns) != 0
                    else 0
                ),
                "best_strategy": max(
                    symbol_results, key=lambda x: x.metrics.get("total_return", 0)
                ).strategy,
            }

        return {
            "summary_metrics": all_metrics,
            "strategy_performance": strategy_performance,
            "asset_performance": asset_performance,
            "total_combinations": len(results),
            "successful_combinations": len(successful_results),
            "success_rate": (
                len(successful_results) / len(results) * 100 if results else 0
            ),
        }

    def _rank_portfolios(
        self, summaries: dict[str, PortfolioSummary]
    ) -> list[tuple[str, PortfolioSummary]]:
        """Rank portfolios by overall score."""
        # Sort by overall score (descending)
        ranked = sorted(
            summaries.items(), key=lambda x: x[1].overall_score, reverse=True
        )

        # Update priority rankings
        for i, (_name, summary) in enumerate(ranked):
            summary.investment_priority = i + 1

        return ranked

    def _generate_investment_recommendations(
        self,
        ranked_portfolios: list[tuple[str, PortfolioSummary]],
        detailed_analysis: dict[str, Any],
    ) -> list[InvestmentRecommendation]:
        """Generate investment recommendations for each portfolio."""
        recommendations = []
        total_score = sum(summary.overall_score for _, summary in ranked_portfolios)

        for i, (name, summary) in enumerate(ranked_portfolios):
            # Calculate recommended allocation based on score
            if total_score > 0:
                base_allocation = (summary.overall_score / total_score) * 100
            else:
                base_allocation = 100 / len(ranked_portfolios)

            # Adjust allocation based on risk category
            risk_adjustment = self._get_risk_adjustment(summary.risk_category)
            recommended_allocation = min(
                base_allocation * risk_adjustment, 40
            )  # Cap at 40%

            # Generate rationale and key points
            rationale = self._generate_investment_rationale(
                summary, detailed_analysis.get(name, {})
            )
            strengths = self._identify_key_strengths(
                summary, detailed_analysis.get(name, {})
            )
            risks = self._identify_key_risks(summary, detailed_analysis.get(name, {}))

            # Calculate confidence score
            confidence = self._calculate_confidence_score(
                summary, detailed_analysis.get(name, {})
            )

            recommendation = InvestmentRecommendation(
                portfolio_name=name,
                priority_rank=i + 1,
                recommended_allocation_pct=recommended_allocation,
                expected_annual_return=summary.avg_return,
                expected_volatility=self._estimate_volatility(
                    detailed_analysis.get(name, {})
                ),
                max_drawdown_risk=abs(summary.max_drawdown),
                confidence_score=confidence,
                risk_category=summary.risk_category,
                investment_rationale=rationale,
                key_strengths=strengths,
                key_risks=risks,
                minimum_investment_period=self._recommend_investment_period(
                    summary.risk_category
                ),
            )

            recommendations.append(recommendation)

        return recommendations

    def _calculate_risk_score(self, results: list[BacktestResult]) -> float:
        """Calculate risk score for portfolio (0-100, higher is better)."""
        risk_metrics = []

        for result in results:
            metrics = result.metrics

            # Individual risk components (normalized to 0-100)
            max_dd = abs(metrics.get("max_drawdown", 0))
            volatility = metrics.get("volatility", 0)
            var_95 = abs(metrics.get("var_95", 0))
            sharpe = metrics.get("sharpe_ratio", 0)
            sortino = metrics.get("sortino_ratio", 0)

            # Convert to scores (lower risk = higher score)
            dd_score = max(0, 100 - max_dd * 2)  # Max drawdown penalty
            vol_score = max(0, 100 - volatility)  # Volatility penalty
            var_score = max(0, 100 - var_95 * 10)  # VaR penalty
            sharpe_score = min(100, sharpe * 20)  # Sharpe bonus
            sortino_score = min(100, sortino * 20)  # Sortino bonus

            # Weighted combination
            risk_score = (
                dd_score * self.risk_weights["max_drawdown"]
                + vol_score * self.risk_weights["volatility"]
                + var_score * self.risk_weights["var_95"]
                + sharpe_score * self.risk_weights["sharpe_ratio"]
                + sortino_score * self.risk_weights["sortino_ratio"]
            )

            risk_metrics.append(risk_score)

        return np.mean(risk_metrics) if risk_metrics else 0

    def _calculate_return_score(self, results: list[BacktestResult]) -> float:
        """Calculate return score for portfolio (0-100, higher is better)."""
        return_metrics = []

        for result in results:
            metrics = result.metrics

            # Individual return components
            total_return = metrics.get("total_return", 0)
            annual_return = metrics.get("annualized_return", 0)
            sharpe = metrics.get("sharpe_ratio", 0)
            win_rate = metrics.get("win_rate", 0)

            # Convert to scores
            total_score = min(100, max(0, total_return))  # Cap at 100%
            annual_score = min(100, max(0, annual_return * 2))  # Scale annual return
            sharpe_score = min(100, sharpe * 20)  # Sharpe bonus
            win_score = win_rate  # Already in percentage

            # Weighted combination
            return_score = (
                total_score * self.return_weights["total_return"]
                + annual_score * self.return_weights["annualized_return"]
                + sharpe_score * self.return_weights["sharpe_ratio"]
                + win_score * self.return_weights["win_rate"]
            )

            return_metrics.append(return_score)

        return np.mean(return_metrics) if return_metrics else 0

    def _determine_risk_category(
        self, risk_score: float, avg_drawdown: float, return_volatility: float
    ) -> str:
        """Determine risk category based on metrics."""
        if risk_score >= 70 and abs(avg_drawdown) <= 10 and return_volatility <= 15:
            return "Conservative"
        if risk_score >= 50 and abs(avg_drawdown) <= 20 and return_volatility <= 25:
            return "Moderate"
        return "Aggressive"

    def _generate_market_analysis(
        self, summaries: dict[str, PortfolioSummary]
    ) -> dict[str, Any]:
        """Generate overall market analysis."""
        if not summaries:
            return {}

        all_returns = [s.avg_return for s in summaries.values()]
        all_sharpes = [s.avg_sharpe for s in summaries.values()]

        return {
            "market_sentiment": (
                "Bullish"
                if np.mean(all_returns) > 5
                else "Bearish"
                if np.mean(all_returns) < -2
                else "Neutral"
            ),
            "average_market_return": np.mean(all_returns),
            "market_volatility": np.std(all_returns),
            "risk_adjusted_performance": np.mean(all_sharpes),
            "top_performing_category": max(
                summaries.keys(), key=lambda k: summaries[k].avg_return
            ),
            "most_consistent_category": max(
                summaries.keys(), key=lambda k: summaries[k].avg_sharpe
            ),
            "recommendations": self._generate_market_recommendations(summaries),
        }

    def _generate_risk_analysis(
        self, summaries: dict[str, PortfolioSummary]
    ) -> dict[str, Any]:
        """Generate risk analysis across portfolios."""
        risk_categories = {}
        for summary in summaries.values():
            category = summary.risk_category
            if category not in risk_categories:
                risk_categories[category] = []
            risk_categories[category].append(summary)

        risk_analysis = {}
        for category, portfolios in risk_categories.items():
            risk_analysis[category] = {
                "count": len(portfolios),
                "avg_return": np.mean([p.avg_return for p in portfolios]),
                "avg_risk_score": np.mean([p.risk_score for p in portfolios]),
                "recommended_allocation": self._get_category_allocation(category),
                "portfolios": [p.name for p in portfolios],
            }

        return {
            "by_category": risk_analysis,
            "overall_risk_level": self._assess_overall_risk_level(summaries),
            "diversification_score": self._calculate_diversification_score(summaries),
            "risk_recommendations": self._generate_risk_recommendations(risk_analysis),
        }

    def _analyze_diversification_opportunities(
        self, portfolios: dict[str, list[BacktestResult]]
    ) -> dict[str, Any]:
        """Analyze diversification opportunities across portfolios."""
        # Asset type analysis
        all_symbols = set()
        all_strategies = set()
        portfolio_overlap = {}

        for name, results in portfolios.items():
            symbols = set(r.symbol for r in results)
            strategies = set(r.strategy for r in results)

            all_symbols.update(symbols)
            all_strategies.update(strategies)

            portfolio_overlap[name] = {
                "symbols": symbols,
                "strategies": strategies,
                "asset_types": self._classify_asset_types(symbols),
            }

        # Calculate overlaps
        overlap_analysis = {}
        portfolio_names = list(portfolio_overlap.keys())

        for i, name1 in enumerate(portfolio_names):
            for name2 in portfolio_names[i + 1 :]:
                symbols1 = portfolio_overlap[name1]["symbols"]
                symbols2 = portfolio_overlap[name2]["symbols"]

                overlap = len(symbols1.intersection(symbols2))
                total_unique = len(symbols1.union(symbols2))

                overlap_analysis[f"{name1}_vs_{name2}"] = {
                    "symbol_overlap": overlap,
                    "total_symbols": total_unique,
                    "overlap_percentage": (
                        (overlap / total_unique * 100) if total_unique > 0 else 0
                    ),
                }

        return {
            "total_unique_symbols": len(all_symbols),
            "total_unique_strategies": len(all_strategies),
            "portfolio_overlaps": overlap_analysis,
            "diversification_opportunities": self._identify_diversification_gaps(
                portfolio_overlap
            ),
            "recommended_portfolio_mix": self._recommend_portfolio_mix(
                portfolio_overlap
            ),
        }

    def _filter_by_risk_tolerance(
        self, recommendations: list[dict], risk_tolerance: str
    ) -> list[dict]:
        """Filter recommendations based on risk tolerance."""
        risk_mapping = {
            "conservative": ["Conservative"],
            "moderate": ["Conservative", "Moderate"],
            "aggressive": ["Conservative", "Moderate", "Aggressive"],
        }

        allowed_categories = risk_mapping.get(
            risk_tolerance, ["Conservative", "Moderate"]
        )

        return [
            rec for rec in recommendations if rec["risk_category"] in allowed_categories
        ]

    def _calculate_capital_allocations(
        self, recommendations: list[dict], total_capital: float, risk_tolerance: str
    ) -> list[dict]:
        """Calculate specific capital allocations."""
        if not recommendations:
            return []

        # Adjust allocations based on risk tolerance
        risk_multipliers = {
            "conservative": {"Conservative": 1.5, "Moderate": 0.5, "Aggressive": 0.1},
            "moderate": {"Conservative": 1.0, "Moderate": 1.2, "Aggressive": 0.8},
            "aggressive": {"Conservative": 0.7, "Moderate": 1.0, "Aggressive": 1.3},
        }

        multipliers = risk_multipliers.get(risk_tolerance, risk_multipliers["moderate"])

        # Apply multipliers
        adjusted_allocations = []
        for rec in recommendations:
            adjusted_pct = rec["recommended_allocation_pct"] * multipliers.get(
                rec["risk_category"], 1.0
            )
            adjusted_allocations.append(adjusted_pct)

        # Normalize to 100%
        total_adjusted = sum(adjusted_allocations)
        if total_adjusted > 0:
            normalized_allocations = [
                pct / total_adjusted * 100 for pct in adjusted_allocations
            ]
        else:
            normalized_allocations = [100 / len(recommendations)] * len(recommendations)

        # Calculate dollar amounts
        allocations = []
        for i, rec in enumerate(recommendations):
            allocation_pct = normalized_allocations[i]
            allocation_amount = total_capital * (allocation_pct / 100)

            allocations.append(
                {
                    "portfolio_name": rec["portfolio_name"],
                    "allocation_percentage": allocation_pct,
                    "allocation_amount": allocation_amount,
                    "priority_rank": rec["priority_rank"],
                    "risk_category": rec["risk_category"],
                    "expected_return": rec["expected_annual_return"],
                }
            )

        return allocations

    def _generate_implementation_plan(self, allocations: list[dict]) -> dict[str, Any]:
        """Generate implementation timeline."""
        # Sort by priority
        sorted_allocations = sorted(allocations, key=lambda x: x["priority_rank"])

        implementation_phases = []
        cumulative_allocation = 0

        for i, allocation in enumerate(sorted_allocations):
            phase_start = i * 2  # 2 weeks between phases
            phase_end = phase_start + 1

            cumulative_allocation += allocation["allocation_percentage"]

            implementation_phases.append(
                {
                    "phase": i + 1,
                    "week_start": phase_start,
                    "week_end": phase_end,
                    "portfolio": allocation["portfolio_name"],
                    "amount": allocation["allocation_amount"],
                    "percentage": allocation["allocation_percentage"],
                    "cumulative_percentage": cumulative_allocation,
                    "priority": allocation["priority_rank"],
                }
            )

        return {
            "total_phases": len(implementation_phases),
            "estimated_duration_weeks": len(implementation_phases) * 2,
            "phases": implementation_phases,
            "risk_management_notes": [
                "Start with highest-ranked portfolios",
                "Monitor performance after each phase",
                "Adjust subsequent allocations based on early results",
                "Maintain 5-10% cash reserve for opportunities",
            ],
        }

    def _generate_risk_management_plan(
        self, allocations: list[dict], analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate risk management plan."""
        sum(a["allocation_amount"] for a in allocations)

        # Calculate portfolio risk metrics
        sum(
            a["expected_return"] * a["allocation_percentage"] / 100 for a in allocations
        )

        return {
            "portfolio_limits": {
                "max_single_portfolio_pct": 40,
                "max_aggressive_allocation_pct": 30,
                "min_conservative_allocation_pct": 20,
            },
            "stop_loss_rules": {
                "individual_portfolio_stop_loss": -15,  # %
                "total_portfolio_stop_loss": -10,  # %
                "review_trigger": -5,  # %
            },
            "rebalancing_triggers": {
                "time_based": "Quarterly",
                "drift_threshold": 5,  # % deviation from target
                "performance_threshold": 10,  # % underperformance
            },
            "monitoring_schedule": {
                "daily": ["Market conditions", "Major news events"],
                "weekly": ["Portfolio performance", "Risk metrics"],
                "monthly": ["Full portfolio review", "Rebalancing assessment"],
                "quarterly": ["Strategy review", "Allocation adjustments"],
            },
            "risk_metrics_targets": {
                "max_portfolio_volatility": 20,
                "target_sharpe_ratio": 1.0,
                "max_correlation_single_asset": 0.3,
            },
        }

    def _calculate_expected_portfolio_metrics(
        self, allocations: list[dict]
    ) -> dict[str, float]:
        """Calculate expected metrics for the combined portfolio."""
        if not allocations:
            return {}

        # Weighted calculations
        weights = [a["allocation_percentage"] / 100 for a in allocations]
        returns = [a["expected_return"] for a in allocations]

        expected_return = sum(w * r for w, r in zip(weights, returns))

        # Simplified risk calculation (would need correlation matrix for full calculation)
        portfolio_volatility = np.sqrt(
            sum(w**2 * (r * 0.5) ** 2 for w, r in zip(weights, returns))
        )

        return {
            "expected_annual_return": expected_return,
            "expected_volatility": portfolio_volatility,
            "expected_sharpe_ratio": (
                expected_return / portfolio_volatility
                if portfolio_volatility > 0
                else 0
            ),
            "diversification_benefit": len(allocations) / 10,  # Simplified
            "risk_score": sum(w * (100 - abs(r)) for w, r in zip(weights, returns)),
        }

    # Helper methods for various calculations...
    def _get_risk_adjustment(self, risk_category: str) -> float:
        """Get risk adjustment multiplier."""
        return {"Conservative": 1.2, "Moderate": 1.0, "Aggressive": 0.8}.get(
            risk_category, 1.0
        )

    def _estimate_volatility(self, detailed_analysis: dict) -> float:
        """Estimate portfolio volatility."""
        if not detailed_analysis or "summary_metrics" not in detailed_analysis:
            return 20.0  # Default estimate

        volatility_data = detailed_analysis["summary_metrics"].get("volatility", {})
        return volatility_data.get("mean", 20.0)

    def _generate_investment_rationale(
        self, summary: PortfolioSummary, detailed_analysis: dict
    ) -> str:
        """Generate investment rationale."""
        if summary.overall_score >= 70:
            return (
                f"Strong performer with {summary.avg_return}% average return and "
                f"{summary.risk_category.lower()} risk profile."
            )
        if summary.overall_score >= 50:
            return "Solid performer with balanced risk-return profile suitable for diversified portfolios."
        return "Higher risk option that may be suitable for aggressive investors seeking potential upside."

    def _identify_key_strengths(
        self, summary: PortfolioSummary, detailed_analysis: dict
    ) -> list[str]:
        """Identify key strengths."""
        strengths = []

        if summary.avg_return > 10:
            strengths.append(f"High average return of {summary.avg_return}%")
        if summary.avg_sharpe > 1:
            strengths.append(
                f"Strong risk-adjusted returns (Sharpe: {summary.avg_sharpe})"
            )
        if abs(summary.max_drawdown) < 10:
            strengths.append("Low drawdown risk")
        if summary.total_assets > 10:
            strengths.append("Well-diversified across multiple assets")

        return strengths[:3]  # Limit to top 3

    def _identify_key_risks(
        self, summary: PortfolioSummary, detailed_analysis: dict
    ) -> list[str]:
        """Identify key risks."""
        risks = []

        if abs(summary.max_drawdown) > 20:
            risks.append(f"High drawdown risk ({abs(summary.max_drawdown):.1f}%)")
        if summary.avg_sharpe < 0.5:
            risks.append("Poor risk-adjusted returns")
        if summary.total_assets < 5:
            risks.append("Limited diversification")
        if summary.risk_category == "Aggressive":
            risks.append("High volatility and risk")

        return risks[:3]  # Limit to top 3

    def _calculate_confidence_score(
        self, summary: PortfolioSummary, detailed_analysis: dict
    ) -> float:
        """Calculate confidence score."""
        base_score = summary.overall_score

        # Adjust based on data quality
        if detailed_analysis.get("success_rate", 0) > 80:
            base_score *= 1.1
        elif detailed_analysis.get("success_rate", 0) < 50:
            base_score *= 0.9

        # Adjust based on consistency
        if summary.total_assets > 10 and summary.total_strategies > 3:
            base_score *= 1.05

        return min(100, base_score)

    def _recommend_investment_period(self, risk_category: str) -> str:
        """Recommend minimum investment period."""
        return {
            "Conservative": "6-12 months",
            "Moderate": "12-24 months",
            "Aggressive": "24+ months",
        }.get(risk_category, "12-24 months")

    def _generate_monitoring_recommendations(self) -> list[str]:
        """Generate monitoring recommendations."""
        return [
            "Review portfolio performance weekly",
            "Monitor individual strategy performance monthly",
            "Assess correlation changes quarterly",
            "Rebalance when allocation drifts >5% from targets",
            "Consider strategy replacement if underperforming for 6+ months",
        ]

    def _generate_rebalancing_strategy(self, allocations: list[dict]) -> dict[str, Any]:
        """Generate rebalancing strategy."""
        return {
            "frequency": "Quarterly",
            "drift_threshold": 5,  # %
            "method": "Threshold-based with time override",
            "rules": [
                "Rebalance if any allocation drifts >5% from target",
                "Mandatory rebalancing every 6 months regardless of drift",
                "Emergency rebalancing if portfolio loses >10%",
                "Consider tax implications before rebalancing",
            ],
        }

    # Additional helper methods would be implemented here...
    def _generate_market_recommendations(self, summaries: dict) -> list[str]:
        return ["Monitor market conditions", "Consider defensive strategies if needed"]

    def _get_category_allocation(self, category: str) -> float:
        return {"Conservative": 40, "Moderate": 35, "Aggressive": 25}.get(category, 30)

    def _assess_overall_risk_level(self, summaries: dict) -> str:
        avg_risk = np.mean([s.risk_score for s in summaries.values()])
        return "Low" if avg_risk > 70 else "Medium" if avg_risk > 50 else "High"

    def _calculate_diversification_score(self, summaries: dict) -> float:
        total_assets = sum(s.total_assets for s in summaries.values())
        return min(100, total_assets * 2)  # Simplified calculation

    def _generate_risk_recommendations(self, risk_analysis: dict) -> list[str]:
        return [
            "Maintain diversification",
            "Monitor correlation changes",
            "Review risk limits regularly",
        ]

    def _classify_asset_types(self, symbols: set) -> dict[str, int]:
        crypto_count = len(
            [
                s
                for s in symbols
                if any(c in s.upper() for c in ["BTC", "ETH", "USD", "USDT"])
            ]
        )
        forex_count = len([s for s in symbols if s.endswith("=X")])
        stock_count = len(symbols) - crypto_count - forex_count

        return {"stocks": stock_count, "crypto": crypto_count, "forex": forex_count}

    def _identify_diversification_gaps(self, portfolio_overlap: dict) -> list[str]:
        return [
            "Consider adding international exposure",
            "Evaluate sector concentration",
        ]

    def _recommend_portfolio_mix(self, portfolio_overlap: dict) -> dict[str, float]:
        return {"Primary": 60, "Secondary": 25, "Satellite": 15}
