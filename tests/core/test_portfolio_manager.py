"""Unit tests for PortfolioManager."""

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import pytest

from src.core.backtest_engine import BacktestResult
from src.core.collection_manager import PortfolioManager


class TestPortfolioManager:
    """Test cases for PortfolioManager."""

    @pytest.fixture
    def mock_backtest_engine(self):
        """Mock backtest engine."""
        engine = Mock()
        engine.batch_backtest.return_value = []
        return engine

    @pytest.fixture
    def portfolio_manager(self, mock_backtest_engine):
        """Create PortfolioManager instance."""
        return PortfolioManager()

    @pytest.fixture
    def sample_backtest_results(self):
        """Sample backtest results."""
        return [
            BacktestResult(
                symbol="AAPL",
                strategy="rsi",
                parameters={},
                config={},
                error=None,
                metrics={
                    "total_return": 0.15,
                    "annualized_return": 0.12,
                    "sharpe_ratio": 1.2,
                    "sortino_ratio": 1.5,
                    "max_drawdown": -0.08,
                    "volatility": 0.18,
                    "beta": 1.1,
                    "alpha": 0.02,
                    "var_95": -0.05,
                    "cvar_95": -0.07,
                    "calmar_ratio": 1.5,
                    "omega_ratio": 1.3,
                    "win_rate": 64.0,
                    "avg_win": 0.05,
                    "avg_loss": -0.03,
                    "profit_factor": 2.1,
                    "kelly_criterion": 0.15,
                    "num_trades": 25,
                },
                start_date="2023-01-01",
                end_date="2023-12-31",
                duration_seconds=365 * 24 * 3600,
                equity_curve=pd.DataFrame({"equity": [10000, 10500, 11000, 11500]}),
                trades=pd.DataFrame(),
                data_points=365,
            ),
            BacktestResult(
                symbol="MSFT",
                strategy="rsi",
                parameters={},
                config={},
                error=None,
                metrics={
                    "total_return": 0.18,
                    "annualized_return": 0.16,
                    "sharpe_ratio": 1.4,
                    "sortino_ratio": 1.7,
                    "max_drawdown": -0.06,
                    "volatility": 0.16,
                    "beta": 0.9,
                    "alpha": 0.04,
                    "var_95": -0.04,
                    "cvar_95": -0.06,
                    "calmar_ratio": 2.67,
                    "omega_ratio": 1.5,
                    "win_rate": 68.0,
                    "avg_win": 0.06,
                    "avg_loss": -0.025,
                    "profit_factor": 2.4,
                    "kelly_criterion": 0.18,
                    "num_trades": 28,
                },
                start_date="2023-01-01",
                end_date="2023-12-31",
                duration_seconds=365 * 24 * 3600,
                equity_curve=pd.DataFrame({"equity": [10000, 10600, 11200, 11800]}),
                trades=pd.DataFrame(),
                data_points=365,
            ),
        ]

    @pytest.fixture
    def sample_portfolios(self):
        """Sample portfolio configurations."""
        return {
            "tech_growth": {
                "name": "Tech Growth",
                "symbols": ["AAPL", "MSFT", "GOOGL"],
                "strategies": ["rsi", "macd"],
                "risk_profile": "aggressive",
                "target_return": 0.15,
            },
            "conservative": {
                "name": "Conservative Mix",
                "symbols": ["SPY", "BND", "VTI"],
                "strategies": ["sma_crossover"],
                "risk_profile": "conservative",
                "target_return": 0.08,
            },
        }

    def test_init(self, portfolio_manager, mock_backtest_engine):
        """Test initialization."""
        assert hasattr(portfolio_manager, "result_analyzer")
        assert hasattr(portfolio_manager, "logger")
        assert hasattr(portfolio_manager, "risk_weights")
        assert hasattr(portfolio_manager, "return_weights")
        assert isinstance(portfolio_manager.risk_weights, dict)
        assert isinstance(portfolio_manager.return_weights, dict)

    def test_analyze_portfolios(self, portfolio_manager, sample_backtest_results):
        """Test portfolio analysis."""
        portfolios = {
            "conservative": sample_backtest_results[:1],
            "aggressive": sample_backtest_results[1:],
        }

        result = portfolio_manager.analyze_portfolios(portfolios)

        assert isinstance(result, dict)
        assert "portfolio_summaries" in result
        assert "investment_recommendations" in result
        assert "market_analysis" in result
        assert "risk_analysis" in result
        assert "diversification_analysis" in result
        assert "detailed_analysis" in result
        assert "ranked_portfolios" in result

    def test_generate_investment_plan(self, portfolio_manager, sample_backtest_results):
        """Test investment plan generation."""
        portfolios = {
            "conservative": sample_backtest_results[:1],
            "aggressive": sample_backtest_results[1:],
        }

        plan = portfolio_manager.generate_investment_plan(
            total_capital=100000, portfolios=portfolios, risk_tolerance="moderate"
        )

        assert isinstance(plan, dict)
        assert "total_capital" in plan
        assert "risk_tolerance" in plan
        assert "allocations" in plan
        assert "implementation_plan" in plan
        assert "risk_management" in plan
        assert "expected_portfolio_metrics" in plan
        assert "monitoring_recommendations" in plan
        assert "rebalancing_strategy" in plan

    def test_risk_weights_configuration(self, portfolio_manager):
        """Test that risk weights are properly configured."""
        assert isinstance(portfolio_manager.risk_weights, dict)
        assert "max_drawdown" in portfolio_manager.risk_weights
        assert "volatility" in portfolio_manager.risk_weights
        assert "var_95" in portfolio_manager.risk_weights
        assert "sharpe_ratio" in portfolio_manager.risk_weights
        assert "sortino_ratio" in portfolio_manager.risk_weights

        # Check weights sum to 1.0
        total_weight = sum(portfolio_manager.risk_weights.values())
        assert total_weight == pytest.approx(1.0, rel=1e-2)

    def test_return_weights_configuration(self, portfolio_manager):
        """Test that return weights are properly configured."""
        assert isinstance(portfolio_manager.return_weights, dict)
        assert "total_return" in portfolio_manager.return_weights
        assert "annualized_return" in portfolio_manager.return_weights
        assert "sharpe_ratio" in portfolio_manager.return_weights
        assert "win_rate" in portfolio_manager.return_weights

        # Check weights sum to 1.0
        total_weight = sum(portfolio_manager.return_weights.values())
        assert total_weight == pytest.approx(1.0, rel=1e-2)

    def test_calculate_risk_score(self, portfolio_manager, sample_backtest_results):
        """Test risk score calculation."""
        risk_score = portfolio_manager._calculate_risk_score(sample_backtest_results)

        assert isinstance(risk_score, float)
        assert 0 <= risk_score <= 100
