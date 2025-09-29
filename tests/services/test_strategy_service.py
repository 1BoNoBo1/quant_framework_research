"""
Tests for Strategy Service
=========================

Suite de tests complète pour le service de gestion des stratégies.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional

from qframe.api.services.strategy_service import StrategyService
from qframe.domain.entities.strategy import Strategy, StrategyStatus, StrategyType
from qframe.domain.entities.backtest import BacktestResult, BacktestStatus
from qframe.domain.entities.portfolio import Portfolio
from qframe.domain.repositories.strategy_repository import StrategyRepository
from qframe.domain.repositories.backtest_repository import BacktestRepository
from qframe.domain.repositories.portfolio_repository import PortfolioRepository
from qframe.domain.services.backtesting_service import BacktestingService
from qframe.core.interfaces import MetricsCollector


@pytest.fixture
def mock_strategy_repository():
    """Repository de stratégies mocké."""
    return Mock(spec=StrategyRepository)


@pytest.fixture
def mock_backtest_repository():
    """Repository de backtests mocké."""
    return Mock(spec=BacktestRepository)


@pytest.fixture
def mock_portfolio_repository():
    """Repository de portfolios mocké."""
    return Mock(spec=PortfolioRepository)


@pytest.fixture
def mock_backtesting_service():
    """Service de backtesting mocké."""
    return Mock(spec=BacktestingService)


@pytest.fixture
def mock_metrics_collector():
    """Collecteur de métriques mocké."""
    return Mock(spec=MetricsCollector)


@pytest.fixture
def strategy_service(mock_strategy_repository, mock_backtest_repository, mock_portfolio_repository,
                    mock_backtesting_service, mock_metrics_collector):
    """Service de stratégies pour les tests."""
    return StrategyService(
        strategy_repository=mock_strategy_repository,
        backtest_repository=mock_backtest_repository,
        portfolio_repository=mock_portfolio_repository,
        backtesting_service=mock_backtesting_service,
        metrics_collector=mock_metrics_collector
    )


@pytest.fixture
def sample_strategy():
    """Stratégie de test."""
    return Strategy(
        id="strategy-001",
        name="Test Mean Reversion",
        type=StrategyType.MEAN_REVERSION,
        description="Test strategy for mean reversion",
        parameters={
            "lookback_period": 20,
            "z_threshold": 2.0,
            "position_size": 0.1
        },
        status=StrategyStatus.INACTIVE,
        created_date=datetime.now(),
        version="1.0.0"
    )


@pytest.fixture
def sample_portfolio():
    """Portfolio de test."""
    return Portfolio(
        id="portfolio-001",
        name="Test Portfolio",
        initial_capital=Decimal("100000.00"),
        base_currency="USD"
    )


@pytest.fixture
def sample_backtest():
    """Backtest de test."""
    return BacktestResult(
        id="backtest-001",
        strategy_id="strategy-001",
        portfolio_id="portfolio-001",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_capital=Decimal("100000.00"),
        status=BacktestStatus.COMPLETED,
        created_date=datetime.now()
    )


@pytest.fixture
def sample_market_data():
    """Données de marché pour tests."""
    dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
    np.random.seed(42)

    # Simulation de prix avec mean reversion
    prices = []
    price = 100.0
    for i in range(252):
        # Mean reversion vers 100 + bruit
        mean_price = 100.0
        reversion_force = (mean_price - price) * 0.05
        noise = np.random.normal(0, 2)
        price += reversion_force + noise
        prices.append(price)

    return pd.DataFrame({
        "date": dates,
        "symbol": "TEST/USD",
        "open": prices,
        "high": [p * 1.02 for p in prices],
        "low": [p * 0.98 for p in prices],
        "close": prices,
        "volume": np.random.uniform(1000, 10000, 252)
    })


class TestStrategyServiceBasic:
    """Tests de base pour StrategyService."""

    async def test_create_strategy(self, strategy_service, mock_strategy_repository):
        """Test de création d'une stratégie."""
        # Arrange
        strategy_data = {
            "name": "New Mean Reversion",
            "type": StrategyType.MEAN_REVERSION,
            "description": "New test strategy",
            "parameters": {
                "lookback_period": 15,
                "z_threshold": 1.5,
                "position_size": 0.05
            }
        }

        expected_strategy = Strategy(
            id="strategy-002",
            **strategy_data,
            status=StrategyStatus.INACTIVE,
            created_date=datetime.now(),
            version="1.0.0"
        )
        mock_strategy_repository.save.return_value = expected_strategy

        # Act
        result = await strategy_service.create_strategy(**strategy_data)

        # Assert
        assert result.name == "New Mean Reversion"
        assert result.type == StrategyType.MEAN_REVERSION
        assert result.status == StrategyStatus.INACTIVE
        assert result.parameters["lookback_period"] == 15
        mock_strategy_repository.save.assert_called_once()

    async def test_get_strategy(self, strategy_service, mock_strategy_repository, sample_strategy):
        """Test de récupération d'une stratégie."""
        # Arrange
        mock_strategy_repository.get_by_id.return_value = sample_strategy

        # Act
        result = await strategy_service.get_strategy("strategy-001")

        # Assert
        assert result == sample_strategy
        mock_strategy_repository.get_by_id.assert_called_once_with("strategy-001")

    async def test_get_all_strategies(self, strategy_service, mock_strategy_repository, sample_strategy):
        """Test de récupération de toutes les stratégies."""
        # Arrange
        strategies = [sample_strategy]
        mock_strategy_repository.get_all.return_value = strategies

        # Act
        result = await strategy_service.get_all_strategies()

        # Assert
        assert len(result) == 1
        assert result[0] == sample_strategy
        mock_strategy_repository.get_all.assert_called_once()

    async def test_update_strategy(self, strategy_service, mock_strategy_repository, sample_strategy):
        """Test de mise à jour d'une stratégie."""
        # Arrange
        mock_strategy_repository.get_by_id.return_value = sample_strategy

        updates = {
            "parameters": {
                "lookback_period": 25,
                "z_threshold": 2.5,
                "position_size": 0.15
            },
            "description": "Updated description"
        }

        updated_strategy = sample_strategy
        updated_strategy.parameters.update(updates["parameters"])
        updated_strategy.description = updates["description"]
        updated_strategy.version = "1.1.0"
        mock_strategy_repository.save.return_value = updated_strategy

        # Act
        result = await strategy_service.update_strategy("strategy-001", updates)

        # Assert
        assert result.parameters["lookback_period"] == 25
        assert result.description == "Updated description"
        assert result.version == "1.1.0"
        mock_strategy_repository.save.assert_called_once()

    async def test_delete_strategy(self, strategy_service, mock_strategy_repository, sample_strategy):
        """Test de suppression d'une stratégie."""
        # Arrange
        mock_strategy_repository.get_by_id.return_value = sample_strategy
        mock_strategy_repository.delete.return_value = True

        # Act
        result = await strategy_service.delete_strategy("strategy-001")

        # Assert
        assert result is True
        mock_strategy_repository.delete.assert_called_once_with("strategy-001")

    async def test_delete_strategy_in_use(self, strategy_service, mock_strategy_repository, sample_strategy):
        """Test de suppression d'une stratégie en cours d'utilisation."""
        # Arrange
        sample_strategy.status = StrategyStatus.ACTIVE
        mock_strategy_repository.get_by_id.return_value = sample_strategy

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot delete active strategy"):
            await strategy_service.delete_strategy("strategy-001")


class TestStrategyServiceLifecycle:
    """Tests du cycle de vie des stratégies."""

    async def test_activate_strategy(self, strategy_service, mock_strategy_repository, sample_strategy):
        """Test d'activation d'une stratégie."""
        # Arrange
        sample_strategy.status = StrategyStatus.INACTIVE
        mock_strategy_repository.get_by_id.return_value = sample_strategy

        activated_strategy = sample_strategy
        activated_strategy.status = StrategyStatus.ACTIVE
        activated_strategy.activated_date = datetime.now()
        mock_strategy_repository.save.return_value = activated_strategy

        # Act
        result = await strategy_service.activate_strategy("strategy-001")

        # Assert
        assert result.status == StrategyStatus.ACTIVE
        assert result.activated_date is not None
        mock_strategy_repository.save.assert_called_once()

    async def test_deactivate_strategy(self, strategy_service, mock_strategy_repository, sample_strategy):
        """Test de désactivation d'une stratégie."""
        # Arrange
        sample_strategy.status = StrategyStatus.ACTIVE
        mock_strategy_repository.get_by_id.return_value = sample_strategy

        deactivated_strategy = sample_strategy
        deactivated_strategy.status = StrategyStatus.INACTIVE
        deactivated_strategy.deactivated_date = datetime.now()
        mock_strategy_repository.save.return_value = deactivated_strategy

        # Act
        result = await strategy_service.deactivate_strategy("strategy-001")

        # Assert
        assert result.status == StrategyStatus.INACTIVE
        assert result.deactivated_date is not None
        mock_strategy_repository.save.assert_called_once()

    async def test_pause_strategy(self, strategy_service, mock_strategy_repository, sample_strategy):
        """Test de mise en pause d'une stratégie."""
        # Arrange
        sample_strategy.status = StrategyStatus.ACTIVE
        mock_strategy_repository.get_by_id.return_value = sample_strategy

        paused_strategy = sample_strategy
        paused_strategy.status = StrategyStatus.PAUSED
        mock_strategy_repository.save.return_value = paused_strategy

        # Act
        result = await strategy_service.pause_strategy("strategy-001")

        # Assert
        assert result.status == StrategyStatus.PAUSED
        mock_strategy_repository.save.assert_called_once()

    async def test_resume_strategy(self, strategy_service, mock_strategy_repository, sample_strategy):
        """Test de reprise d'une stratégie."""
        # Arrange
        sample_strategy.status = StrategyStatus.PAUSED
        mock_strategy_repository.get_by_id.return_value = sample_strategy

        resumed_strategy = sample_strategy
        resumed_strategy.status = StrategyStatus.ACTIVE
        mock_strategy_repository.save.return_value = resumed_strategy

        # Act
        result = await strategy_service.resume_strategy("strategy-001")

        # Assert
        assert result.status == StrategyStatus.ACTIVE
        mock_strategy_repository.save.assert_called_once()


class TestStrategyServiceBacktesting:
    """Tests de backtesting des stratégies."""

    async def test_run_backtest(self, strategy_service, mock_strategy_repository, mock_portfolio_repository,
                              mock_backtesting_service, mock_backtest_repository, sample_strategy, sample_portfolio):
        """Test de lancement d'un backtest."""
        # Arrange
        mock_strategy_repository.get_by_id.return_value = sample_strategy
        mock_portfolio_repository.get_by_id.return_value = sample_portfolio

        backtest_config = {
            "start_date": datetime(2023, 1, 1),
            "end_date": datetime(2023, 12, 31),
            "initial_capital": Decimal("100000.00"),
            "symbols": ["BTC/USD", "ETH/USD"],
            "benchmark": "SPY"
        }

        expected_backtest = BacktestResult(
            id="backtest-001",
            strategy_id="strategy-001",
            portfolio_id="portfolio-001",
            **backtest_config,
            status=BacktestStatus.RUNNING,
            created_date=datetime.now()
        )
        mock_backtesting_service.run_backtest.return_value = expected_backtest
        mock_backtest_repository.save.return_value = expected_backtest

        # Act
        result = await strategy_service.run_backtest("strategy-001", "portfolio-001", backtest_config)

        # Assert
        assert result.strategy_id == "strategy-001"
        assert result.status == BacktestStatus.RUNNING
        assert result.start_date == datetime(2023, 1, 1)
        mock_backtesting_service.run_backtest.assert_called_once()

    async def test_get_backtest_results(self, strategy_service, mock_backtest_repository, sample_backtest):
        """Test de récupération des résultats de backtest."""
        # Arrange
        sample_backtest.status = BacktestStatus.COMPLETED
        sample_backtest.results = {
            "total_return": 0.15,
            "sharpe_ratio": 1.8,
            "max_drawdown": 0.08,
            "win_rate": 0.65,
            "total_trades": 150
        }
        mock_backtest_repository.get_by_id.return_value = sample_backtest

        # Act
        result = await strategy_service.get_backtest_results("backtest-001")

        # Assert
        assert result["total_return"] == 0.15
        assert result["sharpe_ratio"] == 1.8
        assert result["max_drawdown"] == 0.08
        assert result["total_trades"] == 150

    async def test_get_strategy_backtests(self, strategy_service, mock_backtest_repository, sample_backtest):
        """Test de récupération des backtests d'une stratégie."""
        # Arrange
        backtests = [sample_backtest]
        mock_backtest_repository.get_by_strategy_id.return_value = backtests

        # Act
        result = await strategy_service.get_strategy_backtests("strategy-001")

        # Assert
        assert len(result) == 1
        assert result[0] == sample_backtest
        mock_backtest_repository.get_by_strategy_id.assert_called_once_with("strategy-001")

    async def test_compare_strategy_performance(self, strategy_service, mock_backtest_repository):
        """Test de comparaison de performance entre stratégies."""
        # Arrange
        backtest1 = BacktestResult(
            id="bt1", strategy_id="strategy-001", portfolio_id="p1",
            start_date=datetime(2023, 1, 1), end_date=datetime(2023, 12, 31),
            initial_capital=Decimal("100000.00"), status=BacktestStatus.COMPLETED,
            results={"total_return": 0.15, "sharpe_ratio": 1.8, "max_drawdown": 0.08}
        )

        backtest2 = BacktestResult(
            id="bt2", strategy_id="strategy-002", portfolio_id="p1",
            start_date=datetime(2023, 1, 1), end_date=datetime(2023, 12, 31),
            initial_capital=Decimal("100000.00"), status=BacktestStatus.COMPLETED,
            results={"total_return": 0.12, "sharpe_ratio": 1.5, "max_drawdown": 0.10}
        )

        mock_backtest_repository.get_by_strategy_id.side_effect = [
            [backtest1], [backtest2]
        ]

        # Act
        comparison = await strategy_service.compare_strategy_performance(
            ["strategy-001", "strategy-002"]
        )

        # Assert
        assert len(comparison) == 2
        assert comparison["strategy-001"]["total_return"] == 0.15
        assert comparison["strategy-002"]["total_return"] == 0.12
        assert "performance_ranking" in comparison


class TestStrategyServiceOptimization:
    """Tests d'optimisation des stratégies."""

    async def test_optimize_strategy_parameters(self, strategy_service, mock_strategy_repository,
                                              mock_backtesting_service, sample_strategy):
        """Test d'optimisation des paramètres."""
        # Arrange
        mock_strategy_repository.get_by_id.return_value = sample_strategy

        optimization_config = {
            "parameters_to_optimize": {
                "lookback_period": {"min": 10, "max": 50, "step": 5},
                "z_threshold": {"min": 1.0, "max": 3.0, "step": 0.5},
                "position_size": {"min": 0.05, "max": 0.2, "step": 0.05}
            },
            "optimization_metric": "sharpe_ratio",
            "start_date": datetime(2023, 1, 1),
            "end_date": datetime(2023, 12, 31)
        }

        optimization_results = {
            "best_parameters": {
                "lookback_period": 20,
                "z_threshold": 2.0,
                "position_size": 0.1
            },
            "best_metric_value": 2.1,
            "total_combinations_tested": 162,
            "optimization_time_seconds": 145.3
        }
        mock_backtesting_service.optimize_parameters.return_value = optimization_results

        # Act
        result = await strategy_service.optimize_strategy_parameters("strategy-001", optimization_config)

        # Assert
        assert result["best_parameters"]["lookback_period"] == 20
        assert result["best_metric_value"] == 2.1
        assert result["total_combinations_tested"] == 162

    async def test_walk_forward_analysis(self, strategy_service, mock_backtesting_service, sample_strategy):
        """Test d'analyse walk-forward."""
        # Arrange
        walk_forward_config = {
            "in_sample_period_months": 6,
            "out_of_sample_period_months": 3,
            "start_date": datetime(2022, 1, 1),
            "end_date": datetime(2023, 12, 31),
            "reoptimization_frequency": "quarterly"
        }

        walk_forward_results = {
            "periods": [
                {
                    "in_sample_start": datetime(2022, 1, 1),
                    "in_sample_end": datetime(2022, 6, 30),
                    "out_of_sample_start": datetime(2022, 7, 1),
                    "out_of_sample_end": datetime(2022, 9, 30),
                    "optimized_parameters": {"lookback_period": 15},
                    "out_of_sample_return": 0.08
                },
                {
                    "in_sample_start": datetime(2022, 4, 1),
                    "in_sample_end": datetime(2022, 9, 30),
                    "out_of_sample_start": datetime(2022, 10, 1),
                    "out_of_sample_end": datetime(2022, 12, 31),
                    "optimized_parameters": {"lookback_period": 25},
                    "out_of_sample_return": 0.12
                }
            ],
            "overall_return": 0.20,
            "stability_score": 0.85
        }
        mock_backtesting_service.walk_forward_analysis.return_value = walk_forward_results

        # Act
        result = await strategy_service.walk_forward_analysis("strategy-001", walk_forward_config)

        # Assert
        assert len(result["periods"]) == 2
        assert result["overall_return"] == 0.20
        assert result["stability_score"] == 0.85

    async def test_monte_carlo_simulation(self, strategy_service, mock_backtesting_service):
        """Test de simulation Monte Carlo."""
        # Arrange
        monte_carlo_config = {
            "num_simulations": 1000,
            "confidence_levels": [0.05, 0.95],
            "randomize_parameters": True,
            "randomize_start_dates": True,
            "bootstrap_returns": True
        }

        simulation_results = {
            "simulations": 1000,
            "mean_return": 0.145,
            "std_return": 0.067,
            "confidence_intervals": {
                "5%": 0.034,
                "95%": 0.267
            },
            "probability_of_loss": 0.12,
            "value_at_risk_95": 0.089
        }
        mock_backtesting_service.monte_carlo_simulation.return_value = simulation_results

        # Act
        result = await strategy_service.monte_carlo_simulation("strategy-001", monte_carlo_config)

        # Assert
        assert result["simulations"] == 1000
        assert result["mean_return"] == 0.145
        assert result["probability_of_loss"] == 0.12


class TestStrategyServiceAnalytics:
    """Tests d'analytics des stratégies."""

    async def test_get_strategy_performance_metrics(self, strategy_service, mock_backtest_repository):
        """Test de récupération des métriques de performance."""
        # Arrange
        backtests = [
            BacktestResult(
                id=f"bt{i}", strategy_id="strategy-001", portfolio_id="p1",
                start_date=datetime(2023, 1, 1), end_date=datetime(2023, 12, 31),
                initial_capital=Decimal("100000.00"), status=BacktestStatus.COMPLETED,
                results={
                    "total_return": 0.10 + i*0.05,
                    "sharpe_ratio": 1.5 + i*0.3,
                    "max_drawdown": 0.08 + i*0.02
                }
            )
            for i in range(5)
        ]
        mock_backtest_repository.get_by_strategy_id.return_value = backtests

        # Act
        metrics = await strategy_service.get_strategy_performance_metrics("strategy-001")

        # Assert
        assert "average_return" in metrics
        assert "average_sharpe_ratio" in metrics
        assert "consistency_score" in metrics
        assert "best_backtest" in metrics
        assert "worst_backtest" in metrics

    async def test_analyze_strategy_drawdowns(self, strategy_service, mock_backtesting_service):
        """Test d'analyse des drawdowns."""
        # Arrange
        drawdown_analysis = {
            "max_drawdown": 0.15,
            "avg_drawdown": 0.045,
            "drawdown_periods": [
                {
                    "start": datetime(2023, 3, 15),
                    "end": datetime(2023, 4, 20),
                    "duration_days": 36,
                    "magnitude": 0.12
                },
                {
                    "start": datetime(2023, 8, 10),
                    "end": datetime(2023, 9, 5),
                    "duration_days": 26,
                    "magnitude": 0.15
                }
            ],
            "recovery_times": [18, 22],  # jours
            "underwater_percentage": 0.35
        }
        mock_backtesting_service.analyze_drawdowns.return_value = drawdown_analysis

        # Act
        result = await strategy_service.analyze_strategy_drawdowns("strategy-001")

        # Assert
        assert result["max_drawdown"] == 0.15
        assert len(result["drawdown_periods"]) == 2
        assert result["underwater_percentage"] == 0.35

    async def test_get_strategy_correlation_matrix(self, strategy_service, mock_backtesting_service):
        """Test de matrice de corrélation entre stratégies."""
        # Arrange
        strategy_ids = ["strategy-001", "strategy-002", "strategy-003"]

        correlation_matrix = {
            "strategy-001": {
                "strategy-001": 1.0,
                "strategy-002": 0.45,
                "strategy-003": -0.12
            },
            "strategy-002": {
                "strategy-001": 0.45,
                "strategy-002": 1.0,
                "strategy-003": 0.23
            },
            "strategy-003": {
                "strategy-001": -0.12,
                "strategy-002": 0.23,
                "strategy-003": 1.0
            }
        }
        mock_backtesting_service.calculate_strategy_correlations.return_value = correlation_matrix

        # Act
        result = await strategy_service.get_strategy_correlation_matrix(strategy_ids)

        # Assert
        assert result["strategy-001"]["strategy-002"] == 0.45
        assert result["strategy-001"]["strategy-003"] == -0.12
        assert result["strategy-002"]["strategy-003"] == 0.23


class TestStrategyServiceValidation:
    """Tests de validation des stratégies."""

    async def test_validate_strategy_configuration(self, strategy_service):
        """Test de validation de la configuration d'une stratégie."""
        # Arrange
        valid_config = {
            "name": "Valid Strategy",
            "type": StrategyType.MEAN_REVERSION,
            "parameters": {
                "lookback_period": 20,
                "z_threshold": 2.0,
                "position_size": 0.1
            }
        }

        # Act
        validation = await strategy_service.validate_strategy_configuration(valid_config)

        # Assert
        assert validation["is_valid"] is True
        assert validation["errors"] == []

    async def test_validate_invalid_strategy_configuration(self, strategy_service):
        """Test de validation d'une configuration invalide."""
        # Arrange
        invalid_config = {
            "name": "",  # Nom vide
            "type": StrategyType.MEAN_REVERSION,
            "parameters": {
                "lookback_period": -5,  # Valeur négative
                "z_threshold": 0,       # Valeur invalide
                "position_size": 1.5    # > 100%
            }
        }

        # Act
        validation = await strategy_service.validate_strategy_configuration(invalid_config)

        # Assert
        assert validation["is_valid"] is False
        assert len(validation["errors"]) > 0
        assert any("name" in error for error in validation["errors"])
        assert any("lookback_period" in error for error in validation["errors"])

    async def test_check_strategy_data_requirements(self, strategy_service, sample_strategy):
        """Test de vérification des exigences en données."""
        # Arrange
        data_requirements = {
            "required_symbols": ["BTC/USD", "ETH/USD"],
            "required_timeframes": ["1h", "1d"],
            "min_history_days": 100,
            "required_indicators": ["SMA", "RSI"]
        }

        available_data = {
            "symbols": ["BTC/USD", "ETH/USD", "ADA/USD"],
            "timeframes": ["1m", "1h", "1d"],
            "history_days": 365,
            "indicators": ["SMA", "RSI", "MACD"]
        }

        # Act
        check_result = await strategy_service.check_strategy_data_requirements(
            sample_strategy, data_requirements, available_data
        )

        # Assert
        assert check_result["requirements_met"] is True
        assert check_result["missing_symbols"] == []
        assert check_result["missing_timeframes"] == []


class TestStrategyServiceDeployment:
    """Tests de déploiement des stratégies."""

    async def test_deploy_strategy_to_live(self, strategy_service, mock_strategy_repository, sample_strategy):
        """Test de déploiement en live trading."""
        # Arrange
        sample_strategy.status = StrategyStatus.INACTIVE
        mock_strategy_repository.get_by_id.return_value = sample_strategy

        deployment_config = {
            "portfolio_id": "live-portfolio-001",
            "risk_limits": {
                "max_position_size": 0.05,
                "max_daily_loss": 1000.00
            },
            "execution_params": {
                "slippage_tolerance": 0.001,
                "max_order_size": 10000.00
            }
        }

        deployed_strategy = sample_strategy
        deployed_strategy.status = StrategyStatus.LIVE
        deployed_strategy.deployment_config = deployment_config
        mock_strategy_repository.save.return_value = deployed_strategy

        # Act
        result = await strategy_service.deploy_strategy_to_live("strategy-001", deployment_config)

        # Assert
        assert result.status == StrategyStatus.LIVE
        assert result.deployment_config == deployment_config

    async def test_stop_live_strategy(self, strategy_service, mock_strategy_repository, sample_strategy):
        """Test d'arrêt d'une stratégie live."""
        # Arrange
        sample_strategy.status = StrategyStatus.LIVE
        mock_strategy_repository.get_by_id.return_value = sample_strategy

        stopped_strategy = sample_strategy
        stopped_strategy.status = StrategyStatus.STOPPED
        stopped_strategy.stopped_date = datetime.now()
        mock_strategy_repository.save.return_value = stopped_strategy

        # Act
        result = await strategy_service.stop_live_strategy("strategy-001")

        # Assert
        assert result.status == StrategyStatus.STOPPED
        assert result.stopped_date is not None

    async def test_get_live_strategy_performance(self, strategy_service, mock_strategy_repository):
        """Test de récupération des performances live."""
        # Arrange
        live_performance = {
            "start_date": datetime(2024, 1, 1),
            "current_date": datetime.now(),
            "total_return": 0.08,
            "sharpe_ratio": 1.6,
            "max_drawdown": 0.05,
            "total_trades": 45,
            "win_rate": 0.68,
            "avg_trade_return": 0.0018,
            "current_positions": {
                "BTC/USD": 0.05,
                "ETH/USD": -0.03
            }
        }

        # Mock the live performance calculation
        with patch.object(strategy_service, '_calculate_live_performance', return_value=live_performance):
            # Act
            result = await strategy_service.get_live_strategy_performance("strategy-001")

            # Assert
            assert result["total_return"] == 0.08
            assert result["total_trades"] == 45
            assert "BTC/USD" in result["current_positions"]


class TestStrategyServiceIntegration:
    """Tests d'intégration."""

    @pytest.mark.integration
    async def test_complete_strategy_lifecycle(self, strategy_service, mock_strategy_repository,
                                             mock_backtesting_service, mock_backtest_repository):
        """Test du cycle de vie complet d'une stratégie."""
        # Arrange
        strategy_data = {
            "name": "Complete Test Strategy",
            "type": StrategyType.MEAN_REVERSION,
            "description": "End-to-end test",
            "parameters": {"lookback_period": 20, "z_threshold": 2.0}
        }

        created_strategy = Strategy(
            id="strategy-e2e",
            **strategy_data,
            status=StrategyStatus.INACTIVE,
            created_date=datetime.now(),
            version="1.0.0"
        )

        # Mock repository responses
        mock_strategy_repository.save.return_value = created_strategy
        mock_strategy_repository.get_by_id.return_value = created_strategy

        # Backtest setup
        backtest_config = {
            "start_date": datetime(2023, 1, 1),
            "end_date": datetime(2023, 12, 31),
            "initial_capital": Decimal("100000.00")
        }

        completed_backtest = BacktestResult(
            id="bt-e2e",
            strategy_id="strategy-e2e",
            portfolio_id="portfolio-001",
            **backtest_config,
            status=BacktestStatus.COMPLETED,
            results={"total_return": 0.18, "sharpe_ratio": 2.1}
        )

        mock_backtesting_service.run_backtest.return_value = completed_backtest
        mock_backtest_repository.save.return_value = completed_backtest
        mock_backtest_repository.get_by_id.return_value = completed_backtest

        # Act
        # 1. Create strategy
        strategy = await strategy_service.create_strategy(**strategy_data)

        # 2. Run backtest
        backtest = await strategy_service.run_backtest(
            strategy.id, "portfolio-001", backtest_config
        )

        # 3. Get results
        results = await strategy_service.get_backtest_results(backtest.id)

        # 4. Activate strategy
        activated_strategy = await strategy_service.activate_strategy(strategy.id)

        # Assert
        assert strategy.name == "Complete Test Strategy"
        assert backtest.strategy_id == strategy.id
        assert results["total_return"] == 0.18
        assert activated_strategy.status == StrategyStatus.ACTIVE