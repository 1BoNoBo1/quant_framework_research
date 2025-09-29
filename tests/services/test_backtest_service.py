"""
Tests for Backtest Service
==========================

Suite de tests complète pour le service de backtesting.
Teste tous les types de backtest : single period, walk-forward, Monte Carlo.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any

from qframe.domain.services.backtesting_service import BacktestingService, MarketDataPoint
from qframe.domain.entities.backtest import (
    BacktestConfiguration, BacktestResult, BacktestMetrics, BacktestStatus,
    BacktestType, WalkForwardConfig, MonteCarloConfig, RebalanceFrequency,
    TradeExecution
)
from qframe.domain.entities.portfolio import Portfolio, Position
from qframe.domain.entities.order import Order, OrderSide, OrderType, OrderStatus
from qframe.domain.repositories.backtest_repository import BacktestRepository
from qframe.domain.repositories.strategy_repository import StrategyRepository
from qframe.domain.repositories.portfolio_repository import PortfolioRepository


@pytest.fixture
def mock_backtest_repository():
    """Repository de backtest mocké."""
    repo = Mock(spec=BacktestRepository)
    repo.save_result = AsyncMock()
    repo.find_by_id = AsyncMock()
    repo.find_by_configuration_id = AsyncMock()
    return repo


@pytest.fixture
def mock_strategy_repository():
    """Repository de stratégie mocké."""
    repo = Mock(spec=StrategyRepository)
    repo.find_by_id = AsyncMock()
    repo.list_all = AsyncMock()
    return repo


@pytest.fixture
def mock_portfolio_repository():
    """Repository de portfolio mocké."""
    repo = Mock(spec=PortfolioRepository)
    repo.save = AsyncMock()
    repo.find_by_id = AsyncMock()
    return repo


@pytest.fixture
def backtest_service(mock_backtest_repository, mock_strategy_repository, mock_portfolio_repository):
    """Service de backtesting pour les tests."""
    return BacktestingService(
        backtest_repository=mock_backtest_repository,
        strategy_repository=mock_strategy_repository,
        portfolio_repository=mock_portfolio_repository
    )


@pytest.fixture
def sample_backtest_config():
    """Configuration de backtest d'exemple."""
    return BacktestConfiguration(
        name="Test Backtest",
        description="Test configuration",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_capital=Decimal("100000.00"),
        strategy_ids=["strategy-001", "strategy-002"],
        transaction_cost=Decimal("0.001"),
        slippage=Decimal("0.0005"),
        backtest_type=BacktestType.SINGLE_PERIOD
    )


@pytest.fixture
def sample_walk_forward_config():
    """Configuration walk-forward d'exemple."""
    return WalkForwardConfig(
        training_period_months=6,
        testing_period_months=3,
        step_months=1,
        min_training_observations=100
    )


@pytest.fixture
def sample_monte_carlo_config():
    """Configuration Monte Carlo d'exemple."""
    return MonteCarloConfig(
        num_simulations=100,
        confidence_levels=[0.05, 0.95],
        bootstrap_method="stationary",
        block_size=None
    )


@pytest.fixture
def sample_strategy():
    """Stratégie d'exemple pour les tests."""
    strategy = Mock()
    strategy.id = "strategy-001"
    strategy.name = "Test Strategy"
    strategy.generate_signals = AsyncMock(return_value=[])
    return strategy


@pytest.fixture
def sample_portfolio():
    """Portfolio d'exemple."""
    return Portfolio(
        id="portfolio-001",
        name="Test Portfolio",
        initial_capital=Decimal("100000.00"),
        base_currency="USD"
    )


@pytest.fixture
def sample_market_data():
    """Données de marché d'exemple."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    np.random.seed(42)

    n_points = len(dates)
    returns = np.random.normal(0.0005, 0.02, n_points)
    prices = 100 * np.exp(np.cumsum(returns))

    return pd.DataFrame({
        'open': np.roll(prices, 1),
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.lognormal(15, 0.5, n_points)
    }, index=dates)


class TestBacktestingServiceBasic:
    """Tests de base pour BacktestingService."""

    async def test_service_initialization(self, backtest_service):
        """Test d'initialisation du service."""
        assert backtest_service.backtest_repository is not None
        assert backtest_service.strategy_repository is not None
        assert backtest_service.portfolio_repository is not None

    async def test_run_backtest_invalid_config(self, backtest_service, mock_backtest_repository):
        """Test avec une configuration invalide."""
        # Arrange
        invalid_config = BacktestConfiguration(
            name="Invalid Test",
            start_date=datetime(2023, 12, 31),
            end_date=datetime(2023, 1, 1),  # Date fin avant date début
            initial_capital=Decimal("-1000")  # Capital négatif
        )

        # Act
        result = await backtest_service.run_backtest(invalid_config)

        # Assert
        assert result.status == BacktestStatus.FAILED
        assert "Configuration invalid" in result.error_message
        mock_backtest_repository.save_result.assert_called()

    async def test_run_single_period_backtest_success(
        self,
        backtest_service,
        sample_backtest_config,
        mock_strategy_repository,
        mock_backtest_repository,
        sample_strategy
    ):
        """Test de backtest single period réussi."""
        # Arrange
        mock_strategy_repository.find_by_id.return_value = sample_strategy

        # Act
        result = await backtest_service.run_backtest(sample_backtest_config)

        # Assert
        assert result.status == BacktestStatus.COMPLETED
        assert result.configuration_id == sample_backtest_config.id
        assert result.initial_capital == sample_backtest_config.initial_capital
        mock_backtest_repository.save_result.assert_called()

    async def test_run_backtest_no_strategies(
        self,
        backtest_service,
        sample_backtest_config,
        mock_strategy_repository,
        mock_backtest_repository
    ):
        """Test avec aucune stratégie trouvée."""
        # Arrange
        mock_strategy_repository.find_by_id.return_value = None

        # Act
        result = await backtest_service.run_backtest(sample_backtest_config)

        # Assert
        assert result.status == BacktestStatus.FAILED
        assert "No valid strategies found" in result.error_message

    async def test_walk_forward_backtest(
        self,
        backtest_service,
        sample_backtest_config,
        sample_walk_forward_config,
        mock_strategy_repository,
        sample_strategy
    ):
        """Test de backtest walk-forward."""
        # Arrange
        sample_backtest_config.backtest_type = BacktestType.WALK_FORWARD
        sample_backtest_config.walk_forward_config = sample_walk_forward_config
        mock_strategy_repository.find_by_id.return_value = sample_strategy

        # Act
        result = await backtest_service.run_backtest(sample_backtest_config)

        # Assert
        assert result.status == BacktestStatus.COMPLETED
        assert hasattr(result, 'sub_results')

    async def test_monte_carlo_backtest(
        self,
        backtest_service,
        sample_backtest_config,
        sample_monte_carlo_config,
        mock_strategy_repository,
        sample_strategy
    ):
        """Test de backtest Monte Carlo."""
        # Arrange
        sample_backtest_config.backtest_type = BacktestType.MONTE_CARLO
        sample_backtest_config.monte_carlo_config = sample_monte_carlo_config
        mock_strategy_repository.find_by_id.return_value = sample_strategy

        # Act
        result = await backtest_service.run_backtest(sample_backtest_config)

        # Assert
        assert result.status == BacktestStatus.COMPLETED
        assert hasattr(result, 'confidence_intervals')


class TestBacktestingMetrics:
    """Tests des métriques de performance."""

    async def test_calculate_basic_metrics(self, backtest_service, sample_backtest_config):
        """Test de calcul des métriques de base."""
        # Arrange
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        trades = [
            TradeExecution(symbol="BTC/USD", quantity=Decimal("1"), price=Decimal("100"), value=Decimal("10")),
            TradeExecution(symbol="BTC/USD", quantity=Decimal("-1"), price=Decimal("105"), value=Decimal("-5"))
        ]

        # Act
        metrics = await backtest_service._calculate_metrics(returns, trades, sample_backtest_config)

        # Assert
        assert isinstance(metrics, BacktestMetrics)
        assert metrics.total_return > 0
        assert metrics.sharpe_ratio is not None
        assert metrics.total_trades == 2
        assert metrics.winning_trades >= 0
        assert metrics.losing_trades >= 0

    def test_calculate_drawdown_series(self, backtest_service):
        """Test de calcul de la série de drawdown."""
        # Arrange
        cumulative_returns = pd.Series([1.0, 1.1, 1.05, 1.2, 1.15, 1.3])

        # Act
        drawdown = backtest_service._calculate_drawdown_series(cumulative_returns)

        # Assert
        assert len(drawdown) == len(cumulative_returns)
        assert all(drawdown <= 0)  # Drawdown toujours négatif ou zéro

    def test_calculate_max_drawdown_duration(self, backtest_service):
        """Test de calcul de la durée maximale de drawdown."""
        # Arrange
        drawdown_series = pd.Series([0, -0.05, -0.1, -0.05, 0, 0, -0.03, 0])

        # Act
        duration = backtest_service._calculate_max_drawdown_duration(drawdown_series)

        # Assert
        assert duration >= 0
        assert isinstance(duration, int)

    def test_calculate_sortino_ratio(self, backtest_service):
        """Test de calcul du ratio de Sortino."""
        # Arrange
        returns = pd.Series([0.02, -0.01, 0.03, -0.005, 0.01, -0.02, 0.015])

        # Act
        sortino = backtest_service._calculate_sortino_ratio(returns)

        # Assert
        assert isinstance(sortino, float)
        assert sortino > 0  # Pour des returns avec moyenne positive


class TestMarketDataGeneration:
    """Tests de génération de données de marché."""

    async def test_generate_market_data(self, backtest_service):
        """Test de génération de données de marché."""
        # Arrange
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)

        # Act
        data = await backtest_service._generate_market_data(start_date, end_date)

        # Assert
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 31  # 31 jours
        assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        assert all(data['high'] >= data['low'])
        assert all(data['high'] >= data['open'])
        assert all(data['high'] >= data['close'])
        assert all(data['low'] <= data['open'])
        assert all(data['low'] <= data['close'])

    async def test_generate_market_data_consistency(self, backtest_service):
        """Test de cohérence des données générées."""
        # Arrange
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 10)

        # Act
        data1 = await backtest_service._generate_market_data(start_date, end_date)
        data2 = await backtest_service._generate_market_data(start_date, end_date)

        # Assert - Avec la même seed, les données doivent être identiques
        pd.testing.assert_frame_equal(data1, data2)


class TestTradingSimulation:
    """Tests de simulation de trading."""

    async def test_simulate_trading_basic(
        self,
        backtest_service,
        sample_backtest_config,
        sample_market_data,
        sample_portfolio
    ):
        """Test de simulation de trading de base."""
        # Arrange
        strategies = [Mock()]
        strategies[0].generate_signals = AsyncMock(return_value=[])

        # Act
        portfolio_values, trades = await backtest_service._simulate_trading(
            sample_backtest_config, strategies, sample_market_data, sample_portfolio
        )

        # Assert
        assert isinstance(portfolio_values, pd.Series)
        assert isinstance(trades, list)
        assert len(portfolio_values) == len(sample_market_data)

    async def test_update_portfolio_values(self, backtest_service, sample_portfolio):
        """Test de mise à jour des valeurs du portfolio."""
        # Arrange
        market_data = pd.Series({
            'open': 100.0,
            'high': 105.0,
            'low': 95.0,
            'close': 102.0,
            'volume': 1000.0
        })

        # Ajouter une position au portfolio
        position = Position(
            symbol="BTC/USD",
            quantity=Decimal("1.0"),
            average_price=Decimal("100.0"),
            current_price=Decimal("100.0")
        )
        sample_portfolio.positions.append(position)

        # Act
        await backtest_service._update_portfolio_values(sample_portfolio, market_data)

        # Assert
        assert sample_portfolio.positions[0].current_price == Decimal("102.0")


class TestBacktestTypes:
    """Tests des différents types de backtest."""

    async def test_single_period_execution(
        self,
        backtest_service,
        sample_backtest_config,
        mock_strategy_repository,
        sample_strategy
    ):
        """Test d'exécution single period."""
        # Arrange
        mock_strategy_repository.find_by_id.return_value = sample_strategy
        result = BacktestResult(
            configuration_id=sample_backtest_config.id,
            name=sample_backtest_config.name,
            status=BacktestStatus.RUNNING
        )

        # Act
        await backtest_service._run_single_period_backtest(sample_backtest_config, result)

        # Assert
        assert result.metrics is not None
        assert result.portfolio_values is not None
        assert result.returns is not None

    async def test_walk_forward_periods_calculation(
        self,
        backtest_service,
        sample_backtest_config,
        sample_walk_forward_config
    ):
        """Test de calcul des périodes walk-forward."""
        # Arrange
        sample_backtest_config.backtest_type = BacktestType.WALK_FORWARD
        sample_backtest_config.walk_forward_config = sample_walk_forward_config
        sample_backtest_config.start_date = datetime(2023, 1, 1)
        sample_backtest_config.end_date = datetime(2023, 12, 31)

        result = BacktestResult(
            configuration_id=sample_backtest_config.id,
            name=sample_backtest_config.name,
            status=BacktestStatus.RUNNING
        )

        # Act
        await backtest_service._run_walk_forward_backtest(sample_backtest_config, result)

        # Assert
        assert hasattr(result, 'sub_results')
        assert result.metrics is not None


class TestBootstrapAndMonteCarlo:
    """Tests des méthodes bootstrap et Monte Carlo."""

    def test_bootstrap_market_data_stationary(self, backtest_service, sample_market_data):
        """Test de bootstrap stationnaire."""
        # Act
        bootstrapped = backtest_service._bootstrap_market_data(sample_market_data, "stationary")

        # Assert
        assert len(bootstrapped) == len(sample_market_data)
        assert list(bootstrapped.columns) == list(sample_market_data.columns)

    def test_calculate_confidence_intervals(self, backtest_service):
        """Test de calcul des intervalles de confiance."""
        # Arrange
        simulation_results = []
        for i in range(10):
            result = BacktestResult(
                configuration_id=f"config-{i}",
                name=f"Simulation {i}",
                status=BacktestStatus.COMPLETED
            )
            result.metrics = BacktestMetrics(
                sharpe_ratio=Decimal(str(0.5 + i * 0.1)),
                max_drawdown=Decimal(str(-0.1 - i * 0.01)),
                total_return=Decimal(str(0.1 + i * 0.02))
            )
            simulation_results.append(result)

        confidence_levels = [0.05, 0.95]

        # Act
        intervals = backtest_service._calculate_confidence_intervals(
            simulation_results, confidence_levels
        )

        # Assert
        assert 'sharpe_ratio' in intervals
        assert 'max_drawdown' in intervals
        assert 'total_return' in intervals
        assert 'p5' in intervals['sharpe_ratio']
        assert 'p95' in intervals['sharpe_ratio']


class TestErrorHandling:
    """Tests de gestion d'erreur."""

    async def test_backtest_exception_handling(
        self,
        backtest_service,
        sample_backtest_config,
        mock_strategy_repository,
        mock_backtest_repository
    ):
        """Test de gestion d'exception pendant le backtest."""
        # Arrange
        mock_strategy_repository.find_by_id.side_effect = Exception("Database error")

        # Act
        result = await backtest_service.run_backtest(sample_backtest_config)

        # Assert
        assert result.status == BacktestStatus.FAILED
        assert "Database error" in result.error_message
        assert result.end_time is not None

    async def test_walk_forward_missing_config(
        self,
        backtest_service,
        sample_backtest_config
    ):
        """Test walk-forward sans configuration."""
        # Arrange
        sample_backtest_config.backtest_type = BacktestType.WALK_FORWARD
        sample_backtest_config.walk_forward_config = None

        result = BacktestResult(
            configuration_id=sample_backtest_config.id,
            name=sample_backtest_config.name,
            status=BacktestStatus.RUNNING
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Walk-forward configuration is required"):
            await backtest_service._run_walk_forward_backtest(sample_backtest_config, result)

    async def test_monte_carlo_missing_config(
        self,
        backtest_service,
        sample_backtest_config
    ):
        """Test Monte Carlo sans configuration."""
        # Arrange
        sample_backtest_config.backtest_type = BacktestType.MONTE_CARLO
        sample_backtest_config.monte_carlo_config = None

        result = BacktestResult(
            configuration_id=sample_backtest_config.id,
            name=sample_backtest_config.name,
            status=BacktestStatus.RUNNING
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Monte Carlo configuration is required"):
            await backtest_service._run_monte_carlo_backtest(sample_backtest_config, result)


class TestPerformanceAndOptimization:
    """Tests de performance et optimisation."""

    async def test_large_market_data_handling(self, backtest_service):
        """Test de gestion de gros volumes de données."""
        # Arrange
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2023, 12, 31)  # 4 années

        # Act
        start_time = datetime.now()
        data = await backtest_service._generate_market_data(start_date, end_date)
        processing_time = (datetime.now() - start_time).total_seconds()

        # Assert
        assert len(data) > 1000  # Plus de 1000 jours
        assert processing_time < 10.0  # Moins de 10 secondes
        assert data.memory_usage().sum() < 50_000_000  # Moins de 50MB

    async def test_empty_returns_handling(self, backtest_service, sample_backtest_config):
        """Test de gestion des returns vides."""
        # Arrange
        empty_returns = pd.Series([], dtype=float)
        trades = []

        # Act
        metrics = await backtest_service._calculate_metrics(
            empty_returns, trades, sample_backtest_config
        )

        # Assert
        assert isinstance(metrics, BacktestMetrics)
        # Les métriques doivent avoir des valeurs par défaut sensées


class TestMetricsAggregation:
    """Tests d'agrégation de métriques."""

    async def test_aggregate_walk_forward_metrics(self, backtest_service):
        """Test d'agrégation des métriques walk-forward."""
        # Arrange
        sub_results = []
        for i in range(3):
            result = BacktestResult(
                configuration_id=f"config-{i}",
                name=f"Period {i}",
                status=BacktestStatus.COMPLETED
            )
            result.metrics = BacktestMetrics(
                total_return=Decimal(str(0.1 + i * 0.05)),
                sharpe_ratio=Decimal(str(1.0 + i * 0.2)),
                max_drawdown=Decimal(str(-0.05 - i * 0.01)),
                win_rate=Decimal(str(0.6 + i * 0.05)),
                total_trades=10 + i * 5
            )
            sub_results.append(result)

        # Act
        aggregated = await backtest_service._aggregate_walk_forward_metrics(sub_results)

        # Assert
        assert isinstance(aggregated, BacktestMetrics)
        assert aggregated.total_return > 0
        assert aggregated.sharpe_ratio > 0
        assert aggregated.total_trades == 45  # 10 + 15 + 20

    async def test_aggregate_empty_results(self, backtest_service):
        """Test d'agrégation avec résultats vides."""
        # Arrange
        empty_results = []

        # Act
        aggregated = await backtest_service._aggregate_walk_forward_metrics(empty_results)

        # Assert
        assert isinstance(aggregated, BacktestMetrics)


class TestDataTypes:
    """Tests des types de données."""

    def test_market_data_point_creation(self):
        """Test de création d'un point de données de marché."""
        # Act
        point = MarketDataPoint(
            timestamp=datetime.now(),
            symbol="BTC/USD",
            open_price=Decimal("45000.00"),
            high_price=Decimal("46000.00"),
            low_price=Decimal("44000.00"),
            close_price=Decimal("45500.00"),
            volume=Decimal("123.45"),
            metadata={"exchange": "binance"}
        )

        # Assert
        assert point.symbol == "BTC/USD"
        assert point.open_price == Decimal("45000.00")
        assert point.metadata["exchange"] == "binance"

    def test_returns_calculation(self, backtest_service):
        """Test de calcul des returns."""
        # Arrange
        portfolio_values = pd.Series([100000, 101000, 99500, 102000, 103500])

        # Act
        returns = backtest_service._calculate_returns(portfolio_values)

        # Assert
        assert len(returns) == 4  # n-1 returns
        assert abs(returns.iloc[0] - 0.01) < 0.001  # Premier return ~1%