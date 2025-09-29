"""
Tests for Backtest Engine
=========================

Tests complets pour le moteur de backtesting incluant :
- Simulation historique réaliste
- Exécution avec slippage et commissions
- Tests de stratégies multiples
- Application des limites de risque
- Attribution de performance
- Analyse walk-forward
- Simulation Monte Carlo
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

from qframe.domain.services.backtesting_service import (
    BacktestingService,
    MarketDataPoint
)
from qframe.domain.entities.backtest import (
    BacktestConfiguration,
    BacktestResult,
    BacktestMetrics,
    BacktestStatus,
    BacktestType,
    TradeExecution,
    WalkForwardConfig,
    MonteCarloConfig
)
from qframe.domain.entities.order import Order, OrderStatus, OrderSide, OrderType
from qframe.domain.entities.portfolio import Portfolio, Position, PortfolioStatus
from qframe.domain.entities.strategy import Strategy, StrategyStatus
from qframe.core.interfaces import Signal, SignalAction


class TestBacktestEngine:
    """Test suite pour le moteur de backtesting."""

    @pytest.fixture
    def mock_repositories(self):
        """Repositories mockés pour les tests."""
        return {
            'backtest_repository': AsyncMock(),
            'strategy_repository': AsyncMock(),
            'portfolio_repository': AsyncMock()
        }

    @pytest.fixture
    def backtest_service(self, mock_repositories):
        """Service de backtesting initialisé."""
        return BacktestingService(
            backtest_repository=mock_repositories['backtest_repository'],
            strategy_repository=mock_repositories['strategy_repository'],
            portfolio_repository=mock_repositories['portfolio_repository']
        )

    @pytest.fixture
    def sample_strategy(self):
        """Stratégie échantillon pour tests."""
        strategy = Mock(spec=Strategy)
        strategy.id = "test_strategy_001"
        strategy.name = "Test Mean Reversion"
        strategy.status = StrategyStatus.ACTIVE
        strategy.generate_signals = AsyncMock()
        return strategy

    @pytest.fixture
    def sample_market_data(self):
        """Données de marché historiques pour backtesting."""
        np.random.seed(42)
        start_date = datetime(2023, 1, 1)
        n_days = 252  # 1 an de trading

        data_points = []
        current_price = Decimal("50000")

        for i in range(n_days):
            timestamp = start_date + timedelta(days=i)

            # Simulation réaliste avec volatilité et trend
            daily_return = np.random.normal(0.0002, 0.02)  # 0.02% mean, 2% vol
            price_change = current_price * Decimal(str(daily_return))
            new_price = current_price + price_change

            # OHLC realistic simulation
            high_offset = abs(np.random.normal(0, 0.005))
            low_offset = abs(np.random.normal(0, 0.005))

            high_price = new_price * (Decimal("1") + Decimal(str(high_offset)))
            low_price = new_price * (Decimal("1") - Decimal(str(low_offset)))
            open_price = current_price
            close_price = new_price

            volume = Decimal(str(np.random.lognormal(10, 0.5)))

            data_point = MarketDataPoint(
                timestamp=timestamp,
                symbol="BTC/USDT",
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=close_price,
                volume=volume
            )
            data_points.append(data_point)
            current_price = new_price

        return data_points

    @pytest.fixture
    def backtest_config(self):
        """Configuration de backtest standard."""
        return BacktestConfiguration(
            name="Test Backtest",
            strategy_ids=["test_strategy_001"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=Decimal("100000"),
            transaction_cost=Decimal("0.001"),  # 0.1%
            slippage=Decimal("0.0005"),  # 0.05%
            max_position_size=Decimal("0.2"),  # 20% max per position
            backtest_type=BacktestType.HISTORICAL_SIMULATION
        )

    @pytest.fixture
    def sample_signals(self, sample_market_data):
        """Signaux échantillon pour le backtest."""
        signals = []
        for i, data_point in enumerate(sample_market_data[::10]):  # Un signal tous les 10 jours
            action = SignalAction.BUY if i % 2 == 0 else SignalAction.SELL
            signal = Signal(
                timestamp=data_point.timestamp,
                symbol="BTC/USDT",
                action=action,
                strength=0.7,
                price=data_point.close_price,
                size=Decimal("0.1"),  # 10% du capital
                metadata={"strategy": "test_strategy"}
            )
            signals.append(signal)
        return signals

    def test_backtest_configuration_validation(self, backtest_config):
        """Test validation de la configuration de backtest."""
        # Configuration valide
        assert backtest_config.start_date < backtest_config.end_date
        assert backtest_config.initial_capital > 0
        assert backtest_config.transaction_cost >= 0
        assert backtest_config.slippage >= 0

        # Test validation des limites de risque
        assert backtest_config.max_position_size <= 1
        assert backtest_config.max_leverage >= 1

    @pytest.mark.asyncio
    async def test_backtest_initialization(self, backtest_service, backtest_config):
        """Test l'initialisation d'un backtest."""
        backtest_id = await backtest_service.create_backtest(backtest_config)

        assert backtest_id is not None
        assert isinstance(backtest_id, str)

        # Vérifier que le backtest a été sauvegardé
        backtest_service._backtest_repository.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_historical_simulation(self, backtest_service, backtest_config, sample_market_data, sample_strategy, sample_signals):
        """Test simulation historique complète."""
        # Mock strategy signals
        sample_strategy.generate_signals.return_value = sample_signals

        # Mock repository responses
        backtest_service._strategy_repository.find_by_id.return_value = sample_strategy

        # Exécuter le backtest
        result = await backtest_service.run_backtest(
            config=backtest_config,
            market_data=sample_market_data
        )

        assert isinstance(result, BacktestResult)
        assert result.status == BacktestStatus.COMPLETED
        assert result.metrics is not None
        assert len(result.trades) > 0

        # Vérifier les métriques de base
        assert result.metrics.total_return is not None
        assert result.metrics.sharpe_ratio is not None
        assert result.metrics.max_drawdown is not None

    @pytest.mark.asyncio
    async def test_trade_execution_with_slippage(self, backtest_service, backtest_config):
        """Test exécution des trades avec slippage."""
        # Signal d'achat
        buy_signal = Signal(
            timestamp=datetime(2023, 6, 1),
            symbol="BTC/USDT",
            action=SignalAction.BUY,
            strength=0.8,
            price=Decimal("30000"),
            size=Decimal("1.0"),
            metadata={}
        )

        # Market data point correspondant
        market_data = MarketDataPoint(
            timestamp=datetime(2023, 6, 1),
            symbol="BTC/USDT",
            open_price=Decimal("30000"),
            high_price=Decimal("30500"),
            low_price=Decimal("29500"),
            close_price=Decimal("30100"),
            volume=Decimal("1000000")
        )

        execution = await backtest_service._execute_signal(
            signal=buy_signal,
            market_data=market_data,
            config=backtest_config
        )

        assert isinstance(execution, TradeExecution)
        assert execution.executed_price > buy_signal.price  # Slippage pour achat
        assert execution.commission > 0
        assert execution.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_trade_execution_with_commission(self, backtest_service, backtest_config):
        """Test calcul des commissions sur les trades."""
        signal = Signal(
            timestamp=datetime(2023, 6, 1),
            symbol="BTC/USDT",
            action=SignalAction.BUY,
            strength=0.5,
            price=Decimal("30000"),
            size=Decimal("2.0"),
            metadata={}
        )

        market_data = MarketDataPoint(
            timestamp=datetime(2023, 6, 1),
            symbol="BTC/USDT",
            open_price=Decimal("30000"),
            high_price=Decimal("30200"),
            low_price=Decimal("29800"),
            close_price=Decimal("30050"),
            volume=Decimal("500000")
        )

        execution = await backtest_service._execute_signal(signal, market_data, backtest_config)

        # Commission = prix * quantité * taux de commission
        expected_commission = execution.executed_price * execution.quantity * backtest_config.commission_rate
        assert abs(execution.commission - expected_commission) < Decimal("0.01")

    @pytest.mark.asyncio
    async def test_portfolio_tracking(self, backtest_service, backtest_config, sample_market_data):
        """Test le suivi du portefeuille pendant le backtest."""
        # Créer un portfolio initial
        portfolio = Portfolio(
            id="test_portfolio",
            name="Test Portfolio",
            total_value=backtest_config.initial_capital,
            cash_balance=backtest_config.initial_capital,
            positions={},
            status=PortfolioStatus.ACTIVE
        )

        # Exécuter quelques trades
        buy_execution = TradeExecution(
            timestamp=datetime(2023, 6, 1),
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            executed_price=Decimal("30000"),
            commission=Decimal("30"),
            status=OrderStatus.FILLED
        )

        updated_portfolio = await backtest_service._update_portfolio(
            portfolio=portfolio,
            execution=buy_execution
        )

        # Vérifier que la position a été ajoutée
        assert "BTC/USDT" in updated_portfolio.positions
        assert updated_portfolio.positions["BTC/USDT"].quantity == Decimal("1.0")
        assert updated_portfolio.cash_balance < backtest_config.initial_capital

    @pytest.mark.asyncio
    async def test_risk_limits_enforcement(self, backtest_service, backtest_config):
        """Test l'application des limites de risque."""
        # Signal qui dépasserait la limite de position
        oversized_signal = Signal(
            timestamp=datetime(2023, 6, 1),
            symbol="BTC/USDT",
            action=SignalAction.BUY,
            strength=0.9,
            price=Decimal("30000"),
            size=Decimal("0.5"),  # 50% du capital (> limite de 20%)
            metadata={}
        )

        # Vérifier que le signal est ajusté ou rejeté
        risk_check = await backtest_service._check_risk_limits(
            signal=oversized_signal,
            current_portfolio_value=backtest_config.initial_capital,
            risk_limits=backtest_config.risk_limits
        )

        assert risk_check.is_rejected or risk_check.adjusted_size < oversized_signal.size

    @pytest.mark.asyncio
    async def test_performance_metrics_calculation(self, backtest_service):
        """Test calcul des métriques de performance."""
        # Créer une série de rendements
        returns = pd.Series([
            0.02, -0.01, 0.03, -0.015, 0.025, -0.008, 0.01, 0.005, -0.02, 0.015
        ])

        # Créer quelques trades
        trades = [
            TradeExecution(
                timestamp=datetime(2023, 1, 1),
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                quantity=Decimal("1.0"),
                executed_price=Decimal("30000"),
                commission=Decimal("30"),
                status=OrderStatus.FILLED
            ),
            TradeExecution(
                timestamp=datetime(2023, 1, 15),
                symbol="BTC/USDT",
                side=OrderSide.SELL,
                quantity=Decimal("1.0"),
                executed_price=Decimal("31000"),
                commission=Decimal("31"),
                status=OrderStatus.FILLED
            )
        ]

        metrics = await backtest_service._calculate_performance_metrics(
            returns=returns,
            trades=trades,
            initial_capital=Decimal("100000")
        )

        assert isinstance(metrics, BacktestMetrics)
        assert metrics.total_return is not None
        assert metrics.sharpe_ratio is not None
        assert metrics.max_drawdown is not None
        assert metrics.sortino_ratio is not None
        assert metrics.calmar_ratio is not None
        assert metrics.win_rate is not None

        # Vérifier les calculs de base
        assert metrics.total_trades == len(trades)
        assert 0 <= metrics.win_rate <= 1

    @pytest.mark.asyncio
    async def test_drawdown_calculation(self, backtest_service):
        """Test calcul du drawdown maximum."""
        # Série de valeurs de portfolio avec drawdown
        portfolio_values = [
            100000, 105000, 102000, 108000, 95000,  # -12% drawdown ici
            98000, 103000, 110000, 107000, 115000
        ]

        max_drawdown = backtest_service._calculate_max_drawdown(portfolio_values)

        # Le drawdown maximum devrait être d'environ 12%
        assert abs(max_drawdown - 0.12) < 0.01

    @pytest.mark.asyncio
    async def test_sharpe_ratio_calculation(self, backtest_service):
        """Test calcul du ratio de Sharpe."""
        returns = pd.Series([0.01, -0.005, 0.02, 0.015, -0.01, 0.008, 0.012])
        risk_free_rate = 0.02  # 2% annuel

        sharpe_ratio = backtest_service._calculate_sharpe_ratio(
            returns=returns,
            risk_free_rate=risk_free_rate
        )

        assert isinstance(sharpe_ratio, float)
        # Sharpe ratio devrait être calculé correctement
        assert -5 <= sharpe_ratio <= 5  # Plage raisonnable

    @pytest.mark.asyncio
    async def test_walk_forward_analysis(self, backtest_service, sample_market_data, sample_strategy):
        """Test analyse walk-forward."""
        wf_config = WalkForwardConfig(
            training_period_days=120,  # 4 mois d'entraînement
            testing_period_days=30,   # 1 mois de test
            step_size_days=30,        # Avancer de 1 mois
            min_trades_per_period=5
        )

        # Mock strategy training
        sample_strategy.retrain = AsyncMock()

        results = await backtest_service.run_walk_forward_analysis(
            strategy=sample_strategy,
            market_data=sample_market_data,
            config=wf_config
        )

        assert isinstance(results, list)
        assert len(results) > 0

        # Chaque résultat devrait être un backtest complet
        for result in results:
            assert isinstance(result, BacktestResult)
            assert result.metrics is not None

    @pytest.mark.asyncio
    async def test_monte_carlo_simulation(self, backtest_service, sample_market_data, sample_strategy):
        """Test simulation Monte Carlo."""
        mc_config = MonteCarloConfig(
            n_simulations=100,  # Réduit pour les tests
            bootstrap_block_size=10,
            confidence_levels=[0.05, 0.95],
            preserve_correlation=True
        )

        results = await backtest_service.run_monte_carlo_simulation(
            strategy=sample_strategy,
            base_market_data=sample_market_data,
            config=mc_config
        )

        assert isinstance(results, dict)
        assert 'simulations' in results
        assert 'statistics' in results
        assert 'confidence_intervals' in results

        # Vérifier le nombre de simulations
        assert len(results['simulations']) == mc_config.n_simulations

        # Vérifier les statistiques
        stats = results['statistics']
        assert 'mean_return' in stats
        assert 'std_return' in stats
        assert 'sharpe_ratio' in stats

    @pytest.mark.asyncio
    async def test_multiple_strategies_backtest(self, backtest_service, sample_market_data):
        """Test backtest avec stratégies multiples."""
        # Créer plusieurs stratégies mockées
        strategy1 = Mock(spec=Strategy)
        strategy1.id = "strategy_1"
        strategy1.generate_signals = AsyncMock()

        strategy2 = Mock(spec=Strategy)
        strategy2.id = "strategy_2"
        strategy2.generate_signals = AsyncMock()

        strategies = [strategy1, strategy2]

        # Allocation du capital entre stratégies
        allocations = {"strategy_1": 0.6, "strategy_2": 0.4}

        result = await backtest_service.run_multi_strategy_backtest(
            strategies=strategies,
            allocations=allocations,
            market_data=sample_market_data,
            initial_capital=Decimal("100000")
        )

        assert isinstance(result, BacktestResult)
        assert result.metrics is not None

        # Vérifier que les deux stratégies ont été utilisées
        strategy1.generate_signals.assert_called()
        strategy2.generate_signals.assert_called()

    @pytest.mark.asyncio
    async def test_performance_attribution(self, backtest_service):
        """Test attribution de performance par facteur."""
        trades = [
            TradeExecution(
                timestamp=datetime(2023, 1, 1),
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                quantity=Decimal("1.0"),
                executed_price=Decimal("30000"),
                commission=Decimal("30"),
                status=OrderStatus.FILLED
            ),
            TradeExecution(
                timestamp=datetime(2023, 1, 15),
                symbol="ETH/USDT",
                side=OrderSide.BUY,
                quantity=Decimal("10.0"),
                executed_price=Decimal("2000"),
                commission=Decimal("20"),
                status=OrderStatus.FILLED
            )
        ]

        # Mock factor returns
        factor_returns = {
            'market': pd.Series([0.01, 0.02, -0.01, 0.015]),
            'size': pd.Series([0.005, -0.008, 0.012, 0.003]),
            'momentum': pd.Series([-0.002, 0.010, 0.005, -0.008])
        }

        attribution = await backtest_service.calculate_performance_attribution(
            trades=trades,
            factor_returns=factor_returns
        )

        assert isinstance(attribution, dict)
        assert 'factor_contributions' in attribution
        assert 'alpha_contribution' in attribution
        assert 'total_attribution' in attribution

    @pytest.mark.asyncio
    async def test_liquidity_modeling(self, backtest_service):
        """Test modélisation de la liquidité dans le backtest."""
        # Signal avec grosse taille
        large_signal = Signal(
            timestamp=datetime(2023, 6, 1),
            symbol="BTC/USDT",
            action=SignalAction.BUY,
            strength=0.8,
            price=Decimal("30000"),
            size=Decimal("10.0"),  # Gros ordre
            metadata={}
        )

        # Market data avec volume limité
        low_volume_data = MarketDataPoint(
            timestamp=datetime(2023, 6, 1),
            symbol="BTC/USDT",
            open_price=Decimal("30000"),
            high_price=Decimal("30200"),
            low_price=Decimal("29800"),
            close_price=Decimal("30100"),
            volume=Decimal("5.0")  # Volume très faible
        )

        # L'exécution devrait être impactée par la liquidité
        execution = await backtest_service._execute_with_liquidity_impact(
            signal=large_signal,
            market_data=low_volume_data
        )

        # Le prix d'exécution devrait être pire à cause de l'impact sur le marché
        assert execution.executed_price > large_signal.price * Decimal("1.01")  # Au moins 1% de slippage

    @pytest.mark.asyncio
    async def test_transaction_costs_modeling(self, backtest_service, backtest_config):
        """Test modélisation détaillée des coûts de transaction."""
        signal = Signal(
            timestamp=datetime(2023, 6, 1),
            symbol="BTC/USDT",
            action=SignalAction.BUY,
            strength=0.5,
            price=Decimal("30000"),
            size=Decimal("5.0"),
            metadata={}
        )

        costs = await backtest_service._calculate_transaction_costs(
            signal=signal,
            config=backtest_config
        )

        assert isinstance(costs, dict)
        assert 'commission' in costs
        assert 'slippage' in costs
        assert 'market_impact' in costs
        assert 'total_cost' in costs

        # Tous les coûts devraient être positifs
        assert all(cost >= 0 for cost in costs.values())

    @pytest.mark.asyncio
    async def test_backtest_resumption(self, backtest_service, backtest_config):
        """Test reprise d'un backtest interrompu."""
        # Créer un backtest partiellement complété
        partial_result = BacktestResult(
            id="test_backtest",
            config=backtest_config,
            status=BacktestStatus.RUNNING,
            start_time=datetime.now() - timedelta(hours=2),
            progress=0.6,  # 60% complété
            trades=[],
            metrics=None
        )

        # Reprendre le backtest
        resumed_result = await backtest_service.resume_backtest(
            backtest_id="test_backtest",
            partial_result=partial_result
        )

        assert resumed_result.status in [BacktestStatus.COMPLETED, BacktestStatus.RUNNING]
        assert resumed_result.progress >= partial_result.progress

    @pytest.mark.asyncio
    async def test_real_time_backtest_monitoring(self, backtest_service, backtest_config):
        """Test monitoring en temps réel du backtest."""
        # Simuler un backtest en cours
        backtest_id = "test_backtest_monitor"

        # Mock d'un callback de monitoring
        monitoring_callback = AsyncMock()

        # Démarrer le monitoring
        monitor_task = asyncio.create_task(
            backtest_service.monitor_backtest_progress(
                backtest_id=backtest_id,
                callback=monitoring_callback,
                update_interval=0.1  # 100ms pour les tests
            )
        )

        # Laisser le monitoring tourner un peu
        await asyncio.sleep(0.5)

        # Arrêter le monitoring
        monitor_task.cancel()

        # Vérifier que le callback a été appelé
        assert monitoring_callback.called

    def test_backtest_validation_insufficient_data(self, backtest_service):
        """Test validation avec données insuffisantes."""
        insufficient_data = []  # Pas de données

        with pytest.raises(ValueError, match="Insufficient market data"):
            asyncio.run(backtest_service._validate_market_data(insufficient_data))

    def test_backtest_validation_invalid_dates(self, backtest_config):
        """Test validation avec dates invalides."""
        # Dates inversées
        invalid_config = backtest_config
        invalid_config.start_date = datetime(2023, 12, 31)
        invalid_config.end_date = datetime(2023, 1, 1)

        with pytest.raises(ValueError, match="start_date must be before end_date"):
            # Validation dans le service
            pass

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_backtest_performance_large_dataset(self, backtest_service, sample_strategy):
        """Test performance avec un large dataset."""
        import time

        # Créer un dataset de 5 ans (environ 1825 points)
        large_dataset = []
        start_date = datetime(2019, 1, 1)

        for i in range(1825):
            data_point = MarketDataPoint(
                timestamp=start_date + timedelta(days=i),
                symbol="BTC/USDT",
                open_price=Decimal("30000"),
                high_price=Decimal("30500"),
                low_price=Decimal("29500"),
                close_price=Decimal("30100"),
                volume=Decimal("1000000")
            )
            large_dataset.append(data_point)

        # Mock strategy
        sample_strategy.generate_signals.return_value = []

        start_time = time.time()

        # Exécuter le backtest
        config = BacktestConfiguration(
            name="Performance Test",
            strategy_id="test_strategy",
            start_date=datetime(2019, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=Decimal("100000"),
            symbols=["BTC/USDT"]
        )

        result = await backtest_service.run_backtest(config, large_dataset)

        execution_time = time.time() - start_time

        # Le backtest devrait être raisonnablement rapide même avec beaucoup de données
        assert execution_time < 30.0  # Moins de 30 secondes pour 5 ans de données
        assert result.status == BacktestStatus.COMPLETED


class TestBacktestReporting:
    """Tests pour le reporting des résultats de backtest."""

    @pytest.fixture
    def sample_backtest_result(self):
        """Résultat de backtest échantillon."""
        return BacktestResult(
            id="test_backtest_001",
            config=BacktestConfiguration(
                name="Test Strategy Backtest",
                strategy_id="test_strategy",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
                initial_capital=Decimal("100000"),
                symbols=["BTC/USDT"]
            ),
            status=BacktestStatus.COMPLETED,
            start_time=datetime(2023, 1, 1, 9, 0),
            end_time=datetime(2023, 1, 1, 9, 30),
            progress=1.0,
            trades=[],
            metrics=BacktestMetrics(
                total_return=Decimal("0.15"),  # 15% return
                sharpe_ratio=1.2,
                max_drawdown=Decimal("0.08"),  # 8% max drawdown
                sortino_ratio=1.5,
                calmar_ratio=1.875,  # 15% / 8%
                total_trades=150,
                win_rate=0.65,
                profit_factor=1.8
            )
        )

    def test_generate_backtest_report(self, backtest_service, sample_backtest_result):
        """Test génération de rapport de backtest."""
        report = backtest_service.generate_report(sample_backtest_result)

        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'performance_metrics' in report
        assert 'trade_analysis' in report
        assert 'risk_analysis' in report

        # Vérifier les métriques clés dans le résumé
        summary = report['summary']
        assert summary['total_return'] == "15.00%"
        assert summary['sharpe_ratio'] == 1.2
        assert summary['max_drawdown'] == "8.00%"

    def test_export_backtest_results(self, backtest_service, sample_backtest_result):
        """Test export des résultats vers différents formats."""
        # Export CSV
        csv_data = backtest_service.export_to_csv(sample_backtest_result)
        assert isinstance(csv_data, str)
        assert "total_return" in csv_data

        # Export JSON
        json_data = backtest_service.export_to_json(sample_backtest_result)
        assert isinstance(json_data, str)
        assert "15.00%" in json_data  # Total return

    def test_performance_comparison(self, backtest_service):
        """Test comparaison de performance entre backtests."""
        # Créer deux résultats différents
        result1 = Mock()
        result1.metrics.sharpe_ratio = 1.2
        result1.metrics.max_drawdown = Decimal("0.08")

        result2 = Mock()
        result2.metrics.sharpe_ratio = 0.9
        result2.metrics.max_drawdown = Decimal("0.12")

        comparison = backtest_service.compare_backtests([result1, result2])

        assert isinstance(comparison, dict)
        assert 'best_sharpe' in comparison
        assert 'best_drawdown' in comparison

        # Result1 devrait être meilleur sur les deux métriques
        assert comparison['best_sharpe'] == 0  # Index de result1
        assert comparison['best_drawdown'] == 0


# Helpers et utilitaires

def create_synthetic_price_series(start_price=50000, n_points=252, volatility=0.02):
    """Crée une série de prix synthétique réaliste."""
    np.random.seed(42)
    returns = np.random.normal(0.0005, volatility, n_points)
    prices = [start_price]

    for ret in returns:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)

    return prices


def generate_test_signals(dates, symbols, signal_frequency=0.1):
    """Génère des signaux de test pour le backtesting."""
    np.random.seed(42)
    signals = []

    for date in dates:
        if np.random.random() < signal_frequency:
            for symbol in symbols:
                action = np.random.choice([SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD])
                if action != SignalAction.HOLD:
                    signal = Signal(
                        timestamp=date,
                        symbol=symbol,
                        action=action,
                        strength=np.random.uniform(0.3, 0.9),
                        price=Decimal(str(np.random.uniform(25000, 75000))),
                        size=Decimal(str(np.random.uniform(0.05, 0.15))),
                        metadata={}
                    )
                    signals.append(signal)

    return signals