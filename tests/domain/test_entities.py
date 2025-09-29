"""
Tests for Domain Entities
=========================

Suite de tests complète pour les entités du domaine métier.
"""

import pytest
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any

from qframe.domain.entities.portfolio import Portfolio
from qframe.domain.entities.order import Order, OrderStatus, OrderSide, OrderType
from qframe.domain.entities.position import Position
from qframe.domain.entities.strategy import Strategy, StrategyStatus, StrategyType
from qframe.domain.entities.backtest import BacktestConfiguration, BacktestResult, BacktestMetrics, BacktestStatus, BacktestType
from qframe.domain.entities.risk_assessment import RiskAssessment, RiskLevel
from qframe.domain.value_objects.signal import Signal, SignalAction, SignalConfidence
from qframe.domain.value_objects.performance_metrics import PerformanceMetrics


class TestPortfolio:
    """Tests pour l'entité Portfolio"""

    def test_portfolio_creation(self):
        """Test la création d'un portfolio"""
        portfolio = Portfolio(
            id="port-001",
            name="Test Portfolio",
            initial_capital=Decimal("10000.00"),
            base_currency="USD"
        )

        assert portfolio.id == "port-001"
        assert portfolio.name == "Test Portfolio"
        assert portfolio.initial_capital == Decimal("10000.00")
        assert portfolio.base_currency == "USD"
        assert portfolio.total_value == Decimal("10000.00")
        assert portfolio.created_at is not None
        assert portfolio.updated_at is not None

    def test_portfolio_with_minimal_data(self):
        """Test création avec données minimales"""
        portfolio = Portfolio(
            id="min-port",
            name="Minimal",
            initial_capital=Decimal("1000"),
            base_currency="EUR"
        )

        assert portfolio.id == "min-port"
        assert portfolio.total_value == portfolio.initial_capital

    def test_portfolio_value_calculation(self):
        """Test le calcul de valeur du portfolio"""
        portfolio = Portfolio(
            id="calc-port",
            name="Calculation Test",
            initial_capital=Decimal("5000.00"),
            base_currency="USD"
        )

        # Test mise à jour de la valeur
        portfolio.current_value = Decimal("5500.00")
        assert portfolio.current_value == Decimal("5500.00")

        # Calcul du P&L
        pnl = portfolio.current_value - portfolio.initial_capital
        assert pnl == Decimal("500.00")

    def test_portfolio_performance_metrics(self):
        """Test l'ajout de métriques de performance"""
        portfolio = Portfolio(
            id="perf-port",
            name="Performance Test",
            initial_capital=Decimal("10000"),
            base_currency="USD"
        )

        # Simuler une valeur totale avec positions
        portfolio.total_value = Decimal("11500")  # Gain de 15%

        # Vérifier la valeur de marché
        assert portfolio.total_value == Decimal("11500")

        # Calculer le retour
        portfolio_return = (portfolio.total_value - portfolio.initial_capital) / portfolio.initial_capital
        assert portfolio_return == Decimal("0.15")  # 15%

    def test_portfolio_timestamps(self):
        """Test la gestion des timestamps"""
        before_creation = datetime.utcnow()

        portfolio = Portfolio(
            id="time-port",
            name="Timestamp Test",
            initial_capital=Decimal("1000"),
            base_currency="USD"
        )

        after_creation = datetime.utcnow()

        assert before_creation <= portfolio.created_at <= after_creation
        assert before_creation <= portfolio.updated_at <= after_creation
        # Les timestamps peuvent être très proches mais pas exactement égaux
        time_diff = abs((portfolio.updated_at - portfolio.created_at).total_seconds())
        assert time_diff < 1.0  # Moins d'une seconde de différence


class TestOrder:
    """Tests pour l'entité Order"""

    def test_order_creation(self):
        """Test la création d'un ordre"""
        order = Order(
            id="order-001",
            portfolio_id="port-001",
            symbol="BTCUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            created_time=datetime.utcnow()
        )

        assert order.id == "order-001"
        assert order.portfolio_id == "port-001"
        assert order.symbol == "BTCUSD"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.quantity == Decimal("0.1")
        assert order.status == OrderStatus.PENDING  # Status par défaut

    def test_order_with_limit_price(self):
        """Test ordre avec prix limite"""
        order = Order(
            id="limit-order",
            portfolio_id="port-001",
            symbol="ETHUSD",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("2000.00"),
            created_time=datetime.utcnow()
        )

        assert order.order_type == OrderType.LIMIT
        assert order.price == Decimal("2000.00")

    def test_order_status_transitions(self):
        """Test les transitions de statut d'ordre"""
        order = Order(
            id="status-order",
            portfolio_id="port-001",
            symbol="ADAUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100"),
            created_time=datetime.utcnow()
        )

        # État initial
        assert order.status == OrderStatus.PENDING

        # Transition vers filled
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_time = datetime.utcnow()

        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == order.quantity
        assert order.filled_time is not None

    def test_order_partial_fill(self):
        """Test remplissage partiel d'ordre"""
        order = Order(
            id="partial-order",
            portfolio_id="port-001",
            symbol="DOTUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("50"),
            price=Decimal("10.00"),
            created_time=datetime.utcnow()
        )

        # Remplissage partiel
        order.status = OrderStatus.PARTIALLY_FILLED
        order.filled_quantity = Decimal("30")
        order.average_price = Decimal("9.95")

        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.filled_quantity == Decimal("30")
        assert order.filled_quantity < order.quantity

    def test_order_cancellation(self):
        """Test annulation d'ordre"""
        order = Order(
            id="cancel-order",
            portfolio_id="port-001",
            symbol="SOLUSD",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("5"),
            price=Decimal("100.00"),
            created_time=datetime.utcnow()
        )

        # Annulation
        order.status = OrderStatus.CANCELLED
        order.cancelled_time = datetime.utcnow()

        assert order.status == OrderStatus.CANCELLED
        assert order.cancelled_time is not None


class TestPosition:
    """Tests pour l'entité Position"""

    def test_position_creation(self):
        """Test la création d'une position"""
        position = Position(
            symbol="BTCUSD",
            quantity=Decimal("0.5"),
            average_price=Decimal("50000.00"),
            current_price=Decimal("52000.00")
        )

        assert position.symbol == "BTCUSD"
        assert position.quantity == Decimal("0.5")
        assert position.average_price == Decimal("50000.00")
        assert position.current_price == Decimal("52000.00")

    def test_position_pnl_calculation(self):
        """Test le calcul de P&L via market value"""
        position = Position(
            symbol="ETHUSD",
            quantity=Decimal("2.0"),
            average_price=Decimal("1800.00"),
            current_price=Decimal("2000.00")
        )

        # Market value = current_price * quantity
        expected_market_value = Decimal("2000.00") * Decimal("2.0")
        assert position.market_value == expected_market_value

    def test_position_long_vs_short(self):
        """Test positions longues vs courtes"""
        # Position longue
        long_position = Position(
            symbol="ADAUSD",
            quantity=Decimal("1000"),  # Positive = long
            average_price=Decimal("0.50"),
            current_price=Decimal("0.55")
        )

        # Position courte
        short_position = Position(
            symbol="DOTUSD",
            quantity=Decimal("-100"),  # Négative = short
            average_price=Decimal("8.00"),
            current_price=Decimal("7.50")
        )

        assert long_position.quantity > 0  # Long
        assert short_position.quantity < 0  # Short

    def test_position_update_price(self):
        """Test mise à jour du prix"""
        position = Position(
            symbol="SOLUSD",
            quantity=Decimal("10"),
            average_price=Decimal("100.00"),
            current_price=Decimal("100.00")
        )

        # Mise à jour du prix
        old_market_value = position.market_value
        position.current_price = Decimal("110.00")
        new_market_value = position.market_value

        assert position.current_price == Decimal("110.00")
        assert new_market_value > old_market_value


class TestStrategy:
    """Tests pour l'entité Strategy"""

    def test_strategy_creation(self):
        """Test la création d'une stratégie"""
        strategy = Strategy(
            id="strat-001",
            name="DMN LSTM",
            description="Deep Market Network with LSTM",
            strategy_type=StrategyType.MACHINE_LEARNING,
            status=StrategyStatus.ACTIVE
        )

        assert strategy.id == "strat-001"
        assert strategy.name == "DMN LSTM"
        assert strategy.strategy_type == StrategyType.MACHINE_LEARNING
        assert strategy.status == StrategyStatus.ACTIVE

    def test_strategy_configuration(self):
        """Test la configuration de stratégie"""
        config = {
            "window_size": 64,
            "hidden_size": 128,
            "learning_rate": 0.001
        }

        strategy = Strategy(
            id="config-strat",
            name="Configured Strategy",
            description="Test configuration",
            strategy_type=StrategyType.MACHINE_LEARNING,
            status=StrategyStatus.ACTIVE,
            parameters=config
        )

        assert strategy.parameters == config
        assert strategy.parameters["window_size"] == 64

    def test_strategy_performance_tracking(self):
        """Test le suivi de performance"""
        strategy = Strategy(
            id="perf-strat",
            name="Performance Strategy",
            description="Performance tracking test",
            strategy_type=StrategyType.MEAN_REVERSION,
            status=StrategyStatus.ACTIVE
        )

        # Utiliser les attributs existants de performance
        strategy.sharpe_ratio = Decimal("1.8")
        strategy.max_drawdown = Decimal("0.08")
        strategy.total_trades = 100
        strategy.winning_trades = 65

        assert strategy.sharpe_ratio == Decimal("1.8")
        assert strategy.max_drawdown == Decimal("0.08")
        assert strategy.total_trades == 100
        assert strategy.winning_trades == 65

    def test_strategy_status_transitions(self):
        """Test les transitions de statut"""
        strategy = Strategy(
            id="status-strat",
            name="Status Strategy",
            description="Status transition test",
            strategy_type=StrategyType.ARBITRAGE,
            status=StrategyStatus.INACTIVE
        )

        # Activation
        strategy.status = StrategyStatus.ACTIVE
        assert strategy.status == StrategyStatus.ACTIVE

        # Pause
        strategy.status = StrategyStatus.PAUSED
        assert strategy.status == StrategyStatus.PAUSED


class TestBacktestConfiguration:
    """Tests pour l'entité BacktestConfiguration"""

    def test_backtest_configuration_creation(self):
        """Test la création d'une configuration de backtest"""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)

        config = BacktestConfiguration(
            id="bt-001",
            name="Annual Backtest",
            start_date=start_date,
            end_date=end_date,
            initial_capital=Decimal("100000"),
            strategy_ids=["strat-001"]
        )

        assert config.id == "bt-001"
        assert "strat-001" in config.strategy_ids
        assert config.start_date == start_date
        assert config.end_date == end_date
        assert config.initial_capital == Decimal("100000")

    def test_backtest_configuration_validation(self):
        """Test la validation d'une configuration"""
        config = BacktestConfiguration(
            id="valid-bt",
            name="Valid Config",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=Decimal("100000"),
            strategy_ids=["strat-001", "strat-002"],
            strategy_allocations={"strat-001": Decimal("0.6"), "strat-002": Decimal("0.4")}
        )

        errors = config.validate()
        assert len(errors) == 0  # Configuration valide

    def test_backtest_configuration_invalid_dates(self):
        """Test la validation avec dates invalides"""
        config = BacktestConfiguration(
            id="invalid-bt",
            name="Invalid Dates",
            start_date=datetime(2023, 12, 31),
            end_date=datetime(2023, 1, 1),  # End date before start date
            initial_capital=Decimal("100000"),
            strategy_ids=["strat-001"]
        )

        errors = config.validate()
        assert any("End date must be after start date" in error for error in errors)

    def test_backtest_configuration_multi_strategy(self):
        """Test configuration multi-stratégies"""
        config = BacktestConfiguration(
            id="multi-bt",
            name="Multi Strategy Test",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=Decimal("200000"),
            strategy_ids=["mean_reversion", "momentum", "arbitrage"],
            strategy_allocations={
                "mean_reversion": Decimal("0.4"),
                "momentum": Decimal("0.4"),
                "arbitrage": Decimal("0.2")
            },
            backtest_type=BacktestType.MULTI_STRATEGY
        )

        assert len(config.strategy_ids) == 3
        assert config.backtest_type == BacktestType.MULTI_STRATEGY
        assert sum(config.strategy_allocations.values()) == Decimal("1.0")


class TestRiskAssessment:
    """Tests pour l'entité RiskAssessment"""

    def test_risk_assessment_creation(self):
        """Test la création d'une évaluation de risque"""
        assessment = RiskAssessment(
            id="risk-001",
            target_id="port-001",
            assessment_type="portfolio",
            overall_risk_level=RiskLevel.MEDIUM,
            risk_score=Decimal("65")
        )

        assert assessment.id == "risk-001"
        assert assessment.target_id == "port-001"
        assert assessment.assessment_type == "portfolio"
        assert assessment.overall_risk_level == RiskLevel.MEDIUM
        assert assessment.risk_score == Decimal("65")

    def test_risk_levels(self):
        """Test les différents niveaux de risque"""
        # Risque faible
        low_risk = RiskAssessment(
            id="low-risk",
            target_id="port-001",
            assessment_type="portfolio",
            overall_risk_level=RiskLevel.LOW,
            risk_score=Decimal("25")
        )

        # Risque élevé
        high_risk = RiskAssessment(
            id="high-risk",
            target_id="port-002",
            assessment_type="portfolio",
            overall_risk_level=RiskLevel.HIGH,
            risk_score=Decimal("85")
        )

        assert low_risk.overall_risk_level == RiskLevel.LOW
        assert high_risk.overall_risk_level == RiskLevel.HIGH
        assert high_risk.risk_score > low_risk.risk_score

    def test_risk_metrics_validation(self):
        """Test la validation des métriques de risque"""
        from qframe.domain.entities.risk_assessment import RiskMetric, RiskType

        assessment = RiskAssessment(
            id="metrics-risk",
            target_id="port-001",
            assessment_type="portfolio",
            overall_risk_level=RiskLevel.MEDIUM,
            risk_score=Decimal("60")
        )

        # Ajouter des métriques de risque
        var_metric = RiskMetric(
            name="VaR_95",
            value=Decimal("1000.00"),
            threshold=Decimal("1200.00"),
            risk_level=RiskLevel.MEDIUM,
            risk_type=RiskType.MARKET
        )

        assessment.risk_metrics["var_95"] = var_metric
        assert "var_95" in assessment.risk_metrics
        assert assessment.risk_metrics["var_95"].value == Decimal("1000.00")

        # Les valeurs de risque devraient être valides
        assert assessment.risk_score >= 0
        assert assessment.risk_score <= 100


class TestSignal:
    """Tests pour les signaux (value object)"""

    def test_signal_creation(self):
        """Test la création d'un signal"""
        signal = Signal(
            symbol="BTCUSD",
            action=SignalAction.BUY,
            timestamp=datetime.utcnow(),
            strength=Decimal("0.8"),
            confidence=SignalConfidence.HIGH,
            price=Decimal("50000.00"),
            strategy_id="dmn_lstm"
        )

        assert signal.symbol == "BTCUSD"
        assert signal.action == SignalAction.BUY
        assert signal.strength == Decimal("0.8")
        assert signal.confidence == SignalConfidence.HIGH
        assert signal.price == Decimal("50000.00")

    def test_signal_with_metadata(self):
        """Test signal avec métadonnées"""
        metadata = {
            "prediction": 0.75,
            "model_confidence": 0.85,
            "features_used": ["rsi", "macd", "volume"]
        }

        signal = Signal(
            symbol="ETHUSD",
            action=SignalAction.SELL,
            timestamp=datetime.utcnow(),
            strength=Decimal("0.6"),
            confidence=SignalConfidence.MEDIUM,
            price=Decimal("2000.00"),
            strategy_id="mean_reversion",
            metadata=metadata
        )

        assert signal.metadata == metadata
        assert signal.metadata["prediction"] == 0.75

    def test_signal_actions(self):
        """Test les différents types d'actions"""
        buy_signal = Signal(
            symbol="ADAUSD",
            action=SignalAction.BUY,
            timestamp=datetime.utcnow(),
            strength=Decimal("0.9"),
            confidence=SignalConfidence.VERY_HIGH,
            price=Decimal("0.50"),
            strategy_id="rl_alpha"
        )

        sell_signal = Signal(
            symbol="DOTUSD",
            action=SignalAction.SELL,
            timestamp=datetime.utcnow(),
            strength=Decimal("0.7"),
            confidence=SignalConfidence.HIGH,
            price=Decimal("8.00"),
            strategy_id="funding_arbitrage"
        )

        hold_signal = Signal(
            symbol="SOLUSD",
            action=SignalAction.HOLD,
            timestamp=datetime.utcnow(),
            strength=Decimal("0.3"),
            confidence=SignalConfidence.LOW,
            price=Decimal("100.00"),
            strategy_id="adaptive_mr"
        )

        assert buy_signal.action == SignalAction.BUY
        assert sell_signal.action == SignalAction.SELL
        assert hold_signal.action == SignalAction.HOLD

    def test_signal_confidence_levels(self):
        """Test les niveaux de confiance"""
        very_high = SignalConfidence.VERY_HIGH
        high = SignalConfidence.HIGH
        medium = SignalConfidence.MEDIUM
        low = SignalConfidence.LOW

        # Test que les niveaux existent
        assert very_high is not None
        assert high is not None
        assert medium is not None
        assert low is not None


class TestPerformanceMetrics:
    """Tests pour les métriques de performance"""

    def test_performance_metrics_creation(self):
        """Test la création de métriques de performance"""
        metrics = PerformanceMetrics(
            total_return=Decimal("0.15"),
            sharpe_ratio=Decimal("1.2"),
            sortino_ratio=Decimal("1.8"),
            max_drawdown=Decimal("0.08"),
            volatility=Decimal("0.12"),
            alpha=Decimal("0.03"),
            beta=Decimal("0.95")
        )

        assert metrics.total_return == Decimal("0.15")
        assert metrics.sharpe_ratio == Decimal("1.2")
        assert metrics.max_drawdown == Decimal("0.08")

    def test_performance_metrics_calculation(self):
        """Test le calcul de métriques dérivées"""
        metrics = PerformanceMetrics(
            total_return=Decimal("0.20"),
            sharpe_ratio=Decimal("1.5"),
            max_drawdown=Decimal("0.10"),
            volatility=Decimal("0.15")
        )

        # Calculs dérivés
        calmar_ratio = metrics.total_return / metrics.max_drawdown
        assert calmar_ratio == Decimal("2.0")

    def test_performance_metrics_comparison(self):
        """Test la comparaison de métriques"""
        metrics_a = PerformanceMetrics(
            total_return=Decimal("0.15"),
            sharpe_ratio=Decimal("1.2"),
            max_drawdown=Decimal("0.05")
        )

        metrics_b = PerformanceMetrics(
            total_return=Decimal("0.10"),
            sharpe_ratio=Decimal("1.5"),
            max_drawdown=Decimal("0.03")
        )

        # Comparaisons
        assert metrics_a.total_return > metrics_b.total_return
        assert metrics_b.sharpe_ratio > metrics_a.sharpe_ratio
        assert metrics_b.max_drawdown < metrics_a.max_drawdown