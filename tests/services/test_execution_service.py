"""
Tests for Execution Service
===========================

Suite de tests complète pour le service d'exécution des ordres.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from decimal import Decimal
from typing import Dict, List

from qframe.domain.entities.order import Order, OrderStatus, OrderSide, OrderType
from qframe.domain.entities.portfolio import Portfolio
from qframe.domain.entities.position import Position
from qframe.domain.value_objects.signal import Signal, SignalAction, SignalConfidence
from qframe.domain.services.execution_service import ExecutionService
from qframe.domain.repositories.order_repository import OrderRepository
from qframe.domain.repositories.portfolio_repository import PortfolioRepository
from qframe.infrastructure.external.order_execution_adapter import BrokerAdapter
from qframe.core.interfaces import MetricsCollector


class TestExecutionService:
    """Tests pour le service d'exécution"""

    @pytest.fixture
    def mock_order_repository(self):
        """Repository d'ordres mocké"""
        return Mock(spec=OrderRepository)

    @pytest.fixture
    def mock_portfolio_repository(self):
        """Repository de portfolios mocké"""
        return Mock(spec=PortfolioRepository)

    @pytest.fixture
    def mock_broker_adapter(self):
        """Adaptateur broker mocké"""
        return Mock(spec=BrokerAdapter)

    @pytest.fixture
    def mock_metrics(self):
        """Collecteur de métriques mocké"""
        return Mock(spec=MetricsCollector)

    @pytest.fixture
    def execution_service(self, mock_order_repository, mock_portfolio_repository,
                         mock_broker_adapter, mock_metrics):
        """Service d'exécution initialisé"""
        return ExecutionService(
            order_repository=mock_order_repository,
            portfolio_repository=mock_portfolio_repository,
            broker_adapter=mock_broker_adapter,
            metrics_collector=mock_metrics
        )

    @pytest.fixture
    def sample_portfolio(self):
        """Portfolio de test"""
        portfolio = Portfolio(
            id="port-001",
            name="Execution Test Portfolio",
            initial_capital=Decimal("100000"),
            base_currency="USD"
        )
        portfolio.cash_balance = Decimal("50000")
        return portfolio

    @pytest.fixture
    def sample_signal(self):
        """Signal de test"""
        return Signal(
            symbol="BTCUSD",
            action=SignalAction.BUY,
            timestamp=datetime.utcnow(),
            strength=Decimal("0.8"),
            confidence=SignalConfidence.HIGH,
            price=Decimal("50000"),
            strategy_id="test-strategy"
        )

    def test_create_order_from_signal(self, execution_service, mock_portfolio_repository,
                                    sample_portfolio, sample_signal):
        """Test création d'ordre depuis signal"""
        mock_portfolio_repository.get_by_id.return_value = sample_portfolio

        order = execution_service.create_order_from_signal(
            portfolio_id=sample_portfolio.id,
            signal=sample_signal,
            position_size=Decimal("0.02")  # 2% du portfolio
        )

        assert isinstance(order, Order)
        assert order.symbol == sample_signal.symbol
        assert order.side == OrderSide.BUY
        assert order.portfolio_id == sample_portfolio.id
        assert order.quantity > 0

    def test_execute_market_order(self, execution_service, mock_order_repository,
                                mock_broker_adapter, sample_portfolio):
        """Test exécution ordre au marché"""
        order = Order(
            id="order-001",
            portfolio_id=sample_portfolio.id,
            symbol="BTCUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            created_time=datetime.utcnow()
        )

        # Mock broker response
        mock_broker_adapter.execute_order.return_value = {
            "order_id": "broker-001",
            "status": "filled",
            "filled_quantity": Decimal("0.1"),
            "average_price": Decimal("50000"),
            "commission": Decimal("25")
        }

        mock_order_repository.save.return_value = order

        result = execution_service.execute_order(order)

        assert result is not None
        assert result.status == OrderStatus.FILLED
        mock_broker_adapter.execute_order.assert_called_once()

    def test_execute_limit_order(self, execution_service, mock_order_repository,
                                mock_broker_adapter, sample_portfolio):
        """Test exécution ordre à cours limité"""
        order = Order(
            id="order-002",
            portfolio_id=sample_portfolio.id,
            symbol="ETHUSD",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("5.0"),
            price=Decimal("3200"),
            created_time=datetime.utcnow()
        )

        # Mock broker response - ordre en attente
        mock_broker_adapter.execute_order.return_value = {
            "order_id": "broker-002",
            "status": "pending",
            "filled_quantity": Decimal("0"),
            "average_price": None,
            "commission": Decimal("0")
        }

        mock_order_repository.save.return_value = order

        result = execution_service.execute_order(order)

        assert result.status == OrderStatus.PENDING
        assert result.filled_quantity == Decimal("0")

    def test_partial_fill_execution(self, execution_service, mock_order_repository,
                                  mock_broker_adapter, sample_portfolio):
        """Test exécution partielle"""
        order = Order(
            id="order-003",
            portfolio_id=sample_portfolio.id,
            symbol="ADAUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1000"),
            created_time=datetime.utcnow()
        )

        # Mock broker response - exécution partielle
        mock_broker_adapter.execute_order.return_value = {
            "order_id": "broker-003",
            "status": "partially_filled",
            "filled_quantity": Decimal("600"),
            "average_price": Decimal("1.2"),
            "commission": Decimal("0.72")
        }

        mock_order_repository.save.return_value = order

        result = execution_service.execute_order(order)

        assert result.status == OrderStatus.PARTIALLY_FILLED
        assert result.filled_quantity == Decimal("600")
        assert result.remaining_quantity == Decimal("400")

    def test_order_cancellation(self, execution_service, mock_order_repository,
                               mock_broker_adapter):
        """Test annulation d'ordre"""
        order = Order(
            id="order-004",
            portfolio_id="port-001",
            symbol="SOLUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("10"),
            price=Decimal("100"),
            status=OrderStatus.PENDING,
            created_time=datetime.utcnow()
        )

        mock_order_repository.get_by_id.return_value = order
        mock_broker_adapter.cancel_order.return_value = {"status": "cancelled"}
        mock_order_repository.save.return_value = order

        result = execution_service.cancel_order(order.id)

        assert result.status == OrderStatus.CANCELLED
        mock_broker_adapter.cancel_order.assert_called_once()

    def test_position_update_after_fill(self, execution_service, mock_portfolio_repository,
                                      sample_portfolio):
        """Test mise à jour position après exécution"""
        # Ordre d'achat
        filled_order = Order(
            id="order-005",
            portfolio_id=sample_portfolio.id,
            symbol="BTCUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.5"),
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("0.5"),
            average_fill_price=Decimal("50000"),
            created_time=datetime.utcnow()
        )

        mock_portfolio_repository.get_by_id.return_value = sample_portfolio
        mock_portfolio_repository.save.return_value = sample_portfolio

        execution_service.update_portfolio_position(filled_order)

        # Vérifier que la position a été ajoutée
        assert "BTCUSD" in sample_portfolio.positions
        position = sample_portfolio.positions["BTCUSD"]
        assert position.quantity == Decimal("0.5")
        assert position.average_price == Decimal("50000")

    def test_cash_balance_update(self, execution_service, mock_portfolio_repository,
                               sample_portfolio):
        """Test mise à jour du cash après exécution"""
        initial_cash = sample_portfolio.cash_balance

        # Ordre d'achat qui consomme du cash
        filled_order = Order(
            id="order-006",
            portfolio_id=sample_portfolio.id,
            symbol="ETHUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("10"),
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("10"),
            average_fill_price=Decimal("3000"),
            commission=Decimal("15"),
            created_time=datetime.utcnow()
        )

        mock_portfolio_repository.get_by_id.return_value = sample_portfolio
        mock_portfolio_repository.save.return_value = sample_portfolio

        execution_service.update_portfolio_position(filled_order)

        # Cash devrait diminuer
        expected_cash = initial_cash - (Decimal("10") * Decimal("3000")) - Decimal("15")
        assert sample_portfolio.cash_balance == expected_cash

    def test_risk_checks_before_execution(self, execution_service, mock_portfolio_repository,
                                        sample_portfolio):
        """Test vérifications de risque avant exécution"""
        # Ordre qui dépasserait les limites
        large_order = Order(
            id="order-007",
            portfolio_id=sample_portfolio.id,
            symbol="BTCUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("10"),  # Très gros ordre
            created_time=datetime.utcnow()
        )

        mock_portfolio_repository.get_by_id.return_value = sample_portfolio

        # Devrait rejeter l'ordre
        with pytest.raises(ValueError, match="insufficient.*funds|position.*size|risk.*limit"):
            execution_service.validate_order_risk(large_order)

    def test_slippage_calculation(self, execution_service):
        """Test calcul du slippage"""
        expected_price = Decimal("50000")
        actual_price = Decimal("50250")
        quantity = Decimal("1.0")

        slippage = execution_service.calculate_slippage(
            expected_price=expected_price,
            actual_price=actual_price,
            quantity=quantity
        )

        expected_slippage = (actual_price - expected_price) / expected_price
        assert abs(slippage - expected_slippage) < Decimal("0.0001")

    def test_commission_calculation(self, execution_service):
        """Test calcul des commissions"""
        order_value = Decimal("50000")  # $50k order
        commission_rate = Decimal("0.001")  # 0.1%

        commission = execution_service.calculate_commission(
            order_value=order_value,
            commission_rate=commission_rate
        )

        expected_commission = order_value * commission_rate
        assert commission == expected_commission

    def test_order_routing(self, execution_service, mock_broker_adapter):
        """Test routage des ordres"""
        btc_order = Order(
            id="order-008",
            portfolio_id="port-001",
            symbol="BTCUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            created_time=datetime.utcnow()
        )

        # Test que l'ordre est routé vers le bon venue
        venue = execution_service.select_execution_venue(btc_order)
        assert venue is not None
        assert isinstance(venue, str)

    def test_order_timing_analysis(self, execution_service):
        """Test analyse du timing d'exécution"""
        start_time = datetime.utcnow()
        end_time = datetime.utcnow()

        timing_stats = execution_service.analyze_execution_timing(
            order_created=start_time,
            order_filled=end_time
        )

        assert "execution_latency" in timing_stats
        assert "timestamp_created" in timing_stats
        assert "timestamp_filled" in timing_stats

    def test_execution_quality_metrics(self, execution_service, mock_metrics):
        """Test métriques de qualité d'exécution"""
        execution_result = {
            "expected_price": Decimal("50000"),
            "actual_price": Decimal("50100"),
            "slippage": Decimal("0.002"),
            "commission": Decimal("25"),
            "execution_time": 0.5
        }

        execution_service.record_execution_metrics(execution_result)

        # Vérifier que les métriques sont enregistrées
        mock_metrics.record_metric.assert_called()

    def test_batch_order_execution(self, execution_service, mock_order_repository,
                                 mock_broker_adapter, sample_portfolio):
        """Test exécution d'ordres en lot"""
        orders = []
        for i in range(3):
            order = Order(
                id=f"batch-order-{i}",
                portfolio_id=sample_portfolio.id,
                symbol=f"SYMBOL{i}",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
                created_time=datetime.utcnow()
            )
            orders.append(order)

        mock_broker_adapter.execute_batch_orders.return_value = [
            {"order_id": f"batch-order-{i}", "status": "filled"} for i in range(3)
        ]

        results = execution_service.execute_batch_orders(orders)

        assert len(results) == 3
        assert all(r.status == OrderStatus.FILLED for r in results)

    def test_order_state_transitions(self, execution_service):
        """Test transitions d'état des ordres"""
        order = Order(
            id="state-order",
            portfolio_id="port-001",
            symbol="BTCUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            created_time=datetime.utcnow()
        )

        # Test transitions valides
        assert execution_service.can_transition_to(order, OrderStatus.PENDING)
        assert execution_service.can_transition_to(order, OrderStatus.CANCELLED)

        # Transition vers filled depuis new
        order.status = OrderStatus.PENDING
        assert execution_service.can_transition_to(order, OrderStatus.FILLED)
        assert execution_service.can_transition_to(order, OrderStatus.PARTIALLY_FILLED)

        # Transitions invalides
        order.status = OrderStatus.FILLED
        assert not execution_service.can_transition_to(order, OrderStatus.PENDING)

    def test_error_handling_broker_failure(self, execution_service, mock_broker_adapter,
                                         mock_order_repository):
        """Test gestion d'erreurs - panne broker"""
        order = Order(
            id="error-order",
            portfolio_id="port-001",
            symbol="BTCUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            created_time=datetime.utcnow()
        )

        # Simuler panne broker
        mock_broker_adapter.execute_order.side_effect = ConnectionError("Broker unavailable")

        # L'ordre devrait être marqué comme erreur
        with pytest.raises(ConnectionError):
            execution_service.execute_order(order)

        # L'ordre devrait quand même être sauvé avec statut d'erreur
        assert order.status == OrderStatus.REJECTED

    def test_concurrent_order_execution(self, execution_service, mock_order_repository):
        """Test exécution concurrente d'ordres"""
        # Simuler deux ordres concurrents sur le même symbole
        order1 = Order(
            id="concurrent-1",
            portfolio_id="port-001",
            symbol="BTCUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            created_time=datetime.utcnow()
        )

        order2 = Order(
            id="concurrent-2",
            portfolio_id="port-001",
            symbol="BTCUSD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.5"),
            created_time=datetime.utcnow()
        )

        # Les deux ordres devraient pouvoir être traités
        # (implémentation dépend de la synchronisation)
        results = execution_service.handle_concurrent_orders([order1, order2])

        assert len(results) == 2
        assert all(isinstance(r, Order) for r in results)

    def test_order_cleanup_and_archiving(self, execution_service, mock_order_repository):
        """Test nettoyage et archivage des ordres"""
        # Ordres anciens à nettoyer
        old_orders = [
            Order(
                id=f"old-order-{i}",
                portfolio_id="port-001",
                symbol="BTCUSD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
                status=OrderStatus.FILLED,
                created_time=datetime.utcnow() - timedelta(days=30)
            ) for i in range(5)
        ]

        mock_order_repository.get_orders_older_than.return_value = old_orders

        archived_count = execution_service.archive_old_orders(days_old=30)

        assert archived_count >= 0
        mock_order_repository.archive_orders.assert_called_once()

    def test_order_performance_tracking(self, execution_service):
        """Test suivi de performance des ordres"""
        order = Order(
            id="perf-order",
            portfolio_id="port-001",
            symbol="BTCUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            status=OrderStatus.FILLED,
            average_fill_price=Decimal("50100"),
            created_time=datetime.utcnow()
        )

        # Prix de référence au moment de la création du signal
        reference_price = Decimal("50000")

        performance = execution_service.calculate_order_performance(
            order=order,
            reference_price=reference_price
        )

        assert "price_improvement" in performance
        assert "slippage_bps" in performance
        assert "execution_cost" in performance