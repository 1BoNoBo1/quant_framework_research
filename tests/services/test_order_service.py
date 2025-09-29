"""
Tests for Order Service
======================

Suite de tests complète pour le service de gestion des ordres.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional

from qframe.api.services.order_service import OrderService
from qframe.domain.entities.order import Order, OrderStatus, OrderSide, OrderType
from qframe.domain.entities.portfolio import Portfolio
from qframe.domain.repositories.order_repository import OrderRepository
from qframe.domain.repositories.portfolio_repository import PortfolioRepository
from qframe.infrastructure.external.order_execution_adapter import BrokerAdapter
from qframe.core.interfaces import MetricsCollector


@pytest.fixture
def mock_order_repository():
    """Repository d'ordres mocké."""
    return Mock(spec=OrderRepository)


@pytest.fixture
def mock_portfolio_repository():
    """Repository de portfolios mocké."""
    return Mock(spec=PortfolioRepository)


@pytest.fixture
def mock_broker_adapter():
    """Adaptateur de broker mocké."""
    return Mock(spec=BrokerAdapter)


@pytest.fixture
def mock_metrics_collector():
    """Collecteur de métriques mocké."""
    return Mock(spec=MetricsCollector)


@pytest.fixture
def order_service(mock_order_repository, mock_portfolio_repository, mock_broker_adapter, mock_metrics_collector):
    """Service d'ordres pour les tests."""
    return OrderService(
        order_repository=mock_order_repository,
        portfolio_repository=mock_portfolio_repository,
        broker_adapter=mock_broker_adapter,
        metrics_collector=mock_metrics_collector
    )


@pytest.fixture
def sample_portfolio():
    """Portfolio de test."""
    return Portfolio(
        id="portfolio-001",
        name="Test Portfolio",
        initial_capital=Decimal("10000.00"),
        base_currency="USD"
    )


@pytest.fixture
def sample_order():
    """Ordre de test."""
    return Order(
        id="order-001",
        portfolio_id="portfolio-001",
        symbol="BTC/USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.1"),
        created_time=datetime.now(),
        status=OrderStatus.PENDING
    )


@pytest.fixture
def sample_limit_order():
    """Ordre limite de test."""
    return Order(
        id="order-002",
        portfolio_id="portfolio-001",
        symbol="ETH/USD",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=Decimal("2.0"),
        price=Decimal("2000.00"),
        created_time=datetime.now(),
        status=OrderStatus.PENDING
    )


class TestOrderServiceBasic:
    """Tests de base pour OrderService."""

    async def test_create_market_order(self, order_service, mock_order_repository, sample_portfolio):
        """Test de création d'un ordre marché."""
        # Arrange
        order_data = {
            "portfolio_id": "portfolio-001",
            "symbol": "BTC/USD",
            "side": OrderSide.BUY,
            "order_type": OrderType.MARKET,
            "quantity": Decimal("0.1")
        }

        expected_order = Order(
            id="generated-id",
            portfolio_id=order_data["portfolio_id"],
            symbol=order_data["symbol"],
            side=order_data["side"],
            order_type=order_data["order_type"],
            quantity=order_data["quantity"],
            created_time=datetime.now(),
            status=OrderStatus.PENDING
        )

        mock_order_repository.save.return_value = expected_order

        # Act
        result = await order_service.create_order(**order_data)

        # Assert
        assert result.symbol == "BTC/USD"
        assert result.side == OrderSide.BUY
        assert result.order_type == OrderType.MARKET
        assert result.quantity == Decimal("0.1")
        assert result.status == OrderStatus.PENDING
        mock_order_repository.save.assert_called_once()

    async def test_create_limit_order(self, order_service, mock_order_repository):
        """Test de création d'un ordre limite."""
        # Arrange
        order_data = {
            "portfolio_id": "portfolio-001",
            "symbol": "ETH/USD",
            "side": OrderSide.SELL,
            "order_type": OrderType.LIMIT,
            "quantity": Decimal("2.0"),
            "price": Decimal("2000.00")
        }

        expected_order = Order(
            id="generated-id",
            portfolio_id=order_data["portfolio_id"],
            symbol=order_data["symbol"],
            side=order_data["side"],
            order_type=order_data["order_type"],
            quantity=order_data["quantity"],
            price=order_data["price"],
            created_time=datetime.now(),
            status=OrderStatus.PENDING
        )

        mock_order_repository.save.return_value = expected_order

        # Act
        result = await order_service.create_order(**order_data)

        # Assert
        assert result.symbol == "ETH/USD"
        assert result.side == OrderSide.SELL
        assert result.order_type == OrderType.LIMIT
        assert result.price == Decimal("2000.00")
        mock_order_repository.save.assert_called_once()

    async def test_get_order_by_id(self, order_service, mock_order_repository, sample_order):
        """Test de récupération d'un ordre par ID."""
        # Arrange
        mock_order_repository.get_by_id.return_value = sample_order

        # Act
        result = await order_service.get_order("order-001")

        # Assert
        assert result == sample_order
        mock_order_repository.get_by_id.assert_called_once_with("order-001")

    async def test_get_orders_by_portfolio(self, order_service, mock_order_repository, sample_order):
        """Test de récupération des ordres par portfolio."""
        # Arrange
        orders = [sample_order]
        mock_order_repository.get_by_portfolio_id.return_value = orders

        # Act
        result = await order_service.get_orders_by_portfolio("portfolio-001")

        # Assert
        assert len(result) == 1
        assert result[0] == sample_order
        mock_order_repository.get_by_portfolio_id.assert_called_once_with("portfolio-001")

    async def test_get_orders_by_status(self, order_service, mock_order_repository, sample_order):
        """Test de récupération des ordres par statut."""
        # Arrange
        orders = [sample_order]
        mock_order_repository.get_by_status.return_value = orders

        # Act
        result = await order_service.get_orders_by_status(OrderStatus.PENDING)

        # Assert
        assert len(result) == 1
        assert result[0] == sample_order
        mock_order_repository.get_by_status.assert_called_once_with(OrderStatus.PENDING)


class TestOrderServiceExecution:
    """Tests d'exécution des ordres."""

    async def test_execute_market_order_success(self, order_service, mock_order_repository, mock_broker_adapter, sample_order):
        """Test d'exécution réussie d'un ordre marché."""
        # Arrange
        sample_order.status = OrderStatus.PENDING
        mock_order_repository.get_by_id.return_value = sample_order

        # Simulation de l'exécution
        execution_result = {
            "order_id": "order-001",
            "executed_quantity": Decimal("0.1"),
            "executed_price": Decimal("45000.00"),
            "commission": Decimal("4.50"),
            "timestamp": datetime.now()
        }
        mock_broker_adapter.execute_order.return_value = execution_result

        # Ordre mis à jour après exécution
        executed_order = sample_order
        executed_order.status = OrderStatus.FILLED
        executed_order.executed_quantity = execution_result["executed_quantity"]
        executed_order.executed_price = execution_result["executed_price"]
        mock_order_repository.save.return_value = executed_order

        # Act
        result = await order_service.execute_order("order-001")

        # Assert
        assert result.status == OrderStatus.FILLED
        assert result.executed_quantity == Decimal("0.1")
        assert result.executed_price == Decimal("45000.00")
        mock_broker_adapter.execute_order.assert_called_once()
        mock_order_repository.save.assert_called()

    async def test_execute_order_partial_fill(self, order_service, mock_order_repository, mock_broker_adapter, sample_order):
        """Test d'exécution partielle d'un ordre."""
        # Arrange
        sample_order.status = OrderStatus.PENDING
        sample_order.quantity = Decimal("1.0")
        mock_order_repository.get_by_id.return_value = sample_order

        # Simulation d'exécution partielle
        execution_result = {
            "order_id": "order-001",
            "executed_quantity": Decimal("0.5"),  # Seulement la moitié
            "executed_price": Decimal("45000.00"),
            "commission": Decimal("2.25"),
            "timestamp": datetime.now()
        }
        mock_broker_adapter.execute_order.return_value = execution_result

        # Ordre partiellement exécuté
        partial_order = sample_order
        partial_order.status = OrderStatus.PARTIALLY_FILLED
        partial_order.executed_quantity = execution_result["executed_quantity"]
        partial_order.executed_price = execution_result["executed_price"]
        mock_order_repository.save.return_value = partial_order

        # Act
        result = await order_service.execute_order("order-001")

        # Assert
        assert result.status == OrderStatus.PARTIALLY_FILLED
        assert result.executed_quantity == Decimal("0.5")
        assert result.remaining_quantity == Decimal("0.5")

    async def test_execute_order_broker_failure(self, order_service, mock_order_repository, mock_broker_adapter, sample_order):
        """Test d'échec d'exécution côté broker."""
        # Arrange
        mock_order_repository.get_by_id.return_value = sample_order
        mock_broker_adapter.execute_order.side_effect = Exception("Broker connection failed")

        # Act & Assert
        with pytest.raises(Exception, match="Broker connection failed"):
            await order_service.execute_order("order-001")

        # L'ordre devrait être marqué comme échoué
        mock_order_repository.save.assert_called()
        saved_order = mock_order_repository.save.call_args[0][0]
        assert saved_order.status == OrderStatus.FAILED

    async def test_execute_nonexistent_order(self, order_service, mock_order_repository):
        """Test d'exécution d'un ordre inexistant."""
        # Arrange
        mock_order_repository.get_by_id.return_value = None

        # Act & Assert
        with pytest.raises(ValueError, match="Order not found"):
            await order_service.execute_order("nonexistent-order")


class TestOrderServiceCancellation:
    """Tests d'annulation des ordres."""

    async def test_cancel_pending_order(self, order_service, mock_order_repository, mock_broker_adapter, sample_order):
        """Test d'annulation d'un ordre en attente."""
        # Arrange
        sample_order.status = OrderStatus.PENDING
        mock_order_repository.get_by_id.return_value = sample_order
        mock_broker_adapter.cancel_order.return_value = True

        cancelled_order = sample_order
        cancelled_order.status = OrderStatus.CANCELLED
        cancelled_order.cancelled_time = datetime.now()
        mock_order_repository.save.return_value = cancelled_order

        # Act
        result = await order_service.cancel_order("order-001")

        # Assert
        assert result.status == OrderStatus.CANCELLED
        assert result.cancelled_time is not None
        mock_broker_adapter.cancel_order.assert_called_once_with("order-001")
        mock_order_repository.save.assert_called_once()

    async def test_cancel_filled_order(self, order_service, mock_order_repository, sample_order):
        """Test d'annulation d'un ordre déjà exécuté (erreur)."""
        # Arrange
        sample_order.status = OrderStatus.FILLED
        mock_order_repository.get_by_id.return_value = sample_order

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot cancel filled order"):
            await order_service.cancel_order("order-001")

    async def test_cancel_already_cancelled_order(self, order_service, mock_order_repository, sample_order):
        """Test d'annulation d'un ordre déjà annulé."""
        # Arrange
        sample_order.status = OrderStatus.CANCELLED
        mock_order_repository.get_by_id.return_value = sample_order

        # Act & Assert
        with pytest.raises(ValueError, match="Order already cancelled"):
            await order_service.cancel_order("order-001")

    async def test_cancel_order_broker_failure(self, order_service, mock_order_repository, mock_broker_adapter, sample_order):
        """Test d'échec d'annulation côté broker."""
        # Arrange
        sample_order.status = OrderStatus.PENDING
        mock_order_repository.get_by_id.return_value = sample_order
        mock_broker_adapter.cancel_order.side_effect = Exception("Cancellation failed")

        # Act & Assert
        with pytest.raises(Exception, match="Cancellation failed"):
            await order_service.cancel_order("order-001")


class TestOrderServiceValidation:
    """Tests de validation des ordres."""

    async def test_validate_order_sufficient_balance(self, order_service, mock_portfolio_repository):
        """Test de validation avec solde suffisant."""
        # Arrange
        portfolio = Portfolio(
            id="portfolio-001",
            name="Test Portfolio",
            initial_capital=Decimal("10000.00"),
            base_currency="USD"
        )
        portfolio.cash_balance = Decimal("5000.00")
        mock_portfolio_repository.get_by_id.return_value = portfolio

        order_data = {
            "portfolio_id": "portfolio-001",
            "symbol": "BTC/USD",
            "side": OrderSide.BUY,
            "order_type": OrderType.MARKET,
            "quantity": Decimal("0.1")
        }

        # Act
        is_valid = await order_service.validate_order(**order_data)

        # Assert
        assert is_valid is True

    async def test_validate_order_insufficient_balance(self, order_service, mock_portfolio_repository):
        """Test de validation avec solde insuffisant."""
        # Arrange
        portfolio = Portfolio(
            id="portfolio-001",
            name="Test Portfolio",
            initial_capital=Decimal("1000.00"),
            base_currency="USD"
        )
        portfolio.cash_balance = Decimal("100.00")  # Solde insuffisant
        mock_portfolio_repository.get_by_id.return_value = portfolio

        order_data = {
            "portfolio_id": "portfolio-001",
            "symbol": "BTC/USD",
            "side": OrderSide.BUY,
            "order_type": OrderType.MARKET,
            "quantity": Decimal("1.0")  # Ordre important
        }

        # Act
        is_valid = await order_service.validate_order(**order_data)

        # Assert
        assert is_valid is False

    async def test_validate_order_invalid_symbol(self, order_service):
        """Test de validation avec symbole invalide."""
        # Arrange
        order_data = {
            "portfolio_id": "portfolio-001",
            "symbol": "INVALID/SYMBOL",
            "side": OrderSide.BUY,
            "order_type": OrderType.MARKET,
            "quantity": Decimal("0.1")
        }

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid symbol"):
            await order_service.validate_order(**order_data)

    async def test_validate_order_negative_quantity(self, order_service):
        """Test de validation avec quantité négative."""
        # Arrange
        order_data = {
            "portfolio_id": "portfolio-001",
            "symbol": "BTC/USD",
            "side": OrderSide.BUY,
            "order_type": OrderType.MARKET,
            "quantity": Decimal("-0.1")  # Quantité négative
        }

        # Act & Assert
        with pytest.raises(ValueError, match="Quantity must be positive"):
            await order_service.validate_order(**order_data)

    async def test_validate_limit_order_without_price(self, order_service):
        """Test de validation d'un ordre limite sans prix."""
        # Arrange
        order_data = {
            "portfolio_id": "portfolio-001",
            "symbol": "BTC/USD",
            "side": OrderSide.BUY,
            "order_type": OrderType.LIMIT,
            "quantity": Decimal("0.1")
            # Prix manquant
        }

        # Act & Assert
        with pytest.raises(ValueError, match="Limit order requires price"):
            await order_service.validate_order(**order_data)


class TestOrderServiceStatistics:
    """Tests de statistiques des ordres."""

    async def test_get_order_statistics(self, order_service, mock_order_repository):
        """Test de récupération des statistiques d'ordres."""
        # Arrange
        orders = [
            Order(id="1", portfolio_id="p1", symbol="BTC/USD", side=OrderSide.BUY,
                  order_type=OrderType.MARKET, quantity=Decimal("0.1"),
                  status=OrderStatus.FILLED, created_time=datetime.now()),
            Order(id="2", portfolio_id="p1", symbol="ETH/USD", side=OrderSide.SELL,
                  order_type=OrderType.LIMIT, quantity=Decimal("1.0"),
                  status=OrderStatus.CANCELLED, created_time=datetime.now()),
            Order(id="3", portfolio_id="p1", symbol="BTC/USD", side=OrderSide.BUY,
                  order_type=OrderType.MARKET, quantity=Decimal("0.2"),
                  status=OrderStatus.PENDING, created_time=datetime.now())
        ]
        mock_order_repository.get_by_portfolio_id.return_value = orders

        # Act
        stats = await order_service.get_order_statistics("p1")

        # Assert
        assert stats["total_orders"] == 3
        assert stats["filled_orders"] == 1
        assert stats["cancelled_orders"] == 1
        assert stats["pending_orders"] == 1
        assert stats["buy_orders"] == 2
        assert stats["sell_orders"] == 1

    async def test_get_order_performance_metrics(self, order_service, mock_order_repository):
        """Test de récupération des métriques de performance."""
        # Arrange
        filled_orders = [
            Order(id="1", portfolio_id="p1", symbol="BTC/USD", side=OrderSide.BUY,
                  order_type=OrderType.MARKET, quantity=Decimal("0.1"),
                  status=OrderStatus.FILLED, executed_price=Decimal("45000.00"),
                  executed_quantity=Decimal("0.1"), created_time=datetime.now(),
                  executed_time=datetime.now() + timedelta(seconds=30)),
            Order(id="2", portfolio_id="p1", symbol="BTC/USD", side=OrderSide.SELL,
                  order_type=OrderType.LIMIT, quantity=Decimal("0.1"),
                  status=OrderStatus.FILLED, executed_price=Decimal("46000.00"),
                  executed_quantity=Decimal("0.1"), created_time=datetime.now(),
                  executed_time=datetime.now() + timedelta(minutes=5))
        ]
        mock_order_repository.get_by_status.return_value = filled_orders

        # Act
        metrics = await order_service.get_performance_metrics("p1")

        # Assert
        assert "average_execution_time" in metrics
        assert "fill_rate" in metrics
        assert "average_slippage" in metrics
        assert metrics["total_executed_orders"] == 2


class TestOrderServiceRiskManagement:
    """Tests de gestion des risques des ordres."""

    async def test_check_position_limits(self, order_service, mock_portfolio_repository):
        """Test de vérification des limites de position."""
        # Arrange
        portfolio = Portfolio(
            id="portfolio-001",
            name="Test Portfolio",
            initial_capital=Decimal("10000.00"),
            base_currency="USD"
        )
        # Simuler une position existante importante
        portfolio.positions = {"BTC/USD": Decimal("5.0")}  # Position importante
        mock_portfolio_repository.get_by_id.return_value = portfolio

        order_data = {
            "portfolio_id": "portfolio-001",
            "symbol": "BTC/USD",
            "side": OrderSide.BUY,
            "order_type": OrderType.MARKET,
            "quantity": Decimal("2.0")  # Augmenterait trop la position
        }

        # Act
        risk_check = await order_service.check_risk_limits(**order_data)

        # Assert
        assert risk_check["position_limit_exceeded"] is True
        assert "max_position_size" in risk_check

    async def test_check_daily_trading_limits(self, order_service, mock_order_repository):
        """Test de vérification des limites de trading journalier."""
        # Arrange
        today_orders = [
            Order(id="1", portfolio_id="p1", symbol="BTC/USD", side=OrderSide.BUY,
                  order_type=OrderType.MARKET, quantity=Decimal("1.0"),
                  status=OrderStatus.FILLED, created_time=datetime.now()),
            Order(id="2", portfolio_id="p1", symbol="ETH/USD", side=OrderSide.SELL,
                  order_type=OrderType.MARKET, quantity=Decimal("10.0"),
                  status=OrderStatus.FILLED, created_time=datetime.now())
        ]
        mock_order_repository.get_by_date_range.return_value = today_orders

        # Act
        limits_check = await order_service.check_daily_limits("p1")

        # Assert
        assert "daily_order_count" in limits_check
        assert "daily_volume" in limits_check
        assert limits_check["daily_order_count"] == 2


class TestOrderServiceMetrics:
    """Tests de collecte de métriques."""

    async def test_metrics_collection_on_order_creation(self, order_service, mock_metrics_collector, mock_order_repository):
        """Test de collecte de métriques lors de la création d'ordres."""
        # Arrange
        order_data = {
            "portfolio_id": "portfolio-001",
            "symbol": "BTC/USD",
            "side": OrderSide.BUY,
            "order_type": OrderType.MARKET,
            "quantity": Decimal("0.1")
        }

        new_order = Order(
            id="order-001",
            **order_data,
            created_time=datetime.now(),
            status=OrderStatus.PENDING
        )
        mock_order_repository.save.return_value = new_order

        # Act
        await order_service.create_order(**order_data)

        # Assert
        mock_metrics_collector.increment.assert_called_with("orders.created.total")
        mock_metrics_collector.histogram.assert_called()

    async def test_metrics_collection_on_execution(self, order_service, mock_metrics_collector, mock_order_repository, mock_broker_adapter, sample_order):
        """Test de collecte de métriques lors de l'exécution."""
        # Arrange
        mock_order_repository.get_by_id.return_value = sample_order
        mock_broker_adapter.execute_order.return_value = {
            "order_id": "order-001",
            "executed_quantity": Decimal("0.1"),
            "executed_price": Decimal("45000.00"),
            "commission": Decimal("4.50"),
            "timestamp": datetime.now()
        }

        # Act
        await order_service.execute_order("order-001")

        # Assert
        mock_metrics_collector.increment.assert_called_with("orders.executed.total")
        mock_metrics_collector.histogram.assert_called_with("orders.execution_time", mock.ANY)


class TestOrderServiceIntegration:
    """Tests d'intégration."""

    @pytest.mark.integration
    async def test_end_to_end_order_flow(self, order_service, mock_order_repository, mock_portfolio_repository, mock_broker_adapter):
        """Test de flux complet d'un ordre."""
        # Arrange - Portfolio avec solde suffisant
        portfolio = Portfolio(
            id="portfolio-001",
            name="Test Portfolio",
            initial_capital=Decimal("10000.00"),
            base_currency="USD"
        )
        portfolio.cash_balance = Decimal("5000.00")
        mock_portfolio_repository.get_by_id.return_value = portfolio

        # Order creation
        order_data = {
            "portfolio_id": "portfolio-001",
            "symbol": "BTC/USD",
            "side": OrderSide.BUY,
            "order_type": OrderType.MARKET,
            "quantity": Decimal("0.1")
        }

        created_order = Order(
            id="order-001",
            **order_data,
            created_time=datetime.now(),
            status=OrderStatus.PENDING
        )
        mock_order_repository.save.return_value = created_order
        mock_order_repository.get_by_id.return_value = created_order

        # Execution
        execution_result = {
            "order_id": "order-001",
            "executed_quantity": Decimal("0.1"),
            "executed_price": Decimal("45000.00"),
            "commission": Decimal("4.50"),
            "timestamp": datetime.now()
        }
        mock_broker_adapter.execute_order.return_value = execution_result

        # Act
        # 1. Create order
        order = await order_service.create_order(**order_data)

        # 2. Execute order
        executed_order = await order_service.execute_order(order.id)

        # Assert
        assert order.status == OrderStatus.PENDING
        assert executed_order.status == OrderStatus.FILLED
        assert executed_order.executed_price == Decimal("45000.00")
        assert mock_order_repository.save.call_count >= 2  # Create + Update