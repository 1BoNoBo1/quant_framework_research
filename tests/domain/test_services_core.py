"""
Tests for Core Domain Services
==============================

Tests ciblés pour les services domain essentiels.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime
from decimal import Decimal

from qframe.domain.services.portfolio_service import PortfolioService
from qframe.domain.services.execution_service import ExecutionService
from qframe.domain.entities.portfolio import Portfolio
from qframe.domain.entities.order import Order, OrderStatus, OrderSide, OrderType
from qframe.domain.value_objects.signal import Signal, SignalAction


class TestPortfolioService:
    """Tests pour PortfolioService."""

    @pytest.fixture
    def mock_portfolio_repository(self):
        repo = Mock()
        repo.save = Mock()
        repo.find_by_id = Mock()
        repo.update = Mock()
        repo.delete = Mock()
        return repo

    @pytest.fixture
    def portfolio_service(self, mock_portfolio_repository):
        service = PortfolioService(risk_free_rate=Decimal("0.02"))
        # Inject mock repository
        service.portfolio_repository = mock_portfolio_repository
        return service

    @pytest.fixture
    def sample_portfolio(self):
        return Portfolio(
            id="portfolio-001",
            name="Test Portfolio",
            initial_capital=Decimal("10000.00"),
            base_currency="USD"
        )

    def test_create_portfolio(self, portfolio_service, mock_portfolio_repository, sample_portfolio):
        """Test création de portfolio."""
        mock_portfolio_repository.save.return_value = sample_portfolio

        result = portfolio_service.create_portfolio(
            name="Test Portfolio",
            initial_capital=Decimal("10000.00"),
            base_currency="USD"
        )

        assert result is not None
        mock_portfolio_repository.save.assert_called_once()

    def test_get_portfolio(self, portfolio_service, mock_portfolio_repository, sample_portfolio):
        """Test récupération de portfolio."""
        mock_portfolio_repository.find_by_id.return_value = sample_portfolio

        result = portfolio_service.get_portfolio("portfolio-001")

        assert result == sample_portfolio
        mock_portfolio_repository.find_by_id.assert_called_once_with("portfolio-001")

    def test_update_portfolio(self, portfolio_service, mock_portfolio_repository, sample_portfolio):
        """Test mise à jour de portfolio."""
        mock_portfolio_repository.update.return_value = sample_portfolio

        result = portfolio_service.update_portfolio(sample_portfolio)

        assert result == sample_portfolio
        mock_portfolio_repository.update.assert_called_once_with(sample_portfolio)

    def test_delete_portfolio(self, portfolio_service, mock_portfolio_repository):
        """Test suppression de portfolio."""
        mock_portfolio_repository.delete.return_value = True

        result = portfolio_service.delete_portfolio("portfolio-001")

        assert result is True
        mock_portfolio_repository.delete.assert_called_once_with("portfolio-001")

    def test_calculate_portfolio_value(self, portfolio_service, sample_portfolio):
        """Test calcul de valeur de portfolio."""
        # Mock des positions
        portfolio_service.position_service = Mock()
        portfolio_service.position_service.get_portfolio_positions = Mock(return_value=[])

        value = portfolio_service.calculate_portfolio_value("portfolio-001")

        assert isinstance(value, (Decimal, float, int))

    def test_get_portfolio_performance(self, portfolio_service, sample_portfolio):
        """Test récupération de performance de portfolio."""
        portfolio_service.performance_calculator = Mock()
        portfolio_service.performance_calculator.calculate_returns = Mock(return_value=Decimal("0.05"))

        performance = portfolio_service.get_portfolio_performance("portfolio-001")

        assert performance is not None


class TestExecutionService:
    """Tests pour ExecutionService."""

    @pytest.fixture
    def mock_order_repository(self):
        repo = Mock()
        repo.save = Mock()
        repo.find_by_id = Mock()
        repo.update = Mock()
        return repo

    @pytest.fixture
    def mock_broker_adapter(self):
        adapter = Mock()
        adapter.place_order = AsyncMock()
        adapter.cancel_order = AsyncMock()
        adapter.get_order_status = AsyncMock()
        return adapter

    @pytest.fixture
    def execution_service(self, mock_order_repository, mock_broker_adapter):
        service = ExecutionService()
        # Inject mock dependencies
        service.order_repository = mock_order_repository
        service.broker_adapter = mock_broker_adapter
        return service

    @pytest.fixture
    def sample_order(self):
        return Order(
            id="order-001",
            portfolio_id="portfolio-001",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            created_time=datetime.utcnow()
        )

    @pytest.fixture
    def sample_signal(self):
        return Signal(
            symbol="BTC/USD",
            action=SignalAction.BUY,
            confidence=0.8,
            timestamp=datetime.utcnow()
        )

    async def test_execute_signal(self, execution_service, mock_order_repository, sample_signal, sample_order):
        """Test exécution de signal."""
        mock_order_repository.save.return_value = sample_order
        execution_service.broker_adapter.place_order.return_value = {"order_id": "broker-001"}

        result = await execution_service.execute_signal(sample_signal, "portfolio-001")

        assert result is not None
        mock_order_repository.save.assert_called()

    async def test_place_order(self, execution_service, mock_order_repository, sample_order):
        """Test placement d'ordre."""
        mock_order_repository.save.return_value = sample_order
        execution_service.broker_adapter.place_order.return_value = {"order_id": "broker-001"}

        result = await execution_service.place_order(sample_order)

        assert result is not None
        mock_order_repository.save.assert_called()

    async def test_cancel_order(self, execution_service, mock_order_repository, sample_order):
        """Test annulation d'ordre."""
        mock_order_repository.find_by_id.return_value = sample_order
        execution_service.broker_adapter.cancel_order.return_value = {"status": "cancelled"}

        result = await execution_service.cancel_order("order-001")

        assert result is not None
        execution_service.broker_adapter.cancel_order.assert_called_once()

    async def test_get_order_status(self, execution_service, sample_order):
        """Test récupération de statut d'ordre."""
        execution_service.broker_adapter.get_order_status.return_value = {"status": "filled"}

        status = await execution_service.get_order_status("order-001")

        assert status is not None
        execution_service.broker_adapter.get_order_status.assert_called_once()

    def test_validate_order(self, execution_service, sample_order):
        """Test validation d'ordre."""
        # Mock des services de validation
        execution_service.risk_manager = Mock()
        execution_service.risk_manager.validate_order = Mock(return_value=True)

        is_valid = execution_service.validate_order(sample_order)

        assert isinstance(is_valid, bool)

    async def test_bulk_execute_orders(self, execution_service, sample_order):
        """Test exécution en lot d'ordres."""
        orders = [sample_order]
        execution_service.broker_adapter.place_order.return_value = {"order_id": "broker-001"}

        results = await execution_service.bulk_execute_orders(orders)

        assert len(results) == len(orders)

    def test_calculate_order_fees(self, execution_service, sample_order):
        """Test calcul de frais d'ordre."""
        execution_service.fee_calculator = Mock()
        execution_service.fee_calculator.calculate_fees = Mock(return_value=Decimal("5.00"))

        fees = execution_service.calculate_order_fees(sample_order)

        assert isinstance(fees, (Decimal, float))

    def test_get_execution_statistics(self, execution_service):
        """Test statistiques d'exécution."""
        execution_service.order_repository.find_all = Mock(return_value=[])

        stats = execution_service.get_execution_statistics()

        assert isinstance(stats, dict)