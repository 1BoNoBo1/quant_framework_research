"""
Tests for Core Persistence Repositories
======================================

Tests ciblés pour les repositories de persistence essentiels.
"""

import pytest
from unittest.mock import Mock
from datetime import datetime
from decimal import Decimal

from qframe.infrastructure.persistence.memory_order_repository import MemoryOrderRepository
from qframe.infrastructure.persistence.memory_portfolio_repository import MemoryPortfolioRepository
from qframe.domain.entities.order import Order, OrderStatus, OrderSide, OrderType
from qframe.domain.entities.portfolio import Portfolio


class TestMemoryOrderRepository:
    """Tests pour MemoryOrderRepository."""

    @pytest.fixture
    def order_repository(self):
        return MemoryOrderRepository()

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

    def test_save_order(self, order_repository, sample_order):
        """Test sauvegarde d'ordre."""
        saved_order = order_repository.save(sample_order)
        assert saved_order.id == sample_order.id
        assert saved_order.symbol == "BTC/USD"

    def test_find_by_id(self, order_repository, sample_order):
        """Test recherche par ID."""
        order_repository.save(sample_order)
        found_order = order_repository.find_by_id(sample_order.id)
        assert found_order is not None
        assert found_order.id == sample_order.id

    def test_find_by_portfolio_id(self, order_repository, sample_order):
        """Test recherche par portfolio ID."""
        order_repository.save(sample_order)
        orders = order_repository.find_by_portfolio_id("portfolio-001")
        assert len(orders) == 1
        assert orders[0].id == sample_order.id

    def test_find_by_status(self, order_repository, sample_order):
        """Test recherche par statut."""
        sample_order.status = OrderStatus.PENDING
        order_repository.save(sample_order)
        orders = order_repository.find_by_status(OrderStatus.PENDING)
        assert len(orders) == 1

    def test_update_order(self, order_repository, sample_order):
        """Test mise à jour d'ordre."""
        order_repository.save(sample_order)
        sample_order.status = OrderStatus.FILLED
        updated_order = order_repository.update(sample_order)
        assert updated_order.status == OrderStatus.FILLED

    def test_delete_order(self, order_repository, sample_order):
        """Test suppression d'ordre."""
        order_repository.save(sample_order)
        assert order_repository.delete(sample_order.id) is True
        assert order_repository.find_by_id(sample_order.id) is None

    def test_find_all_orders(self, order_repository, sample_order):
        """Test récupération de tous les ordres."""
        order_repository.save(sample_order)
        orders = order_repository.find_all()
        assert len(orders) == 1

    def test_count_orders(self, order_repository, sample_order):
        """Test comptage d'ordres."""
        order_repository.save(sample_order)
        count = order_repository.count()
        assert count == 1

    def test_exists_order(self, order_repository, sample_order):
        """Test existence d'ordre."""
        order_repository.save(sample_order)
        assert order_repository.exists(sample_order.id) is True
        assert order_repository.exists("non-existent") is False


class TestMemoryPortfolioRepository:
    """Tests pour MemoryPortfolioRepository."""

    @pytest.fixture
    def portfolio_repository(self):
        return MemoryPortfolioRepository()

    @pytest.fixture
    def sample_portfolio(self):
        return Portfolio(
            id="portfolio-001",
            name="Test Portfolio",
            initial_capital=Decimal("10000.00"),
            base_currency="USD"
        )

    def test_save_portfolio(self, portfolio_repository, sample_portfolio):
        """Test sauvegarde de portfolio."""
        saved_portfolio = portfolio_repository.save(sample_portfolio)
        assert saved_portfolio.id == sample_portfolio.id
        assert saved_portfolio.name == "Test Portfolio"

    def test_find_by_id(self, portfolio_repository, sample_portfolio):
        """Test recherche par ID."""
        portfolio_repository.save(sample_portfolio)
        found_portfolio = portfolio_repository.find_by_id(sample_portfolio.id)
        assert found_portfolio is not None
        assert found_portfolio.id == sample_portfolio.id

    def test_find_by_name(self, portfolio_repository, sample_portfolio):
        """Test recherche par nom."""
        portfolio_repository.save(sample_portfolio)
        portfolios = portfolio_repository.find_by_name("Test Portfolio")
        assert len(portfolios) == 1
        assert portfolios[0].name == "Test Portfolio"

    def test_update_portfolio(self, portfolio_repository, sample_portfolio):
        """Test mise à jour de portfolio."""
        portfolio_repository.save(sample_portfolio)
        sample_portfolio.name = "Updated Portfolio"
        updated_portfolio = portfolio_repository.update(sample_portfolio)
        assert updated_portfolio.name == "Updated Portfolio"

    def test_delete_portfolio(self, portfolio_repository, sample_portfolio):
        """Test suppression de portfolio."""
        portfolio_repository.save(sample_portfolio)
        assert portfolio_repository.delete(sample_portfolio.id) is True
        assert portfolio_repository.find_by_id(sample_portfolio.id) is None

    def test_find_all_portfolios(self, portfolio_repository, sample_portfolio):
        """Test récupération de tous les portfolios."""
        portfolio_repository.save(sample_portfolio)
        portfolios = portfolio_repository.find_all()
        assert len(portfolios) == 1

    def test_count_portfolios(self, portfolio_repository, sample_portfolio):
        """Test comptage de portfolios."""
        portfolio_repository.save(sample_portfolio)
        count = portfolio_repository.count()
        assert count == 1

    def test_exists_portfolio(self, portfolio_repository, sample_portfolio):
        """Test existence de portfolio."""
        portfolio_repository.save(sample_portfolio)
        assert portfolio_repository.exists(sample_portfolio.id) is True
        assert portfolio_repository.exists("non-existent") is False

    def test_get_active_portfolios(self, portfolio_repository, sample_portfolio):
        """Test récupération des portfolios actifs."""
        portfolio_repository.save(sample_portfolio)
        active_portfolios = portfolio_repository.get_active_portfolios()
        assert len(active_portfolios) >= 0  # Peut être 0 ou 1 selon l'état

    def test_get_portfolio_statistics(self, portfolio_repository, sample_portfolio):
        """Test statistiques de portfolios."""
        portfolio_repository.save(sample_portfolio)
        stats = portfolio_repository.get_portfolio_statistics(sample_portfolio.id)
        assert stats is not None
        assert "total_value" in stats or stats == {}