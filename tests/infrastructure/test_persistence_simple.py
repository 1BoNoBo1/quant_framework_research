"""
Tests for Infrastructure Persistence (Simple)
=============================================

Tests ciblés pour la persistance des données.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional

from qframe.infrastructure.persistence.memory_order_repository import MemoryOrderRepository
from qframe.infrastructure.persistence.memory_portfolio_repository import MemoryPortfolioRepository
from qframe.domain.entities.order import Order, OrderStatus, OrderSide, OrderType
from qframe.domain.entities.portfolio import Portfolio
from qframe.domain.entities.position import Position


@pytest.fixture
def order_repository():
    return MemoryOrderRepository()


@pytest.fixture
def portfolio_repository():
    return MemoryPortfolioRepository()


@pytest.fixture
def sample_order():
    return Order(
        id="order-001",
        portfolio_id="portfolio-001",
        symbol="BTC/USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("1.0"),
        created_time=datetime.utcnow()
    )


@pytest.fixture
def sample_portfolio():
    return Portfolio(
        id="portfolio-001",
        name="Test Portfolio",
        initial_capital=Decimal("100000.00"),
        base_currency="USD"
    )


class TestMemoryOrderRepository:
    """Tests pour MemoryOrderRepository."""

    def test_save_order(self, order_repository, sample_order):
        """Test sauvegarde d'ordre."""
        saved_order = order_repository.save(sample_order)

        assert saved_order.id == sample_order.id
        assert saved_order.symbol == sample_order.symbol

    def test_get_order_by_id(self, order_repository, sample_order):
        """Test récupération par ID."""
        order_repository.save(sample_order)

        retrieved_order = order_repository.get_by_id(sample_order.id)

        assert retrieved_order is not None
        assert retrieved_order.id == sample_order.id

    def test_get_nonexistent_order(self, order_repository):
        """Test récupération ordre inexistant."""
        retrieved_order = order_repository.get_by_id("nonexistent")

        assert retrieved_order is None

    def test_get_orders_by_portfolio(self, order_repository):
        """Test récupération par portfolio."""
        # Créer plusieurs ordres pour le même portfolio
        orders = []
        for i in range(3):
            order = Order(
                id=f"order-{i}",
                portfolio_id="portfolio-001",
                symbol=f"SYMBOL{i}",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
                created_time=datetime.utcnow()
            )
            order_repository.save(order)
            orders.append(order)

        # Récupérer tous les ordres du portfolio
        portfolio_orders = order_repository.get_by_portfolio_id("portfolio-001")

        assert len(portfolio_orders) == 3
        assert all(o.portfolio_id == "portfolio-001" for o in portfolio_orders)

    def test_get_orders_by_status(self, order_repository, sample_order):
        """Test récupération par statut."""
        sample_order.status = OrderStatus.FILLED
        order_repository.save(sample_order)

        filled_orders = order_repository.get_by_status(OrderStatus.FILLED)

        assert len(filled_orders) == 1
        assert filled_orders[0].status == OrderStatus.FILLED

    def test_get_active_orders(self, order_repository):
        """Test récupération ordres actifs."""
        # Créer ordres avec différents statuts
        active_order = Order(
            id="active-001",
            portfolio_id="portfolio-001",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            status=OrderStatus.PENDING,
            created_time=datetime.utcnow()
        )

        filled_order = Order(
            id="filled-001",
            portfolio_id="portfolio-001",
            symbol="ETH/USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("2.0"),
            status=OrderStatus.FILLED,
            created_time=datetime.utcnow()
        )

        order_repository.save(active_order)
        order_repository.save(filled_order)

        active_orders = order_repository.get_active_orders()

        assert len(active_orders) == 1
        assert active_orders[0].id == "active-001"

    def test_update_order(self, order_repository, sample_order):
        """Test mise à jour d'ordre."""
        order_repository.save(sample_order)

        # Modifier l'ordre
        sample_order.status = OrderStatus.FILLED
        sample_order.filled_quantity = sample_order.quantity
        sample_order.filled_time = datetime.utcnow()

        updated_order = order_repository.save(sample_order)

        assert updated_order.status == OrderStatus.FILLED
        assert updated_order.filled_quantity == sample_order.quantity

    def test_delete_order(self, order_repository, sample_order):
        """Test suppression d'ordre."""
        order_repository.save(sample_order)

        # Vérifier que l'ordre existe
        assert order_repository.get_by_id(sample_order.id) is not None

        # Supprimer
        order_repository.delete(sample_order.id)

        # Vérifier que l'ordre n'existe plus
        assert order_repository.get_by_id(sample_order.id) is None

    def test_get_order_statistics(self, order_repository):
        """Test statistiques des ordres."""
        # Créer plusieurs ordres avec différents statuts
        for i in range(5):
            order = Order(
                id=f"stat-order-{i}",
                portfolio_id="portfolio-001",
                symbol="BTC/USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
                status=OrderStatus.FILLED if i < 3 else OrderStatus.PENDING,
                created_time=datetime.utcnow()
            )
            order_repository.save(order)

        stats = order_repository.get_order_statistics("portfolio-001")

        assert stats["total_orders"] == 5
        assert stats["filled_orders"] == 3
        assert stats["pending_orders"] == 2


class TestMemoryPortfolioRepository:
    """Tests pour MemoryPortfolioRepository."""

    def test_save_portfolio(self, portfolio_repository, sample_portfolio):
        """Test sauvegarde de portfolio."""
        saved_portfolio = portfolio_repository.save(sample_portfolio)

        assert saved_portfolio.id == sample_portfolio.id
        assert saved_portfolio.name == sample_portfolio.name

    def test_get_portfolio_by_id(self, portfolio_repository, sample_portfolio):
        """Test récupération par ID."""
        portfolio_repository.save(sample_portfolio)

        retrieved_portfolio = portfolio_repository.get_by_id(sample_portfolio.id)

        assert retrieved_portfolio is not None
        assert retrieved_portfolio.id == sample_portfolio.id

    def test_update_portfolio(self, portfolio_repository, sample_portfolio):
        """Test mise à jour de portfolio."""
        portfolio_repository.save(sample_portfolio)

        # Ajouter une position
        position = Position(
            symbol="BTC/USD",
            quantity=Decimal("1.0"),
            average_price=Decimal("45000.00")
        )
        sample_portfolio.add_position(position)
        sample_portfolio.cash_balance = Decimal("55000.00")

        updated_portfolio = portfolio_repository.save(sample_portfolio)

        assert "BTC/USD" in updated_portfolio.positions
        assert updated_portfolio.cash_balance == Decimal("55000.00")

    def test_get_all_portfolios(self, portfolio_repository):
        """Test récupération de tous les portfolios."""
        # Créer plusieurs portfolios
        portfolios = []
        for i in range(3):
            portfolio = Portfolio(
                id=f"portfolio-{i}",
                name=f"Portfolio {i}",
                initial_capital=Decimal("10000.00"),
                base_currency="USD"
            )
            portfolio_repository.save(portfolio)
            portfolios.append(portfolio)

        all_portfolios = portfolio_repository.get_all()

        assert len(all_portfolios) == 3
        assert all(p.name.startswith("Portfolio") for p in all_portfolios)

    def test_delete_portfolio(self, portfolio_repository, sample_portfolio):
        """Test suppression de portfolio."""
        portfolio_repository.save(sample_portfolio)

        # Vérifier existence
        assert portfolio_repository.get_by_id(sample_portfolio.id) is not None

        # Supprimer
        portfolio_repository.delete(sample_portfolio.id)

        # Vérifier suppression
        assert portfolio_repository.get_by_id(sample_portfolio.id) is None

    def test_portfolio_with_positions(self, portfolio_repository):
        """Test portfolio avec positions."""
        portfolio = Portfolio(
            id="position-portfolio",
            name="Portfolio with Positions",
            initial_capital=Decimal("100000.00"),
            base_currency="USD"
        )

        # Ajouter des positions
        positions = [
            Position("BTC/USD", Decimal("1.0"), Decimal("45000.00")),
            Position("ETH/USD", Decimal("10.0"), Decimal("3000.00")),
            Position("ADA/USD", Decimal("1000.0"), Decimal("1.50"))
        ]

        for position in positions:
            portfolio.add_position(position)

        # Ajuster le cash
        portfolio.cash_balance = Decimal("22500.00")  # 100k - 45k - 30k - 1.5k

        saved_portfolio = portfolio_repository.save(portfolio)
        retrieved_portfolio = portfolio_repository.get_by_id("position-portfolio")

        assert len(retrieved_portfolio.positions) == 3
        assert "BTC/USD" in retrieved_portfolio.positions
        assert retrieved_portfolio.cash_balance == Decimal("22500.00")

    def test_portfolio_performance_tracking(self, portfolio_repository):
        """Test suivi de performance du portfolio."""
        portfolio = Portfolio(
            id="perf-portfolio",
            name="Performance Portfolio",
            initial_capital=Decimal("50000.00"),
            base_currency="USD"
        )

        # Simuler une performance
        portfolio.total_value = Decimal("57500.00")  # Gain de 15%

        saved_portfolio = portfolio_repository.save(portfolio)

        # Calculer le retour
        portfolio_return = (saved_portfolio.total_value - saved_portfolio.initial_capital) / saved_portfolio.initial_capital
        assert portfolio_return == Decimal("0.15")  # 15%

    def test_concurrent_access(self, portfolio_repository):
        """Test accès concurrent (simulation)."""
        portfolio = Portfolio(
            id="concurrent-portfolio",
            name="Concurrent Portfolio",
            initial_capital=Decimal("10000.00"),
            base_currency="USD"
        )

        # Sauvegarder initialement
        portfolio_repository.save(portfolio)

        # Simuler deux modifications concurrentes
        portfolio1 = portfolio_repository.get_by_id("concurrent-portfolio")
        portfolio2 = portfolio_repository.get_by_id("concurrent-portfolio")

        # Modification 1
        portfolio1.cash_balance = Decimal("9000.00")
        portfolio_repository.save(portfolio1)

        # Modification 2
        portfolio2.cash_balance = Decimal("8500.00")
        portfolio_repository.save(portfolio2)

        # La dernière modification gagne
        final_portfolio = portfolio_repository.get_by_id("concurrent-portfolio")
        assert final_portfolio.cash_balance == Decimal("8500.00")

    def test_portfolio_validation(self, portfolio_repository):
        """Test validation des données de portfolio."""
        # Portfolio avec capital négatif (devrait être autorisé mais surveillé)
        portfolio = Portfolio(
            id="validation-portfolio",
            name="Validation Test",
            initial_capital=Decimal("-1000.00"),  # Capital négatif
            base_currency="USD"
        )

        # Le repository devrait quand même sauvegarder
        saved_portfolio = portfolio_repository.save(portfolio)
        assert saved_portfolio.initial_capital == Decimal("-1000.00")

    def test_portfolio_search_by_name(self, portfolio_repository):
        """Test recherche par nom."""
        # Créer des portfolios avec noms similaires
        portfolios = [
            Portfolio("search-1", "Trading Portfolio", Decimal("10000"), "USD"),
            Portfolio("search-2", "Investment Portfolio", Decimal("20000"), "USD"),
            Portfolio("search-3", "Crypto Trading", Decimal("5000"), "USD")
        ]

        for portfolio in portfolios:
            portfolio_repository.save(portfolio)

        # Rechercher portfolios contenant "Trading"
        trading_portfolios = [p for p in portfolio_repository.get_all()
                            if "Trading" in p.name]

        assert len(trading_portfolios) == 2
        assert all("Trading" in p.name for p in trading_portfolios)