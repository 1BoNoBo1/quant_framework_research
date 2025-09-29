"""
Tests d'Exécution Réelle - Infrastructure Repositories
======================================================

Tests qui EXÉCUTENT vraiment le code des repositories infrastructure
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional

# Memory Repositories
from qframe.infrastructure.persistence.memory_order_repository import MemoryOrderRepository
from qframe.infrastructure.persistence.memory_portfolio_repository import MemoryPortfolioRepository

# Entities pour tests
from qframe.domain.entities.order import (
    Order, OrderStatus, OrderSide, OrderType, TimeInForce, OrderPriority,
    OrderExecution, create_market_order, create_limit_order
)
from qframe.domain.entities.portfolio import Portfolio, Position, PortfolioSnapshot, PortfolioConstraints
from qframe.domain.entities.strategy import Strategy


class TestMemoryOrderRepositoryExecution:
    """Tests d'exécution réelle pour MemoryOrderRepository."""

    @pytest.fixture
    def order_repository(self):
        """Repository d'ordres en mémoire."""
        return MemoryOrderRepository()

    @pytest.fixture
    def sample_orders(self):
        """Ordres de test."""
        orders = []

        # Ordre de marché BTC
        market_order = create_market_order(
            portfolio_id="portfolio-001",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0")
        )
        orders.append(market_order)

        # Ordre limite ETH
        limit_order = create_limit_order(
            portfolio_id="portfolio-001",
            symbol="ETH/USD",
            side=OrderSide.SELL,
            quantity=Decimal("5.0"),
            price=Decimal("3200")
        )
        orders.append(limit_order)

        # Ordre limite BTC avec différent portfolio
        btc_limit = create_limit_order(
            portfolio_id="portfolio-002",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.5"),
            price=Decimal("49000")
        )
        orders.append(btc_limit)

        return orders

    @pytest.mark.asyncio
    async def test_order_repository_initialization_execution(self, order_repository):
        """Test initialisation du repository."""
        # Vérifier état initial
        assert isinstance(order_repository._orders, dict)
        assert len(order_repository._orders) == 0

        # Vérifier que les méthodes existent
        assert hasattr(order_repository, 'save')
        assert hasattr(order_repository, 'find_by_id')
        assert hasattr(order_repository, 'find_by_portfolio_id')
        assert hasattr(order_repository, 'find_by_symbol')
        assert hasattr(order_repository, 'find_by_status')

    @pytest.mark.asyncio
    async def test_save_and_find_by_id_execution(self, order_repository, sample_orders):
        """Test sauvegarde et récupération par ID."""
        order = sample_orders[0]

        # Exécuter sauvegarde
        saved_order = await order_repository.save(order)

        # Vérifier sauvegarde
        assert saved_order is not None
        assert saved_order.id == order.id
        assert saved_order.symbol == order.symbol

        # Exécuter récupération par ID
        found_order = await order_repository.find_by_id(order.id)

        # Vérifier récupération
        assert found_order is not None
        assert found_order.id == order.id
        assert found_order.portfolio_id == order.portfolio_id
        assert found_order.symbol == order.symbol
        assert found_order.side == order.side
        assert found_order.quantity == order.quantity

    @pytest.mark.asyncio
    async def test_find_by_portfolio_id_execution(self, order_repository, sample_orders):
        """Test recherche par ID de portfolio."""
        # Sauvegarder tous les ordres
        for order in sample_orders:
            await order_repository.save(order)

        # Exécuter recherche pour portfolio-001
        portfolio_orders = await order_repository.find_by_portfolio_id("portfolio-001")

        # Vérifier résultats
        assert isinstance(portfolio_orders, list)
        assert len(portfolio_orders) == 2  # BTC market et ETH limit

        for order in portfolio_orders:
            assert order.portfolio_id == "portfolio-001"

        # Exécuter recherche pour portfolio-002
        portfolio2_orders = await order_repository.find_by_portfolio_id("portfolio-002")

        # Vérifier résultats
        assert len(portfolio2_orders) == 1  # BTC limit seulement
        assert portfolio2_orders[0].portfolio_id == "portfolio-002"

    @pytest.mark.asyncio
    async def test_find_by_symbol_execution(self, order_repository, sample_orders):
        """Test recherche par symbole."""
        # Sauvegarder tous les ordres
        for order in sample_orders:
            await order_repository.save(order)

        # Exécuter recherche pour BTC/USD
        btc_orders = await order_repository.find_by_symbol("BTC/USD")

        # Vérifier résultats
        assert isinstance(btc_orders, list)
        assert len(btc_orders) == 2  # Market BTC et Limit BTC

        for order in btc_orders:
            assert order.symbol == "BTC/USD"

        # Exécuter recherche pour ETH/USD
        eth_orders = await order_repository.find_by_symbol("ETH/USD")

        # Vérifier résultats
        assert len(eth_orders) == 1
        assert eth_orders[0].symbol == "ETH/USD"

    @pytest.mark.asyncio
    async def test_find_by_status_execution(self, order_repository, sample_orders):
        """Test recherche par statut."""
        # Modifier statuts des ordres
        sample_orders[0].status = OrderStatus.FILLED
        sample_orders[1].status = OrderStatus.PENDING
        sample_orders[2].status = OrderStatus.CANCELLED

        # Sauvegarder tous les ordres
        for order in sample_orders:
            await order_repository.save(order)

        # Exécuter recherche par statut FILLED
        filled_orders = await order_repository.find_by_status(OrderStatus.FILLED)

        # Vérifier résultats
        assert len(filled_orders) == 1
        assert filled_orders[0].status == OrderStatus.FILLED

        # Exécuter recherche par statut PENDING
        pending_orders = await order_repository.find_by_status(OrderStatus.PENDING)

        # Vérifier résultats
        assert len(pending_orders) == 1
        assert pending_orders[0].status == OrderStatus.PENDING

    @pytest.mark.asyncio
    async def test_find_by_side_execution(self, order_repository, sample_orders):
        """Test recherche par côté (BUY/SELL)."""
        # Sauvegarder tous les ordres
        for order in sample_orders:
            await order_repository.save(order)

        # Exécuter recherche ordres d'achat
        buy_orders = await order_repository.find_by_side(OrderSide.BUY)

        # Vérifier résultats
        assert isinstance(buy_orders, list)
        assert len(buy_orders) == 2  # 2 ordres BUY

        for order in buy_orders:
            assert order.side == OrderSide.BUY

        # Exécuter recherche ordres de vente
        sell_orders = await order_repository.find_by_side(OrderSide.SELL)

        # Vérifier résultats
        assert len(sell_orders) == 1  # 1 ordre SELL
        assert sell_orders[0].side == OrderSide.SELL

    @pytest.mark.asyncio
    async def test_find_by_type_execution(self, order_repository, sample_orders):
        """Test recherche par type d'ordre."""
        # Sauvegarder tous les ordres
        for order in sample_orders:
            await order_repository.save(order)

        # Exécuter recherche ordres de marché
        market_orders = await order_repository.find_by_type(OrderType.MARKET)

        # Vérifier résultats
        assert len(market_orders) == 1
        assert market_orders[0].order_type == OrderType.MARKET

        # Exécuter recherche ordres limite
        limit_orders = await order_repository.find_by_type(OrderType.LIMIT)

        # Vérifier résultats
        assert len(limit_orders) == 2
        for order in limit_orders:
            assert order.order_type == OrderType.LIMIT

    @pytest.mark.asyncio
    async def test_find_by_date_range_execution(self, order_repository, sample_orders):
        """Test recherche par plage de dates."""
        # Modifier dates des ordres
        base_time = datetime.utcnow()
        sample_orders[0].created_time = base_time - timedelta(days=2)
        sample_orders[1].created_time = base_time - timedelta(days=1)
        sample_orders[2].created_time = base_time

        # Sauvegarder tous les ordres
        for order in sample_orders:
            await order_repository.save(order)

        # Exécuter recherche dernières 24h
        start_date = base_time - timedelta(days=1, hours=1)
        end_date = base_time + timedelta(hours=1)

        recent_orders = await order_repository.find_by_date_range(start_date, end_date)

        # Vérifier résultats
        assert len(recent_orders) == 2  # Les 2 ordres les plus récents

        for order in recent_orders:
            assert start_date <= order.created_time <= end_date

    @pytest.mark.asyncio
    async def test_update_order_execution(self, order_repository, sample_orders):
        """Test mise à jour d'ordre."""
        order = sample_orders[0]

        # Sauvegarder ordre initial
        await order_repository.save(order)

        # Modifier l'ordre
        order.status = OrderStatus.PARTIALLY_FILLED
        order.executed_quantity = Decimal("0.5")
        order.updated_at = datetime.utcnow()

        # Exécuter mise à jour
        updated_order = await order_repository.save(order)

        # Vérifier mise à jour
        assert updated_order.status == OrderStatus.PARTIALLY_FILLED
        assert updated_order.executed_quantity == Decimal("0.5")

        # Vérifier persistance
        found_order = await order_repository.find_by_id(order.id)
        assert found_order.status == OrderStatus.PARTIALLY_FILLED
        assert found_order.executed_quantity == Decimal("0.5")

    @pytest.mark.asyncio
    async def test_delete_order_execution(self, order_repository, sample_orders):
        """Test suppression d'ordre."""
        order = sample_orders[0]

        # Sauvegarder ordre
        await order_repository.save(order)

        # Vérifier existence
        found = await order_repository.find_by_id(order.id)
        assert found is not None

        # Exécuter suppression
        deleted = await order_repository.delete(order.id)

        # Vérifier suppression
        assert deleted is True

        # Vérifier que l'ordre n'existe plus
        not_found = await order_repository.find_by_id(order.id)
        assert not_found is None

    @pytest.mark.asyncio
    async def test_get_all_orders_execution(self, order_repository, sample_orders):
        """Test récupération de tous les ordres."""
        # Sauvegarder tous les ordres
        for order in sample_orders:
            await order_repository.save(order)

        # Exécuter récupération de tous
        all_orders = await order_repository.get_all()

        # Vérifier résultats
        assert isinstance(all_orders, list)
        assert len(all_orders) == len(sample_orders)

        # Vérifier que tous les ordres sont présents
        order_ids = {order.id for order in all_orders}
        expected_ids = {order.id for order in sample_orders}
        assert order_ids == expected_ids

    @pytest.mark.asyncio
    async def test_count_orders_execution(self, order_repository, sample_orders):
        """Test comptage d'ordres."""
        # Compter ordres vides
        initial_count = await order_repository.count()
        assert initial_count == 0

        # Sauvegarder quelques ordres
        for order in sample_orders[:2]:
            await order_repository.save(order)

        # Exécuter comptage
        count_after_save = await order_repository.count()
        assert count_after_save == 2

        # Ajouter un ordre de plus
        await order_repository.save(sample_orders[2])

        final_count = await order_repository.count()
        assert final_count == 3

    @pytest.mark.asyncio
    async def test_repository_with_executions_execution(self, order_repository):
        """Test avec ordres ayant des exécutions."""
        # Créer ordre avec exécutions
        order = create_limit_order(
            portfolio_id="portfolio-001",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000")
        )

        # Ajouter exécutions
        execution1 = OrderExecution(
            executed_quantity=Decimal("0.5"),
            execution_price=Decimal("49950"),
            commission=Decimal("12.50"),
            venue="binance"
        )

        execution2 = OrderExecution(
            executed_quantity=Decimal("0.5"),
            execution_price=Decimal("50000"),
            commission=Decimal("12.50"),
            venue="coinbase"
        )

        order.add_execution(execution1)
        order.add_execution(execution2)

        # Sauvegarder ordre avec exécutions
        saved_order = await order_repository.save(order)

        # Vérifier sauvegarde des exécutions
        assert len(saved_order.executions) == 2
        assert saved_order.executed_quantity == Decimal("1.0")

        # Récupérer et vérifier persistance
        found_order = await order_repository.find_by_id(order.id)
        assert len(found_order.executions) == 2
        assert found_order.executed_quantity == Decimal("1.0")


class TestMemoryPortfolioRepositoryExecution:
    """Tests d'exécution réelle pour MemoryPortfolioRepository."""

    @pytest.fixture
    def portfolio_repository(self):
        """Repository de portfolios en mémoire."""
        return MemoryPortfolioRepository()

    @pytest.fixture
    def sample_portfolios(self):
        """Portfolios de test."""
        portfolios = []

        # Portfolio crypto
        crypto_positions = {
            "BTC/USD": Position(
                symbol="BTC/USD",
                quantity=Decimal("2.0"),
                average_price=Decimal("45000"),
                current_price=Decimal("50000"),
                market_value=Decimal("100000")
            ),
            "ETH/USD": Position(
                symbol="ETH/USD",
                quantity=Decimal("10"),
                average_price=Decimal("2800"),
                current_price=Decimal("3000"),
                market_value=Decimal("30000")
            )
        }

        crypto_portfolio = Portfolio(
            id="crypto-portfolio",
            name="Crypto Investment Portfolio",
            initial_capital=Decimal("120000"),
            base_currency="USD",
            positions=crypto_positions
        )
        crypto_portfolio.total_value = Decimal("130000")
        portfolios.append(crypto_portfolio)

        # Portfolio conservateur
        conservative_positions = {
            "CASH": Position(
                symbol="CASH",
                quantity=Decimal("50000"),
                average_price=Decimal("1"),
                current_price=Decimal("1"),
                market_value=Decimal("50000")
            )
        }

        conservative_portfolio = Portfolio(
            id="conservative-portfolio",
            name="Conservative Portfolio",
            initial_capital=Decimal("50000"),
            base_currency="USD",
            positions=conservative_positions
        )
        conservative_portfolio.total_value = Decimal("50000")
        portfolios.append(conservative_portfolio)

        return portfolios

    @pytest.mark.asyncio
    async def test_portfolio_repository_initialization_execution(self, portfolio_repository):
        """Test initialisation du repository."""
        # Vérifier état initial
        assert isinstance(portfolio_repository._portfolios, dict)
        assert len(portfolio_repository._portfolios) == 0

        # Vérifier que les méthodes existent
        assert hasattr(portfolio_repository, 'save')
        assert hasattr(portfolio_repository, 'find_by_id')
        assert hasattr(portfolio_repository, 'find_by_name')
        assert hasattr(portfolio_repository, 'get_all')

    @pytest.mark.asyncio
    async def test_save_and_find_portfolio_execution(self, portfolio_repository, sample_portfolios):
        """Test sauvegarde et récupération de portfolio."""
        portfolio = sample_portfolios[0]

        # Exécuter sauvegarde
        saved_portfolio = await portfolio_repository.save(portfolio)

        # Vérifier sauvegarde
        assert saved_portfolio is not None
        assert saved_portfolio.id == portfolio.id
        assert saved_portfolio.name == portfolio.name
        assert saved_portfolio.total_value == portfolio.total_value

        # Exécuter récupération par ID
        found_portfolio = await portfolio_repository.find_by_id(portfolio.id)

        # Vérifier récupération
        assert found_portfolio is not None
        assert found_portfolio.id == portfolio.id
        assert found_portfolio.initial_capital == portfolio.initial_capital
        assert found_portfolio.base_currency == portfolio.base_currency

        # Vérifier positions
        assert len(found_portfolio.positions) == len(portfolio.positions)
        for symbol, position in portfolio.positions.items():
            assert symbol in found_portfolio.positions
            found_position = found_portfolio.positions[symbol]
            assert found_position.quantity == position.quantity
            assert found_position.market_value == position.market_value

    @pytest.mark.asyncio
    async def test_find_by_name_execution(self, portfolio_repository, sample_portfolios):
        """Test recherche par nom."""
        # Sauvegarder portfolios
        for portfolio in sample_portfolios:
            await portfolio_repository.save(portfolio)

        # Exécuter recherche par nom
        found_portfolio = await portfolio_repository.find_by_name("Crypto Investment Portfolio")

        # Vérifier résultat
        assert found_portfolio is not None
        assert found_portfolio.name == "Crypto Investment Portfolio"
        assert found_portfolio.id == "crypto-portfolio"

        # Test recherche nom inexistant
        not_found = await portfolio_repository.find_by_name("Non-existent Portfolio")
        assert not_found is None

    @pytest.mark.asyncio
    async def test_update_portfolio_execution(self, portfolio_repository, sample_portfolios):
        """Test mise à jour de portfolio."""
        portfolio = sample_portfolios[0]

        # Sauvegarder portfolio initial
        await portfolio_repository.save(portfolio)

        # Modifier le portfolio
        portfolio.total_value = Decimal("140000")
        portfolio.updated_at = datetime.utcnow()

        # Ajouter une nouvelle position
        portfolio.positions["ADA/USD"] = Position(
            symbol="ADA/USD",
            quantity=Decimal("1000"),
            average_price=Decimal("1.50"),
            current_price=Decimal("1.60"),
            market_value=Decimal("1600")
        )

        # Exécuter mise à jour
        updated_portfolio = await portfolio_repository.save(portfolio)

        # Vérifier mise à jour
        assert updated_portfolio.total_value == Decimal("140000")
        assert "ADA/USD" in updated_portfolio.positions

        # Vérifier persistance
        found_portfolio = await portfolio_repository.find_by_id(portfolio.id)
        assert found_portfolio.total_value == Decimal("140000")
        assert "ADA/USD" in found_portfolio.positions

    @pytest.mark.asyncio
    async def test_delete_portfolio_execution(self, portfolio_repository, sample_portfolios):
        """Test suppression de portfolio."""
        portfolio = sample_portfolios[0]

        # Sauvegarder portfolio
        await portfolio_repository.save(portfolio)

        # Vérifier existence
        found = await portfolio_repository.find_by_id(portfolio.id)
        assert found is not None

        # Exécuter suppression
        deleted = await portfolio_repository.delete(portfolio.id)

        # Vérifier suppression
        assert deleted is True

        # Vérifier que le portfolio n'existe plus
        not_found = await portfolio_repository.find_by_id(portfolio.id)
        assert not_found is None

    @pytest.mark.asyncio
    async def test_get_all_portfolios_execution(self, portfolio_repository, sample_portfolios):
        """Test récupération de tous les portfolios."""
        # Sauvegarder tous les portfolios
        for portfolio in sample_portfolios:
            await portfolio_repository.save(portfolio)

        # Exécuter récupération de tous
        all_portfolios = await portfolio_repository.get_all()

        # Vérifier résultats
        assert isinstance(all_portfolios, list)
        assert len(all_portfolios) == len(sample_portfolios)

        # Vérifier que tous les portfolios sont présents
        portfolio_ids = {p.id for p in all_portfolios}
        expected_ids = {p.id for p in sample_portfolios}
        assert portfolio_ids == expected_ids

    @pytest.mark.asyncio
    async def test_count_portfolios_execution(self, portfolio_repository, sample_portfolios):
        """Test comptage de portfolios."""
        # Compter portfolios vides
        initial_count = await portfolio_repository.count()
        assert initial_count == 0

        # Sauvegarder un portfolio
        await portfolio_repository.save(sample_portfolios[0])

        # Exécuter comptage
        count_after_save = await portfolio_repository.count()
        assert count_after_save == 1

        # Ajouter un autre portfolio
        await portfolio_repository.save(sample_portfolios[1])

        final_count = await portfolio_repository.count()
        assert final_count == 2

    @pytest.mark.asyncio
    async def test_portfolio_with_snapshots_execution(self, portfolio_repository):
        """Test portfolio avec historique de snapshots."""
        # Créer portfolio avec snapshots
        portfolio = Portfolio(
            id="snapshot-portfolio",
            name="Portfolio with History",
            initial_capital=Decimal("100000"),
            base_currency="USD"
        )

        # Ajouter snapshots historiques
        base_date = datetime.utcnow() - timedelta(days=5)
        for i in range(5):
            snapshot = PortfolioSnapshot(
                timestamp=base_date + timedelta(days=i),
                total_value=Decimal("100000") + Decimal(str(i * 1000)),
                positions_snapshot={},
                cash_balance=Decimal("10000")
            )
            portfolio.snapshots.append(snapshot)

        # Sauvegarder portfolio avec snapshots
        saved_portfolio = await portfolio_repository.save(portfolio)

        # Vérifier sauvegarde des snapshots
        assert len(saved_portfolio.snapshots) == 5

        # Récupérer et vérifier persistance
        found_portfolio = await portfolio_repository.find_by_id(portfolio.id)
        assert len(found_portfolio.snapshots) == 5

        # Vérifier ordre chronologique
        timestamps = [s.timestamp for s in found_portfolio.snapshots]
        assert timestamps == sorted(timestamps)

    @pytest.mark.asyncio
    async def test_repository_concurrent_access_execution(self, portfolio_repository, sample_portfolios):
        """Test accès concurrent au repository."""
        # Sauvegardes concurrentes
        save_tasks = []
        for portfolio in sample_portfolios:
            task = portfolio_repository.save(portfolio)
            save_tasks.append(task)

        # Exécuter toutes les sauvegardes en parallèle
        saved_portfolios = await asyncio.gather(*save_tasks)

        # Vérifier que toutes les sauvegardes ont réussi
        assert len(saved_portfolios) == len(sample_portfolios)

        # Lectures concurrentes
        read_tasks = []
        for portfolio in sample_portfolios:
            task = portfolio_repository.find_by_id(portfolio.id)
            read_tasks.append(task)

        # Exécuter toutes les lectures en parallèle
        found_portfolios = await asyncio.gather(*read_tasks)

        # Vérifier que toutes les lectures ont réussi
        assert len(found_portfolios) == len(sample_portfolios)
        for found in found_portfolios:
            assert found is not None

    @pytest.mark.asyncio
    async def test_repository_integration_execution(self, order_repository, portfolio_repository, sample_portfolios, sample_orders):
        """Test d'intégration repositories."""
        # Workflow intégré: Portfolio → Ordres → Exécutions

        # 1. Sauvegarder portfolios
        for portfolio in sample_portfolios:
            await portfolio_repository.save(portfolio)

        # 2. Créer et sauvegarder ordres pour portfolios
        for order in sample_orders:
            await order_repository.save(order)

        # 3. Vérifier relations
        crypto_portfolio = await portfolio_repository.find_by_id("crypto-portfolio")
        assert crypto_portfolio is not None

        # Récupérer ordres de ce portfolio
        portfolio_orders = await order_repository.find_by_portfolio_id("crypto-portfolio")
        assert len(portfolio_orders) > 0

        # 4. Simuler exécution et mise à jour
        for order in portfolio_orders:
            # Simuler exécution partielle
            order.executed_quantity = order.quantity / 2
            order.status = OrderStatus.PARTIALLY_FILLED
            await order_repository.save(order)

        # 5. Vérifier état final
        updated_orders = await order_repository.find_by_portfolio_id("crypto-portfolio")
        for order in updated_orders:
            assert order.executed_quantity > 0
            assert order.status == OrderStatus.PARTIALLY_FILLED

        # 6. Comptes finaux
        total_portfolios = await portfolio_repository.count()
        total_orders = await order_repository.count()

        assert total_portfolios == len(sample_portfolios)
        assert total_orders == len(sample_orders)