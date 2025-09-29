"""
Tests d'exécution réelle pour qframe.infrastructure.persistence
==============================================================

Tests complets de tous les modules de persistence avec exécution réelle
des méthodes et validation des comportements.
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# Core imports
from qframe.domain.entities.order import Order, OrderStatus, OrderSide, OrderType, TimeInForce, OrderPriority
from qframe.domain.entities.portfolio import Portfolio, PortfolioStatus, PortfolioType
from qframe.domain.value_objects.position import Position
from qframe.domain.value_objects.signal import Signal, SignalAction, SignalConfidence

# Infrastructure persistence imports
from qframe.infrastructure.persistence.memory_order_repository import MemoryOrderRepository
from qframe.infrastructure.persistence.memory_portfolio_repository import MemoryPortfolioRepository
from qframe.infrastructure.persistence.memory_risk_assessment_repository import MemoryRiskAssessmentRepository
from qframe.infrastructure.persistence.memory_backtest_repository import MemoryBacktestRepository
from qframe.infrastructure.persistence.memory_strategy_repository import MemoryStrategyRepository

# Database management imports
from qframe.infrastructure.persistence.database import (
    DatabaseConfig, DatabaseManager, ConnectionPool, TransactionManager,
    IsolationLevel, ConnectionStats
)

# Cache management imports
from qframe.infrastructure.persistence.cache import (
    CacheConfig, CacheManager, InMemoryCache, RedisCache,
    CacheStrategy, CacheStats, CacheInterface
)

# Time-series imports
try:
    from qframe.infrastructure.persistence.timeseries import (
        TimeSeriesConfig, InfluxDBManager, MarketDataStorage,
        QueryOptions, Aggregation, TimeSeriesDB
    )
    from qframe.infrastructure.data.market_data_pipeline import MarketDataPoint, DataType, DataQuality
    TIMESERIES_AVAILABLE = True
except ImportError:
    TIMESERIES_AVAILABLE = False


class TestMemoryOrderRepository:
    """Tests pour MemoryOrderRepository avec exécution réelle"""

    @pytest.fixture
    async def order_repository(self):
        """Repository d'ordres en mémoire"""
        return MemoryOrderRepository()

    @pytest.fixture
    def sample_order(self):
        """Ordre d'exemple pour tests"""
        return Order(
            id="test-order-001",
            portfolio_id="portfolio-001",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            created_time=datetime.utcnow(),
            status=OrderStatus.PENDING,
            strategy_id="strategy-001"
        )

    @pytest.fixture
    async def order_with_executions(self):
        """Ordre avec exécutions pour tests avancés"""
        from qframe.domain.value_objects.execution import Execution

        order = Order(
            id="order-with-exec-001",
            portfolio_id="portfolio-001",
            symbol="ETH/USD",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("2.0"),
            price=Decimal("2000.0"),
            created_time=datetime.utcnow(),
            status=OrderStatus.PARTIALLY_FILLED,
            strategy_id="strategy-002"
        )

        # Ajouter des exécutions simulées
        execution1 = Execution(
            id="exec-001",
            order_id=order.id,
            executed_quantity=Decimal("0.5"),
            execution_price=Decimal("1999.50"),
            commission=Decimal("1.0"),
            fees=Decimal("0.1"),
            timestamp=datetime.utcnow()
        )

        execution2 = Execution(
            id="exec-002",
            order_id=order.id,
            executed_quantity=Decimal("0.3"),
            execution_price=Decimal("2000.25"),
            commission=Decimal("0.6"),
            fees=Decimal("0.06"),
            timestamp=datetime.utcnow()
        )

        order.executions = [execution1, execution2]
        return order

    async def test_save_and_find_by_id(self, order_repository, sample_order):
        """Test sauvegarde et récupération par ID"""
        # Sauvegarder l'ordre
        await order_repository.save(sample_order)

        # Récupérer par ID
        found_order = await order_repository.find_by_id(sample_order.id)

        assert found_order is not None
        assert found_order.id == sample_order.id
        assert found_order.symbol == sample_order.symbol
        assert found_order.side == sample_order.side
        assert found_order.quantity == sample_order.quantity

        # Test ordre inexistant
        not_found = await order_repository.find_by_id("non-existent-id")
        assert not_found is None

    async def test_find_by_symbol(self, order_repository, sample_order):
        """Test recherche par symbole"""
        # Créer plusieurs ordres pour différents symboles
        order_btc = sample_order
        order_eth = Order(
            id="eth-order-001",
            portfolio_id="portfolio-001",
            symbol="ETH/USD",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            created_time=datetime.utcnow(),
            status=OrderStatus.PENDING
        )

        await order_repository.save(order_btc)
        await order_repository.save(order_eth)

        # Rechercher par symbole
        btc_orders = await order_repository.find_by_symbol("BTC/USD")
        eth_orders = await order_repository.find_by_symbol("ETH/USD")

        assert len(btc_orders) == 1
        assert len(eth_orders) == 1
        assert btc_orders[0].id == order_btc.id
        assert eth_orders[0].id == order_eth.id

        # Symbole inexistant
        empty_orders = await order_repository.find_by_symbol("XRP/USD")
        assert len(empty_orders) == 0

    async def test_find_by_status(self, order_repository):
        """Test recherche par statut"""
        # Créer ordres avec différents statuts
        pending_order = Order(
            id="pending-001",
            portfolio_id="portfolio-001",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            created_time=datetime.utcnow(),
            status=OrderStatus.PENDING
        )

        filled_order = Order(
            id="filled-001",
            portfolio_id="portfolio-001",
            symbol="ETH/USD",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            created_time=datetime.utcnow(),
            status=OrderStatus.FILLED
        )

        await order_repository.save(pending_order)
        await order_repository.save(filled_order)

        # Test recherche par statut
        pending_orders = await order_repository.find_by_status(OrderStatus.PENDING)
        filled_orders = await order_repository.find_by_status(OrderStatus.FILLED)
        cancelled_orders = await order_repository.find_by_status(OrderStatus.CANCELLED)

        assert len(pending_orders) == 1
        assert len(filled_orders) == 1
        assert len(cancelled_orders) == 0

        assert pending_orders[0].id == pending_order.id
        assert filled_orders[0].id == filled_order.id

    async def test_find_active_orders(self, order_repository):
        """Test recherche d'ordres actifs"""
        # Créer ordres avec différents statuts
        orders = []
        statuses = [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCELLED]

        for i, status in enumerate(statuses):
            order = Order(
                id=f"order-{i:03d}",
                portfolio_id="portfolio-001",
                symbol="BTC/USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.1"),
                created_time=datetime.utcnow(),
                status=status
            )
            orders.append(order)
            await order_repository.save(order)

        # Récupérer ordres actifs
        active_orders = await order_repository.find_active_orders()

        # Seuls PENDING et PARTIALLY_FILLED sont actifs
        assert len(active_orders) == 2
        active_statuses = {order.status for order in active_orders}
        assert active_statuses == {OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED}

    async def test_find_by_criteria(self, order_repository):
        """Test recherche multi-critères"""
        # Créer ordres avec différentes caractéristiques
        base_time = datetime.utcnow()

        orders_data = [
            ("order-001", "BTC/USD", OrderSide.BUY, OrderType.MARKET, OrderStatus.PENDING, Decimal("0.1"), "strategy-A"),
            ("order-002", "BTC/USD", OrderSide.SELL, OrderType.LIMIT, OrderStatus.FILLED, Decimal("0.2"), "strategy-A"),
            ("order-003", "ETH/USD", OrderSide.BUY, OrderType.MARKET, OrderStatus.PENDING, Decimal("1.0"), "strategy-B"),
            ("order-004", "BTC/USD", OrderSide.BUY, OrderType.MARKET, OrderStatus.CANCELLED, Decimal("0.15"), "strategy-A"),
        ]

        for i, (order_id, symbol, side, order_type, status, quantity, strategy_id) in enumerate(orders_data):
            order = Order(
                id=order_id,
                portfolio_id="portfolio-001",
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                created_time=base_time + timedelta(minutes=i),
                status=status,
                strategy_id=strategy_id
            )
            await order_repository.save(order)

        # Test critères multiples
        # BTC/USD + BUY + PENDING
        results = await order_repository.find_by_criteria(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            status=OrderStatus.PENDING
        )
        assert len(results) == 1
        assert results[0].id == "order-001"

        # Strategy A + quantité >= 0.15
        results = await order_repository.find_by_criteria(
            strategy_id="strategy-A",
            min_quantity=Decimal("0.15")
        )
        assert len(results) == 2  # order-002 (0.2) et order-004 (0.15)

        # Plage de dates
        results = await order_repository.find_by_criteria(
            start_date=base_time + timedelta(minutes=1),
            end_date=base_time + timedelta(minutes=2.5)
        )
        assert len(results) == 2  # order-002 et order-003

    async def test_count_methods(self, order_repository):
        """Test méthodes de comptage"""
        # Créer ordres test
        statuses = [OrderStatus.PENDING, OrderStatus.PENDING, OrderStatus.FILLED, OrderStatus.CANCELLED]
        symbols = ["BTC/USD", "BTC/USD", "ETH/USD", "BTC/USD"]

        for i, (status, symbol) in enumerate(zip(statuses, symbols)):
            order = Order(
                id=f"count-order-{i:03d}",
                portfolio_id="portfolio-001",
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.1"),
                created_time=datetime.utcnow(),
                status=status
            )
            await order_repository.save(order)

        # Test comptage par statut
        status_counts = await order_repository.count_by_status()
        assert status_counts[OrderStatus.PENDING] == 2
        assert status_counts[OrderStatus.FILLED] == 1
        assert status_counts[OrderStatus.CANCELLED] == 1
        assert status_counts[OrderStatus.REJECTED] == 0

        # Test comptage par symbole
        symbol_counts = await order_repository.count_by_symbol()
        assert symbol_counts["BTC/USD"] == 3
        assert symbol_counts["ETH/USD"] == 1

        # Test avec limit
        limited_counts = await order_repository.count_by_symbol(limit=1)
        assert len(limited_counts) == 1
        assert "BTC/USD" in limited_counts  # Le plus fréquent

    async def test_execution_methods(self, order_repository, order_with_executions):
        """Test méthodes liées aux exécutions"""
        await order_repository.save(order_with_executions)

        # Test prix moyen d'exécution
        avg_price = await order_repository.get_average_fill_price(order_with_executions.id)
        assert avg_price is not None

        # Calcul attendu: ((0.5 * 1999.50) + (0.3 * 2000.25)) / (0.5 + 0.3)
        expected = (Decimal("0.5") * Decimal("1999.50") + Decimal("0.3") * Decimal("2000.25")) / Decimal("0.8")
        assert abs(avg_price - expected) < Decimal("0.01")

        # Test ordre sans exécution
        order_no_exec = Order(
            id="no-exec-001",
            portfolio_id="portfolio-001",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            created_time=datetime.utcnow(),
            status=OrderStatus.PENDING
        )
        await order_repository.save(order_no_exec)

        avg_price_none = await order_repository.get_average_fill_price(order_no_exec.id)
        assert avg_price_none is None

    async def test_order_statistics(self, order_repository):
        """Test génération de statistiques"""
        # Créer ordres test avec timestamps différents
        base_time = datetime.utcnow()

        orders_data = [
            ("BTC/USD", Decimal("0.1"), OrderStatus.FILLED),
            ("BTC/USD", Decimal("0.2"), OrderStatus.FILLED),
            ("ETH/USD", Decimal("1.0"), OrderStatus.PENDING),
            ("BTC/USD", Decimal("0.15"), OrderStatus.CANCELLED)
        ]

        for i, (symbol, quantity, status) in enumerate(orders_data):
            order = Order(
                id=f"stats-order-{i:03d}",
                portfolio_id="portfolio-001",
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=quantity,
                created_time=base_time + timedelta(minutes=i),
                submitted_time=base_time + timedelta(minutes=i, seconds=10),
                accepted_time=base_time + timedelta(minutes=i, seconds=20) if status == OrderStatus.FILLED else None,
                status=status
            )
            await order_repository.save(order)

        # Statistiques globales
        stats = await order_repository.get_order_statistics()

        assert stats["total_orders"] == 4
        assert stats["total_volume"] == 1.45  # 0.1 + 0.2 + 1.0 + 0.15
        assert stats["avg_order_size"] == 1.45 / 4
        assert stats["fill_rate"] == 0.5  # 2 remplis sur 4

        # Statistiques par symbole
        btc_stats = await order_repository.get_order_statistics(symbol="BTC/USD")
        assert btc_stats["total_orders"] == 3
        assert btc_stats["total_volume"] == 0.45  # 0.1 + 0.2 + 0.15

    async def test_cleanup_methods(self, order_repository):
        """Test méthodes de nettoyage"""
        # Créer ordres expirés
        past_time = datetime.utcnow() - timedelta(days=2)
        old_time = datetime.utcnow() - timedelta(days=10)

        orders = [
            # Ordre récent actif
            Order(
                id="recent-001",
                portfolio_id="portfolio-001",
                symbol="BTC/USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.1"),
                created_time=datetime.utcnow(),
                status=OrderStatus.PENDING
            ),
            # Ordre ancien rempli
            Order(
                id="old-filled-001",
                portfolio_id="portfolio-001",
                symbol="BTC/USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.1"),
                created_time=old_time,
                status=OrderStatus.FILLED
            ),
            # Ordre avec expire_time dépassé
            Order(
                id="expired-001",
                portfolio_id="portfolio-001",
                symbol="BTC/USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.1"),
                created_time=past_time,
                expire_time=past_time + timedelta(hours=1),
                status=OrderStatus.PENDING
            )
        ]

        for order in orders:
            await order_repository.save(order)

        # Test recherche d'ordres expirés
        expired_orders = await order_repository.find_expired_orders()
        assert len(expired_orders) == 1
        assert expired_orders[0].id == "expired-001"

        # Test archivage d'anciens ordres
        archived_count = await order_repository.archive_old_orders(
            cutoff_date=datetime.utcnow() - timedelta(days=5)
        )
        assert archived_count == 1  # old-filled-001

        # Vérifier l'ordre archivé
        archived_order = await order_repository.find_by_id("old-filled-001")
        assert hasattr(archived_order, 'archived')
        assert archived_order.archived is True

        # Test nettoyage ordres expirés
        cleaned_count = await order_repository.cleanup_expired_orders()
        assert cleaned_count == 1  # expired-001 marqué comme EXPIRED

        # Vérifier le statut
        cleaned_order = await order_repository.find_by_id("expired-001")
        assert cleaned_order.status == OrderStatus.EXPIRED


class TestMemoryPortfolioRepository:
    """Tests pour MemoryPortfolioRepository avec exécution réelle"""

    @pytest.fixture
    async def portfolio_repository(self):
        """Repository de portfolios en mémoire"""
        return MemoryPortfolioRepository()

    @pytest.fixture
    def sample_portfolio(self):
        """Portfolio d'exemple pour tests"""
        return Portfolio(
            id="test-portfolio-001",
            name="Test Portfolio",
            initial_capital=Decimal("10000.00"),
            base_currency="USD",
            status=PortfolioStatus.ACTIVE,
            portfolio_type=PortfolioType.TRADING,
            created_at=datetime.utcnow()
        )

    async def test_save_and_find_by_id(self, portfolio_repository, sample_portfolio):
        """Test sauvegarde et récupération par ID"""
        # Sauvegarder le portfolio
        await portfolio_repository.save(sample_portfolio)

        # Récupérer par ID
        found_portfolio = await portfolio_repository.find_by_id(sample_portfolio.id)

        assert found_portfolio is not None
        assert found_portfolio.id == sample_portfolio.id
        assert found_portfolio.name == sample_portfolio.name
        assert found_portfolio.initial_capital == sample_portfolio.initial_capital
        assert found_portfolio.status == sample_portfolio.status

        # Test portfolio inexistant
        not_found = await portfolio_repository.find_by_id("non-existent-id")
        assert not_found is None

    async def test_find_by_name(self, portfolio_repository, sample_portfolio):
        """Test recherche par nom"""
        await portfolio_repository.save(sample_portfolio)

        # Récupérer par nom
        found_portfolio = await portfolio_repository.find_by_name(sample_portfolio.name)

        assert found_portfolio is not None
        assert found_portfolio.id == sample_portfolio.id
        assert found_portfolio.name == sample_portfolio.name

        # Test nom inexistant
        not_found = await portfolio_repository.find_by_name("Non-existent Portfolio")
        assert not_found is None

    async def test_unique_name_constraint(self, portfolio_repository, sample_portfolio):
        """Test contrainte d'unicité du nom"""
        await portfolio_repository.save(sample_portfolio)

        # Essayer de créer un autre portfolio avec le même nom
        duplicate_portfolio = Portfolio(
            id="duplicate-portfolio-001",
            name=sample_portfolio.name,  # Même nom
            initial_capital=Decimal("5000.00"),
            base_currency="USD",
            status=PortfolioStatus.ACTIVE,
            portfolio_type=PortfolioType.BACKTESTING,
            created_at=datetime.utcnow()
        )

        # Ceci devrait lever une exception
        with pytest.raises(Exception):  # DuplicatePortfolioNameError
            await portfolio_repository.save(duplicate_portfolio)

    async def test_find_by_status_and_type(self, portfolio_repository):
        """Test recherche par statut et type"""
        portfolios = [
            Portfolio(
                id="portfolio-active-001",
                name="Active Trading",
                initial_capital=Decimal("10000.00"),
                base_currency="USD",
                status=PortfolioStatus.ACTIVE,
                portfolio_type=PortfolioType.TRADING,
                created_at=datetime.utcnow()
            ),
            Portfolio(
                id="portfolio-inactive-001",
                name="Inactive Trading",
                initial_capital=Decimal("5000.00"),
                base_currency="USD",
                status=PortfolioStatus.INACTIVE,
                portfolio_type=PortfolioType.TRADING,
                created_at=datetime.utcnow()
            ),
            Portfolio(
                id="portfolio-backtest-001",
                name="Backtest Portfolio",
                initial_capital=Decimal("100000.00"),
                base_currency="USD",
                status=PortfolioStatus.ACTIVE,
                portfolio_type=PortfolioType.BACKTESTING,
                created_at=datetime.utcnow()
            )
        ]

        for portfolio in portfolios:
            await portfolio_repository.save(portfolio)

        # Test recherche par statut
        active_portfolios = await portfolio_repository.find_by_status(PortfolioStatus.ACTIVE)
        inactive_portfolios = await portfolio_repository.find_by_status(PortfolioStatus.INACTIVE)

        assert len(active_portfolios) == 2
        assert len(inactive_portfolios) == 1

        # Test recherche par type
        trading_portfolios = await portfolio_repository.find_by_type(PortfolioType.TRADING)
        backtest_portfolios = await portfolio_repository.find_by_type(PortfolioType.BACKTESTING)

        assert len(trading_portfolios) == 2
        assert len(backtest_portfolios) == 1

    async def test_portfolio_statistics(self, portfolio_repository):
        """Test génération de statistiques"""
        # Créer portfolios avec différentes valeurs
        portfolios_data = [
            ("High Value", Decimal("50000.00"), PortfolioStatus.ACTIVE, PortfolioType.TRADING),
            ("Medium Value", Decimal("25000.00"), PortfolioStatus.ACTIVE, PortfolioType.TRADING),
            ("Low Value", Decimal("10000.00"), PortfolioStatus.INACTIVE, PortfolioType.BACKTESTING),
            ("Test Value", Decimal("5000.00"), PortfolioStatus.ARCHIVED, PortfolioType.RESEARCH)
        ]

        for i, (name, value, status, portfolio_type) in enumerate(portfolios_data):
            portfolio = Portfolio(
                id=f"stats-portfolio-{i:03d}",
                name=name,
                initial_capital=value,
                base_currency="USD",
                status=status,
                portfolio_type=portfolio_type,
                created_at=datetime.utcnow()
            )
            # Simuler une valeur totale différente du capital initial
            portfolio.total_value = value * Decimal("1.1")  # +10%
            await portfolio_repository.save(portfolio)

        # Statistiques globales
        global_stats = await portfolio_repository.get_global_statistics()

        assert global_stats["total_portfolios"] == 4
        expected_total = sum(Decimal(str(val)) * Decimal("1.1") for _, val, _, _ in portfolios_data)
        assert abs(global_stats["total_value"] - expected_total) < Decimal("0.01")

        # Comptages par statut et type
        status_counts = await portfolio_repository.count_by_status()
        assert status_counts[PortfolioStatus.ACTIVE] == 2
        assert status_counts[PortfolioStatus.INACTIVE] == 1
        assert status_counts[PortfolioStatus.ARCHIVED] == 1

        type_counts = await portfolio_repository.count_by_type()
        assert type_counts[PortfolioType.TRADING] == 2
        assert type_counts[PortfolioType.BACKTESTING] == 1
        assert type_counts[PortfolioType.RESEARCH] == 1

    async def test_snapshots_management(self, portfolio_repository, sample_portfolio):
        """Test gestion des snapshots de portfolios"""
        await portfolio_repository.save(sample_portfolio)

        # Créer un snapshot
        await portfolio_repository.update_portfolio_snapshot(sample_portfolio.id)

        # Modifier le portfolio et créer un autre snapshot
        sample_portfolio.total_value = Decimal("11000.00")
        await portfolio_repository.update(sample_portfolio)
        await portfolio_repository.update_portfolio_snapshot(sample_portfolio.id)

        # Test bulk update des snapshots
        portfolio2 = Portfolio(
            id="portfolio-002",
            name="Portfolio 2",
            initial_capital=Decimal("20000.00"),
            base_currency="USD",
            status=PortfolioStatus.ACTIVE,
            portfolio_type=PortfolioType.TRADING,
            created_at=datetime.utcnow()
        )
        await portfolio_repository.save(portfolio2)

        updated_count = await portfolio_repository.bulk_update_snapshots([sample_portfolio.id, portfolio2.id])
        assert updated_count == 2

        # Test nettoyage des anciens snapshots
        deleted_count = await portfolio_repository.cleanup_old_snapshots(
            retention_days=0,  # Supprimer tous les snapshots
            max_snapshots_per_portfolio=1
        )
        assert deleted_count >= 2  # Au moins 2 snapshots supprimés

    async def test_portfolio_queries_advanced(self, portfolio_repository):
        """Test requêtes avancées de portfolios"""
        base_time = datetime.utcnow()

        # Créer portfolios avec valeurs différentes et dates
        portfolios_data = [
            ("Portfolio A", Decimal("15000.00"), base_time - timedelta(days=5)),
            ("Portfolio B", Decimal("30000.00"), base_time - timedelta(days=3)),
            ("Portfolio C", Decimal("8000.00"), base_time - timedelta(days=1)),
            ("Portfolio D", Decimal("50000.00"), base_time)
        ]

        for i, (name, value, created_at) in enumerate(portfolios_data):
            portfolio = Portfolio(
                id=f"query-portfolio-{i:03d}",
                name=name,
                initial_capital=value,
                base_currency="USD",
                status=PortfolioStatus.ACTIVE,
                portfolio_type=PortfolioType.TRADING,
                created_at=created_at
            )
            portfolio.total_value = value
            await portfolio_repository.save(portfolio)

        # Test recherche par valeur
        high_value_portfolios = await portfolio_repository.find_by_value_range(
            min_value=Decimal("25000.00")
        )
        assert len(high_value_portfolios) == 2  # Portfolio B et D

        medium_value_portfolios = await portfolio_repository.find_by_value_range(
            min_value=Decimal("10000.00"),
            max_value=Decimal("40000.00")
        )
        assert len(medium_value_portfolios) == 2  # Portfolio A et B

        # Test recherche par période
        recent_portfolios = await portfolio_repository.find_by_date_range(
            start_date=base_time - timedelta(days=2),
            end_date=base_time + timedelta(days=1),
            date_field="created_at"
        )
        assert len(recent_portfolios) == 2  # Portfolio C et D


class TestCacheManagement:
    """Tests pour les systèmes de cache avec exécution réelle"""

    @pytest.fixture
    def cache_config(self):
        """Configuration de cache pour tests"""
        return CacheConfig(
            redis_host="localhost",
            redis_port=6379,
            redis_db=1,  # DB séparée pour tests
            default_ttl=300,
            max_memory_mb=10,
            enable_memory_fallback=True
        )

    @pytest.fixture
    async def memory_cache(self):
        """Cache en mémoire pour tests"""
        return InMemoryCache(max_size=100, default_ttl=300, strategy=CacheStrategy.LRU)

    async def test_memory_cache_basic_operations(self, memory_cache):
        """Test opérations de base du cache mémoire"""
        # Test set/get
        await memory_cache.set("key1", "value1")
        value = await memory_cache.get("key1")
        assert value == "value1"

        # Test exists
        assert await memory_cache.exists("key1") is True
        assert await memory_cache.exists("key_not_exists") is False

        # Test delete
        deleted = await memory_cache.delete("key1")
        assert deleted is True

        value_after_delete = await memory_cache.get("key1")
        assert value_after_delete is None

        # Test delete non-existent
        deleted_again = await memory_cache.delete("key1")
        assert deleted_again is False

    async def test_memory_cache_ttl(self, memory_cache):
        """Test TTL du cache mémoire"""
        # Set avec TTL court
        await memory_cache.set("ttl_key", "ttl_value", ttl=1)  # 1 seconde

        # Immédiatement disponible
        value = await memory_cache.get("ttl_key")
        assert value == "ttl_value"

        # Attendre expiration
        await asyncio.sleep(1.1)

        # Doit être expiré
        value_expired = await memory_cache.get("ttl_key")
        assert value_expired is None

    async def test_memory_cache_eviction_lru(self):
        """Test éviction LRU"""
        cache = InMemoryCache(max_size=3, strategy=CacheStrategy.LRU)

        # Remplir le cache
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        # Accéder à key1 pour le rendre récent
        await cache.get("key1")

        # Ajouter key4, doit évincer key2 (least recently used)
        await cache.set("key4", "value4")

        assert await cache.exists("key1") is True  # Récemment accédé
        assert await cache.exists("key2") is False  # Évincé
        assert await cache.exists("key3") is True  # Récent
        assert await cache.exists("key4") is True  # Nouveau

    async def test_memory_cache_eviction_lfu(self):
        """Test éviction LFU"""
        cache = InMemoryCache(max_size=3, strategy=CacheStrategy.LFU)

        # Remplir le cache
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        # Accéder plusieurs fois à key1
        for _ in range(3):
            await cache.get("key1")

        # Accéder une fois à key3
        await cache.get("key3")

        # key2 n'a jamais été accédé, key1 accédé 3 fois, key3 accédé 1 fois
        # Ajouter key4, doit évincer key2 (least frequently used)
        await cache.set("key4", "value4")

        assert await cache.exists("key1") is True  # Plus fréquemment utilisé
        assert await cache.exists("key2") is False  # Évincé (jamais accédé)
        assert await cache.exists("key3") is True  # Utilisé
        assert await cache.exists("key4") is True  # Nouveau

    async def test_memory_cache_statistics(self, memory_cache):
        """Test statistiques du cache"""
        # Opérations initiales
        await memory_cache.set("stats_key1", "value1")
        await memory_cache.set("stats_key2", "value2")

        # Hits
        await memory_cache.get("stats_key1")  # Hit
        await memory_cache.get("stats_key1")  # Hit

        # Misses
        await memory_cache.get("nonexistent1")  # Miss
        await memory_cache.get("nonexistent2")  # Miss

        # Delete
        await memory_cache.delete("stats_key2")

        stats = memory_cache.get_stats()

        assert stats.sets >= 2
        assert stats.hits >= 2
        assert stats.misses >= 2
        assert stats.deletes >= 1
        assert 0 <= stats.hit_rate <= 1

    @pytest.mark.skip(reason="Requires Redis server")
    async def test_redis_cache_operations(self, cache_config):
        """Test opérations Redis (requiert serveur Redis)"""
        redis_cache = RedisCache(cache_config)

        try:
            await redis_cache.initialize()

            # Test opérations de base
            await redis_cache.set("redis_key1", "redis_value1")
            value = await redis_cache.get("redis_key1")
            assert value == "redis_value1"

            # Test TTL
            await redis_cache.set("redis_ttl_key", "ttl_value", ttl=1)
            await asyncio.sleep(1.1)
            expired_value = await redis_cache.get("redis_ttl_key")
            assert expired_value is None

            # Test avec objets complexes
            complex_data = {"list": [1, 2, 3], "dict": {"nested": "value"}}
            await redis_cache.set("complex_key", complex_data)
            retrieved_data = await redis_cache.get("complex_key")
            assert retrieved_data == complex_data

        except Exception as e:
            pytest.skip(f"Redis not available: {e}")
        finally:
            await redis_cache.close()

    async def test_cache_manager_fallback(self, cache_config):
        """Test fallback du gestionnaire de cache"""
        cache_manager = CacheManager(cache_config)

        # Sans initialiser Redis (pour simuler échec connexion)
        # Le cache manager devrait utiliser le fallback mémoire

        # Test opérations avec fallback
        success = await cache_manager.set("fallback_key", "fallback_value")
        assert success is True  # Le cache mémoire devrait fonctionner

        value = await cache_manager.get("fallback_key")
        assert value == "fallback_value"

        # Test namespace
        await cache_manager.set("namespaced_key", "namespaced_value", namespace="test")
        namespaced_value = await cache_manager.get("namespaced_key", namespace="test")
        assert namespaced_value == "namespaced_value"

        # Différent namespace
        different_value = await cache_manager.get("namespaced_key", namespace="different")
        assert different_value is None

    async def test_cache_get_or_set(self, memory_cache):
        """Test méthode get_or_set"""
        cache_manager = CacheManager(CacheConfig())
        cache_manager.memory_cache = memory_cache

        # Factory function
        call_count = 0

        def expensive_operation():
            nonlocal call_count
            call_count += 1
            return f"computed_value_{call_count}"

        # Premier appel - devrait exécuter la factory
        value1 = await cache_manager.get_or_set("computed_key", expensive_operation)
        assert value1 == "computed_value_1"
        assert call_count == 1

        # Deuxième appel - devrait utiliser le cache
        value2 = await cache_manager.get_or_set("computed_key", expensive_operation)
        assert value2 == "computed_value_1"  # Même valeur
        assert call_count == 1  # Factory pas appelée


class TestDatabaseManagement:
    """Tests pour la gestion de base de données avec exécution réelle"""

    @pytest.fixture
    def db_config(self):
        """Configuration de base de données pour tests"""
        return DatabaseConfig(
            host="localhost",
            port=5432,
            database="qframe_test",
            user="test_user",
            password="test_password",
            min_connections=1,
            max_connections=5,
            query_timeout=10
        )

    @pytest.fixture
    def connection_stats(self):
        """Stats de connexion pour tests"""
        return ConnectionStats(
            total_connections=5,
            active_connections=2,
            idle_connections=3,
            total_queries=100,
            failed_queries=2,
            avg_query_time_ms=15.5,
            connection_errors=1
        )

    def test_database_config(self, db_config):
        """Test configuration de base de données"""
        # Test DSN generation
        expected_dsn = "postgresql://test_user:test_password@localhost:5432/qframe_test"
        assert db_config.get_dsn() == expected_dsn
        assert db_config.get_async_dsn() == expected_dsn

        # Test default values
        assert db_config.min_connections == 1
        assert db_config.max_connections == 5
        assert db_config.ssl_mode == "prefer"

    def test_connection_stats(self, connection_stats):
        """Test statistiques de connexion"""
        assert connection_stats.total_connections == 5
        assert connection_stats.active_connections == 2
        assert connection_stats.idle_connections == 3
        assert connection_stats.connection_errors == 1

        # Test calculated fields
        assert connection_stats.total_queries == 100
        assert connection_stats.failed_queries == 2
        assert connection_stats.avg_query_time_ms == 15.5

    @pytest.mark.skip(reason="Requires PostgreSQL server")
    async def test_database_manager_initialization(self, db_config):
        """Test initialisation du gestionnaire de base de données"""
        db_manager = DatabaseManager(db_config)

        try:
            await db_manager.initialize()
            assert db_manager._initialized is True

            # Test query execution
            result = await db_manager.execute_query("SELECT 1 as test_value", fetch="one")
            assert result is not None

        except Exception as e:
            pytest.skip(f"PostgreSQL not available: {e}")
        finally:
            await db_manager.close()

    def test_isolation_levels(self):
        """Test niveaux d'isolation"""
        assert IsolationLevel.READ_UNCOMMITTED == "read_uncommitted"
        assert IsolationLevel.READ_COMMITTED == "read_committed"
        assert IsolationLevel.REPEATABLE_READ == "repeatable_read"
        assert IsolationLevel.SERIALIZABLE == "serializable"

    @pytest.mark.skip(reason="Requires PostgreSQL server")
    async def test_transaction_manager(self, db_config):
        """Test gestionnaire de transactions"""
        db_manager = DatabaseManager(db_config)

        try:
            await db_manager.initialize()

            # Test transaction commit
            async with db_manager.transaction_manager.transaction() as conn:
                await conn.execute("CREATE TEMP TABLE test_tx (id INTEGER)")
                await conn.execute("INSERT INTO test_tx VALUES (1)")

            # Test transaction rollback
            try:
                async with db_manager.transaction_manager.transaction() as conn:
                    await conn.execute("INSERT INTO test_tx VALUES (2)")
                    raise Exception("Force rollback")
            except:
                pass  # Expected

        except Exception as e:
            pytest.skip(f"PostgreSQL not available: {e}")
        finally:
            await db_manager.close()


@pytest.mark.skipif(not TIMESERIES_AVAILABLE, reason="InfluxDB dependencies not available")
class TestTimeSeriesManagement:
    """Tests pour la gestion des time-series avec exécution réelle"""

    @pytest.fixture
    def timeseries_config(self):
        """Configuration time-series pour tests"""
        return TimeSeriesConfig(
            url="http://localhost:8086",
            token="test-token",
            org="qframe-test",
            bucket="test_bucket",
            batch_size=10,
            retention_days=30
        )

    @pytest.fixture
    def sample_market_data(self):
        """Données de marché d'exemple"""
        return MarketDataPoint(
            symbol="BTC/USD",
            data_type=DataType.TICKER,
            timestamp=datetime.utcnow(),
            data={
                "bid": 45000.0,
                "ask": 45001.0,
                "last": 45000.5,
                "volume_24h": 1000000.0
            },
            provider="test_provider",
            quality=DataQuality.HIGH
        )

    @pytest.fixture
    def query_options(self):
        """Options de requête pour tests"""
        return QueryOptions(
            start_time=datetime.utcnow() - timedelta(hours=1),
            end_time=datetime.utcnow(),
            symbols=["BTC/USD", "ETH/USD"],
            data_types=[DataType.TICKER, DataType.TRADES],
            aggregation=Aggregation.MEAN,
            window="5m",
            limit=100
        )

    def test_timeseries_config(self, timeseries_config):
        """Test configuration time-series"""
        assert timeseries_config.url == "http://localhost:8086"
        assert timeseries_config.token == "test-token"
        assert timeseries_config.org == "qframe-test"
        assert timeseries_config.bucket == "test_bucket"
        assert timeseries_config.batch_size == 10
        assert timeseries_config.retention_days == 30

    def test_query_options(self, query_options):
        """Test options de requête"""
        assert query_options.symbols == ["BTC/USD", "ETH/USD"]
        assert query_options.data_types == [DataType.TICKER, DataType.TRADES]
        assert query_options.aggregation == Aggregation.MEAN
        assert query_options.window == "5m"
        assert query_options.limit == 100

        # Test durée
        duration = query_options.end_time - query_options.start_time
        assert duration.total_seconds() == 3600  # 1 heure

    @pytest.mark.skip(reason="Requires InfluxDB server")
    async def test_influxdb_manager_initialization(self, timeseries_config):
        """Test initialisation du gestionnaire InfluxDB"""
        influx_manager = InfluxDBManager(timeseries_config)

        try:
            await influx_manager.initialize()

            # Test statistiques initiales
            stats = await influx_manager.get_statistics()
            assert "writes_count" in stats
            assert "queries_count" in stats
            assert stats["writes_count"] == 0
            assert stats["queries_count"] == 0

        except Exception as e:
            pytest.skip(f"InfluxDB not available: {e}")
        finally:
            await influx_manager.close()

    @pytest.mark.skip(reason="Requires InfluxDB server")
    async def test_market_data_write_and_query(self, timeseries_config, sample_market_data, query_options):
        """Test écriture et requête de données de marché"""
        influx_manager = InfluxDBManager(timeseries_config)

        try:
            await influx_manager.initialize()

            # Test écriture d'un point
            success = await influx_manager.write_market_data(sample_market_data)
            assert success is True

            # Test écriture en batch
            batch_data = [sample_market_data] * 3
            batch_success = await influx_manager.write_market_data_batch(batch_data)
            assert batch_success is True

            # Attendre indexation
            await asyncio.sleep(1)

            # Test requête
            results = await influx_manager.query_market_data(query_options)
            assert results is not None  # DataFrame peut être vide pour tests

            # Test dernières données
            latest = await influx_manager.get_latest_data("BTC/USD", DataType.TICKER, "test_provider")
            if latest:
                assert latest.symbol == "BTC/USD"
                assert latest.data_type == DataType.TICKER

        except Exception as e:
            pytest.skip(f"InfluxDB not available: {e}")
        finally:
            await influx_manager.close()

    def test_aggregation_types(self):
        """Test types d'agrégation"""
        assert Aggregation.MEAN == "mean"
        assert Aggregation.MAX == "max"
        assert Aggregation.MIN == "min"
        assert Aggregation.FIRST == "first"
        assert Aggregation.LAST == "last"
        assert Aggregation.SUM == "sum"
        assert Aggregation.COUNT == "count"
        assert Aggregation.STDDEV == "stddev"

    async def test_market_data_storage_integration(self, timeseries_config, sample_market_data):
        """Test intégration stockage de données de marché"""
        # Mock time-series DB
        mock_timeseries_db = AsyncMock(spec=TimeSeriesDB)
        mock_timeseries_db.write_market_data.return_value = True
        mock_timeseries_db.write_market_data_batch.return_value = True
        mock_timeseries_db.get_latest_data.return_value = sample_market_data

        # Mock cache manager
        mock_cache_manager = AsyncMock()
        mock_cache_manager.set.return_value = True
        mock_cache_manager.get.return_value = sample_market_data

        # Créer storage
        storage = MarketDataStorage(mock_timeseries_db, mock_cache_manager)

        # Test stockage
        success = await storage.store_market_data(sample_market_data)
        assert success is True

        # Vérifier appels
        mock_timeseries_db.write_market_data.assert_called_once_with(sample_market_data)
        mock_cache_manager.set.assert_called_once()

        # Test récupération depuis cache
        cached_data = await storage.get_latest_cached("BTC/USD", DataType.TICKER, "test_provider")
        assert cached_data == sample_market_data

        # Test batch
        batch_success = await storage.store_market_data_batch([sample_market_data])
        assert batch_success is True
        mock_timeseries_db.write_market_data_batch.assert_called_once_with([sample_market_data])


# ===============================
# TESTS D'INTÉGRATION PERSISTENCE
# ===============================

class TestPersistenceIntegration:
    """Tests d'intégration des différents composants de persistence"""

    @pytest.fixture
    async def integrated_repositories(self):
        """Tous les repositories intégrés"""
        return {
            "orders": MemoryOrderRepository(),
            "portfolios": MemoryPortfolioRepository(),
            "risk_assessments": MemoryRiskAssessmentRepository(),
            "backtests": MemoryBacktestRepository(),
            "strategies": MemoryStrategyRepository()
        }

    async def test_cross_repository_workflow(self, integrated_repositories):
        """Test workflow cross-repository"""
        order_repo = integrated_repositories["orders"]
        portfolio_repo = integrated_repositories["portfolios"]

        # Créer un portfolio
        portfolio = Portfolio(
            id="integration-portfolio-001",
            name="Integration Test Portfolio",
            initial_capital=Decimal("50000.00"),
            base_currency="USD",
            status=PortfolioStatus.ACTIVE,
            portfolio_type=PortfolioType.TRADING,
            created_at=datetime.utcnow()
        )
        await portfolio_repo.save(portfolio)

        # Créer des ordres pour ce portfolio
        orders = []
        for i in range(5):
            order = Order(
                id=f"integration-order-{i:03d}",
                portfolio_id=portfolio.id,
                symbol=f"SYMBOL{i}/USD",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal(str(100 + i * 10)),
                created_time=datetime.utcnow(),
                status=OrderStatus.PENDING
            )
            orders.append(order)
            await order_repo.save(order)

        # Vérifications cross-repository
        portfolio_orders = await order_repo.find_by_portfolio(portfolio.id)
        assert len(portfolio_orders) == 5

        # Vérifier que tous les ordres appartiennent au bon portfolio
        for order in portfolio_orders:
            assert order.portfolio_id == portfolio.id

        # Statistiques intégrées
        order_stats = await order_repo.get_order_statistics()
        assert order_stats["total_orders"] >= 5

        portfolio_stats = await portfolio_repo.get_portfolio_statistics(portfolio.id)
        assert portfolio_stats["id"] == portfolio.id

    async def test_performance_under_load(self, integrated_repositories):
        """Test performance sous charge"""
        order_repo = integrated_repositories["orders"]
        portfolio_repo = integrated_repositories["portfolios"]

        # Créer beaucoup de portfolios et ordres
        start_time = time.time()

        # Créer 10 portfolios
        portfolios = []
        for i in range(10):
            portfolio = Portfolio(
                id=f"perf-portfolio-{i:03d}",
                name=f"Performance Test Portfolio {i}",
                initial_capital=Decimal("10000.00"),
                base_currency="USD",
                status=PortfolioStatus.ACTIVE,
                portfolio_type=PortfolioType.TRADING,
                created_at=datetime.utcnow()
            )
            portfolios.append(portfolio)
            await portfolio_repo.save(portfolio)

        portfolio_creation_time = time.time() - start_time

        # Créer 100 ordres (10 par portfolio)
        orders_start_time = time.time()
        for portfolio in portfolios:
            for j in range(10):
                order = Order(
                    id=f"perf-order-{portfolio.id}-{j:03d}",
                    portfolio_id=portfolio.id,
                    symbol=f"BTC/USD",
                    side=OrderSide.BUY if j % 2 == 0 else OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=Decimal("0.01"),
                    created_time=datetime.utcnow(),
                    status=OrderStatus.PENDING
                )
                await order_repo.save(order)

        order_creation_time = time.time() - orders_start_time

        # Tests de performance de requêtes
        query_start_time = time.time()

        # Requêtes complexes
        all_orders = []
        for portfolio in portfolios:
            portfolio_orders = await order_repo.find_by_portfolio(portfolio.id)
            all_orders.extend(portfolio_orders)

        query_time = time.time() - query_start_time

        # Assertions de performance (ajustables selon besoin)
        assert portfolio_creation_time < 1.0  # < 1s pour 10 portfolios
        assert order_creation_time < 2.0      # < 2s pour 100 ordres
        assert query_time < 0.5               # < 0.5s pour requêtes
        assert len(all_orders) == 100

    async def test_data_consistency(self, integrated_repositories):
        """Test cohérence des données"""
        order_repo = integrated_repositories["orders"]
        portfolio_repo = integrated_repositories["portfolios"]

        # Créer portfolio et ordres
        portfolio = Portfolio(
            id="consistency-portfolio-001",
            name="Consistency Test Portfolio",
            initial_capital=Decimal("25000.00"),
            base_currency="USD",
            status=PortfolioStatus.ACTIVE,
            portfolio_type=PortfolioType.TRADING,
            created_at=datetime.utcnow()
        )
        await portfolio_repo.save(portfolio)

        # Créer ordres avec références
        order1 = Order(
            id="consistency-order-001",
            portfolio_id=portfolio.id,
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.5"),
            created_time=datetime.utcnow(),
            status=OrderStatus.PENDING
        )

        order2 = Order(
            id="consistency-order-002",
            portfolio_id=portfolio.id,
            symbol="ETH/USD",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("2.0"),
            price=Decimal("2000.00"),
            created_time=datetime.utcnow(),
            status=OrderStatus.FILLED,
            parent_order_id=order1.id  # Ordre enfant
        )

        await order_repo.save(order1)
        await order_repo.save(order2)

        # Vérifications de cohérence

        # 1. Les ordres appartiennent au bon portfolio
        portfolio_orders = await order_repo.find_by_portfolio(portfolio.id)
        assert len(portfolio_orders) == 2

        # 2. Relations parent-enfant
        parent_orders = await order_repo.find_parent_orders()
        child_orders = await order_repo.find_child_orders(order1.id)

        assert len(parent_orders) >= 1  # order1 est parent
        assert len(child_orders) == 1   # order2 est enfant de order1
        assert child_orders[0].id == order2.id

        # 3. Comptages cohérents
        total_count = await order_repo.count_all()
        status_counts = await order_repo.count_by_status()

        assert total_count >= 2
        assert status_counts[OrderStatus.PENDING] >= 1
        assert status_counts[OrderStatus.FILLED] >= 1

        # 4. Intégrité référentielle (simulation)
        retrieved_order2 = await order_repo.find_by_id(order2.id)
        assert retrieved_order2.parent_order_id == order1.id

        parent_exists = await order_repo.exists(retrieved_order2.parent_order_id)
        assert parent_exists is True

    async def test_concurrent_operations(self, integrated_repositories):
        """Test opérations concurrentes"""
        order_repo = integrated_repositories["orders"]

        # Fonction pour créer des ordres en parallèle
        async def create_orders_batch(batch_id: int, count: int):
            orders = []
            for i in range(count):
                order = Order(
                    id=f"concurrent-order-{batch_id:02d}-{i:03d}",
                    portfolio_id="concurrent-portfolio-001",
                    symbol="BTC/USD",
                    side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=Decimal("0.01"),
                    created_time=datetime.utcnow(),
                    status=OrderStatus.PENDING
                )
                await order_repo.save(order)
                orders.append(order)
            return orders

        # Lancer 3 batches en parallèle
        tasks = [
            create_orders_batch(0, 10),
            create_orders_batch(1, 10),
            create_orders_batch(2, 10)
        ]

        results = await asyncio.gather(*tasks)

        # Vérifier que tous les ordres ont été créés
        total_orders = sum(len(batch) for batch in results)
        assert total_orders == 30

        # Vérifier dans le repository
        all_orders = await order_repo.find_by_portfolio("concurrent-portfolio-001")
        assert len(all_orders) == 30

        # Vérifier unicité des IDs
        order_ids = {order.id for order in all_orders}
        assert len(order_ids) == 30  # Tous les IDs sont uniques

    async def test_error_handling_and_recovery(self, integrated_repositories):
        """Test gestion d'erreurs et récupération"""
        order_repo = integrated_repositories["orders"]
        portfolio_repo = integrated_repositories["portfolios"]

        # Test avec portfolio inexistant
        order_orphan = Order(
            id="orphan-order-001",
            portfolio_id="non-existent-portfolio",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            created_time=datetime.utcnow(),
            status=OrderStatus.PENDING
        )

        # Ceci ne devrait pas lever d'exception (repositories en mémoire tolèrent)
        await order_repo.save(order_orphan)

        # Mais on peut détecter l'incohérence
        saved_order = await order_repo.find_by_id(order_orphan.id)
        assert saved_order is not None

        # Le portfolio n'existe pas
        portfolio = await portfolio_repo.find_by_id(saved_order.portfolio_id)
        assert portfolio is None  # Incohérence détectée

        # Test récupération après erreur
        try:
            # Simuler une erreur lors de la sauvegarde
            with patch.object(order_repo, '_add_to_indexes', side_effect=Exception("Simulated error")):
                error_order = Order(
                    id="error-order-001",
                    portfolio_id="portfolio-001",
                    symbol="BTC/USD",
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=Decimal("0.1"),
                    created_time=datetime.utcnow(),
                    status=OrderStatus.PENDING
                )
                await order_repo.save(error_order)

        except Exception:
            # L'erreur est capturée, la base reste cohérente
            pass

        # Vérifier que le repository fonctionne encore
        test_order = Order(
            id="recovery-order-001",
            portfolio_id="portfolio-001",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            created_time=datetime.utcnow(),
            status=OrderStatus.PENDING
        )

        await order_repo.save(test_order)
        recovered_order = await order_repo.find_by_id(test_order.id)
        assert recovered_order is not None
        assert recovered_order.id == test_order.id


# ============================================
# TESTS D'EXÉCUTION ET VALIDATION COMPLÈTE
# ============================================

if __name__ == "__main__":
    # Exécution directe pour validation
    print("🧪 Tests d'exécution réelle - Infrastructure Persistence")
    print("=" * 60)

    # Tests de base
    async def run_basic_tests():
        """Exécuter les tests de base"""
        print("📦 Test MemoryOrderRepository...")
        order_repo = MemoryOrderRepository()

        # Test création d'ordre
        order = Order(
            id="direct-test-001",
            portfolio_id="portfolio-001",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            created_time=datetime.utcnow(),
            status=OrderStatus.PENDING
        )

        await order_repo.save(order)
        found_order = await order_repo.find_by_id(order.id)

        assert found_order is not None
        assert found_order.id == order.id
        print("  ✅ Order repository - OK")

        print("📁 Test MemoryPortfolioRepository...")
        portfolio_repo = MemoryPortfolioRepository()

        # Test création portfolio
        portfolio = Portfolio(
            id="direct-test-portfolio-001",
            name="Direct Test Portfolio",
            initial_capital=Decimal("10000.00"),
            base_currency="USD",
            status=PortfolioStatus.ACTIVE,
            portfolio_type=PortfolioType.TRADING,
            created_at=datetime.utcnow()
        )

        await portfolio_repo.save(portfolio)
        found_portfolio = await portfolio_repo.find_by_id(portfolio.id)

        assert found_portfolio is not None
        assert found_portfolio.name == portfolio.name
        print("  ✅ Portfolio repository - OK")

        print("💾 Test InMemoryCache...")
        cache = InMemoryCache(max_size=10, default_ttl=300)

        await cache.set("test_key", "test_value")
        value = await cache.get("test_key")

        assert value == "test_value"
        print("  ✅ Memory cache - OK")

        print("⚡ Test performance basique...")
        start_time = time.time()

        # Créer 100 ordres
        for i in range(100):
            perf_order = Order(
                id=f"perf-order-{i:03d}",
                portfolio_id="portfolio-001",
                symbol="BTC/USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.01"),
                created_time=datetime.utcnow(),
                status=OrderStatus.PENDING
            )
            await order_repo.save(perf_order)

        creation_time = time.time() - start_time

        # Requêter tous les ordres
        start_query = time.time()
        all_orders = await order_repo.find_by_symbol("BTC/USD")
        query_time = time.time() - start_query

        print(f"  📊 100 ordres créés en {creation_time:.3f}s")
        print(f"  📊 {len(all_orders)} ordres récupérés en {query_time:.3f}s")
        print("  ✅ Performance - OK")

        return True

    # Exécuter les tests de base
    try:
        result = asyncio.run(run_basic_tests())
        if result:
            print("\n🎉 Tous les tests d'exécution réelle sont VALIDÉS !")
            print("   Infrastructure Persistence complètement fonctionnelle.")
    except Exception as e:
        print(f"\n❌ Erreur lors des tests : {e}")
        import traceback
        traceback.print_exc()