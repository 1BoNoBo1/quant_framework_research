"""
Tests for Memory Repositories
=============================

Suite de tests pour les repositories memory critiques.
Teste les implémentations in-memory des repositories.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional
from unittest.mock import Mock, patch

from qframe.infrastructure.persistence.memory_order_repository import MemoryOrderRepository
from qframe.infrastructure.persistence.memory_portfolio_repository import MemoryPortfolioRepository
from qframe.infrastructure.persistence.memory_strategy_repository import MemoryStrategyRepository
from qframe.infrastructure.persistence.memory_backtest_repository import MemoryBacktestRepository
from qframe.infrastructure.persistence.memory_risk_assessment_repository import MemoryRiskAssessmentRepository

from qframe.domain.entities.order import Order, OrderSide, OrderType, OrderStatus
from qframe.domain.entities.portfolio import Portfolio, Position
from qframe.domain.entities.strategy import Strategy, StrategyStatus, StrategyType
from qframe.domain.entities.backtest import BacktestResult, BacktestStatus, BacktestMetrics
from qframe.domain.entities.risk_assessment import RiskAssessment, RiskLevel


@pytest.fixture
def sample_order():
    """Ordre d'exemple."""
    return Order(
        id="order-001",
        portfolio_id="portfolio-001",
        symbol="BTC/USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.1"),
        price=Decimal("45000.00"),
        status=OrderStatus.PENDING,
        created_time=datetime.utcnow()
    )


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
def sample_strategy():
    """Stratégie d'exemple."""
    return Strategy(
        id="strategy-001",
        name="Test Strategy",
        strategy_type=StrategyType.MEAN_REVERSION,
        status=StrategyStatus.ACTIVE,
        description="Test strategy",
        parameters={"param1": "value1"},
        created_at=datetime.utcnow()
    )


@pytest.fixture
def sample_backtest_result():
    """Résultat de backtest d'exemple."""
    return BacktestResult(
        id="backtest-001",
        configuration_id="config-001",
        name="Test Backtest",
        status=BacktestStatus.COMPLETED,
        initial_capital=Decimal("100000.00"),
        start_time=datetime.utcnow() - timedelta(hours=1),
        end_time=datetime.utcnow()
    )


@pytest.fixture
def sample_risk_assessment():
    """Évaluation de risque d'exemple."""
    return RiskAssessment(
        id="risk-001",
        portfolio_id="portfolio-001",
        timestamp=datetime.utcnow(),
        risk_level=RiskLevel.MEDIUM,
        var_95=Decimal("5000.00"),
        cvar_95=Decimal("7500.00"),
        max_drawdown=Decimal("0.15"),
        volatility=Decimal("0.20")
    )


class TestMemoryOrderRepository:
    """Tests pour MemoryOrderRepository."""

    @pytest.fixture
    def order_repository(self):
        return MemoryOrderRepository()

    async def test_save_order(self, order_repository, sample_order):
        """Test de sauvegarde d'ordre."""
        # Act
        await order_repository.save(sample_order)

        # Assert
        saved_order = await order_repository.find_by_id(sample_order.id)
        assert saved_order is not None
        assert saved_order.id == sample_order.id
        assert saved_order.symbol == sample_order.symbol

    async def test_find_by_id_not_found(self, order_repository):
        """Test de recherche par ID inexistant."""
        # Act
        result = await order_repository.find_by_id("non-existent")

        # Assert
        assert result is None

    async def test_find_by_portfolio_id(self, order_repository, sample_order):
        """Test de recherche par portfolio ID."""
        # Arrange
        await order_repository.save(sample_order)

        # Act
        orders = await order_repository.find_by_portfolio_id("portfolio-001")

        # Assert
        assert len(orders) == 1
        assert orders[0].id == sample_order.id

    async def test_find_by_status(self, order_repository, sample_order):
        """Test de recherche par statut."""
        # Arrange
        await order_repository.save(sample_order)

        # Act
        orders = await order_repository.find_by_status(OrderStatus.PENDING)

        # Assert
        assert len(orders) == 1
        assert orders[0].status == OrderStatus.PENDING

    async def test_find_by_symbol(self, order_repository, sample_order):
        """Test de recherche par symbole."""
        # Arrange
        await order_repository.save(sample_order)

        # Act
        orders = await order_repository.find_by_symbol("BTC/USD")

        # Assert
        assert len(orders) == 1
        assert orders[0].symbol == "BTC/USD"

    async def test_update_order(self, order_repository, sample_order):
        """Test de mise à jour d'ordre."""
        # Arrange
        await order_repository.save(sample_order)
        sample_order.status = OrderStatus.FILLED

        # Act
        await order_repository.update(sample_order)

        # Assert
        updated_order = await order_repository.find_by_id(sample_order.id)
        assert updated_order.status == OrderStatus.FILLED

    async def test_delete_order(self, order_repository, sample_order):
        """Test de suppression d'ordre."""
        # Arrange
        await order_repository.save(sample_order)

        # Act
        await order_repository.delete(sample_order.id)

        # Assert
        deleted_order = await order_repository.find_by_id(sample_order.id)
        assert deleted_order is None

    async def test_list_all_orders(self, order_repository, sample_order):
        """Test de listage de tous les ordres."""
        # Arrange
        await order_repository.save(sample_order)

        # Act
        orders = await order_repository.list_all()

        # Assert
        assert len(orders) == 1
        assert orders[0].id == sample_order.id

    async def test_get_active_orders(self, order_repository):
        """Test de récupération des ordres actifs."""
        # Arrange
        pending_order = Order(
            id="order-pending",
            portfolio_id="portfolio-001",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            status=OrderStatus.PENDING
        )
        filled_order = Order(
            id="order-filled",
            portfolio_id="portfolio-001",
            symbol="ETH/USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            status=OrderStatus.FILLED
        )

        await order_repository.save(pending_order)
        await order_repository.save(filled_order)

        # Act
        active_orders = await order_repository.get_active_orders("portfolio-001")

        # Assert
        assert len(active_orders) == 1
        assert active_orders[0].status == OrderStatus.PENDING


class TestMemoryPortfolioRepository:
    """Tests pour MemoryPortfolioRepository."""

    @pytest.fixture
    def portfolio_repository(self):
        return MemoryPortfolioRepository()

    async def test_save_portfolio(self, portfolio_repository, sample_portfolio):
        """Test de sauvegarde de portfolio."""
        # Act
        await portfolio_repository.save(sample_portfolio)

        # Assert
        saved_portfolio = await portfolio_repository.find_by_id(sample_portfolio.id)
        assert saved_portfolio is not None
        assert saved_portfolio.name == sample_portfolio.name

    async def test_find_by_name(self, portfolio_repository, sample_portfolio):
        """Test de recherche par nom."""
        # Arrange
        await portfolio_repository.save(sample_portfolio)

        # Act
        portfolio = await portfolio_repository.find_by_name("Test Portfolio")

        # Assert
        assert portfolio is not None
        assert portfolio.id == sample_portfolio.id

    async def test_get_portfolio_value(self, portfolio_repository, sample_portfolio):
        """Test de calcul de la valeur du portfolio."""
        # Arrange
        await portfolio_repository.save(sample_portfolio)

        # Act
        value = await portfolio_repository.get_portfolio_value(sample_portfolio.id)

        # Assert
        assert value == sample_portfolio.initial_capital

    async def test_get_positions(self, portfolio_repository, sample_portfolio):
        """Test de récupération des positions."""
        # Arrange
        position = Position(
            symbol="BTC/USD",
            quantity=Decimal("0.5"),
            average_price=Decimal("45000.00"),
            current_price=Decimal("46000.00")
        )
        sample_portfolio.positions.append(position)
        await portfolio_repository.save(sample_portfolio)

        # Act
        positions = await portfolio_repository.get_positions(sample_portfolio.id)

        # Assert
        assert len(positions) == 1
        assert positions[0].symbol == "BTC/USD"

    async def test_update_position(self, portfolio_repository, sample_portfolio):
        """Test de mise à jour de position."""
        # Arrange
        position = Position(
            symbol="BTC/USD",
            quantity=Decimal("0.5"),
            average_price=Decimal("45000.00"),
            current_price=Decimal("46000.00")
        )
        sample_portfolio.positions.append(position)
        await portfolio_repository.save(sample_portfolio)

        # Act
        await portfolio_repository.update_position(
            sample_portfolio.id,
            "BTC/USD",
            Decimal("1.0"),
            Decimal("47000.00")
        )

        # Assert
        updated_portfolio = await portfolio_repository.find_by_id(sample_portfolio.id)
        updated_position = next(p for p in updated_portfolio.positions if p.symbol == "BTC/USD")
        assert updated_position.quantity == Decimal("1.0")
        assert updated_position.current_price == Decimal("47000.00")


class TestMemoryStrategyRepository:
    """Tests pour MemoryStrategyRepository."""

    @pytest.fixture
    def strategy_repository(self):
        return MemoryStrategyRepository()

    async def test_save_strategy(self, strategy_repository, sample_strategy):
        """Test de sauvegarde de stratégie."""
        # Act
        await strategy_repository.save(sample_strategy)

        # Assert
        saved_strategy = await strategy_repository.find_by_id(sample_strategy.id)
        assert saved_strategy is not None
        assert saved_strategy.name == sample_strategy.name

    async def test_find_by_name(self, strategy_repository, sample_strategy):
        """Test de recherche par nom."""
        # Arrange
        await strategy_repository.save(sample_strategy)

        # Act
        strategy = await strategy_repository.find_by_name("Test Strategy")

        # Assert
        assert strategy is not None
        assert strategy.id == sample_strategy.id

    async def test_find_by_type(self, strategy_repository, sample_strategy):
        """Test de recherche par type."""
        # Arrange
        await strategy_repository.save(sample_strategy)

        # Act
        strategies = await strategy_repository.find_by_type(StrategyType.MEAN_REVERSION)

        # Assert
        assert len(strategies) == 1
        assert strategies[0].strategy_type == StrategyType.MEAN_REVERSION

    async def test_find_active_strategies(self, strategy_repository, sample_strategy):
        """Test de recherche des stratégies actives."""
        # Arrange
        await strategy_repository.save(sample_strategy)

        # Act
        active_strategies = await strategy_repository.find_active_strategies()

        # Assert
        assert len(active_strategies) == 1
        assert active_strategies[0].status == StrategyStatus.ACTIVE

    async def test_update_strategy_status(self, strategy_repository, sample_strategy):
        """Test de mise à jour du statut."""
        # Arrange
        await strategy_repository.save(sample_strategy)

        # Act
        await strategy_repository.update_status(sample_strategy.id, StrategyStatus.PAUSED)

        # Assert
        updated_strategy = await strategy_repository.find_by_id(sample_strategy.id)
        assert updated_strategy.status == StrategyStatus.PAUSED


class TestMemoryBacktestRepository:
    """Tests pour MemoryBacktestRepository."""

    @pytest.fixture
    def backtest_repository(self):
        return MemoryBacktestRepository()

    async def test_save_backtest_result(self, backtest_repository, sample_backtest_result):
        """Test de sauvegarde de résultat de backtest."""
        # Act
        await backtest_repository.save_result(sample_backtest_result)

        # Assert
        saved_result = await backtest_repository.find_result_by_id(sample_backtest_result.id)
        assert saved_result is not None
        assert saved_result.name == sample_backtest_result.name

    async def test_find_by_configuration_id(self, backtest_repository, sample_backtest_result):
        """Test de recherche par configuration ID."""
        # Arrange
        await backtest_repository.save_result(sample_backtest_result)

        # Act
        results = await backtest_repository.find_by_configuration_id("config-001")

        # Assert
        assert len(results) == 1
        assert results[0].configuration_id == "config-001"

    async def test_find_by_status(self, backtest_repository, sample_backtest_result):
        """Test de recherche par statut."""
        # Arrange
        await backtest_repository.save_result(sample_backtest_result)

        # Act
        results = await backtest_repository.find_by_status(BacktestStatus.COMPLETED)

        # Assert
        assert len(results) == 1
        assert results[0].status == BacktestStatus.COMPLETED

    async def test_find_recent_results(self, backtest_repository, sample_backtest_result):
        """Test de recherche des résultats récents."""
        # Arrange
        await backtest_repository.save_result(sample_backtest_result)

        # Act
        recent_results = await backtest_repository.find_recent_results(limit=10)

        # Assert
        assert len(recent_results) == 1
        assert recent_results[0].id == sample_backtest_result.id

    async def test_get_performance_summary(self, backtest_repository, sample_backtest_result):
        """Test de récupération du résumé de performance."""
        # Arrange
        sample_backtest_result.metrics = BacktestMetrics(
            total_return=Decimal("0.15"),
            sharpe_ratio=Decimal("1.5"),
            max_drawdown=Decimal("-0.05")
        )
        await backtest_repository.save_result(sample_backtest_result)

        # Act
        summary = await backtest_repository.get_performance_summary(sample_backtest_result.id)

        # Assert
        assert summary is not None
        assert "total_return" in summary
        assert summary["total_return"] == Decimal("0.15")


class TestMemoryRiskAssessmentRepository:
    """Tests pour MemoryRiskAssessmentRepository."""

    @pytest.fixture
    def risk_repository(self):
        return MemoryRiskAssessmentRepository()

    async def test_save_risk_assessment(self, risk_repository, sample_risk_assessment):
        """Test de sauvegarde d'évaluation de risque."""
        # Act
        await risk_repository.save(sample_risk_assessment)

        # Assert
        saved_assessment = await risk_repository.find_by_id(sample_risk_assessment.id)
        assert saved_assessment is not None
        assert saved_assessment.portfolio_id == sample_risk_assessment.portfolio_id

    async def test_find_by_portfolio_id(self, risk_repository, sample_risk_assessment):
        """Test de recherche par portfolio ID."""
        # Arrange
        await risk_repository.save(sample_risk_assessment)

        # Act
        assessments = await risk_repository.find_by_portfolio_id("portfolio-001")

        # Assert
        assert len(assessments) == 1
        assert assessments[0].portfolio_id == "portfolio-001"

    async def test_find_by_risk_level(self, risk_repository, sample_risk_assessment):
        """Test de recherche par niveau de risque."""
        # Arrange
        await risk_repository.save(sample_risk_assessment)

        # Act
        assessments = await risk_repository.find_by_risk_level(RiskLevel.MEDIUM)

        # Assert
        assert len(assessments) == 1
        assert assessments[0].risk_level == RiskLevel.MEDIUM

    async def test_find_recent_assessments(self, risk_repository, sample_risk_assessment):
        """Test de recherche des évaluations récentes."""
        # Arrange
        await risk_repository.save(sample_risk_assessment)

        # Act
        recent_assessments = await risk_repository.find_recent_assessments(
            "portfolio-001", hours=24
        )

        # Assert
        assert len(recent_assessments) == 1
        assert recent_assessments[0].id == sample_risk_assessment.id

    async def test_get_latest_assessment(self, risk_repository, sample_risk_assessment):
        """Test de récupération de la dernière évaluation."""
        # Arrange
        await risk_repository.save(sample_risk_assessment)

        # Act
        latest = await risk_repository.get_latest_assessment("portfolio-001")

        # Assert
        assert latest is not None
        assert latest.id == sample_risk_assessment.id


class TestRepositoryPerformance:
    """Tests de performance des repositories."""

    async def test_concurrent_operations(self):
        """Test d'opérations concurrentes."""
        # Arrange
        order_repo = MemoryOrderRepository()
        tasks = []

        # Act - Créer plusieurs ordres en parallèle
        for i in range(100):
            order = Order(
                id=f"order-{i}",
                portfolio_id="portfolio-001",
                symbol="BTC/USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.1")
            )
            tasks.append(order_repo.save(order))

        await asyncio.gather(*tasks)

        # Assert
        all_orders = await order_repo.list_all()
        assert len(all_orders) == 100

    async def test_large_dataset_handling(self):
        """Test de gestion de gros volumes de données."""
        # Arrange
        portfolio_repo = MemoryPortfolioRepository()

        # Act - Créer un portfolio avec beaucoup de positions
        portfolio = Portfolio(
            id="large-portfolio",
            name="Large Portfolio",
            initial_capital=Decimal("1000000.00")
        )

        for i in range(1000):
            position = Position(
                symbol=f"SYMBOL{i}",
                quantity=Decimal("1.0"),
                average_price=Decimal("100.00"),
                current_price=Decimal("101.00")
            )
            portfolio.positions.append(position)

        await portfolio_repo.save(portfolio)

        # Assert
        saved_portfolio = await portfolio_repo.find_by_id("large-portfolio")
        assert len(saved_portfolio.positions) == 1000

    async def test_memory_efficiency(self):
        """Test d'efficacité mémoire."""
        # Arrange
        strategy_repo = MemoryStrategyRepository()

        # Act - Créer et supprimer beaucoup de stratégies
        for i in range(1000):
            strategy = Strategy(
                id=f"strategy-{i}",
                name=f"Strategy {i}",
                strategy_type=StrategyType.MEAN_REVERSION,
                status=StrategyStatus.ACTIVE
            )
            await strategy_repo.save(strategy)

        # Supprimer la moitié
        for i in range(0, 1000, 2):
            await strategy_repo.delete(f"strategy-{i}")

        # Assert
        remaining_strategies = await strategy_repo.list_all()
        assert len(remaining_strategies) == 500


class TestRepositoryErrorHandling:
    """Tests de gestion d'erreur des repositories."""

    async def test_duplicate_id_handling(self):
        """Test de gestion des IDs dupliqués."""
        # Arrange
        order_repo = MemoryOrderRepository()
        order1 = Order(
            id="duplicate-id",
            portfolio_id="portfolio-001",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1")
        )
        order2 = Order(
            id="duplicate-id",
            portfolio_id="portfolio-002",
            symbol="ETH/USD",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0")
        )

        # Act
        await order_repo.save(order1)
        await order_repo.save(order2)  # Devrait écraser le premier

        # Assert
        saved_order = await order_repo.find_by_id("duplicate-id")
        assert saved_order.portfolio_id == "portfolio-002"

    async def test_invalid_id_operations(self):
        """Test d'opérations avec IDs invalides."""
        # Arrange
        portfolio_repo = MemoryPortfolioRepository()

        # Act & Assert
        result = await portfolio_repo.find_by_id("")
        assert result is None

        result = await portfolio_repo.find_by_id(None)
        assert result is None

    async def test_empty_search_results(self):
        """Test de résultats de recherche vides."""
        # Arrange
        backtest_repo = MemoryBacktestRepository()

        # Act
        results = await backtest_repo.find_by_status(BacktestStatus.FAILED)

        # Assert
        assert results == []
        assert isinstance(results, list)