"""
Tests d'Exécution Réelle CORRIGÉS - Services et Repositories
=============================================================

Tests qui EXÉCUTENT vraiment le code avec les BONNES signatures
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional

# Services
from qframe.domain.services.portfolio_service import (
    PortfolioService, RebalancingPlan, AllocationOptimization,
    PortfolioPerformanceAnalysis
)
from qframe.domain.services.execution_service import (
    ExecutionService, VenueQuote, ExecutionPlan, ExecutionReport,
    RoutingStrategy, ExecutionAlgorithm
)

# Entities avec signatures CORRECTES
from qframe.domain.entities.portfolio import Portfolio, PortfolioSnapshot, PortfolioConstraints, RebalancingFrequency
from qframe.domain.entities.position import Position  # Position(symbol, quantity, average_price, current_price=None)
from qframe.domain.entities.order import Order, OrderSide, OrderType, create_market_order, create_limit_order

# Repositories
from qframe.infrastructure.persistence.memory_order_repository import MemoryOrderRepository
from qframe.infrastructure.persistence.memory_portfolio_repository import MemoryPortfolioRepository


class TestPortfolioServiceExecutionFixed:
    """Tests d'exécution réelle CORRIGÉS pour PortfolioService."""

    @pytest.fixture
    def portfolio_service(self):
        """Service avec taux sans risque configuré."""
        return PortfolioService(risk_free_rate=Decimal("0.03"))

    @pytest.fixture
    def sample_portfolio(self):
        """Portfolio de test avec positions CORRECTES."""
        # Positions avec signature CORRECTE : Position(symbol, quantity, average_price, current_price=None)
        positions = {
            "BTC/USD": Position(
                symbol="BTC/USD",
                quantity=Decimal("0.5"),
                average_price=Decimal("45000"),
                current_price=Decimal("50000")  # Optionnel
            ),
            "ETH/USD": Position(
                symbol="ETH/USD",
                quantity=Decimal("10"),
                average_price=Decimal("2800"),
                current_price=Decimal("3000")
            ),
            "CASH": Position(
                symbol="CASH",
                quantity=Decimal("15000"),
                average_price=Decimal("1"),
                current_price=Decimal("1")
            )
        }

        # Contraintes réelles
        constraints = PortfolioConstraints(
            max_position_size=Decimal("0.4"),
            max_leverage=Decimal("1.0"),
            allowed_symbols=["BTC/USD", "ETH/USD", "CASH"],
            rebalancing_frequency=RebalancingFrequency.WEEKLY
        )

        # Allocations cibles
        target_allocations = {
            "BTC/USD": Decimal("0.35"),
            "ETH/USD": Decimal("0.40"),
            "CASH": Decimal("0.25")
        }

        portfolio = Portfolio(
            id="test-portfolio-001",
            name="Test Portfolio",
            initial_capital=Decimal("60000"),
            base_currency="USD",
            positions=positions,
            target_allocations=target_allocations,
            constraints=constraints
        )

        # Calculer la valeur totale via les positions
        portfolio.total_value = sum(pos.market_value for pos in positions.values())

        return portfolio

    @pytest.fixture
    def portfolio_with_snapshots(self, sample_portfolio):
        """Portfolio avec historique de snapshots CORRECTS."""
        portfolio = sample_portfolio

        # Créer historique avec signature CORRECTE de PortfolioSnapshot
        base_date = datetime.utcnow() - timedelta(days=30)
        snapshots = []

        base_value = Decimal("65000")
        for i in range(30):
            variation = Decimal(str((i % 7 - 3) * 0.02))
            daily_value = base_value * (Decimal("1") + variation)

            # PortfolioSnapshot(timestamp, total_value, cash, positions_count, largest_position_weight)
            snapshot = PortfolioSnapshot(
                timestamp=base_date + timedelta(days=i),
                total_value=daily_value,
                cash=Decimal("15000"),
                positions_count=3,
                largest_position_weight=Decimal("0.45"),
                daily_pnl=variation * Decimal("1000")  # Optionnel
            )
            snapshots.append(snapshot)
            base_value = daily_value

        portfolio.snapshots = snapshots
        return portfolio

    def test_create_rebalancing_plan_execution(self, portfolio_service, sample_portfolio):
        """Test création plan de rééquilibrage."""
        # Exécuter création du plan
        plan = portfolio_service.create_rebalancing_plan(
            portfolio=sample_portfolio,
            rebalancing_threshold=Decimal("0.05"),
            transaction_cost_rate=Decimal("0.002")
        )

        # Vérifier que le plan a été créé (peut être None si pas nécessaire)
        if plan:
            assert isinstance(plan, RebalancingPlan)
            assert plan.portfolio_id == sample_portfolio.id
            assert plan.estimated_cost > 0
            assert plan.get_trade_value() > 0

    def test_portfolio_performance_analysis_execution(self, portfolio_service, portfolio_with_snapshots):
        """Test analyse complète de performance."""
        # Exécuter analyse de performance
        analysis = portfolio_service.analyze_portfolio_performance(
            portfolio_with_snapshots,
            analysis_period_days=30
        )

        # Vérifier création de l'analyse
        assert isinstance(analysis, PortfolioPerformanceAnalysis)
        assert analysis.portfolio_id == portfolio_with_snapshots.id
        assert analysis.analysis_period_days == 30

        # Vérifier calculs de base
        assert analysis.total_return != Decimal("0")
        assert analysis.volatility >= Decimal("0")
        assert analysis.max_drawdown >= Decimal("0")

        # Vérifier sérialisation
        analysis_dict = analysis.to_dict()
        assert isinstance(analysis_dict, dict)
        assert "total_return" in analysis_dict
        assert isinstance(analysis_dict["total_return"], float)

    def test_optimize_allocation_equal_weight_execution(self, portfolio_service):
        """Test optimisation allocation équi-pondérée."""
        symbols = ["BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD"]

        # Exécuter optimisation
        optimization = portfolio_service.optimize_allocation_equal_weight(symbols)

        # Vérifier résultat
        assert isinstance(optimization, AllocationOptimization)
        assert optimization.optimization_method == "equal_weight"

        # Vérifier allocation équitable
        expected_weight = Decimal("1") / len(symbols)
        for symbol in symbols:
            assert optimization.optimized_allocations[symbol] == expected_weight

        # Vérifier somme = 1
        total_weight = sum(optimization.optimized_allocations.values())
        assert abs(total_weight - Decimal("1")) < Decimal("0.0001")

    def test_calculate_risk_metrics_execution(self, portfolio_service, portfolio_with_snapshots):
        """Test calcul métriques de risque."""
        # Exécuter calcul des métriques
        risk_metrics = portfolio_service.calculate_risk_metrics(portfolio_with_snapshots)

        # Vérifier résultat
        if risk_metrics:  # Peut être None si pas assez de données
            assert isinstance(risk_metrics, dict)
            assert "volatility" in risk_metrics
            assert "sharpe_ratio" in risk_metrics
            assert risk_metrics["volatility"] >= 0

    def test_daily_returns_calculation_execution(self, portfolio_service, portfolio_with_snapshots):
        """Test calcul des rendements journaliers."""
        # Exécuter calcul des rendements
        returns = portfolio_service._calculate_daily_returns(portfolio_with_snapshots.snapshots)

        # Vérifier résultat
        assert isinstance(returns, list)
        assert len(returns) == len(portfolio_with_snapshots.snapshots) - 1

        # Vérifier types des rendements
        for ret in returns:
            assert isinstance(ret, Decimal)


class TestExecutionServiceExecutionFixed:
    """Tests d'exécution réelle CORRIGÉS pour ExecutionService."""

    @pytest.fixture
    def execution_service(self):
        """Service d'exécution configuré."""
        return ExecutionService()

    @pytest.fixture
    def sample_order(self):
        """Ordre de test."""
        return create_market_order(
            portfolio_id="portfolio-001",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0")
        )

    @pytest.fixture
    def market_data(self):
        """Données de marché multi-venues."""
        return {
            "binance": VenueQuote(
                venue="binance",
                symbol="BTC/USD",
                bid_price=Decimal("49900"),
                ask_price=Decimal("50000"),
                bid_size=Decimal("3.0"),
                ask_size=Decimal("2.5"),
                timestamp=datetime.utcnow()
            ),
            "coinbase": VenueQuote(
                venue="coinbase",
                symbol="BTC/USD",
                bid_price=Decimal("49950"),
                ask_price=Decimal("50050"),
                bid_size=Decimal("1.8"),
                ask_size=Decimal("1.2"),
                timestamp=datetime.utcnow()
            )
        }

    def test_create_execution_plan_best_price_execution(self, execution_service, sample_order, market_data):
        """Test création plan avec stratégie BEST_PRICE."""
        # Exécuter création du plan
        plan = execution_service.create_execution_plan(
            order=sample_order,
            market_data=market_data,
            routing_strategy=RoutingStrategy.BEST_PRICE,
            execution_algorithm=ExecutionAlgorithm.IMMEDIATE
        )

        # Vérifier création du plan
        assert isinstance(plan, ExecutionPlan)
        assert plan.order_id == sample_order.id
        assert plan.routing_strategy == RoutingStrategy.BEST_PRICE
        assert plan.risk_checks_passed is True
        assert plan.estimated_cost > 0

    def test_execute_order_immediate_execution(self, execution_service, sample_order, market_data):
        """Test exécution immédiate d'un ordre."""
        # Créer plan d'exécution
        plan = execution_service.create_execution_plan(
            order=sample_order,
            market_data=market_data,
            routing_strategy=RoutingStrategy.BEST_PRICE,
            execution_algorithm=ExecutionAlgorithm.IMMEDIATE
        )

        # Exécuter l'ordre
        executions = execution_service.execute_order(sample_order, plan, market_data)

        # Vérifier exécutions
        assert isinstance(executions, list)
        assert len(executions) > 0

        # Vérifier que l'ordre a été mis à jour
        assert len(sample_order.executions) > 0
        assert sample_order.executed_quantity > 0


class TestMemoryOrderRepositoryExecutionFixed:
    """Tests d'exécution réelle CORRIGÉS pour MemoryOrderRepository."""

    @pytest.fixture
    def order_repository(self):
        """Repository d'ordres en mémoire."""
        return MemoryOrderRepository()

    @pytest.fixture
    def sample_orders(self):
        """Ordres de test."""
        return [
            create_market_order(
                portfolio_id="portfolio-001",
                symbol="BTC/USD",
                side=OrderSide.BUY,
                quantity=Decimal("1.0")
            ),
            create_limit_order(
                portfolio_id="portfolio-001",
                symbol="ETH/USD",
                side=OrderSide.SELL,
                quantity=Decimal("5.0"),
                price=Decimal("3200")
            )
        ]

    @pytest.mark.asyncio
    async def test_save_and_find_by_id_execution(self, order_repository, sample_orders):
        """Test sauvegarde et récupération par ID."""
        order = sample_orders[0]

        # Exécuter sauvegarde
        saved_order = await order_repository.save(order)

        # Vérifier sauvegarde
        assert saved_order is not None
        assert saved_order.id == order.id

        # Exécuter récupération par ID
        found_order = await order_repository.find_by_id(order.id)

        # Vérifier récupération
        assert found_order is not None
        assert found_order.symbol == order.symbol
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
        assert len(portfolio_orders) == 2

        for order in portfolio_orders:
            assert order.portfolio_id == "portfolio-001"

    @pytest.mark.asyncio
    async def test_count_orders_execution(self, order_repository, sample_orders):
        """Test comptage d'ordres."""
        # Compter ordres vides
        initial_count = await order_repository.count()
        assert initial_count == 0

        # Sauvegarder ordres
        for order in sample_orders:
            await order_repository.save(order)

        # Exécuter comptage
        final_count = await order_repository.count()
        assert final_count == len(sample_orders)


class TestMemoryPortfolioRepositoryExecutionFixed:
    """Tests d'exécution réelle CORRIGÉS pour MemoryPortfolioRepository."""

    @pytest.fixture
    def portfolio_repository(self):
        """Repository de portfolios en mémoire."""
        return MemoryPortfolioRepository()

    @pytest.fixture
    def sample_portfolios(self):
        """Portfolios de test avec positions CORRECTES."""
        portfolios = []

        # Portfolio crypto avec Position CORRECTE
        crypto_positions = {
            "BTC/USD": Position(
                symbol="BTC/USD",
                quantity=Decimal("2.0"),
                average_price=Decimal("45000"),
                current_price=Decimal("50000")
            ),
            "ETH/USD": Position(
                symbol="ETH/USD",
                quantity=Decimal("10"),
                average_price=Decimal("2800"),
                current_price=Decimal("3000")
            )
        }

        crypto_portfolio = Portfolio(
            id="crypto-portfolio",
            name="Crypto Investment Portfolio",
            initial_capital=Decimal("120000"),
            base_currency="USD",
            positions=crypto_positions
        )

        # Calculer total_value via market_value des positions
        crypto_portfolio.total_value = sum(pos.market_value for pos in crypto_positions.values())
        portfolios.append(crypto_portfolio)

        # Portfolio conservateur
        conservative_positions = {
            "CASH": Position(
                symbol="CASH",
                quantity=Decimal("50000"),
                average_price=Decimal("1"),
                current_price=Decimal("1")
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
    async def test_save_and_find_portfolio_execution(self, portfolio_repository, sample_portfolios):
        """Test sauvegarde et récupération de portfolio."""
        portfolio = sample_portfolios[0]

        # Exécuter sauvegarde
        saved_portfolio = await portfolio_repository.save(portfolio)

        # Vérifier sauvegarde
        assert saved_portfolio is not None
        assert saved_portfolio.id == portfolio.id
        assert saved_portfolio.name == portfolio.name

        # Exécuter récupération par ID
        found_portfolio = await portfolio_repository.find_by_id(portfolio.id)

        # Vérifier récupération
        assert found_portfolio is not None
        assert found_portfolio.initial_capital == portfolio.initial_capital
        assert len(found_portfolio.positions) == len(portfolio.positions)

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

    @pytest.mark.asyncio
    async def test_portfolio_with_snapshots_execution(self, portfolio_repository):
        """Test portfolio avec historique de snapshots CORRECTS."""
        portfolio = Portfolio(
            id="snapshot-portfolio",
            name="Portfolio with History",
            initial_capital=Decimal("100000"),
            base_currency="USD"
        )

        # Ajouter snapshots avec signature CORRECTE
        base_date = datetime.utcnow() - timedelta(days=5)
        for i in range(5):
            snapshot = PortfolioSnapshot(
                timestamp=base_date + timedelta(days=i),
                total_value=Decimal("100000") + Decimal(str(i * 1000)),
                cash=Decimal("10000"),
                positions_count=1,
                largest_position_weight=Decimal("0.90")
            )
            portfolio.snapshots.append(snapshot)

        # Sauvegarder portfolio avec snapshots
        saved_portfolio = await portfolio_repository.save(portfolio)

        # Vérifier sauvegarde des snapshots
        assert len(saved_portfolio.snapshots) == 5

        # Récupérer et vérifier persistance
        found_portfolio = await portfolio_repository.find_by_id(portfolio.id)
        assert len(found_portfolio.snapshots) == 5