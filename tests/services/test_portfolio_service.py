"""
Tests for Portfolio Service
===========================

Suite de tests pour le service de gestion de portefeuille.
"""

import pytest
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any

from qframe.domain.entities.portfolio import Portfolio, PortfolioStatus
from qframe.domain.entities.position import Position
from qframe.domain.services.portfolio_service import (
    PortfolioService,
    RebalancingPlan,
    AllocationOptimization,
    PortfolioPerformanceAnalysis
)


class TestPortfolioService:
    """Tests pour le service Portfolio"""

    @pytest.fixture
    def portfolio_service(self):
        """Service initialisé pour les tests"""
        return PortfolioService(risk_free_rate=Decimal("0.02"))

    @pytest.fixture
    def sample_portfolio(self):
        """Portfolio de test avec positions"""
        portfolio = Portfolio(
            id="port-001",
            name="Test Portfolio",
            initial_capital=Decimal("100000"),
            base_currency="USD"
        )

        # Ajouter des positions
        portfolio.positions["BTCUSD"] = Position(
            symbol="BTCUSD",
            quantity=Decimal("1.0"),
            average_price=Decimal("50000"),
            current_price=Decimal("55000")
        )

        portfolio.positions["ETHUSD"] = Position(
            symbol="ETHUSD",
            quantity=Decimal("10.0"),
            average_price=Decimal("3000"),
            current_price=Decimal("3200")
        )

        # Définir allocations cibles
        portfolio.target_allocations = {
            "BTCUSD": Decimal("0.6"),
            "ETHUSD": Decimal("0.4")
        }

        return portfolio

    def test_create_rebalancing_plan(self, portfolio_service, sample_portfolio):
        """Test création d'un plan de rééquilibrage"""
        plan = portfolio_service.create_rebalancing_plan(
            portfolio=sample_portfolio,
            target_allocations={"BTCUSD": Decimal("0.5"), "ETHUSD": Decimal("0.5")},
            rebalancing_threshold=Decimal("0.05")
        )

        if plan:  # Le plan peut être None si pas de rééquilibrage nécessaire
            assert isinstance(plan, RebalancingPlan)
            assert plan.portfolio_id == sample_portfolio.id
            assert isinstance(plan.target_allocations, dict)
            assert isinstance(plan.trades_required, dict)

    def test_equal_weight_optimization(self, portfolio_service, sample_portfolio):
        """Test optimisation d'allocation équipondérée"""
        result = portfolio_service.optimize_allocation_equal_weight(
            portfolio=sample_portfolio,
            symbols=["BTCUSD", "ETHUSD", "ADAUSD"]
        )

        assert isinstance(result, AllocationOptimization)
        assert len(result.optimized_allocations) == 3

        # Vérifier que toutes les allocations sont égales
        allocations = list(result.optimized_allocations.values())
        expected_allocation = Decimal("1") / Decimal("3")
        for allocation in allocations:
            assert abs(allocation - expected_allocation) < Decimal("0.001")

    def test_risk_parity_optimization(self, portfolio_service, sample_portfolio):
        """Test optimisation risk parity"""
        # Mock des données de volatilité
        volatilities = {
            "BTCUSD": Decimal("0.8"),  # Haute volatilité
            "ETHUSD": Decimal("0.6"),  # Volatilité moyenne
            "ADAUSD": Decimal("0.4")   # Volatilité plus faible
        }

        result = portfolio_service.optimize_allocation_risk_parity(
            portfolio=sample_portfolio,
            symbols=list(volatilities.keys()),
            volatilities=volatilities
        )

        assert isinstance(result, AllocationOptimization)
        assert len(result.optimized_allocations) == 3

        # Les actifs moins volatils devraient avoir plus d'allocation
        ada_allocation = result.optimized_allocations["ADAUSD"]
        btc_allocation = result.optimized_allocations["BTCUSD"]
        assert ada_allocation > btc_allocation

    def test_momentum_optimization(self, portfolio_service, sample_portfolio):
        """Test optimisation momentum"""
        # Mock des scores de momentum
        momentum_scores = {
            "BTCUSD": Decimal("0.8"),   # Momentum fort
            "ETHUSD": Decimal("0.3"),   # Momentum faible
            "ADAUSD": Decimal("0.6")    # Momentum moyen
        }

        result = portfolio_service.optimize_allocation_momentum(
            portfolio=sample_portfolio,
            momentum_scores=momentum_scores,
            lookback_window=30
        )

        assert isinstance(result, AllocationOptimization)

        # L'actif avec le plus fort momentum devrait avoir plus d'allocation
        btc_allocation = result.optimized_allocations.get("BTCUSD", Decimal("0"))
        eth_allocation = result.optimized_allocations.get("ETHUSD", Decimal("0"))
        assert btc_allocation > eth_allocation

    def test_portfolio_performance_analysis(self, portfolio_service, sample_portfolio):
        """Test analyse de performance"""
        # Mock historique des valeurs
        value_history = [
            (datetime.now(), Decimal("100000")),
            (datetime.now(), Decimal("105000")),
            (datetime.now(), Decimal("110000")),
            (datetime.now(), Decimal("108000")),
            (datetime.now(), Decimal("115000"))
        ]

        analysis = portfolio_service.analyze_portfolio_performance(
            portfolio=sample_portfolio,
            value_history=value_history,
            benchmark_returns=[Decimal("0.02"), Decimal("0.01"), Decimal("-0.01"), Decimal("0.03")]
        )

        assert isinstance(analysis, PortfolioPerformanceAnalysis)
        assert analysis.portfolio_id == sample_portfolio.id
        assert isinstance(analysis.total_return, Decimal)
        assert isinstance(analysis.volatility, Decimal)
        assert isinstance(analysis.sharpe_ratio, Decimal)

    def test_portfolio_comparison(self, portfolio_service, sample_portfolio):
        """Test comparaison de portfolios"""
        # Créer un second portfolio pour comparaison
        portfolio2 = Portfolio(
            id="port-002",
            name="Test Portfolio 2",
            initial_capital=Decimal("100000"),
            base_currency="USD"
        )

        portfolio2.positions["BTCUSD"] = Position(
            symbol="BTCUSD",
            quantity=Decimal("0.5"),
            average_price=Decimal("52000"),
            current_price=Decimal("55000")
        )

        comparison = portfolio_service.compare_portfolios(
            portfolio1=sample_portfolio,
            portfolio2=portfolio2,
            comparison_metrics=["total_return", "sharpe_ratio", "max_drawdown"]
        )

        assert isinstance(comparison, dict)
        assert "portfolio1" in comparison
        assert "portfolio2" in comparison
        assert "comparison_summary" in comparison

    def test_should_rebalance_by_frequency(self, portfolio_service, sample_portfolio):
        """Test vérification de rééquilibrage par fréquence"""
        from qframe.domain.entities.portfolio import RebalancingFrequency

        # Test fréquence mensuelle
        should_rebalance = portfolio_service.should_rebalance_by_frequency(
            portfolio=sample_portfolio,
            frequency=RebalancingFrequency.MONTHLY
        )

        assert isinstance(should_rebalance, bool)

    def test_execute_rebalancing_plan(self, portfolio_service, sample_portfolio):
        """Test exécution d'un plan de rééquilibrage"""
        # Créer un plan de rééquilibrage
        plan = RebalancingPlan(
            portfolio_id=sample_portfolio.id,
            timestamp=datetime.now(),
            target_allocations={"BTCUSD": Decimal("0.5"), "ETHUSD": Decimal("0.5")},
            current_allocations={"BTCUSD": Decimal("0.6"), "ETHUSD": Decimal("0.4")},
            trades_required={"BTCUSD": Decimal("-10000"), "ETHUSD": Decimal("10000")},
            estimated_cost=Decimal("100"),
            reason="Target allocation drift"
        )

        # Mock des prix actuels
        current_prices = {
            "BTCUSD": Decimal("55000"),
            "ETHUSD": Decimal("3200")
        }

        executed_portfolio = portfolio_service.execute_rebalancing_plan(
            portfolio=sample_portfolio,
            plan=plan,
            current_prices=current_prices
        )

        assert isinstance(executed_portfolio, Portfolio)
        assert executed_portfolio.id == sample_portfolio.id

    def test_risk_metrics_calculation(self, portfolio_service, sample_portfolio):
        """Test calcul des métriques de risque"""
        risk_metrics = portfolio_service.calculate_risk_metrics(sample_portfolio)

        if risk_metrics:  # Peut être None si données insuffisantes
            assert isinstance(risk_metrics, dict)
            # Vérifier présence de métriques de base
            expected_metrics = ["concentration_risk", "correlation_risk", "total_exposure"]
            for metric in expected_metrics:
                if metric in risk_metrics:
                    assert isinstance(risk_metrics[metric], (Decimal, float, int))

    def test_concentration_risk_calculation(self, portfolio_service, sample_portfolio):
        """Test calcul du risque de concentration"""
        concentration_risk = portfolio_service.calculate_concentration_risk(sample_portfolio)

        assert isinstance(concentration_risk, Decimal)
        assert concentration_risk >= 0
        assert concentration_risk <= 1  # Concentration entre 0 et 1

    def test_correlation_risk_estimation(self, portfolio_service, sample_portfolio):
        """Test estimation du risque de corrélation"""
        correlation_risk = portfolio_service.estimate_correlation_risk(sample_portfolio)

        assert isinstance(correlation_risk, Decimal)
        assert correlation_risk >= 0

    def test_rebalancing_plan_calculations(self, portfolio_service, sample_portfolio):
        """Test calculs de plan de rééquilibrage"""
        # Test avec allocations cibles différentes
        target_allocations = {"BTCUSD": Decimal("0.7"), "ETHUSD": Decimal("0.3")}
        current_prices = {"BTCUSD": Decimal("55000"), "ETHUSD": Decimal("3200")}

        plan = portfolio_service.calculate_rebalancing_plan(
            portfolio=sample_portfolio,
            target_allocations=target_allocations,
            current_prices=current_prices
        )

        if plan:
            assert isinstance(plan, RebalancingPlan)
            assert plan.portfolio_id == sample_portfolio.id

            # Vérifier que la somme des allocations cibles = 1
            total_allocation = sum(plan.target_allocations.values())
            assert abs(total_allocation - Decimal("1")) < Decimal("0.001")

    def test_empty_portfolio_handling(self, portfolio_service):
        """Test gestion d'un portfolio vide"""
        empty_portfolio = Portfolio(
            id="empty-port",
            name="Empty Portfolio",
            initial_capital=Decimal("10000"),
            base_currency="USD"
        )

        # Ne devrait pas lever d'exception
        concentration_risk = portfolio_service.calculate_concentration_risk(empty_portfolio)
        assert concentration_risk == Decimal("0")

        risk_metrics = portfolio_service.calculate_risk_metrics(empty_portfolio)
        # Peut être None ou dict vide
        assert risk_metrics is None or isinstance(risk_metrics, dict)

    def test_single_position_portfolio(self, portfolio_service):
        """Test portfolio avec une seule position"""
        single_position_portfolio = Portfolio(
            id="single-pos",
            name="Single Position Portfolio",
            initial_capital=Decimal("100000"),
            base_currency="USD"
        )

        single_position_portfolio.positions["BTCUSD"] = Position(
            symbol="BTCUSD",
            quantity=Decimal("1.0"),
            average_price=Decimal("50000"),
            current_price=Decimal("55000")
        )

        # Concentration devrait être maximale (1.0)
        concentration_risk = portfolio_service.calculate_concentration_risk(single_position_portfolio)
        assert concentration_risk == Decimal("1.0")

    def test_allocation_optimization_edge_cases(self, portfolio_service, sample_portfolio):
        """Test cas limites d'optimisation d'allocation"""
        # Test avec liste vide
        result = portfolio_service.optimize_allocation_equal_weight(
            portfolio=sample_portfolio,
            symbols=[]
        )

        assert isinstance(result, AllocationOptimization)
        assert len(result.optimized_allocations) == 0

        # Test avec un seul symbole
        result = portfolio_service.optimize_allocation_equal_weight(
            portfolio=sample_portfolio,
            symbols=["BTCUSD"]
        )

        assert result.optimized_allocations["BTCUSD"] == Decimal("1.0")