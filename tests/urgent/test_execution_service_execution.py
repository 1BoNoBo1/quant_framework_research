"""
Tests d'Exécution Réelle - Execution Service
===========================================

Tests qui EXÉCUTENT vraiment le code qframe.domain.services.execution_service
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List

from qframe.domain.services.execution_service import (
    ExecutionService, VenueQuote, ExecutionPlan, ExecutionReport,
    RoutingStrategy, ExecutionAlgorithm
)
from qframe.domain.entities.order import (
    Order, OrderStatus, OrderSide, OrderType, TimeInForce, OrderPriority,
    OrderExecution, create_market_order, create_limit_order
)


class TestVenueQuoteExecution:
    """Tests d'exécution réelle pour VenueQuote."""

    def test_venue_quote_creation_execution(self):
        """Test création VenueQuote avec TOUS les calculs."""
        # Exécuter création avec données réelles
        quote = VenueQuote(
            venue="binance",
            symbol="BTC/USD",
            bid_price=Decimal("49800"),
            ask_price=Decimal("50000"),
            bid_size=Decimal("2.5"),
            ask_size=Decimal("1.8"),
            timestamp=datetime.utcnow()
        )

        # Vérifier calcul automatique du spread
        assert quote.spread == Decimal("200")  # 50000 - 49800

        # Exécuter méthodes métier
        buy_price = quote.get_price_for_side(OrderSide.BUY)
        sell_price = quote.get_price_for_side(OrderSide.SELL)
        buy_size = quote.get_size_for_side(OrderSide.BUY)
        sell_size = quote.get_size_for_side(OrderSide.SELL)

        # Vérifier logique métier
        assert buy_price == quote.ask_price  # Achat au ask
        assert sell_price == quote.bid_price  # Vente au bid
        assert buy_size == quote.ask_size    # Taille disponible à l'ask
        assert sell_size == quote.bid_size   # Taille disponible au bid

    def test_venue_quote_spread_manual_execution(self):
        """Test avec spread défini manuellement."""
        quote = VenueQuote(
            venue="coinbase",
            symbol="ETH/USD",
            bid_price=Decimal("2980"),
            ask_price=Decimal("3000"),
            bid_size=Decimal("5.0"),
            ask_size=Decimal("3.2"),
            timestamp=datetime.utcnow(),
            spread=Decimal("25")  # Spread manuel différent du calculé
        )

        # Vérifier que le spread manuel est conservé
        assert quote.spread == Decimal("25")  # Pas 20 (3000-2980)


class TestExecutionPlanExecution:
    """Tests d'exécution réelle pour ExecutionPlan."""

    def test_execution_plan_creation_execution(self):
        """Test création plan d'exécution complet."""
        # Instructions de slicing réalistes
        slice_instructions = [
            {
                "venue": "binance",
                "quantity": 1.0,
                "timing": "immediate",
                "order_type": "market"
            },
            {
                "venue": "coinbase",
                "quantity": 0.5,
                "timing": "delay_5_minutes",
                "order_type": "limit"
            }
        ]

        # Exécuter création complète
        plan = ExecutionPlan(
            order_id="order-001",
            target_venues=["binance", "coinbase", "kraken"],
            routing_strategy=RoutingStrategy.SMART_ORDER_ROUTING,
            execution_algorithm=ExecutionAlgorithm.TWAP,
            estimated_cost=Decimal("75.50"),
            estimated_duration=timedelta(minutes=30),
            slice_instructions=slice_instructions,
            risk_checks_passed=True,
            created_time=datetime.utcnow()
        )

        # Exécuter sérialisation
        plan_dict = plan.to_dict()

        # Vérifier structure sérialisée
        assert isinstance(plan_dict, dict)
        assert plan_dict["order_id"] == "order-001"
        assert plan_dict["routing_strategy"] == "smart_order_routing"
        assert plan_dict["execution_algorithm"] == "twap"
        assert plan_dict["estimated_cost"] == 75.50
        assert plan_dict["estimated_duration_seconds"] == 1800.0  # 30 minutes
        assert plan_dict["risk_checks_passed"] is True
        assert len(plan_dict["slice_instructions"]) == 2


class TestExecutionReportExecution:
    """Tests d'exécution réelle pour ExecutionReport."""

    def test_execution_report_creation_execution(self):
        """Test création rapport d'exécution."""
        # Exécuter création avec métriques réalistes
        report = ExecutionReport(
            order_id="order-001",
            total_executed_quantity=Decimal("1.5"),
            average_execution_price=Decimal("49950"),
            total_commission=Decimal("12.50"),
            total_fees=Decimal("3.75"),
            execution_time_seconds=125.5,
            venues_used=["binance", "coinbase"],
            slippage=Decimal("0.002"),  # 0.2% slippage
            implementation_shortfall=Decimal("0.0015"),  # 0.15%
            execution_quality="good"
        )

        # Exécuter sérialisation
        report_dict = report.to_dict()

        # Vérifier calculs et sérialisation
        assert isinstance(report_dict, dict)
        assert report_dict["total_executed_quantity"] == 1.5
        assert report_dict["average_execution_price"] == 49950.0
        assert report_dict["execution_time_seconds"] == 125.5
        assert report_dict["venues_used"] == ["binance", "coinbase"]
        assert report_dict["slippage"] == 0.002
        assert report_dict["execution_quality"] == "good"


class TestExecutionServiceExecution:
    """Tests d'exécution réelle pour ExecutionService."""

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
    def large_order(self):
        """Ordre important pour tests d'algorithmes."""
        return create_limit_order(
            portfolio_id="portfolio-001",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=Decimal("10.0"),
            price=Decimal("50000")
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
            ),
            "kraken": VenueQuote(
                venue="kraken",
                symbol="BTC/USD",
                bid_price=Decimal("49875"),
                ask_price=Decimal("49975"),
                bid_size=Decimal("4.0"),
                ask_size=Decimal("3.5"),
                timestamp=datetime.utcnow()
            )
        }

    def test_execution_service_initialization_execution(self, execution_service):
        """Test initialisation du service."""
        # Vérifier configuration des venues
        assert isinstance(execution_service.supported_venues, dict)
        assert "binance" in execution_service.supported_venues
        assert "coinbase" in execution_service.supported_venues
        assert "kraken" in execution_service.supported_venues

        # Vérifier structure de configuration
        binance_config = execution_service.supported_venues["binance"]
        assert "commission_rate" in binance_config
        assert "min_size" in binance_config
        assert isinstance(binance_config["commission_rate"], Decimal)
        assert isinstance(binance_config["min_size"], Decimal)

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
        assert plan.execution_algorithm == ExecutionAlgorithm.IMMEDIATE
        assert plan.risk_checks_passed is True

        # Vérifier sélection du meilleur venue
        # Pour un achat, kraken a le meilleur ask (49975)
        assert "kraken" in plan.target_venues

        # Vérifier calculs des coûts
        assert plan.estimated_cost > 0
        assert isinstance(plan.estimated_duration, timedelta)

    def test_create_execution_plan_lowest_cost_execution(self, execution_service, sample_order, market_data):
        """Test création plan avec stratégie LOWEST_COST."""
        # Exécuter création du plan
        plan = execution_service.create_execution_plan(
            order=sample_order,
            market_data=market_data,
            routing_strategy=RoutingStrategy.LOWEST_COST,
            execution_algorithm=ExecutionAlgorithm.IMMEDIATE
        )

        # Vérifier stratégie de coût
        assert plan.routing_strategy == RoutingStrategy.LOWEST_COST
        assert len(plan.target_venues) <= 3  # Maximum 3 venues
        assert plan.estimated_cost > 0

        # Vérifier que les venues sont triés par coût total (prix + commission)
        for venue in plan.target_venues:
            assert venue in execution_service.supported_venues

    def test_create_execution_plan_smart_routing_execution(self, execution_service, sample_order, market_data):
        """Test création plan avec SMART_ORDER_ROUTING."""
        # Exécuter création du plan
        plan = execution_service.create_execution_plan(
            order=sample_order,
            market_data=market_data,
            routing_strategy=RoutingStrategy.SMART_ORDER_ROUTING,
            execution_algorithm=ExecutionAlgorithm.IMMEDIATE
        )

        # Vérifier routing intelligent
        assert plan.routing_strategy == RoutingStrategy.SMART_ORDER_ROUTING
        assert len(plan.target_venues) <= 3
        assert plan.estimated_cost > 0

        # Le smart routing devrait considérer prix, taille et commission
        for venue in plan.target_venues:
            assert venue in market_data
            assert venue in execution_service.supported_venues

    def test_venue_selection_best_price_execution(self, execution_service, sample_order, market_data):
        """Test sélection venues avec stratégie BEST_PRICE."""
        # Exécuter sélection
        venues = execution_service._select_venues(
            order=sample_order,
            market_data=market_data,
            strategy=RoutingStrategy.BEST_PRICE
        )

        # Vérifier sélection logique
        assert isinstance(venues, list)
        assert len(venues) > 0

        # Pour un ordre d'achat, le premier venue devrait avoir le meilleur ask
        if venues:
            best_venue = venues[0]
            best_ask = market_data[best_venue].ask_price

            for venue_name, quote in market_data.items():
                if venue_name in execution_service.supported_venues:
                    assert best_ask <= quote.ask_price

    def test_venue_selection_minimize_impact_execution(self, execution_service, large_order, market_data):
        """Test sélection venues pour minimiser l'impact."""
        # Exécuter sélection pour gros ordre
        venues = execution_service._select_venues(
            order=large_order,
            market_data=market_data,
            strategy=RoutingStrategy.MINIMIZE_IMPACT
        )

        # Pour minimiser l'impact, devrait distribuer sur plusieurs venues
        assert len(venues) >= 2  # Au moins 2 venues
        assert len(venues) <= 3  # Maximum 3 venues

    def test_execution_cost_estimation_execution(self, execution_service, sample_order, market_data):
        """Test estimation coût d'exécution."""
        target_venues = ["binance", "kraken"]

        # Exécuter estimation
        estimated_cost = execution_service._estimate_execution_cost(
            order=sample_order,
            market_data=market_data,
            target_venues=target_venues
        )

        # Vérifier calcul du coût
        assert isinstance(estimated_cost, Decimal)
        assert estimated_cost > 0

        # Le coût devrait inclure prix + commissions
        # Pour 1 BTC à ~50000 + commissions, le coût devrait être raisonnable
        assert estimated_cost >= Decimal("49000")  # Au moins le prix de base
        assert estimated_cost <= Decimal("55000")  # Pas trop élevé avec commissions

    def test_execution_duration_estimation_execution(self, execution_service, sample_order):
        """Test estimation durée d'exécution."""
        # Test algorithmes différents
        algorithms = [
            ExecutionAlgorithm.IMMEDIATE,
            ExecutionAlgorithm.TWAP,
            ExecutionAlgorithm.VWAP,
            ExecutionAlgorithm.ICEBERG,
            ExecutionAlgorithm.PARTICIPATE
        ]

        for algorithm in algorithms:
            # Exécuter estimation
            duration = execution_service._estimate_execution_duration(sample_order, algorithm)

            # Vérifier résultat
            assert isinstance(duration, timedelta)
            assert duration.total_seconds() > 0

            # Vérifier logique selon l'algorithme
            if algorithm == ExecutionAlgorithm.IMMEDIATE:
                assert duration.total_seconds() <= 60  # Très rapide
            elif algorithm == ExecutionAlgorithm.TWAP:
                assert duration.total_seconds() >= 1800  # Au moins 30 minutes
            elif algorithm == ExecutionAlgorithm.VWAP:
                assert duration.total_seconds() >= 3600  # Au moins 1 heure

    def test_slice_instructions_immediate_execution(self, execution_service, sample_order):
        """Test création instructions pour exécution immédiate."""
        target_venues = ["binance", "coinbase"]

        # Exécuter création d'instructions
        instructions = execution_service._create_slice_instructions(
            order=sample_order,
            target_venues=target_venues,
            algorithm=ExecutionAlgorithm.IMMEDIATE
        )

        # Vérifier instructions immédiates
        assert isinstance(instructions, list)
        assert len(instructions) == 1  # Une seule instruction

        instruction = instructions[0]
        assert instruction["venue"] == "binance"  # Premier venue
        assert instruction["quantity"] == float(sample_order.quantity)
        assert instruction["timing"] == "immediate"
        assert instruction["order_type"] == "market"

    def test_slice_instructions_twap_execution(self, execution_service, large_order):
        """Test création instructions pour TWAP."""
        target_venues = ["binance", "coinbase", "kraken"]

        # Exécuter création d'instructions TWAP
        instructions = execution_service._create_slice_instructions(
            order=large_order,
            target_venues=target_venues,
            algorithm=ExecutionAlgorithm.TWAP
        )

        # Vérifier instructions TWAP
        assert isinstance(instructions, list)
        assert len(instructions) == 6  # 6 tranches par défaut

        # Vérifier distribution temporelle
        total_quantity = sum(instruction["quantity"] for instruction in instructions)
        expected_quantity = float(large_order.quantity)
        assert abs(total_quantity - expected_quantity) < 0.001

        # Vérifier timing graduel
        for i, instruction in enumerate(instructions):
            assert f"delay_{i * 5}_minutes" in instruction["timing"]
            assert instruction["order_type"] == "limit"

    def test_slice_instructions_iceberg_execution(self, execution_service, large_order):
        """Test création instructions pour ICEBERG."""
        target_venues = ["binance"]

        # Exécuter création d'instructions iceberg
        instructions = execution_service._create_slice_instructions(
            order=large_order,
            target_venues=target_venues,
            algorithm=ExecutionAlgorithm.ICEBERG
        )

        # Vérifier instructions iceberg
        assert isinstance(instructions, list)
        assert len(instructions) == 5  # 5 tranches iceberg

        # Vérifier caractéristiques iceberg
        for instruction in instructions:
            assert instruction["timing"] == "after_previous_fill"
            assert instruction["order_type"] == "limit"
            assert instruction["hidden"] is True

        # Vérifier quantités égales
        quantities = [instruction["quantity"] for instruction in instructions]
        assert len(set(quantities)) == 1  # Toutes identiques

    def test_slice_instructions_vwap_execution(self, execution_service, large_order):
        """Test création instructions pour VWAP."""
        target_venues = ["binance", "coinbase"]

        # Exécuter création d'instructions VWAP
        instructions = execution_service._create_slice_instructions(
            order=large_order,
            target_venues=target_venues,
            algorithm=ExecutionAlgorithm.VWAP
        )

        # Vérifier instructions VWAP
        assert isinstance(instructions, list)
        assert len(instructions) == 4  # 4 fenêtres de volume

        # Vérifier profil de volume
        total_quantity = sum(instruction["quantity"] for instruction in instructions)
        expected_quantity = float(large_order.quantity)
        assert abs(total_quantity - expected_quantity) < 0.001

        # Vérifier timing basé sur volume
        for i, instruction in enumerate(instructions):
            assert f"volume_window_{i}" in instruction["timing"]

    def test_pre_execution_risk_checks_execution(self, execution_service):
        """Test vérifications de risque pré-exécution."""
        # Ordre valide
        valid_order = create_market_order(
            portfolio_id="portfolio-001",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0")
        )

        # Exécuter vérifications
        assert execution_service._perform_pre_execution_risk_checks(valid_order) is True

        # Ordre avec quantité négative
        invalid_order_qty = create_market_order(
            portfolio_id="portfolio-001",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=Decimal("-1.0")
        )

        assert execution_service._perform_pre_execution_risk_checks(invalid_order_qty) is False

        # Ordre avec statut invalide
        rejected_order = create_market_order(
            portfolio_id="portfolio-001",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0")
        )
        rejected_order.status = OrderStatus.REJECTED

        assert execution_service._perform_pre_execution_risk_checks(rejected_order) is False

        # Ordre trop gros (limite de 1M)
        huge_order = create_market_order(
            portfolio_id="portfolio-001",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=Decimal("25.0")  # 25 BTC * 50k = 1.25M > limite
        )
        huge_order.notional_value = Decimal("1250000")

        assert execution_service._perform_pre_execution_risk_checks(huge_order) is False

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

        execution = executions[0]
        assert isinstance(execution, OrderExecution)
        assert execution.executed_quantity > 0
        assert execution.execution_price > 0
        assert execution.commission >= 0
        assert execution.venue in execution_service.supported_venues

        # Vérifier que l'ordre a été mis à jour
        assert len(sample_order.executions) > 0
        assert sample_order.executed_quantity > 0

    def test_execute_order_failed_risk_checks_execution(self, execution_service, market_data):
        """Test exécution avec vérifications de risque échouées."""
        # Ordre avec problème
        bad_order = create_market_order(
            portfolio_id="portfolio-001",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=Decimal("0")  # Quantité nulle
        )

        # Créer plan (qui échouera aux vérifications)
        plan = execution_service.create_execution_plan(
            order=bad_order,
            market_data=market_data
        )

        # L'exécution devrait échouer
        with pytest.raises(ValueError, match="Risk checks failed"):
            execution_service.execute_order(bad_order, plan, market_data)

    def test_execute_immediate_detailed_execution(self, execution_service, sample_order, market_data):
        """Test détaillé d'exécution immédiate."""
        # Plan pour exécution immédiate sur kraken (meilleur prix)
        plan = ExecutionPlan(
            order_id=sample_order.id,
            target_venues=["kraken"],
            routing_strategy=RoutingStrategy.BEST_PRICE,
            execution_algorithm=ExecutionAlgorithm.IMMEDIATE,
            estimated_cost=Decimal("50000"),
            estimated_duration=timedelta(seconds=5),
            slice_instructions=[],
            risk_checks_passed=True,
            created_time=datetime.utcnow()
        )

        # Exécuter
        executions = execution_service._execute_immediate(sample_order, plan, market_data)

        # Vérifier exécution détaillée
        assert len(executions) == 1
        execution = executions[0]

        # Vérifier calculs corrects
        kraken_quote = market_data["kraken"]
        expected_price = kraken_quote.ask_price  # Achat au ask
        expected_quantity = min(sample_order.quantity, kraken_quote.ask_size)

        assert execution.execution_price == expected_price
        assert execution.executed_quantity == expected_quantity
        assert execution.venue == "kraken"
        assert execution.liquidity_flag == "taker"

        # Vérifier calcul commission
        execution_value = execution.executed_quantity * execution.execution_price
        kraken_commission_rate = execution_service.supported_venues["kraken"]["commission_rate"]
        expected_commission = execution_value * kraken_commission_rate

        assert execution.commission == expected_commission

    def test_execution_service_integration_execution(self, execution_service, large_order, market_data):
        """Test d'intégration complète du service d'exécution."""
        # Workflow complet: Plan → Exécution → Rapport

        # 1. Créer plan d'exécution sophistiqué
        plan = execution_service.create_execution_plan(
            order=large_order,
            market_data=market_data,
            routing_strategy=RoutingStrategy.SMART_ORDER_ROUTING,
            execution_algorithm=ExecutionAlgorithm.TWAP
        )

        # 2. Vérifier plan détaillé
        assert isinstance(plan, ExecutionPlan)
        assert plan.routing_strategy == RoutingStrategy.SMART_ORDER_ROUTING
        assert plan.execution_algorithm == ExecutionAlgorithm.TWAP
        assert len(plan.target_venues) > 1  # Multi-venue
        assert len(plan.slice_instructions) > 1  # Multi-slice
        assert plan.estimated_cost > 0

        # 3. Exécuter (simulation TWAP immédiate pour test)
        # Pour le test, on simule une exécution immédiate
        immediate_plan = ExecutionPlan(
            order_id=large_order.id,
            target_venues=plan.target_venues,
            routing_strategy=plan.routing_strategy,
            execution_algorithm=ExecutionAlgorithm.IMMEDIATE,  # Changé pour test
            estimated_cost=plan.estimated_cost,
            estimated_duration=timedelta(seconds=5),
            slice_instructions=[],
            risk_checks_passed=True,
            created_time=datetime.utcnow()
        )

        executions = execution_service.execute_order(large_order, immediate_plan, market_data)

        # 4. Vérifier exécutions
        assert len(executions) > 0
        total_executed = sum(ex.executed_quantity for ex in executions)
        assert total_executed > 0

        # 5. Analyser qualité d'exécution
        total_cost = sum(ex.executed_quantity * ex.execution_price + ex.commission for ex in executions)
        avg_price = total_cost / total_executed if total_executed > 0 else Decimal("0")

        # Vérifier cohérence des prix
        assert avg_price > Decimal("49000")  # Prix raisonnable
        assert avg_price < Decimal("51000")  # Pas trop élevé

        # 6. Créer rapport synthétique (simulé)
        venues_used = list(set(ex.venue for ex in executions))
        total_commission = sum(ex.commission for ex in executions)

        report = ExecutionReport(
            order_id=large_order.id,
            total_executed_quantity=total_executed,
            average_execution_price=avg_price,
            total_commission=total_commission,
            total_fees=Decimal("0"),
            execution_time_seconds=5.0,
            venues_used=venues_used,
            slippage=Decimal("0.001"),  # 0.1% slippage estimé
            implementation_shortfall=Decimal("0.0005"),
            execution_quality="good"
        )

        # 7. Vérifier rapport final
        assert isinstance(report, ExecutionReport)
        assert report.total_executed_quantity > 0
        assert report.average_execution_price > 0
        assert len(report.venues_used) > 0

        # 8. Test sérialisation complète
        plan_dict = plan.to_dict()
        report_dict = report.to_dict()

        assert isinstance(plan_dict, dict)
        assert isinstance(report_dict, dict)
        assert "routing_strategy" in plan_dict
        assert "execution_quality" in report_dict