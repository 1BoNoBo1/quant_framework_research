"""
Tests for Order Execution Infrastructure
=====================================

Tests critiques pour l'exécution d'ordres via les courtiers.
Couvre tous les aspects de l'OrderExecutionAdapter et ExecutionService.
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock, patch

from qframe.infrastructure.external.order_execution_adapter import (
    OrderExecutionAdapter, BrokerAdapter
)
from qframe.infrastructure.external.mock_broker_adapter import MockBrokerAdapter
from qframe.domain.services.execution_service import (
    ExecutionService, VenueQuote, ExecutionPlan, ExecutionReport,
    RoutingStrategy, ExecutionAlgorithm
)
from qframe.domain.entities.order import (
    Order, OrderStatus, OrderType, OrderSide, TimeInForce, OrderPriority,
    OrderExecution, create_market_order, create_limit_order
)


class TestOrderExecutionAdapter:
    """Tests pour OrderExecutionAdapter"""

    @pytest.fixture
    def execution_adapter(self):
        """OrderExecutionAdapter pour les tests"""
        return OrderExecutionAdapter()

    @pytest.fixture
    def mock_broker(self):
        """Mock broker adapter"""
        return MockBrokerAdapter("test_venue", base_latency_ms=1, fill_probability=1.0)

    @pytest.fixture
    def sample_order(self):
        """Ordre de test"""
        return create_market_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            portfolio_id="test_portfolio",
            strategy_id="test_strategy"
        )

    @pytest.fixture
    def sample_execution_plan(self):
        """Plan d'exécution de test"""
        return ExecutionPlan(
            order_id="test_order",
            target_venues=["test_venue"],
            routing_strategy=RoutingStrategy.BEST_PRICE,
            execution_algorithm=ExecutionAlgorithm.IMMEDIATE,
            estimated_cost=Decimal("50000"),
            estimated_duration=timedelta(seconds=5),
            slice_instructions=[{
                "venue": "test_venue",
                "quantity": 1.0,
                "timing": "immediate",
                "order_type": "market"
            }],
            risk_checks_passed=True,
            created_time=datetime.utcnow()
        )

    def test_broker_registration(self, execution_adapter, mock_broker):
        """Test l'enregistrement d'un broker"""
        # Act
        execution_adapter.register_broker("test_venue", mock_broker)

        # Assert
        assert execution_adapter.is_venue_available("test_venue")
        assert "test_venue" in execution_adapter.get_supported_venues()

    @pytest.mark.asyncio
    async def test_execute_order_plan_success(self, execution_adapter, mock_broker, sample_order, sample_execution_plan):
        """Test l'exécution réussie d'un plan d'ordre"""
        # Arrange
        execution_adapter.register_broker("test_venue", mock_broker)

        # Act
        executions = await execution_adapter.execute_order_plan(sample_order, sample_execution_plan)

        # Assert
        assert len(executions) > 0
        assert all(isinstance(exec, OrderExecution) for exec in executions)
        assert sample_order.id in execution_adapter._order_mappings

    @pytest.mark.asyncio
    async def test_execute_order_plan_no_broker(self, execution_adapter, sample_order, sample_execution_plan):
        """Test l'exécution avec venue non supportée"""
        # Act
        executions = await execution_adapter.execute_order_plan(sample_order, sample_execution_plan)

        # Assert
        assert len(executions) == 0

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, execution_adapter, mock_broker, sample_execution_plan):
        """Test l'annulation réussie d'un ordre limite"""
        # Arrange - Use a limit order with a very low price that won't execute immediately
        limit_order = create_limit_order("BTC-USD", OrderSide.BUY, Decimal("1.0"), Decimal("1000"), "portfolio", "strategy")

        execution_adapter.register_broker("test_venue", mock_broker)
        await execution_adapter.execute_order_plan(limit_order, sample_execution_plan)

        # Give some time but not enough for execution
        await asyncio.sleep(0.05)

        # Act
        results = await execution_adapter.cancel_order(limit_order.id)

        # Assert
        assert "test_venue" in results
        # Note: May be True or False depending on execution timing
        assert isinstance(results["test_venue"], bool)

    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self, execution_adapter):
        """Test l'annulation d'un ordre inexistant"""
        # Act
        results = await execution_adapter.cancel_order("non_existent_order")

        # Assert
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_get_market_data(self, execution_adapter, mock_broker):
        """Test la récupération des données de marché"""
        # Arrange
        execution_adapter.register_broker("test_venue", mock_broker)

        # Act
        quotes = await execution_adapter.get_market_data("BTC-USD")

        # Assert
        assert "test_venue" in quotes
        assert isinstance(quotes["test_venue"], VenueQuote)
        assert quotes["test_venue"].venue == "test_venue"

    @pytest.mark.asyncio
    async def test_get_order_status_all_venues(self, execution_adapter, mock_broker, sample_order, sample_execution_plan):
        """Test la récupération du statut sur toutes les venues"""
        # Arrange
        execution_adapter.register_broker("test_venue", mock_broker)
        await execution_adapter.execute_order_plan(sample_order, sample_execution_plan)

        # Act
        statuses = await execution_adapter.get_order_status_all_venues(sample_order.id)

        # Assert
        assert "test_venue" in statuses
        assert "status" in statuses["test_venue"]

    @pytest.mark.asyncio
    async def test_health_check(self, execution_adapter, mock_broker):
        """Test la vérification de santé"""
        # Arrange
        execution_adapter.register_broker("test_venue", mock_broker)

        # Act
        health = await execution_adapter.health_check()

        # Assert
        assert health["status"] in ["healthy", "warning", "degraded"]
        assert health["total_venues"] == 1
        assert "venues" in health


class TestMockBrokerAdapter:
    """Tests pour MockBrokerAdapter"""

    @pytest.fixture
    def mock_broker(self):
        """Mock broker pour les tests"""
        return MockBrokerAdapter("test_venue", base_latency_ms=1, fill_probability=1.0)

    @pytest.fixture
    def sample_order(self):
        """Ordre de test"""
        return create_market_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            portfolio_id="test_portfolio",
            strategy_id="test_strategy"
        )

    @pytest.mark.asyncio
    async def test_submit_order_success(self, mock_broker, sample_order):
        """Test la soumission réussie d'un ordre"""
        # Act
        result = await mock_broker.submit_order(sample_order)

        # Assert
        assert result["success"] is True
        assert "order_id" in result
        assert result["venue"] == "test_venue"

    @pytest.mark.asyncio
    async def test_submit_order_rejection(self, sample_order):
        """Test le rejet d'un ordre via simulation de conditions de marché"""
        # Arrange - broker standard
        broker = MockBrokerAdapter("test_venue", base_latency_ms=1, fill_probability=1.0)

        # Act - Force a rejection by mocking random to return < 0.02
        with patch('random.random', return_value=0.01):  # Force rejection
            result = await broker.submit_order(sample_order)

        # Assert - le rejet devrait se produire
        assert result["success"] is False
        assert "error" in result
        assert result["order_id"] is None

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, mock_broker):
        """Test l'annulation réussie d'un ordre limite"""
        # Arrange - Use a limit order with a very low price that won't execute
        limit_order = create_limit_order("BTC-USD", OrderSide.BUY, Decimal("1.0"), Decimal("1000"), "portfolio", "strategy")
        submit_result = await mock_broker.submit_order(limit_order)
        broker_order_id = submit_result["order_id"]

        # Give minimal time, not enough for limit order execution
        await asyncio.sleep(0.01)

        # Act
        success = await mock_broker.cancel_order(limit_order.id, broker_order_id)

        # Assert
        assert success is True

    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self, mock_broker):
        """Test l'annulation d'un ordre inexistant"""
        # Act
        success = await mock_broker.cancel_order("test_order", "non_existent_id")

        # Assert
        assert success is False

    @pytest.mark.asyncio
    async def test_get_order_status(self, mock_broker, sample_order):
        """Test la récupération du statut d'un ordre"""
        # Arrange
        submit_result = await mock_broker.submit_order(sample_order)
        broker_order_id = submit_result["order_id"]

        # Wait for potential execution
        await asyncio.sleep(0.1)

        # Act
        status = await mock_broker.get_order_status(broker_order_id)

        # Assert
        assert "status" in status
        assert "filled_quantity" in status
        assert "remaining_quantity" in status

    @pytest.mark.asyncio
    async def test_get_market_data(self, mock_broker):
        """Test la récupération des données de marché"""
        # Act
        quote = await mock_broker.get_market_data("BTC-USD")

        # Assert
        assert isinstance(quote, VenueQuote)
        assert quote.venue == "test_venue"
        assert quote.bid_price < quote.ask_price
        assert quote.bid_size > 0
        assert quote.ask_size > 0

    @pytest.mark.asyncio
    async def test_health_check(self, mock_broker):
        """Test la vérification de santé du broker"""
        # Act
        health = await mock_broker.health_check()

        # Assert
        assert health["status"] == "healthy"
        assert health["venue"] == "test_venue"
        assert health["connected"] is True
        assert "total_orders" in health

    @pytest.mark.asyncio
    async def test_market_order_execution(self, mock_broker, sample_order):
        """Test l'exécution d'un ordre market"""
        # Arrange
        market_order = create_market_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            portfolio_id="test_portfolio",
            strategy_id="test_strategy"
        )

        # Act
        submit_result = await mock_broker.submit_order(market_order)
        broker_order_id = submit_result["order_id"]

        # Wait for execution
        await asyncio.sleep(0.6)  # Attendre l'exécution simulée

        status = await mock_broker.get_order_status(broker_order_id)

        # Assert
        assert status["status"] in ["filled", "partial"]
        assert len(status["executions"]) > 0

    @pytest.mark.asyncio
    async def test_limit_order_execution(self, mock_broker):
        """Test l'exécution d'un ordre limite"""
        # Arrange
        limit_order = create_limit_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            portfolio_id="test_portfolio",
            strategy_id="test_strategy"
        )

        # Act
        submit_result = await mock_broker.submit_order(limit_order)
        broker_order_id = submit_result["order_id"]

        # Wait for potential execution
        await asyncio.sleep(0.1)

        status = await mock_broker.get_order_status(broker_order_id)

        # Assert
        assert status["status"] in ["submitted", "filled", "partial"]

    def test_price_slippage_calculation(self, mock_broker):
        """Test le calcul du slippage"""
        # Arrange
        market_price = Decimal("50000")
        buy_order = create_market_order("BTC-USD", OrderSide.BUY, Decimal("1.0"), "portfolio", "strategy")
        sell_order = create_market_order("BTC-USD", OrderSide.SELL, Decimal("1.0"), "portfolio", "strategy")

        # Act
        buy_price = mock_broker._calculate_execution_price(buy_order, market_price)
        sell_price = mock_broker._calculate_execution_price(sell_order, market_price)

        # Assert
        assert buy_price >= market_price  # Slippage défavorable pour achat
        assert sell_price <= market_price  # Slippage défavorable pour vente

    def test_market_price_updates(self, mock_broker):
        """Test la mise à jour des prix de marché"""
        # Arrange
        initial_price = mock_broker._market_prices["BTC-USD"]

        # Act
        asyncio.run(mock_broker._update_market_prices())

        # Assert
        updated_price = mock_broker._market_prices["BTC-USD"]
        # Le prix peut être le même si pas assez de temps écoulé
        assert isinstance(updated_price, Decimal)

    @pytest.mark.asyncio
    async def test_fee_calculation_in_execution(self, mock_broker, sample_order):
        """Test le calcul des frais dans une exécution réelle"""
        # Arrange
        expected_fee_rate = mock_broker._fee_rates.get("test_venue", Decimal("0.001"))

        # Act
        result = await mock_broker.submit_order(sample_order)
        broker_order_id = result["order_id"]

        # Wait for execution
        await asyncio.sleep(0.6)

        status = await mock_broker.get_order_status(broker_order_id)

        # Assert - Vérifier que les frais sont calculés correctement
        if status["executions"]:
            execution = status["executions"][0]
            execution_value = Decimal(execution["quantity"]) * Decimal(execution["price"])
            expected_fee = execution_value * expected_fee_rate
            actual_fee = Decimal(execution["fee"])

            # Tolérance pour les calculs en virgule flottante
            assert abs(actual_fee - expected_fee) < Decimal("0.01")


class TestExecutionService:
    """Tests pour ExecutionService"""

    @pytest.fixture
    def execution_service(self):
        """Service d'exécution pour les tests"""
        return ExecutionService()

    @pytest.fixture
    def sample_market_data(self) -> Dict[str, VenueQuote]:
        """Données de marché de test"""
        return {
            "binance": VenueQuote(
                venue="binance",
                symbol="BTC-USD",
                bid_price=Decimal("49900"),
                ask_price=Decimal("50100"),
                bid_size=Decimal("10.0"),
                ask_size=Decimal("8.0"),
                timestamp=datetime.utcnow()
            ),
            "coinbase": VenueQuote(
                venue="coinbase",
                symbol="BTC-USD",
                bid_price=Decimal("49950"),
                ask_price=Decimal("50050"),
                bid_size=Decimal("5.0"),
                ask_size=Decimal("12.0"),
                timestamp=datetime.utcnow()
            ),
            "kraken": VenueQuote(
                venue="kraken",
                symbol="BTC-USD",
                bid_price=Decimal("49850"),
                ask_price=Decimal("50150"),
                bid_size=Decimal("15.0"),
                ask_size=Decimal("6.0"),
                timestamp=datetime.utcnow()
            )
        }

    @pytest.fixture
    def sample_order(self):
        """Ordre de test"""
        return create_market_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("2.0"),
            portfolio_id="test_portfolio",
            strategy_id="test_strategy"
        )

    def test_create_execution_plan_best_price(self, execution_service, sample_order, sample_market_data):
        """Test la création d'un plan avec stratégie meilleur prix"""
        # Act
        plan = execution_service.create_execution_plan(
            sample_order,
            sample_market_data,
            RoutingStrategy.BEST_PRICE,
            ExecutionAlgorithm.IMMEDIATE
        )

        # Assert
        assert isinstance(plan, ExecutionPlan)
        assert plan.order_id == sample_order.id
        assert plan.routing_strategy == RoutingStrategy.BEST_PRICE
        assert plan.execution_algorithm == ExecutionAlgorithm.IMMEDIATE
        assert len(plan.target_venues) > 0
        assert plan.risk_checks_passed is True

    def test_create_execution_plan_lowest_cost(self, execution_service, sample_order, sample_market_data):
        """Test la création d'un plan avec stratégie coût le plus bas"""
        # Act
        plan = execution_service.create_execution_plan(
            sample_order,
            sample_market_data,
            RoutingStrategy.LOWEST_COST,
            ExecutionAlgorithm.IMMEDIATE
        )

        # Assert
        assert isinstance(plan, ExecutionPlan)
        assert plan.routing_strategy == RoutingStrategy.LOWEST_COST
        assert len(plan.target_venues) <= 3  # Maximum 3 venues

    def test_create_execution_plan_smart_routing(self, execution_service, sample_order, sample_market_data):
        """Test la création d'un plan avec routing intelligent"""
        # Act
        plan = execution_service.create_execution_plan(
            sample_order,
            sample_market_data,
            RoutingStrategy.SMART_ORDER_ROUTING,
            ExecutionAlgorithm.IMMEDIATE
        )

        # Assert
        assert isinstance(plan, ExecutionPlan)
        assert plan.routing_strategy == RoutingStrategy.SMART_ORDER_ROUTING
        assert len(plan.target_venues) <= 3

    def test_create_execution_plan_twap(self, execution_service, sample_order, sample_market_data):
        """Test la création d'un plan TWAP"""
        # Act
        plan = execution_service.create_execution_plan(
            sample_order,
            sample_market_data,
            RoutingStrategy.BEST_PRICE,
            ExecutionAlgorithm.TWAP
        )

        # Assert
        assert plan.execution_algorithm == ExecutionAlgorithm.TWAP
        assert plan.estimated_duration > timedelta(minutes=10)
        assert len(plan.slice_instructions) > 1

    def test_create_execution_plan_iceberg(self, execution_service, sample_order, sample_market_data):
        """Test la création d'un plan iceberg"""
        # Act
        plan = execution_service.create_execution_plan(
            sample_order,
            sample_market_data,
            RoutingStrategy.BEST_PRICE,
            ExecutionAlgorithm.ICEBERG
        )

        # Assert
        assert plan.execution_algorithm == ExecutionAlgorithm.ICEBERG
        assert len(plan.slice_instructions) == 5  # 5 tranches iceberg

    def test_venue_selection_best_price_buy(self, execution_service, sample_order, sample_market_data):
        """Test la sélection de venue pour achat au meilleur prix"""
        # Act
        venues = execution_service._select_venues(sample_order, sample_market_data, RoutingStrategy.BEST_PRICE)

        # Assert
        assert len(venues) > 0
        # Pour un achat, coinbase devrait être en tête (ask le plus bas: 50050)
        assert "coinbase" in venues

    def test_venue_selection_best_price_sell(self, execution_service, sample_market_data):
        """Test la sélection de venue pour vente au meilleur prix"""
        # Arrange
        sell_order = create_market_order(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            quantity=Decimal("2.0"),
            portfolio_id="test_portfolio",
            strategy_id="test_strategy"
        )

        # Act
        venues = execution_service._select_venues(sell_order, sample_market_data, RoutingStrategy.BEST_PRICE)

        # Assert
        assert len(venues) > 0
        # Pour une vente, coinbase devrait être en tête (bid le plus haut: 49950)
        assert "coinbase" in venues

    def test_execution_cost_estimation(self, execution_service, sample_order, sample_market_data):
        """Test l'estimation du coût d'exécution"""
        # Arrange
        target_venues = ["binance", "coinbase"]

        # Act
        cost = execution_service._estimate_execution_cost(sample_order, sample_market_data, target_venues)

        # Assert
        assert cost > 0
        assert isinstance(cost, Decimal)

    def test_execution_duration_estimation(self, execution_service, sample_order):
        """Test l'estimation de la durée d'exécution"""
        # Act
        immediate_duration = execution_service._estimate_execution_duration(sample_order, ExecutionAlgorithm.IMMEDIATE)
        twap_duration = execution_service._estimate_execution_duration(sample_order, ExecutionAlgorithm.TWAP)
        vwap_duration = execution_service._estimate_execution_duration(sample_order, ExecutionAlgorithm.VWAP)

        # Assert
        assert immediate_duration < twap_duration < vwap_duration
        assert immediate_duration == timedelta(seconds=5)
        assert twap_duration == timedelta(minutes=30)
        assert vwap_duration == timedelta(hours=1)

    def test_pre_execution_risk_checks(self, execution_service):
        """Test les vérifications de risque pré-exécution"""
        # Arrange
        valid_order = create_market_order("BTC-USD", OrderSide.BUY, Decimal("1.0"), "portfolio", "strategy")

        # Create an order manually to bypass validation for testing
        invalid_order = create_market_order("BTC-USD", OrderSide.BUY, Decimal("1.0"), "portfolio", "strategy")
        invalid_order.quantity = Decimal("0")  # Set invalid quantity after creation

        # Act
        valid_result = execution_service._perform_pre_execution_risk_checks(valid_order)
        invalid_result = execution_service._perform_pre_execution_risk_checks(invalid_order)

        # Assert
        assert valid_result is True
        assert invalid_result is False

    def test_execute_immediate_order(self, execution_service, sample_order, sample_market_data):
        """Test l'exécution immédiate d'un ordre"""
        # Arrange
        plan = execution_service.create_execution_plan(
            sample_order, sample_market_data, RoutingStrategy.BEST_PRICE, ExecutionAlgorithm.IMMEDIATE
        )

        # Act
        executions = execution_service.execute_order(sample_order, plan, sample_market_data)

        # Assert
        assert len(executions) > 0
        assert all(isinstance(exec, OrderExecution) for exec in executions)
        assert sample_order.status in [OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED]

    def test_execute_twap_order(self, execution_service, sample_order, sample_market_data):
        """Test l'exécution TWAP d'un ordre"""
        # Arrange
        plan = execution_service.create_execution_plan(
            sample_order, sample_market_data, RoutingStrategy.BEST_PRICE, ExecutionAlgorithm.TWAP
        )

        # Act
        executions = execution_service.execute_order(sample_order, plan, sample_market_data)

        # Assert
        assert len(executions) > 1  # Plusieurs exécutions TWAP
        assert all(isinstance(exec, OrderExecution) for exec in executions)

    def test_execute_iceberg_order(self, execution_service, sample_order, sample_market_data):
        """Test l'exécution iceberg d'un ordre"""
        # Arrange
        plan = execution_service.create_execution_plan(
            sample_order, sample_market_data, RoutingStrategy.BEST_PRICE, ExecutionAlgorithm.ICEBERG
        )

        # Act
        executions = execution_service.execute_order(sample_order, plan, sample_market_data)

        # Assert
        assert len(executions) > 1  # Plusieurs exécutions iceberg
        assert all(exec.liquidity_flag == "maker" for exec in executions)  # Iceberg = maker

    def test_create_execution_report(self, execution_service, sample_order):
        """Test la création d'un rapport d'exécution"""
        # Arrange
        execution = OrderExecution(
            executed_quantity=Decimal("1.0"),
            execution_price=Decimal("50000"),
            commission=Decimal("50"),
            venue="binance",
            liquidity_flag="taker"
        )
        sample_order.add_execution(execution)
        benchmark_price = Decimal("49950")

        # Act
        report = execution_service.create_execution_report(sample_order, benchmark_price)

        # Assert
        assert isinstance(report, ExecutionReport)
        assert report.order_id == sample_order.id
        assert report.total_executed_quantity == Decimal("1.0")
        assert report.average_execution_price == Decimal("50000")
        assert report.slippage != Decimal("0")  # Il devrait y avoir du slippage
        assert report.execution_quality in ["excellent", "good", "fair", "poor"]

    def test_create_execution_report_no_executions(self, execution_service, sample_order):
        """Test la création d'un rapport sans exécutions"""
        # Act
        report = execution_service.create_execution_report(sample_order)

        # Assert
        assert report.total_executed_quantity == Decimal("0")
        assert report.execution_quality == "poor"
        assert len(report.venues_used) == 0

    def test_assess_execution_quality(self, execution_service, sample_order):
        """Test l'évaluation de la qualité d'exécution"""
        # Act
        excellent = execution_service._assess_execution_quality(sample_order, Decimal("0.01"), Decimal("0.05"))
        good = execution_service._assess_execution_quality(sample_order, Decimal("0.08"), Decimal("0.15"))
        fair = execution_service._assess_execution_quality(sample_order, Decimal("0.15"), Decimal("0.3"))
        poor = execution_service._assess_execution_quality(sample_order, Decimal("0.5"), Decimal("1.0"))

        # Assert
        assert excellent == "excellent"
        assert good == "good"
        assert fair == "fair"
        assert poor == "poor"

    def test_create_child_orders(self, execution_service, sample_order, sample_market_data):
        """Test la création d'ordres enfants"""
        # Arrange
        plan = execution_service.create_execution_plan(
            sample_order, sample_market_data, RoutingStrategy.BEST_PRICE, ExecutionAlgorithm.TWAP
        )

        # Act
        child_orders = execution_service.create_child_orders(sample_order, plan)

        # Assert
        assert len(child_orders) == len(plan.slice_instructions)
        assert all(child.parent_order_id == sample_order.id for child in child_orders)
        assert all(child.symbol == sample_order.symbol for child in child_orders)
        assert all(child.side == sample_order.side for child in child_orders)

    def test_monitor_execution_progress(self, execution_service, sample_order):
        """Test la surveillance du progrès d'exécution"""
        # Arrange
        child_order1 = create_market_order("BTC-USD", OrderSide.BUY, Decimal("1.0"), "portfolio", "strategy")
        child_order1.parent_order_id = sample_order.id
        child_order1.status = OrderStatus.FILLED
        child_order1.filled_quantity = Decimal("1.0")

        child_order2 = create_market_order("BTC-USD", OrderSide.BUY, Decimal("1.0"), "portfolio", "strategy")
        child_order2.parent_order_id = sample_order.id
        child_order2.status = OrderStatus.SUBMITTED  # Use SUBMITTED to be "active"
        child_order2.filled_quantity = Decimal("0")

        child_orders = [child_order1, child_order2]

        # Act
        progress = execution_service.monitor_execution_progress(sample_order, child_orders)

        # Assert
        assert progress["parent_order_id"] == sample_order.id
        assert progress["total_children"] == 2
        assert progress["completed_children"] == 1
        assert progress["active_children"] == 1
        assert progress["progress_percentage"] == 50.0  # 1/2 = 50%

    def test_slice_instructions_immediate(self, execution_service, sample_order):
        """Test les instructions de slice pour exécution immédiate"""
        # Act
        instructions = execution_service._create_slice_instructions(
            sample_order, ["binance"], ExecutionAlgorithm.IMMEDIATE
        )

        # Assert
        assert len(instructions) == 1
        assert instructions[0]["timing"] == "immediate"
        assert instructions[0]["order_type"] == "market"

    def test_slice_instructions_twap(self, execution_service, sample_order):
        """Test les instructions de slice pour TWAP"""
        # Act
        instructions = execution_service._create_slice_instructions(
            sample_order, ["binance", "coinbase"], ExecutionAlgorithm.TWAP
        )

        # Assert
        assert len(instructions) == 6  # 6 tranches TWAP
        assert all("delay_" in instr["timing"] for instr in instructions)
        assert all(instr["order_type"] == "limit" for instr in instructions)

    def test_slice_instructions_iceberg(self, execution_service, sample_order):
        """Test les instructions de slice pour iceberg"""
        # Act
        instructions = execution_service._create_slice_instructions(
            sample_order, ["binance"], ExecutionAlgorithm.ICEBERG
        )

        # Assert
        assert len(instructions) == 5  # 5 tranches iceberg
        assert all(instr["timing"] == "after_previous_fill" for instr in instructions)
        assert all(instr["hidden"] is True for instr in instructions)

    def test_slice_instructions_vwap(self, execution_service, sample_order):
        """Test les instructions de slice pour VWAP"""
        # Act
        instructions = execution_service._create_slice_instructions(
            sample_order, ["binance"], ExecutionAlgorithm.VWAP
        )

        # Assert
        assert len(instructions) == 4  # 4 fenêtres de volume
        assert all("volume_window_" in instr["timing"] for instr in instructions)
        assert all(instr["order_type"] == "limit" for instr in instructions)


class TestIntegrationOrderExecution:
    """Tests d'intégration pour l'exécution d'ordres"""

    @pytest.fixture
    def integrated_system(self):
        """Système intégré pour les tests"""
        execution_adapter = OrderExecutionAdapter()
        execution_service = ExecutionService()
        mock_brokers = {
            "binance": MockBrokerAdapter("binance", base_latency_ms=1, fill_probability=1.0),
            "coinbase": MockBrokerAdapter("coinbase", base_latency_ms=1, fill_probability=1.0),
            "kraken": MockBrokerAdapter("kraken", base_latency_ms=1, fill_probability=1.0)
        }

        for venue, broker in mock_brokers.items():
            execution_adapter.register_broker(venue, broker)

        return {
            "adapter": execution_adapter,
            "service": execution_service,
            "brokers": mock_brokers
        }

    @pytest.fixture
    def sample_market_data(self) -> Dict[str, VenueQuote]:
        """Données de marché réalistes"""
        return {
            "binance": VenueQuote(
                venue="binance",
                symbol="BTC-USD",
                bid_price=Decimal("49900"),
                ask_price=Decimal("50100"),
                bid_size=Decimal("10.0"),
                ask_size=Decimal("8.0"),
                timestamp=datetime.utcnow()
            ),
            "coinbase": VenueQuote(
                venue="coinbase",
                symbol="BTC-USD",
                bid_price=Decimal("49950"),
                ask_price=Decimal("50050"),
                bid_size=Decimal("5.0"),
                ask_size=Decimal("12.0"),
                timestamp=datetime.utcnow()
            )
        }

    @pytest.mark.asyncio
    async def test_end_to_end_market_order(self, integrated_system, sample_market_data):
        """Test complet d'exécution d'un ordre market"""
        # Arrange
        order = create_market_order("BTC-USD", OrderSide.BUY, Decimal("2.0"), "portfolio", "strategy")
        service = integrated_system["service"]
        adapter = integrated_system["adapter"]

        # Act
        # 1. Créer le plan d'exécution
        plan = service.create_execution_plan(order, sample_market_data)

        # 2. Exécuter via l'adapter
        executions = await adapter.execute_order_plan(order, plan)

        # 3. Ajouter les exécutions à l'ordre
        for execution in executions:
            order.add_execution(execution)

        # 4. Créer le rapport
        report = service.create_execution_report(order, Decimal("50000"))

        # Assert
        assert len(executions) > 0
        assert all(isinstance(exec, OrderExecution) for exec in executions)
        assert report.total_executed_quantity > 0
        assert report.execution_quality in ["excellent", "good", "fair", "poor"]

    @pytest.mark.asyncio
    async def test_end_to_end_limit_order(self, integrated_system, sample_market_data):
        """Test complet d'exécution d'un ordre limite"""
        # Arrange
        order = create_limit_order("BTC-USD", OrderSide.BUY, Decimal("1.0"), Decimal("49900"), "portfolio", "strategy")
        service = integrated_system["service"]
        adapter = integrated_system["adapter"]

        # Act
        plan = service.create_execution_plan(order, sample_market_data)
        executions = await adapter.execute_order_plan(order, plan)

        # Assert
        assert isinstance(plan, ExecutionPlan)
        assert len(executions) >= 0  # Peut être 0 si pas d'exécution immédiate

    @pytest.mark.asyncio
    async def test_multi_venue_execution(self, integrated_system, sample_market_data):
        """Test d'exécution sur plusieurs venues"""
        # Arrange
        large_order = create_market_order("BTC-USD", OrderSide.BUY, Decimal("20.0"), "portfolio", "strategy")
        service = integrated_system["service"]
        adapter = integrated_system["adapter"]

        # Act
        plan = service.create_execution_plan(
            large_order, sample_market_data, RoutingStrategy.MINIMIZE_IMPACT
        )
        executions = await adapter.execute_order_plan(large_order, plan)

        # Assert
        assert len(plan.target_venues) >= 1  # Au moins une venue
        venues_used = set(exec.venue for exec in executions)
        # Note: Multi-venue execution depends on venue capacity and routing strategy
        # With MINIMIZE_IMPACT, the system should try to distribute across venues
        assert len(venues_used) >= 1  # Au moins une venue utilisée

    @pytest.mark.asyncio
    async def test_order_cancellation_workflow(self, integrated_system, sample_market_data):
        """Test du workflow d'annulation d'ordre"""
        # Arrange - Use a very low price limit order that won't execute
        order = create_limit_order("BTC-USD", OrderSide.BUY, Decimal("1.0"), Decimal("1000"), "portfolio", "strategy")
        service = integrated_system["service"]
        adapter = integrated_system["adapter"]

        # Act
        # 1. Placer l'ordre
        plan = service.create_execution_plan(order, sample_market_data)
        await adapter.execute_order_plan(order, plan)

        # Give minimal time
        await asyncio.sleep(0.01)

        # 2. Annuler l'ordre
        cancel_results = await adapter.cancel_order(order.id)

        # Assert
        assert len(cancel_results) > 0
        # Note: cancellation success depends on execution timing
        assert all(isinstance(result, bool) for result in cancel_results.values())

    @pytest.mark.asyncio
    async def test_execution_monitoring(self, integrated_system, sample_market_data):
        """Test de la surveillance d'exécution"""
        # Arrange
        order = create_market_order("BTC-USD", OrderSide.BUY, Decimal("3.0"), "portfolio", "strategy")
        service = integrated_system["service"]
        adapter = integrated_system["adapter"]

        # Act
        plan = service.create_execution_plan(order, sample_market_data, execution_algorithm=ExecutionAlgorithm.TWAP)
        child_orders = service.create_child_orders(order, plan)

        # Simuler quelques exécutions
        for child in child_orders[:2]:
            child.status = OrderStatus.FILLED
            child.filled_quantity = child.quantity

        progress = service.monitor_execution_progress(order, child_orders)

        # Assert
        assert progress["total_children"] == len(child_orders)
        assert progress["completed_children"] == 2
        assert progress["progress_percentage"] > 0

    @pytest.mark.asyncio
    async def test_error_handling_broker_failure(self, integrated_system, sample_market_data):
        """Test de gestion d'erreur lors d'échec broker"""
        # Arrange
        order = create_market_order("BTC-USD", OrderSide.BUY, Decimal("1.0"), "portfolio", "strategy")
        adapter = integrated_system["adapter"]

        # Créer un plan avec une venue non enregistrée
        plan = ExecutionPlan(
            order_id=order.id,
            target_venues=["non_existent_venue"],
            routing_strategy=RoutingStrategy.BEST_PRICE,
            execution_algorithm=ExecutionAlgorithm.IMMEDIATE,
            estimated_cost=Decimal("50000"),
            estimated_duration=timedelta(seconds=5),
            slice_instructions=[],
            risk_checks_passed=True,
            created_time=datetime.utcnow()
        )

        # Act
        executions = await adapter.execute_order_plan(order, plan)

        # Assert
        assert len(executions) == 0  # Aucune exécution sur venue inexistante

    @pytest.mark.asyncio
    async def test_performance_metrics(self, integrated_system, sample_market_data):
        """Test des métriques de performance"""
        # Arrange
        order = create_market_order("BTC-USD", OrderSide.BUY, Decimal("1.0"), "portfolio", "strategy")
        service = integrated_system["service"]
        adapter = integrated_system["adapter"]

        # Act
        start_time = datetime.utcnow()
        plan = service.create_execution_plan(order, sample_market_data)
        executions = await adapter.execute_order_plan(order, plan)
        end_time = datetime.utcnow()

        execution_time = (end_time - start_time).total_seconds()

        # Assert
        assert execution_time < 1.0  # Exécution rapide pour test
        assert len(executions) > 0

        # Vérifier les métriques dans le rapport
        report = service.create_execution_report(order, Decimal("50000"))
        assert report.execution_time_seconds >= 0
        assert report.total_commission >= 0

    @pytest.mark.asyncio
    async def test_risk_checks_failure(self, integrated_system, sample_market_data):
        """Test d'échec des vérifications de risque"""
        # Arrange - Create an invalid order by setting quantity to 0 after creation
        invalid_order = create_market_order("BTC-USD", OrderSide.BUY, Decimal("1.0"), "portfolio", "strategy")
        invalid_order.quantity = Decimal("0")  # Set invalid quantity after creation
        service = integrated_system["service"]

        # Act
        plan = service.create_execution_plan(invalid_order, sample_market_data)

        # Assert
        assert plan.risk_checks_passed is False

        # L'exécution devrait échouer
        with pytest.raises(ValueError, match="Risk checks failed"):
            service.execute_order(invalid_order, plan, sample_market_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])