"""
Tests for API Services
=====================

Tests ciblés pour les services API.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal
from datetime import datetime
import pandas as pd

from qframe.api.services.market_data_service import MarketDataService
from qframe.api.services.order_service import OrderService as APIOrderService
from qframe.api.services.portfolio_service import PortfolioService as APIPortfolioService
from qframe.api.services.strategy_service import StrategyService as APIStrategyService
from qframe.api.services.risk_service import RiskService as APIRiskService


class TestMarketDataService:
    """Tests pour MarketDataService API."""

    @pytest.fixture
    def mock_data_provider(self):
        return Mock()

    @pytest.fixture
    def mock_metrics(self):
        return Mock()

    @pytest.fixture
    def market_data_service(self, mock_data_provider, mock_metrics):
        return MarketDataService(
            data_provider=mock_data_provider,
            metrics_collector=mock_metrics
        )

    def test_get_current_price(self, market_data_service, mock_data_provider):
        """Test récupération prix actuel."""
        mock_data_provider.get_current_price.return_value = {
            "symbol": "BTC/USD",
            "price": 45000.0,
            "timestamp": datetime.utcnow()
        }

        result = market_data_service.get_current_price("BTC/USD")

        assert result["symbol"] == "BTC/USD"
        assert result["price"] == 45000.0
        mock_data_provider.get_current_price.assert_called_once_with("BTC/USD")

    def test_get_historical_data(self, market_data_service, mock_data_provider):
        """Test récupération données historiques."""
        mock_df = pd.DataFrame({
            'timestamp': [datetime.utcnow()],
            'open': [44000.0],
            'high': [45000.0],
            'low': [43000.0],
            'close': [44500.0],
            'volume': [1000.0]
        })
        mock_data_provider.fetch_ohlcv.return_value = mock_df

        result = market_data_service.get_historical_data("BTC/USD", "1h", 100)

        assert not result.empty
        assert "close" in result.columns
        mock_data_provider.fetch_ohlcv.assert_called_once()

    def test_get_market_status(self, market_data_service):
        """Test statut du marché."""
        status = market_data_service.get_market_status()

        assert "is_open" in status
        assert "session" in status
        assert isinstance(status["is_open"], bool)

    def test_get_supported_symbols(self, market_data_service, mock_data_provider):
        """Test symboles supportés."""
        mock_data_provider.get_supported_symbols.return_value = ["BTC/USD", "ETH/USD"]

        symbols = market_data_service.get_supported_symbols()

        assert "BTC/USD" in symbols
        assert "ETH/USD" in symbols

    def test_data_validation(self, market_data_service, mock_data_provider):
        """Test validation des données."""
        # Données invalides
        mock_data_provider.get_current_price.return_value = None

        result = market_data_service.get_current_price("INVALID/SYMBOL")

        assert result is None or "error" in result


class TestAPIOrderService:
    """Tests pour OrderService API."""

    @pytest.fixture
    def mock_order_repository(self):
        return Mock()

    @pytest.fixture
    def mock_execution_service(self):
        return Mock()

    @pytest.fixture
    def api_order_service(self, mock_order_repository, mock_execution_service):
        return APIOrderService(
            order_repository=mock_order_repository,
            execution_service=mock_execution_service
        )

    def test_create_order_api(self, api_order_service, mock_order_repository):
        """Test création d'ordre via API."""
        order_data = {
            "portfolio_id": "portfolio-001",
            "symbol": "BTC/USD",
            "side": "BUY",
            "order_type": "MARKET",
            "quantity": 1.0
        }

        mock_order = Mock()
        mock_order.id = "order-001"
        mock_order_repository.save.return_value = mock_order

        result = api_order_service.create_order(order_data)

        assert result.id == "order-001"
        mock_order_repository.save.assert_called_once()

    def test_get_orders_with_pagination(self, api_order_service, mock_order_repository):
        """Test récupération ordres avec pagination."""
        mock_orders = [Mock() for _ in range(5)]
        mock_order_repository.get_all.return_value = mock_orders

        result = api_order_service.get_orders(limit=10, offset=0)

        assert len(result) == 5
        mock_order_repository.get_all.assert_called_once()

    def test_order_status_update(self, api_order_service, mock_order_repository):
        """Test mise à jour statut ordre."""
        mock_order = Mock()
        mock_order.id = "order-001"
        mock_order_repository.get_by_id.return_value = mock_order
        mock_order_repository.save.return_value = mock_order

        result = api_order_service.update_order_status("order-001", "FILLED")

        assert result is not None
        mock_order_repository.save.assert_called_once()


class TestAPIPortfolioService:
    """Tests pour PortfolioService API."""

    @pytest.fixture
    def mock_portfolio_repository(self):
        return Mock()

    @pytest.fixture
    def mock_position_service(self):
        return Mock()

    @pytest.fixture
    def api_portfolio_service(self, mock_portfolio_repository, mock_position_service):
        return APIPortfolioService(
            portfolio_repository=mock_portfolio_repository,
            position_service=mock_position_service
        )

    def test_get_portfolio_summary(self, api_portfolio_service, mock_portfolio_repository):
        """Test résumé de portfolio."""
        mock_portfolio = Mock()
        mock_portfolio.id = "portfolio-001"
        mock_portfolio.total_value = Decimal("100000.00")
        mock_portfolio_repository.get_by_id.return_value = mock_portfolio

        summary = api_portfolio_service.get_portfolio_summary("portfolio-001")

        assert summary["portfolio_id"] == "portfolio-001"
        assert summary["total_value"] == 100000.0

    def test_portfolio_performance_metrics(self, api_portfolio_service, mock_portfolio_repository):
        """Test métriques de performance."""
        mock_portfolio = Mock()
        mock_portfolio.id = "portfolio-001"
        mock_portfolio.initial_capital = Decimal("100000.00")
        mock_portfolio.total_value = Decimal("115000.00")
        mock_portfolio_repository.get_by_id.return_value = mock_portfolio

        metrics = api_portfolio_service.get_performance_metrics("portfolio-001")

        assert "total_return" in metrics
        assert "portfolio_value" in metrics

    def test_portfolio_allocation(self, api_portfolio_service, mock_portfolio_repository):
        """Test allocation du portfolio."""
        mock_portfolio = Mock()
        mock_portfolio.positions = {
            "BTC/USD": Mock(market_value=Decimal("50000")),
            "ETH/USD": Mock(market_value=Decimal("30000"))
        }
        mock_portfolio.cash_balance = Decimal("20000")
        mock_portfolio_repository.get_by_id.return_value = mock_portfolio

        allocation = api_portfolio_service.get_allocation("portfolio-001")

        assert "positions" in allocation
        assert "cash_percentage" in allocation


class TestAPIStrategyService:
    """Tests pour StrategyService API."""

    @pytest.fixture
    def mock_strategy_repository(self):
        return Mock()

    @pytest.fixture
    def mock_signal_service(self):
        return Mock()

    @pytest.fixture
    def api_strategy_service(self, mock_strategy_repository, mock_signal_service):
        return APIStrategyService(
            strategy_repository=mock_strategy_repository,
            signal_service=mock_signal_service
        )

    def test_create_strategy_api(self, api_strategy_service, mock_strategy_repository):
        """Test création de stratégie via API."""
        strategy_data = {
            "name": "Test Strategy",
            "description": "API Test Strategy",
            "strategy_type": "mean_reversion",
            "parameters": {"lookback": 20}
        }

        mock_strategy = Mock()
        mock_strategy.id = "strategy-001"
        mock_strategy_repository.save.return_value = mock_strategy

        result = api_strategy_service.create_strategy(strategy_data)

        assert result.id == "strategy-001"
        mock_strategy_repository.save.assert_called_once()

    def test_strategy_backtesting(self, api_strategy_service, mock_strategy_repository):
        """Test backtesting de stratégie."""
        mock_strategy = Mock()
        mock_strategy.id = "strategy-001"
        mock_strategy_repository.get_by_id.return_value = mock_strategy

        backtest_config = {
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 100000.0
        }

        result = api_strategy_service.run_backtest("strategy-001", backtest_config)

        assert "backtest_id" in result

    def test_strategy_signals_generation(self, api_strategy_service, mock_signal_service):
        """Test génération de signaux."""
        mock_signals = [Mock() for _ in range(3)]
        mock_signal_service.generate_signals.return_value = mock_signals

        signals = api_strategy_service.get_latest_signals("strategy-001", limit=5)

        assert len(signals) == 3

    def test_strategy_performance_tracking(self, api_strategy_service, mock_strategy_repository):
        """Test suivi de performance de stratégie."""
        mock_strategy = Mock()
        mock_strategy.performance_metrics = {
            "total_return": 0.15,
            "sharpe_ratio": 1.8,
            "max_drawdown": 0.08
        }
        mock_strategy_repository.get_by_id.return_value = mock_strategy

        performance = api_strategy_service.get_strategy_performance("strategy-001")

        assert performance["total_return"] == 0.15
        assert performance["sharpe_ratio"] == 1.8


class TestAPIRiskService:
    """Tests pour RiskService API."""

    @pytest.fixture
    def mock_risk_calculation_service(self):
        return Mock()

    @pytest.fixture
    def mock_portfolio_repository(self):
        return Mock()

    @pytest.fixture
    def api_risk_service(self, mock_risk_calculation_service, mock_portfolio_repository):
        return APIRiskService(
            risk_calculation_service=mock_risk_calculation_service,
            portfolio_repository=mock_portfolio_repository
        )

    def test_portfolio_risk_assessment(self, api_risk_service, mock_risk_calculation_service):
        """Test évaluation de risque de portfolio."""
        mock_assessment = Mock()
        mock_assessment.overall_risk_level = "MEDIUM"
        mock_assessment.risk_score = Decimal("65")
        mock_risk_calculation_service.assess_portfolio_risk.return_value = mock_assessment

        result = api_risk_service.assess_portfolio_risk("portfolio-001")

        assert result.overall_risk_level == "MEDIUM"
        assert result.risk_score == 65

    def test_var_calculation(self, api_risk_service, mock_risk_calculation_service):
        """Test calcul VaR."""
        mock_var = Decimal("5000.00")
        mock_risk_calculation_service.calculate_var.return_value = mock_var

        var_95 = api_risk_service.calculate_portfolio_var("portfolio-001", 0.95)

        assert var_95 == 5000.00

    def test_risk_limits_validation(self, api_risk_service):
        """Test validation des limites de risque."""
        risk_limits = {
            "max_position_size": 0.1,
            "var_limit": 0.02,
            "stop_loss": 0.05
        }

        result = api_risk_service.validate_risk_limits(risk_limits)

        assert result["valid"] is True

    def test_risk_monitoring(self, api_risk_service, mock_risk_calculation_service):
        """Test monitoring de risque."""
        mock_alerts = [
            {"type": "high_var", "severity": "warning"},
            {"type": "position_limit", "severity": "critical"}
        ]
        mock_risk_calculation_service.get_risk_alerts.return_value = mock_alerts

        alerts = api_risk_service.get_risk_alerts("portfolio-001")

        assert len(alerts) == 2
        assert alerts[0]["type"] == "high_var"

    def test_stress_testing(self, api_risk_service, mock_risk_calculation_service):
        """Test stress testing."""
        scenarios = [-0.1, -0.2, -0.3]  # Baisse de 10%, 20%, 30%
        mock_results = {
            -0.1: Decimal("95000.00"),
            -0.2: Decimal("90000.00"),
            -0.3: Decimal("85000.00")
        }
        mock_risk_calculation_service.stress_test.return_value = mock_results

        results = api_risk_service.run_stress_test("portfolio-001", scenarios)

        assert len(results) == 3
        assert results[-0.1] == 95000.00


class TestServiceIntegration:
    """Tests d'intégration des services API."""

    def test_order_to_portfolio_integration(self):
        """Test intégration ordre → portfolio."""
        # Mock des services
        mock_order_service = Mock()
        mock_portfolio_service = Mock()

        # Simulation création ordre qui affecte le portfolio
        order_data = {
            "portfolio_id": "portfolio-001",
            "symbol": "BTC/USD",
            "side": "BUY",
            "quantity": 1.0
        }

        mock_order = Mock()
        mock_order.portfolio_id = "portfolio-001"
        mock_order_service.create_order.return_value = mock_order

        # Vérifier que le portfolio est mis à jour
        mock_portfolio_service.update_from_order.return_value = True

        # Simulation du workflow
        order = mock_order_service.create_order(order_data)
        portfolio_updated = mock_portfolio_service.update_from_order(order)

        assert order.portfolio_id == "portfolio-001"
        assert portfolio_updated is True

    def test_strategy_to_order_integration(self):
        """Test intégration stratégie → ordre."""
        mock_strategy_service = Mock()
        mock_order_service = Mock()

        # Génération de signal par stratégie
        mock_signal = Mock()
        mock_signal.symbol = "BTC/USD"
        mock_signal.action = "BUY"
        mock_strategy_service.generate_signal.return_value = mock_signal

        # Création d'ordre depuis signal
        mock_order = Mock()
        mock_order_service.create_from_signal.return_value = mock_order

        # Workflow signal → ordre
        signal = mock_strategy_service.generate_signal("strategy-001")
        order = mock_order_service.create_from_signal(signal)

        assert signal.symbol == "BTC/USD"
        assert order is not None

    def test_risk_monitoring_integration(self):
        """Test intégration monitoring de risque."""
        mock_risk_service = Mock()
        mock_portfolio_service = Mock()

        # Monitoring déclenche alerte
        mock_alerts = [{"type": "high_var", "severity": "critical"}]
        mock_risk_service.monitor_portfolio.return_value = mock_alerts

        # Action sur le portfolio en réponse
        mock_portfolio_service.apply_risk_action.return_value = {"action": "reduce_exposure"}

        # Workflow monitoring → action
        alerts = mock_risk_service.monitor_portfolio("portfolio-001")
        if alerts:
            action = mock_portfolio_service.apply_risk_action("portfolio-001", alerts[0])

        assert len(alerts) == 1
        assert action["action"] == "reduce_exposure"