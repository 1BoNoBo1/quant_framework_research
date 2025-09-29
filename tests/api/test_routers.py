"""
Tests for API Routers
====================

Tests ciblés pour les routeurs FastAPI.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from decimal import Decimal
from datetime import datetime

from qframe.api.main import app
from qframe.domain.entities.order import Order, OrderStatus, OrderSide, OrderType
from qframe.domain.entities.portfolio import Portfolio
from qframe.domain.entities.position import Position


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_container():
    """Mock du container DI."""
    return Mock()


class TestOrdersRouter:
    """Tests pour le routeur des ordres."""

    def test_get_orders_endpoint(self, client):
        """Test endpoint GET /orders."""
        with patch('qframe.api.routers.orders.get_container') as mock_get_container:
            mock_service = Mock()
            mock_service.get_all_orders.return_value = []
            mock_get_container.return_value.resolve.return_value = mock_service

            response = client.get("/api/v1/orders")

            assert response.status_code == 200
            assert response.json() == []

    def test_create_order_endpoint(self, client):
        """Test endpoint POST /orders."""
        order_data = {
            "portfolio_id": "portfolio-001",
            "symbol": "BTC/USD",
            "side": "BUY",
            "order_type": "MARKET",
            "quantity": 1.0
        }

        with patch('qframe.api.routers.orders.get_container') as mock_get_container:
            mock_service = Mock()
            mock_order = Order(
                id="order-001",
                portfolio_id="portfolio-001",
                symbol="BTC/USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
                created_time=datetime.utcnow()
            )
            mock_service.create_order.return_value = mock_order
            mock_get_container.return_value.resolve.return_value = mock_service

            response = client.post("/api/v1/orders", json=order_data)

            assert response.status_code == 201
            assert response.json()["id"] == "order-001"

    def test_get_order_by_id(self, client):
        """Test endpoint GET /orders/{order_id}."""
        with patch('qframe.api.routers.orders.get_container') as mock_get_container:
            mock_service = Mock()
            mock_order = Order(
                id="order-001",
                portfolio_id="portfolio-001",
                symbol="BTC/USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
                created_time=datetime.utcnow()
            )
            mock_service.get_order_by_id.return_value = mock_order
            mock_get_container.return_value.resolve.return_value = mock_service

            response = client.get("/api/v1/orders/order-001")

            assert response.status_code == 200
            assert response.json()["id"] == "order-001"

    def test_cancel_order(self, client):
        """Test endpoint DELETE /orders/{order_id}."""
        with patch('qframe.api.routers.orders.get_container') as mock_get_container:
            mock_service = Mock()
            mock_order = Order(
                id="order-001",
                portfolio_id="portfolio-001",
                symbol="BTC/USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
                status=OrderStatus.CANCELLED,
                created_time=datetime.utcnow()
            )
            mock_service.cancel_order.return_value = mock_order
            mock_get_container.return_value.resolve.return_value = mock_service

            response = client.delete("/api/v1/orders/order-001")

            assert response.status_code == 200
            assert response.json()["status"] == "CANCELLED"


class TestPositionsRouter:
    """Tests pour le routeur des positions."""

    def test_get_positions_endpoint(self, client):
        """Test endpoint GET /positions."""
        with patch('qframe.api.routers.positions.get_container') as mock_get_container:
            mock_service = Mock()
            mock_service.get_all_positions.return_value = []
            mock_get_container.return_value.resolve.return_value = mock_service

            response = client.get("/api/v1/positions")

            assert response.status_code == 200
            assert response.json() == []

    def test_get_positions_by_portfolio(self, client):
        """Test endpoint GET /positions/portfolio/{portfolio_id}."""
        with patch('qframe.api.routers.positions.get_container') as mock_get_container:
            mock_service = Mock()
            mock_position = Position(
                symbol="BTC/USD",
                quantity=Decimal("1.0"),
                average_price=Decimal("45000.00")
            )
            mock_service.get_positions_by_portfolio.return_value = [mock_position]
            mock_get_container.return_value.resolve.return_value = mock_service

            response = client.get("/api/v1/positions/portfolio/portfolio-001")

            assert response.status_code == 200
            assert len(response.json()) == 1

    def test_close_position(self, client):
        """Test endpoint DELETE /positions/{position_id}."""
        with patch('qframe.api.routers.positions.get_container') as mock_get_container:
            mock_service = Mock()
            mock_service.close_position.return_value = {"status": "closed"}
            mock_get_container.return_value.resolve.return_value = mock_service

            response = client.delete("/api/v1/positions/BTC-USD-001")

            assert response.status_code == 200
            assert response.json()["status"] == "closed"


class TestStrategiesRouter:
    """Tests pour le routeur des stratégies."""

    def test_get_strategies_endpoint(self, client):
        """Test endpoint GET /strategies."""
        with patch('qframe.api.routers.strategies.get_container') as mock_get_container:
            mock_service = Mock()
            mock_service.get_all_strategies.return_value = []
            mock_get_container.return_value.resolve.return_value = mock_service

            response = client.get("/api/v1/strategies")

            assert response.status_code == 200
            assert response.json() == []

    def test_create_strategy_endpoint(self, client):
        """Test endpoint POST /strategies."""
        strategy_data = {
            "name": "Test Strategy",
            "description": "A test strategy",
            "strategy_type": "mean_reversion",
            "parameters": {"lookback": 20}
        }

        with patch('qframe.api.routers.strategies.get_container') as mock_get_container:
            mock_service = Mock()
            mock_strategy = Mock()
            mock_strategy.id = "strategy-001"
            mock_strategy.name = "Test Strategy"
            mock_service.create_strategy.return_value = mock_strategy
            mock_get_container.return_value.resolve.return_value = mock_service

            response = client.post("/api/v1/strategies", json=strategy_data)

            assert response.status_code == 201
            assert response.json()["id"] == "strategy-001"

    def test_activate_strategy(self, client):
        """Test endpoint POST /strategies/{strategy_id}/activate."""
        with patch('qframe.api.routers.strategies.get_container') as mock_get_container:
            mock_service = Mock()
            mock_strategy = Mock()
            mock_strategy.id = "strategy-001"
            mock_strategy.status = "ACTIVE"
            mock_service.activate_strategy.return_value = mock_strategy
            mock_get_container.return_value.resolve.return_value = mock_service

            response = client.post("/api/v1/strategies/strategy-001/activate")

            assert response.status_code == 200
            assert response.json()["status"] == "ACTIVE"


class TestRiskRouter:
    """Tests pour le routeur de gestion des risques."""

    def test_get_portfolio_risk(self, client):
        """Test endpoint GET /risk/portfolio/{portfolio_id}."""
        with patch('qframe.api.routers.risk.get_container') as mock_get_container:
            mock_service = Mock()
            mock_assessment = {
                "portfolio_id": "portfolio-001",
                "risk_level": "MEDIUM",
                "var_95": 1000.0,
                "max_drawdown": 0.05
            }
            mock_service.assess_portfolio_risk.return_value = mock_assessment
            mock_get_container.return_value.resolve.return_value = mock_service

            response = client.get("/api/v1/risk/portfolio/portfolio-001")

            assert response.status_code == 200
            assert response.json()["risk_level"] == "MEDIUM"

    def test_update_risk_limits(self, client):
        """Test endpoint PUT /risk/limits."""
        risk_limits = {
            "max_position_size": 0.1,
            "var_limit": 0.02,
            "stop_loss": 0.05
        }

        with patch('qframe.api.routers.risk.get_container') as mock_get_container:
            mock_service = Mock()
            mock_service.update_risk_limits.return_value = risk_limits
            mock_get_container.return_value.resolve.return_value = mock_service

            response = client.put("/api/v1/risk/limits", json=risk_limits)

            assert response.status_code == 200
            assert response.json()["max_position_size"] == 0.1


class TestMarketDataRouter:
    """Tests pour le routeur des données de marché."""

    def test_get_market_data(self, client):
        """Test endpoint GET /market-data/{symbol}."""
        with patch('qframe.api.routers.market_data.get_container') as mock_get_container:
            mock_service = Mock()
            mock_data = {
                "symbol": "BTC/USD",
                "price": 45000.0,
                "volume": 1000000.0,
                "timestamp": "2023-01-01T00:00:00Z"
            }
            mock_service.get_current_data.return_value = mock_data
            mock_get_container.return_value.resolve.return_value = mock_service

            response = client.get("/api/v1/market-data/BTC/USD")

            assert response.status_code == 200
            assert response.json()["symbol"] == "BTC/USD"

    def test_get_historical_data(self, client):
        """Test endpoint GET /market-data/{symbol}/history."""
        with patch('qframe.api.routers.market_data.get_container') as mock_get_container:
            mock_service = Mock()
            mock_data = [
                {"timestamp": "2023-01-01T00:00:00Z", "price": 45000.0},
                {"timestamp": "2023-01-01T01:00:00Z", "price": 45100.0}
            ]
            mock_service.get_historical_data.return_value = mock_data
            mock_get_container.return_value.resolve.return_value = mock_service

            response = client.get("/api/v1/market-data/BTC/USD/history?period=1h&limit=100")

            assert response.status_code == 200
            assert len(response.json()) == 2

    def test_get_portfolio_summary(self, client):
        """Test endpoint GET /portfolios/{portfolio_id}/summary."""
        with patch('qframe.api.routers.positions.get_container') as mock_get_container:
            mock_service = Mock()
            mock_summary = {
                "portfolio_id": "portfolio-001",
                "total_value": 100000.0,
                "cash_balance": 50000.0,
                "positions_count": 5,
                "unrealized_pnl": 2500.0
            }
            mock_service.get_portfolio_summary.return_value = mock_summary
            mock_get_container.return_value.resolve.return_value = mock_service

            response = client.get("/api/v1/portfolios/portfolio-001/summary")

            assert response.status_code == 200
            assert response.json()["total_value"] == 100000.0


class TestErrorHandling:
    """Tests pour la gestion d'erreurs."""

    def test_invalid_order_data(self, client):
        """Test création d'ordre avec données invalides."""
        invalid_data = {
            "symbol": "INVALID",
            "quantity": -1.0  # Quantité négative
        }

        response = client.post("/api/v1/orders", json=invalid_data)

        assert response.status_code == 422  # Validation error

    def test_order_not_found(self, client):
        """Test récupération d'ordre inexistant."""
        with patch('qframe.api.routers.orders.get_container') as mock_get_container:
            mock_service = Mock()
            mock_service.get_order_by_id.return_value = None
            mock_get_container.return_value.resolve.return_value = mock_service

            response = client.get("/api/v1/orders/nonexistent")

            assert response.status_code == 404

    def test_portfolio_not_found(self, client):
        """Test récupération de portfolio inexistant."""
        with patch('qframe.api.routers.positions.get_container') as mock_get_container:
            mock_service = Mock()
            mock_service.get_positions_by_portfolio.return_value = []
            mock_get_container.return_value.resolve.return_value = mock_service

            response = client.get("/api/v1/positions/portfolio/nonexistent")

            assert response.status_code == 200  # Retourne liste vide, pas d'erreur
            assert response.json() == []

    def test_strategy_activation_error(self, client):
        """Test erreur d'activation de stratégie."""
        with patch('qframe.api.routers.strategies.get_container') as mock_get_container:
            mock_service = Mock()
            mock_service.activate_strategy.side_effect = ValueError("Strategy not found")
            mock_get_container.return_value.resolve.return_value = mock_service

            response = client.post("/api/v1/strategies/nonexistent/activate")

            assert response.status_code == 500  # Internal server error

    def test_risk_calculation_error(self, client):
        """Test erreur de calcul de risque."""
        with patch('qframe.api.routers.risk.get_container') as mock_get_container:
            mock_service = Mock()
            mock_service.assess_portfolio_risk.side_effect = Exception("Calculation error")
            mock_get_container.return_value.resolve.return_value = mock_service

            response = client.get("/api/v1/risk/portfolio/portfolio-001")

            assert response.status_code == 500