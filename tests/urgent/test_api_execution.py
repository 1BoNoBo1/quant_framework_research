"""
Tests d'Exécution Réelle - API FastAPI
=======================================

Tests qui EXÉCUTENT vraiment le code qframe.api
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional
from unittest.mock import Mock, AsyncMock, patch

# API modules
from qframe.api.main import create_app, app
from qframe.api.models.requests import (
    CreateOrderRequest, UpdateOrderRequest,
    CreateStrategyRequest, BacktestRequest
)
from qframe.api.models.responses import (
    OrderResponse, PortfolioResponse, PositionResponse,
    MarketDataResponse, StrategyResponse, RiskMetricsResponse
)


class TestAPIMainExecution:
    """Tests d'exécution réelle pour qframe.api.main."""

    @pytest.fixture
    def app(self):
        """Application FastAPI de test."""
        # Créer l'application
        app = create_app()
        return app

    @pytest.fixture
    def client(self, app):
        """Client de test FastAPI."""
        return TestClient(app)

    def test_create_app_execution(self):
        """Test création de l'application FastAPI."""
        # Exécuter création
        app = create_app()

        # Vérifier création
        assert isinstance(app, FastAPI)
        assert app.title is not None
        assert app.version is not None

        # Vérifier que les routes sont enregistrées
        routes = [route.path for route in app.routes]
        assert "/" in routes  # Route racine
        assert "/health" in routes  # Health check

    def test_get_application_execution(self):
        """Test récupération de l'application."""
        # Exécuter récupération - utiliser app directement
        from qframe.api.main import app as application

        # Vérifier application
        assert isinstance(application, FastAPI)
        assert hasattr(application, 'routes')
        assert hasattr(application, 'middleware')

    def test_health_check_execution(self, client):
        """Test endpoint health check."""
        # Exécuter requête health
        response = client.get("/health")

        # Vérifier réponse
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "ok"]

    def test_root_endpoint_execution(self, client):
        """Test endpoint racine."""
        # Exécuter requête racine
        response = client.get("/")

        # Vérifier réponse
        assert response.status_code == 200
        data = response.json()
        assert "message" in data or "name" in data
        assert "version" in data

    def test_api_docs_available_execution(self, client):
        """Test disponibilité documentation API."""
        # Exécuter requête docs
        response = client.get("/docs")

        # Vérifier réponse (redirect ou HTML)
        assert response.status_code in [200, 307]  # OK ou redirect

        # Test OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema


class TestAPIModelsExecution:
    """Tests d'exécution réelle pour qframe.api.models."""

    def test_create_order_request_execution(self):
        """Test modèle CreateOrderRequest."""
        # Exécuter création avec données valides
        request = CreateOrderRequest(
            portfolio_id="portfolio-001",
            symbol="BTC/USD",
            side="buy",
            order_type="market",
            quantity=1.5,
            price=50000.0,
            time_in_force="GTC"
        )

        # Vérifier sérialisation
        request_dict = request.model_dump()
        assert isinstance(request_dict, dict)
        assert request_dict["portfolio_id"] == "portfolio-001"
        assert request_dict["symbol"] == "BTC/USD"
        assert request_dict["quantity"] == 1.5

        # Vérifier validation JSON
        json_str = request.model_dump_json()
        assert isinstance(json_str, str)
        assert "portfolio-001" in json_str

    def test_update_order_request_execution(self):
        """Test modèle UpdateOrderRequest."""
        # Exécuter création avec mise à jour partielle
        request = UpdateOrderRequest(
            status="filled",
            executed_quantity=0.75,
            average_price=49950.0
        )

        # Vérifier sérialisation
        request_dict = request.model_dump(exclude_none=True)
        assert "status" in request_dict
        assert "executed_quantity" in request_dict
        assert request_dict["executed_quantity"] == 0.75

    def test_create_strategy_request_execution(self):
        """Test modèle CreateStrategyRequest."""
        # Exécuter création
        request = CreateStrategyRequest(
            name="Test Strategy",
            strategy_type="mean_reversion",
            config={"lookback": 20, "threshold": 0.02}
        )

        # Vérifier sérialisation
        request_dict = request.model_dump()
        assert request_dict["name"] == "Test Strategy"
        assert request_dict["strategy_type"] == "mean_reversion"
        assert "config" in request_dict
        assert request_dict["config"]["lookback"] == 20

    def test_order_response_execution(self):
        """Test modèle OrderResponse."""
        # Exécuter création de réponse
        response = OrderResponse(
            id="order-001",
            portfolio_id="portfolio-001",
            symbol="BTC/USD",
            side="buy",
            order_type="market",
            quantity=1.0,
            price=50000.0,
            status="pending",
            created_at=datetime.utcnow(),
            executed_quantity=0.0,
            average_price=0.0
        )

        # Vérifier sérialisation
        response_dict = response.model_dump()
        assert response_dict["id"] == "order-001"
        assert response_dict["status"] == "pending"
        assert isinstance(response_dict["created_at"], datetime)

        # Vérifier JSON
        json_str = response.model_dump_json()
        assert isinstance(json_str, str)
        assert "order-001" in json_str

    def test_portfolio_response_execution(self):
        """Test modèle PortfolioResponse."""
        # Exécuter création
        response = PortfolioResponse(
            id="portfolio-001",
            name="Test Portfolio",
            total_value=125000.0,
            cash_balance=25000.0,
            positions=[
                {
                    "symbol": "BTC/USD",
                    "quantity": 2.0,
                    "average_price": 45000.0,
                    "current_price": 50000.0,
                    "market_value": 100000.0,
                    "unrealized_pnl": 10000.0
                }
            ],
            performance={
                "total_return": 0.25,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.1
            },
            created_at=datetime.utcnow()
        )

        # Vérifier sérialisation
        response_dict = response.model_dump()
        assert response_dict["id"] == "portfolio-001"
        assert response_dict["total_value"] == 125000.0
        assert len(response_dict["positions"]) == 1
        assert response_dict["performance"]["total_return"] == 0.25

    def test_risk_metrics_response_execution(self):
        """Test modèle RiskMetricsResponse."""
        # Exécuter création
        response = RiskMetricsResponse(
            portfolio_id="portfolio-001",
            var_95=5000.0,
            cvar_95=6000.0,
            volatility=0.25,
            sharpe_ratio=1.2,
            max_drawdown=0.15,
            beta=0.9,
            correlation_matrix={"BTC": {"ETH": 0.7}},
            risk_score=65.0,
            calculated_at=datetime.utcnow()
        )

        # Vérifier sérialisation
        response_dict = response.model_dump()
        assert response_dict["portfolio_id"] == "portfolio-001"
        assert response_dict["var_95"] == 5000.0
        assert response_dict["volatility"] == 0.25
        assert isinstance(response_dict["correlation_matrix"], dict)


class TestAPIRoutesExecution:
    """Tests d'exécution réelle pour les routes API."""

    @pytest.fixture
    def app(self):
        """Application avec services mockés."""
        app = create_app()
        return app

    @pytest.fixture
    def client(self, app):
        """Client de test."""
        return TestClient(app)

    @patch('qframe.api.routers.orders.OrderService')
    def test_create_order_route_execution(self, mock_order_service, client):
        """Test route création d'ordre."""
        # Mock du service
        mock_service = mock_order_service.return_value
        mock_service.create_order.return_value = {
            "id": "order-123",
            "status": "pending",
            "symbol": "BTC/USD"
        }

        # Données de requête
        order_data = {
            "portfolio_id": "portfolio-001",
            "symbol": "BTC/USD",
            "side": "buy",
            "order_type": "market",
            "quantity": 1.0
        }

        # Exécuter requête POST
        response = client.post("/api/v1/orders", json=order_data)

        # Vérifier réponse
        assert response.status_code in [200, 201]
        data = response.json()
        assert "id" in data
        assert data["symbol"] == "BTC/USD"

    @patch('qframe.api.routers.portfolios.PortfolioService')
    def test_get_portfolio_route_execution(self, mock_portfolio_service, client):
        """Test route récupération portfolio."""
        # Mock du service
        mock_service = mock_portfolio_service.return_value
        mock_service.get_portfolio.return_value = {
            "id": "portfolio-001",
            "name": "Test Portfolio",
            "total_value": 100000.0
        }

        # Exécuter requête GET
        response = client.get("/api/v1/portfolios/portfolio-001")

        # Vérifier réponse
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "portfolio-001"
        assert data["total_value"] == 100000.0

    @patch('qframe.api.routers.market_data.MarketDataService')
    def test_get_market_data_route_execution(self, mock_market_service, client):
        """Test route données de marché."""
        # Mock du service
        mock_service = mock_market_service.return_value
        mock_service.get_ohlcv.return_value = {
            "symbol": "BTC/USD",
            "timeframe": "1h",
            "data": [
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "open": 49000,
                    "high": 50000,
                    "low": 48500,
                    "close": 49500,
                    "volume": 1000
                }
            ]
        }

        # Exécuter requête GET
        response = client.get("/api/v1/market-data/ohlcv?symbol=BTC/USD&timeframe=1h")

        # Vérifier réponse
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "BTC/USD"
        assert "data" in data
        assert len(data["data"]) > 0

    @patch('qframe.api.routers.risk.RiskService')
    def test_calculate_risk_metrics_route_execution(self, mock_risk_service, client):
        """Test route calcul métriques de risque."""
        # Mock du service
        mock_service = mock_risk_service.return_value
        mock_service.calculate_portfolio_risk.return_value = {
            "portfolio_id": "portfolio-001",
            "var_95": 5000.0,
            "volatility": 0.25,
            "sharpe_ratio": 1.5
        }

        # Exécuter requête POST
        response = client.post(
            "/api/v1/risk/calculate",
            json={"portfolio_id": "portfolio-001"}
        )

        # Vérifier réponse
        assert response.status_code == 200
        data = response.json()
        assert data["portfolio_id"] == "portfolio-001"
        assert data["var_95"] == 5000.0
        assert data["volatility"] == 0.25

    @patch('qframe.api.routers.strategies.StrategyService')
    def test_list_strategies_route_execution(self, mock_strategy_service, client):
        """Test route liste des stratégies."""
        # Mock du service
        mock_service = mock_strategy_service.return_value
        mock_service.list_strategies.return_value = [
            {
                "id": "strategy-001",
                "name": "Mean Reversion",
                "type": "mean_reversion",
                "status": "active"
            },
            {
                "id": "strategy-002",
                "name": "DMN LSTM",
                "type": "ml",
                "status": "inactive"
            }
        ]

        # Exécuter requête GET
        response = client.get("/api/v1/strategies")

        # Vérifier réponse
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["name"] == "Mean Reversion"


class TestAPIMiddlewareExecution:
    """Tests d'exécution réelle pour les middlewares API."""

    @pytest.fixture
    def app(self):
        """Application avec middlewares."""
        from qframe.api.middleware.auth import AuthMiddleware

        app = create_app()
        # Ajouter middleware d'authentification
        app.add_middleware(AuthMiddleware)
        return app

    @pytest.fixture
    def client(self, app):
        """Client de test."""
        return TestClient(app)

    def test_cors_middleware_execution(self, client):
        """Test middleware CORS."""
        # Exécuter requête avec origin
        response = client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"}
        )

        # Vérifier headers CORS
        assert response.status_code == 200
        # Les headers CORS peuvent être présents
        headers = response.headers
        # Vérifier que la réponse est valide

    def test_request_id_middleware_execution(self, client):
        """Test middleware request ID."""
        # Exécuter requête
        response = client.get("/health")

        # Vérifier headers
        assert response.status_code == 200
        # Request ID peut être dans les headers
        headers = dict(response.headers)
        # Certains middlewares ajoutent x-request-id


class TestAPIServicesExecution:
    """Tests d'exécution réelle pour les services API."""

    @pytest.fixture
    def mock_repository(self):
        """Repository mocké."""
        repo = AsyncMock()
        repo.find_by_id.return_value = {
            "id": "test-001",
            "value": 100
        }
        repo.save.return_value = {"id": "test-001", "saved": True}
        return repo

    def test_base_service_initialization_execution(self):
        """Test initialisation service de base."""
        from qframe.api.services.base_service import BaseService

        # Exécuter création
        service = BaseService()

        # Vérifier initialisation
        assert service is not None
        assert hasattr(service, '__class__')

    @pytest.mark.asyncio
    async def test_order_service_execution(self, mock_repository):
        """Test service d'ordres."""
        from qframe.api.services.order_service import OrderService

        # Créer service avec repo mocké
        service = OrderService(repository=mock_repository)

        # Exécuter méthode get_order
        result = await service.get_order("order-001")

        # Vérifier appel
        mock_repository.find_by_id.assert_called_once_with("order-001")
        assert result["id"] == "test-001"

    @pytest.mark.asyncio
    async def test_portfolio_service_execution(self, mock_repository):
        """Test service de portfolios."""
        from qframe.api.services.portfolio_service import PortfolioService

        # Créer service
        service = PortfolioService(repository=mock_repository)

        # Exécuter méthode
        result = await service.get_portfolio("portfolio-001")

        # Vérifier
        mock_repository.find_by_id.assert_called_once()
        assert result is not None


class TestAPIIntegrationExecution:
    """Tests d'intégration complète de l'API."""

    @pytest.fixture
    def app(self):
        """Application complète."""
        return create_app()

    @pytest.fixture
    def client(self, app):
        """Client de test."""
        return TestClient(app)

    def test_api_workflow_execution(self, client):
        """Test workflow complet API."""
        # 1. Vérifier health
        response = client.get("/health")
        assert response.status_code == 200

        # 2. Récupérer info API
        response = client.get("/")
        assert response.status_code == 200
        api_info = response.json()
        assert "version" in api_info

        # 3. Vérifier documentation
        response = client.get("/openapi.json")
        assert response.status_code == 200
        openapi = response.json()
        assert "paths" in openapi

        # 4. Vérifier au moins une route API existe
        paths = openapi.get("paths", {})
        assert len(paths) > 0

        # Vérifier structure OpenAPI
        assert "info" in openapi
        assert "title" in openapi["info"]