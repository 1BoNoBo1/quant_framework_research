"""
Infrastructure Layer: FastAPI REST API Service
==============================================

Service REST API complet avec FastAPI pour exposer toutes les fonctionnalités
du framework de trading quantitatif.
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, Response, status, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn

from ...core.container import DIContainer
from ...domain.entities.strategy import StrategyStatus
from ...domain.entities.portfolio import PortfolioStatus
from ...domain.entities.order import OrderStatus, OrderSide, OrderType
from ...domain.entities.risk_assessment import RiskLevel

from ..observability.logging import LoggerFactory
from ..observability.metrics import get_business_metrics
from ..observability.tracing import get_tracer, trace
from ..observability.health import get_health_monitor
from ..observability.dashboard import get_dashboard

from ..data.market_data_pipeline import get_market_data_pipeline, DataType
from ..data.real_time_streaming import get_streaming_service, SubscriptionLevel

from ..events.core import get_event_bus
from ..events.event_store import get_event_store
from ..events.saga import get_saga_manager
from ..events.projections import get_projection_manager


# ===============================
# Modèles Pydantic pour l'API
# ===============================

class APIResponse(BaseModel):
    """Réponse API standard"""
    success: bool = True
    data: Optional[Any] = None
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None

class ErrorResponse(BaseModel):
    """Réponse d'erreur API"""
    success: bool = False
    error: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None

class HealthResponse(BaseModel):
    """Réponse de health check"""
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    environment: str = "development"
    uptime_seconds: int = 0
    components: Dict[str, Any] = Field(default_factory=dict)

# Modèles pour Strategy
class CreateStrategyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    risk_limits: Dict[str, float] = Field(default_factory=dict)

class StrategyResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    status: str
    parameters: Dict[str, Any]
    risk_limits: Dict[str, float]
    created_at: datetime
    updated_at: Optional[datetime]

# Modèles pour Portfolio
class CreatePortfolioRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    initial_cash: float = Field(..., gt=0)
    currency: str = Field(default="USD", pattern=r"^[A-Z]{3}$")

class PortfolioResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    status: str
    total_value: float
    cash_balance: float
    currency: str
    positions_count: int
    created_at: datetime
    updated_at: Optional[datetime]

# Modèles pour Order
class CreateOrderRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    side: str = Field(..., pattern=r"^(buy|sell)$")
    order_type: str = Field(..., pattern=r"^(market|limit|stop|stop_limit)$")
    quantity: float = Field(..., gt=0)
    price: Optional[float] = Field(None, gt=0)
    portfolio_id: str

    @validator('price')
    def validate_price(cls, v, values):
        order_type = values.get('order_type')
        if order_type in ['limit', 'stop_limit'] and v is None:
            raise ValueError('Price is required for limit and stop_limit orders')
        return v

class OrderResponse(BaseModel):
    id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    status: str
    portfolio_id: str
    created_at: datetime
    updated_at: Optional[datetime]

# Modèles pour Market Data
class MarketDataRequest(BaseModel):
    symbols: List[str] = Field(..., min_items=1, max_items=100)
    data_types: List[str] = Field(..., min_items=1)
    providers: Optional[List[str]] = None

class MarketDataResponse(BaseModel):
    symbol: str
    data_type: str
    timestamp: datetime
    data: Dict[str, Any]
    provider: str
    quality: str

# Modèles pour WebSocket Streaming
class StreamSubscriptionRequest(BaseModel):
    level: str = Field(..., pattern=r"^(symbol|type|provider|all)$")
    filter_criteria: Dict[str, Any] = Field(default_factory=dict)
    max_queue_size: int = Field(default=1000, gt=0, le=10000)


class FastAPIService:
    """
    Service FastAPI principal pour l'API REST.
    Gère tous les endpoints et middleware.
    """

    def __init__(self, container: DIContainer):
        self.container = container
        self.logger = LoggerFactory.get_logger(__name__)
        self.metrics = get_business_metrics()
        self.tracer = get_tracer()

        # Services
        self.health_monitor = get_health_monitor()
        self.dashboard = get_dashboard()
        self.market_data_pipeline = get_market_data_pipeline()
        self.streaming_service = get_streaming_service()
        self.event_bus = get_event_bus()
        self.event_store = get_event_store()

        # FastAPI app
        self.app = self._create_app()
        self._setup_middleware()
        self._setup_routes()

        # Métadonnées
        self._start_time = time.time()

    def _create_app(self) -> FastAPI:
        """Créer l'application FastAPI"""
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            self.logger.info("Starting FastAPI application")
            yield
            # Shutdown
            self.logger.info("Shutting down FastAPI application")

        return FastAPI(
            title="QFrame Trading API",
            description="API REST pour le framework de trading quantitatif QFrame",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json",
            lifespan=lifespan
        )

    def _setup_middleware(self):
        """Configurer les middleware"""
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # À restreindre en production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Compression
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)

        # Request ID et logging
        @self.app.middleware("http")
        async def add_request_id(request: Request, call_next):
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id

            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time

            # Headers de réponse
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)

            # Logging
            self.logger.info(
                f"{request.method} {request.url.path}",
                request_id=request_id,
                method=request.method,
                path=str(request.url.path),
                status_code=response.status_code,
                process_time_ms=process_time * 1000
            )

            # Métriques
            self.metrics.collector.increment_counter(
                "api.requests",
                labels={
                    "method": request.method,
                    "endpoint": str(request.url.path),
                    "status_code": str(response.status_code)
                }
            )

            self.metrics.collector.record_histogram(
                "api.request_duration",
                process_time * 1000,
                labels={"method": request.method, "endpoint": str(request.url.path)}
            )

            return response

    def _setup_routes(self):
        """Configurer toutes les routes"""
        self._setup_health_routes()
        self._setup_strategy_routes()
        self._setup_portfolio_routes()
        self._setup_order_routes()
        self._setup_market_data_routes()
        self._setup_streaming_routes()
        self._setup_observability_routes()
        self._setup_event_routes()

    def _setup_health_routes(self):
        """Routes de health check"""
        @self.app.get("/health", response_model=HealthResponse, tags=["Health"])
        async def health_check():
            """Health check simple"""
            uptime = int(time.time() - self._start_time)
            system_health = self.health_monitor.get_system_health()

            return HealthResponse(
                status=system_health.get("status", "unknown"),
                uptime_seconds=uptime,
                components=system_health.get("components", {})
            )

        @self.app.get("/health/detailed", response_model=Dict[str, Any], tags=["Health"])
        async def detailed_health_check():
            """Health check détaillé"""
            return {
                "system": self.health_monitor.get_system_health(),
                "dashboard": self.dashboard.get_current_metrics().__dict__,
                "market_data": self.market_data_pipeline.get_statistics(),
                "streaming": self.streaming_service.get_statistics().__dict__,
                "events": self.event_bus.get_statistics()
            }

    def _setup_strategy_routes(self):
        """Routes pour les stratégies"""
        @self.app.post("/api/v1/strategies", response_model=APIResponse, tags=["Strategies"])
        async def create_strategy(request: CreateStrategyRequest):
            """Créer une nouvelle stratégie"""
            try:
                # Utiliser les handlers d'application via le container
                from ...application.handlers.strategy_command_handler import StrategyCommandHandler
                from ...application.commands.strategy_commands import CreateStrategyCommand

                handler = self.container.resolve(StrategyCommandHandler)
                command = CreateStrategyCommand(
                    name=request.name,
                    description=request.description,
                    parameters=request.parameters,
                    risk_limits=request.risk_limits
                )

                strategy = await handler.handle(command)

                return APIResponse(
                    data=StrategyResponse(
                        id=strategy.id,
                        name=strategy.name,
                        description=strategy.description,
                        status=strategy.status.value,
                        parameters=strategy.parameters,
                        risk_limits=strategy.risk_limits,
                        created_at=strategy.created_at,
                        updated_at=strategy.updated_at
                    ).dict(),
                    message="Strategy created successfully"
                )

            except Exception as e:
                self.logger.error(f"Error creating strategy: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/strategies", response_model=APIResponse, tags=["Strategies"])
        async def list_strategies(
            status: Optional[str] = None,
            limit: int = Query(default=100, ge=1, le=1000),
            offset: int = Query(default=0, ge=0)
        ):
            """Lister les stratégies"""
            try:
                # Simuler pour l'instant
                strategies = []
                return APIResponse(
                    data={
                        "strategies": strategies,
                        "total": len(strategies),
                        "limit": limit,
                        "offset": offset
                    }
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    def _setup_portfolio_routes(self):
        """Routes pour les portfolios"""
        @self.app.post("/api/v1/portfolios", response_model=APIResponse, tags=["Portfolios"])
        async def create_portfolio(request: CreatePortfolioRequest):
            """Créer un nouveau portfolio"""
            try:
                # Simuler pour l'instant
                portfolio_id = str(uuid.uuid4())
                return APIResponse(
                    data=PortfolioResponse(
                        id=portfolio_id,
                        name=request.name,
                        description=request.description,
                        status="active",
                        total_value=request.initial_cash,
                        cash_balance=request.initial_cash,
                        currency=request.currency,
                        positions_count=0,
                        created_at=datetime.utcnow()
                    ).dict(),
                    message="Portfolio created successfully"
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/portfolios", response_model=APIResponse, tags=["Portfolios"])
        async def list_portfolios():
            """Lister les portfolios"""
            portfolios = []
            return APIResponse(data={"portfolios": portfolios})

    def _setup_order_routes(self):
        """Routes pour les ordres"""
        @self.app.post("/api/v1/orders", response_model=APIResponse, tags=["Orders"])
        async def create_order(request: CreateOrderRequest):
            """Créer un nouvel ordre"""
            try:
                order_id = str(uuid.uuid4())
                return APIResponse(
                    data=OrderResponse(
                        id=order_id,
                        symbol=request.symbol,
                        side=request.side,
                        order_type=request.order_type,
                        quantity=request.quantity,
                        price=request.price,
                        status="pending",
                        portfolio_id=request.portfolio_id,
                        created_at=datetime.utcnow()
                    ).dict(),
                    message="Order created successfully"
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/orders", response_model=APIResponse, tags=["Orders"])
        async def list_orders(
            portfolio_id: Optional[str] = None,
            status: Optional[str] = None,
            symbol: Optional[str] = None
        ):
            """Lister les ordres"""
            orders = []
            return APIResponse(data={"orders": orders})

    def _setup_market_data_routes(self):
        """Routes pour les données de marché"""
        @self.app.post("/api/v1/market-data/subscribe", response_model=APIResponse, tags=["Market Data"])
        async def subscribe_market_data(request: MarketDataRequest):
            """S'abonner aux données de marché"""
            try:
                providers = request.providers or list(self.market_data_pipeline._providers.keys())

                for symbol in request.symbols:
                    data_types = [DataType(dt) for dt in request.data_types]
                    await self.market_data_pipeline.subscribe_symbol(symbol, data_types, providers)

                return APIResponse(
                    message=f"Subscribed to {len(request.symbols)} symbols for {len(request.data_types)} data types"
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/market-data/latest/{symbol}", response_model=APIResponse, tags=["Market Data"])
        async def get_latest_market_data(symbol: str, data_type: str):
            """Obtenir les dernières données de marché"""
            try:
                data_point = self.market_data_pipeline.get_latest_data(symbol, DataType(data_type))
                if not data_point:
                    raise HTTPException(status_code=404, detail="No data found")

                return APIResponse(
                    data=MarketDataResponse(
                        symbol=data_point.symbol,
                        data_type=data_point.data_type.value,
                        timestamp=data_point.timestamp,
                        data=data_point.data,
                        provider=data_point.provider,
                        quality=data_point.quality.value
                    ).dict()
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    def _setup_streaming_routes(self):
        """Routes pour le streaming temps réel"""
        @self.app.post("/api/v1/streaming/subscribe", response_model=APIResponse, tags=["Streaming"])
        async def create_streaming_subscription(request: StreamSubscriptionRequest):
            """Créer une subscription streaming"""
            try:
                client_id = str(uuid.uuid4())
                subscription_id = self.streaming_service.subscribe_callback(
                    client_id=client_id,
                    callback=lambda data: None,  # Placeholder
                    level=SubscriptionLevel(request.level),
                    **request.filter_criteria
                )

                return APIResponse(
                    data={
                        "subscription_id": subscription_id,
                        "client_id": client_id
                    },
                    message="Streaming subscription created"
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/streaming/subscriptions", response_model=APIResponse, tags=["Streaming"])
        async def list_streaming_subscriptions():
            """Lister les subscriptions streaming"""
            subscriptions = self.streaming_service.list_subscriptions()
            return APIResponse(data={"subscriptions": subscriptions})

    def _setup_observability_routes(self):
        """Routes pour l'observabilité"""
        @self.app.get("/api/v1/metrics", response_model=Dict[str, Any], tags=["Observability"])
        async def get_metrics():
            """Obtenir les métriques Prometheus"""
            prometheus_data = self.dashboard.export_prometheus_metrics()
            return {"prometheus": prometheus_data}

        @self.app.get("/api/v1/dashboard", response_model=APIResponse, tags=["Observability"])
        async def get_dashboard_data():
            """Obtenir les données du dashboard"""
            dashboard_data = self.dashboard.export_dashboard_data()
            return APIResponse(data=dashboard_data)

        @self.app.get("/api/v1/health-report", response_model=APIResponse, tags=["Observability"])
        async def get_health_report():
            """Obtenir le rapport de santé textuel"""
            report = self.dashboard.create_health_report()
            return APIResponse(data={"report": report})

    def _setup_event_routes(self):
        """Routes pour les événements"""
        @self.app.get("/api/v1/events/statistics", response_model=APIResponse, tags=["Events"])
        async def get_event_statistics():
            """Obtenir les statistiques des événements"""
            return APIResponse(data={
                "event_bus": self.event_bus.get_statistics(),
                "event_store": self.event_store.get_statistics().__dict__,
                "saga_manager": get_saga_manager().get_statistics(),
                "projection_manager": get_projection_manager().get_global_statistics()
            })

        @self.app.get("/api/v1/events/streams/{stream_id}", response_model=APIResponse, tags=["Events"])
        async def get_event_stream(stream_id: str):
            """Obtenir les informations d'un stream d'événements"""
            stream_info = await self.event_store.get_stream_info(stream_id)
            if not stream_info:
                raise HTTPException(status_code=404, detail="Stream not found")

            return APIResponse(data=stream_info.to_dict())

    async def start(self, host: str = "0.0.0.0", port: int = 8000):
        """Démarrer le serveur API"""
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
        server = uvicorn.Server(config)

        self.logger.info(f"Starting FastAPI server on {host}:{port}")
        await server.serve()


# Instance globale
_global_api_service: Optional[FastAPIService] = None


def get_api_service() -> Optional[FastAPIService]:
    """Obtenir l'instance globale du service API"""
    return _global_api_service


def create_api_service(container: DIContainer) -> FastAPIService:
    """Créer l'instance globale du service API"""
    global _global_api_service
    _global_api_service = FastAPIService(container)
    return _global_api_service


# Router pour modularité
class APIRouter:
    """Router pour organiser les endpoints par modules"""

    def __init__(self, prefix: str = "", tags: List[str] = None):
        self.prefix = prefix
        self.tags = tags or []
        self.routes = []

    def get(self, path: str, **kwargs):
        """Décorateur pour routes GET"""
        def decorator(func):
            self.routes.append(("GET", self.prefix + path, func, kwargs))
            return func
        return decorator

    def post(self, path: str, **kwargs):
        """Décorateur pour routes POST"""
        def decorator(func):
            self.routes.append(("POST", self.prefix + path, func, kwargs))
            return func
        return decorator

    def include_router(self, app: FastAPI):
        """Inclure ce router dans l'app FastAPI"""
        for method, path, func, kwargs in self.routes:
            if method == "GET":
                app.get(path, tags=self.tags, **kwargs)(func)
            elif method == "POST":
                app.post(path, tags=self.tags, **kwargs)(func)