"""
Infrastructure Layer: GraphQL API Service
=========================================

Service GraphQL pour requêtes complexes et efficaces des données
de trading avec support des subscriptions temps réel.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator
from enum import Enum

import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.types import Info
from strawberry.subscriptions import GRAPHQL_TRANSPORT_WS_PROTOCOL, GRAPHQL_WS_PROTOCOL

from ..observability.logging import LoggerFactory
from ..observability.metrics import get_business_metrics
from ..observability.tracing import get_tracer, trace

from ..data.market_data_pipeline import get_market_data_pipeline, DataType, DataQuality
from ..data.real_time_streaming import get_streaming_service, SubscriptionLevel

from ..events.core import get_event_bus
from ..events.event_store import get_event_store
from ..observability.dashboard import get_dashboard


# ===============================
# Types GraphQL
# ===============================

@strawberry.enum
class OrderSideEnum(Enum):
    BUY = "buy"
    SELL = "sell"


@strawberry.enum
class OrderTypeEnum(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@strawberry.enum
class OrderStatusEnum(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@strawberry.enum
class DataTypeEnum(Enum):
    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    TRADES = "trades"
    CANDLES = "candles"


@strawberry.type
class MarketData:
    """Données de marché"""
    symbol: str
    data_type: DataTypeEnum
    timestamp: datetime
    data: strawberry.scalars.JSON
    provider: str
    quality: str


@strawberry.type
class Strategy:
    """Stratégie de trading"""
    id: str
    name: str
    description: Optional[str]
    status: str
    parameters: strawberry.scalars.JSON
    risk_limits: strawberry.scalars.JSON
    created_at: datetime
    updated_at: Optional[datetime]


@strawberry.type
class Portfolio:
    """Portfolio de trading"""
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


@strawberry.type
class Order:
    """Ordre de trading"""
    id: str
    symbol: str
    side: OrderSideEnum
    order_type: OrderTypeEnum
    quantity: float
    price: Optional[float]
    status: OrderStatusEnum
    portfolio_id: str
    created_at: datetime
    updated_at: Optional[datetime]


@strawberry.type
class Position:
    """Position dans un portfolio"""
    symbol: str
    quantity: float
    average_price: float
    current_price: float
    unrealized_pnl: float
    portfolio_id: str


@strawberry.type
class DashboardMetrics:
    """Métriques du dashboard"""
    timestamp: datetime
    system_health: str
    total_trades: int
    total_volume: float
    current_pnl: float
    avg_trade_latency: float
    api_requests_per_minute: int
    error_rate: float
    current_var: float
    max_drawdown: float
    risk_breaches_count: int
    open_alerts: int
    critical_alerts: int
    healthy_components: int
    total_components: int


@strawberry.type
class SystemHealth:
    """Santé du système"""
    status: str
    timestamp: datetime
    uptime_seconds: int
    components: strawberry.scalars.JSON


@strawberry.type
class EventStatistics:
    """Statistiques des événements"""
    events_published: int
    events_processed: int
    events_failed: int
    handlers_executed: int
    handlers_failed: int
    queue_size: int
    running: bool


# ===============================
# Inputs GraphQL
# ===============================

@strawberry.input
class CreateStrategyInput:
    name: str
    description: Optional[str] = None
    parameters: strawberry.scalars.JSON = strawberry.field(default_factory=dict)
    risk_limits: strawberry.scalars.JSON = strawberry.field(default_factory=dict)


@strawberry.input
class CreatePortfolioInput:
    name: str
    description: Optional[str] = None
    initial_cash: float
    currency: str = "USD"


@strawberry.input
class CreateOrderInput:
    symbol: str
    side: OrderSideEnum
    order_type: OrderTypeEnum
    quantity: float
    price: Optional[float] = None
    portfolio_id: str


@strawberry.input
class MarketDataFilter:
    symbols: Optional[List[str]] = None
    data_types: Optional[List[DataTypeEnum]] = None
    providers: Optional[List[str]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


# ===============================
# Resolvers et Context
# ===============================

class GraphQLContext:
    """Contexte GraphQL avec services"""

    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)
        self.metrics = get_business_metrics()
        self.tracer = get_tracer()

        # Services
        self.market_data_pipeline = get_market_data_pipeline()
        self.streaming_service = get_streaming_service()
        self.dashboard = get_dashboard()
        self.event_bus = get_event_bus()
        self.event_store = get_event_store()


# ===============================
# Queries
# ===============================

@strawberry.type
class Query:
    """Requêtes GraphQL"""

    @strawberry.field
    @trace("graphql.market_data")
    async def market_data(
        self,
        info: Info,
        symbol: str,
        data_type: DataTypeEnum,
        provider: Optional[str] = None
    ) -> Optional[MarketData]:
        """Obtenir les dernières données de marché"""
        context: GraphQLContext = info.context

        data_point = context.market_data_pipeline.get_latest_data(
            symbol,
            DataType(data_type.value),
            provider
        )

        if not data_point:
            return None

        return MarketData(
            symbol=data_point.symbol,
            data_type=DataTypeEnum(data_point.data_type.value),
            timestamp=data_point.timestamp,
            data=data_point.data,
            provider=data_point.provider,
            quality=data_point.quality.value
        )

    @strawberry.field
    async def market_data_list(
        self,
        info: Info,
        filter: Optional[MarketDataFilter] = None
    ) -> List[MarketData]:
        """Lister les données de marché avec filtres"""
        # Implémentation simplifiée
        return []

    @strawberry.field
    async def strategies(
        self,
        info: Info,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Strategy]:
        """Lister les stratégies"""
        # Implémentation à connecter avec les handlers
        return []

    @strawberry.field
    async def strategy(self, info: Info, id: str) -> Optional[Strategy]:
        """Obtenir une stratégie par ID"""
        # Implémentation à connecter avec les handlers
        return None

    @strawberry.field
    async def portfolios(
        self,
        info: Info,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Portfolio]:
        """Lister les portfolios"""
        return []

    @strawberry.field
    async def portfolio(self, info: Info, id: str) -> Optional[Portfolio]:
        """Obtenir un portfolio par ID"""
        return None

    @strawberry.field
    async def orders(
        self,
        info: Info,
        portfolio_id: Optional[str] = None,
        status: Optional[OrderStatusEnum] = None,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Order]:
        """Lister les ordres"""
        return []

    @strawberry.field
    async def order(self, info: Info, id: str) -> Optional[Order]:
        """Obtenir un ordre par ID"""
        return None

    @strawberry.field
    async def positions(
        self,
        info: Info,
        portfolio_id: str
    ) -> List[Position]:
        """Obtenir les positions d'un portfolio"""
        return []

    @strawberry.field
    @trace("graphql.dashboard_metrics")
    async def dashboard_metrics(self, info: Info) -> DashboardMetrics:
        """Obtenir les métriques du dashboard"""
        context: GraphQLContext = info.context
        metrics = context.dashboard.get_current_metrics()

        return DashboardMetrics(
            timestamp=metrics.timestamp,
            system_health=metrics.system_health,
            total_trades=metrics.total_trades,
            total_volume=metrics.total_volume,
            current_pnl=metrics.current_pnl,
            avg_trade_latency=metrics.avg_trade_latency,
            api_requests_per_minute=metrics.api_requests_per_minute,
            error_rate=metrics.error_rate,
            current_var=metrics.current_var,
            max_drawdown=metrics.max_drawdown,
            risk_breaches_count=metrics.risk_breaches_count,
            open_alerts=metrics.open_alerts,
            critical_alerts=metrics.critical_alerts,
            healthy_components=metrics.healthy_components,
            total_components=metrics.total_components
        )

    @strawberry.field
    async def system_health(self, info: Info) -> SystemHealth:
        """Obtenir la santé du système"""
        from ..observability.health import get_health_monitor
        health_monitor = get_health_monitor()
        health_data = health_monitor.get_system_health()

        return SystemHealth(
            status=health_data.get("status", "unknown"),
            timestamp=datetime.utcnow(),
            uptime_seconds=0,  # À calculer
            components=health_data.get("components", {})
        )

    @strawberry.field
    async def event_statistics(self, info: Info) -> EventStatistics:
        """Obtenir les statistiques des événements"""
        context: GraphQLContext = info.context
        stats = context.event_bus.get_statistics()

        return EventStatistics(
            events_published=stats["events_published"],
            events_processed=stats["events_processed"],
            events_failed=stats["events_failed"],
            handlers_executed=stats["handlers_executed"],
            handlers_failed=stats["handlers_failed"],
            queue_size=stats["queue_size"],
            running=stats["running"]
        )


# ===============================
# Mutations
# ===============================

@strawberry.type
class Mutation:
    """Mutations GraphQL"""

    @strawberry.mutation
    async def create_strategy(
        self,
        info: Info,
        input: CreateStrategyInput
    ) -> Strategy:
        """Créer une nouvelle stratégie"""
        # Implémentation à connecter avec les command handlers
        import uuid
        strategy_id = str(uuid.uuid4())

        return Strategy(
            id=strategy_id,
            name=input.name,
            description=input.description,
            status="active",
            parameters=input.parameters,
            risk_limits=input.risk_limits,
            created_at=datetime.utcnow(),
            updated_at=None
        )

    @strawberry.mutation
    async def create_portfolio(
        self,
        info: Info,
        input: CreatePortfolioInput
    ) -> Portfolio:
        """Créer un nouveau portfolio"""
        import uuid
        portfolio_id = str(uuid.uuid4())

        return Portfolio(
            id=portfolio_id,
            name=input.name,
            description=input.description,
            status="active",
            total_value=input.initial_cash,
            cash_balance=input.initial_cash,
            currency=input.currency,
            positions_count=0,
            created_at=datetime.utcnow(),
            updated_at=None
        )

    @strawberry.mutation
    async def create_order(
        self,
        info: Info,
        input: CreateOrderInput
    ) -> Order:
        """Créer un nouvel ordre"""
        import uuid
        order_id = str(uuid.uuid4())

        return Order(
            id=order_id,
            symbol=input.symbol,
            side=input.side,
            order_type=input.order_type,
            quantity=input.quantity,
            price=input.price,
            status=OrderStatusEnum.PENDING,
            portfolio_id=input.portfolio_id,
            created_at=datetime.utcnow(),
            updated_at=None
        )


# ===============================
# Subscriptions
# ===============================

@strawberry.type
class Subscription:
    """Subscriptions GraphQL pour données temps réel"""

    @strawberry.subscription
    async def market_data_stream(
        self,
        info: Info,
        symbols: Optional[List[str]] = None,
        data_types: Optional[List[DataTypeEnum]] = None
    ) -> AsyncGenerator[MarketData, None]:
        """Stream des données de marché en temps réel"""
        context: GraphQLContext = info.context

        # Créer une subscription au streaming service
        client_id = "graphql_" + str(id(info))

        filter_criteria = {}
        if symbols:
            filter_criteria["symbols"] = symbols
        if data_types:
            filter_criteria["data_types"] = [dt.value for dt in data_types]

        subscription_id, stream = context.streaming_service.subscribe_async_generator(
            client_id=client_id,
            level=SubscriptionLevel.ALL,
            **filter_criteria
        )

        try:
            async for data_point in stream:
                yield MarketData(
                    symbol=data_point.symbol,
                    data_type=DataTypeEnum(data_point.data_type.value),
                    timestamp=data_point.timestamp,
                    data=data_point.data,
                    provider=data_point.provider,
                    quality=data_point.quality.value
                )
        finally:
            # Nettoyer la subscription
            await context.streaming_service.unsubscribe(subscription_id)

    @strawberry.subscription
    async def dashboard_metrics_stream(
        self,
        info: Info,
        interval_seconds: int = 5
    ) -> AsyncGenerator[DashboardMetrics, None]:
        """Stream des métriques du dashboard"""
        context: GraphQLContext = info.context

        while True:
            metrics = context.dashboard.get_current_metrics()

            yield DashboardMetrics(
                timestamp=metrics.timestamp,
                system_health=metrics.system_health,
                total_trades=metrics.total_trades,
                total_volume=metrics.total_volume,
                current_pnl=metrics.current_pnl,
                avg_trade_latency=metrics.avg_trade_latency,
                api_requests_per_minute=metrics.api_requests_per_minute,
                error_rate=metrics.error_rate,
                current_var=metrics.current_var,
                max_drawdown=metrics.max_drawdown,
                risk_breaches_count=metrics.risk_breaches_count,
                open_alerts=metrics.open_alerts,
                critical_alerts=metrics.critical_alerts,
                healthy_components=metrics.healthy_components,
                total_components=metrics.total_components
            )

            await asyncio.sleep(interval_seconds)

    @strawberry.subscription
    async def order_updates(
        self,
        info: Info,
        portfolio_id: Optional[str] = None
    ) -> AsyncGenerator[Order, None]:
        """Stream des mises à jour d'ordres"""
        # Implémentation via event bus
        while True:
            # Simuler pour l'instant
            await asyncio.sleep(10)
            yield Order(
                id="test",
                symbol="BTC/USD",
                side=OrderSideEnum.BUY,
                order_type=OrderTypeEnum.MARKET,
                quantity=0.1,
                price=None,
                status=OrderStatusEnum.FILLED,
                portfolio_id=portfolio_id or "default",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )


# ===============================
# Schema et Service
# ===============================

schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription
)


class GraphQLService:
    """Service GraphQL principal"""

    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)
        self.metrics = get_business_metrics()
        self.context = GraphQLContext()

    def create_router(self) -> GraphQLRouter:
        """Créer le router GraphQL pour FastAPI"""
        return GraphQLRouter(
            schema,
            context_getter=lambda: self.context,
            subscription_protocols=[
                GRAPHQL_TRANSPORT_WS_PROTOCOL,
                GRAPHQL_WS_PROTOCOL
            ]
        )

    async def execute_query(self, query: str, variables: Dict[str, Any] = None) -> Dict[str, Any]:
        """Exécuter une requête GraphQL programmatiquement"""
        result = await schema.execute(
            query,
            variable_values=variables or {},
            context_value=self.context
        )

        return {
            "data": result.data,
            "errors": [str(error) for error in result.errors] if result.errors else None
        }


# Instance globale
_global_graphql_service: Optional[GraphQLService] = None


def get_graphql_service() -> GraphQLService:
    """Obtenir l'instance globale du service GraphQL"""
    global _global_graphql_service
    if _global_graphql_service is None:
        _global_graphql_service = GraphQLService()
    return _global_graphql_service


# Requêtes prédéfinies utiles
PREDEFINED_QUERIES = {
    "dashboard_overview": """
        query DashboardOverview {
            dashboardMetrics {
                timestamp
                systemHealth
                totalTrades
                totalVolume
                currentPnl
                avgTradeLatency
                errorRate
                openAlerts
                criticalAlerts
                healthyComponents
                totalComponents
            }
            systemHealth {
                status
                timestamp
                components
            }
        }
    """,

    "market_data_overview": """
        query MarketDataOverview($symbols: [String!]!) {
            marketDataList(filter: { symbols: $symbols }) {
                symbol
                dataType
                timestamp
                data
                provider
                quality
            }
        }
    """,

    "portfolio_summary": """
        query PortfolioSummary($portfolioId: String!) {
            portfolio(id: $portfolioId) {
                id
                name
                status
                totalValue
                cashBalance
                currency
                positionsCount
            }
            positions(portfolioId: $portfolioId) {
                symbol
                quantity
                averagePrice
                currentPrice
                unrealizedPnl
            }
        }
    """
}