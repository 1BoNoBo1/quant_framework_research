"""
Service Configuration for Dependency Injection
==============================================

Configuration centralisée de tous les services pour l'architecture hexagonale.
Intègre les repositories, services domain, et adapters infrastructure.
"""

from typing import Type, Dict, Any
import logging

from ...core.container import DIContainer, LifetimeScope
from ...core.config import FrameworkConfig, get_config

# Domain interfaces
from ...domain.repositories.strategy_repository import StrategyRepository
from ...domain.repositories.risk_assessment_repository import RiskAssessmentRepository
from ...domain.repositories.portfolio_repository import PortfolioRepository
from ...domain.repositories.order_repository import OrderRepository
from ...domain.repositories.backtest_repository import BacktestRepository
from ...domain.services.signal_service import SignalService
from ...domain.services.risk_calculation_service import RiskCalculationService
from ...domain.services.portfolio_service import PortfolioService
from ...domain.services.execution_service import ExecutionService
from ...domain.services.backtesting_service import BacktestingService

# Application services (à créer)
from ...application.handlers.strategy_command_handler import StrategyCommandHandler
from ...application.handlers.signal_query_handler import SignalQueryHandler

# Risk Management Application Services
from ...application.risk_management.commands import (
    CreateRiskAssessmentHandler,
    UpdateRiskAssessmentHandler,
    AddRiskMetricHandler,
    SetRiskLimitsHandler,
    AddRiskRecommendationHandler,
    AddRiskAlertHandler,
    PerformStressTestHandler,
    ArchiveOldAssessmentsHandler
)
from ...application.risk_management.queries import (
    GetRiskAssessmentHandler,
    GetLatestRiskAssessmentHandler,
    GetRiskAssessmentsByTargetHandler,
    GetCriticalRiskAssessmentsHandler,
    GetRiskAssessmentsByLevelHandler,
    GetBreachedMetricsHandler,
    GetRiskStatisticsHandler,
    GetRiskTrendsHandler,
    SearchRiskAssessmentsHandler,
    GetRiskDashboardHandler
)

# Portfolio Management Application Services
from ...application.portfolio_management.commands import (
    CreatePortfolioHandler,
    UpdatePortfolioHandler,
    AddPositionHandler,
    RemovePositionHandler,
    AdjustCashHandler,
    SetTargetAllocationHandler,
    RebalancePortfolioHandler,
    OptimizeAllocationHandler,
    AddStrategyToPortfolioHandler,
    CreateSnapshotHandler,
    ArchivePortfolioHandler
)
from ...application.portfolio_management.queries import (
    GetPortfolioHandler,
    GetPortfolioByNameHandler,
    GetAllPortfoliosHandler,
    GetPortfoliosByStatusHandler,
    GetActivePortfoliosHandler,
    GetPortfoliosByStrategyHandler,
    GetPortfoliosNeedingRebalancingHandler,
    GetPortfolioPerformanceHandler,
    GetPortfolioStatisticsHandler,
    GetGlobalPortfolioStatisticsHandler,
    SearchPortfoliosHandler,
    GetPortfolioComparisonHandler,
    GetPortfolioRebalancingPlanHandler,
    GetPortfolioAllocationAnalysisHandler
)

# Execution Management Application Services
from ...application.execution_management.commands import (
    CreateOrderHandler,
    SubmitOrderHandler,
    ModifyOrderHandler,
    CancelOrderHandler,
    ExecuteOrderHandler,
    AddExecutionHandler,
    CreateExecutionPlanHandler,
    CreateChildOrdersHandler,
    BulkCancelOrdersHandler
)
from ...application.execution_management.queries import (
    GetOrderHandler,
    GetOrderByClientIdHandler,
    GetOrdersByStatusHandler,
    GetActiveOrdersHandler,
    GetOrdersBySymbolHandler,
    GetOrdersByPortfolioHandler,
    GetParentOrdersHandler,
    GetChildOrdersHandler,
    GetExecutionReportHandler,
    GetOrderStatisticsHandler,
    GetExecutionStatisticsHandler,
    SearchOrdersHandler,
    GetOrderBookHandler,
    GetExecutionProgressHandler
)

# Backtesting Application Services
from ...application.backtesting.commands import (
    CreateBacktestConfigurationHandler,
    UpdateBacktestConfigurationHandler,
    DeleteBacktestConfigurationHandler,
    RunBacktestHandler,
    StopBacktestHandler,
    DeleteBacktestResultHandler,
    ArchiveBacktestResultHandler,
    RestoreBacktestResultHandler,
    CleanupOldBacktestResultsHandler,
    ExportBacktestResultsHandler,
    ImportBacktestResultsHandler
)
from ...application.backtesting.queries import (
    GetBacktestConfigurationHandler,
    GetAllBacktestConfigurationsHandler,
    FindBacktestConfigurationsByNameHandler,
    GetBacktestResultHandler,
    FindBestPerformingBacktestsHandler,
    GetBacktestPerformanceComparisonHandler,
    GetBacktestStatisticsHandler,
    GetBacktestDashboardHandler
)

# Infrastructure implementations (à créer)
from ..persistence.memory_strategy_repository import MemoryStrategyRepository
from ..persistence.postgres_strategy_repository import PostgresStrategyRepository
from ..external.market_data_service import MarketDataService
from ..external.broker_service import BrokerService

# Execution Infrastructure
from ..persistence.memory_order_repository import MemoryOrderRepository
from ..persistence.postgres_order_repository import PostgresOrderRepository
from ..external.order_execution_adapter import OrderExecutionAdapter
from ..external.mock_broker_adapter import MockBrokerAdapter

# Backtesting Infrastructure
from ..persistence.memory_backtest_repository import MemoryBacktestRepository
from ..analytics.performance_calculator import PerformanceCalculator

# Observability Infrastructure
from ..observability.logging import LoggerFactory, LogContext
from ..observability.metrics import MetricsCollector, BusinessMetrics
from ..observability.tracing import TradingTracer
from ..observability.health import HealthMonitor
from ..observability.alerting import AlertManager
from ..observability.dashboard import ObservabilityDashboard

# Data Pipeline Infrastructure
from ..data.market_data_pipeline import MarketDataPipeline, MockDataProvider
from ..data.binance_provider import BinanceProvider
from ..data.coinbase_provider import CoinbaseProvider
from ..data.ccxt_provider import CCXTProvider, CCXTProviderFactory
from ..data.real_time_streaming import RealTimeStreamingService

# Event-Driven Architecture Infrastructure
from ..events.core import EventBus, get_event_bus
from ..events.event_store import EventStore, InMemoryEventStore, get_event_store
from ..events.saga import SagaManager, get_saga_manager
from ..events.projections import ProjectionManager, get_projection_manager

# API Services Infrastructure
from ..api.rest import FastAPIService, create_api_service
from ..api.websocket import WebSocketManager, get_websocket_manager
from ..api.auth import AuthService, get_auth_service
# from ..api.graphql import GraphQLService, get_graphql_service  # Disabled due to Python 3.13 compatibility

logger = logging.getLogger(__name__)


class ServiceConfiguration:
    """
    Configuration centralisée des services pour l'injection de dépendances.

    Organise l'enregistrement des services selon l'architecture hexagonale:
    - Domain services
    - Application handlers
    - Infrastructure adapters
    """

    def __init__(self, container: DIContainer, config: FrameworkConfig):
        self.container = container
        self.config = config

    def configure_all_services(self) -> None:
        """Configure tous les services du framework."""
        logger.info("🔧 Configuration des services DI...")

        # Configuration par ordre de dépendance
        self._configure_core_services()
        self._configure_domain_services()
        self._configure_infrastructure_services()
        self._configure_application_services()
        self._configure_presentation_services()

        logger.info("✅ Configuration DI terminée")

    def _configure_core_services(self) -> None:
        """Configure les services core (configuration, logging, etc.)."""

        # Configuration framework
        self.container.register_singleton(
            FrameworkConfig,
            factory=lambda: self.config
        )

        # Logger factory
        self.container.register_transient(
            logging.Logger,
            factory=lambda: logging.getLogger(__name__)
        )

    def _configure_domain_services(self) -> None:
        """Configure les services du domaine."""

        # Domain services - Singletons car ils n'ont pas d'état
        self.container.register_singleton(SignalService, SignalService)
        self.container.register_singleton(RiskCalculationService, RiskCalculationService)
        self.container.register_singleton(PortfolioService, PortfolioService)
        self.container.register_singleton(ExecutionService, ExecutionService)
        self.container.register_singleton(BacktestingService, BacktestingService)

    def _configure_infrastructure_services(self) -> None:
        """Configure les adapters infrastructure."""

        # Choix du repository selon la configuration
        if self.config.environment.value == "testing":
            # En mode test: repository en mémoire
            self.container.register_singleton(
                StrategyRepository,
                MemoryStrategyRepository
            )
            # Risk Assessment Repository en mémoire pour tests
            from ..persistence.memory_risk_assessment_repository import MemoryRiskAssessmentRepository
            self.container.register_singleton(
                RiskAssessmentRepository,
                MemoryRiskAssessmentRepository
            )
            # Portfolio Repository en mémoire pour tests
            from ..persistence.memory_portfolio_repository import MemoryPortfolioRepository
            self.container.register_singleton(
                PortfolioRepository,
                MemoryPortfolioRepository
            )
            # Order Repository en mémoire pour tests
            self.container.register_singleton(
                OrderRepository,
                MemoryOrderRepository
            )
            # Backtest Repository en mémoire pour tests
            self.container.register_singleton(
                BacktestRepository,
                MemoryBacktestRepository
            )
        else:
            # En production: repository PostgreSQL
            self.container.register_singleton(
                StrategyRepository,
                PostgresStrategyRepository
            )
            # Risk Assessment Repository - Use memory for now (PostgreSQL implementation TODO)
            from ..persistence.memory_risk_assessment_repository import MemoryRiskAssessmentRepository
            self.container.register_singleton(
                RiskAssessmentRepository,
                MemoryRiskAssessmentRepository
            )
            # Portfolio Repository - Use memory for now (PostgreSQL implementation TODO)
            from ..persistence.memory_portfolio_repository import MemoryPortfolioRepository
            self.container.register_singleton(
                PortfolioRepository,
                MemoryPortfolioRepository
            )
            # Order Repository - Use memory for now (PostgreSQL implementation TODO)
            self.container.register_singleton(
                OrderRepository,
                MemoryOrderRepository
            )
            # Backtest Repository PostgreSQL pour production (TODO: créer l'implémentation)
            self.container.register_singleton(
                BacktestRepository,
                MemoryBacktestRepository  # Temporaire: utiliser memory même en prod
            )

        # Services externes
        self.container.register_singleton(MarketDataService, MarketDataService)
        self.container.register_singleton(BrokerService, BrokerService)

        # Order Execution Adapter avec brokers configurés
        self.container.register_singleton(
            OrderExecutionAdapter,
            factory=self._create_order_execution_adapter
        )

        # Performance Calculator pour backtesting
        self.container.register_singleton(PerformanceCalculator, PerformanceCalculator)

        # Observability services
        self._configure_observability_services()

        # Data pipeline and market data services
        self._configure_data_pipeline_services()

        # Event-driven architecture services
        self._configure_event_driven_services()

        # API services
        self._configure_api_services()

        # Persistence services (database, cache, timeseries, migrations)
        self._configure_persistence_services()

        # Data providers selon configuration
        self._configure_data_providers()

        # Autres adapters infrastructure
        # self.container.register_singleton(EmailService, EmailService)
        # self.container.register_singleton(SlackService, SlackService)

    def _configure_application_services(self) -> None:
        """Configure les handlers et use cases de l'application."""

        # Command handlers - Transient for simplicity (change to scoped if needed)
        self.container.register_transient(
            StrategyCommandHandler,
            StrategyCommandHandler
        )

        # Risk Management Command Handlers
        self.container.register_scoped(CreateRiskAssessmentHandler, CreateRiskAssessmentHandler)
        self.container.register_scoped(UpdateRiskAssessmentHandler, UpdateRiskAssessmentHandler)
        self.container.register_scoped(AddRiskMetricHandler, AddRiskMetricHandler)
        self.container.register_scoped(SetRiskLimitsHandler, SetRiskLimitsHandler)
        self.container.register_scoped(AddRiskRecommendationHandler, AddRiskRecommendationHandler)
        self.container.register_scoped(AddRiskAlertHandler, AddRiskAlertHandler)
        self.container.register_scoped(PerformStressTestHandler, PerformStressTestHandler)
        self.container.register_scoped(ArchiveOldAssessmentsHandler, ArchiveOldAssessmentsHandler)

        # Portfolio Management Command Handlers
        self.container.register_scoped(CreatePortfolioHandler, CreatePortfolioHandler)
        self.container.register_scoped(UpdatePortfolioHandler, UpdatePortfolioHandler)
        self.container.register_scoped(AddPositionHandler, AddPositionHandler)
        self.container.register_scoped(RemovePositionHandler, RemovePositionHandler)
        self.container.register_scoped(AdjustCashHandler, AdjustCashHandler)
        self.container.register_scoped(SetTargetAllocationHandler, SetTargetAllocationHandler)
        self.container.register_scoped(RebalancePortfolioHandler, RebalancePortfolioHandler)
        self.container.register_scoped(OptimizeAllocationHandler, OptimizeAllocationHandler)
        self.container.register_scoped(AddStrategyToPortfolioHandler, AddStrategyToPortfolioHandler)
        self.container.register_scoped(CreateSnapshotHandler, CreateSnapshotHandler)
        self.container.register_scoped(ArchivePortfolioHandler, ArchivePortfolioHandler)

        # Execution Management Command Handlers
        self.container.register_scoped(CreateOrderHandler, CreateOrderHandler)
        self.container.register_scoped(SubmitOrderHandler, SubmitOrderHandler)
        self.container.register_scoped(ModifyOrderHandler, ModifyOrderHandler)
        self.container.register_scoped(CancelOrderHandler, CancelOrderHandler)
        self.container.register_scoped(ExecuteOrderHandler, ExecuteOrderHandler)
        self.container.register_scoped(AddExecutionHandler, AddExecutionHandler)
        self.container.register_scoped(CreateExecutionPlanHandler, CreateExecutionPlanHandler)
        self.container.register_scoped(CreateChildOrdersHandler, CreateChildOrdersHandler)
        self.container.register_scoped(BulkCancelOrdersHandler, BulkCancelOrdersHandler)

        # Backtesting Command Handlers
        self.container.register_scoped(CreateBacktestConfigurationHandler, CreateBacktestConfigurationHandler)
        self.container.register_scoped(UpdateBacktestConfigurationHandler, UpdateBacktestConfigurationHandler)
        self.container.register_scoped(DeleteBacktestConfigurationHandler, DeleteBacktestConfigurationHandler)
        self.container.register_scoped(RunBacktestHandler, RunBacktestHandler)
        self.container.register_scoped(StopBacktestHandler, StopBacktestHandler)
        self.container.register_scoped(DeleteBacktestResultHandler, DeleteBacktestResultHandler)
        self.container.register_scoped(ArchiveBacktestResultHandler, ArchiveBacktestResultHandler)
        self.container.register_scoped(RestoreBacktestResultHandler, RestoreBacktestResultHandler)
        self.container.register_scoped(CleanupOldBacktestResultsHandler, CleanupOldBacktestResultsHandler)
        self.container.register_scoped(ExportBacktestResultsHandler, ExportBacktestResultsHandler)
        self.container.register_scoped(ImportBacktestResultsHandler, ImportBacktestResultsHandler)

        # Query handlers - Transient car stateless
        self.container.register_transient(
            SignalQueryHandler,
            SignalQueryHandler
        )

        # Risk Management Query Handlers
        self.container.register_transient(GetRiskAssessmentHandler, GetRiskAssessmentHandler)
        self.container.register_transient(GetLatestRiskAssessmentHandler, GetLatestRiskAssessmentHandler)
        self.container.register_transient(GetRiskAssessmentsByTargetHandler, GetRiskAssessmentsByTargetHandler)
        self.container.register_transient(GetCriticalRiskAssessmentsHandler, GetCriticalRiskAssessmentsHandler)
        self.container.register_transient(GetRiskAssessmentsByLevelHandler, GetRiskAssessmentsByLevelHandler)
        self.container.register_transient(GetBreachedMetricsHandler, GetBreachedMetricsHandler)
        self.container.register_transient(GetRiskStatisticsHandler, GetRiskStatisticsHandler)
        self.container.register_transient(GetRiskTrendsHandler, GetRiskTrendsHandler)
        self.container.register_transient(SearchRiskAssessmentsHandler, SearchRiskAssessmentsHandler)
        self.container.register_transient(GetRiskDashboardHandler, GetRiskDashboardHandler)

        # Portfolio Management Query Handlers
        self.container.register_transient(GetPortfolioHandler, GetPortfolioHandler)
        self.container.register_transient(GetPortfolioByNameHandler, GetPortfolioByNameHandler)
        self.container.register_transient(GetAllPortfoliosHandler, GetAllPortfoliosHandler)
        self.container.register_transient(GetPortfoliosByStatusHandler, GetPortfoliosByStatusHandler)
        self.container.register_transient(GetActivePortfoliosHandler, GetActivePortfoliosHandler)
        self.container.register_transient(GetPortfoliosByStrategyHandler, GetPortfoliosByStrategyHandler)
        self.container.register_transient(GetPortfoliosNeedingRebalancingHandler, GetPortfoliosNeedingRebalancingHandler)
        self.container.register_transient(GetPortfolioPerformanceHandler, GetPortfolioPerformanceHandler)
        self.container.register_transient(GetPortfolioStatisticsHandler, GetPortfolioStatisticsHandler)
        self.container.register_transient(GetGlobalPortfolioStatisticsHandler, GetGlobalPortfolioStatisticsHandler)
        self.container.register_transient(SearchPortfoliosHandler, SearchPortfoliosHandler)
        self.container.register_transient(GetPortfolioComparisonHandler, GetPortfolioComparisonHandler)
        self.container.register_transient(GetPortfolioRebalancingPlanHandler, GetPortfolioRebalancingPlanHandler)
        self.container.register_transient(GetPortfolioAllocationAnalysisHandler, GetPortfolioAllocationAnalysisHandler)

        # Execution Management Query Handlers
        self.container.register_transient(GetOrderHandler, GetOrderHandler)
        self.container.register_transient(GetOrderByClientIdHandler, GetOrderByClientIdHandler)
        self.container.register_transient(GetOrdersByStatusHandler, GetOrdersByStatusHandler)
        self.container.register_transient(GetActiveOrdersHandler, GetActiveOrdersHandler)
        self.container.register_transient(GetOrdersBySymbolHandler, GetOrdersBySymbolHandler)
        self.container.register_transient(GetOrdersByPortfolioHandler, GetOrdersByPortfolioHandler)
        self.container.register_transient(GetParentOrdersHandler, GetParentOrdersHandler)
        self.container.register_transient(GetChildOrdersHandler, GetChildOrdersHandler)
        self.container.register_transient(GetExecutionReportHandler, GetExecutionReportHandler)
        self.container.register_transient(GetOrderStatisticsHandler, GetOrderStatisticsHandler)
        self.container.register_transient(GetExecutionStatisticsHandler, GetExecutionStatisticsHandler)
        self.container.register_transient(SearchOrdersHandler, SearchOrdersHandler)
        self.container.register_transient(GetOrderBookHandler, GetOrderBookHandler)
        self.container.register_transient(GetExecutionProgressHandler, GetExecutionProgressHandler)

        # Backtesting Query Handlers
        self.container.register_transient(GetBacktestConfigurationHandler, GetBacktestConfigurationHandler)
        self.container.register_transient(GetAllBacktestConfigurationsHandler, GetAllBacktestConfigurationsHandler)
        self.container.register_transient(FindBacktestConfigurationsByNameHandler, FindBacktestConfigurationsByNameHandler)
        self.container.register_transient(GetBacktestResultHandler, GetBacktestResultHandler)
        self.container.register_transient(FindBestPerformingBacktestsHandler, FindBestPerformingBacktestsHandler)
        self.container.register_transient(GetBacktestPerformanceComparisonHandler, GetBacktestPerformanceComparisonHandler)
        self.container.register_transient(GetBacktestStatisticsHandler, GetBacktestStatisticsHandler)
        self.container.register_transient(GetBacktestDashboardHandler, GetBacktestDashboardHandler)

        # Autres handlers...
        # self.container.register_scoped(PortfolioCommandHandler)
        # self.container.register_transient(PerformanceQueryHandler)

    def _configure_presentation_services(self) -> None:
        """Configure les services de présentation (CLI, API, Web)."""

        # Ces services seront configurés quand on créera les controllers
        pass

    def _configure_data_providers(self) -> None:
        """Configure les fournisseurs de données selon la configuration."""

        for provider_name, provider_config in self.config.data_providers.items():

            if provider_name == "binance":
                # Factory pour Binance avec configuration
                self.container.register_singleton(
                    f"DataProvider_{provider_name}",
                    factory=lambda: self._create_binance_provider(provider_config.dict())
                )

            elif provider_name == "yfinance":
                # Factory pour YFinance
                self.container.register_singleton(
                    f"DataProvider_{provider_name}",
                    factory=lambda: self._create_yfinance_provider(provider_config.dict())
                )

            elif provider_name.startswith("ccxt_"):
                # CCXT exchanges (ccxt_binance, ccxt_okx, ccxt_bybit, etc.)
                exchange_name = provider_name.replace("ccxt_", "")
                self.container.register_singleton(
                    f"DataProvider_{provider_name}",
                    factory=lambda: self._create_ccxt_provider(exchange_name, provider_config.dict())
                )

    def _create_binance_provider(self, config: Dict[str, Any]):
        """Factory pour créer un provider Binance configuré."""
        # Import local pour éviter les dépendances circulaires
        from ...data.providers.ccxt_provider import CCXTProvider

        return CCXTProvider(
            exchange_name="binance",
            api_key=config.get("api_key"),
            api_secret=config.get("api_secret"),
            testnet=True  # Sécurité par défaut
        )

    def _create_yfinance_provider(self, config: Dict[str, Any]):
        """Factory pour créer un provider YFinance configuré."""
        # Import local pour éviter les dépendances circulaires
        from ...data.providers.yfinance_provider import YFinanceProvider

        return YFinanceProvider()

    def _create_ccxt_provider(self, exchange_name: str, config: Dict[str, Any]):
        """Factory pour créer un provider CCXT configuré."""
        return CCXTProvider(
            exchange_name=exchange_name,
            api_key=config.get("api_key"),
            secret=config.get("secret"),
            password=config.get("password"),  # Pour certains exchanges
            sandbox=config.get("sandbox", True),  # Sécurité par défaut
            rate_limit=config.get("rate_limit", True),
            options=config.get("options", {})
        )

    def _create_order_execution_adapter(self) -> OrderExecutionAdapter:
        """Factory pour créer l'adaptateur d'exécution avec brokers configurés."""
        adapter = OrderExecutionAdapter()

        # Enregistrer des brokers mock pour le développement/tests
        if self.config.environment.value in ["development", "testing"]:
            # Mock brokers pour différentes venues
            venues_to_configure = [
                "binance",
                "coinbase",
                "kraken",
                "alpaca",
                "interactive_brokers"
            ]

            for venue in venues_to_configure:
                mock_broker = MockBrokerAdapter(
                    venue=venue,
                    base_latency_ms=50 if venue in ["binance", "coinbase"] else 100,
                    fill_probability=0.98 if venue in ["binance", "coinbase"] else 0.95,
                    price_slippage_bps=2.0 if venue == "binance" else 5.0
                )
                adapter.register_broker(venue, mock_broker)

        # En production, on enregistrerait de vrais adaptateurs de courtier
        # else:
        #     adapter.register_broker("binance", RealBinanceAdapter(...))
        #     adapter.register_broker("coinbase", RealCoinbaseAdapter(...))
        #     etc.

        logger.info(f"🏗️ OrderExecutionAdapter créé avec {len(adapter.get_supported_venues())} venues")
        return adapter

    def _configure_observability_services(self) -> None:
        """Configure les services d'observabilité"""

        # Logger Factory avec contexte environnement
        log_context = LogContext(
            environment=self.config.environment.value,
            service_name="qframe",
            service_version="1.0.0"
        )

        LoggerFactory.configure_defaults(
            level="DEBUG" if self.config.environment.value == "development" else "INFO",
            format="console" if self.config.environment.value == "development" else "json",
            context=log_context
        )

        # Metrics Collector - Singleton global
        from ..observability.metrics import get_metrics_collector, get_business_metrics
        self.container.register_singleton(
            MetricsCollector,
            factory=lambda: get_metrics_collector()
        )

        self.container.register_singleton(
            BusinessMetrics,
            factory=lambda: get_business_metrics()
        )

        # Tracer - Singleton global
        from ..observability.tracing import get_tracer
        self.container.register_singleton(
            TradingTracer,
            factory=lambda: get_tracer()
        )

        # Health Monitor - Singleton avec auto-start
        from ..observability.health import get_health_monitor
        health_monitor = get_health_monitor()

        # Enregistrer les health checks par défaut
        if self.config.environment.value != "testing":
            health_monitor.start()  # Démarrer le monitoring

        self.container.register_singleton(
            HealthMonitor,
            factory=lambda: health_monitor
        )

        # Alert Manager - Singleton global avec règles par défaut
        from ..observability.alerting import get_alert_manager
        alert_manager = get_alert_manager()
        alert_manager.register_trading_rules()  # Enregistrer les règles de trading

        self.container.register_singleton(
            AlertManager,
            factory=lambda: alert_manager
        )

        # Dashboard - Singleton global
        from ..observability.dashboard import get_dashboard
        dashboard = get_dashboard()

        # Démarrer la capture automatique de snapshots en développement
        if self.config.environment.value == "development":
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(dashboard.start_auto_snapshot(interval_seconds=30))
            except RuntimeError:
                pass  # Pas de loop, on skip

        self.container.register_singleton(
            ObservabilityDashboard,
            factory=lambda: dashboard
        )

        logger.info("📊 Observability services configured successfully")

    def _configure_data_pipeline_services(self) -> None:
        """Configure les services de pipeline de données"""

        # Market Data Pipeline - Singleton global
        from ..data.market_data_pipeline import get_market_data_pipeline
        pipeline = get_market_data_pipeline()

        # Enregistrer des providers selon l'environnement
        if self.config.environment.value == "testing":
            # Mock provider pour les tests
            mock_provider = MockDataProvider("test_mock")
            pipeline.register_provider(mock_provider)

        elif self.config.environment.value == "development":
            # Mock + quelques providers réels pour le développement
            mock_provider = MockDataProvider("dev_mock")
            pipeline.register_provider(mock_provider)

            # Binance testnet
            binance_provider = BinanceProvider(testnet=True, market_type="spot")
            pipeline.register_provider(binance_provider)

            # Coinbase sandbox
            coinbase_provider = CoinbaseProvider(sandbox=True)
            pipeline.register_provider(coinbase_provider)

            # CCXT providers pour plus d'exchanges
            # OKX sandbox
            okx_provider = CCXTProviderFactory.create_provider("okx", sandbox=True)
            pipeline.register_provider(okx_provider)

            # Bybit sandbox
            bybit_provider = CCXTProviderFactory.create_provider("bybit", sandbox=True)
            pipeline.register_provider(bybit_provider)

        else:
            # Production: providers réels
            # Binance spot
            binance_spot = BinanceProvider(testnet=False, market_type="spot")
            pipeline.register_provider(binance_spot)

            # Binance futures
            binance_futures = BinanceProvider(testnet=False, market_type="futures")
            pipeline.register_provider(binance_futures)

            # Coinbase Pro
            coinbase_provider = CoinbaseProvider(sandbox=False)
            pipeline.register_provider(coinbase_provider)

            # CCXT providers production
            # OKX live
            okx_provider = CCXTProviderFactory.create_provider("okx", sandbox=False)
            pipeline.register_provider(okx_provider)

            # Bybit live
            bybit_provider = CCXTProviderFactory.create_provider("bybit", sandbox=False)
            pipeline.register_provider(bybit_provider)

            # Kraken (pas de sandbox, donc toujours live)
            kraken_provider = CCXTProviderFactory.create_provider("kraken", sandbox=False)
            pipeline.register_provider(kraken_provider)

        # Enregistrer le pipeline dans le container
        self.container.register_singleton(
            MarketDataPipeline,
            factory=lambda: pipeline
        )

        # Démarrer le pipeline automatiquement
        if self.config.environment.value != "testing":
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(pipeline.start())
            except RuntimeError:
                pass  # Pas de loop, on démarrera plus tard

        # Real-Time Streaming Service - Singleton global
        from ..data.real_time_streaming import get_streaming_service
        streaming_service = get_streaming_service()

        self.container.register_singleton(
            RealTimeStreamingService,
            factory=lambda: streaming_service
        )

        # Démarrer le service de streaming
        if self.config.environment.value != "testing":
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(streaming_service.start())
            except RuntimeError:
                pass  # Pas de loop, on démarrera plus tard

        logger.info("📈 Market data pipeline and streaming configured successfully")

    def _configure_event_driven_services(self) -> None:
        """Configure les services d'architecture événementielle"""

        # Event Bus - Singleton global
        event_bus = get_event_bus()
        self.container.register_singleton(
            EventBus,
            factory=lambda: event_bus
        )

        # Event Store - Singleton global
        event_store = get_event_store()
        self.container.register_singleton(
            EventStore,
            factory=lambda: event_store
        )

        # Saga Manager - Singleton global
        saga_manager = get_saga_manager()
        self.container.register_singleton(
            SagaManager,
            factory=lambda: saga_manager
        )

        # Projection Manager - Singleton global
        projection_manager = get_projection_manager()
        self.container.register_singleton(
            ProjectionManager,
            factory=lambda: projection_manager
        )

        # Démarrer les services automatiquement
        if self.config.environment.value != "testing":
            import asyncio
            try:
                loop = asyncio.get_event_loop()

                # Démarrer l'event bus
                loop.create_task(event_bus.start())

                # Démarrer le gestionnaire de projections
                loop.create_task(projection_manager.start_all())

                logger.info("Started event-driven services")

            except RuntimeError:
                pass  # Pas de loop, on démarrera plus tard

        logger.info("⚡ Event-driven architecture services configured successfully")

    def _configure_api_services(self) -> None:
        """Configure les services API"""

        # Authentication Service - Singleton global
        auth_service = get_auth_service()
        self.container.register_singleton(
            AuthService,
            factory=lambda: auth_service
        )

        # WebSocket Manager - Singleton global
        websocket_manager = get_websocket_manager()
        self.container.register_singleton(
            WebSocketManager,
            factory=lambda: websocket_manager
        )

        # # GraphQL Service - Singleton global (Disabled due to Python 3.13 compatibility)
        # graphql_service = get_graphql_service()
        # self.container.register_singleton(
        #     GraphQLService,
        #     factory=lambda: graphql_service
        # )

        # FastAPI Service - Singleton avec container
        api_service = create_api_service(self.container)
        self.container.register_singleton(
            FastAPIService,
            factory=lambda: api_service
        )

        # Démarrer les services automatiquement
        if self.config.environment.value != "testing":
            import asyncio
            try:
                loop = asyncio.get_event_loop()

                # Démarrer le WebSocket manager
                loop.create_task(websocket_manager.start())

                logger.info("Started API services")

            except RuntimeError:
                pass  # Pas de loop, on démarrera plus tard

        logger.info("🌐 API services configured successfully")

    def _configure_persistence_services(self) -> None:
        """Configure les services de persistence avancée"""
        from pathlib import Path
        from ..persistence import (
            DatabaseManager,
            DatabaseConfig,
            CacheManager,
            CacheConfig,
            TimeSeriesDB,
            InfluxDBConfig,
            MigrationManager,
            get_database_manager,
            get_cache_manager,
            get_timeseries_db,
            get_migration_manager,
            create_database_manager,
            create_cache_manager,
            create_timeseries_db,
            create_migration_manager
        )

        # Database Manager - Singleton global
        db_config = DatabaseConfig(
            host=self.config.database_host if hasattr(self.config, 'database_host') else "localhost",
            port=self.config.database_port if hasattr(self.config, 'database_port') else 5432,
            database=self.config.database_name if hasattr(self.config, 'database_name') else "qframe",
            user=self.config.database_user if hasattr(self.config, 'database_user') else "qframe",
            password=self.config.database_password if hasattr(self.config, 'database_password') else "qframe",
            min_connections=5,
            max_connections=20
        )

        db_manager = get_database_manager()
        if not db_manager:
            db_manager = create_database_manager(db_config)

        self.container.register_singleton(
            DatabaseManager,
            factory=lambda: db_manager
        )

        # Cache Manager - Singleton global
        cache_config = CacheConfig(
            redis_host=self.config.redis.host if hasattr(self.config, 'redis') else "localhost",
            redis_port=self.config.redis.port if hasattr(self.config, 'redis') else 6379,
            default_ttl=3600,
            max_memory_mb=512
        )

        cache_manager = get_cache_manager()
        if not cache_manager:
            cache_manager = create_cache_manager(cache_config)

        self.container.register_singleton(
            CacheManager,
            factory=lambda: cache_manager
        )

        # Time-series Database - Singleton global
        if self.config.environment.value != "testing":
            influx_config = InfluxDBConfig(
                url=self.config.influxdb_url if hasattr(self.config, 'influxdb_url') else "http://localhost:8086",
                token=self.config.influxdb_token if hasattr(self.config, 'influxdb_token') else "",
                org=self.config.influxdb_org if hasattr(self.config, 'influxdb_org') else "qframe",
                bucket=self.config.influxdb_bucket if hasattr(self.config, 'influxdb_bucket') else "market_data"
            )

            timeseries_db = get_timeseries_db()
            if not timeseries_db:
                timeseries_db = create_timeseries_db(influx_config)

            self.container.register_singleton(
                TimeSeriesDB,
                factory=lambda: timeseries_db
            )

        # Migration Manager - Singleton global
        migrations_path = Path(__file__).parent.parent / "persistence" / "migrations"
        migration_manager = get_migration_manager()
        if not migration_manager and db_manager:
            migration_manager = create_migration_manager(db_manager, migrations_path)

        if migration_manager:
            self.container.register_singleton(
                MigrationManager,
                factory=lambda: migration_manager
            )

        # Démarrer les services automatiquement
        if self.config.environment.value != "testing":
            import asyncio
            try:
                loop = asyncio.get_event_loop()

                # Initialiser le database manager
                if db_manager and hasattr(db_manager, '_initialized') and not db_manager._initialized:
                    loop.create_task(db_manager.initialize())

                # Initialiser le cache manager
                if cache_manager and hasattr(cache_manager, '_initialized') and not cache_manager._initialized:
                    loop.create_task(cache_manager.initialize())

                # Initialiser la time-series DB
                if timeseries_db and hasattr(timeseries_db, '_initialized') and not timeseries_db._initialized:
                    loop.create_task(timeseries_db.initialize())

                # Initialiser le migration manager
                if migration_manager:
                    loop.create_task(migration_manager.initialize())

                logger.info("Started persistence services")

            except RuntimeError:
                pass  # Pas de loop, on démarrera plus tard

        logger.info("💾 Persistence services configured successfully")


class ServiceModule:
    """
    Module de services pour organisation modulaire.
    Permet d'organiser les services par domaine fonctionnel.
    """

    def __init__(self, name: str):
        self.name = name
        self.services: Dict[Type, Dict[str, Any]] = {}

    def add_singleton(self, interface: Type, implementation: Type = None, factory=None):
        """Ajoute un service singleton au module."""
        self.services[interface] = {
            "implementation": implementation or interface,
            "lifetime": LifetimeScope.SINGLETON,
            "factory": factory
        }
        return self

    def add_transient(self, interface: Type, implementation: Type = None, factory=None):
        """Ajoute un service transient au module."""
        self.services[interface] = {
            "implementation": implementation or interface,
            "lifetime": LifetimeScope.TRANSIENT,
            "factory": factory
        }
        return self

    def add_scoped(self, interface: Type, implementation: Type = None, factory=None):
        """Ajoute un service scoped au module."""
        self.services[interface] = {
            "implementation": implementation or interface,
            "lifetime": LifetimeScope.SCOPED,
            "factory": factory
        }
        return self

    def register_to_container(self, container: DIContainer):
        """Enregistre tous les services du module dans le container."""
        logger.info(f"📦 Enregistrement du module: {self.name}")

        for interface, config in self.services.items():
            lifetime = config["lifetime"]
            implementation = config["implementation"]
            factory = config.get("factory")

            if lifetime == LifetimeScope.SINGLETON:
                container.register_singleton(interface, implementation, factory)
            elif lifetime == LifetimeScope.SCOPED:
                container.register_scoped(interface, implementation, factory)
            else:  # TRANSIENT
                container.register_transient(interface, implementation, factory)

        logger.info(f"✅ Module {self.name}: {len(self.services)} services enregistrés")


def create_trading_module() -> ServiceModule:
    """Crée le module des services de trading."""
    module = ServiceModule("Trading")

    # Services de trading
    module.add_singleton(SignalService)
    # module.add_singleton(PositionService)
    # module.add_singleton(OrderService)

    return module


def create_data_module() -> ServiceModule:
    """Crée le module des services de données."""
    module = ServiceModule("Data")

    # Services de données
    # module.add_singleton(DataValidationService)
    # module.add_transient(DataTransformationService)

    return module


def create_risk_module() -> ServiceModule:
    """Crée le module des services de risque."""
    module = ServiceModule("Risk")

    # Domain Services de risque
    module.add_singleton(RiskCalculationService)

    # Command Handlers de risque
    module.add_scoped(CreateRiskAssessmentHandler)
    module.add_scoped(UpdateRiskAssessmentHandler)
    module.add_scoped(AddRiskMetricHandler)
    module.add_scoped(SetRiskLimitsHandler)
    module.add_scoped(AddRiskRecommendationHandler)
    module.add_scoped(AddRiskAlertHandler)
    module.add_scoped(PerformStressTestHandler)
    module.add_scoped(ArchiveOldAssessmentsHandler)

    # Query Handlers de risque
    module.add_transient(GetRiskAssessmentHandler)
    module.add_transient(GetLatestRiskAssessmentHandler)
    module.add_transient(GetRiskAssessmentsByTargetHandler)
    module.add_transient(GetCriticalRiskAssessmentsHandler)
    module.add_transient(GetRiskAssessmentsByLevelHandler)
    module.add_transient(GetBreachedMetricsHandler)
    module.add_transient(GetRiskStatisticsHandler)
    module.add_transient(GetRiskTrendsHandler)
    module.add_transient(SearchRiskAssessmentsHandler)
    module.add_transient(GetRiskDashboardHandler)

    return module


def configure_production_services(container: DIContainer) -> None:
    """Configuration complète pour l'environnement de production."""
    config = get_config()
    service_config = ServiceConfiguration(container, config)

    # Configuration complète
    service_config.configure_all_services()

    # Modules additionnels
    trading_module = create_trading_module()
    trading_module.register_to_container(container)

    data_module = create_data_module()
    data_module.register_to_container(container)

    risk_module = create_risk_module()
    risk_module.register_to_container(container)

    logger.info("🚀 Configuration de production terminée")


def configure_testing_services(container: DIContainer) -> None:
    """Configuration simplifiée pour les tests."""
    config = get_config()
    service_config = ServiceConfiguration(container, config)

    # Configuration de base seulement
    service_config._configure_core_services()
    service_config._configure_domain_services()

    # Repository en mémoire pour les tests
    container.register_singleton(
        StrategyRepository,
        MemoryStrategyRepository
    )

    logger.info("🧪 Configuration de test terminée")


def get_service_statistics(container: DIContainer) -> Dict[str, Any]:
    """Retourne les statistiques des services enregistrés."""
    registrations = container.get_registrations()

    stats = {
        "total_services": len(registrations),
        "singletons": len([r for r in registrations.values()
                          if r.lifetime == LifetimeScope.SINGLETON]),
        "transients": len([r for r in registrations.values()
                          if r.lifetime == LifetimeScope.TRANSIENT]),
        "scoped": len([r for r in registrations.values()
                      if r.lifetime == LifetimeScope.SCOPED]),
        "with_factories": len([r for r in registrations.values()
                              if r.factory is not None]),
        "services": [
            {
                "interface": r.interface.__name__,
                "implementation": r.implementation.__name__ if r.implementation else "Factory",
                "lifetime": r.lifetime
            }
            for r in registrations.values()
        ]
    }

    return stats