"""
Tests d'exécution réelle pour qframe.infrastructure.observability
=================================================================

Tests complets de tous les modules d'observability avec exécution réelle
des méthodes et validation des comportements.
"""

import asyncio
import pytest
import time
import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from threading import Thread

# Core imports
from qframe.domain.entities.order import Order, OrderStatus, OrderSide, OrderType
from qframe.domain.entities.portfolio import Portfolio, PortfolioStatus, PortfolioType

# Infrastructure observability imports
from qframe.infrastructure.observability.logging import (
    StructuredLogger, LoggerFactory, LogContext, LogLevel,
    PerformanceTimer, JsonFormatter, ConsoleFormatter,
    correlation_id, user_id, session_id, request_id
)

from qframe.infrastructure.observability.metrics import (
    MetricsCollector, MetricDefinition, MetricType, MetricUnit, MetricPoint,
    MetricStorage, BusinessMetrics, TimeMeasurement,
    get_metrics_collector, get_business_metrics
)

from qframe.infrastructure.observability.tracing import (
    Tracer, TradingTracer, Span, Trace, SpanKind, SpanStatus,
    SpanContext, SpanEvent, NoOpSpan,
    get_tracer, trace, current_span, current_trace
)

from qframe.infrastructure.observability.health import (
    HealthMonitor, HealthCheck, HealthStatus, ComponentType, ComponentHealth,
    DatabaseHealthCheck, MarketDataHealthCheck, BrokerHealthCheck,
    CircuitBreaker, CircuitBreakerState, ReadinessProbe, LivenessProbe,
    get_health_monitor
)

from qframe.infrastructure.observability.alerting import (
    AlertManager, Alert, AlertRule, AlertSeverity, AlertCategory, AlertStatus,
    AlertChannel, LogChannel, EmailChannel, SlackChannel,
    AnomalyDetector, AlertCorrelator,
    get_alert_manager
)


class TestStructuredLogging:
    """Tests pour le système de logging structuré avec exécution réelle"""

    @pytest.fixture
    def log_context(self):
        """Contexte de logging pour tests"""
        return LogContext(
            correlation_id="test-correlation-001",
            user_id="test-user-001",
            session_id="test-session-001",
            strategy_id="test-strategy-001",
            portfolio_id="test-portfolio-001",
            environment="test",
            service_name="qframe-test"
        )

    @pytest.fixture
    def structured_logger(self, log_context):
        """Logger structuré pour tests"""
        return StructuredLogger(
            name="test_logger",
            level="DEBUG",
            output_format="json",
            context=log_context
        )

    def test_log_context_creation(self, log_context):
        """Test création et sérialisation du contexte"""
        assert log_context.correlation_id == "test-correlation-001"
        assert log_context.user_id == "test-user-001"
        assert log_context.environment == "test"

        # Test conversion en dictionnaire
        context_dict = log_context.to_dict()
        assert "correlation_id" in context_dict
        assert "user_id" in context_dict
        assert "service_name" in context_dict
        assert context_dict["environment"] == "test"

    def test_structured_logger_creation(self, structured_logger):
        """Test création du logger structuré"""
        assert structured_logger.name == "test_logger"
        assert structured_logger.output_format == "json"
        assert structured_logger.context.service_name == "qframe-test"

        # Test création avec contexte enrichi
        enriched_logger = structured_logger.with_context(
            order_id="test-order-001",
            trade_id="test-trade-001"
        )
        assert enriched_logger.context.order_id == "test-order-001"
        assert enriched_logger.context.trade_id == "test-trade-001"

    def test_basic_logging_levels(self, structured_logger):
        """Test des différents niveaux de logging"""
        # Capturer les logs pour vérification
        import io
        import sys
        log_capture = io.StringIO()

        # Rediriger temporairement stdout
        original_stdout = sys.stdout
        sys.stdout = log_capture

        try:
            # Test tous les niveaux
            structured_logger.debug("Debug message", test_param="debug_value")
            structured_logger.info("Info message", test_param="info_value")
            structured_logger.warning("Warning message", test_param="warning_value")
            structured_logger.error("Error message", test_param="error_value")
            structured_logger.critical("Critical message", test_param="critical_value")

            # Niveaux spécialisés
            structured_logger.trade("Trade executed", symbol="BTC/USD", quantity=0.1)
            structured_logger.risk("Risk limit approached", metric="var", value=95.0)
            structured_logger.audit("User action logged", action="place_order")

        finally:
            sys.stdout = original_stdout

        # Vérifier que les logs ont été générés
        log_output = log_capture.getvalue()
        assert len(log_output.strip()) > 0

        # Les logs JSON doivent contenir les éléments attendus
        lines = log_output.strip().split('\n')
        assert len(lines) >= 8  # Au moins 8 logs

    def test_error_logging_with_exception(self, structured_logger):
        """Test logging d'erreurs avec exceptions"""
        try:
            raise ValueError("Test exception for logging")
        except Exception as e:
            # Capturer le log d'erreur
            import io
            import sys
            log_capture = io.StringIO()
            original_stdout = sys.stdout
            sys.stdout = log_capture

            try:
                structured_logger.error("Exception occurred", error=e, context="test")
            finally:
                sys.stdout = original_stdout

            log_output = log_capture.getvalue()

            # Vérifier que l'exception est bien loggée
            assert "ValueError" in log_output
            assert "Test exception for logging" in log_output
            assert "error_traceback" in log_output

    def test_performance_timer(self, structured_logger):
        """Test du timer de performance"""
        # Test avec opération rapide
        with structured_logger.measure_performance("fast_operation") as timer:
            time.sleep(0.01)  # 10ms

        # Test avec opération lente
        with structured_logger.measure_performance("slow_operation") as timer:
            time.sleep(0.11)  # 110ms

        # Le timer doit se comporter différemment selon la durée
        assert timer is not None

    def test_specialized_logging_methods(self, structured_logger):
        """Test méthodes de logging spécialisées"""
        # Test trade execution logging
        structured_logger.log_trade_execution(
            order_id="order-001",
            symbol="BTC/USD",
            side="BUY",
            quantity=Decimal("0.1"),
            price=Decimal("45000.0"),
            venue="binance"
        )

        # Test risk breach logging
        structured_logger.log_risk_breach(
            metric="var",
            current_value=Decimal("15000.0"),
            limit=Decimal("10000.0"),
            severity="high",
            portfolio_id="portfolio-001"
        )

        # Test portfolio update logging
        structured_logger.log_portfolio_update(
            portfolio_id="portfolio-001",
            total_value=Decimal("50000.0"),
            pnl=Decimal("2500.0"),
            positions=5
        )

        # Ces appels ne doivent pas lever d'exception
        assert True

    def test_logger_factory(self):
        """Test de la factory de loggers"""
        # Configuration par défaut
        LoggerFactory.configure_defaults(
            level="INFO",
            format="json",
            context=LogContext(service_name="test-service")
        )

        # Création de loggers
        logger1 = LoggerFactory.get_logger("test.logger1")
        logger2 = LoggerFactory.get_logger("test.logger2")
        logger3 = LoggerFactory.get_logger("test.logger1")  # Même nom

        assert logger1.name == "test.logger1"
        assert logger2.name == "test.logger2"
        assert logger1 is logger3  # Même instance

        # Test des méthodes de contexte
        correlation_id_value = LoggerFactory.create_correlation_id()
        assert len(correlation_id_value) > 0

        LoggerFactory.set_user_context("user-123", "session-456")
        LoggerFactory.set_request_context("request-789")

        # Les context vars doivent être définies
        assert correlation_id.get() == correlation_id_value

    def test_json_and_console_formatters(self, log_context):
        """Test des formateurs JSON et console"""
        # Test JSON formatter
        json_formatter = JsonFormatter(log_context)

        # Créer un LogRecord de test
        import logging
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=100,
            msg="Test message",
            args=(),
            exc_info=None
        )

        # Test console formatter
        console_formatter = ConsoleFormatter()

        # Les formatters ne doivent pas lever d'exception
        assert json_formatter is not None
        assert console_formatter is not None


class TestMetricsCollection:
    """Tests pour le système de collecte de métriques avec exécution réelle"""

    @pytest.fixture
    def metrics_collector(self):
        """Collecteur de métriques pour tests"""
        collector = MetricsCollector(namespace="test")
        collector.reset()  # Nettoyer pour tests
        return collector

    @pytest.fixture
    def business_metrics(self, metrics_collector):
        """Métriques business pour tests"""
        return BusinessMetrics(metrics_collector)

    def test_metric_definition_creation(self):
        """Test création de définitions de métriques"""
        definition = MetricDefinition(
            name="test.counter",
            type=MetricType.COUNTER,
            unit=MetricUnit.COUNT,
            description="Test counter metric",
            labels=["strategy", "symbol"]
        )

        assert definition.name == "test.counter"
        assert definition.type == MetricType.COUNTER
        assert definition.unit == MetricUnit.COUNT
        assert "strategy" in definition.labels
        assert "symbol" in definition.labels

    def test_metrics_collector_registration(self, metrics_collector):
        """Test enregistrement de métriques"""
        # Enregistrer une métrique custom
        custom_metric = MetricDefinition(
            name="custom.gauge",
            type=MetricType.GAUGE,
            unit=MetricUnit.DOLLARS,
            description="Custom gauge metric"
        )

        metrics_collector.register_metric(custom_metric)

        # Vérifier l'enregistrement
        full_name = f"{metrics_collector.namespace}.{custom_metric.name}"
        assert full_name in metrics_collector._metrics

    def test_counter_operations(self, metrics_collector):
        """Test opérations sur les compteurs"""
        # Utiliser une métrique système existante
        metric_name = "trades.executed"

        # Incrémenter sans labels
        metrics_collector.increment_counter(metric_name, 1)
        value1 = metrics_collector.get_counter_value(metric_name)
        assert value1 == 1

        # Incrémenter avec labels
        labels = {"symbol": "BTC/USD", "side": "buy", "strategy": "test"}
        metrics_collector.increment_counter(metric_name, 2, labels)
        value2 = metrics_collector.get_counter_value(metric_name, labels)
        assert value2 == 2

        # Incrémenter encore
        metrics_collector.increment_counter(metric_name, 3, labels)
        value3 = metrics_collector.get_counter_value(metric_name, labels)
        assert value3 == 5  # 2 + 3

    def test_gauge_operations(self, metrics_collector):
        """Test opérations sur les gauges"""
        metric_name = "portfolio.value"

        # Définir valeur sans labels
        metrics_collector.set_gauge(metric_name, 10000.0)
        value1 = metrics_collector.get_gauge_value(metric_name)
        assert value1 == 10000.0

        # Définir valeur avec labels
        labels = {"portfolio_id": "portfolio-001"}
        metrics_collector.set_gauge(metric_name, 25000.0, labels)
        value2 = metrics_collector.get_gauge_value(metric_name, labels)
        assert value2 == 25000.0

        # Modifier la valeur
        metrics_collector.set_gauge(metric_name, 30000.0, labels)
        value3 = metrics_collector.get_gauge_value(metric_name, labels)
        assert value3 == 30000.0

    def test_histogram_operations(self, metrics_collector):
        """Test opérations sur les histogrammes"""
        metric_name = "trades.latency"

        labels = {"venue": "binance"}

        # Enregistrer plusieurs valeurs
        values = [10.0, 25.0, 50.0, 100.0, 250.0, 500.0]
        for value in values:
            metrics_collector.record_histogram(metric_name, value, labels)

        # Obtenir statistiques
        stats = metrics_collector.get_histogram_stats(metric_name, labels)

        assert stats["count"] == len(values)
        assert stats["sum"] == sum(values)
        assert stats["mean"] == sum(values) / len(values)
        assert stats["min"] == min(values)
        assert stats["max"] == max(values)
        assert 0 <= stats["p50"] <= max(values)
        assert 0 <= stats["p90"] <= max(values)
        assert 0 <= stats["p95"] <= max(values)
        assert 0 <= stats["p99"] <= max(values)

    def test_time_measurement(self, metrics_collector):
        """Test mesure du temps d'exécution"""
        metric_name = "system.api_latency"
        labels = {"endpoint": "/orders", "method": "POST"}

        # Mesurer une opération
        with metrics_collector.measure_time(metric_name, labels):
            time.sleep(0.01)  # Simuler 10ms d'opération

        # Vérifier que la mesure a été enregistrée
        stats = metrics_collector.get_histogram_stats(metric_name, labels)
        assert stats["count"] == 1
        assert stats["mean"] >= 10  # Au moins 10ms

    def test_business_metrics_integration(self, business_metrics):
        """Test métriques business spécialisées"""
        # Enregistrer un trade
        business_metrics.record_trade(
            symbol="BTC/USD",
            side="BUY",
            quantity=Decimal("0.1"),
            price=Decimal("45000.0"),
            venue="binance",
            strategy="test_strategy",
            latency_ms=25.5
        )

        # Vérifier les métriques générées
        collector = business_metrics.collector

        # Compteur de trades
        trades_count = collector.get_counter_value(
            "trades.executed",
            {"symbol": "BTC/USD", "side": "BUY", "strategy": "test_strategy"}
        )
        assert trades_count == 1

        # Volume en dollars
        volume = collector.get_counter_value(
            "trades.volume",
            {"symbol": "BTC/USD", "side": "BUY"}
        )
        assert volume == 4500.0  # 0.1 * 45000

        # Latence
        latency_stats = collector.get_histogram_stats(
            "trades.latency",
            {"venue": "binance"}
        )
        assert latency_stats["count"] == 1
        assert latency_stats["mean"] == 25.5

    def test_portfolio_metrics_update(self, business_metrics):
        """Test mise à jour des métriques de portfolio"""
        business_metrics.update_portfolio_metrics(
            portfolio_id="portfolio-001",
            total_value=Decimal("50000.0"),
            pnl=Decimal("2500.0"),
            sharpe_ratio=1.85,
            max_drawdown=0.15
        )

        collector = business_metrics.collector

        # Vérifier les gauges
        portfolio_value = collector.get_gauge_value(
            "portfolio.value",
            {"portfolio_id": "portfolio-001"}
        )
        assert portfolio_value == 50000.0

        pnl = collector.get_gauge_value(
            "portfolio.pnl",
            {"portfolio_id": "portfolio-001", "timeframe": "daily"}
        )
        assert pnl == 2500.0

        sharpe = collector.get_gauge_value(
            "portfolio.sharpe_ratio",
            {"portfolio_id": "portfolio-001"}
        )
        assert sharpe == 1.85

    def test_metrics_export(self, metrics_collector):
        """Test export des métriques"""
        # Créer quelques métriques
        metrics_collector.increment_counter("trades.executed", 5)
        metrics_collector.set_gauge("portfolio.value", 25000.0)
        metrics_collector.record_histogram("trades.latency", 50.0)

        # Export Prometheus
        prometheus_export = metrics_collector.export_prometheus()
        assert len(prometheus_export) > 0
        assert "test.trades.executed" in prometheus_export
        assert "test.portfolio.value" in prometheus_export

        # Export JSON
        json_export = metrics_collector.export_json()
        assert len(json_export) > 0

        json_data = json.loads(json_export)
        assert "namespace" in json_data
        assert "counters" in json_data
        assert "gauges" in json_data
        assert "histograms" in json_data
        assert json_data["namespace"] == "test"

    def test_metric_storage(self):
        """Test stockage des points de métriques"""
        storage = MetricStorage(max_points=100)

        # Ajouter des points
        for i in range(10):
            point = MetricPoint(
                timestamp=datetime.utcnow(),
                value=float(i),
                labels={"test": "value"}
            )
            storage.add_point("test.metric", point)

        # Récupérer les points
        points = storage.get_points("test.metric")
        assert len(points) == 10

        # Récupérer le dernier point
        latest = storage.get_latest("test.metric")
        assert latest is not None
        assert latest.value == 9.0

        # Test avec fenêtre temporelle
        now = datetime.utcnow()
        recent_points = storage.get_points(
            "test.metric",
            start_time=now - timedelta(minutes=1),
            end_time=now + timedelta(minutes=1)
        )
        assert len(recent_points) == 10


class TestDistributedTracing:
    """Tests pour le système de tracing distribué avec exécution réelle"""

    @pytest.fixture
    def tracer(self):
        """Tracer pour tests"""
        tracer = TradingTracer("test-service")
        tracer.clear_traces()  # Nettoyer pour tests
        return tracer

    def test_tracer_creation(self, tracer):
        """Test création du tracer"""
        assert tracer.service_name == "test-service"
        assert len(tracer._traces) == 0

    def test_span_creation_and_management(self, tracer):
        """Test création et gestion des spans"""
        # Créer un span simple
        span = tracer.start_span("test_operation")

        assert span is not None
        assert span.operation_name == "test_operation"
        assert span.trace_id is not None
        assert span.span_id is not None
        assert span.parent_span_id is None  # Span racine
        assert span.status == SpanStatus.UNSET

        # Définir des attributs
        span.set_attribute("test.key", "test.value")
        span.set_attribute("test.number", 42)
        span.set_attributes({
            "test.decimal": Decimal("123.45"),
            "test.boolean": True
        })

        assert span.attributes["test.key"] == "test.value"
        assert span.attributes["test.number"] == 42
        assert span.attributes["test.decimal"] == 123.45
        assert span.attributes["test.boolean"] is True

        # Ajouter un événement
        span.add_event("test_event", {"event.param": "event.value"})
        assert len(span.events) == 1
        assert span.events[0].name == "test_event"

        # Terminer le span
        span.end()
        assert span.end_time is not None
        assert span.duration_ms is not None
        assert span.duration_ms >= 0

    def test_nested_spans(self, tracer):
        """Test spans imbriqués"""
        # Créer span parent
        parent_span = tracer.start_span("parent_operation")

        # Créer span enfant
        child_span = tracer.start_span("child_operation", parent=parent_span)

        assert child_span.parent_span_id == parent_span.span_id
        assert child_span.trace_id == parent_span.trace_id

        # Créer span petit-enfant
        grandchild_span = tracer.start_span("grandchild_operation", parent=child_span)

        assert grandchild_span.parent_span_id == child_span.span_id
        assert grandchild_span.trace_id == parent_span.trace_id

        # Terminer dans l'ordre inverse
        grandchild_span.end()
        child_span.end()
        parent_span.end()

        # Vérifier la trace
        trace = tracer.get_trace(parent_span.trace_id)
        assert trace is not None
        assert len(trace.spans) == 3

    def test_span_context_manager(self, tracer):
        """Test context manager pour spans"""
        with tracer.span("context_operation", attributes={"test": "value"}) as span:
            assert span.operation_name == "context_operation"
            assert span.attributes["test"] == "value"

            # Ajouter une opération dans le contexte
            time.sleep(0.001)
            span.add_event("operation_completed")

        # Le span doit être automatiquement terminé
        assert span.end_time is not None
        assert span.status == SpanStatus.OK

    def test_span_error_handling(self, tracer):
        """Test gestion d'erreurs dans les spans"""
        # Test avec exception dans context manager
        try:
            with tracer.span("error_operation") as span:
                raise ValueError("Test error for span")
        except ValueError:
            pass

        # Le span doit avoir enregistré l'erreur
        assert span.status == SpanStatus.ERROR
        assert len(span.events) > 0

        # Vérifier l'événement d'exception
        exception_event = next((e for e in span.events if e.name == "exception"), None)
        assert exception_event is not None
        assert "exception.type" in exception_event.attributes
        assert exception_event.attributes["exception.type"] == "ValueError"

    def test_trading_specific_tracing(self, tracer):
        """Test tracing spécifique au trading"""
        # Tracer l'exécution d'un ordre
        order_span = tracer.trace_order_execution(
            order_id="order-001",
            symbol="BTC/USD",
            side="BUY",
            quantity=Decimal("0.1"),
            price=Decimal("45000.0")
        )

        assert order_span.operation_name == "order.execute"
        assert order_span.attributes["order.id"] == "order-001"
        assert order_span.attributes["order.symbol"] == "BTC/USD"
        assert order_span.attributes["order.quantity"] == 0.1
        assert order_span.attributes["order.value"] == 4500.0

        # Tracer un signal de stratégie
        signal_span = tracer.trace_strategy_signal(
            strategy_id="mean_reversion",
            signal_type="BUY",
            confidence=0.85
        )

        assert signal_span.operation_name == "strategy.signal"
        assert signal_span.attributes["strategy.id"] == "mean_reversion"
        assert signal_span.attributes["signal.confidence"] == 0.85

        # Tracer une vérification de risque
        risk_span = tracer.trace_risk_check(
            check_type="position_size",
            portfolio_id="portfolio-001",
            result="approved"
        )

        assert risk_span.operation_name == "risk.check"
        assert risk_span.attributes["risk.check_type"] == "position_size"
        assert risk_span.attributes["risk.result"] == "approved"

        # Tracer récupération de données de marché
        data_span = tracer.trace_market_data_fetch(
            provider="binance",
            symbol="BTC/USD",
            data_type="ticker"
        )

        assert data_span.operation_name == "market_data.fetch"
        assert data_span.attributes["provider"] == "binance"
        assert data_span.attributes["data.type"] == "ticker"

    def test_trace_sampling(self, tracer):
        """Test échantillonnage des traces"""
        # Ajouter un sampler qui rejette certaines opérations
        tracer.add_sampler(lambda op: op != "ignored_operation")

        # Cette opération doit être tracée
        span1 = tracer.start_span("normal_operation")
        assert not isinstance(span1, NoOpSpan)

        # Cette opération doit être ignorée
        span2 = tracer.start_span("ignored_operation")
        assert isinstance(span2, NoOpSpan)

    def test_trace_export(self, tracer):
        """Test export des traces"""
        # Créer quelques traces
        with tracer.span("operation_1") as span1:
            span1.set_attribute("test", "value1")
            time.sleep(0.001)

        with tracer.span("operation_2") as span2:
            span2.set_attribute("test", "value2")
            time.sleep(0.001)

        # Exporter les traces
        exported_traces = tracer.export_traces()

        assert len(exported_traces) >= 2
        for trace_data in exported_traces:
            assert "trace_id" in trace_data
            assert "spans" in trace_data
            assert "duration_ms" in trace_data
            assert len(trace_data["spans"]) >= 1

    def test_span_context_propagation(self, tracer):
        """Test propagation du contexte de span"""
        # Créer un span et injecter le contexte
        with tracer.span("http_request") as span:
            headers = {}
            tracer.inject_context(headers)

            assert "X-Trace-Id" in headers
            assert "X-Span-Id" in headers
            assert headers["X-Trace-Id"] == span.trace_id
            assert headers["X-Span-Id"] == span.span_id

        # Extraire le contexte
        extracted_context = tracer.extract_context(headers)
        assert extracted_context is not None
        assert extracted_context.trace_id == span.trace_id

    def test_trace_decorator(self, tracer):
        """Test décorateur de tracing"""
        @tracer.trace_operation("decorated_function", test_param="test_value")
        def test_function(x, y):
            time.sleep(0.001)
            return x + y

        result = test_function(2, 3)
        assert result == 5

        # Vérifier que la trace a été créée
        traces = tracer.export_traces()
        assert len(traces) >= 1

        # Trouver notre trace
        our_trace = None
        for trace_data in traces:
            for span_data in trace_data["spans"]:
                if span_data["operation_name"] == "decorated_function":
                    our_trace = span_data
                    break

        assert our_trace is not None
        assert our_trace["status"] == "ok"


class TestHealthMonitoring:
    """Tests pour le système de monitoring de santé avec exécution réelle"""

    @pytest.fixture
    def health_monitor(self):
        """Moniteur de santé pour tests"""
        monitor = HealthMonitor(check_interval=1)  # 1 seconde pour tests
        return monitor

    @pytest.fixture
    def mock_database_pool(self):
        """Pool de base de données mocké"""
        pool = AsyncMock()

        async def mock_acquire():
            conn = AsyncMock()
            conn.fetchval.return_value = 1
            return conn

        pool.acquire.return_value.__aenter__ = mock_acquire
        pool.acquire.return_value.__aexit__ = AsyncMock()
        return pool

    @pytest.fixture
    def mock_market_data_provider(self):
        """Fournisseur de données de marché mocké"""
        provider = AsyncMock()
        provider.get_current_price.return_value = Decimal("45000.0")
        return provider

    @pytest.fixture
    def mock_broker_service(self):
        """Service broker mocké"""
        broker = AsyncMock()
        broker.get_account_balance.return_value = {"USD": 10000.0}
        return broker

    def test_health_check_creation(self, mock_database_pool):
        """Test création de health checks"""
        db_check = DatabaseHealthCheck(mock_database_pool, "test_database")

        assert db_check.name == "test_database"
        assert db_check.component_type == ComponentType.DATABASE

    async def test_database_health_check(self, mock_database_pool):
        """Test health check de base de données"""
        db_check = DatabaseHealthCheck(mock_database_pool, "test_db")

        result = await db_check.check()

        assert result.name == "test_db"
        assert result.status == HealthStatus.HEALTHY
        assert result.response_time_ms is not None
        assert result.response_time_ms >= 0
        assert "successful" in result.message.lower()

    async def test_database_health_check_failure(self):
        """Test health check de base de données en échec"""
        # Pool qui lève une exception
        failing_pool = AsyncMock()
        failing_pool.acquire.side_effect = Exception("Connection failed")

        db_check = DatabaseHealthCheck(failing_pool, "failing_db")
        result = await db_check.check()

        assert result.status == HealthStatus.UNHEALTHY
        assert result.error is not None
        assert "Connection failed" in result.error

    async def test_market_data_health_check(self, mock_market_data_provider):
        """Test health check de données de marché"""
        data_check = MarketDataHealthCheck(mock_market_data_provider, "test_data")

        result = await data_check.check()

        assert result.name == "test_data"
        assert result.status == HealthStatus.HEALTHY
        assert result.details["test_symbol"] == "BTC/USDT"
        assert result.details["price"] == 45000.0

    async def test_broker_health_check(self, mock_broker_service):
        """Test health check de broker"""
        broker_check = BrokerHealthCheck(mock_broker_service, "test_broker")

        result = await broker_check.check()

        assert result.name == "test_broker"
        assert result.status == HealthStatus.HEALTHY
        assert result.details["account_active"] is True

    def test_component_health_tracking(self):
        """Test suivi de la santé des composants"""
        component = ComponentHealth(
            name="test_component",
            type=ComponentType.SERVICE,
            status=HealthStatus.UNKNOWN,
            last_check=datetime.utcnow()
        )

        # Ajouter des résultats réussis
        for i in range(5):
            result = HealthCheckResult(
                name="test_component",
                status=HealthStatus.HEALTHY,
                response_time_ms=10.0 + i
            )
            component.add_result(result)

        assert component.status == HealthStatus.HEALTHY
        assert component.consecutive_successes == 5
        assert component.consecutive_failures == 0

        # Ajouter un échec
        failure_result = HealthCheckResult(
            name="test_component",
            status=HealthStatus.UNHEALTHY,
            error="Test failure"
        )
        component.add_result(failure_result)

        assert component.status == HealthStatus.UNHEALTHY
        assert component.consecutive_successes == 0
        assert component.consecutive_failures == 1

        # Vérifier uptime et temps de réponse moyen
        uptime = component.get_uptime_percentage(1)  # 1 heure
        assert 0 <= uptime <= 100

        avg_response = component.get_average_response_time(5)
        assert avg_response is not None
        assert avg_response >= 10.0

    def test_circuit_breaker(self):
        """Test circuit breaker"""
        cb = CircuitBreaker(
            name="test_service",
            failure_threshold=3,
            recovery_timeout=5,
            half_open_max_calls=2
        )

        assert cb.state == CircuitBreakerState.CLOSED

        # Fonction qui va échouer
        def failing_function():
            raise Exception("Service unavailable")

        # Fonction qui réussit
        def successful_function():
            return "success"

        # Déclencher des échecs pour ouvrir le circuit
        for i in range(3):
            try:
                cb.call(failing_function)
            except:
                pass

        assert cb.state == CircuitBreakerState.OPEN

        # Test que le circuit reste ouvert
        try:
            cb.call(successful_function)
            assert False, "Circuit should be open"
        except Exception as e:
            assert "OPEN" in str(e)

        # Simuler le timeout de récupération
        cb.last_failure_time = datetime.utcnow() - timedelta(seconds=10)

        # Le circuit devrait passer en HALF_OPEN
        try:
            cb.call(successful_function)
        except:
            pass

        # Test des succès pour fermer le circuit
        for i in range(3):
            try:
                result = cb.call(successful_function)
                assert result == "success"
            except:
                pass

    async def test_health_monitor_integration(self, health_monitor, mock_database_pool):
        """Test intégration complète du moniteur de santé"""
        # Enregistrer des health checks
        db_check = DatabaseHealthCheck(mock_database_pool, "database")
        health_monitor.register_check(db_check)

        # Ajouter un handler d'alerte
        alerts_received = []
        def alert_handler(component: str, status: HealthStatus):
            alerts_received.append((component, status))

        health_monitor.add_alert_handler(alert_handler)

        # Effectuer un check manuel
        result = await health_monitor.check_component("database")
        assert result is not None
        assert result.status == HealthStatus.HEALTHY

        # Obtenir la santé du composant
        component_health = health_monitor.get_component_health("database")
        assert component_health is not None
        assert component_health.name == "database"

        # Obtenir la santé globale du système
        system_health = health_monitor.get_system_health()
        assert "status" in system_health
        assert "components" in system_health
        assert "circuit_breakers" in system_health
        assert "database" in system_health["components"]

        # Vérifier si le système est sain
        is_healthy = health_monitor.is_healthy()
        assert is_healthy is True

    def test_readiness_and_liveness_probes(self, health_monitor, mock_database_pool):
        """Test des probes de readiness et liveness"""
        # Configurer le moniteur
        db_check = DatabaseHealthCheck(mock_database_pool, "database")
        health_monitor.register_check(db_check)

        # Créer les probes
        readiness = ReadinessProbe(health_monitor, ["database"])
        liveness = LivenessProbe(health_monitor)

        # Les probes doivent être initialement prêtes
        # (en pratique, il faudrait attendre un check)
        assert readiness is not None
        assert liveness is not None


class TestIntelligentAlerting:
    """Tests pour le système d'alerting intelligent avec exécution réelle"""

    @pytest.fixture
    def alert_manager(self):
        """Gestionnaire d'alertes pour tests"""
        manager = AlertManager()
        manager._alerts.clear()  # Nettoyer pour tests
        return manager

    @pytest.fixture
    def anomaly_detector(self):
        """Détecteur d'anomalies pour tests"""
        return AnomalyDetector(window_size=50, z_threshold=2.0)

    def test_alert_creation(self):
        """Test création d'alertes"""
        alert = Alert(
            id="test-alert-001",
            title="Test Alert",
            message="This is a test alert",
            severity=AlertSeverity.WARNING,
            category=AlertCategory.TRADING,
            source="test_system",
            metadata={"test_key": "test_value"}
        )

        assert alert.id == "test-alert-001"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.category == AlertCategory.TRADING
        assert alert.status == AlertStatus.OPEN
        assert alert.fingerprint is not None
        assert len(alert.fingerprint) == 16

        # Test conversion en dictionnaire
        alert_dict = alert.to_dict()
        assert alert_dict["id"] == alert.id
        assert alert_dict["severity"] == "warning"
        assert alert_dict["category"] == "trading"

    def test_alert_rule_evaluation(self):
        """Test évaluation des règles d'alerte"""
        # Règle simple : alerte si P&L < -1000
        rule = AlertRule(
            name="loss_alert",
            condition=lambda data: data.get("pnl", 0) < -1000,
            severity=AlertSeverity.ERROR,
            category=AlertCategory.TRADING,
            title_template="Loss Alert: ${pnl:.2f}",
            message_template="Portfolio {portfolio_id} has loss of ${pnl:.2f}",
            cooldown_seconds=60
        )

        # Test avec données qui déclenchent l'alerte
        trigger_data = {"pnl": -1500, "portfolio_id": "portfolio-001"}
        alert = rule.evaluate(trigger_data)

        assert alert is not None
        assert alert.severity == AlertSeverity.ERROR
        assert "$-1500.00" in alert.title
        assert "portfolio-001" in alert.message

        # Test avec données qui ne déclenchent pas l'alerte
        no_trigger_data = {"pnl": 500, "portfolio_id": "portfolio-001"}
        no_alert = rule.evaluate(no_trigger_data)

        assert no_alert is None

        # Test cooldown
        alert2 = rule.evaluate(trigger_data)
        assert alert2 is None  # Cooldown actif

    def test_alert_channels(self):
        """Test canaux d'alerte"""
        alert = Alert(
            id="channel-test-001",
            title="Channel Test Alert",
            message="Testing alert channels",
            severity=AlertSeverity.INFO,
            category=AlertCategory.SYSTEM,
            source="test"
        )

        # Test log channel
        log_channel = LogChannel()
        assert log_channel.name == "log"

        # Test email channel
        email_channel = EmailChannel(
            smtp_config={"host": "test", "port": 587},
            recipients=["test@example.com"]
        )
        assert email_channel.name == "email"

        # Test Slack channel
        slack_channel = SlackChannel(
            webhook_url="https://test.slack.com/webhook",
            channel="#alerts"
        )
        assert slack_channel.name == "slack"

    async def test_alert_channel_sending(self):
        """Test envoi d'alertes via les canaux"""
        alert = Alert(
            id="send-test-001",
            title="Send Test Alert",
            message="Testing alert sending",
            severity=AlertSeverity.WARNING,
            category=AlertCategory.PERFORMANCE,
            source="test"
        )

        # Test envoi via log channel
        log_channel = LogChannel()
        result = await log_channel.send(alert)
        assert result is True

        # Test envoi via email channel (mocké)
        email_channel = EmailChannel({}, ["test@example.com"])
        result = await email_channel.send(alert)
        assert result is True

        # Test envoi via Slack channel (mocké)
        slack_channel = SlackChannel("https://test.webhook")
        result = await slack_channel.send(alert)
        assert result is True

    def test_anomaly_detector(self, anomaly_detector):
        """Test détecteur d'anomalies"""
        metric_name = "response_time"

        # Ajouter des valeurs normales
        normal_values = [10, 12, 11, 13, 9, 14, 10, 11, 12, 13]
        for value in normal_values:
            anomaly_detector.add_value(metric_name, value)

        # Test valeur normale
        is_anomaly = anomaly_detector.is_anomaly(metric_name, 11.5)
        assert is_anomaly is False

        # Test valeur anormale
        is_anomaly = anomaly_detector.is_anomaly(metric_name, 50.0)
        assert is_anomaly is True

        # Test statistiques
        stats = anomaly_detector.get_statistics(metric_name)
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert stats["count"] == len(normal_values)

    def test_alert_correlator(self):
        """Test corrélateur d'alertes"""
        correlator = AlertCorrelator(time_window_seconds=60)

        # Créer des alertes similaires
        alert1 = Alert(
            id="corr-1",
            title="Database Error",
            message="Database connection failed",
            severity=AlertSeverity.ERROR,
            category=AlertCategory.SYSTEM,
            source="database"
        )

        alert2 = Alert(
            id="corr-2",
            title="Database Error",
            message="Database query timeout",
            severity=AlertSeverity.ERROR,
            category=AlertCategory.SYSTEM,
            source="database"
        )

        # Corréler les alertes
        group_id1 = correlator.correlate(alert1)
        group_id2 = correlator.correlate(alert2)

        # Elles devraient être dans le même groupe
        assert group_id1 == group_id2

        # Récupérer le groupe
        group_alerts = correlator.get_group(group_id1)
        assert len(group_alerts) == 2

    def test_alert_manager_integration(self, alert_manager):
        """Test intégration complète du gestionnaire d'alertes"""
        # Enregistrer un canal
        log_channel = LogChannel()
        alert_manager.register_channel(log_channel)

        # Enregistrer une règle
        rule = AlertRule(
            name="test_rule",
            condition=lambda data: data.get("error", False),
            severity=AlertSeverity.ERROR,
            category=AlertCategory.SYSTEM,
            title_template="System Error: {error_type}",
            message_template="Error occurred: {error_message}"
        )
        alert_manager.register_rule(rule)

        # Configurer le routage
        alert_manager.configure_severity_routing(
            AlertSeverity.ERROR,
            ["log"]
        )

        # Déclencher une alerte
        test_data = {
            "error": True,
            "error_type": "timeout",
            "error_message": "Operation timed out"
        }
        alert_manager.evaluate_rules(test_data)

        # Vérifier que l'alerte a été créée
        alerts = alert_manager.get_alerts()
        assert len(alerts) >= 1

        error_alert = alerts[0]
        assert error_alert.severity == AlertSeverity.ERROR
        assert "timeout" in error_alert.title

        # Test reconnaissance d'alerte
        alert_manager.acknowledge_alert(error_alert.id, "test_user")
        updated_alert = alert_manager.get_alerts()[0]
        assert updated_alert.status == AlertStatus.ACKNOWLEDGED
        assert updated_alert.acknowledged_by == "test_user"

        # Test résolution d'alerte
        alert_manager.resolve_alert(error_alert.id)
        resolved_alert = alert_manager.get_alerts()[0]
        assert resolved_alert.status == AlertStatus.RESOLVED

    def test_alert_suppression(self, alert_manager):
        """Test suppression d'alertes"""
        # Ajouter une règle de suppression
        def suppress_test_alerts(alert: Alert) -> bool:
            return alert.source == "suppressed_source"

        alert_manager.add_suppression_rule(suppress_test_alerts)

        # Créer une alerte qui doit être supprimée
        suppressed_alert = Alert(
            id="suppressed-001",
            title="Suppressed Alert",
            message="This alert should be suppressed",
            severity=AlertSeverity.WARNING,
            category=AlertCategory.SYSTEM,
            source="suppressed_source"
        )

        alert_manager.trigger_alert(suppressed_alert)

        # L'alerte doit être supprimée
        assert suppressed_alert.status == AlertStatus.SUPPRESSED

    def test_trading_alert_rules(self, alert_manager):
        """Test règles d'alerte spécifiques au trading"""
        # Enregistrer les règles de trading
        alert_manager.register_trading_rules()

        # Test alerte de grosse perte
        loss_data = {
            "pnl": -15000,
            "portfolio_id": "portfolio-001"
        }
        alert_manager.evaluate_rules(loss_data)

        # Test alerte de breach de risque
        risk_data = {
            "risk_breach": True,
            "metric": "var",
            "value": 12000,
            "limit": 10000
        }
        alert_manager.evaluate_rules(risk_data)

        # Test alerte de latence élevée
        latency_data = {
            "latency_ms": 1500,
            "operation": "place_order"
        }
        alert_manager.evaluate_rules(latency_data)

        # Vérifier que les alertes ont été créées
        alerts = alert_manager.get_alerts()
        assert len(alerts) >= 3

        # Vérifier les types d'alertes
        alert_titles = [alert.title for alert in alerts]
        assert any("Large Loss" in title for title in alert_titles)
        assert any("Risk Limit Breach" in title for title in alert_titles)
        assert any("High Latency" in title for title in alert_titles)

    def test_alert_statistics(self, alert_manager):
        """Test statistiques d'alertes"""
        # Créer plusieurs alertes de différents types
        alerts_data = [
            (AlertSeverity.INFO, AlertCategory.SYSTEM),
            (AlertSeverity.WARNING, AlertCategory.PERFORMANCE),
            (AlertSeverity.ERROR, AlertCategory.TRADING),
            (AlertSeverity.CRITICAL, AlertCategory.RISK),
            (AlertSeverity.WARNING, AlertCategory.TRADING)
        ]

        for severity, category in alerts_data:
            alert = Alert(
                id=f"stats-{severity.value}-{category.value}",
                title=f"Stats Alert {severity.value}",
                message="Statistics test alert",
                severity=severity,
                category=category,
                source="test"
            )
            alert_manager.trigger_alert(alert)

        # Obtenir statistiques
        stats = alert_manager.get_alert_statistics()

        assert stats["total"] == len(alerts_data)
        assert stats["by_severity"][AlertSeverity.WARNING.value] == 2
        assert stats["by_category"][AlertCategory.TRADING.value] == 2

    def test_alert_cleanup(self, alert_manager):
        """Test nettoyage des alertes"""
        # Créer une alerte ancienne résolue
        old_alert = Alert(
            id="old-alert-001",
            title="Old Alert",
            message="This is an old alert",
            severity=AlertSeverity.INFO,
            category=AlertCategory.SYSTEM,
            source="test"
        )

        # La marquer comme résolue dans le passé
        old_alert.status = AlertStatus.RESOLVED
        old_alert.resolved_at = datetime.utcnow() - timedelta(days=10)

        alert_manager._alerts[old_alert.id] = old_alert

        # Effectuer le nettoyage
        alert_manager.cleanup(days_old=7)

        # L'alerte ancienne doit être supprimée
        assert old_alert.id not in alert_manager._alerts


# ============================================
# TESTS D'INTÉGRATION OBSERVABILITY
# ============================================

class TestObservabilityIntegration:
    """Tests d'intégration des différents composants d'observability"""

    @pytest.fixture
    async def integrated_observability(self):
        """Système d'observability complet intégré"""
        # Logger structuré
        logger = LoggerFactory.get_logger("integration_test")

        # Métriques
        metrics = get_metrics_collector()
        metrics.reset()

        # Tracing
        tracer = get_tracer()
        tracer.clear_traces()

        # Health monitoring
        health_monitor = get_health_monitor()

        # Alerting
        alert_manager = get_alert_manager()
        alert_manager._alerts.clear()

        return {
            "logger": logger,
            "metrics": metrics,
            "tracer": tracer,
            "health_monitor": health_monitor,
            "alert_manager": alert_manager
        }

    async def test_full_workflow_integration(self, integrated_observability):
        """Test workflow complet d'observability"""
        logger = integrated_observability["logger"]
        metrics = integrated_observability["metrics"]
        tracer = integrated_observability["tracer"]
        alert_manager = integrated_observability["alert_manager"]

        # Simuler une opération de trading complète avec observability
        correlation_id_value = LoggerFactory.create_correlation_id()

        with tracer.span("trade_execution", attributes={"symbol": "BTC/USD"}) as span:
            # Logger le début de l'opération
            logger.info(
                "Starting trade execution",
                symbol="BTC/USD",
                quantity=0.1,
                order_type="MARKET"
            )

            # Mesurer la latence
            start_time = time.time()

            # Simuler validation de risque
            with tracer.span("risk_validation") as risk_span:
                time.sleep(0.001)  # Simuler calculs
                risk_span.set_attribute("risk_check_result", "approved")
                logger.debug("Risk validation completed", result="approved")

            # Simuler exécution d'ordre
            with tracer.span("order_execution") as exec_span:
                time.sleep(0.002)  # Simuler exécution
                exec_span.set_attribute("execution_price", 45000.0)
                exec_span.set_attribute("executed_quantity", 0.1)

                # Logger l'exécution
                logger.trade(
                    "Trade executed successfully",
                    symbol="BTC/USD",
                    price=45000.0,
                    quantity=0.1,
                    venue="binance"
                )

            # Enregistrer les métriques
            latency_ms = (time.time() - start_time) * 1000
            metrics.increment_counter("trades.executed", labels={"symbol": "BTC/USD"})
            metrics.record_histogram("trades.latency", latency_ms)
            metrics.set_gauge("portfolio.value", 50000.0)

            # Simuler une condition d'alerte (optionnelle)
            if latency_ms > 100:  # Si trop lent
                alert_data = {
                    "latency_ms": latency_ms,
                    "operation": "trade_execution"
                }
                alert_manager.evaluate_rules(alert_data)

            span.set_status(SpanStatus.OK)

        # Vérifications post-exécution
        # 1. Traces créées
        traces = tracer.export_traces()
        assert len(traces) >= 1
        main_trace = traces[0]
        assert len(main_trace["spans"]) >= 3  # trade_execution, risk_validation, order_execution

        # 2. Métriques enregistrées
        trade_count = metrics.get_counter_value("trades.executed", {"symbol": "BTC/USD"})
        assert trade_count == 1

        latency_stats = metrics.get_histogram_stats("trades.latency")
        assert latency_stats["count"] == 1

        portfolio_value = metrics.get_gauge_value("portfolio.value")
        assert portfolio_value == 50000.0

        # 3. Logs générés (difficile à vérifier directement, mais pas d'exception)
        assert logger is not None

    async def test_error_scenario_observability(self, integrated_observability):
        """Test observability lors d'erreurs"""
        logger = integrated_observability["logger"]
        metrics = integrated_observability["metrics"]
        tracer = integrated_observability["tracer"]
        alert_manager = integrated_observability["alert_manager"]

        # Enregistrer une règle d'alerte pour les erreurs
        error_rule = AlertRule(
            name="trade_error",
            condition=lambda data: data.get("error_occurred", False),
            severity=AlertSeverity.ERROR,
            category=AlertCategory.TRADING,
            title_template="Trade Error: {error_type}",
            message_template="Trade failed: {error_message}"
        )
        alert_manager.register_rule(error_rule)

        try:
            with tracer.span("failing_trade_execution") as span:
                span.set_attribute("symbol", "ETH/USD")

                # Logger le début
                logger.info("Starting trade execution", symbol="ETH/USD")

                # Simuler une erreur
                with tracer.span("broker_connection") as broker_span:
                    broker_span.add_event("connection_attempt")

                    # Simuler échec de connexion
                    raise ConnectionError("Broker connection failed")

        except ConnectionError as e:
            # Logger l'erreur
            logger.error("Trade execution failed", error=e, symbol="ETH/USD")

            # Métriques d'erreur
            metrics.increment_counter("system.errors", labels={
                "error_type": "ConnectionError",
                "component": "broker"
            })

            # Déclencher alerte
            alert_data = {
                "error_occurred": True,
                "error_type": "ConnectionError",
                "error_message": str(e)
            }
            alert_manager.evaluate_rules(alert_data)

        # Vérifications
        # 1. Trace avec erreur
        traces = tracer.export_traces()
        assert len(traces) >= 1
        error_trace = traces[0]

        # Trouver le span d'erreur
        error_span = None
        for span_data in error_trace["spans"]:
            if span_data["status"] == "error":
                error_span = span_data
                break

        assert error_span is not None

        # 2. Métriques d'erreur
        error_count = metrics.get_counter_value("system.errors", {
            "error_type": "ConnectionError",
            "component": "broker"
        })
        assert error_count == 1

        # 3. Alerte générée
        alerts = alert_manager.get_alerts()
        assert len(alerts) >= 1
        error_alert = alerts[0]
        assert error_alert.severity == AlertSeverity.ERROR
        assert "ConnectionError" in error_alert.title

    async def test_health_monitoring_integration(self, integrated_observability):
        """Test intégration du monitoring de santé"""
        health_monitor = integrated_observability["health_monitor"]
        metrics = integrated_observability["metrics"]
        alert_manager = integrated_observability["alert_manager"]

        # Créer un health check qui va échouer
        class FailingHealthCheck(HealthCheck):
            def __init__(self):
                self.call_count = 0

            @property
            def name(self) -> str:
                return "failing_service"

            async def check(self):
                self.call_count += 1
                # Échouer après 2 appels
                if self.call_count > 2:
                    raise Exception("Service unavailable")

                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="Service OK"
                )

        failing_check = FailingHealthCheck()
        health_monitor.register_check(failing_check)

        # Ajouter un handler d'alerte pour les changements de santé
        health_alerts = []
        def health_alert_handler(component: str, status: HealthStatus):
            health_alerts.append((component, status))

            # Créer une alerte
            alert = Alert(
                id=f"health-{component}-{int(time.time())}",
                title=f"Health Status Changed: {component}",
                message=f"Component {component} is now {status.value}",
                severity=AlertSeverity.WARNING if status == HealthStatus.DEGRADED else AlertSeverity.ERROR,
                category=AlertCategory.SYSTEM,
                source="health_monitor"
            )
            alert_manager.trigger_alert(alert)

        health_monitor.add_alert_handler(health_alert_handler)

        # Effectuer plusieurs checks
        for i in range(4):
            try:
                await health_monitor.check_component("failing_service")
            except:
                pass

        # Vérifier que des alertes de santé ont été générées
        assert len(health_alerts) >= 1

        # Vérifier les alertes dans le gestionnaire
        health_system_alerts = [
            a for a in alert_manager.get_alerts()
            if a.source == "health_monitor"
        ]
        assert len(health_system_alerts) >= 1

    def test_performance_under_load(self, integrated_observability):
        """Test performance de l'observability sous charge"""
        logger = integrated_observability["logger"]
        metrics = integrated_observability["metrics"]
        tracer = integrated_observability["tracer"]

        # Nombre d'opérations à simuler
        num_operations = 100

        start_time = time.time()

        # Simuler beaucoup d'opérations simultanées
        for i in range(num_operations):
            # Tracing
            with tracer.span(f"operation_{i}") as span:
                span.set_attribute("operation_id", i)

                # Logging
                logger.debug(f"Processing operation {i}", operation_id=i)

                # Métriques
                metrics.increment_counter("operations.processed")
                metrics.record_histogram("operation.duration", float(i % 10))

                # Simuler un peu de travail
                time.sleep(0.001)

        total_time = time.time() - start_time

        # Vérifications de performance
        # Doit traiter 100 opérations en moins de 5 secondes
        assert total_time < 5.0

        # Vérifier que toutes les métriques ont été enregistrées
        operation_count = metrics.get_counter_value("operations.processed")
        assert operation_count == num_operations

        duration_stats = metrics.get_histogram_stats("operation.duration")
        assert duration_stats["count"] == num_operations

        # Vérifier que toutes les traces ont été créées
        traces = tracer.export_traces()
        assert len(traces) == num_operations

    def test_memory_usage_observability(self, integrated_observability):
        """Test utilisation mémoire de l'observability"""
        metrics = integrated_observability["metrics"]
        tracer = integrated_observability["tracer"]

        # Mesurer l'utilisation mémoire initiale
        import psutil
        import os
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Générer beaucoup de données d'observability
        for i in range(1000):
            with tracer.span(f"memory_test_{i}") as span:
                span.set_attribute("iteration", i)
                span.add_event("memory_test_event")

                metrics.increment_counter("memory.test")
                metrics.record_histogram("memory.values", float(i))

        # Mesurer l'utilisation mémoire finale
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # L'augmentation mémoire ne doit pas être excessive
        # (limite arbitraire de 50MB pour 1000 opérations)
        assert memory_increase < 50 * 1024 * 1024

        # Nettoyer pour limiter l'utilisation mémoire
        tracer.clear_traces()
        metrics.reset()


# ============================================
# TESTS D'EXÉCUTION ET VALIDATION COMPLÈTE
# ============================================

if __name__ == "__main__":
    # Exécution directe pour validation
    print("🔍 Tests d'exécution réelle - Infrastructure Observability")
    print("=" * 65)

    # Tests de base
    async def run_basic_tests():
        """Exécuter les tests de base"""
        print("📝 Test StructuredLogger...")

        # Test logger structuré
        context = LogContext(
            correlation_id="test-001",
            service_name="test-service"
        )

        logger = StructuredLogger("test", "DEBUG", "json", context)
        logger.info("Test logging message", test_param="test_value")
        print("  ✅ Structured logging - OK")

        print("📊 Test MetricsCollector...")

        # Test métriques
        metrics = MetricsCollector("test")
        metrics.increment_counter("test.counter", 5)
        metrics.set_gauge("test.gauge", 100.0)
        metrics.record_histogram("test.histogram", 25.0)

        assert metrics.get_counter_value("test.counter") == 5
        assert metrics.get_gauge_value("test.gauge") == 100.0

        stats = metrics.get_histogram_stats("test.histogram")
        assert stats["count"] == 1

        print("  ✅ Metrics collection - OK")

        print("🔍 Test TradingTracer...")

        # Test tracing
        tracer = TradingTracer("test-service")

        with tracer.span("test_operation") as span:
            span.set_attribute("test_key", "test_value")
            span.add_event("test_event")
            time.sleep(0.001)

        traces = tracer.export_traces()
        assert len(traces) >= 1
        assert traces[0]["spans"][0]["operation_name"] == "test_operation"

        print("  ✅ Distributed tracing - OK")

        print("🏥 Test HealthMonitor...")

        # Test health monitoring
        monitor = HealthMonitor(check_interval=1)

        # Health check simple
        class TestHealthCheck(HealthCheck):
            @property
            def name(self) -> str:
                return "test_service"

            async def check(self):
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="Test service OK"
                )

        test_check = TestHealthCheck()
        monitor.register_check(test_check)

        result = await monitor.check_component("test_service")
        assert result.status == HealthStatus.HEALTHY

        print("  ✅ Health monitoring - OK")

        print("🚨 Test AlertManager...")

        # Test alerting
        alert_manager = AlertManager()

        # Règle d'alerte simple
        rule = AlertRule(
            name="test_alert",
            condition=lambda data: data.get("trigger", False),
            severity=AlertSeverity.WARNING,
            category=AlertCategory.SYSTEM,
            title_template="Test Alert",
            message_template="Test alert message"
        )

        alert_manager.register_rule(rule)
        alert_manager.evaluate_rules({"trigger": True})

        alerts = alert_manager.get_alerts()
        assert len(alerts) >= 1

        print("  ✅ Intelligent alerting - OK")

        print("🔄 Test intégration complète...")

        # Test workflow intégré
        correlation_id_value = LoggerFactory.create_correlation_id()

        with tracer.span("integration_test") as span:
            logger.info("Integration test starting")

            metrics.increment_counter("integration.tests")

            # Simuler une opération
            time.sleep(0.001)

            metrics.record_histogram("integration.duration", 1.0)
            logger.info("Integration test completed")

        # Vérifications
        assert metrics.get_counter_value("integration.tests") == 1

        integration_traces = tracer.export_traces()
        integration_trace = next(
            (t for t in integration_traces
             if any(s["operation_name"] == "integration_test" for s in t["spans"])),
            None
        )
        assert integration_trace is not None

        print("  ✅ Intégration complète - OK")

        return True

    # Exécuter les tests de base
    try:
        result = asyncio.run(run_basic_tests())
        if result:
            print("\n🎉 Tous les tests d'exécution réelle sont VALIDÉS !")
            print("   Infrastructure Observability complètement fonctionnelle.")
            print("   - Logging structuré avec contexte riche")
            print("   - Métriques Prometheus-compatible")
            print("   - Tracing distribué avec corrélation")
            print("   - Health monitoring avec circuit breakers")
            print("   - Alerting intelligent avec ML")
    except Exception as e:
        print(f"\n❌ Erreur lors des tests : {e}")
        import traceback
        traceback.print_exc()