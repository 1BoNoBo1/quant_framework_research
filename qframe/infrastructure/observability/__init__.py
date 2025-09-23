"""
Infrastructure Layer: Observability Module
=========================================

Module complet d'observabilité pour monitoring, logging, tracing et alerting.
Fournit une visibilité complète sur le système en production.
"""

from .logging import (
    LogLevel,
    LogContext,
    StructuredLogger,
    LoggerFactory,
    logger
)

from .metrics import (
    MetricType,
    MetricUnit,
    MetricDefinition,
    MetricsCollector,
    BusinessMetrics,
    get_metrics_collector,
    get_business_metrics
)

from .tracing import (
    SpanKind,
    SpanStatus,
    Span,
    Trace,
    Tracer,
    TradingTracer,
    get_tracer,
    trace
)

from .health import (
    HealthStatus,
    ComponentType,
    CircuitBreakerState,
    HealthCheckResult,
    ComponentHealth,
    HealthCheck,
    CircuitBreaker,
    HealthMonitor,
    ReadinessProbe,
    LivenessProbe,
    get_health_monitor
)

from .alerting import (
    AlertSeverity,
    AlertCategory,
    AlertStatus,
    Alert,
    AlertRule,
    AlertChannel,
    AlertManager,
    AnomalyDetector,
    AlertCorrelator,
    get_alert_manager
)

__all__ = [
    # Logging
    'LogLevel',
    'LogContext',
    'StructuredLogger',
    'LoggerFactory',
    'logger',

    # Metrics
    'MetricType',
    'MetricUnit',
    'MetricDefinition',
    'MetricsCollector',
    'BusinessMetrics',
    'get_metrics_collector',
    'get_business_metrics',

    # Tracing
    'SpanKind',
    'SpanStatus',
    'Span',
    'Trace',
    'Tracer',
    'TradingTracer',
    'get_tracer',
    'trace',

    # Health
    'HealthStatus',
    'ComponentType',
    'CircuitBreakerState',
    'HealthCheckResult',
    'ComponentHealth',
    'HealthCheck',
    'CircuitBreaker',
    'HealthMonitor',
    'ReadinessProbe',
    'LivenessProbe',
    'get_health_monitor',

    # Alerting
    'AlertSeverity',
    'AlertCategory',
    'AlertStatus',
    'Alert',
    'AlertRule',
    'AlertChannel',
    'AlertManager',
    'AnomalyDetector',
    'AlertCorrelator',
    'get_alert_manager'
]