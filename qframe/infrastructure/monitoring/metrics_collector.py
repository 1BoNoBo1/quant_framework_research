"""
Advanced Metrics Collection and Monitoring for QFrame Enterprise
===============================================================

Système de collecte de métriques et monitoring enterprise avec
intégration Prometheus, alerting, et tableaux de bord temps réel.
"""

import time
import threading
import json
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable
import statistics

try:
    import prometheus_client
    from prometheus_client import Counter, Gauge, Histogram, Summary
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from qframe.infrastructure.observability.structured_logging import (
    StructuredLogger, LoggerFactory, PerformanceLogger
)


class MetricType(str, Enum):
    """Types de métriques supportées."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


class AlertSeverity(str, Enum):
    """Niveaux de sévérité des alertes."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricDefinition:
    """Définition d'une métrique."""
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # Pour histogrammes
    unit: Optional[str] = None


@dataclass
class MetricValue:
    """Valeur de métrique avec métadonnées."""
    name: str
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    unit: Optional[str] = None


class AlertRule:
    """Règle d'alerte sur métriques."""

    def __init__(
        self,
        name: str,
        metric_name: str,
        condition: Callable[[float], bool],
        severity: AlertSeverity = AlertSeverity.WARNING,
        message_template: str = "Alert {name}: {metric_name} = {value}",
        cooldown_minutes: int = 5
    ):
        self.name = name
        self.metric_name = metric_name
        self.condition = condition
        self.severity = severity
        self.message_template = message_template
        self.cooldown_minutes = cooldown_minutes
        self.last_alert_time: Optional[datetime] = None

    def should_alert(self, value: float) -> bool:
        """Vérifie si une alerte doit être déclenchée."""
        if not self.condition(value):
            return False

        # Vérifier cooldown
        if self.last_alert_time:
            time_since_last = datetime.now(timezone.utc) - self.last_alert_time
            if time_since_last.total_seconds() < self.cooldown_minutes * 60:
                return False

        return True

    def format_message(self, value: float) -> str:
        """Formate le message d'alerte."""
        return self.message_template.format(
            name=self.name,
            metric_name=self.metric_name,
            value=value
        )


@dataclass
class Alert:
    """Alerte déclenchée."""
    rule_name: str
    metric_name: str
    value: float
    severity: AlertSeverity
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class MetricStorage(ABC):
    """Interface abstraite pour stockage de métriques."""

    @abstractmethod
    def store_metric(self, metric: MetricValue) -> None:
        """Stocke une métrique."""
        pass

    @abstractmethod
    def get_metrics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        labels: Optional[Dict[str, str]] = None
    ) -> List[MetricValue]:
        """Récupère des métriques."""
        pass

    @abstractmethod
    def get_latest_value(self, metric_name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Récupère la dernière valeur d'une métrique."""
        pass


class InMemoryMetricStorage(MetricStorage):
    """Stockage en mémoire pour développement/tests."""

    def __init__(self, max_retention_hours: int = 24):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.max_retention_hours = max_retention_hours
        self.lock = threading.RLock()

    def store_metric(self, metric: MetricValue) -> None:
        """Stocke une métrique en mémoire."""
        key = self._make_key(metric.name, metric.labels)

        with self.lock:
            self.metrics[key].append(metric)
            self._cleanup_old_metrics()

    def get_metrics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        labels: Optional[Dict[str, str]] = None
    ) -> List[MetricValue]:
        """Récupère des métriques par période."""
        key = self._make_key(metric_name, labels or {})

        with self.lock:
            if key not in self.metrics:
                return []

            filtered_metrics = []
            for metric in self.metrics[key]:
                if start_time <= metric.timestamp <= end_time:
                    filtered_metrics.append(metric)

            return sorted(filtered_metrics, key=lambda m: m.timestamp)

    def get_latest_value(self, metric_name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Récupère la dernière valeur."""
        key = self._make_key(metric_name, labels or {})

        with self.lock:
            if key not in self.metrics or not self.metrics[key]:
                return None

            return self.metrics[key][-1].value

    def _make_key(self, metric_name: str, labels: Dict[str, str]) -> str:
        """Crée une clé unique pour la métrique."""
        if not labels:
            return metric_name

        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{metric_name}#{label_str}"

    def _cleanup_old_metrics(self) -> None:
        """Nettoie les métriques expirées."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.max_retention_hours)

        for key, metric_deque in self.metrics.items():
            # Supprimer métriques expirées du début de la deque
            while metric_deque and metric_deque[0].timestamp < cutoff_time:
                metric_deque.popleft()


class PrometheusAdapter:
    """Adaptateur pour Prometheus."""

    def __init__(self, enabled: bool = HAS_PROMETHEUS):
        self.enabled = enabled and HAS_PROMETHEUS
        self.metrics: Dict[str, Any] = {}

        if self.enabled:
            # Registre Prometheus personnalisé
            self.registry = prometheus_client.CollectorRegistry()

    def register_metric(self, definition: MetricDefinition) -> None:
        """Enregistre une métrique Prometheus."""
        if not self.enabled:
            return

        if definition.name in self.metrics:
            return

        if definition.metric_type == MetricType.COUNTER:
            metric = Counter(
                definition.name,
                definition.description,
                labelnames=definition.labels,
                registry=self.registry
            )
        elif definition.metric_type == MetricType.GAUGE:
            metric = Gauge(
                definition.name,
                definition.description,
                labelnames=definition.labels,
                registry=self.registry
            )
        elif definition.metric_type == MetricType.HISTOGRAM:
            # Buckets par défaut si pas spécifiés
            default_buckets = (.005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5, 5.0, 7.5, 10.0, float('inf'))
            buckets = definition.buckets or default_buckets
            metric = Histogram(
                definition.name,
                definition.description,
                labelnames=definition.labels,
                buckets=buckets,
                registry=self.registry
            )
        elif definition.metric_type == MetricType.SUMMARY:
            metric = Summary(
                definition.name,
                definition.description,
                labelnames=definition.labels,
                registry=self.registry
            )
        else:
            return

        self.metrics[definition.name] = metric

    def update_metric(self, metric_value: MetricValue) -> None:
        """Met à jour une métrique Prometheus."""
        if not self.enabled or metric_value.name not in self.metrics:
            return

        metric = self.metrics[metric_value.name]
        label_values = [metric_value.labels.get(label, "") for label in metric._labelnames]

        if isinstance(metric, Counter):
            if label_values:
                metric.labels(*label_values).inc(metric_value.value)
            else:
                metric.inc(metric_value.value)

        elif isinstance(metric, Gauge):
            if label_values:
                metric.labels(*label_values).set(metric_value.value)
            else:
                metric.set(metric_value.value)

        elif isinstance(metric, (Histogram, Summary)):
            if label_values:
                metric.labels(*label_values).observe(metric_value.value)
            else:
                metric.observe(metric_value.value)

    def get_metrics_text(self) -> str:
        """Retourne les métriques au format Prometheus."""
        if not self.enabled:
            return ""

        return prometheus_client.generate_latest(self.registry).decode('utf-8')


class SystemMetricsCollector:
    """Collecteur de métriques système."""

    def __init__(self, enabled: bool = HAS_PSUTIL):
        self.enabled = enabled and HAS_PSUTIL
        self.logger = LoggerFactory.get_logger("system_metrics")

    def collect_system_metrics(self) -> List[MetricValue]:
        """Collecte des métriques système."""
        if not self.enabled:
            return []

        metrics = []
        timestamp = datetime.now(timezone.utc)

        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=None)
            metrics.append(MetricValue(
                name="system_cpu_usage_percent",
                value=cpu_percent,
                timestamp=timestamp,
                unit="percent"
            ))

            # Mémoire
            memory = psutil.virtual_memory()
            metrics.append(MetricValue(
                name="system_memory_usage_percent",
                value=memory.percent,
                timestamp=timestamp,
                unit="percent"
            ))

            metrics.append(MetricValue(
                name="system_memory_used_bytes",
                value=memory.used,
                timestamp=timestamp,
                unit="bytes"
            ))

            # Disque
            disk = psutil.disk_usage('/')
            metrics.append(MetricValue(
                name="system_disk_usage_percent",
                value=(disk.used / disk.total) * 100,
                timestamp=timestamp,
                unit="percent"
            ))

            # Réseau (si disponible)
            try:
                net_io = psutil.net_io_counters()
                metrics.append(MetricValue(
                    name="system_network_bytes_sent",
                    value=net_io.bytes_sent,
                    timestamp=timestamp,
                    unit="bytes"
                ))

                metrics.append(MetricValue(
                    name="system_network_bytes_recv",
                    value=net_io.bytes_recv,
                    timestamp=timestamp,
                    unit="bytes"
                ))
            except Exception:
                pass  # Réseau non disponible

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")

        return metrics


class MetricsCollector:
    """Collecteur principal de métriques enterprise."""

    def __init__(
        self,
        storage: Optional[MetricStorage] = None,
        enable_prometheus: bool = HAS_PROMETHEUS,
        enable_system_metrics: bool = HAS_PSUTIL
    ):
        self.storage = storage or InMemoryMetricStorage()
        self.prometheus = PrometheusAdapter(enable_prometheus)
        self.system_collector = SystemMetricsCollector(enable_system_metrics)

        self.logger = LoggerFactory.get_logger("metrics_collector")
        self.perf_logger = PerformanceLogger(self.logger)

        # Registre des définitions de métriques
        self.metric_definitions: Dict[str, MetricDefinition] = {}

        # Système d'alertes
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: List[Alert] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []

        # Threading pour collecte automatique
        self.collection_interval = 30  # secondes
        self.auto_collection_enabled = False
        self.collection_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Statistiques internes
        self.stats = {
            'metrics_collected': 0,
            'alerts_triggered': 0,
            'errors': 0
        }

        # Enregistrer métriques de base
        self._register_core_metrics()

    def _register_core_metrics(self) -> None:
        """Enregistre les métriques de base du framework."""

        core_metrics = [
            # Métriques système
            MetricDefinition("system_cpu_usage_percent", MetricType.GAUGE, "CPU usage percentage"),
            MetricDefinition("system_memory_usage_percent", MetricType.GAUGE, "Memory usage percentage"),
            MetricDefinition("system_disk_usage_percent", MetricType.GAUGE, "Disk usage percentage"),

            # Métriques trading
            MetricDefinition("trading_signals_generated", MetricType.COUNTER, "Number of trading signals generated", ["strategy", "symbol"]),
            MetricDefinition("trading_orders_executed", MetricType.COUNTER, "Number of orders executed", ["side", "symbol"]),
            MetricDefinition("trading_pnl_realized", MetricType.GAUGE, "Realized PnL", ["strategy", "symbol"]),
            MetricDefinition("trading_portfolio_value", MetricType.GAUGE, "Portfolio value", ["portfolio_id"]),

            # Métriques performance
            MetricDefinition("processing_duration_seconds", MetricType.HISTOGRAM, "Processing duration", ["operation"],
                           buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]),
            MetricDefinition("feature_calculation_duration", MetricType.HISTOGRAM, "Feature calculation duration", ["feature_type"]),

            # Métriques qualité
            MetricDefinition("data_quality_score", MetricType.GAUGE, "Data quality score", ["symbol", "timeframe"]),
            MetricDefinition("model_accuracy", MetricType.GAUGE, "Model accuracy", ["model_name", "symbol"]),

            # Métriques erreurs
            MetricDefinition("errors_total", MetricType.COUNTER, "Total number of errors", ["error_type", "component"]),
            MetricDefinition("api_requests_total", MetricType.COUNTER, "Total API requests", ["endpoint", "status"])
        ]

        for metric_def in core_metrics:
            self.register_metric(metric_def)

    def register_metric(self, definition: MetricDefinition) -> None:
        """Enregistre une nouvelle métrique."""
        self.metric_definitions[definition.name] = definition
        self.prometheus.register_metric(definition)

        self.logger.debug(f"Registered metric: {definition.name}")

    def record_metric(
        self,
        name: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None,
        unit: Optional[str] = None
    ) -> None:
        """Enregistre une valeur de métrique."""

        if name not in self.metric_definitions:
            self.logger.warning(f"Unknown metric: {name}")
            return

        metric_value = MetricValue(
            name=name,
            value=value,
            labels=labels or {},
            unit=unit
        )

        # Stocker
        self.storage.store_metric(metric_value)

        # Prometheus
        self.prometheus.update_metric(metric_value)

        # Vérifier alertes
        self._check_alerts(metric_value)

        self.stats['metrics_collected'] += 1

    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None, amount: float = 1.0) -> None:
        """Incrémente un compteur."""
        self.record_metric(name, amount, labels)

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Définit la valeur d'une gauge."""
        self.record_metric(name, value, labels)

    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe une valeur pour un histogramme."""
        self.record_metric(name, value, labels)

    def time_operation(self, operation_name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager pour mesurer la durée d'une opération."""
        return TimerContext(self, operation_name, labels)

    def add_alert_rule(self, rule: AlertRule) -> None:
        """Ajoute une règle d'alerte."""
        self.alert_rules[rule.name] = rule
        self.logger.info(f"Added alert rule: {rule.name} for metric {rule.metric_name}")

    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Ajoute un callback pour les alertes."""
        self.alert_callbacks.append(callback)

    def _check_alerts(self, metric_value: MetricValue) -> None:
        """Vérifie les règles d'alerte pour une métrique."""
        for rule in self.alert_rules.values():
            if rule.metric_name != metric_value.name:
                continue

            if rule.should_alert(metric_value.value):
                alert = Alert(
                    rule_name=rule.name,
                    metric_name=metric_value.name,
                    value=metric_value.value,
                    severity=rule.severity,
                    message=rule.format_message(metric_value.value)
                )

                self.active_alerts.append(alert)
                rule.last_alert_time = datetime.now(timezone.utc)

                # Notifier callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        self.logger.error(f"Alert callback error: {e}")

                self.stats['alerts_triggered'] += 1
                self.logger.warning(f"Alert triggered: {alert.message}")

    def get_metric_history(
        self,
        metric_name: str,
        hours: int = 1,
        labels: Optional[Dict[str, str]] = None
    ) -> List[MetricValue]:
        """Récupère l'historique d'une métrique."""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)

        return self.storage.get_metrics(metric_name, start_time, end_time, labels)

    def get_current_value(self, metric_name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Récupère la valeur actuelle d'une métrique."""
        return self.storage.get_latest_value(metric_name, labels)

    def get_metric_summary(self, metric_name: str, hours: int = 1) -> Dict[str, float]:
        """Calcule un résumé statistique d'une métrique."""
        history = self.get_metric_history(metric_name, hours)

        if not history:
            return {}

        values = [m.value for m in history]

        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0
        }

    def start_auto_collection(self, interval_seconds: int = 30) -> None:
        """Démarre la collecte automatique de métriques."""
        if self.auto_collection_enabled:
            return

        self.collection_interval = interval_seconds
        self.auto_collection_enabled = True
        self.stop_event.clear()

        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True
        )
        self.collection_thread.start()

        self.logger.info(f"Started automatic metrics collection (interval: {interval_seconds}s)")

    def stop_auto_collection(self) -> None:
        """Arrête la collecte automatique."""
        if not self.auto_collection_enabled:
            return

        self.auto_collection_enabled = False
        self.stop_event.set()

        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)

        self.logger.info("Stopped automatic metrics collection")

    def _collection_loop(self) -> None:
        """Boucle de collecte automatique."""
        while self.auto_collection_enabled and not self.stop_event.wait(self.collection_interval):
            try:
                # Collecter métriques système
                system_metrics = self.system_collector.collect_system_metrics()
                for metric in system_metrics:
                    self.storage.store_metric(metric)
                    self.prometheus.update_metric(metric)

                # Métriques internes du collecteur
                self.set_gauge("metrics_collector_total_collected", self.stats['metrics_collected'])
                self.set_gauge("metrics_collector_alerts_triggered", self.stats['alerts_triggered'])
                self.set_gauge("metrics_collector_active_alerts", len(self.active_alerts))

            except Exception as e:
                self.stats['errors'] += 1
                self.logger.error(f"Error in metrics collection loop: {e}")

    def get_prometheus_metrics(self) -> str:
        """Retourne les métriques au format Prometheus."""
        return self.prometheus.get_metrics_text()

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Récupère les données pour tableau de bord."""
        dashboard_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'stats': self.stats.copy(),
            'active_alerts': [
                {
                    'rule_name': alert.rule_name,
                    'metric_name': alert.metric_name,
                    'value': alert.value,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in self.active_alerts[-10:]  # Dernières 10 alertes
            ],
            'system_status': {}
        }

        # Ajouter métriques système actuelles
        current_metrics = [
            'system_cpu_usage_percent',
            'system_memory_usage_percent',
            'system_disk_usage_percent'
        ]

        for metric_name in current_metrics:
            value = self.get_current_value(metric_name)
            if value is not None:
                dashboard_data['system_status'][metric_name] = value

        return dashboard_data


class TimerContext:
    """Context manager pour mesurer les durées."""

    def __init__(self, collector: MetricsCollector, operation_name: str, labels: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.operation_name = operation_name
        self.labels = labels or {}
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.collector.observe_histogram(
                "processing_duration_seconds",
                duration,
                {**self.labels, "operation": self.operation_name}
            )


# Décorateur pour mesurer automatiquement les performances
def monitored(metric_name: str = None, labels: Optional[Dict[str, str]] = None):
    """Décorateur pour monitoring automatique des fonctions."""

    def decorator(func: Callable) -> Callable:
        operation_name = metric_name or f"{func.__module__}.{func.__name__}"

        def wrapper(*args, **kwargs):
            # Essayer de récupérer le collector depuis les arguments
            collector = None
            for arg in args:
                if hasattr(arg, 'metrics_collector') and isinstance(arg.metrics_collector, MetricsCollector):
                    collector = arg.metrics_collector
                    break

            if not collector:
                # Utiliser collector global par défaut
                collector = get_default_collector()

            with collector.time_operation(operation_name, labels):
                return func(*args, **kwargs)

        return wrapper
    return decorator


# Collector global
_default_collector: Optional[MetricsCollector] = None

def get_default_collector() -> MetricsCollector:
    """Récupère le collector par défaut."""
    global _default_collector
    if _default_collector is None:
        _default_collector = MetricsCollector()
    return _default_collector


def configure_monitoring(
    enable_prometheus: bool = True,
    enable_system_metrics: bool = True,
    storage: Optional[MetricStorage] = None,
    auto_collection_interval: int = 30
) -> MetricsCollector:
    """Configure le système de monitoring."""
    global _default_collector

    _default_collector = MetricsCollector(
        storage=storage,
        enable_prometheus=enable_prometheus,
        enable_system_metrics=enable_system_metrics
    )

    if auto_collection_interval > 0:
        _default_collector.start_auto_collection(auto_collection_interval)

    return _default_collector


if __name__ == "__main__":
    # Démonstration du système de monitoring
    print("📊 Démonstration Metrics Collector QFrame")

    # Configuration
    collector = configure_monitoring(
        enable_prometheus=HAS_PROMETHEUS,
        enable_system_metrics=HAS_PSUTIL,
        auto_collection_interval=5
    )

    # Ajouter règles d'alerte
    high_cpu_rule = AlertRule(
        name="high_cpu_usage",
        metric_name="system_cpu_usage_percent",
        condition=lambda x: x > 80,
        severity=AlertSeverity.WARNING,
        message_template="🚨 High CPU usage: {value:.1f}%"
    )
    collector.add_alert_rule(high_cpu_rule)

    # Callback d'alerte
    def alert_handler(alert: Alert):
        print(f"🔔 ALERT: {alert.message}")

    collector.add_alert_callback(alert_handler)

    # Enregistrer quelques métriques
    print("📈 Recording sample metrics...")

    with collector.time_operation("sample_calculation"):
        time.sleep(0.1)

    collector.increment_counter("trading_signals_generated", {"strategy": "momentum", "symbol": "BTC/USD"})
    collector.set_gauge("trading_portfolio_value", 150000.0, {"portfolio_id": "main"})
    collector.observe_histogram("feature_calculation_duration", 0.25, {"feature_type": "technical"})

    # Afficher statistiques
    stats = collector.get_dashboard_data()
    print(f"✅ Dashboard data available")

    # Arrêter collecte
    collector.stop_auto_collection()
    print("🏁 Monitoring demo completed")