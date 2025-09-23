"""
Infrastructure Layer: Metrics Collection System
==============================================

Système de collecte de métriques pour monitoring temps réel.
Compatible avec Prometheus, StatsD, DataDog, etc.
"""

import time
import threading
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Deque
import numpy as np
from threading import Lock
import json


class MetricType(str, Enum):
    """Types de métriques supportés"""
    COUNTER = "counter"      # Toujours croissant (ex: nombre de trades)
    GAUGE = "gauge"          # Peut monter/descendre (ex: P&L)
    HISTOGRAM = "histogram"  # Distribution (ex: latences)
    SUMMARY = "summary"      # Percentiles (ex: trade sizes)


class MetricUnit(str, Enum):
    """Unités de mesure pour les métriques"""
    COUNT = "count"
    PERCENTAGE = "percentage"
    MILLISECONDS = "milliseconds"
    SECONDS = "seconds"
    BYTES = "bytes"
    DOLLARS = "dollars"
    BASIS_POINTS = "basis_points"


@dataclass
class MetricDefinition:
    """Définition d'une métrique"""
    name: str
    type: MetricType
    unit: MetricUnit
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # Pour histogrammes
    max_age_seconds: int = 3600  # Temps max de rétention


@dataclass
class MetricPoint:
    """Point de données pour une métrique"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


class MetricStorage:
    """Stockage en mémoire pour les métriques"""

    def __init__(self, max_points: int = 10000):
        self.max_points = max_points
        self._data: Dict[str, Deque[MetricPoint]] = defaultdict(lambda: deque(maxlen=max_points))
        self._lock = Lock()

    def add_point(self, metric_name: str, point: MetricPoint):
        """Ajouter un point de métrique"""
        with self._lock:
            self._data[metric_name].append(point)

    def get_points(
        self,
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[MetricPoint]:
        """Récupérer les points d'une métrique"""
        with self._lock:
            points = list(self._data.get(metric_name, []))

        if start_time or end_time:
            filtered_points = []
            for point in points:
                if start_time and point.timestamp < start_time:
                    continue
                if end_time and point.timestamp > end_time:
                    continue
                filtered_points.append(point)
            return filtered_points

        return points

    def get_latest(self, metric_name: str) -> Optional[MetricPoint]:
        """Récupérer le dernier point d'une métrique"""
        with self._lock:
            points = self._data.get(metric_name)
            return points[-1] if points else None

    def clear(self, metric_name: Optional[str] = None):
        """Effacer les données"""
        with self._lock:
            if metric_name:
                self._data[metric_name].clear()
            else:
                self._data.clear()


class MetricsCollector:
    """
    Collecteur principal de métriques.
    Supporte différents types de métriques et export vers différents backends.
    """

    def __init__(self, namespace: str = "qframe"):
        self.namespace = namespace
        self._metrics: Dict[str, MetricDefinition] = {}
        self._storage = MetricStorage()
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = Lock()

        # Métriques système par défaut
        self._register_system_metrics()

    def register_metric(self, definition: MetricDefinition):
        """Enregistrer une nouvelle métrique"""
        full_name = f"{self.namespace}.{definition.name}"
        self._metrics[full_name] = definition

    def _register_system_metrics(self):
        """Enregistrer les métriques système par défaut"""
        # Trading metrics
        self.register_metric(MetricDefinition(
            name="trades.executed",
            type=MetricType.COUNTER,
            unit=MetricUnit.COUNT,
            description="Number of trades executed",
            labels=["symbol", "side", "strategy"]
        ))

        self.register_metric(MetricDefinition(
            name="trades.volume",
            type=MetricType.COUNTER,
            unit=MetricUnit.DOLLARS,
            description="Total trade volume in dollars",
            labels=["symbol", "side"]
        ))

        self.register_metric(MetricDefinition(
            name="trades.latency",
            type=MetricType.HISTOGRAM,
            unit=MetricUnit.MILLISECONDS,
            description="Trade execution latency",
            labels=["venue"],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
        ))

        # Portfolio metrics
        self.register_metric(MetricDefinition(
            name="portfolio.value",
            type=MetricType.GAUGE,
            unit=MetricUnit.DOLLARS,
            description="Total portfolio value",
            labels=["portfolio_id"]
        ))

        self.register_metric(MetricDefinition(
            name="portfolio.pnl",
            type=MetricType.GAUGE,
            unit=MetricUnit.DOLLARS,
            description="Portfolio P&L",
            labels=["portfolio_id", "timeframe"]
        ))

        self.register_metric(MetricDefinition(
            name="portfolio.sharpe_ratio",
            type=MetricType.GAUGE,
            unit=MetricUnit.COUNT,
            description="Portfolio Sharpe ratio",
            labels=["portfolio_id"]
        ))

        # Risk metrics
        self.register_metric(MetricDefinition(
            name="risk.var",
            type=MetricType.GAUGE,
            unit=MetricUnit.DOLLARS,
            description="Value at Risk",
            labels=["portfolio_id", "confidence_level"]
        ))

        self.register_metric(MetricDefinition(
            name="risk.exposure",
            type=MetricType.GAUGE,
            unit=MetricUnit.DOLLARS,
            description="Market exposure",
            labels=["portfolio_id", "asset_class", "sector"]
        ))

        self.register_metric(MetricDefinition(
            name="risk.breaches",
            type=MetricType.COUNTER,
            unit=MetricUnit.COUNT,
            description="Risk limit breaches",
            labels=["metric", "severity"]
        ))

        # System metrics
        self.register_metric(MetricDefinition(
            name="system.api_requests",
            type=MetricType.COUNTER,
            unit=MetricUnit.COUNT,
            description="API requests",
            labels=["endpoint", "method", "status"]
        ))

        self.register_metric(MetricDefinition(
            name="system.api_latency",
            type=MetricType.HISTOGRAM,
            unit=MetricUnit.MILLISECONDS,
            description="API request latency",
            labels=["endpoint", "method"],
            buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
        ))

        self.register_metric(MetricDefinition(
            name="system.database_queries",
            type=MetricType.COUNTER,
            unit=MetricUnit.COUNT,
            description="Database queries",
            labels=["query_type", "table"]
        ))

        self.register_metric(MetricDefinition(
            name="system.errors",
            type=MetricType.COUNTER,
            unit=MetricUnit.COUNT,
            description="System errors",
            labels=["error_type", "severity", "component"]
        ))

    def increment_counter(self, name: str, value: float = 1, labels: Optional[Dict[str, str]] = None):
        """Incrémenter un compteur"""
        full_name = f"{self.namespace}.{name}"

        if full_name not in self._metrics:
            raise ValueError(f"Metric {full_name} not registered")

        if self._metrics[full_name].type != MetricType.COUNTER:
            raise ValueError(f"Metric {full_name} is not a counter")

        key = self._make_key(full_name, labels)

        with self._lock:
            self._counters[key] += value

        # Stocker le point
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=self._counters[key],
            labels=labels or {}
        )
        self._storage.add_point(full_name, point)

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Définir la valeur d'un gauge"""
        full_name = f"{self.namespace}.{name}"

        if full_name not in self._metrics:
            raise ValueError(f"Metric {full_name} not registered")

        if self._metrics[full_name].type != MetricType.GAUGE:
            raise ValueError(f"Metric {full_name} is not a gauge")

        key = self._make_key(full_name, labels)

        with self._lock:
            self._gauges[key] = value

        # Stocker le point
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=value,
            labels=labels or {}
        )
        self._storage.add_point(full_name, point)

    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Enregistrer une valeur dans un histogramme"""
        full_name = f"{self.namespace}.{name}"

        if full_name not in self._metrics:
            raise ValueError(f"Metric {full_name} not registered")

        if self._metrics[full_name].type != MetricType.HISTOGRAM:
            raise ValueError(f"Metric {full_name} is not a histogram")

        key = self._make_key(full_name, labels)

        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(value)

            # Limiter la taille de l'histogramme
            if len(self._histograms[key]) > 10000:
                self._histograms[key] = self._histograms[key][-10000:]

        # Stocker le point
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=value,
            labels=labels or {}
        )
        self._storage.add_point(full_name, point)

    def _make_key(self, metric_name: str, labels: Optional[Dict[str, str]]) -> str:
        """Créer une clé unique pour une métrique avec labels"""
        if not labels:
            return metric_name

        sorted_labels = sorted(labels.items())
        label_str = ",".join(f"{k}={v}" for k, v in sorted_labels)
        return f"{metric_name}{{{label_str}}}"

    def get_counter_value(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Obtenir la valeur actuelle d'un compteur"""
        full_name = f"{self.namespace}.{name}"
        key = self._make_key(full_name, labels)
        return self._counters.get(key, 0)

    def get_gauge_value(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Obtenir la valeur actuelle d'un gauge"""
        full_name = f"{self.namespace}.{name}"
        key = self._make_key(full_name, labels)
        return self._gauges.get(key, 0)

    def get_histogram_stats(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """Obtenir les statistiques d'un histogramme"""
        full_name = f"{self.namespace}.{name}"
        key = self._make_key(full_name, labels)

        with self._lock:
            values = self._histograms.get(key, [])

        if not values:
            return {
                "count": 0,
                "sum": 0,
                "mean": 0,
                "min": 0,
                "max": 0,
                "p50": 0,
                "p90": 0,
                "p95": 0,
                "p99": 0
            }

        return {
            "count": len(values),
            "sum": sum(values),
            "mean": np.mean(values),
            "min": min(values),
            "max": max(values),
            "p50": np.percentile(values, 50),
            "p90": np.percentile(values, 90),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99)
        }

    def measure_time(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager pour mesurer le temps d'exécution"""
        return TimeMeasurement(self, name, labels)

    def export_prometheus(self) -> str:
        """Exporter les métriques au format Prometheus"""
        lines = []

        # Counters
        for key, value in self._counters.items():
            lines.append(f"{key} {value}")

        # Gauges
        for key, value in self._gauges.items():
            lines.append(f"{key} {value}")

        # Histograms
        for key, values in self._histograms.items():
            if values:
                stats = self.get_histogram_stats(key.split("{")[0], None)
                base_name = key.split("{")[0]
                labels = key.split("{")[1].rstrip("}") if "{" in key else ""

                lines.append(f'{base_name}_count{{{labels}}} {stats["count"]}')
                lines.append(f'{base_name}_sum{{{labels}}} {stats["sum"]}')

                # Percentiles
                for percentile in [50, 90, 95, 99]:
                    p_label = f'quantile="{percentile/100}"'
                    if labels:
                        p_label = f'{labels},{p_label}'
                    lines.append(f'{base_name}{{{p_label}}} {stats[f"p{percentile}"]}')

        return "\n".join(lines)

    def export_json(self) -> str:
        """Exporter les métriques au format JSON"""
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "namespace": self.namespace,
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {}
        }

        for key, values in self._histograms.items():
            if values:
                data["histograms"][key] = self.get_histogram_stats(key.split("{")[0], None)

        return json.dumps(data, indent=2)

    def reset(self):
        """Réinitialiser toutes les métriques"""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
        self._storage.clear()


class TimeMeasurement:
    """Context manager pour mesurer le temps d'exécution"""

    def __init__(self, collector: MetricsCollector, metric_name: str, labels: Optional[Dict[str, str]]):
        self.collector = collector
        self.metric_name = metric_name
        self.labels = labels
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        self.collector.record_histogram(self.metric_name, duration_ms, self.labels)


class BusinessMetrics:
    """
    Métriques business spécifiques au trading.
    Wrapper autour de MetricsCollector avec des méthodes spécialisées.
    """

    def __init__(self, collector: MetricsCollector):
        self.collector = collector

    def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        venue: str,
        strategy: str,
        latency_ms: float
    ):
        """Enregistrer les métriques d'un trade"""
        # Compteur de trades
        self.collector.increment_counter(
            "trades.executed",
            labels={"symbol": symbol, "side": side, "strategy": strategy}
        )

        # Volume en dollars
        volume = float(quantity * price)
        self.collector.increment_counter(
            "trades.volume",
            value=volume,
            labels={"symbol": symbol, "side": side}
        )

        # Latence d'exécution
        self.collector.record_histogram(
            "trades.latency",
            latency_ms,
            labels={"venue": venue}
        )

    def update_portfolio_metrics(
        self,
        portfolio_id: str,
        total_value: Decimal,
        pnl: Decimal,
        sharpe_ratio: float,
        max_drawdown: float
    ):
        """Mettre à jour les métriques du portfolio"""
        self.collector.set_gauge(
            "portfolio.value",
            float(total_value),
            labels={"portfolio_id": portfolio_id}
        )

        self.collector.set_gauge(
            "portfolio.pnl",
            float(pnl),
            labels={"portfolio_id": portfolio_id, "timeframe": "daily"}
        )

        self.collector.set_gauge(
            "portfolio.sharpe_ratio",
            sharpe_ratio,
            labels={"portfolio_id": portfolio_id}
        )

        self.collector.set_gauge(
            "portfolio.max_drawdown",
            max_drawdown,
            labels={"portfolio_id": portfolio_id}
        )

    def record_risk_breach(
        self,
        metric: str,
        severity: str,
        portfolio_id: str,
        current_value: float,
        limit: float
    ):
        """Enregistrer une violation de limite de risque"""
        self.collector.increment_counter(
            "risk.breaches",
            labels={"metric": metric, "severity": severity}
        )

        # Log l'écart par rapport à la limite
        breach_percentage = ((current_value - limit) / limit) * 100 if limit != 0 else 0

        self.collector.set_gauge(
            "risk.breach_percentage",
            breach_percentage,
            labels={"portfolio_id": portfolio_id, "metric": metric}
        )

    def record_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        latency_ms: float
    ):
        """Enregistrer une requête API"""
        self.collector.increment_counter(
            "system.api_requests",
            labels={"endpoint": endpoint, "method": method, "status": str(status_code)}
        )

        self.collector.record_histogram(
            "system.api_latency",
            latency_ms,
            labels={"endpoint": endpoint, "method": method}
        )

    def record_error(
        self,
        error_type: str,
        severity: str,
        component: str
    ):
        """Enregistrer une erreur système"""
        self.collector.increment_counter(
            "system.errors",
            labels={"error_type": error_type, "severity": severity, "component": component}
        )


# Instance globale pour faciliter l'accès
_global_collector = MetricsCollector()
business_metrics = BusinessMetrics(_global_collector)


def get_metrics_collector() -> MetricsCollector:
    """Obtenir l'instance globale du collecteur de métriques"""
    return _global_collector


def get_business_metrics() -> BusinessMetrics:
    """Obtenir l'instance globale des métriques business"""
    return business_metrics