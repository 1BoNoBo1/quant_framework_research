"""
Performance Monitor
==================

System performance monitoring for production trading environment.
"""

from typing import Dict, List, Optional, Any, NamedTuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass
import asyncio
import logging
import psutil
import time
import statistics
from collections import deque

from ...core.container import injectable
from .metrics_collector import MetricsCollector


logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """Métriques système à un instant donné"""
    timestamp: datetime

    # CPU
    cpu_percent: float
    cpu_count: int
    load_average: Optional[List[float]]  # 1, 5, 15 minutes

    # Mémoire
    memory_total: int  # bytes
    memory_available: int  # bytes
    memory_percent: float
    memory_used: int  # bytes

    # Disque
    disk_total: int  # bytes
    disk_used: int  # bytes
    disk_free: int  # bytes
    disk_percent: float

    # Réseau
    network_bytes_sent: int
    network_bytes_recv: int
    network_packets_sent: int
    network_packets_recv: int

    # Processus Python
    process_memory_rss: int  # bytes
    process_memory_vms: int  # bytes
    process_cpu_percent: float
    process_num_threads: int
    process_open_files: int

    # Métriques trading spécifiques
    active_strategies: int
    active_orders: int
    websocket_connections: int
    database_connections: int


class PerformanceThreshold(NamedTuple):
    """Seuil de performance"""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    unit: str


@injectable
class PerformanceMonitor:
    """
    Moniteur de performance système pour environnement de trading.

    Surveille:
    - Utilisation CPU, mémoire, disque
    - Performance réseau
    - Latence des composants
    - Health checks des services
    - Métriques application-spécifiques
    """

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        monitoring_interval: int = 30,  # secondes
        history_retention: timedelta = timedelta(hours=24)
    ):
        self.metrics_collector = metrics_collector
        self.monitoring_interval = monitoring_interval
        self.history_retention = history_retention

        # Historique des métriques
        self.metrics_history: deque = deque(maxlen=2880)  # 24h à 30s d'intervalle

        # État du monitoring
        self.is_monitoring = False
        self.last_metrics: Optional[SystemMetrics] = None

        # Configuration des seuils
        self.performance_thresholds = self._setup_default_thresholds()

        # Cache des processus et informations système
        self.process = psutil.Process()
        self.boot_time = psutil.boot_time()

        # Métriques dérivées
        self.performance_scores: Dict[str, float] = {}
        self.health_status = "unknown"

    async def start_monitoring(self) -> None:
        """Démarre le monitoring de performance"""
        if self.is_monitoring:
            logger.warning("Performance monitoring already started")
            return

        self.is_monitoring = True
        logger.info("Starting performance monitoring")

        # Démarrer boucle de monitoring
        asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self) -> None:
        """Arrête le monitoring de performance"""
        self.is_monitoring = False
        logger.info("Performance monitoring stopped")

    async def get_current_metrics(self) -> SystemMetrics:
        """Collecte les métriques système actuelles"""
        current_time = datetime.utcnow()

        try:
            # Métriques CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            try:
                load_avg = list(psutil.getloadavg())
            except AttributeError:
                load_avg = None  # Pas disponible sur Windows

            # Métriques mémoire
            memory = psutil.virtual_memory()

            # Métriques disque (partition principale)
            disk = psutil.disk_usage('/')

            # Métriques réseau
            network = psutil.net_io_counters()

            # Métriques du processus Python actuel
            proc_memory = self.process.memory_info()
            proc_cpu = self.process.cpu_percent()

            try:
                proc_threads = self.process.num_threads()
                proc_files = len(self.process.open_files())
            except (psutil.AccessDenied, OSError):
                proc_threads = 0
                proc_files = 0

            # Métriques trading (simulées - dans la réalité viendraient des services)
            trading_metrics = await self._get_trading_metrics()

            metrics = SystemMetrics(
                timestamp=current_time,
                cpu_percent=cpu_percent,
                cpu_count=cpu_count,
                load_average=load_avg,
                memory_total=memory.total,
                memory_available=memory.available,
                memory_percent=memory.percent,
                memory_used=memory.used,
                disk_total=disk.total,
                disk_used=disk.used,
                disk_free=disk.free,
                disk_percent=disk.used / disk.total * 100,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                network_packets_sent=network.packets_sent,
                network_packets_recv=network.packets_recv,
                process_memory_rss=proc_memory.rss,
                process_memory_vms=proc_memory.vms,
                process_cpu_percent=proc_cpu,
                process_num_threads=proc_threads,
                process_open_files=proc_files,
                **trading_metrics
            )

            self.last_metrics = metrics
            return metrics

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            # Retourner métriques minimales en cas d'erreur
            return SystemMetrics(
                timestamp=current_time,
                cpu_percent=0, cpu_count=1, load_average=None,
                memory_total=0, memory_available=0, memory_percent=0, memory_used=0,
                disk_total=0, disk_used=0, disk_free=0, disk_percent=0,
                network_bytes_sent=0, network_bytes_recv=0,
                network_packets_sent=0, network_packets_recv=0,
                process_memory_rss=0, process_memory_vms=0,
                process_cpu_percent=0, process_num_threads=0, process_open_files=0,
                active_strategies=0, active_orders=0,
                websocket_connections=0, database_connections=0
            )

    async def get_performance_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de performance"""
        if not self.last_metrics:
            await self.get_current_metrics()

        current = self.last_metrics
        if not current:
            return {"error": "No metrics available"}

        # Calculer scores de performance
        await self._calculate_performance_scores()

        # Analyser tendances
        trends = await self._analyze_trends()

        # Vérifier seuils
        threshold_violations = await self._check_thresholds(current)

        return {
            "timestamp": current.timestamp.isoformat(),
            "health_status": self.health_status,
            "overall_score": self.performance_scores.get("overall", 0),
            "current_metrics": {
                "cpu_percent": current.cpu_percent,
                "memory_percent": current.memory_percent,
                "disk_percent": current.disk_percent,
                "process_cpu": current.process_cpu_percent,
                "process_memory_mb": current.process_memory_rss / 1024 / 1024,
                "active_strategies": current.active_strategies,
                "active_orders": current.active_orders
            },
            "performance_scores": self.performance_scores,
            "trends": trends,
            "threshold_violations": threshold_violations,
            "uptime_hours": (time.time() - self.boot_time) / 3600,
            "monitoring_duration": len(self.metrics_history) * self.monitoring_interval / 3600
        }

    async def get_detailed_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques détaillées"""
        current = await self.get_current_metrics()

        # Métriques réseau dérivées
        network_stats = await self._calculate_network_stats()

        # Métriques de latence
        latency_stats = await self._measure_component_latency()

        return {
            "system": {
                "cpu": {
                    "percent": current.cpu_percent,
                    "count": current.cpu_count,
                    "load_average": current.load_average
                },
                "memory": {
                    "total_gb": current.memory_total / 1024**3,
                    "available_gb": current.memory_available / 1024**3,
                    "used_gb": current.memory_used / 1024**3,
                    "percent": current.memory_percent
                },
                "disk": {
                    "total_gb": current.disk_total / 1024**3,
                    "used_gb": current.disk_used / 1024**3,
                    "free_gb": current.disk_free / 1024**3,
                    "percent": current.disk_percent
                }
            },
            "process": {
                "cpu_percent": current.process_cpu_percent,
                "memory_rss_mb": current.process_memory_rss / 1024**2,
                "memory_vms_mb": current.process_memory_vms / 1024**2,
                "threads": current.process_num_threads,
                "open_files": current.process_open_files
            },
            "network": network_stats,
            "latency": latency_stats,
            "trading": {
                "active_strategies": current.active_strategies,
                "active_orders": current.active_orders,
                "websocket_connections": current.websocket_connections,
                "database_connections": current.database_connections
            }
        }

    async def get_historical_data(
        self,
        metric_name: str,
        time_range: timedelta = timedelta(hours=1)
    ) -> List[Dict[str, Any]]:
        """Retourne les données historiques pour une métrique"""
        cutoff_time = datetime.utcnow() - time_range

        historical_data = []
        for metrics in self.metrics_history:
            if metrics.timestamp >= cutoff_time:
                value = getattr(metrics, metric_name, None)
                if value is not None:
                    historical_data.append({
                        "timestamp": metrics.timestamp.isoformat(),
                        "value": value
                    })

        return historical_data

    async def run_health_check(self) -> Dict[str, Any]:
        """Effectue un health check complet du système"""
        health_results = {}

        # Check métriques système
        current = await self.get_current_metrics()

        # CPU Health
        cpu_health = "healthy"
        if current.cpu_percent > 90:
            cpu_health = "critical"
        elif current.cpu_percent > 70:
            cpu_health = "warning"

        # Memory Health
        memory_health = "healthy"
        if current.memory_percent > 95:
            memory_health = "critical"
        elif current.memory_percent > 85:
            memory_health = "warning"

        # Disk Health
        disk_health = "healthy"
        if current.disk_percent > 95:
            disk_health = "critical"
        elif current.disk_percent > 85:
            disk_health = "warning"

        # Trading Services Health
        trading_health = await self._check_trading_services_health()

        health_results = {
            "overall": "healthy",  # Sera calculé
            "components": {
                "cpu": {"status": cpu_health, "value": current.cpu_percent},
                "memory": {"status": memory_health, "value": current.memory_percent},
                "disk": {"status": disk_health, "value": current.disk_percent},
                "trading": trading_health
            },
            "timestamp": datetime.utcnow().isoformat()
        }

        # Calculer statut global
        component_statuses = [comp["status"] for comp in health_results["components"].values()]
        if "critical" in component_statuses:
            health_results["overall"] = "critical"
        elif "warning" in component_statuses:
            health_results["overall"] = "warning"

        self.health_status = health_results["overall"]
        return health_results

    # === Méthodes privées ===

    async def _monitoring_loop(self) -> None:
        """Boucle principale de monitoring"""
        while self.is_monitoring:
            try:
                # Collecter métriques
                metrics = await self.get_current_metrics()

                # Enregistrer dans historique
                self.metrics_history.append(metrics)

                # Envoyer vers collecteur de métriques
                await self._send_to_metrics_collector(metrics)

                # Vérifier seuils et alertes
                violations = await self._check_thresholds(metrics)
                if violations:
                    await self._handle_threshold_violations(violations)

                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(60)  # Pause plus longue en cas d'erreur

    async def _send_to_metrics_collector(self, metrics: SystemMetrics) -> None:
        """Envoie les métriques au collecteur principal"""
        try:
            # Envoyer métriques principales
            await self.metrics_collector.record_gauge("system.cpu_percent", metrics.cpu_percent)
            await self.metrics_collector.record_gauge("system.memory_percent", metrics.memory_percent)
            await self.metrics_collector.record_gauge("system.disk_percent", metrics.disk_percent)

            # Métriques processus
            await self.metrics_collector.record_gauge("process.cpu_percent", metrics.process_cpu_percent)
            await self.metrics_collector.record_gauge("process.memory_rss_mb", metrics.process_memory_rss / 1024**2)
            await self.metrics_collector.record_gauge("process.threads", metrics.process_num_threads)

            # Métriques trading
            await self.metrics_collector.record_gauge("trading.active_strategies", metrics.active_strategies)
            await self.metrics_collector.record_gauge("trading.active_orders", metrics.active_orders)

        except Exception as e:
            logger.error(f"Error sending metrics to collector: {e}")

    async def _get_trading_metrics(self) -> Dict[str, int]:
        """Récupère les métriques spécifiques au trading"""
        # Simulation - dans la réalité, interrogerait les services réels
        return {
            "active_strategies": 4,  # Nombre de stratégies actives
            "active_orders": 12,     # Ordres en cours
            "websocket_connections": 8,  # Connexions WebSocket
            "database_connections": 5    # Connexions DB
        }

    async def _calculate_performance_scores(self) -> None:
        """Calcule les scores de performance composite"""
        if not self.last_metrics:
            return

        current = self.last_metrics

        # Score CPU (inversé, plus bas = meilleur)
        cpu_score = max(0, 100 - current.cpu_percent)

        # Score mémoire
        memory_score = max(0, 100 - current.memory_percent)

        # Score disque
        disk_score = max(0, 100 - current.disk_percent)

        # Score processus
        process_score = max(0, 100 - current.process_cpu_percent)

        # Score trading (basé sur activité)
        trading_score = min(100, (current.active_strategies * 20 + current.active_orders * 5))

        # Score global (moyenne pondérée)
        overall_score = (
            cpu_score * 0.25 +
            memory_score * 0.25 +
            disk_score * 0.15 +
            process_score * 0.20 +
            trading_score * 0.15
        )

        self.performance_scores = {
            "cpu": cpu_score,
            "memory": memory_score,
            "disk": disk_score,
            "process": process_score,
            "trading": trading_score,
            "overall": overall_score
        }

    async def _analyze_trends(self) -> Dict[str, str]:
        """Analyse les tendances des métriques"""
        if len(self.metrics_history) < 10:
            return {"status": "insufficient_data"}

        # Analyser tendances sur les 10 dernières mesures
        recent_metrics = list(self.metrics_history)[-10:]

        trends = {}

        # Tendance CPU
        cpu_values = [m.cpu_percent for m in recent_metrics]
        trends["cpu"] = self._calculate_trend(cpu_values)

        # Tendance mémoire
        memory_values = [m.memory_percent for m in recent_metrics]
        trends["memory"] = self._calculate_trend(memory_values)

        # Tendance trading
        order_values = [m.active_orders for m in recent_metrics]
        trends["trading_activity"] = self._calculate_trend(order_values)

        return trends

    def _calculate_trend(self, values: List[float]) -> str:
        """Calcule la tendance d'une série de valeurs"""
        if len(values) < 3:
            return "stable"

        # Regression linéaire simple
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(values)

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        # Classifier la tendance
        if slope > 1:
            return "increasing"
        elif slope < -1:
            return "decreasing"
        else:
            return "stable"

    async def _check_thresholds(self, metrics: SystemMetrics) -> List[Dict[str, Any]]:
        """Vérifie les seuils de performance"""
        violations = []

        for threshold in self.performance_thresholds:
            value = getattr(metrics, threshold.metric_name, None)
            if value is None:
                continue

            if value >= threshold.critical_threshold:
                violations.append({
                    "metric": threshold.metric_name,
                    "level": "critical",
                    "value": value,
                    "threshold": threshold.critical_threshold,
                    "unit": threshold.unit
                })
            elif value >= threshold.warning_threshold:
                violations.append({
                    "metric": threshold.metric_name,
                    "level": "warning",
                    "value": value,
                    "threshold": threshold.warning_threshold,
                    "unit": threshold.unit
                })

        return violations

    async def _handle_threshold_violations(self, violations: List[Dict[str, Any]]) -> None:
        """Traite les violations de seuils"""
        for violation in violations:
            logger.warning(
                f"Performance threshold violation: {violation['metric']} = "
                f"{violation['value']:.2f}{violation['unit']} "
                f"(threshold: {violation['threshold']}{violation['unit']})"
            )

            # Enregistrer métrique de violation
            await self.metrics_collector.record_counter(
                f"performance.threshold_violation.{violation['metric']}"
            )

    async def _calculate_network_stats(self) -> Dict[str, Any]:
        """Calcule les statistiques réseau dérivées"""
        if len(self.metrics_history) < 2:
            return {"status": "insufficient_data"}

        # Calculer taux de transfert
        current = self.metrics_history[-1]
        previous = self.metrics_history[-2]

        time_diff = (current.timestamp - previous.timestamp).total_seconds()
        if time_diff <= 0:
            return {"status": "invalid_time_diff"}

        bytes_sent_rate = (current.network_bytes_sent - previous.network_bytes_sent) / time_diff
        bytes_recv_rate = (current.network_bytes_recv - previous.network_bytes_recv) / time_diff

        return {
            "bytes_sent_per_sec": bytes_sent_rate,
            "bytes_recv_per_sec": bytes_recv_rate,
            "total_bytes_sent": current.network_bytes_sent,
            "total_bytes_recv": current.network_bytes_recv,
            "packets_sent": current.network_packets_sent,
            "packets_recv": current.network_packets_recv
        }

    async def _measure_component_latency(self) -> Dict[str, float]:
        """Mesure la latence des composants"""
        latency_stats = {}

        # Latence collecteur de métriques
        start_time = time.perf_counter()
        await self.metrics_collector.record_counter("latency_test", 1)
        metrics_latency = (time.perf_counter() - start_time) * 1000

        latency_stats["metrics_collector_ms"] = metrics_latency

        # Latence disque (lecture/écriture test)
        start_time = time.perf_counter()
        try:
            # Test simple de lecture/écriture
            test_data = b"performance_test"
            with open("/tmp/perf_test", "wb") as f:
                f.write(test_data)
            with open("/tmp/perf_test", "rb") as f:
                f.read()

            disk_latency = (time.perf_counter() - start_time) * 1000
            latency_stats["disk_io_ms"] = disk_latency

        except Exception:
            latency_stats["disk_io_ms"] = -1  # Erreur

        return latency_stats

    async def _check_trading_services_health(self) -> Dict[str, Any]:
        """Vérifie la santé des services de trading"""
        # Simulation de health checks
        services_health = {
            "order_manager": {"status": "healthy", "response_time_ms": 5},
            "portfolio_service": {"status": "healthy", "response_time_ms": 3},
            "data_provider": {"status": "healthy", "response_time_ms": 12},
            "risk_monitor": {"status": "healthy", "response_time_ms": 8}
        }

        # Statut global des services trading
        all_healthy = all(service["status"] == "healthy" for service in services_health.values())

        return {
            "overall_status": "healthy" if all_healthy else "degraded",
            "services": services_health
        }

    def _setup_default_thresholds(self) -> List[PerformanceThreshold]:
        """Configure les seuils de performance par défaut"""
        return [
            PerformanceThreshold("cpu_percent", 70.0, 90.0, "%"),
            PerformanceThreshold("memory_percent", 80.0, 95.0, "%"),
            PerformanceThreshold("disk_percent", 85.0, 95.0, "%"),
            PerformanceThreshold("process_cpu_percent", 50.0, 80.0, "%"),
            PerformanceThreshold("process_num_threads", 50, 100, ""),
            PerformanceThreshold("process_open_files", 500, 1000, "")
        ]