"""
Infrastructure Layer: Observability Dashboard
============================================

Dashboard unifié pour monitoring en temps réel.
Combine métriques, logs, traces, health checks et alertes.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

from .logging import LoggerFactory
from .metrics import get_metrics_collector, get_business_metrics
from .tracing import get_tracer
from .health import get_health_monitor
from .alerting import get_alert_manager, AlertSeverity, AlertStatus


@dataclass
class DashboardMetrics:
    """Métriques consolidées pour le dashboard"""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Métriques système
    system_health: str = "unknown"
    total_trades: int = 0
    total_volume: float = 0.0
    current_pnl: float = 0.0

    # Métriques de performance
    avg_trade_latency: float = 0.0
    api_requests_per_minute: int = 0
    error_rate: float = 0.0

    # Métriques de risque
    current_var: float = 0.0
    max_drawdown: float = 0.0
    risk_breaches_count: int = 0

    # Alertes
    open_alerts: int = 0
    critical_alerts: int = 0

    # Composants
    healthy_components: int = 0
    total_components: int = 0

    # Traces
    active_traces: int = 0
    avg_trace_duration: float = 0.0


class ObservabilityDashboard:
    """
    Dashboard unifié pour toutes les données d'observabilité.
    """

    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)
        self.metrics_collector = get_metrics_collector()
        self.business_metrics = get_business_metrics()
        self.tracer = get_tracer()
        self.health_monitor = get_health_monitor()
        self.alert_manager = get_alert_manager()

        # Historique des métriques
        self._metrics_history: List[DashboardMetrics] = []
        self._max_history = 1000

    def get_current_metrics(self) -> DashboardMetrics:
        """Obtenir les métriques actuelles consolidées"""
        # Métriques système
        system_health_data = self.health_monitor.get_system_health()

        # Métriques de trading
        total_trades = self.metrics_collector.get_counter_value("trades.executed")
        total_volume = self.metrics_collector.get_counter_value("trades.volume")

        # P&L actuel (supposé être dans les métriques)
        current_pnl = self.metrics_collector.get_gauge_value("portfolio.pnl")

        # Latence moyenne des trades
        trade_latency_stats = self.metrics_collector.get_histogram_stats("trades.latency")
        avg_trade_latency = trade_latency_stats.get("mean", 0.0)

        # Requêtes API par minute
        api_requests = self.metrics_collector.get_counter_value("system.api_requests")

        # Taux d'erreur
        total_requests = max(1, api_requests)  # Éviter division par zéro
        errors = self.metrics_collector.get_counter_value("system.errors")
        error_rate = (errors / total_requests) * 100

        # Métriques de risque
        current_var = self.metrics_collector.get_gauge_value("risk.var")
        risk_breaches = self.metrics_collector.get_counter_value("risk.breaches")

        # Alertes
        alert_stats = self.alert_manager.get_alert_statistics()
        open_alerts = alert_stats.get("by_status", {}).get(AlertStatus.OPEN.value, 0)
        critical_alerts = alert_stats.get("by_severity", {}).get(AlertSeverity.CRITICAL.value, 0)

        # Composants
        components_status = system_health_data.get("components", {})
        healthy_components = len([c for c in components_status.values() if c.get("status") == "healthy"])
        total_components = len(components_status)

        # Traces (approximation basée sur les exports)
        traces_data = self.tracer.export_traces()
        active_traces = len(traces_data)

        # Calculer la durée moyenne des traces
        if traces_data:
            durations = [t.get("duration_ms", 0) for t in traces_data if t.get("duration_ms")]
            avg_trace_duration = sum(durations) / len(durations) if durations else 0.0
        else:
            avg_trace_duration = 0.0

        return DashboardMetrics(
            system_health=system_health_data.get("status", "unknown"),
            total_trades=int(total_trades),
            total_volume=float(total_volume),
            current_pnl=float(current_pnl),
            avg_trade_latency=avg_trade_latency,
            api_requests_per_minute=int(api_requests),  # Approximation
            error_rate=error_rate,
            current_var=float(current_var),
            risk_breaches_count=int(risk_breaches),
            open_alerts=open_alerts,
            critical_alerts=critical_alerts,
            healthy_components=healthy_components,
            total_components=total_components,
            active_traces=active_traces,
            avg_trace_duration=avg_trace_duration
        )

    def get_historical_metrics(
        self,
        hours: int = 24,
        resolution_minutes: int = 5
    ) -> List[DashboardMetrics]:
        """Obtenir l'historique des métriques"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        # Filtrer par temps et échantillonner selon la résolution
        filtered_metrics = []
        last_time = None

        for metric in reversed(self._metrics_history):
            if metric.timestamp < cutoff_time:
                break

            if (last_time is None or
                (metric.timestamp - last_time).total_seconds() >= resolution_minutes * 60):
                filtered_metrics.append(metric)
                last_time = metric.timestamp

        return list(reversed(filtered_metrics))

    def get_component_details(self) -> Dict[str, Any]:
        """Obtenir les détails des composants"""
        system_health = self.health_monitor.get_system_health()
        components = {}

        for name, component_data in system_health.get("components", {}).items():
            health = self.health_monitor.get_component_health(name)

            if health:
                components[name] = {
                    "status": component_data.get("status"),
                    "type": component_data.get("type"),
                    "last_check": component_data.get("last_check"),
                    "uptime_24h": component_data.get("uptime_24h"),
                    "avg_response_time": component_data.get("avg_response_time"),
                    "consecutive_failures": component_data.get("consecutive_failures"),
                    "recent_errors": health.consecutive_failures
                }

        return components

    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtenir les alertes récentes"""
        recent_time = datetime.utcnow() - timedelta(hours=1)
        alerts = self.alert_manager.get_alerts(since=recent_time)

        return [alert.to_dict() for alert in alerts[:limit]]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Obtenir un résumé des performances"""
        # Latences par endpoint
        api_latency = self.metrics_collector.get_histogram_stats("system.api_latency")
        trade_latency = self.metrics_collector.get_histogram_stats("trades.latency")

        # Throughput
        total_trades = self.metrics_collector.get_counter_value("trades.executed")
        total_requests = self.metrics_collector.get_counter_value("system.api_requests")

        return {
            "api_latency": {
                "p50": api_latency.get("p50", 0),
                "p90": api_latency.get("p90", 0),
                "p99": api_latency.get("p99", 0)
            },
            "trade_latency": {
                "p50": trade_latency.get("p50", 0),
                "p90": trade_latency.get("p90", 0),
                "p99": trade_latency.get("p99", 0)
            },
            "throughput": {
                "trades_total": int(total_trades),
                "requests_total": int(total_requests)
            }
        }

    def get_risk_dashboard(self) -> Dict[str, Any]:
        """Obtenir le tableau de bord des risques"""
        return {
            "current_var": float(self.metrics_collector.get_gauge_value("risk.var")),
            "exposures": {
                # Ces métriques devraient être alimentées par le risk engine
                "total": float(self.metrics_collector.get_gauge_value("risk.exposure")),
                "by_sector": {},  # À implémenter avec des labels
                "by_asset_class": {}  # À implémenter avec des labels
            },
            "breaches": {
                "total": int(self.metrics_collector.get_counter_value("risk.breaches")),
                "recent": len(self.alert_manager.get_alerts(
                    category=AlertSeverity.ERROR,
                    since=datetime.utcnow() - timedelta(hours=1)
                ))
            },
            "limits": {
                # À configurer selon les limites définies
                "var_limit": 50000,
                "exposure_limit": 1000000,
                "drawdown_limit": 0.1
            }
        }

    def get_trading_summary(self) -> Dict[str, Any]:
        """Obtenir un résumé du trading"""
        # Métriques de base
        total_trades = self.metrics_collector.get_counter_value("trades.executed")
        total_volume = self.metrics_collector.get_counter_value("trades.volume")
        current_pnl = self.metrics_collector.get_gauge_value("portfolio.pnl")

        # Répartition par symbole/venue (nécessiterait des métriques avec labels)
        return {
            "total_trades": int(total_trades),
            "total_volume": float(total_volume),
            "current_pnl": float(current_pnl),
            "avg_trade_size": float(total_volume / max(1, total_trades)),
            "top_symbols": [],  # À implémenter avec des métriques labelisées
            "venue_distribution": {},  # À implémenter avec des métriques labelisées
            "recent_trades": []  # À implémenter avec un historique des trades
        }

    def get_traces_summary(self) -> Dict[str, Any]:
        """Obtenir un résumé des traces"""
        traces = self.tracer.export_traces()

        if not traces:
            return {
                "total_traces": 0,
                "avg_duration": 0,
                "slowest_operations": [],
                "error_traces": []
            }

        # Analyser les traces
        durations = []
        error_traces = []
        operation_stats = {}

        for trace in traces:
            duration = trace.get("duration_ms", 0)
            durations.append(duration)

            # Analyser les spans pour trouver les erreurs
            has_error = False
            for span in trace.get("spans", []):
                if span.get("status") == "error":
                    error_traces.append(trace)
                    has_error = True
                    break

            # Statistiques par opération
            if trace.get("spans"):
                root_operation = trace["spans"][0].get("operation_name", "unknown")
                if root_operation not in operation_stats:
                    operation_stats[root_operation] = {"count": 0, "total_duration": 0}
                operation_stats[root_operation]["count"] += 1
                operation_stats[root_operation]["total_duration"] += duration

        # Trier les opérations par durée moyenne
        slowest_operations = []
        for op, stats in operation_stats.items():
            avg_duration = stats["total_duration"] / stats["count"]
            slowest_operations.append({
                "operation": op,
                "avg_duration": avg_duration,
                "count": stats["count"]
            })

        slowest_operations.sort(key=lambda x: x["avg_duration"], reverse=True)

        return {
            "total_traces": len(traces),
            "avg_duration": sum(durations) / len(durations) if durations else 0,
            "slowest_operations": slowest_operations[:5],
            "error_traces": len(error_traces),
            "error_rate": (len(error_traces) / len(traces)) * 100
        }

    def export_dashboard_data(self) -> Dict[str, Any]:
        """Exporter toutes les données du dashboard"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "current_metrics": self.get_current_metrics().__dict__,
            "components": self.get_component_details(),
            "alerts": self.get_recent_alerts(),
            "performance": self.get_performance_summary(),
            "risk": self.get_risk_dashboard(),
            "trading": self.get_trading_summary(),
            "traces": self.get_traces_summary()
        }

    def export_prometheus_metrics(self) -> str:
        """Exporter les métriques au format Prometheus"""
        return self.metrics_collector.export_prometheus()

    def export_json_metrics(self) -> str:
        """Exporter les métriques au format JSON"""
        return self.metrics_collector.export_json()

    def record_snapshot(self):
        """Enregistrer un snapshot des métriques actuelles"""
        current_metrics = self.get_current_metrics()
        self._metrics_history.append(current_metrics)

        # Limiter la taille de l'historique
        if len(self._metrics_history) > self._max_history:
            self._metrics_history = self._metrics_history[-self._max_history:]

        self.logger.debug(
            "Dashboard snapshot recorded",
            total_trades=current_metrics.total_trades,
            system_health=current_metrics.system_health,
            open_alerts=current_metrics.open_alerts
        )

    async def start_auto_snapshot(self, interval_seconds: int = 60):
        """Démarrer la capture automatique de snapshots"""
        while True:
            try:
                self.record_snapshot()
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                self.logger.error("Error recording snapshot", error=e)
                await asyncio.sleep(interval_seconds)

    def create_health_report(self) -> str:
        """Créer un rapport de santé textuel"""
        current = self.get_current_metrics()
        components = self.get_component_details()

        report = f"""
=== QFrame System Health Report ===
Generated: {datetime.utcnow().isoformat()}

🔍 SYSTEM OVERVIEW
Status: {current.system_health.upper()}
Components: {current.healthy_components}/{current.total_components} healthy
Open Alerts: {current.open_alerts} (Critical: {current.critical_alerts})

📈 TRADING METRICS
Total Trades: {current.total_trades:,}
Total Volume: ${current.total_volume:,.2f}
Current P&L: ${current.current_pnl:,.2f}
Avg Trade Latency: {current.avg_trade_latency:.1f}ms

⚡ PERFORMANCE
API Requests/min: {current.api_requests_per_minute}
Error Rate: {current.error_rate:.2f}%
Active Traces: {current.active_traces}
Avg Trace Duration: {current.avg_trace_duration:.1f}ms

🛡️ RISK METRICS
Current VaR: ${current.current_var:,.2f}
Max Drawdown: {current.max_drawdown:.2%}
Risk Breaches: {current.risk_breaches_count}

📊 COMPONENT STATUS
"""

        for name, comp in components.items():
            status_emoji = "✅" if comp["status"] == "healthy" else "⚠️" if comp["status"] == "degraded" else "❌"
            uptime = comp.get("uptime_24h", 0)
            report += f"{status_emoji} {name}: {comp['status']} (Uptime: {uptime:.1f}%)\n"

        return report


# Instance globale
_global_dashboard = ObservabilityDashboard()


def get_dashboard() -> ObservabilityDashboard:
    """Obtenir l'instance globale du dashboard"""
    return _global_dashboard