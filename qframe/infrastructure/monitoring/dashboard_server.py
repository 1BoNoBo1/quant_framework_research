"""
Dashboard Server
===============

Real-time web dashboard for monitoring trading operations.
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio
import logging
import json
from pathlib import Path

from ...core.container import injectable
from .metrics_collector import MetricsCollector


logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration du dashboard"""
    host: str = "localhost"
    port: int = 8080
    title: str = "QFrame Trading Dashboard"
    refresh_interval: int = 5  # secondes
    theme: str = "dark"  # "dark" ou "light"
    enable_auth: bool = False
    auth_token: Optional[str] = None


@injectable
class DashboardServer:
    """
    Serveur de dashboard web temps réel.

    Fonctionnalités:
    - Interface web responsive
    - Graphiques temps réel avec Chart.js
    - WebSocket pour mises à jour live
    - Panels configurables
    - Export données et rapports
    """

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        config: DashboardConfig = DashboardConfig()
    ):
        self.metrics_collector = metrics_collector
        self.config = config

        # État du serveur
        self.is_running = False
        self.connected_clients: List = []

        # Données du dashboard
        self.dashboard_data: Dict[str, Any] = {}
        self.last_update = datetime.utcnow()

        # Configuration des panels
        self.panels = self._create_default_panels()

    async def start_server(self) -> None:
        """Démarre le serveur web"""
        if self.is_running:
            logger.warning("Dashboard server already running")
            return

        try:
            # Pour une vraie implémentation, utiliserait FastAPI ou Quart
            # Ici on simule le démarrage

            self.is_running = True
            logger.info(f"Dashboard server starting on {self.config.host}:{self.config.port}")

            # Démarrer boucle de mise à jour des données
            asyncio.create_task(self._data_update_loop())

            # Simulation d'endpoints
            await self._setup_routes()

            logger.info("Dashboard server started successfully")

        except Exception as e:
            logger.error(f"Failed to start dashboard server: {e}")
            self.is_running = False

    async def stop_server(self) -> None:
        """Arrête le serveur web"""
        self.is_running = False
        self.connected_clients.clear()
        logger.info("Dashboard server stopped")

    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Retourne les données complètes du dashboard"""
        try:
            # Métriques de base
            current_metrics = await self.metrics_collector.get_current_metrics()
            metrics_summary = await self.metrics_collector.get_metrics_summary()

            # Données des panels
            panel_data = {}
            for panel_id, panel_config in self.panels.items():
                panel_data[panel_id] = await self._get_panel_data(panel_config)

            # Données de navigation
            nav_data = await self._get_navigation_data()

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "config": {
                    "title": self.config.title,
                    "theme": self.config.theme,
                    "refresh_interval": self.config.refresh_interval
                },
                "metrics": current_metrics,
                "summary": metrics_summary,
                "panels": panel_data,
                "navigation": nav_data,
                "status": {
                    "server_running": self.is_running,
                    "connected_clients": len(self.connected_clients),
                    "last_update": self.last_update.isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {"error": str(e)}

    async def add_custom_panel(
        self,
        panel_id: str,
        title: str,
        panel_type: str,
        config: Dict[str, Any]
    ) -> None:
        """Ajoute un panel personnalisé"""
        self.panels[panel_id] = {
            "title": title,
            "type": panel_type,
            "config": config,
            "created_at": datetime.utcnow().isoformat()
        }

        logger.info(f"Added custom panel: {panel_id}")

    async def remove_panel(self, panel_id: str) -> bool:
        """Supprime un panel"""
        if panel_id in self.panels:
            del self.panels[panel_id]
            logger.info(f"Removed panel: {panel_id}")
            return True
        return False

    async def export_dashboard_config(self) -> Dict[str, Any]:
        """Exporte la configuration du dashboard"""
        return {
            "config": self.config.__dict__,
            "panels": self.panels,
            "export_timestamp": datetime.utcnow().isoformat()
        }

    async def import_dashboard_config(self, config_data: Dict[str, Any]) -> bool:
        """Importe une configuration de dashboard"""
        try:
            if "panels" in config_data:
                self.panels.update(config_data["panels"])

            if "config" in config_data:
                for key, value in config_data["config"].items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)

            logger.info("Dashboard configuration imported successfully")
            return True

        except Exception as e:
            logger.error(f"Error importing dashboard config: {e}")
            return False

    # === Génération HTML ===

    async def generate_html_dashboard(self) -> str:
        """Génère le HTML complet du dashboard"""
        dashboard_data = await self.get_dashboard_data()

        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body class="{self.config.theme}">
    <div id="app">
        <header>
            <h1>{self.config.title}</h1>
            <div class="status-bar">
                <span class="status-indicator {'green' if self.is_running else 'red'}"></span>
                <span>Last Update: {self.last_update.strftime('%H:%M:%S')}</span>
            </div>
        </header>

        <main>
            {await self._generate_panels_html(dashboard_data.get('panels', {}))}
        </main>

        <footer>
            <p>QFrame Trading Dashboard - Connected Clients: {len(self.connected_clients)}</p>
        </footer>
    </div>

    <script>
        // Dashboard data
        const dashboardData = {json.dumps(dashboard_data, default=str)};

        // Auto-refresh
        setInterval(() => {{
            location.reload();
        }}, {self.config.refresh_interval * 1000});

        {self._get_javascript_code()}
    </script>
</body>
</html>
        """

        return html_template

    # === Méthodes privées ===

    async def _setup_routes(self) -> None:
        """Configure les routes de l'API"""
        # Dans une vraie implémentation avec FastAPI:
        # @app.get("/")
        # @app.get("/api/metrics")
        # @app.get("/api/dashboard")
        # @app.websocket("/ws")

        routes_info = {
            "/": "Dashboard HTML page",
            "/api/metrics": "Current metrics JSON",
            "/api/dashboard": "Full dashboard data",
            "/api/panels/{panel_id}": "Individual panel data",
            "/ws": "WebSocket for real-time updates",
            "/export/prometheus": "Prometheus metrics format"
        }

        logger.info(f"Configured {len(routes_info)} dashboard routes")

    async def _data_update_loop(self) -> None:
        """Boucle de mise à jour des données"""
        while self.is_running:
            try:
                # Mettre à jour données dashboard
                self.dashboard_data = await self.get_dashboard_data()
                self.last_update = datetime.utcnow()

                # Notifier clients WebSocket (simulation)
                await self._notify_websocket_clients(self.dashboard_data)

                await asyncio.sleep(self.config.refresh_interval)

            except Exception as e:
                logger.error(f"Error in data update loop: {e}")
                await asyncio.sleep(10)

    async def _notify_websocket_clients(self, data: Dict[str, Any]) -> None:
        """Notifie les clients WebSocket des mises à jour"""
        if not self.connected_clients:
            return

        # Simulation de notification WebSocket
        logger.debug(f"Notifying {len(self.connected_clients)} WebSocket clients")

    def _create_default_panels(self) -> Dict[str, Dict[str, Any]]:
        """Crée les panels par défaut du dashboard"""
        return {
            "overview": {
                "title": "System Overview",
                "type": "metrics_grid",
                "config": {
                    "metrics": [
                        "total_metrics_collected",
                        "metrics_per_second",
                        "buffer_utilization"
                    ]
                }
            },
            "performance": {
                "title": "Performance Metrics",
                "type": "line_chart",
                "config": {
                    "metrics": ["execution_time", "throughput"],
                    "time_range": "1h"
                }
            },
            "trading_stats": {
                "title": "Trading Statistics",
                "type": "bar_chart",
                "config": {
                    "metrics": ["orders_per_minute", "success_rate", "avg_slippage"],
                    "time_range": "24h"
                }
            },
            "risk_monitor": {
                "title": "Risk Monitoring",
                "type": "gauge",
                "config": {
                    "metrics": ["portfolio_var", "max_drawdown", "concentration_risk"],
                    "thresholds": {
                        "portfolio_var": {"warning": 0.05, "critical": 0.10},
                        "max_drawdown": {"warning": 0.03, "critical": 0.07}
                    }
                }
            },
            "alerts": {
                "title": "Recent Alerts",
                "type": "alert_list",
                "config": {
                    "max_alerts": 10,
                    "severity_filter": ["warning", "critical", "emergency"]
                }
            },
            "strategy_breakdown": {
                "title": "Strategy Performance",
                "type": "pie_chart",
                "config": {
                    "metric": "strategy_returns",
                    "time_range": "24h"
                }
            }
        }

    async def _get_panel_data(self, panel_config: Dict[str, Any]) -> Dict[str, Any]:
        """Récupère les données pour un panel spécifique"""
        panel_type = panel_config.get("type")
        config = panel_config.get("config", {})

        try:
            if panel_type == "metrics_grid":
                return await self._get_metrics_grid_data(config)
            elif panel_type == "line_chart":
                return await self._get_line_chart_data(config)
            elif panel_type == "bar_chart":
                return await self._get_bar_chart_data(config)
            elif panel_type == "gauge":
                return await self._get_gauge_data(config)
            elif panel_type == "alert_list":
                return await self._get_alert_list_data(config)
            elif panel_type == "pie_chart":
                return await self._get_pie_chart_data(config)
            else:
                return {"error": f"Unknown panel type: {panel_type}"}

        except Exception as e:
            logger.error(f"Error getting panel data for {panel_type}: {e}")
            return {"error": str(e)}

    async def _get_metrics_grid_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Données pour grille de métriques"""
        metrics = config.get("metrics", [])
        current_metrics = await self.metrics_collector.get_current_metrics()

        grid_data = []
        for metric_name in metrics:
            value = current_metrics.get("collection_stats", {}).get(metric_name, "N/A")
            grid_data.append({
                "name": metric_name,
                "value": value,
                "format": "number"
            })

        return {"type": "metrics_grid", "data": grid_data}

    async def _get_line_chart_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Données pour graphique linéaire"""
        # Simulation de données de série temporelle
        time_range = config.get("time_range", "1h")
        metrics = config.get("metrics", [])

        # Générer points de données simulés
        datasets = []
        for metric in metrics:
            # Dans la réalité, récupérerait les données agrégées
            data_points = [
                {"x": (datetime.utcnow() - timedelta(minutes=i)).isoformat(), "y": 50 + i * 2}
                for i in range(60, 0, -5)
            ]

            datasets.append({
                "label": metric,
                "data": data_points,
                "borderColor": f"hsl({hash(metric) % 360}, 70%, 50%)",
                "tension": 0.4
            })

        return {
            "type": "line_chart",
            "data": {
                "datasets": datasets
            },
            "options": {
                "responsive": True,
                "scales": {
                    "x": {"type": "time"},
                    "y": {"beginAtZero": True}
                }
            }
        }

    async def _get_bar_chart_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Données pour graphique en barres"""
        metrics = config.get("metrics", [])

        return {
            "type": "bar_chart",
            "data": {
                "labels": metrics,
                "datasets": [{
                    "label": "Current Values",
                    "data": [100, 85, 92, 78, 95],  # Données simulées
                    "backgroundColor": "rgba(54, 162, 235, 0.6)"
                }]
            }
        }

    async def _get_gauge_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Données pour jauges"""
        metrics = config.get("metrics", [])
        thresholds = config.get("thresholds", {})

        gauges = []
        for metric in metrics:
            # Valeur simulée
            value = 0.03 if metric == "max_drawdown" else 0.02

            gauge_data = {
                "name": metric,
                "value": value,
                "min": 0,
                "max": 0.10,
                "thresholds": thresholds.get(metric, {})
            }
            gauges.append(gauge_data)

        return {"type": "gauge", "data": gauges}

    async def _get_alert_list_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Données pour liste d'alertes"""
        max_alerts = config.get("max_alerts", 10)

        # Simulation d'alertes
        alerts = [
            {
                "id": "alert_1",
                "severity": "warning",
                "message": "High volatility detected in BTC position",
                "timestamp": datetime.utcnow().isoformat(),
                "acknowledged": False
            },
            {
                "id": "alert_2",
                "severity": "info",
                "message": "Strategy rebalancing completed",
                "timestamp": (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
                "acknowledged": True
            }
        ]

        return {"type": "alert_list", "data": alerts[:max_alerts]}

    async def _get_pie_chart_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Données pour graphique en secteurs"""
        return {
            "type": "pie_chart",
            "data": {
                "labels": ["Strategy A", "Strategy B", "Strategy C"],
                "datasets": [{
                    "data": [45, 35, 20],
                    "backgroundColor": [
                        "rgba(255, 99, 132, 0.6)",
                        "rgba(54, 162, 235, 0.6)",
                        "rgba(255, 205, 86, 0.6)"
                    ]
                }]
            }
        }

    async def _get_navigation_data(self) -> Dict[str, Any]:
        """Données de navigation"""
        return {
            "sections": [
                {"id": "overview", "title": "Overview", "icon": "dashboard"},
                {"id": "trading", "title": "Trading", "icon": "trending_up"},
                {"id": "risk", "title": "Risk Management", "icon": "security"},
                {"id": "strategies", "title": "Strategies", "icon": "psychology"},
                {"id": "alerts", "title": "Alerts", "icon": "notification_important"}
            ]
        }

    async def _generate_panels_html(self, panels_data: Dict[str, Any]) -> str:
        """Génère le HTML pour tous les panels"""
        panels_html = []

        for panel_id, panel_data in panels_data.items():
            panel_config = self.panels.get(panel_id, {})
            title = panel_config.get("title", panel_id)

            panel_html = f"""
            <div class="panel" id="panel-{panel_id}">
                <div class="panel-header">
                    <h3>{title}</h3>
                    <div class="panel-controls">
                        <button onclick="refreshPanel('{panel_id}')">⟳</button>
                    </div>
                </div>
                <div class="panel-content">
                    {await self._generate_panel_content_html(panel_data)}
                </div>
            </div>
            """

            panels_html.append(panel_html)

        return "\\n".join(panels_html)

    async def _generate_panel_content_html(self, panel_data: Dict[str, Any]) -> str:
        """Génère le contenu HTML d'un panel"""
        panel_type = panel_data.get("type")

        if panel_type == "metrics_grid":
            return self._generate_metrics_grid_html(panel_data.get("data", []))
        elif panel_type in ["line_chart", "bar_chart", "pie_chart"]:
            return f'<canvas id="chart-{panel_type}"></canvas>'
        elif panel_type == "gauge":
            return self._generate_gauge_html(panel_data.get("data", []))
        elif panel_type == "alert_list":
            return self._generate_alert_list_html(panel_data.get("data", []))
        else:
            return f'<p>Panel type "{panel_type}" not implemented</p>'

    def _generate_metrics_grid_html(self, metrics_data: List[Dict]) -> str:
        """Génère HTML pour grille de métriques"""
        grid_items = []

        for metric in metrics_data:
            grid_items.append(f"""
                <div class="metric-item">
                    <div class="metric-label">{metric['name']}</div>
                    <div class="metric-value">{metric['value']}</div>
                </div>
            """)

        return f'<div class="metrics-grid">{"".join(grid_items)}</div>'

    def _generate_gauge_html(self, gauges_data: List[Dict]) -> str:
        """Génère HTML pour jauges"""
        gauges_html = []

        for gauge in gauges_data:
            value_percent = (gauge['value'] / gauge['max']) * 100

            gauges_html.append(f"""
                <div class="gauge-container">
                    <div class="gauge-label">{gauge['name']}</div>
                    <div class="gauge">
                        <div class="gauge-fill" style="width: {value_percent}%"></div>
                    </div>
                    <div class="gauge-value">{gauge['value']:.3f}</div>
                </div>
            """)

        return f'<div class="gauges-grid">{"".join(gauges_html)}</div>'

    def _generate_alert_list_html(self, alerts_data: List[Dict]) -> str:
        """Génère HTML pour liste d'alertes"""
        alerts_html = []

        for alert in alerts_data:
            severity_class = f"alert-{alert['severity']}"
            ack_class = "acknowledged" if alert['acknowledged'] else ""

            alerts_html.append(f"""
                <div class="alert-item {severity_class} {ack_class}">
                    <div class="alert-severity">{alert['severity'].upper()}</div>
                    <div class="alert-message">{alert['message']}</div>
                    <div class="alert-timestamp">{alert['timestamp']}</div>
                </div>
            """)

        return f'<div class="alerts-list">{"".join(alerts_html)}</div>'

    def _get_css_styles(self) -> str:
        """Retourne les styles CSS du dashboard"""
        return """
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body.dark {
            background: #1a1a1a;
            color: #e0e0e0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        header {
            background: #2d2d2d;
            padding: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #444;
        }

        .status-bar {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }

        .status-indicator.green { background: #4caf50; }
        .status-indicator.red { background: #f44336; }

        main {
            padding: 1rem;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1rem;
        }

        .panel {
            background: #2d2d2d;
            border-radius: 8px;
            border: 1px solid #444;
            overflow: hidden;
        }

        .panel-header {
            background: #333;
            padding: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .panel-content {
            padding: 1rem;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
        }

        .metric-item {
            background: #333;
            padding: 1rem;
            border-radius: 4px;
            text-align: center;
        }

        .metric-label {
            font-size: 0.9rem;
            color: #aaa;
            margin-bottom: 0.5rem;
        }

        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #4caf50;
        }

        .gauges-grid {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .gauge-container {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .gauge-label {
            flex: 1;
            font-size: 0.9rem;
        }

        .gauge {
            flex: 2;
            height: 20px;
            background: #444;
            border-radius: 10px;
            overflow: hidden;
        }

        .gauge-fill {
            height: 100%;
            background: linear-gradient(90deg, #4caf50, #ff9800, #f44336);
            transition: width 0.3s ease;
        }

        .gauge-value {
            flex: 0 0 60px;
            text-align: right;
            font-weight: bold;
        }

        .alerts-list {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .alert-item {
            padding: 0.75rem;
            border-radius: 4px;
            border-left: 4px solid;
        }

        .alert-warning { border-left-color: #ff9800; background: rgba(255, 152, 0, 0.1); }
        .alert-critical { border-left-color: #f44336; background: rgba(244, 67, 54, 0.1); }
        .alert-info { border-left-color: #2196f3; background: rgba(33, 150, 243, 0.1); }

        .alert-severity {
            font-size: 0.8rem;
            font-weight: bold;
            margin-bottom: 0.25rem;
        }

        .alert-message {
            margin-bottom: 0.25rem;
        }

        .alert-timestamp {
            font-size: 0.8rem;
            color: #aaa;
        }

        footer {
            background: #2d2d2d;
            padding: 1rem;
            text-align: center;
            border-top: 1px solid #444;
            margin-top: 2rem;
        }
        """

    def _get_javascript_code(self) -> str:
        """Retourne le code JavaScript du dashboard"""
        return """
        // Initialize charts
        function initCharts() {
            // Chart.js initialization would go here
            console.log('Charts initialized');
        }

        // Refresh panel data
        function refreshPanel(panelId) {
            console.log('Refreshing panel:', panelId);
            // AJAX call to refresh panel data
        }

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
            console.log('Dashboard initialized');
        });
        """