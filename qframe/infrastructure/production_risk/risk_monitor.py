"""
Production Risk Monitor
======================

Real-time risk monitoring with intelligent alerts and automatic actions.
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging

from ...core.container import injectable
from ...domain.entities.portfolio import Portfolio
from ...domain.entities.position import Position
from .circuit_breakers import CircuitBreaker
from .position_limits import PositionLimitManager
from .risk_metrics import RealTimeRiskCalculator, RiskMetrics


logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Niveaux de sévérité des alertes"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class RiskCategory(str, Enum):
    """Catégories de risques"""
    POSITION_SIZE = "position_size"
    PORTFOLIO_EXPOSURE = "portfolio_exposure"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    CONCENTRATION = "concentration"
    MARKET_CONDITIONS = "market_conditions"


@dataclass
class RiskAlert:
    """Alerte de risque"""
    id: str
    category: RiskCategory
    severity: AlertSeverity
    message: str
    current_value: Decimal
    threshold_value: Decimal
    symbol: Optional[str] = None
    strategy_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    auto_action_taken: Optional[str] = None


@dataclass
class RiskThreshold:
    """Seuil de risque configuré"""
    category: RiskCategory
    severity: AlertSeverity
    threshold_value: Decimal
    comparison: str = "greater_than"  # "greater_than", "less_than", "equal"
    auto_action: Optional[str] = None  # "reduce_position", "stop_trading", "alert_only"


@injectable
class ProductionRiskMonitor:
    """
    Moniteur de risque en temps réel pour environnement de production.

    Responsabilités:
    - Surveillance continue des métriques de risque
    - Alertes intelligentes avec seuils adaptatifs
    - Actions automatiques (circuit breakers, réduction positions)
    - Reporting détaillé pour compliance
    """

    def __init__(
        self,
        circuit_breaker: CircuitBreaker,
        position_limit_manager: PositionLimitManager,
        risk_calculator: RealTimeRiskCalculator,
        monitoring_interval: timedelta = timedelta(seconds=10)
    ):
        self.circuit_breaker = circuit_breaker
        self.position_limit_manager = position_limit_manager
        self.risk_calculator = risk_calculator
        self.monitoring_interval = monitoring_interval

        # Configuration des seuils
        self.risk_thresholds: List[RiskThreshold] = []
        self._setup_default_thresholds()

        # État du monitoring
        self.is_monitoring = False
        self.active_alerts: Dict[str, RiskAlert] = {}
        self.alert_history: List[RiskAlert] = []

        # Métriques historiques
        self.risk_metrics_history: List[RiskMetrics] = []
        self.max_history_length = 1000

        # Callbacks
        self.on_alert_triggered: Optional[Callable] = None
        self.on_emergency_stop: Optional[Callable] = None

    async def start_monitoring(self) -> None:
        """Démarre le monitoring en temps réel"""
        if self.is_monitoring:
            logger.warning("Risk monitoring already started")
            return

        self.is_monitoring = True
        logger.info("Starting production risk monitoring")

        # Démarrer boucle de monitoring
        asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self) -> None:
        """Arrête le monitoring"""
        self.is_monitoring = False
        logger.info("Production risk monitoring stopped")

    async def check_portfolio_risk(self, portfolio: Portfolio) -> List[RiskAlert]:
        """
        Évalue les risques d'un portfolio et retourne les alertes.

        Args:
            portfolio: Portfolio à analyser

        Returns:
            Liste des alertes générées
        """
        alerts = []

        try:
            # Calculer métriques de risque
            risk_metrics = await self.risk_calculator.calculate_risk_metrics(portfolio)

            # Enregistrer dans historique
            self._record_risk_metrics(risk_metrics)

            # Vérifier chaque seuil configuré
            for threshold in self.risk_thresholds:
                alert = await self._check_threshold(risk_metrics, portfolio, threshold)
                if alert:
                    alerts.append(alert)

            # Vérifier limits de positions
            position_alerts = await self._check_position_limits(portfolio)
            alerts.extend(position_alerts)

            # Vérifier conditions de circuit breaker
            circuit_alerts = await self._check_circuit_conditions(risk_metrics)
            alerts.extend(circuit_alerts)

            # Traiter nouvelles alertes
            for alert in alerts:
                await self._process_alert(alert)

            return alerts

        except Exception as e:
            logger.error(f"Error checking portfolio risk: {e}")
            return []

    async def add_risk_threshold(self, threshold: RiskThreshold) -> None:
        """Ajoute un seuil de risque personnalisé"""
        self.risk_thresholds.append(threshold)
        logger.info(f"Added risk threshold: {threshold.category.value} - {threshold.severity.value}")

    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Marque une alerte comme acquittée"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"Alert {alert_id} acknowledged")
            return True
        return False

    async def get_active_alerts(self) -> List[RiskAlert]:
        """Retourne toutes les alertes actives"""
        return list(self.active_alerts.values())

    async def get_risk_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de l'état des risques"""
        active_alerts = list(self.active_alerts.values())

        alert_counts = {}
        for severity in AlertSeverity:
            alert_counts[severity.value] = len([a for a in active_alerts if a.severity == severity])

        latest_metrics = self.risk_metrics_history[-1] if self.risk_metrics_history else None

        return {
            "monitoring_active": self.is_monitoring,
            "active_alerts_count": len(active_alerts),
            "alerts_by_severity": alert_counts,
            "circuit_breaker_status": await self.circuit_breaker.get_status(),
            "latest_risk_metrics": {
                "portfolio_var": float(latest_metrics.portfolio_var) if latest_metrics else None,
                "max_drawdown": float(latest_metrics.max_drawdown) if latest_metrics else None,
                "concentration_risk": float(latest_metrics.concentration_risk) if latest_metrics else None,
                "timestamp": latest_metrics.timestamp.isoformat() if latest_metrics else None
            },
            "total_alerts_today": len([
                a for a in self.alert_history
                if a.timestamp.date() == datetime.utcnow().date()
            ])
        }

    # === Méthodes de monitoring ===

    async def _monitoring_loop(self) -> None:
        """Boucle principale de monitoring"""
        while self.is_monitoring:
            try:
                # Simuler obtention du portfolio (à adapter selon architecture)
                # Dans une vraie implémentation, ceci viendrait du portfolio service
                current_portfolio = await self._get_current_portfolio()

                if current_portfolio:
                    await self.check_portfolio_risk(current_portfolio)

                # Nettoyer anciennes alertes
                await self._cleanup_resolved_alerts()

                await asyncio.sleep(self.monitoring_interval.total_seconds())

            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(30)  # Pause plus longue en cas d'erreur

    async def _check_threshold(
        self,
        risk_metrics: RiskMetrics,
        portfolio: Portfolio,
        threshold: RiskThreshold
    ) -> Optional[RiskAlert]:
        """Vérifie un seuil spécifique"""
        current_value = None

        # Extraire valeur selon catégorie
        if threshold.category == RiskCategory.DRAWDOWN:
            current_value = risk_metrics.max_drawdown
        elif threshold.category == RiskCategory.PORTFOLIO_EXPOSURE:
            current_value = portfolio.total_value
        elif threshold.category == RiskCategory.CONCENTRATION:
            current_value = risk_metrics.concentration_risk
        elif threshold.category == RiskCategory.VOLATILITY:
            current_value = risk_metrics.portfolio_volatility

        if current_value is None:
            return None

        # Vérifier condition
        threshold_breached = False
        if threshold.comparison == "greater_than":
            threshold_breached = current_value > threshold.threshold_value
        elif threshold.comparison == "less_than":
            threshold_breached = current_value < threshold.threshold_value

        if threshold_breached:
            alert_id = f"{threshold.category.value}_{threshold.severity.value}_{int(datetime.utcnow().timestamp())}"

            return RiskAlert(
                id=alert_id,
                category=threshold.category,
                severity=threshold.severity,
                message=f"{threshold.category.value} threshold breached: {current_value} vs {threshold.threshold_value}",
                current_value=current_value,
                threshold_value=threshold.threshold_value,
                auto_action_taken=threshold.auto_action
            )

        return None

    async def _check_position_limits(self, portfolio: Portfolio) -> List[RiskAlert]:
        """Vérifie les limites de positions"""
        alerts = []

        for position in portfolio.positions:
            violations = await self.position_limit_manager.check_position_limits(position)

            for violation in violations:
                alert = RiskAlert(
                    id=f"position_limit_{position.symbol}_{int(datetime.utcnow().timestamp())}",
                    category=RiskCategory.POSITION_SIZE,
                    severity=AlertSeverity.WARNING if violation['severity'] == 'warning' else AlertSeverity.CRITICAL,
                    message=f"Position limit exceeded: {position.symbol} - {violation['message']}",
                    current_value=violation['current_value'],
                    threshold_value=violation['limit_value'],
                    symbol=position.symbol
                )
                alerts.append(alert)

        return alerts

    async def _check_circuit_conditions(self, risk_metrics: RiskMetrics) -> List[RiskAlert]:
        """Vérifie les conditions de circuit breaker"""
        alerts = []

        # Vérifier si circuit breaker doit être déclenché
        should_trigger, reason = await self.circuit_breaker.should_trigger(risk_metrics)

        if should_trigger:
            alert = RiskAlert(
                id=f"circuit_breaker_{int(datetime.utcnow().timestamp())}",
                category=RiskCategory.MARKET_CONDITIONS,
                severity=AlertSeverity.EMERGENCY,
                message=f"Circuit breaker triggered: {reason}",
                current_value=Decimal("0"),  # Placeholder
                threshold_value=Decimal("0"),  # Placeholder
                auto_action_taken="emergency_stop"
            )
            alerts.append(alert)

            # Déclencher circuit breaker
            await self.circuit_breaker.trigger(reason)

        return alerts

    async def _process_alert(self, alert: RiskAlert) -> None:
        """Traite une nouvelle alerte"""
        # Éviter duplicatas
        similar_alerts = [
            a for a in self.active_alerts.values()
            if (a.category == alert.category and
                a.symbol == alert.symbol and
                a.severity == alert.severity)
        ]

        if similar_alerts:
            return  # Alerte similaire déjà active

        # Ajouter aux alertes actives
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)

        # Log selon sévérité
        if alert.severity == AlertSeverity.EMERGENCY:
            logger.critical(f"EMERGENCY ALERT: {alert.message}")
        elif alert.severity == AlertSeverity.CRITICAL:
            logger.error(f"CRITICAL ALERT: {alert.message}")
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(f"WARNING ALERT: {alert.message}")
        else:
            logger.info(f"INFO ALERT: {alert.message}")

        # Actions automatiques
        if alert.auto_action_taken:
            await self._execute_auto_action(alert)

        # Callbacks
        if self.on_alert_triggered:
            await self.on_alert_triggered(alert)

        if alert.severity == AlertSeverity.EMERGENCY and self.on_emergency_stop:
            await self.on_emergency_stop(alert)

    async def _execute_auto_action(self, alert: RiskAlert) -> None:
        """Exécute une action automatique basée sur l'alerte"""
        try:
            if alert.auto_action_taken == "emergency_stop":
                logger.critical("EXECUTING EMERGENCY STOP")
                # Implémenter arrêt d'urgence
                # Ceci devrait déclencher l'arrêt complet du trading

            elif alert.auto_action_taken == "reduce_position":
                logger.warning(f"AUTO ACTION: Reducing position for {alert.symbol}")
                # Implémenter réduction de position

            elif alert.auto_action_taken == "stop_trading":
                logger.warning("AUTO ACTION: Stopping trading operations")
                # Implémenter arrêt du trading

        except Exception as e:
            logger.error(f"Error executing auto action {alert.auto_action_taken}: {e}")

    async def _cleanup_resolved_alerts(self) -> None:
        """Nettoie les alertes résolues (anciennes)"""
        cutoff_time = datetime.utcnow() - timedelta(hours=1)

        resolved_alerts = [
            alert_id for alert_id, alert in self.active_alerts.items()
            if alert.timestamp < cutoff_time and alert.acknowledged
        ]

        for alert_id in resolved_alerts:
            del self.active_alerts[alert_id]

        if resolved_alerts:
            logger.info(f"Cleaned up {len(resolved_alerts)} resolved alerts")

    def _record_risk_metrics(self, metrics: RiskMetrics) -> None:
        """Enregistre les métriques dans l'historique"""
        self.risk_metrics_history.append(metrics)

        # Limiter taille historique
        if len(self.risk_metrics_history) > self.max_history_length:
            self.risk_metrics_history = self.risk_metrics_history[-self.max_history_length:]

    def _setup_default_thresholds(self) -> None:
        """Configure les seuils par défaut"""
        # Seuils de drawdown
        self.risk_thresholds.extend([
            RiskThreshold(
                category=RiskCategory.DRAWDOWN,
                severity=AlertSeverity.WARNING,
                threshold_value=Decimal("0.05"),  # 5%
                comparison="greater_than",
                auto_action="alert_only"
            ),
            RiskThreshold(
                category=RiskCategory.DRAWDOWN,
                severity=AlertSeverity.CRITICAL,
                threshold_value=Decimal("0.10"),  # 10%
                comparison="greater_than",
                auto_action="reduce_position"
            ),
            RiskThreshold(
                category=RiskCategory.DRAWDOWN,
                severity=AlertSeverity.EMERGENCY,
                threshold_value=Decimal("0.20"),  # 20%
                comparison="greater_than",
                auto_action="emergency_stop"
            ),

            # Seuils de concentration
            RiskThreshold(
                category=RiskCategory.CONCENTRATION,
                severity=AlertSeverity.WARNING,
                threshold_value=Decimal("0.30"),  # 30% dans un seul asset
                comparison="greater_than",
                auto_action="alert_only"
            ),
            RiskThreshold(
                category=RiskCategory.CONCENTRATION,
                severity=AlertSeverity.CRITICAL,
                threshold_value=Decimal("0.50"),  # 50% dans un seul asset
                comparison="greater_than",
                auto_action="reduce_position"
            )
        ])

    async def _get_current_portfolio(self) -> Optional[Portfolio]:
        """Obtient le portfolio actuel (placeholder)"""
        # Dans une vraie implémentation, ceci viendrait du portfolio service
        # ou serait injecté comme dépendance
        return None