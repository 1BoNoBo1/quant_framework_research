"""
Infrastructure Layer: Intelligent Alerting System
================================================

Système d'alerting intelligent avec ML pour détecter les anomalies,
groupement d'alertes, et suppression de bruit.
"""

import hashlib
import time
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Deque
import numpy as np
from threading import Lock, Thread
import json
import logging


class AlertSeverity(str, Enum):
    """Niveaux de sévérité des alertes"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertCategory(str, Enum):
    """Catégories d'alertes"""
    PERFORMANCE = "performance"
    RISK = "risk"
    TRADING = "trading"
    SYSTEM = "system"
    SECURITY = "security"
    DATA_QUALITY = "data_quality"


class AlertStatus(str, Enum):
    """Statut d'une alerte"""
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Représentation d'une alerte"""
    id: str
    title: str
    message: str
    severity: AlertSeverity
    category: AlertCategory
    source: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: AlertStatus = AlertStatus.OPEN
    metadata: Dict[str, Any] = field(default_factory=dict)
    fingerprint: Optional[str] = None
    group_id: Optional[str] = None
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    suppression_reason: Optional[str] = None

    def __post_init__(self):
        if not self.fingerprint:
            self.fingerprint = self._generate_fingerprint()

    def _generate_fingerprint(self) -> str:
        """Générer une empreinte unique pour l'alerte"""
        data = f"{self.title}:{self.source}:{self.category.value}:{self.severity.value}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convertir en dictionnaire"""
        return {
            "id": self.id,
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "metadata": self.metadata,
            "fingerprint": self.fingerprint,
            "group_id": self.group_id,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "suppression_reason": self.suppression_reason
        }


@dataclass
class AlertRule:
    """Règle de déclenchement d'alerte"""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity
    category: AlertCategory
    title_template: str
    message_template: str
    cooldown_seconds: int = 300  # Temps minimum entre alertes identiques
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_triggered: Optional[datetime] = None

    def evaluate(self, data: Dict[str, Any]) -> Optional[Alert]:
        """Évaluer la règle et générer une alerte si nécessaire"""
        if not self.enabled:
            return None

        # Vérifier le cooldown
        if self.last_triggered:
            if (datetime.utcnow() - self.last_triggered).total_seconds() < self.cooldown_seconds:
                return None

        # Évaluer la condition
        if self.condition(data):
            self.last_triggered = datetime.utcnow()
            return Alert(
                id=f"{self.name}_{int(time.time()*1000)}",
                title=self.title_template.format(**data),
                message=self.message_template.format(**data),
                severity=self.severity,
                category=self.category,
                source=self.name,
                metadata={**self.metadata, **data}
            )

        return None


class AlertChannel(ABC):
    """Interface pour un canal d'alerte"""

    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """Envoyer une alerte"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Nom du canal"""
        pass


class LogChannel(AlertChannel):
    """Canal d'alerte qui log les alertes"""

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

    @property
    def name(self) -> str:
        return "log"

    async def send(self, alert: Alert) -> bool:
        level_map = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }

        self.logger.log(
            level_map[alert.severity],
            f"[ALERT] [{alert.severity.value.upper()}] {alert.title}: {alert.message}",
            extra=alert.to_dict()
        )
        return True


class EmailChannel(AlertChannel):
    """Canal d'alerte par email"""

    def __init__(self, smtp_config: Dict[str, Any], recipients: List[str]):
        self.smtp_config = smtp_config
        self.recipients = recipients

    @property
    def name(self) -> str:
        return "email"

    async def send(self, alert: Alert) -> bool:
        # Implémentation simplifiée
        logging.info(f"Would send email alert to {self.recipients}: {alert.title}")
        return True


class SlackChannel(AlertChannel):
    """Canal d'alerte Slack"""

    def __init__(self, webhook_url: str, channel: str = "#alerts"):
        self.webhook_url = webhook_url
        self.channel = channel

    @property
    def name(self) -> str:
        return "slack"

    async def send(self, alert: Alert) -> bool:
        # Implémentation simplifiée
        color_map = {
            AlertSeverity.INFO: "good",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "danger",
            AlertSeverity.CRITICAL: "danger"
        }

        payload = {
            "channel": self.channel,
            "attachments": [{
                "color": color_map[alert.severity],
                "title": alert.title,
                "text": alert.message,
                "fields": [
                    {"title": "Severity", "value": alert.severity.value, "short": True},
                    {"title": "Category", "value": alert.category.value, "short": True},
                    {"title": "Source", "value": alert.source, "short": True},
                    {"title": "Time", "value": alert.timestamp.isoformat(), "short": True}
                ]
            }]
        }

        logging.info(f"Would send Slack alert: {json.dumps(payload)}")
        return True


class AnomalyDetector:
    """
    Détecteur d'anomalies basé sur des statistiques.
    Utilise une approche simple basée sur l'écart-type.
    """

    def __init__(self, window_size: int = 100, z_threshold: float = 3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self._data: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window_size))
        self._lock = Lock()

    def add_value(self, metric: str, value: float):
        """Ajouter une valeur pour une métrique"""
        with self._lock:
            self._data[metric].append(value)

    def is_anomaly(self, metric: str, value: float) -> bool:
        """Détecter si une valeur est anomale"""
        with self._lock:
            data = list(self._data.get(metric, []))

        if len(data) < 10:  # Pas assez de données
            return False

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return False

        z_score = abs((value - mean) / std)
        return z_score > self.z_threshold

    def get_statistics(self, metric: str) -> Dict[str, float]:
        """Obtenir les statistiques pour une métrique"""
        with self._lock:
            data = list(self._data.get(metric, []))

        if not data:
            return {}

        return {
            "mean": np.mean(data),
            "std": np.std(data),
            "min": np.min(data),
            "max": np.max(data),
            "count": len(data)
        }


class AlertCorrelator:
    """
    Corrélateur d'alertes pour grouper les alertes liées.
    """

    def __init__(self, time_window_seconds: int = 300):
        self.time_window_seconds = time_window_seconds
        self._groups: Dict[str, List[Alert]] = defaultdict(list)
        self._lock = Lock()

    def correlate(self, alert: Alert) -> str:
        """Corréler une alerte et retourner son group_id"""
        with self._lock:
            # Chercher un groupe existant
            for group_id, alerts in self._groups.items():
                if self._should_group(alert, alerts):
                    alerts.append(alert)
                    return group_id

            # Créer un nouveau groupe
            group_id = f"group_{int(time.time()*1000)}"
            self._groups[group_id] = [alert]
            return group_id

    def _should_group(self, alert: Alert, alerts: List[Alert]) -> bool:
        """Déterminer si une alerte doit être groupée"""
        if not alerts:
            return False

        # Vérifier la fenêtre temporelle
        latest_alert = max(alerts, key=lambda a: a.timestamp)
        if (alert.timestamp - latest_alert.timestamp).total_seconds() > self.time_window_seconds:
            return False

        # Grouper par fingerprint ou catégorie/source similaires
        for existing_alert in alerts:
            if alert.fingerprint == existing_alert.fingerprint:
                return True
            if (alert.category == existing_alert.category and
                alert.source == existing_alert.source):
                return True

        return False

    def get_group(self, group_id: str) -> List[Alert]:
        """Obtenir les alertes d'un groupe"""
        with self._lock:
            return self._groups.get(group_id, [])

    def cleanup(self):
        """Nettoyer les vieux groupes"""
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.time_window_seconds * 2)

        with self._lock:
            groups_to_remove = []
            for group_id, alerts in self._groups.items():
                latest = max(alerts, key=lambda a: a.timestamp)
                if latest.timestamp < cutoff_time:
                    groups_to_remove.append(group_id)

            for group_id in groups_to_remove:
                del self._groups[group_id]


class AlertManager:
    """
    Gestionnaire principal des alertes.
    """

    def __init__(self):
        self._alerts: Dict[str, Alert] = {}
        self._rules: Dict[str, AlertRule] = {}
        self._channels: Dict[str, AlertChannel] = {}
        self._anomaly_detector = AnomalyDetector()
        self._correlator = AlertCorrelator()
        self._suppression_rules: List[Callable[[Alert], bool]] = []
        self._lock = Lock()
        self._running = False

        # Configuration par défaut
        self._severity_channels = {
            AlertSeverity.INFO: ["log"],
            AlertSeverity.WARNING: ["log", "slack"],
            AlertSeverity.ERROR: ["log", "slack", "email"],
            AlertSeverity.CRITICAL: ["log", "slack", "email"]
        }

    def register_rule(self, rule: AlertRule):
        """Enregistrer une règle d'alerte"""
        with self._lock:
            self._rules[rule.name] = rule

    def register_channel(self, channel: AlertChannel):
        """Enregistrer un canal d'alerte"""
        with self._lock:
            self._channels[channel.name] = channel

    def add_suppression_rule(self, rule: Callable[[Alert], bool]):
        """Ajouter une règle de suppression"""
        self._suppression_rules.append(rule)

    def configure_severity_routing(self, severity: AlertSeverity, channels: List[str]):
        """Configurer le routage par sévérité"""
        self._severity_channels[severity] = channels

    def evaluate_rules(self, data: Dict[str, Any]):
        """Évaluer toutes les règles avec des données"""
        for rule in self._rules.values():
            alert = rule.evaluate(data)
            if alert:
                self.trigger_alert(alert)

    def trigger_alert(self, alert: Alert):
        """Déclencher une alerte"""
        # Vérifier la suppression
        if self._should_suppress(alert):
            alert.status = AlertStatus.SUPPRESSED
            alert.suppression_reason = "Matched suppression rule"
            logging.info(f"Alert suppressed: {alert.title}")
            return

        # Corréler avec d'autres alertes
        alert.group_id = self._correlator.correlate(alert)

        # Stocker l'alerte
        with self._lock:
            self._alerts[alert.id] = alert

        # Router vers les canaux appropriés
        self._route_alert(alert)

    def _should_suppress(self, alert: Alert) -> bool:
        """Vérifier si une alerte doit être supprimée"""
        for rule in self._suppression_rules:
            if rule(alert):
                return True
        return False

    def _route_alert(self, alert: Alert):
        """Router une alerte vers les canaux appropriés"""
        channels_to_use = self._severity_channels.get(alert.severity, ["log"])

        for channel_name in channels_to_use:
            channel = self._channels.get(channel_name)
            if channel:
                try:
                    import asyncio
                    asyncio.create_task(channel.send(alert))
                except Exception as e:
                    logging.error(f"Error sending alert to {channel_name}: {e}")

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Reconnaître une alerte"""
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.utcnow()
                alert.acknowledged_by = acknowledged_by

    def resolve_alert(self, alert_id: str):
        """Résoudre une alerte"""
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.utcnow()

    def get_alerts(
        self,
        status: Optional[AlertStatus] = None,
        severity: Optional[AlertSeverity] = None,
        category: Optional[AlertCategory] = None,
        since: Optional[datetime] = None
    ) -> List[Alert]:
        """Obtenir les alertes filtrées"""
        with self._lock:
            alerts = list(self._alerts.values())

        # Appliquer les filtres
        if status:
            alerts = [a for a in alerts if a.status == status]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if category:
            alerts = [a for a in alerts if a.category == category]
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Obtenir les statistiques des alertes"""
        with self._lock:
            alerts = list(self._alerts.values())

        stats = {
            "total": len(alerts),
            "by_status": {},
            "by_severity": {},
            "by_category": {}
        }

        for status in AlertStatus:
            stats["by_status"][status.value] = len([a for a in alerts if a.status == status])

        for severity in AlertSeverity:
            stats["by_severity"][severity.value] = len([a for a in alerts if a.severity == severity])

        for category in AlertCategory:
            stats["by_category"][category.value] = len([a for a in alerts if a.category == category])

        # Alertes récentes (dernière heure)
        recent_time = datetime.utcnow() - timedelta(hours=1)
        recent_alerts = [a for a in alerts if a.timestamp >= recent_time]
        stats["recent_count"] = len(recent_alerts)

        return stats

    def cleanup(self, days_old: int = 7):
        """Nettoyer les vieilles alertes résolues"""
        cutoff_time = datetime.utcnow() - timedelta(days=days_old)

        with self._lock:
            alerts_to_remove = []
            for alert_id, alert in self._alerts.items():
                if (alert.status == AlertStatus.RESOLVED and
                    alert.resolved_at and alert.resolved_at < cutoff_time):
                    alerts_to_remove.append(alert_id)

            for alert_id in alerts_to_remove:
                del self._alerts[alert_id]

        self._correlator.cleanup()

    def register_trading_rules(self):
        """Enregistrer les règles d'alerte spécifiques au trading"""

        # Alerte de P&L négatif important
        self.register_rule(AlertRule(
            name="large_loss",
            condition=lambda data: data.get("pnl", 0) < -10000,
            severity=AlertSeverity.ERROR,
            category=AlertCategory.TRADING,
            title_template="Large Loss Detected: ${pnl:.2f}",
            message_template="Portfolio {portfolio_id} has incurred a loss of ${pnl:.2f}",
            cooldown_seconds=600
        ))

        # Alerte de breach de limite de risque
        self.register_rule(AlertRule(
            name="risk_limit_breach",
            condition=lambda data: data.get("risk_breach", False),
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.RISK,
            title_template="Risk Limit Breach: {metric}",
            message_template="Risk metric {metric} = {value:.2f} exceeds limit {limit:.2f}",
            cooldown_seconds=300
        ))

        # Alerte de latence élevée
        self.register_rule(AlertRule(
            name="high_latency",
            condition=lambda data: data.get("latency_ms", 0) > 1000,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.PERFORMANCE,
            title_template="High Latency: {latency_ms}ms",
            message_template="Operation {operation} took {latency_ms}ms to complete",
            cooldown_seconds=300
        ))

        # Alerte d'échec d'ordre
        self.register_rule(AlertRule(
            name="order_failure",
            condition=lambda data: data.get("order_status") == "rejected",
            severity=AlertSeverity.ERROR,
            category=AlertCategory.TRADING,
            title_template="Order Rejected: {order_id}",
            message_template="Order {order_id} was rejected: {reason}",
            cooldown_seconds=60
        ))

        # Alerte de connexion perdue
        self.register_rule(AlertRule(
            name="connection_lost",
            condition=lambda data: data.get("connection_status") == "disconnected",
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.SYSTEM,
            title_template="Connection Lost: {service}",
            message_template="Lost connection to {service}: {error}",
            cooldown_seconds=60
        ))


# Instance globale
_global_alert_manager = AlertManager()

# Configurer les canaux par défaut
_global_alert_manager.register_channel(LogChannel())


def get_alert_manager() -> AlertManager:
    """Obtenir l'instance globale du gestionnaire d'alertes"""
    return _global_alert_manager