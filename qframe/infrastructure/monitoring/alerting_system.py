"""
Alerting System
==============

Multi-channel alerting system with intelligent routing and escalation.
"""

from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
import json
from abc import ABC, abstractmethod

from ...core.container import injectable


logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Niveaux de sévérité des alertes"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(str, Enum):
    """États d'une alerte"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class AlertChannel(str, Enum):
    """Canaux de notification"""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    DISCORD = "discord"
    WEBHOOK = "webhook"
    PHONE = "phone"
    DASHBOARD = "dashboard"


@dataclass
class Alert:
    """Alerte système"""
    id: str
    title: str
    message: str
    severity: AlertSeverity
    source: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: AlertStatus = AlertStatus.ACTIVE

    # Métadonnées
    tags: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Union[float, int, str]] = field(default_factory=dict)

    # Gestion des alertes
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    escalation_level: int = 0

    # Configuration notification
    channels: List[AlertChannel] = field(default_factory=list)
    cooldown_until: Optional[datetime] = None


@dataclass
class AlertRule:
    """Règle de déclenchement d'alerte"""
    id: str
    name: str
    condition: str  # Expression évaluable
    severity: AlertSeverity
    channels: List[AlertChannel]

    # Configuration
    enabled: bool = True
    cooldown_minutes: int = 5
    escalation_rules: List[Dict[str, Any]] = field(default_factory=list)

    # Filtres
    tags_filter: Dict[str, str] = field(default_factory=dict)
    time_filter: Optional[Dict[str, Any]] = None  # ex: {"start": "09:00", "end": "17:00"}

    # Historique
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class NotificationConfig:
    """Configuration d'un canal de notification"""
    channel: AlertChannel
    enabled: bool = True

    # Configuration spécifique au canal
    config: Dict[str, Any] = field(default_factory=dict)

    # Filtres
    severity_filter: List[AlertSeverity] = field(default_factory=lambda: list(AlertSeverity))
    time_restrictions: Optional[Dict[str, Any]] = None


class NotificationChannel(ABC):
    """Interface pour canaux de notification"""

    @abstractmethod
    async def send_notification(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Envoie une notification"""
        pass

    @abstractmethod
    async def test_connection(self, config: Dict[str, Any]) -> bool:
        """Teste la connexion au canal"""
        pass


@injectable
class AlertingSystem:
    """
    Système d'alerting intelligent.

    Fonctionnalités:
    - Multiples canaux de notification
    - Règles d'escalade automatique
    - Suppression d'alertes dupliquées
    - Cooldowns et rate limiting
    - Corrélation d'alertes
    - Dashboard d'alerting
    """

    def __init__(
        self,
        default_channels: Optional[List[AlertChannel]] = None,
        max_alerts_history: int = 10000,
        auto_escalation_enabled: bool = True
    ):
        self.default_channels = default_channels or [AlertChannel.DASHBOARD]
        self.max_alerts_history = max_alerts_history
        self.auto_escalation_enabled = auto_escalation_enabled

        # État des alertes
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_rules: Dict[str, AlertRule] = {}

        # Configuration des canaux
        self.notification_configs: Dict[AlertChannel, NotificationConfig] = {}
        self.notification_channels: Dict[AlertChannel, NotificationChannel] = {}

        # Statistiques
        self.total_alerts_sent = 0
        self.alerts_by_severity = {severity: 0 for severity in AlertSeverity}
        self.alerts_by_channel = {channel: 0 for channel in AlertChannel}

        # Corrélation et suppression
        self.correlation_rules: List[Dict[str, Any]] = []
        self.suppression_rules: List[Dict[str, Any]] = []

        self._setup_default_channels()
        self._setup_default_rules()

        # Démarrer tâches en arrière-plan
        asyncio.create_task(self._escalation_loop())
        asyncio.create_task(self._cleanup_loop())

    # === Gestion des alertes ===

    async def send_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        source: str,
        tags: Optional[Dict[str, str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        channels: Optional[List[AlertChannel]] = None
    ) -> str:
        """
        Envoie une nouvelle alerte.

        Returns:
            ID de l'alerte créée
        """
        # Générer ID unique
        alert_id = f"alert_{int(datetime.utcnow().timestamp())}_{source}"

        # Créer alerte
        alert = Alert(
            id=alert_id,
            title=title,
            message=message,
            severity=severity,
            source=source,
            tags=tags or {},
            metrics=metrics or {},
            channels=channels or self.default_channels
        )

        # Vérifier suppression
        if await self._should_suppress_alert(alert):
            logger.info(f"Alert suppressed: {alert_id}")
            return alert_id

        # Vérifier règles d'alerte
        matching_rules = await self._find_matching_rules(alert)
        if matching_rules:
            # Utiliser configuration de la première règle
            rule = matching_rules[0]
            alert.channels = rule.channels
            alert.severity = rule.severity  # Peut override la sévérité

            # Vérifier cooldown
            if rule.last_triggered and rule.cooldown_minutes > 0:
                cooldown_end = rule.last_triggered + timedelta(minutes=rule.cooldown_minutes)
                if datetime.utcnow() < cooldown_end:
                    logger.info(f"Alert in cooldown: {alert_id}")
                    return alert_id

            # Mettre à jour règle
            rule.last_triggered = datetime.utcnow()
            rule.trigger_count += 1

        # Corrélation avec alertes existantes
        correlated_alert = await self._find_correlated_alert(alert)
        if correlated_alert:
            # Mettre à jour alerte existante au lieu d'en créer une nouvelle
            await self._update_correlated_alert(correlated_alert, alert)
            return correlated_alert.id

        # Ajouter aux alertes actives
        self.active_alerts[alert_id] = alert

        # Envoyer notifications
        await self._send_notifications(alert)

        # Enregistrer dans historique
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_alerts_history:
            self.alert_history = self.alert_history[-self.max_alerts_history:]

        # Statistiques
        self.total_alerts_sent += 1
        self.alerts_by_severity[severity] += 1

        logger.info(f"Alert sent: {alert_id} ({severity.value}) - {title}")
        return alert_id

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acquitte une alerte"""
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.utcnow()

        logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
        return True

    async def resolve_alert(self, alert_id: str, resolved_by: Optional[str] = None) -> bool:
        """Résout une alerte"""
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()

        # Retirer des alertes actives
        del self.active_alerts[alert_id]

        logger.info(f"Alert resolved: {alert_id}")
        return True

    async def suppress_alert(self, alert_id: str, reason: str) -> bool:
        """Supprime une alerte"""
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.SUPPRESSED
        alert.tags["suppression_reason"] = reason

        # Retirer des alertes actives
        del self.active_alerts[alert_id]

        logger.info(f"Alert suppressed: {alert_id} - {reason}")
        return True

    # === Configuration des règles ===

    async def add_alert_rule(self, rule: AlertRule) -> None:
        """Ajoute une règle d'alerte"""
        self.alert_rules[rule.id] = rule
        logger.info(f"Added alert rule: {rule.id}")

    async def remove_alert_rule(self, rule_id: str) -> bool:
        """Supprime une règle d'alerte"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
            return True
        return False

    async def update_alert_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Met à jour une règle d'alerte"""
        if rule_id not in self.alert_rules:
            return False

        rule = self.alert_rules[rule_id]
        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)

        logger.info(f"Updated alert rule: {rule_id}")
        return True

    # === Configuration des canaux ===

    async def configure_notification_channel(
        self,
        channel: AlertChannel,
        config: NotificationConfig
    ) -> None:
        """Configure un canal de notification"""
        self.notification_configs[channel] = config

        # Initialiser le canal si pas déjà fait
        if channel not in self.notification_channels:
            channel_impl = self._create_notification_channel(channel)
            if channel_impl:
                self.notification_channels[channel] = channel_impl

        logger.info(f"Configured notification channel: {channel.value}")

    async def test_notification_channel(self, channel: AlertChannel) -> bool:
        """Teste un canal de notification"""
        if channel not in self.notification_channels:
            return False

        config = self.notification_configs.get(channel, {})
        return await self.notification_channels[channel].test_connection(config.config)

    # === Requêtes et statistiques ===

    async def get_active_alerts(
        self,
        severity_filter: Optional[List[AlertSeverity]] = None,
        source_filter: Optional[str] = None
    ) -> List[Alert]:
        """Retourne les alertes actives avec filtres optionnels"""
        alerts = list(self.active_alerts.values())

        if severity_filter:
            alerts = [a for a in alerts if a.severity in severity_filter]

        if source_filter:
            alerts = [a for a in alerts if source_filter in a.source]

        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)

    async def get_alert_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques d'alerting"""
        # Alertes des dernières 24h
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        recent_alerts = [a for a in self.alert_history if a.timestamp >= cutoff_time]

        # Temps moyen de résolution
        resolved_alerts = [a for a in recent_alerts if a.resolved_at]
        avg_resolution_time = None

        if resolved_alerts:
            resolution_times = [
                (a.resolved_at - a.timestamp).total_seconds() / 60
                for a in resolved_alerts
            ]
            avg_resolution_time = sum(resolution_times) / len(resolution_times)

        return {
            "total_alerts_sent": self.total_alerts_sent,
            "active_alerts_count": len(self.active_alerts),
            "alerts_last_24h": len(recent_alerts),
            "alerts_by_severity_24h": {
                severity.value: len([a for a in recent_alerts if a.severity == severity])
                for severity in AlertSeverity
            },
            "alerts_by_channel": dict(self.alerts_by_channel),
            "avg_resolution_time_minutes": avg_resolution_time,
            "configured_channels": len(self.notification_configs),
            "alert_rules_count": len(self.alert_rules)
        }

    async def get_alert_dashboard_data(self) -> Dict[str, Any]:
        """Retourne les données pour le dashboard d'alerting"""
        active_alerts = await self.get_active_alerts()
        stats = await self.get_alert_statistics()

        # Alertes critiques non acquittées
        critical_unack = [
            a for a in active_alerts
            if a.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
            and a.status == AlertStatus.ACTIVE
        ]

        # Tendances d'alertes (dernières heures)
        hourly_counts = {}
        for i in range(24):
            hour_start = datetime.utcnow() - timedelta(hours=i+1)
            hour_end = datetime.utcnow() - timedelta(hours=i)
            count = len([
                a for a in self.alert_history
                if hour_start <= a.timestamp < hour_end
            ])
            hourly_counts[f"-{i}h"] = count

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "active_alerts": [
                {
                    "id": a.id,
                    "title": a.title,
                    "severity": a.severity.value,
                    "source": a.source,
                    "timestamp": a.timestamp.isoformat(),
                    "status": a.status.value,
                    "escalation_level": a.escalation_level
                }
                for a in active_alerts[:20]  # Top 20
            ],
            "critical_unacknowledged": len(critical_unack),
            "statistics": stats,
            "hourly_trends": hourly_counts,
            "channel_status": {
                channel.value: {
                    "configured": channel in self.notification_configs,
                    "enabled": self.notification_configs.get(channel, NotificationConfig(channel)).enabled
                }
                for channel in AlertChannel
            }
        }

    # === Méthodes privées ===

    async def _send_notifications(self, alert: Alert) -> None:
        """Envoie les notifications pour une alerte"""
        for channel in alert.channels:
            try:
                # Vérifier configuration du canal
                if channel not in self.notification_configs:
                    continue

                config = self.notification_configs[channel]
                if not config.enabled:
                    continue

                # Vérifier filtre de sévérité
                if alert.severity not in config.severity_filter:
                    continue

                # Vérifier restrictions temporelles
                if not await self._check_time_restrictions(config.time_restrictions):
                    continue

                # Envoyer notification
                if channel in self.notification_channels:
                    success = await self.notification_channels[channel].send_notification(alert, config.config)
                    if success:
                        self.alerts_by_channel[channel] += 1
                        logger.debug(f"Notification sent via {channel.value} for alert {alert.id}")

            except Exception as e:
                logger.error(f"Error sending notification via {channel.value}: {e}")

    async def _find_matching_rules(self, alert: Alert) -> List[AlertRule]:
        """Trouve les règles correspondant à une alerte"""
        matching_rules = []

        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue

            # Vérifier filtres de tags
            if rule.tags_filter:
                if not all(alert.tags.get(k) == v for k, v in rule.tags_filter.items()):
                    continue

            # Vérifier condition (expression simplifiée)
            if await self._evaluate_condition(rule.condition, alert):
                matching_rules.append(rule)

        return matching_rules

    async def _evaluate_condition(self, condition: str, alert: Alert) -> bool:
        """Évalue une condition d'alerte (implémentation simplifiée)"""
        try:
            # Remplacer variables dans la condition
            context = {
                "severity": alert.severity.value,
                "source": alert.source,
                **alert.tags,
                **{f"metric_{k}": v for k, v in alert.metrics.items()}
            }

            # Évaluation simplifiée (dans la réalité, utiliserait un parser sécurisé)
            if "severity == 'critical'" in condition:
                return alert.severity == AlertSeverity.CRITICAL
            elif "source.startswith('trading')" in condition:
                return alert.source.startswith('trading')

            return True  # Par défaut, accepter

        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False

    async def _should_suppress_alert(self, alert: Alert) -> bool:
        """Détermine si une alerte doit être supprimée"""
        # Vérifier règles de suppression globales
        for rule in self.suppression_rules:
            if await self._matches_suppression_rule(alert, rule):
                return True

        # Vérifier doublons récents
        recent_cutoff = datetime.utcnow() - timedelta(minutes=5)
        similar_alerts = [
            a for a in self.alert_history
            if (a.timestamp >= recent_cutoff and
                a.title == alert.title and
                a.source == alert.source and
                a.severity == alert.severity)
        ]

        return len(similar_alerts) > 0

    async def _find_correlated_alert(self, alert: Alert) -> Optional[Alert]:
        """Trouve une alerte corrélée existante"""
        for existing_alert in self.active_alerts.values():
            if await self._are_alerts_correlated(existing_alert, alert):
                return existing_alert
        return None

    async def _are_alerts_correlated(self, alert1: Alert, alert2: Alert) -> bool:
        """Détermine si deux alertes sont corrélées"""
        # Corrélation simple basée sur source et tags
        if alert1.source == alert2.source:
            # Vérifier tags communs
            common_tags = set(alert1.tags.keys()) & set(alert2.tags.keys())
            if len(common_tags) >= 2:  # Au moins 2 tags en commun
                return all(alert1.tags[tag] == alert2.tags[tag] for tag in common_tags)

        return False

    async def _update_correlated_alert(self, existing_alert: Alert, new_alert: Alert) -> None:
        """Met à jour une alerte corrélée"""
        # Mettre à jour timestamp et escalation
        existing_alert.timestamp = new_alert.timestamp
        existing_alert.escalation_level += 1

        # Combiner métriques
        existing_alert.metrics.update(new_alert.metrics)

        logger.info(f"Updated correlated alert: {existing_alert.id} (escalation level {existing_alert.escalation_level})")

    async def _escalation_loop(self) -> None:
        """Boucle d'escalade automatique des alertes"""
        while True:
            try:
                if self.auto_escalation_enabled:
                    await self._process_escalations()

                await asyncio.sleep(60)  # Vérifier toutes les minutes

            except Exception as e:
                logger.error(f"Error in escalation loop: {e}")
                await asyncio.sleep(300)  # Pause plus longue en cas d'erreur

    async def _process_escalations(self) -> None:
        """Traite les escalades d'alertes"""
        current_time = datetime.utcnow()

        for alert in self.active_alerts.values():
            if alert.status != AlertStatus.ACTIVE:
                continue

            # Vérifier si escalade nécessaire
            alert_age_minutes = (current_time - alert.timestamp).total_seconds() / 60

            # Escalade basée sur sévérité et âge
            escalation_thresholds = {
                AlertSeverity.EMERGENCY: 5,    # 5 minutes
                AlertSeverity.CRITICAL: 15,    # 15 minutes
                AlertSeverity.WARNING: 60,     # 1 heure
                AlertSeverity.INFO: 240        # 4 heures
            }

            threshold = escalation_thresholds.get(alert.severity, 60)

            if alert_age_minutes > threshold and alert.escalation_level == 0:
                await self._escalate_alert(alert)

    async def _escalate_alert(self, alert: Alert) -> None:
        """Escalade une alerte"""
        alert.escalation_level += 1

        # Envoyer via canaux d'escalade
        escalation_channels = [AlertChannel.EMAIL, AlertChannel.SMS]

        for channel in escalation_channels:
            if channel in self.notification_channels:
                config = self.notification_configs.get(channel, NotificationConfig(channel))
                await self.notification_channels[channel].send_notification(alert, config.config)

        logger.warning(f"Alert escalated: {alert.id} (level {alert.escalation_level})")

    async def _cleanup_loop(self) -> None:
        """Boucle de nettoyage des anciennes alertes"""
        while True:
            try:
                # Nettoyer alertes résolues anciennes
                cutoff_time = datetime.utcnow() - timedelta(hours=24)

                old_alerts = [
                    alert_id for alert_id, alert in self.active_alerts.items()
                    if alert.resolved_at and alert.resolved_at < cutoff_time
                ]

                for alert_id in old_alerts:
                    del self.active_alerts[alert_id]

                if old_alerts:
                    logger.info(f"Cleaned up {len(old_alerts)} old resolved alerts")

                await asyncio.sleep(3600)  # Nettoyer toutes les heures

            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)

    async def _check_time_restrictions(self, restrictions: Optional[Dict[str, Any]]) -> bool:
        """Vérifie les restrictions temporelles"""
        if not restrictions:
            return True

        current_time = datetime.utcnow().time()

        # Vérifier plage horaire
        if "start" in restrictions and "end" in restrictions:
            start_time = datetime.strptime(restrictions["start"], "%H:%M").time()
            end_time = datetime.strptime(restrictions["end"], "%H:%M").time()

            if start_time <= end_time:
                return start_time <= current_time <= end_time
            else:  # Plage qui traverse minuit
                return current_time >= start_time or current_time <= end_time

        return True

    async def _matches_suppression_rule(self, alert: Alert, rule: Dict[str, Any]) -> bool:
        """Vérifie si une alerte correspond à une règle de suppression"""
        # Implémentation simplifiée
        if "severity" in rule and alert.severity.value not in rule["severity"]:
            return False

        if "source_pattern" in rule and rule["source_pattern"] not in alert.source:
            return False

        return True

    def _setup_default_channels(self) -> None:
        """Configure les canaux par défaut"""
        # Dashboard (toujours actif)
        self.notification_configs[AlertChannel.DASHBOARD] = NotificationConfig(
            channel=AlertChannel.DASHBOARD,
            enabled=True
        )

        # Email (exemple de configuration)
        self.notification_configs[AlertChannel.EMAIL] = NotificationConfig(
            channel=AlertChannel.EMAIL,
            enabled=False,  # Désactivé par défaut
            config={
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "from_email": "alerts@qframe.com",
                "recipients": ["admin@qframe.com"]
            },
            severity_filter=[AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
        )

    def _setup_default_rules(self) -> None:
        """Configure les règles par défaut"""
        default_rules = [
            AlertRule(
                id="critical_trading_error",
                name="Critical Trading Error",
                condition="severity == 'critical' and source.startswith('trading')",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.DASHBOARD],
                cooldown_minutes=10
            ),
            AlertRule(
                id="high_drawdown",
                name="High Portfolio Drawdown",
                condition="metric_drawdown > 0.05",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.DASHBOARD],
                cooldown_minutes=30
            )
        ]

        for rule in default_rules:
            self.alert_rules[rule.id] = rule

    def _create_notification_channel(self, channel: AlertChannel) -> Optional[NotificationChannel]:
        """Crée une instance de canal de notification"""
        # Factory pour les canaux
        if channel == AlertChannel.EMAIL:
            return EmailNotificationChannel()
        elif channel == AlertChannel.SLACK:
            return SlackNotificationChannel()
        elif channel == AlertChannel.WEBHOOK:
            return WebhookNotificationChannel()
        elif channel == AlertChannel.DASHBOARD:
            return DashboardNotificationChannel()

        return None


# === Implémentations des canaux ===

class EmailNotificationChannel(NotificationChannel):
    """Canal de notification email"""

    async def send_notification(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Envoie une notification email"""
        try:
            # Simulation d'envoi email
            logger.info(f"[EMAIL] Alert: {alert.title} - {alert.message}")
            return True

        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            return False

    async def test_connection(self, config: Dict[str, Any]) -> bool:
        """Teste la connexion SMTP"""
        # Simulation de test de connexion
        return True


class SlackNotificationChannel(NotificationChannel):
    """Canal de notification Slack"""

    async def send_notification(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Envoie une notification Slack"""
        try:
            logger.info(f"[SLACK] Alert: {alert.title} - {alert.message}")
            return True

        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return False

    async def test_connection(self, config: Dict[str, Any]) -> bool:
        """Teste la connexion Slack"""
        return True


class WebhookNotificationChannel(NotificationChannel):
    """Canal de notification webhook"""

    async def send_notification(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Envoie une notification webhook"""
        try:
            logger.info(f"[WEBHOOK] Alert: {alert.title} - {alert.message}")
            return True

        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
            return False

    async def test_connection(self, config: Dict[str, Any]) -> bool:
        """Teste la connexion webhook"""
        return True


class DashboardNotificationChannel(NotificationChannel):
    """Canal de notification dashboard"""

    async def send_notification(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Affiche notification sur dashboard"""
        logger.info(f"[DASHBOARD] Alert: {alert.title} - {alert.message}")
        return True

    async def test_connection(self, config: Dict[str, Any]) -> bool:
        """Pas de test nécessaire pour dashboard"""
        return True