"""
Paper Alert System
==================

Automated monitoring and alerting for new research publications.
Configurable rules, multiple channels, and intelligent filtering.
"""

from typing import Dict, List, Optional, Any, Tuple, Union, Callable, Set
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import logging
import asyncio
import json
import re
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import requests
from pathlib import Path

from qframe.core.container import injectable
from qframe.core.config import FrameworkConfig
from qframe.research.external.research_api_manager import ResearchAPIManager

logger = logging.getLogger(__name__)


class AlertTrigger(Enum):
    """Types de déclencheurs d'alerte"""
    NEW_PAPER = "new_paper"                    # Nouveau papier publié
    KEYWORD_MATCH = "keyword_match"            # Correspondance de mots-clés
    AUTHOR_PUBLICATION = "author_publication"  # Publication d'un auteur suivi
    CITATION_THRESHOLD = "citation_threshold"  # Seuil de citations atteint
    IMPACT_FACTOR = "impact_factor"           # Facteur d'impact du journal
    ARXIV_CATEGORY = "arxiv_category"         # Catégorie arXiv spécifique
    COLLABORATIVE_FILTERING = "collaborative" # Recommandation collaborative


class AlertChannel(Enum):
    """Canaux de notification"""
    EMAIL = "email"
    SLACK = "slack"
    DISCORD = "discord"
    WEBHOOK = "webhook"
    SMS = "sms"
    TELEGRAM = "telegram"
    FILE_EXPORT = "file_export"


class AlertPriority(Enum):
    """Priorités d'alerte"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class FilterOperator(Enum):
    """Opérateurs de filtrage"""
    CONTAINS = "contains"
    EQUALS = "equals"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    IN_LIST = "in_list"
    NOT_IN_LIST = "not_in_list"


@dataclass
class AlertFilter:
    """Filtre pour alertes"""
    field: str  # "title", "abstract", "authors", "journal", "year", "citations"
    operator: FilterOperator
    value: Any
    case_sensitive: bool = False
    weight: float = 1.0  # Poids pour scoring


@dataclass
class AlertRule:
    """Règle d'alerte"""
    rule_id: str
    name: str
    description: str
    trigger: AlertTrigger
    priority: AlertPriority

    # Critères de filtrage
    filters: List[AlertFilter] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    authors: List[str] = field(default_factory=list)
    journals: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)

    # Seuils
    min_citation_count: Optional[int] = None
    min_impact_factor: Optional[float] = None
    min_relevance_score: float = 0.5

    # Configuration temporelle
    check_frequency: timedelta = timedelta(hours=24)
    lookback_period: timedelta = timedelta(days=7)

    # Canaux de notification
    notification_channels: List[AlertChannel] = field(default_factory=list)

    # État
    is_active: bool = True
    last_check: Optional[datetime] = None
    total_alerts_sent: int = 0

    # Métadonnées
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"


@dataclass
class PaperAlert:
    """Alerte générée pour un papier"""
    alert_id: str
    rule_id: str
    paper_data: Dict[str, Any]

    # Scoring
    relevance_score: float
    matched_filters: List[str]
    matched_keywords: List[str]

    # Métadonnées
    triggered_at: datetime = field(default_factory=datetime.utcnow)
    priority: AlertPriority = AlertPriority.MEDIUM
    sent_to_channels: List[AlertChannel] = field(default_factory=list)

    # État
    is_sent: bool = False
    is_dismissed: bool = False
    user_feedback: Optional[str] = None


@dataclass
class NotificationChannel:
    """Configuration d'un canal de notification"""
    channel_type: AlertChannel
    name: str
    config: Dict[str, Any]
    is_active: bool = True

    # Rate limiting
    max_alerts_per_hour: int = 10
    max_alerts_per_day: int = 50

    # Filtres spécifiques au canal
    min_priority: AlertPriority = AlertPriority.LOW
    blacklisted_keywords: List[str] = field(default_factory=list)


@injectable
class PaperAlertSystem:
    """
    Système d'alerte automatique pour publications de recherche.

    Fonctionnalités:
    - Monitoring continu de multiples sources
    - Règles d'alerte sophistiquées avec scoring
    - Filtrage intelligent et dédoublonnage
    - Notifications multi-canaux
    - Apprentissage des préférences utilisateur
    - Analytics et métriques détaillées
    """

    def __init__(self, config: FrameworkConfig, api_manager: ResearchAPIManager):
        self.config = config
        self.api_manager = api_manager

        # Règles et alertes
        self.rules: Dict[str, AlertRule] = {}
        self.notification_channels: Dict[str, NotificationChannel] = {}
        self.alert_history: List[PaperAlert] = []

        # Cache et state
        self.processed_papers: Set[str] = set()  # IDs des papiers déjà traités
        self.monitoring_task: Optional[asyncio.Task] = None

        # Métriques
        self.metrics = {
            "total_papers_processed": 0,
            "total_alerts_generated": 0,
            "alerts_by_rule": {},
            "alerts_by_priority": {p.name: 0 for p in AlertPriority},
            "notifications_sent": 0,
            "notification_failures": 0,
            "false_positive_rate": 0.0,
            "user_engagement": {
                "clicks": 0,
                "dismissals": 0,
                "positive_feedback": 0,
                "negative_feedback": 0
            }
        }

        logger.info("Paper Alert System initialized")

    async def start_monitoring(self):
        """Démarre le monitoring automatique"""
        if self.monitoring_task is None or self.monitoring_task.done():
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Paper monitoring started")
        else:
            logger.warning("Monitoring already running")

    async def stop_monitoring(self):
        """Arrête le monitoring"""
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Paper monitoring stopped")

    def add_rule(self, rule: AlertRule):
        """Ajoute une règle d'alerte"""
        self.rules[rule.rule_id] = rule
        self.metrics["alerts_by_rule"][rule.rule_id] = 0
        logger.info(f"Alert rule added: {rule.rule_id}")

    def remove_rule(self, rule_id: str):
        """Supprime une règle d'alerte"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Alert rule removed: {rule_id}")

    def add_notification_channel(self, channel: NotificationChannel):
        """Ajoute un canal de notification"""
        self.notification_channels[channel.name] = channel
        logger.info(f"Notification channel added: {channel.name}")

    async def process_new_papers(self, papers: List[Dict[str, Any]]) -> List[PaperAlert]:
        """Traite de nouveaux papiers et génère des alertes"""
        generated_alerts = []

        for paper in papers:
            paper_id = self._extract_paper_id(paper)

            # Éviter les doublons
            if paper_id in self.processed_papers:
                continue

            self.processed_papers.add(paper_id)
            self.metrics["total_papers_processed"] += 1

            # Évaluer contre toutes les règles actives
            for rule in self.rules.values():
                if not rule.is_active:
                    continue

                alert = await self._evaluate_paper_against_rule(paper, rule)
                if alert:
                    generated_alerts.append(alert)
                    self.alert_history.append(alert)

                    # Mettre à jour métriques
                    self.metrics["total_alerts_generated"] += 1
                    self.metrics["alerts_by_rule"][rule.rule_id] += 1
                    self.metrics["alerts_by_priority"][alert.priority.name] += 1

                    # Envoyer notifications
                    await self._send_alert_notifications(alert, rule)

        return generated_alerts

    async def manual_check(self, rule_id: Optional[str] = None) -> List[PaperAlert]:
        """Vérification manuelle immédiate"""
        logger.info(f"Manual check triggered for rule: {rule_id or 'all'}")

        # Sélectionner les règles à vérifier
        rules_to_check = []
        if rule_id and rule_id in self.rules:
            rules_to_check = [self.rules[rule_id]]
        else:
            rules_to_check = list(self.rules.values())

        all_alerts = []

        for rule in rules_to_check:
            if not rule.is_active:
                continue

            # Récupérer les papiers récents
            papers = await self._fetch_papers_for_rule(rule)

            # Traiter les papiers
            alerts = await self.process_new_papers(papers)
            all_alerts.extend(alerts)

        return all_alerts

    async def _monitoring_loop(self):
        """Boucle principale de monitoring"""
        logger.info("Monitoring loop started")

        try:
            while True:
                # Vérifier chaque règle selon sa fréquence
                for rule in self.rules.values():
                    if not rule.is_active:
                        continue

                    # Vérifier si il est temps de checker cette règle
                    now = datetime.utcnow()
                    if rule.last_check is None or (now - rule.last_check) >= rule.check_frequency:
                        try:
                            await self._check_rule(rule)
                            rule.last_check = now
                        except Exception as e:
                            logger.error(f"Failed to check rule {rule.rule_id}: {e}")

                # Attendre avant la prochaine itération
                await asyncio.sleep(300)  # 5 minutes

        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
            raise

    async def _check_rule(self, rule: AlertRule):
        """Vérifie une règle spécifique"""
        logger.debug(f"Checking rule: {rule.rule_id}")

        # Récupérer les nouveaux papiers
        papers = await self._fetch_papers_for_rule(rule)

        # Filtrer les papiers déjà traités
        new_papers = [p for p in papers if self._extract_paper_id(p) not in self.processed_papers]

        if new_papers:
            logger.info(f"Found {len(new_papers)} new papers for rule {rule.rule_id}")
            await self.process_new_papers(new_papers)

    async def _fetch_papers_for_rule(self, rule: AlertRule) -> List[Dict[str, Any]]:
        """Récupère les papiers pour une règle"""
        papers = []

        try:
            if rule.trigger == AlertTrigger.NEW_PAPER or rule.trigger == AlertTrigger.KEYWORD_MATCH:
                # Recherche par mots-clés
                if rule.keywords:
                    query = " ".join(rule.keywords)
                    search_results = await self.api_manager.search_papers(
                        query=query,
                        providers=["arxiv", "semantic_scholar"],
                        limit=50
                    )
                    papers.extend(search_results)

            elif rule.trigger == AlertTrigger.AUTHOR_PUBLICATION:
                # Recherche par auteur
                for author in rule.authors:
                    search_results = await self.api_manager.search_papers(
                        query=f'author:"{author}"',
                        providers=["semantic_scholar"],
                        limit=20
                    )
                    papers.extend(search_results)

            elif rule.trigger == AlertTrigger.ARXIV_CATEGORY:
                # Recherche par catégorie arXiv
                for category in rule.categories:
                    search_results = await self.api_manager.search_papers(
                        query=f"cat:{category}",
                        providers=["arxiv"],
                        limit=30
                    )
                    papers.extend(search_results)

        except Exception as e:
            logger.error(f"Failed to fetch papers for rule {rule.rule_id}: {e}")

        return papers

    async def _evaluate_paper_against_rule(self, paper: Dict[str, Any], rule: AlertRule) -> Optional[PaperAlert]:
        """Évalue un papier contre une règle"""

        # Calculer le score de pertinence
        relevance_score = 0.0
        matched_filters = []
        matched_keywords = []

        # Évaluer les filtres
        for filter_def in rule.filters:
            if self._evaluate_filter(paper, filter_def):
                relevance_score += filter_def.weight
                matched_filters.append(f"{filter_def.field}_{filter_def.operator.value}")

        # Évaluer les mots-clés
        for keyword in rule.keywords:
            if self._keyword_matches_paper(keyword, paper):
                relevance_score += 0.5
                matched_keywords.append(keyword)

        # Évaluer les auteurs
        for author in rule.authors:
            if self._author_matches_paper(author, paper):
                relevance_score += 1.0
                matched_filters.append(f"author_{author}")

        # Normaliser le score (0-1)
        max_possible_score = len(rule.filters) + len(rule.keywords) * 0.5 + len(rule.authors)
        if max_possible_score > 0:
            relevance_score = min(relevance_score / max_possible_score, 1.0)

        # Vérifier seuil minimum
        if relevance_score < rule.min_relevance_score:
            return None

        # Vérifier autres critères
        if rule.min_citation_count:
            citations = paper.get("citations", 0)
            if isinstance(citations, (int, float)) and citations < rule.min_citation_count:
                return None

        # Générer l'alerte
        alert_id = f"alert_{rule.rule_id}_{int(datetime.utcnow().timestamp() * 1000)}"

        return PaperAlert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            paper_data=paper,
            relevance_score=relevance_score,
            matched_filters=matched_filters,
            matched_keywords=matched_keywords,
            priority=rule.priority
        )

    def _evaluate_filter(self, paper: Dict[str, Any], filter_def: AlertFilter) -> bool:
        """Évalue un filtre contre un papier"""
        field_value = paper.get(filter_def.field, "")

        if field_value is None:
            return False

        # Convertir en string si nécessaire
        if not isinstance(field_value, str) and filter_def.operator in [
            FilterOperator.CONTAINS, FilterOperator.STARTS_WITH,
            FilterOperator.ENDS_WITH, FilterOperator.REGEX
        ]:
            field_value = str(field_value)

        # Gérer la casse
        if isinstance(field_value, str) and not filter_def.case_sensitive:
            field_value = field_value.lower()
            if isinstance(filter_def.value, str):
                filter_value = filter_def.value.lower()
            else:
                filter_value = filter_def.value
        else:
            filter_value = filter_def.value

        # Évaluer selon l'opérateur
        if filter_def.operator == FilterOperator.CONTAINS:
            return filter_value in field_value
        elif filter_def.operator == FilterOperator.EQUALS:
            return field_value == filter_value
        elif filter_def.operator == FilterOperator.STARTS_WITH:
            return field_value.startswith(filter_value)
        elif filter_def.operator == FilterOperator.ENDS_WITH:
            return field_value.endswith(filter_value)
        elif filter_def.operator == FilterOperator.REGEX:
            try:
                return bool(re.search(filter_value, field_value))
            except re.error:
                return False
        elif filter_def.operator == FilterOperator.GREATER_THAN:
            try:
                return float(field_value) > float(filter_value)
            except (ValueError, TypeError):
                return False
        elif filter_def.operator == FilterOperator.LESS_THAN:
            try:
                return float(field_value) < float(filter_value)
            except (ValueError, TypeError):
                return False
        elif filter_def.operator == FilterOperator.IN_LIST:
            return field_value in filter_value
        elif filter_def.operator == FilterOperator.NOT_IN_LIST:
            return field_value not in filter_value

        return False

    def _keyword_matches_paper(self, keyword: str, paper: Dict[str, Any]) -> bool:
        """Vérifie si un mot-clé correspond à un papier"""
        searchable_text = ""

        # Construire le texte de recherche
        for field in ["title", "abstract", "keywords"]:
            value = paper.get(field, "")
            if value:
                if isinstance(value, list):
                    searchable_text += " " + " ".join(str(v) for v in value)
                else:
                    searchable_text += " " + str(value)

        return keyword.lower() in searchable_text.lower()

    def _author_matches_paper(self, author: str, paper: Dict[str, Any]) -> bool:
        """Vérifie si un auteur correspond à un papier"""
        paper_authors = paper.get("authors", [])

        if not isinstance(paper_authors, list):
            return False

        author_lower = author.lower()

        for paper_author in paper_authors:
            if isinstance(paper_author, str) and author_lower in paper_author.lower():
                return True

        return False

    async def _send_alert_notifications(self, alert: PaperAlert, rule: AlertRule):
        """Envoie les notifications pour une alerte"""

        for channel_type in rule.notification_channels:
            try:
                # Trouver le canal configuré
                channel = None
                for ch in self.notification_channels.values():
                    if ch.channel_type == channel_type and ch.is_active:
                        channel = ch
                        break

                if not channel:
                    logger.warning(f"No active channel found for type: {channel_type}")
                    continue

                # Vérifier les filtres du canal
                if alert.priority.value < channel.min_priority.value:
                    continue

                # Vérifier rate limiting
                if not self._check_channel_rate_limit(channel):
                    logger.warning(f"Rate limit exceeded for channel: {channel.name}")
                    continue

                # Envoyer notification
                success = await self._send_notification(channel, alert)

                if success:
                    alert.sent_to_channels.append(channel_type)
                    self.metrics["notifications_sent"] += 1
                else:
                    self.metrics["notification_failures"] += 1

            except Exception as e:
                logger.error(f"Failed to send notification via {channel_type}: {e}")
                self.metrics["notification_failures"] += 1

    async def _send_notification(self, channel: NotificationChannel, alert: PaperAlert) -> bool:
        """Envoie une notification via un canal spécifique"""

        try:
            if channel.channel_type == AlertChannel.EMAIL:
                return await self._send_email_notification(channel, alert)
            elif channel.channel_type == AlertChannel.SLACK:
                return await self._send_slack_notification(channel, alert)
            elif channel.channel_type == AlertChannel.WEBHOOK:
                return await self._send_webhook_notification(channel, alert)
            elif channel.channel_type == AlertChannel.FILE_EXPORT:
                return await self._export_alert_to_file(channel, alert)
            else:
                logger.warning(f"Notification type not implemented: {channel.channel_type}")
                return False

        except Exception as e:
            logger.error(f"Notification failed for channel {channel.name}: {e}")
            return False

    async def _send_email_notification(self, channel: NotificationChannel, alert: PaperAlert) -> bool:
        """Envoie notification par email"""
        config = channel.config

        smtp_server = config.get("smtp_server", "smtp.gmail.com")
        smtp_port = config.get("smtp_port", 587)
        sender_email = config.get("sender_email")
        sender_password = config.get("sender_password")
        recipient_emails = config.get("recipient_emails", [])

        if not sender_email or not sender_password or not recipient_emails:
            logger.error("Email configuration incomplete")
            return False

        # Construire le message
        subject = f"🔬 Research Alert: {alert.paper_data.get('title', 'New Paper')}"

        body = self._format_alert_email(alert)

        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = ", ".join(recipient_emails)
        message["Subject"] = subject

        message.attach(MIMEText(body, "html"))

        try:
            # Envoyer l'email
            context = ssl.create_default_context()
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls(context=context)
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, recipient_emails, message.as_string())

            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    async def _send_slack_notification(self, channel: NotificationChannel, alert: PaperAlert) -> bool:
        """Envoie notification Slack"""
        webhook_url = channel.config.get("webhook_url")

        if not webhook_url:
            logger.error("Slack webhook URL not configured")
            return False

        # Construire le message Slack
        slack_message = {
            "text": f"🔬 New Research Alert",
            "attachments": [
                {
                    "color": self._get_priority_color(alert.priority),
                    "title": alert.paper_data.get("title", "Untitled Paper"),
                    "title_link": alert.paper_data.get("url", ""),
                    "fields": [
                        {
                            "title": "Authors",
                            "value": ", ".join(alert.paper_data.get("authors", [])),
                            "short": True
                        },
                        {
                            "title": "Relevance Score",
                            "value": f"{alert.relevance_score:.2f}",
                            "short": True
                        },
                        {
                            "title": "Matched Keywords",
                            "value": ", ".join(alert.matched_keywords) or "None",
                            "short": False
                        }
                    ],
                    "text": alert.paper_data.get("abstract", "")[:300] + "..." if len(alert.paper_data.get("abstract", "")) > 300 else alert.paper_data.get("abstract", ""),
                    "footer": f"QFrame Alert System | Rule: {alert.rule_id}",
                    "ts": int(alert.triggered_at.timestamp())
                }
            ]
        }

        try:
            response = requests.post(webhook_url, json=slack_message, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False

    async def _send_webhook_notification(self, channel: NotificationChannel, alert: PaperAlert) -> bool:
        """Envoie notification via webhook"""
        webhook_url = channel.config.get("url")
        headers = channel.config.get("headers", {})

        if not webhook_url:
            return False

        payload = {
            "alert_id": alert.alert_id,
            "rule_id": alert.rule_id,
            "priority": alert.priority.name,
            "relevance_score": alert.relevance_score,
            "paper": alert.paper_data,
            "matched_keywords": alert.matched_keywords,
            "triggered_at": alert.triggered_at.isoformat()
        }

        try:
            response = requests.post(webhook_url, json=payload, headers=headers, timeout=10)
            return 200 <= response.status_code < 300
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False

    async def _export_alert_to_file(self, channel: NotificationChannel, alert: PaperAlert) -> bool:
        """Exporte l'alerte vers un fichier"""
        file_path = channel.config.get("file_path", "alerts.jsonl")

        try:
            alert_data = {
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "priority": alert.priority.name,
                "relevance_score": alert.relevance_score,
                "paper": alert.paper_data,
                "matched_keywords": alert.matched_keywords,
                "triggered_at": alert.triggered_at.isoformat()
            }

            # Append to JSONL file
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert_data) + "\n")

            return True

        except Exception as e:
            logger.error(f"Failed to export alert to file: {e}")
            return False

    def _format_alert_email(self, alert: PaperAlert) -> str:
        """Formate l'email d'alerte"""
        paper = alert.paper_data

        html = f"""
        <html>
        <body>
            <h2>🔬 Research Alert</h2>

            <h3>{paper.get('title', 'Untitled Paper')}</h3>

            <p><strong>Authors:</strong> {', '.join(paper.get('authors', []))}</p>

            <p><strong>Relevance Score:</strong> {alert.relevance_score:.2f}</p>

            <p><strong>Matched Keywords:</strong> {', '.join(alert.matched_keywords) if alert.matched_keywords else 'None'}</p>

            <p><strong>Priority:</strong> {alert.priority.name}</p>

            <p><strong>Abstract:</strong></p>
            <p style="background-color: #f5f5f5; padding: 10px; border-left: 3px solid #007cba;">
                {paper.get('abstract', 'No abstract available')}
            </p>

            {f'<p><strong>URL:</strong> <a href="{paper.get("url")}">{paper.get("url")}</a></p>' if paper.get("url") else ''}

            <hr>
            <p style="color: #666; font-size: 12px;">
                Generated by QFrame Alert System | Rule: {alert.rule_id} | {alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S')}
            </p>
        </body>
        </html>
        """

        return html

    def _get_priority_color(self, priority: AlertPriority) -> str:
        """Retourne une couleur pour la priorité"""
        colors = {
            AlertPriority.LOW: "#36a64f",      # Vert
            AlertPriority.MEDIUM: "#ffaa00",    # Orange
            AlertPriority.HIGH: "#ff6b6b",     # Rouge clair
            AlertPriority.CRITICAL: "#cc0000"  # Rouge foncé
        }
        return colors.get(priority, "#36a64f")

    def _check_channel_rate_limit(self, channel: NotificationChannel) -> bool:
        """Vérifie les limites de taux pour un canal"""
        # Implémentation simplifiée - à améliorer avec un vrai rate limiter
        return True

    def _extract_paper_id(self, paper: Dict[str, Any]) -> str:
        """Extrait un ID unique pour un papier"""
        # Essayer différents champs d'ID
        for id_field in ["id", "paperId", "doi", "arxiv_id"]:
            if paper.get(id_field):
                return str(paper[id_field])

        # Fallback: hash du titre et premier auteur
        title = paper.get("title", "")
        authors = paper.get("authors", [])
        first_author = authors[0] if authors else ""

        return str(hash(f"{title}|{first_author}"))

    def get_alert_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'alerte"""

        # Statistiques des dernières 24h
        last_24h = datetime.utcnow() - timedelta(days=1)
        recent_alerts = [a for a in self.alert_history if a.triggered_at > last_24h]

        # Répartition par priorité des alertes récentes
        recent_by_priority = {}
        for priority in AlertPriority:
            recent_by_priority[priority.name] = len([a for a in recent_alerts if a.priority == priority])

        return {
            "global": self.metrics.copy(),
            "recent_24h": {
                "total_alerts": len(recent_alerts),
                "by_priority": recent_by_priority,
                "avg_relevance_score": np.mean([a.relevance_score for a in recent_alerts]) if recent_alerts else 0
            },
            "rules": {
                "total_active": len([r for r in self.rules.values() if r.is_active]),
                "total_inactive": len([r for r in self.rules.values() if not r.is_active]),
                "by_trigger": {trigger.value: len([r for r in self.rules.values() if r.trigger == trigger]) for trigger in AlertTrigger}
            },
            "channels": {
                "total_configured": len(self.notification_channels),
                "total_active": len([c for c in self.notification_channels.values() if c.is_active])
            }
        }

    def get_top_papers(self, limit: int = 10, days: int = 7) -> List[PaperAlert]:
        """Retourne les papiers les mieux notés récemment"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_alerts = [a for a in self.alert_history if a.triggered_at > cutoff_date]

        # Trier par score de pertinence
        sorted_alerts = sorted(recent_alerts, key=lambda a: a.relevance_score, reverse=True)

        return sorted_alerts[:limit]

    async def setup_default_rules(self):
        """Configure des règles par défaut"""

        # Règle pour Machine Learning en finance
        ml_finance_rule = AlertRule(
            rule_id="ml_finance",
            name="Machine Learning in Finance",
            description="Papers about ML applications in finance",
            trigger=AlertTrigger.KEYWORD_MATCH,
            priority=AlertPriority.HIGH,
            keywords=["machine learning", "deep learning", "neural network", "artificial intelligence", "quantitative finance", "algorithmic trading"],
            filters=[
                AlertFilter(field="title", operator=FilterOperator.CONTAINS, value="finance", weight=1.0),
                AlertFilter(field="abstract", operator=FilterOperator.CONTAINS, value="trading", weight=0.5)
            ],
            min_relevance_score=0.3,
            notification_channels=[AlertChannel.EMAIL, AlertChannel.FILE_EXPORT]
        )
        self.add_rule(ml_finance_rule)

        # Règle pour crypto et blockchain
        crypto_rule = AlertRule(
            rule_id="crypto_blockchain",
            name="Cryptocurrency and Blockchain",
            description="Papers about cryptocurrency and blockchain technology",
            trigger=AlertTrigger.KEYWORD_MATCH,
            priority=AlertPriority.MEDIUM,
            keywords=["cryptocurrency", "blockchain", "bitcoin", "ethereum", "defi", "smart contract"],
            min_relevance_score=0.4,
            notification_channels=[AlertChannel.FILE_EXPORT]
        )
        self.add_rule(crypto_rule)

        # Règle pour risk management
        risk_rule = AlertRule(
            rule_id="risk_management",
            name="Risk Management",
            description="Papers about financial risk management",
            trigger=AlertTrigger.KEYWORD_MATCH,
            priority=AlertPriority.HIGH,
            keywords=["risk management", "value at risk", "var", "stress testing", "portfolio optimization"],
            min_relevance_score=0.5,
            notification_channels=[AlertChannel.EMAIL]
        )
        self.add_rule(risk_rule)

    async def setup_default_channels(self):
        """Configure les canaux par défaut"""

        # Canal email par défaut
        email_channel = NotificationChannel(
            channel_type=AlertChannel.EMAIL,
            name="default_email",
            config={
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender_email": "alerts@qframe.com",  # À configurer
                "sender_password": "",  # À configurer
                "recipient_emails": ["researcher@qframe.com"]  # À configurer
            },
            min_priority=AlertPriority.MEDIUM
        )
        self.add_notification_channel(email_channel)

        # Canal fichier par défaut
        file_channel = NotificationChannel(
            channel_type=AlertChannel.FILE_EXPORT,
            name="default_file",
            config={
                "file_path": "research_alerts.jsonl"
            },
            min_priority=AlertPriority.LOW
        )
        self.add_notification_channel(file_channel)