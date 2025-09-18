#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Système de Monitoring et Alertes Quantitatif
Surveillance temps réel des performances et détection d'anomalies
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
from datetime import datetime, timedelta
import json
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from dataclasses import dataclass, field
from enum import Enum
import warnings

import requests
from mlpipeline.utils.risk_metrics import comprehensive_metrics

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class AlertSeverity(Enum):
    """Niveaux de sévérité des alertes"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertChannel(Enum):
    """Canaux de notification"""
    EMAIL = "email"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    SLACK = "slack"
    LOG_ONLY = "log"

@dataclass
class Alert:
    """Structure d'une alerte"""
    
    timestamp: datetime
    title: str
    message: str
    severity: AlertSeverity
    category: str
    metrics: Dict = field(default_factory=dict)
    alert_id: str = field(default_factory=lambda: f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    def to_dict(self) -> Dict:
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'title': self.title,
            'message': self.message,
            'severity': self.severity.value,
            'category': self.category,
            'metrics': self.metrics
        }

@dataclass
class MonitoringConfig:
    """Configuration du monitoring"""
    
    # Alertes performances
    max_drawdown_threshold: float = 0.15  # 15%
    min_sharpe_threshold: float = 0.5
    max_vol_threshold: float = 0.4  # 40% annualisé
    min_return_threshold: float = -0.05  # -5% journalier
    
    # Alertes techniques
    max_correlation_threshold: float = 0.8
    min_diversification_threshold: float = 0.3
    max_position_concentration: float = 0.4
    
    # Alertes données
    max_missing_data_ratio: float = 0.05  # 5%
    max_data_staleness_hours: int = 2
    min_data_quality_score: float = 0.8
    
    # Notifications
    notification_channels: List[AlertChannel] = field(default_factory=lambda: [AlertChannel.LOG_ONLY])
    email_config: Dict = field(default_factory=dict)
    discord_webhook: Optional[str] = None
    telegram_config: Dict = field(default_factory=dict)
    
    # Fréquences monitoring
    check_frequency_minutes: int = 15
    daily_report_time: str = "08:00"
    weekly_report_day: int = 1  # Lundi

class QuantMonitor:
    """
    Système de monitoring quantitatif avancé
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        
        # État du monitoring
        self.last_check_time = None
        self.alert_history: List[Alert] = []
        self.active_alerts: Dict[str, Alert] = {}
        
        # Métriques courantes
        self.current_metrics = {}
        self.performance_buffer = []
        self.data_quality_buffer = []
        
        # Seuils adaptatifs
        self.adaptive_thresholds = {}
        
        logger.info("📊 QuantMonitor initialisé")
    
    def check_system_health(self, 
                          portfolio_data: pd.DataFrame,
                          alpha_performance: Dict[str, Dict],
                          market_data: pd.DataFrame) -> List[Alert]:
        """
        Vérification complète de la santé du système
        """
        
        alerts = []
        self.last_check_time = datetime.now()
        
        logger.debug("🔍 Vérification santé système...")
        
        try:
            # 1. Performance Portfolio
            perf_alerts = self._check_portfolio_performance(portfolio_data)
            alerts.extend(perf_alerts)
            
            # 2. Performance Alphas
            alpha_alerts = self._check_alpha_performance(alpha_performance)
            alerts.extend(alpha_alerts)
            
            # 3. Qualité données
            data_alerts = self._check_data_quality(market_data)
            alerts.extend(data_alerts)
            
            # 4. Corrélations et diversification
            div_alerts = self._check_diversification(alpha_performance)
            alerts.extend(div_alerts)
            
            # 5. Détection anomalies
            anomaly_alerts = self._detect_anomalies(portfolio_data, alpha_performance)
            alerts.extend(anomaly_alerts)
            
            # 6. Mise à jour seuils adaptatifs
            self._update_adaptive_thresholds(portfolio_data, alpha_performance)
            
            # 7. Notification des nouvelles alertes
            self._process_alerts(alerts)
            
            logger.info(f"📊 Check terminé: {len(alerts)} alertes générées")
            
        except Exception as e:
            error_alert = Alert(
                timestamp=datetime.now(),
                title="Erreur Monitoring",
                message=f"Erreur lors du check système: {e}",
                severity=AlertSeverity.CRITICAL,
                category="system"
            )
            alerts.append(error_alert)
            logger.error(f"❌ Erreur monitoring: {e}")
        
        return alerts
    
    def _check_portfolio_performance(self, portfolio_data: pd.DataFrame) -> List[Alert]:
        """Vérification performance du portfolio"""
        
        alerts = []
        
        if portfolio_data.empty or len(portfolio_data) < 2:
            return alerts
        
        try:
            # Calcul métriques récentes
            returns = portfolio_data['value'].pct_change().dropna()
            
            if len(returns) == 0:
                return alerts
            
            # Métriques actuelles
            current_metrics = comprehensive_metrics(returns.values[-252:])  # 1 an
            
            # 1. Drawdown excessif
            max_dd = current_metrics.get('max_drawdown', 0)
            if max_dd > self.config.max_drawdown_threshold:
                alerts.append(Alert(
                    timestamp=datetime.now(),
                    title="Drawdown Critique",
                    message=f"Max drawdown: {max_dd:.2%} > seuil {self.config.max_drawdown_threshold:.2%}",
                    severity=AlertSeverity.CRITICAL,
                    category="performance",
                    metrics={'max_drawdown': max_dd, 'threshold': self.config.max_drawdown_threshold}
                ))
            
            # 2. Sharpe ratio faible
            sharpe = current_metrics.get('sharpe_ratio', 0)
            if sharpe < self.config.min_sharpe_threshold:
                alerts.append(Alert(
                    timestamp=datetime.now(),
                    title="Sharpe Ratio Faible",
                    message=f"Sharpe ratio: {sharpe:.3f} < seuil {self.config.min_sharpe_threshold:.3f}",
                    severity=AlertSeverity.WARNING,
                    category="performance",
                    metrics={'sharpe_ratio': sharpe, 'threshold': self.config.min_sharpe_threshold}
                ))
            
            # 3. Volatilité excessive
            volatility = returns.std() * np.sqrt(252)  # Annualisée
            if volatility > self.config.max_vol_threshold:
                alerts.append(Alert(
                    timestamp=datetime.now(),
                    title="Volatilité Excessive",
                    message=f"Volatilité: {volatility:.2%} > seuil {self.config.max_vol_threshold:.2%}",
                    severity=AlertSeverity.WARNING,
                    category="risk",
                    metrics={'volatility': volatility, 'threshold': self.config.max_vol_threshold}
                ))
            
            # 4. Return journalier critique
            if len(returns) > 0:
                last_return = returns.iloc[-1]
                if last_return < self.config.min_return_threshold:
                    alerts.append(Alert(
                        timestamp=datetime.now(),
                        title="Return Journalier Critique",
                        message=f"Return: {last_return:.2%} < seuil {self.config.min_return_threshold:.2%}",
                        severity=AlertSeverity.CRITICAL,
                        category="performance",
                        metrics={'daily_return': last_return, 'threshold': self.config.min_return_threshold}
                    ))
            
        except Exception as e:
            logger.error(f"❌ Erreur check performance: {e}")
        
        return alerts
    
    def _check_alpha_performance(self, alpha_performance: Dict[str, Dict]) -> List[Alert]:
        """Vérification performance individuelle des alphas"""
        
        alerts = []
        
        try:
            for alpha_name, metrics in alpha_performance.items():
                
                # 1. Performance dégradée
                sharpe = metrics.get('sharpe_ratio', 0)
                if sharpe < 0:
                    alerts.append(Alert(
                        timestamp=datetime.now(),
                        title=f"Alpha {alpha_name} Défaillant",
                        message=f"{alpha_name}: Sharpe ratio négatif ({sharpe:.3f})",
                        severity=AlertSeverity.WARNING,
                        category="alpha_performance",
                        metrics={'alpha': alpha_name, 'sharpe': sharpe}
                    ))
                
                # 2. Corrélation trop forte avec marché
                market_corr = metrics.get('market_correlation', 0)
                if abs(market_corr) > 0.9:
                    alerts.append(Alert(
                        timestamp=datetime.now(),
                        title=f"Alpha {alpha_name} Trop Corrélé",
                        message=f"{alpha_name}: Corrélation marché {market_corr:.3f}",
                        severity=AlertSeverity.WARNING,
                        category="diversification",
                        metrics={'alpha': alpha_name, 'correlation': market_corr}
                    ))
                
                # 3. Absence de signaux
                signals_count = metrics.get('signals_generated', 0)
                if signals_count == 0:
                    alerts.append(Alert(
                        timestamp=datetime.now(),
                        title=f"Alpha {alpha_name} Inactif",
                        message=f"{alpha_name}: Aucun signal généré récemment",
                        severity=AlertSeverity.WARNING,
                        category="alpha_activity",
                        metrics={'alpha': alpha_name, 'signals_count': signals_count}
                    ))
        
        except Exception as e:
            logger.error(f"❌ Erreur check alphas: {e}")
        
        return alerts
    
    def _check_data_quality(self, market_data: pd.DataFrame) -> List[Alert]:
        """Vérification qualité des données"""
        
        alerts = []
        
        try:
            if market_data.empty:
                alerts.append(Alert(
                    timestamp=datetime.now(),
                    title="Données Manquantes",
                    message="Aucune donnée de marché disponible",
                    severity=AlertSeverity.EMERGENCY,
                    category="data_quality"
                ))
                return alerts
            
            # 1. Données manquantes
            missing_ratio = market_data.isnull().sum().sum() / (len(market_data) * len(market_data.columns))
            if missing_ratio > self.config.max_missing_data_ratio:
                alerts.append(Alert(
                    timestamp=datetime.now(),
                    title="Données Incomplètes",
                    message=f"Ratio données manquantes: {missing_ratio:.2%}",
                    severity=AlertSeverity.WARNING,
                    category="data_quality",
                    metrics={'missing_ratio': missing_ratio}
                ))
            
            # 2. Fraîcheur des données
            if hasattr(market_data.index, 'max'):
                last_data_time = market_data.index.max()
                if isinstance(last_data_time, pd.Timestamp):
                    staleness_hours = (datetime.now(last_data_time.tz) - last_data_time).total_seconds() / 3600
                    
                    if staleness_hours > self.config.max_data_staleness_hours:
                        alerts.append(Alert(
                            timestamp=datetime.now(),
                            title="Données Périmées",
                            message=f"Dernières données: {staleness_hours:.1f}h",
                            severity=AlertSeverity.CRITICAL,
                            category="data_quality",
                            metrics={'staleness_hours': staleness_hours}
                        ))
            
            # 3. Prix suspects (variations extrêmes)
            if 'close' in market_data.columns:
                price_changes = market_data['close'].pct_change().dropna()
                extreme_changes = price_changes[abs(price_changes) > 0.2]  # >20%
                
                if len(extreme_changes) > 0:
                    alerts.append(Alert(
                        timestamp=datetime.now(),
                        title="Prix Suspects Détectés",
                        message=f"{len(extreme_changes)} variations >20% détectées",
                        severity=AlertSeverity.WARNING,
                        category="data_quality",
                        metrics={'extreme_changes_count': len(extreme_changes)}
                    ))
        
        except Exception as e:
            logger.error(f"❌ Erreur check données: {e}")
        
        return alerts
    
    def _check_diversification(self, alpha_performance: Dict[str, Dict]) -> List[Alert]:
        """Vérification diversification du portfolio"""
        
        alerts = []
        
        try:
            if len(alpha_performance) < 2:
                return alerts
            
            # Calcul matrice corrélations
            correlations = []
            alpha_names = list(alpha_performance.keys())
            
            for i, alpha1 in enumerate(alpha_names):
                for j, alpha2 in enumerate(alpha_names[i+1:], i+1):
                    corr = alpha_performance[alpha1].get('correlation_with', {}).get(alpha2, 0)
                    correlations.append(abs(corr))
            
            # Corrélation moyenne
            if correlations:
                avg_correlation = np.mean(correlations)
                max_correlation = np.max(correlations)
                
                # Alerte corrélation excessive
                if max_correlation > self.config.max_correlation_threshold:
                    alerts.append(Alert(
                        timestamp=datetime.now(),
                        title="Corrélation Excessive Entre Alphas",
                        message=f"Max corrélation: {max_correlation:.3f}",
                        severity=AlertSeverity.WARNING,
                        category="diversification",
                        metrics={'max_correlation': max_correlation, 'avg_correlation': avg_correlation}
                    ))
        
        except Exception as e:
            logger.error(f"❌ Erreur check diversification: {e}")
        
        return alerts
    
    def _detect_anomalies(self, portfolio_data: pd.DataFrame, alpha_performance: Dict) -> List[Alert]:
        """Détection d'anomalies par ML/statistiques"""
        
        alerts = []
        
        try:
            if portfolio_data.empty or len(portfolio_data) < 50:
                return alerts
            
            returns = portfolio_data['value'].pct_change().dropna()
            
            # 1. Détection outliers (Z-score)
            z_scores = np.abs((returns - returns.mean()) / returns.std())
            outliers = z_scores > 3
            
            if outliers.sum() > len(returns) * 0.05:  # >5% outliers
                alerts.append(Alert(
                    timestamp=datetime.now(),
                    title="Outliers Détectés",
                    message=f"{outliers.sum()} returns outliers ({outliers.sum()/len(returns):.1%})",
                    severity=AlertSeverity.WARNING,
                    category="anomaly",
                    metrics={'outliers_count': int(outliers.sum()), 'outliers_ratio': float(outliers.sum()/len(returns))}
                ))
            
            # 2. Changement de régime (variance)
            if len(returns) > 100:
                recent_vol = returns.iloc[-30:].std()
                historical_vol = returns.iloc[-100:-30].std()
                vol_ratio = recent_vol / historical_vol
                
                if vol_ratio > 2.0:  # Volatilité doublée
                    alerts.append(Alert(
                        timestamp=datetime.now(),
                        title="Changement Régime Volatilité",
                        message=f"Volatilité récente x{vol_ratio:.1f}",
                        severity=AlertSeverity.WARNING,
                        category="regime_change",
                        metrics={'volatility_ratio': vol_ratio}
                    ))
        
        except Exception as e:
            logger.error(f"❌ Erreur détection anomalies: {e}")
        
        return alerts
    
    def _update_adaptive_thresholds(self, portfolio_data: pd.DataFrame, alpha_performance: Dict):
        """Mise à jour seuils adaptatifs"""
        
        try:
            if portfolio_data.empty:
                return
            
            returns = portfolio_data['value'].pct_change().dropna()
            
            if len(returns) > 100:
                # Seuil drawdown adaptatif (95e percentile historique)
                historical_dds = []
                for i in range(50, len(returns)):
                    period_returns = returns.iloc[i-50:i]
                    period_dd = (period_returns.cumsum().cummax() - period_returns.cumsum()).max()
                    historical_dds.append(period_dd)
                
                if historical_dds:
                    adaptive_dd_threshold = np.percentile(historical_dds, 95)
                    self.adaptive_thresholds['max_drawdown'] = adaptive_dd_threshold
                    
                    logger.debug(f"Seuil DD adaptatif: {adaptive_dd_threshold:.2%}")
        
        except Exception as e:
            logger.error(f"❌ Erreur seuils adaptatifs: {e}")
    
    def _process_alerts(self, alerts: List[Alert]):
        """Traitement et notification des alertes"""
        
        new_alerts = []
        
        for alert in alerts:
            # Éviter doublons récents
            if not self._is_duplicate_alert(alert):
                new_alerts.append(alert)
                self.alert_history.append(alert)
                self.active_alerts[alert.category] = alert
        
        # Notification
        if new_alerts:
            self._send_notifications(new_alerts)
            logger.info(f"📢 {len(new_alerts)} nouvelles alertes envoyées")
    
    def _is_duplicate_alert(self, alert: Alert) -> bool:
        """Vérification doublons d'alertes"""
        
        # Chercher alertes similaires dans les 30 dernières minutes
        cutoff_time = datetime.now() - timedelta(minutes=30)
        
        for existing_alert in reversed(self.alert_history[-50:]):  # Check 50 dernières
            if (existing_alert.timestamp > cutoff_time and
                existing_alert.category == alert.category and
                existing_alert.title == alert.title):
                return True
        
        return False
    
    def _send_notifications(self, alerts: List[Alert]):
        """Envoi notifications multi-canal"""
        
        for channel in self.config.notification_channels:
            try:
                if channel == AlertChannel.EMAIL:
                    self._send_email_alerts(alerts)
                elif channel == AlertChannel.DISCORD:
                    self._send_discord_alerts(alerts)
                elif channel == AlertChannel.TELEGRAM:
                    self._send_telegram_alerts(alerts)
                elif channel == AlertChannel.LOG_ONLY:
                    self._log_alerts(alerts)
                    
            except Exception as e:
                logger.error(f"❌ Erreur notification {channel.value}: {e}")
    
    def _send_email_alerts(self, alerts: List[Alert]):
        """Envoi alertes par email"""
        
        if not self.config.email_config:
            return
        
        try:
            smtp_server = self.config.email_config.get('smtp_server')
            smtp_port = self.config.email_config.get('smtp_port', 587)
            username = self.config.email_config.get('username')
            password = self.config.email_config.get('password')
            to_addresses = self.config.email_config.get('to_addresses', [])
            
            if not all([smtp_server, username, password, to_addresses]):
                return
            
            # Composition email
            subject = f"[QuantBot] {len(alerts)} Alertes - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            body_html = self._format_alerts_email(alerts)
            
            msg = MimeMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = username
            msg['To'] = ', '.join(to_addresses)
            
            html_part = MimeText(body_html, 'html')
            msg.attach(html_part)
            
            # Envoi
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
            
            logger.debug("📧 Email alertes envoyé")
            
        except Exception as e:
            logger.error(f"❌ Erreur email: {e}")
    
    def _send_discord_alerts(self, alerts: List[Alert]):
        """Envoi alertes Discord via webhook"""
        
        if not self.config.discord_webhook:
            return
        
        try:
            for alert in alerts:
                embed = {
                    "title": alert.title,
                    "description": alert.message,
                    "color": self._get_color_for_severity(alert.severity),
                    "timestamp": alert.timestamp.isoformat(),
                    "fields": [
                        {"name": "Catégorie", "value": alert.category, "inline": True},
                        {"name": "Sévérité", "value": alert.severity.value.upper(), "inline": True}
                    ]
                }
                
                if alert.metrics:
                    metrics_str = "\\n".join([f"{k}: {v}" for k, v in alert.metrics.items()])
                    embed["fields"].append({"name": "Métriques", "value": f"```{metrics_str}```"})
                
                payload = {"embeds": [embed]}
                
                response = requests.post(self.config.discord_webhook, json=payload)
                response.raise_for_status()
            
            logger.debug("💬 Discord alertes envoyées")
            
        except Exception as e:
            logger.error(f"❌ Erreur Discord: {e}")
    
    def _send_telegram_alerts(self, alerts: List[Alert]):
        """Envoi alertes Telegram"""
        
        if not self.config.telegram_config:
            return
        
        try:
            bot_token = self.config.telegram_config.get('bot_token')
            chat_id = self.config.telegram_config.get('chat_id')
            
            if not bot_token or not chat_id:
                return
            
            for alert in alerts:
                message = f"🤖 *{alert.title}*\\n\\n{alert.message}\\n\\n_Sévérité: {alert.severity.value}_"
                
                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                payload = {
                    'chat_id': chat_id,
                    'text': message,
                    'parse_mode': 'Markdown'
                }
                
                response = requests.post(url, json=payload)
                response.raise_for_status()
            
            logger.debug("📱 Telegram alertes envoyées")
            
        except Exception as e:
            logger.error(f"❌ Erreur Telegram: {e}")
    
    def _log_alerts(self, alerts: List[Alert]):
        """Log des alertes"""
        
        for alert in alerts:
            level = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.CRITICAL: logging.ERROR,
                AlertSeverity.EMERGENCY: logging.CRITICAL
            }.get(alert.severity, logging.INFO)
            
            logger.log(level, f"🚨 {alert.title}: {alert.message}")
    
    def _format_alerts_email(self, alerts: List[Alert]) -> str:
        """Formatage HTML pour email"""
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .alert {{ border-left: 4px solid; padding: 10px; margin: 10px 0; }}
                .critical {{ border-color: #dc3545; background: #f8d7da; }}
                .warning {{ border-color: #ffc107; background: #fff3cd; }}
                .info {{ border-color: #17a2b8; background: #d1ecf1; }}
                .metrics {{ background: #f8f9fa; padding: 8px; font-family: monospace; }}
            </style>
        </head>
        <body>
            <h2>Alertes Système Quantitatif - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h2>
        """
        
        for alert in alerts:
            css_class = alert.severity.value
            html += f"""
            <div class="alert {css_class}">
                <h3>{alert.title}</h3>
                <p>{alert.message}</p>
                <small>Catégorie: {alert.category} | Sévérité: {alert.severity.value.upper()}</small>
                """
            
            if alert.metrics:
                html += '<div class="metrics">'
                for key, value in alert.metrics.items():
                    html += f"{key}: {value}<br>"
                html += '</div>'
            
            html += '</div>'
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _get_color_for_severity(self, severity: AlertSeverity) -> int:
        """Couleur Discord par sévérité"""
        
        colors = {
            AlertSeverity.INFO: 0x17a2b8,      # Bleu
            AlertSeverity.WARNING: 0xffc107,   # Jaune
            AlertSeverity.CRITICAL: 0xdc3545,  # Rouge
            AlertSeverity.EMERGENCY: 0x6f42c1  # Violet
        }
        
        return colors.get(severity, 0x17a2b8)
    
    def generate_daily_report(self, portfolio_data: pd.DataFrame, 
                            alpha_performance: Dict) -> Dict:
        """Génération rapport journalier"""
        
        report = {
            'date': datetime.now().date(),
            'portfolio_summary': {},
            'alpha_summary': {},
            'alerts_summary': {},
            'recommendations': []
        }
        
        try:
            # Portfolio summary
            if not portfolio_data.empty:
                returns = portfolio_data['value'].pct_change().dropna()
                if len(returns) > 0:
                    report['portfolio_summary'] = {
                        'total_return_1d': returns.iloc[-1] if len(returns) >= 1 else 0,
                        'total_return_7d': returns.iloc[-7:].sum() if len(returns) >= 7 else 0,
                        'volatility_7d': returns.iloc[-7:].std() * np.sqrt(7) if len(returns) >= 7 else 0,
                        'sharpe_30d': (returns.iloc[-30:].mean() / returns.iloc[-30:].std()) * np.sqrt(30) if len(returns) >= 30 else 0
                    }
            
            # Alerts summary
            recent_alerts = [a for a in self.alert_history if a.timestamp.date() == datetime.now().date()]
            report['alerts_summary'] = {
                'total_alerts': len(recent_alerts),
                'critical_alerts': len([a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL]),
                'warning_alerts': len([a for a in recent_alerts if a.severity == AlertSeverity.WARNING]),
                'categories': list(set([a.category for a in recent_alerts]))
            }
            
            # Recommandations
            if report['alerts_summary']['critical_alerts'] > 0:
                report['recommendations'].append("Vérifier alertes critiques et ajuster positions si nécessaire")
            
            if report['portfolio_summary'].get('volatility_7d', 0) > 0.3:
                report['recommendations'].append("Volatilité élevée - considérer réduction exposition")
            
        except Exception as e:
            logger.error(f"❌ Erreur génération rapport: {e}")
        
        return report
    
    def get_alert_history(self, days: int = 7) -> pd.DataFrame:
        """Historique des alertes"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_alerts = [a for a in self.alert_history if a.timestamp > cutoff_date]
        
        if not recent_alerts:
            return pd.DataFrame()
        
        data = []
        for alert in recent_alerts:
            data.append({
                'timestamp': alert.timestamp,
                'title': alert.title,
                'message': alert.message,
                'severity': alert.severity.value,
                'category': alert.category,
                'alert_id': alert.alert_id
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df


# ==============================================
# FONCTIONS UTILITAIRES
# ==============================================

def create_monitoring_config(config_dict: Dict) -> MonitoringConfig:
    """Création configuration depuis dictionnaire"""
    
    config = MonitoringConfig()
    
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def run_monitoring_check(portfolio_data: pd.DataFrame,
                        alpha_performance: Dict,
                        market_data: pd.DataFrame,
                        config: Optional[MonitoringConfig] = None) -> List[Alert]:
    """
    Interface simplifiée pour check monitoring
    """
    
    if config is None:
        config = MonitoringConfig()
    
    monitor = QuantMonitor(config)
    return monitor.check_system_health(portfolio_data, alpha_performance, market_data)