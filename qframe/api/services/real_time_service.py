"""
⚡ Real Time Service
Service pour les données et événements temps réel
"""

import asyncio
import logging
from typing import Dict, List, Set, Any, Optional, Callable
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, field
from enum import Enum

from qframe.core.container import injectable
from qframe.api.services.base_service import BaseService

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types d'événements temps réel."""
    PRICE_UPDATE = "price_update"
    ORDER_UPDATE = "order_update"
    POSITION_UPDATE = "position_update"
    TRADE_EXECUTED = "trade_executed"
    RISK_ALERT = "risk_alert"
    STRATEGY_SIGNAL = "strategy_signal"
    SYSTEM_ALERT = "system_alert"


@dataclass
class RealTimeEvent:
    """Événement temps réel."""
    type: EventType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: Optional[str] = None
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'événement en dictionnaire."""
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "source": self.source
        }

    def to_json(self) -> str:
        """Convertit l'événement en JSON."""
        return json.dumps(self.to_dict())


@dataclass
class Subscription:
    """Abonnement à des événements temps réel."""
    client_id: str
    event_types: Set[EventType]
    symbols: Optional[Set[str]] = None
    callback: Optional[Callable] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    def matches_event(self, event: RealTimeEvent) -> bool:
        """Vérifie si l'abonnement correspond à l'événement."""
        # Vérifier le type d'événement
        if event.type not in self.event_types:
            return False

        # Vérifier le symbole si spécifié
        if self.symbols and event.symbol and event.symbol not in self.symbols:
            return False

        return True


@injectable
class RealTimeService(BaseService):
    """Service de gestion des données et événements temps réel."""

    def __init__(self):
        super().__init__()

        # Gestion des abonnements
        self._subscriptions: Dict[str, Subscription] = {}

        # Queue des événements
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)

        # Statistiques
        self._stats = {
            "events_processed": 0,
            "events_dropped": 0,
            "active_subscriptions": 0,
            "total_subscribers": 0
        }

        # Tâches en arrière-plan
        self._event_processor_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Configuration
        self.max_events_per_second = 1000
        self.cleanup_interval = 300  # 5 minutes
        self.subscription_timeout = 3600  # 1 heure

    async def start(self):
        """Démarre le service temps réel."""
        logger.info("Starting Real Time Service...")

        self._start_time = datetime.now()

        # Démarrer les tâches en arrière-plan
        self._event_processor_task = asyncio.create_task(self._process_events())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        self._is_running = True
        logger.info("Real Time Service started successfully")

    async def stop(self):
        """Arrête le service temps réel."""
        logger.info("Stopping Real Time Service...")

        self._is_running = False

        # Arrêter les tâches
        if self._event_processor_task:
            self._event_processor_task.cancel()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        # Attendre l'arrêt des tâches
        tasks = [self._event_processor_task, self._heartbeat_task]
        for task in tasks:
            if task:
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Nettoyer les abonnements
        self._subscriptions.clear()

        logger.info("Real Time Service stopped")

    async def subscribe(
        self,
        client_id: str,
        event_types: List[EventType],
        symbols: Optional[List[str]] = None,
        callback: Optional[Callable] = None
    ) -> bool:
        """Crée un abonnement aux événements temps réel."""
        try:
            subscription = Subscription(
                client_id=client_id,
                event_types=set(event_types),
                symbols=set(symbols) if symbols else None,
                callback=callback
            )

            self._subscriptions[client_id] = subscription
            self._stats["active_subscriptions"] = len(self._subscriptions)
            self._stats["total_subscribers"] += 1

            logger.info(f"Client {client_id} subscribed to {len(event_types)} event types")
            return True

        except Exception as e:
            logger.error(f"Error subscribing client {client_id}: {e}")
            return False

    async def unsubscribe(self, client_id: str) -> bool:
        """Supprime un abonnement."""
        try:
            if client_id in self._subscriptions:
                del self._subscriptions[client_id]
                self._stats["active_subscriptions"] = len(self._subscriptions)
                logger.info(f"Client {client_id} unsubscribed")
                return True
            return False

        except Exception as e:
            logger.error(f"Error unsubscribing client {client_id}: {e}")
            return False

    async def publish_event(self, event: RealTimeEvent) -> bool:
        """Publie un événement dans le système."""
        try:
            # Vérifier la limite d'événements
            if self._event_queue.qsize() >= self._event_queue.maxsize:
                self._stats["events_dropped"] += 1
                logger.warning("Event queue full, dropping event")
                return False

            # Ajouter l'événement à la queue
            await self._event_queue.put(event)
            return True

        except Exception as e:
            logger.error(f"Error publishing event: {e}")
            self._stats["events_dropped"] += 1
            return False

    async def publish_price_update(self, symbol: str, price_data: Dict[str, Any]):
        """Publie une mise à jour de prix."""
        event = RealTimeEvent(
            type=EventType.PRICE_UPDATE,
            data=price_data,
            symbol=symbol,
            source="market_data"
        )
        await self.publish_event(event)

    async def publish_order_update(self, order_data: Dict[str, Any]):
        """Publie une mise à jour d'ordre."""
        event = RealTimeEvent(
            type=EventType.ORDER_UPDATE,
            data=order_data,
            symbol=order_data.get("symbol"),
            source="order_management"
        )
        await self.publish_event(event)

    async def publish_position_update(self, position_data: Dict[str, Any]):
        """Publie une mise à jour de position."""
        event = RealTimeEvent(
            type=EventType.POSITION_UPDATE,
            data=position_data,
            symbol=position_data.get("symbol"),
            source="portfolio_management"
        )
        await self.publish_event(event)

    async def publish_risk_alert(self, alert_data: Dict[str, Any]):
        """Publie une alerte de risque."""
        event = RealTimeEvent(
            type=EventType.RISK_ALERT,
            data=alert_data,
            source="risk_management"
        )
        await self.publish_event(event)

    async def publish_trade_execution(self, trade_data: Dict[str, Any]):
        """Publie l'exécution d'un trade."""
        event = RealTimeEvent(
            type=EventType.TRADE_EXECUTED,
            data=trade_data,
            symbol=trade_data.get("symbol"),
            source="execution_engine"
        )
        await self.publish_event(event)

    async def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques du service."""
        return {
            **self._stats,
            "queue_size": self._event_queue.qsize(),
            "queue_capacity": self._event_queue.maxsize,
            "uptime": self.get_uptime(),
            "events_per_minute": self._calculate_events_per_minute(),
            "subscription_details": [
                {
                    "client_id": sub.client_id,
                    "event_types": [et.value for et in sub.event_types],
                    "symbols": list(sub.symbols) if sub.symbols else None,
                    "created_at": sub.created_at.isoformat(),
                    "last_activity": sub.last_activity.isoformat()
                }
                for sub in self._subscriptions.values()
            ]
        }

    async def get_recent_events(
        self,
        event_types: Optional[List[EventType]] = None,
        symbols: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Récupère les événements récents (simulation)."""
        # En production, ceci viendrait d'un store persistant
        events = []

        # Simuler quelques événements récents
        for i in range(min(limit, 20)):
            event_type = EventType.PRICE_UPDATE
            if event_types and event_type not in event_types:
                continue

            symbol = "BTC/USDT"
            if symbols and symbol not in symbols:
                continue

            events.append({
                "type": event_type.value,
                "data": {
                    "symbol": symbol,
                    "price": 43250.0 + i * 10,
                    "change": 0.5
                },
                "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
                "symbol": symbol,
                "source": "simulation"
            })

        return events

    async def _process_events(self):
        """Traite les événements de la queue."""
        while self._is_running:
            try:
                # Récupérer l'événement avec timeout
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)

                # Traiter l'événement
                await self._handle_event(event)

                self._stats["events_processed"] += 1

            except asyncio.TimeoutError:
                # Pas d'événement, continuer
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing event: {e}")

    async def _handle_event(self, event: RealTimeEvent):
        """Traite un événement spécifique."""
        # Envoyer l'événement à tous les abonnés intéressés
        matching_subscriptions = [
            sub for sub in self._subscriptions.values()
            if sub.matches_event(event)
        ]

        for subscription in matching_subscriptions:
            try:
                # Mettre à jour l'activité
                subscription.last_activity = datetime.now()

                # Appeler le callback si défini
                if subscription.callback:
                    await self._safe_callback(subscription.callback, event)

            except Exception as e:
                logger.error(f"Error handling event for client {subscription.client_id}: {e}")

    async def _safe_callback(self, callback: Callable, event: RealTimeEvent):
        """Appelle un callback de manière sécurisée."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)
        except Exception as e:
            logger.error(f"Error in event callback: {e}")

    async def _heartbeat_loop(self):
        """Boucle de heartbeat et nettoyage."""
        while self._is_running:
            try:
                # Nettoyer les abonnements expirés
                await self._cleanup_expired_subscriptions()

                # Publier un heartbeat
                await self._publish_heartbeat()

                # Attendre le prochain cycle
                await asyncio.sleep(self.cleanup_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")

    async def _cleanup_expired_subscriptions(self):
        """Nettoie les abonnements expirés."""
        now = datetime.now()
        expired_clients = []

        for client_id, subscription in self._subscriptions.items():
            time_since_activity = (now - subscription.last_activity).total_seconds()
            if time_since_activity > self.subscription_timeout:
                expired_clients.append(client_id)

        for client_id in expired_clients:
            await self.unsubscribe(client_id)
            logger.info(f"Cleaned up expired subscription for client {client_id}")

    async def _publish_heartbeat(self):
        """Publie un heartbeat système."""
        heartbeat_event = RealTimeEvent(
            type=EventType.SYSTEM_ALERT,
            data={
                "type": "heartbeat",
                "uptime": self.get_uptime(),
                "active_subscriptions": len(self._subscriptions),
                "events_processed": self._stats["events_processed"]
            },
            source="real_time_service"
        )
        await self.publish_event(heartbeat_event)

    def _calculate_events_per_minute(self) -> float:
        """Calcule le nombre d'événements par minute."""
        uptime = self.get_uptime()
        if uptime and uptime > 0:
            return (self._stats["events_processed"] / uptime) * 60
        return 0.0