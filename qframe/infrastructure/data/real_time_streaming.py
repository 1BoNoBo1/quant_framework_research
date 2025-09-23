"""
Infrastructure Layer: Real-Time Market Data Streaming
====================================================

Service de streaming en temps réel pour distribuer les données de marché
vers différents consommateurs avec gestion des subscriptions et backpressure.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set, Deque, AsyncGenerator
from threading import Lock
import weakref

from ..observability.logging import LoggerFactory
from ..observability.metrics import get_business_metrics
from ..observability.tracing import get_tracer, trace
from .market_data_pipeline import MarketDataPoint, DataType, get_market_data_pipeline


class StreamingProtocol(str, Enum):
    """Protocoles de streaming supportés"""
    WEBSOCKET = "websocket"
    SSE = "sse"  # Server-Sent Events
    CALLBACK = "callback"
    ASYNC_GENERATOR = "async_generator"
    REDIS_PUBSUB = "redis_pubsub"


class SubscriptionLevel(str, Enum):
    """Niveaux de subscription"""
    SYMBOL = "symbol"      # Données pour un symbole spécifique
    TYPE = "type"          # Tous les symboles d'un type de données
    PROVIDER = "provider"  # Toutes les données d'un provider
    ALL = "all"           # Toutes les données


@dataclass
class StreamingSubscription:
    """Représente une subscription streaming"""
    subscription_id: str
    client_id: str
    level: SubscriptionLevel
    filter_criteria: Dict[str, Any]
    protocol: StreamingProtocol
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)

    # Configuration
    max_queue_size: int = 1000
    backpressure_enabled: bool = True

    # État
    active: bool = True
    message_count: int = 0
    dropped_messages: int = 0

    def matches(self, data_point: MarketDataPoint) -> bool:
        """Vérifier si un point de données correspond aux critères"""
        if self.level == SubscriptionLevel.ALL:
            return True

        if self.level == SubscriptionLevel.SYMBOL:
            return data_point.symbol == self.filter_criteria.get("symbol")

        if self.level == SubscriptionLevel.TYPE:
            return data_point.data_type.value == self.filter_criteria.get("data_type")

        if self.level == SubscriptionLevel.PROVIDER:
            return data_point.provider == self.filter_criteria.get("provider")

        return False


class StreamingConsumer(ABC):
    """Interface pour les consommateurs de streaming"""

    @abstractmethod
    async def send_data(self, data_point: MarketDataPoint) -> bool:
        """Envoyer des données au consommateur"""
        pass

    @abstractmethod
    async def close(self):
        """Fermer la connexion"""
        pass

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """Vérifier si le consommateur est actif"""
        pass


class CallbackConsumer(StreamingConsumer):
    """Consommateur utilisant des callbacks"""

    def __init__(self, callback: Callable[[MarketDataPoint], None]):
        self.callback = callback
        self._active = True

    async def send_data(self, data_point: MarketDataPoint) -> bool:
        try:
            if self._active:
                self.callback(data_point)
                return True
        except Exception:
            self._active = False
        return False

    async def close(self):
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active


class AsyncGeneratorConsumer(StreamingConsumer):
    """Consommateur utilisant des async generators"""

    def __init__(self, max_queue_size: int = 1000):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._active = True

    async def send_data(self, data_point: MarketDataPoint) -> bool:
        if not self._active:
            return False

        try:
            self.queue.put_nowait(data_point)
            return True
        except asyncio.QueueFull:
            # Backpressure: supprimer les anciens messages
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(data_point)
                return True
            except asyncio.QueueEmpty:
                return False

    async def close(self):
        self._active = False
        # Vider la queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    @property
    def is_active(self) -> bool:
        return self._active

    async def stream(self) -> AsyncGenerator[MarketDataPoint, None]:
        """Générateur asynchrone pour les données"""
        while self._active:
            try:
                data_point = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=1.0
                )
                yield data_point
            except asyncio.TimeoutError:
                continue
            except Exception:
                break


@dataclass
class StreamingStatistics:
    """Statistiques du service de streaming"""
    total_subscriptions: int = 0
    active_subscriptions: int = 0
    total_consumers: int = 0
    active_consumers: int = 0
    messages_sent: int = 0
    messages_dropped: int = 0

    # Par protocole
    subscriptions_by_protocol: Dict[str, int] = field(default_factory=dict)

    # Par niveau
    subscriptions_by_level: Dict[str, int] = field(default_factory=dict)

    # Performance
    avg_processing_time_ms: float = 0.0
    max_queue_size: int = 0
    current_queue_size: int = 0


class RealTimeStreamingService:
    """
    Service de streaming en temps réel pour distribuer les données de marché.
    Gère les subscriptions, la distribution et la backpressure.
    """

    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)
        self.metrics = get_business_metrics()
        self.tracer = get_tracer()

        # État des subscriptions
        self._subscriptions: Dict[str, StreamingSubscription] = {}
        self._consumers: Dict[str, StreamingConsumer] = {}
        self._subscription_consumers: Dict[str, str] = {}  # subscription_id -> consumer_id

        # Gestion des données
        self._message_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self._processing_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None

        # Statistiques
        self._stats = StreamingStatistics()
        self._processing_times: Deque[float] = deque(maxlen=1000)

        # Configuration
        self._running = False
        self._cleanup_interval = 300  # 5 minutes
        self._max_inactive_time = 3600  # 1 heure

        # Weak references pour éviter les fuites mémoire
        self._consumer_refs = weakref.WeakValueDictionary()

    async def start(self):
        """Démarrer le service de streaming"""
        if self._running:
            return

        self._running = True

        # S'abonner au pipeline de données
        pipeline = get_market_data_pipeline()
        pipeline.subscribe("*", self._on_market_data)

        # Démarrer les tâches de traitement
        self._processing_task = asyncio.create_task(self._process_messages())
        self._cleanup_task = asyncio.create_task(self._cleanup_inactive_subscriptions())

        self.logger.info("Real-time streaming service started")

    async def stop(self):
        """Arrêter le service de streaming"""
        self._running = False

        # Arrêter les tâches
        if self._processing_task:
            self._processing_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()

        # Fermer tous les consommateurs
        for consumer in self._consumers.values():
            await consumer.close()

        self._subscriptions.clear()
        self._consumers.clear()
        self._subscription_consumers.clear()

        self.logger.info("Real-time streaming service stopped")

    def _on_market_data(self, data_point: MarketDataPoint):
        """Callback pour recevoir les données du pipeline"""
        try:
            if self._running and not self._message_queue.full():
                self._message_queue.put_nowait(data_point)
            else:
                self._stats.messages_dropped += 1
        except Exception as e:
            self.logger.error(f"Error queuing market data: {e}")

    async def _process_messages(self):
        """Traiter les messages en attente"""
        while self._running:
            try:
                # Récupérer un message avec timeout
                data_point = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0
                )

                start_time = time.time()
                await self._distribute_data_point(data_point)

                # Mesurer les performances
                processing_time = (time.time() - start_time) * 1000
                self._processing_times.append(processing_time)

                if len(self._processing_times) > 0:
                    self._stats.avg_processing_time_ms = sum(self._processing_times) / len(self._processing_times)

                self._stats.messages_sent += 1

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")

    @trace("streaming.distribute")
    async def _distribute_data_point(self, data_point: MarketDataPoint):
        """Distribuer un point de données aux subscriptions correspondantes"""
        active_subscriptions = []

        for subscription in self._subscriptions.values():
            if subscription.active and subscription.matches(data_point):
                active_subscriptions.append(subscription)

        # Distribuer aux consommateurs
        for subscription in active_subscriptions:
            consumer_id = self._subscription_consumers.get(subscription.subscription_id)
            if consumer_id and consumer_id in self._consumers:
                consumer = self._consumers[consumer_id]

                try:
                    success = await consumer.send_data(data_point)
                    if success:
                        subscription.message_count += 1
                        subscription.last_activity = datetime.utcnow()
                    else:
                        subscription.dropped_messages += 1

                        # Marquer comme inactif si trop d'échecs
                        if subscription.dropped_messages > 100:
                            subscription.active = False

                except Exception as e:
                    self.logger.error(f"Error sending data to consumer {consumer_id}: {e}")
                    subscription.dropped_messages += 1

    def subscribe_callback(
        self,
        client_id: str,
        callback: Callable[[MarketDataPoint], None],
        level: SubscriptionLevel = SubscriptionLevel.ALL,
        **filter_criteria
    ) -> str:
        """Créer une subscription avec callback"""
        subscription_id = f"sub_{int(time.time() * 1000)}_{client_id}"
        consumer_id = f"cons_{subscription_id}"

        # Créer la subscription
        subscription = StreamingSubscription(
            subscription_id=subscription_id,
            client_id=client_id,
            level=level,
            filter_criteria=filter_criteria,
            protocol=StreamingProtocol.CALLBACK
        )

        # Créer le consommateur
        consumer = CallbackConsumer(callback)

        # Enregistrer
        self._subscriptions[subscription_id] = subscription
        self._consumers[consumer_id] = consumer
        self._subscription_consumers[subscription_id] = consumer_id

        self._update_statistics()

        self.logger.info(f"Created callback subscription {subscription_id} for client {client_id}")
        return subscription_id

    def subscribe_async_generator(
        self,
        client_id: str,
        level: SubscriptionLevel = SubscriptionLevel.ALL,
        max_queue_size: int = 1000,
        **filter_criteria
    ) -> tuple[str, AsyncGenerator[MarketDataPoint, None]]:
        """Créer une subscription avec async generator"""
        subscription_id = f"sub_{int(time.time() * 1000)}_{client_id}"
        consumer_id = f"cons_{subscription_id}"

        # Créer la subscription
        subscription = StreamingSubscription(
            subscription_id=subscription_id,
            client_id=client_id,
            level=level,
            filter_criteria=filter_criteria,
            protocol=StreamingProtocol.ASYNC_GENERATOR,
            max_queue_size=max_queue_size
        )

        # Créer le consommateur
        consumer = AsyncGeneratorConsumer(max_queue_size)

        # Enregistrer
        self._subscriptions[subscription_id] = subscription
        self._consumers[consumer_id] = consumer
        self._subscription_consumers[subscription_id] = consumer_id

        self._update_statistics()

        self.logger.info(f"Created async generator subscription {subscription_id} for client {client_id}")
        return subscription_id, consumer.stream()

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Se désabonner"""
        if subscription_id not in self._subscriptions:
            return False

        # Récupérer et fermer le consommateur
        consumer_id = self._subscription_consumers.get(subscription_id)
        if consumer_id and consumer_id in self._consumers:
            await self._consumers[consumer_id].close()
            del self._consumers[consumer_id]

        # Supprimer la subscription
        del self._subscriptions[subscription_id]
        if subscription_id in self._subscription_consumers:
            del self._subscription_consumers[subscription_id]

        self._update_statistics()

        self.logger.info(f"Unsubscribed {subscription_id}")
        return True

    async def unsubscribe_client(self, client_id: str) -> int:
        """Se désabonner de toutes les subscriptions d'un client"""
        client_subscriptions = [
            sub_id for sub_id, sub in self._subscriptions.items()
            if sub.client_id == client_id
        ]

        count = 0
        for subscription_id in client_subscriptions:
            if await self.unsubscribe(subscription_id):
                count += 1

        self.logger.info(f"Unsubscribed {count} subscriptions for client {client_id}")
        return count

    async def _cleanup_inactive_subscriptions(self):
        """Nettoyer les subscriptions inactives"""
        while self._running:
            try:
                await asyncio.sleep(self._cleanup_interval)

                now = datetime.utcnow()
                to_remove = []

                for subscription_id, subscription in self._subscriptions.items():
                    # Vérifier l'inactivité
                    inactive_time = (now - subscription.last_activity).total_seconds()

                    if not subscription.active or inactive_time > self._max_inactive_time:
                        to_remove.append(subscription_id)

                # Supprimer les subscriptions inactives
                for subscription_id in to_remove:
                    await self.unsubscribe(subscription_id)

                if to_remove:
                    self.logger.info(f"Cleaned up {len(to_remove)} inactive subscriptions")

            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")

    def _update_statistics(self):
        """Mettre à jour les statistiques"""
        self._stats.total_subscriptions = len(self._subscriptions)
        self._stats.active_subscriptions = len([s for s in self._subscriptions.values() if s.active])
        self._stats.total_consumers = len(self._consumers)
        self._stats.active_consumers = len([c for c in self._consumers.values() if c.is_active])

        # Par protocole
        protocol_counts = defaultdict(int)
        for sub in self._subscriptions.values():
            protocol_counts[sub.protocol.value] += 1
        self._stats.subscriptions_by_protocol = dict(protocol_counts)

        # Par niveau
        level_counts = defaultdict(int)
        for sub in self._subscriptions.values():
            level_counts[sub.level.value] += 1
        self._stats.subscriptions_by_level = dict(level_counts)

        # Queue size
        self._stats.current_queue_size = self._message_queue.qsize()
        self._stats.max_queue_size = self._message_queue.maxsize

    def get_statistics(self) -> StreamingStatistics:
        """Obtenir les statistiques actuelles"""
        self._update_statistics()
        return self._stats

    def get_subscription_info(self, subscription_id: str) -> Optional[Dict[str, Any]]:
        """Obtenir les informations d'une subscription"""
        if subscription_id not in self._subscriptions:
            return None

        subscription = self._subscriptions[subscription_id]
        return {
            "subscription_id": subscription.subscription_id,
            "client_id": subscription.client_id,
            "level": subscription.level.value,
            "filter_criteria": subscription.filter_criteria,
            "protocol": subscription.protocol.value,
            "created_at": subscription.created_at.isoformat(),
            "last_activity": subscription.last_activity.isoformat(),
            "active": subscription.active,
            "message_count": subscription.message_count,
            "dropped_messages": subscription.dropped_messages
        }

    def list_subscriptions(self, client_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lister les subscriptions"""
        subscriptions = []

        for subscription in self._subscriptions.values():
            if client_id is None or subscription.client_id == client_id:
                info = self.get_subscription_info(subscription.subscription_id)
                if info:
                    subscriptions.append(info)

        return subscriptions


# Instance globale
_global_streaming_service = RealTimeStreamingService()


def get_streaming_service() -> RealTimeStreamingService:
    """Obtenir l'instance globale du service de streaming"""
    return _global_streaming_service