"""
Infrastructure Layer: Core Event System
=======================================

Système d'événements de base pour l'architecture événementielle.
Event, EventHandler, EventBus pour découpler les composants.
"""

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Type, Union, Set
import weakref

from ..observability.logging import LoggerFactory
from ..observability.metrics import get_business_metrics
from ..observability.tracing import get_tracer, trace


class EventType(str, Enum):
    """Types d'événements"""
    # Trading events
    ORDER_CREATED = "order.created"
    ORDER_EXECUTED = "order.executed"
    ORDER_CANCELLED = "order.cancelled"
    TRADE_EXECUTED = "trade.executed"

    # Portfolio events
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"
    PORTFOLIO_REBALANCED = "portfolio.rebalanced"

    # Risk events
    RISK_LIMIT_BREACHED = "risk.limit_breached"
    RISK_ASSESSMENT_CREATED = "risk.assessment_created"

    # Market data events
    PRICE_UPDATE = "market_data.price_update"
    MARKET_DATA_RECEIVED = "market_data.received"

    # System events
    SERVICE_STARTED = "system.service_started"
    SERVICE_STOPPED = "system.service_stopped"
    ERROR_OCCURRED = "system.error_occurred"

    # Strategy events
    STRATEGY_STARTED = "strategy.started"
    STRATEGY_STOPPED = "strategy.stopped"
    SIGNAL_GENERATED = "strategy.signal_generated"


@dataclass
class EventMetadata:
    """Métadonnées d'un événement"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    source: str = ""

    # Métriques
    processing_time_ms: float = 0.0
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convertir en dictionnaire"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventMetadata':
        """Créer depuis un dictionnaire"""
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class Event(ABC):
    """
    Classe de base pour tous les événements.
    Encapsule les données et métadonnées.
    """

    def __init__(
        self,
        event_type: str,
        data: Dict[str, Any],
        metadata: Optional[EventMetadata] = None,
        correlation_id: Optional[str] = None,
        causation_id: Optional[str] = None,
        source: str = ""
    ):
        self.metadata = metadata or EventMetadata()
        self.metadata.event_type = event_type
        self.metadata.correlation_id = correlation_id or self.metadata.correlation_id
        self.metadata.causation_id = causation_id or self.metadata.causation_id
        self.metadata.source = source or self.metadata.source

        self.data = data

    @property
    def event_id(self) -> str:
        return self.metadata.event_id

    @property
    def event_type(self) -> str:
        return self.metadata.event_type

    @property
    def timestamp(self) -> datetime:
        return self.metadata.timestamp

    @property
    def correlation_id(self) -> Optional[str]:
        return self.metadata.correlation_id

    def to_dict(self) -> Dict[str, Any]:
        """Sérialiser l'événement"""
        return {
            "metadata": self.metadata.to_dict(),
            "data": self.data
        }

    @classmethod
    def from_dict(cls, event_dict: Dict[str, Any]) -> 'Event':
        """Désérialiser un événement"""
        metadata = EventMetadata.from_dict(event_dict["metadata"])
        return cls(
            event_type=metadata.event_type,
            data=event_dict["data"],
            metadata=metadata
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(event_type='{self.event_type}', event_id='{self.event_id}')"


class DomainEvent(Event):
    """Événement métier du domaine"""

    def __init__(
        self,
        event_type: EventType,
        aggregate_id: str,
        data: Dict[str, Any],
        aggregate_version: int = 1,
        **kwargs
    ):
        super().__init__(event_type.value, data, **kwargs)
        self.aggregate_id = aggregate_id
        self.aggregate_version = aggregate_version
        self.metadata.version = aggregate_version


class SystemEvent(Event):
    """Événement système/infrastructure"""

    def __init__(
        self,
        event_type: EventType,
        component: str,
        data: Dict[str, Any],
        **kwargs
    ):
        super().__init__(event_type.value, data, **kwargs)
        self.component = component


class EventHandler(ABC):
    """Interface pour les gestionnaires d'événements"""

    @abstractmethod
    async def handle(self, event: Event) -> None:
        """Traiter un événement"""
        pass

    @abstractmethod
    def can_handle(self, event: Event) -> bool:
        """Vérifier si ce handler peut traiter l'événement"""
        pass

    @property
    @abstractmethod
    def handler_name(self) -> str:
        """Nom du handler pour logging/debugging"""
        pass


class AsyncEventHandler(EventHandler):
    """Handler asynchrone avec callback"""

    def __init__(
        self,
        name: str,
        event_types: Union[str, List[str]],
        callback: Callable[[Event], None]
    ):
        self._name = name
        self._event_types = [event_types] if isinstance(event_types, str) else event_types
        self._callback = callback

    async def handle(self, event: Event) -> None:
        """Traiter l'événement"""
        try:
            if asyncio.iscoroutinefunction(self._callback):
                await self._callback(event)
            else:
                self._callback(event)
        except Exception as e:
            # Le EventBus s'occupera de l'erreur
            raise

    def can_handle(self, event: Event) -> bool:
        """Vérifier si peut traiter cet événement"""
        return event.event_type in self._event_types

    @property
    def handler_name(self) -> str:
        return self._name


class EventHandlerRegistry:
    """Registre des handlers d'événements"""

    def __init__(self):
        self._handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        self._wildcard_handlers: List[EventHandler] = []
        self.logger = LoggerFactory.get_logger(__name__)

    def register(self, event_type: str, handler: EventHandler):
        """Enregistrer un handler pour un type d'événement"""
        if event_type == "*":
            self._wildcard_handlers.append(handler)
        else:
            self._handlers[event_type].append(handler)

        self.logger.info(f"Registered handler '{handler.handler_name}' for event type '{event_type}'")

    def register_callback(
        self,
        event_type: str,
        callback: Callable[[Event], None],
        handler_name: Optional[str] = None
    ):
        """Enregistrer un callback simple"""
        name = handler_name or f"callback_{int(time.time())}"
        handler = AsyncEventHandler(name, event_type, callback)
        self.register(event_type, handler)

    def unregister(self, event_type: str, handler: EventHandler):
        """Désenregistrer un handler"""
        if event_type == "*":
            if handler in self._wildcard_handlers:
                self._wildcard_handlers.remove(handler)
        else:
            if event_type in self._handlers and handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)

    def get_handlers(self, event_type: str) -> List[EventHandler]:
        """Obtenir tous les handlers pour un type d'événement"""
        handlers = []

        # Handlers spécifiques
        handlers.extend(self._handlers.get(event_type, []))

        # Handlers wildcard
        handlers.extend(self._wildcard_handlers)

        return handlers

    def get_statistics(self) -> Dict[str, Any]:
        """Obtenir les statistiques du registre"""
        specific_handlers = sum(len(handlers) for handlers in self._handlers.values())
        wildcard_handlers = len(self._wildcard_handlers)

        return {
            "total_handlers": specific_handlers + wildcard_handlers,
            "specific_handlers": specific_handlers,
            "wildcard_handlers": wildcard_handlers,
            "event_types_registered": len(self._handlers),
            "handlers_by_type": {
                event_type: len(handlers)
                for event_type, handlers in self._handlers.items()
            }
        }


class EventBus:
    """
    Bus d'événements central pour publier et distribuer les événements.
    Gère l'orchestration des handlers et la propagation des erreurs.
    """

    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)
        self.metrics = get_business_metrics()
        self.tracer = get_tracer()

        self._registry = EventHandlerRegistry()
        self._running = False

        # Queue pour les événements
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self._processing_task: Optional[asyncio.Task] = None

        # Statistiques
        self._stats = {
            "events_published": 0,
            "events_processed": 0,
            "events_failed": 0,
            "handlers_executed": 0,
            "handlers_failed": 0
        }

        # Configuration
        self._max_retries = 3
        self._retry_delay_ms = 100

    async def start(self):
        """Démarrer le bus d'événements"""
        if self._running:
            return

        self._running = True
        self._processing_task = asyncio.create_task(self._process_events())

        self.logger.info("Event bus started")

    async def stop(self):
        """Arrêter le bus d'événements"""
        self._running = False

        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Event bus stopped")

    @trace("event_bus.publish")
    async def publish(self, event: Event) -> bool:
        """Publier un événement"""
        try:
            if not self._running:
                self.logger.warning("Event bus not running, dropping event")
                return False

            # Enqueue l'événement
            if self._event_queue.full():
                self.logger.warning("Event queue full, dropping event")
                self._stats["events_failed"] += 1
                return False

            self._event_queue.put_nowait(event)
            self._stats["events_published"] += 1

            # Métriques
            self.metrics.collector.increment_counter(
                "events.published",
                labels={"event_type": event.event_type, "source": event.metadata.source}
            )

            self.logger.debug(
                f"Published event {event.event_type}",
                event_id=event.event_id,
                event_type=event.event_type,
                correlation_id=event.correlation_id
            )

            return True

        except Exception as e:
            self.logger.error(f"Error publishing event: {e}", error=e)
            self._stats["events_failed"] += 1
            return False

    async def publish_and_wait(self, event: Event) -> bool:
        """Publier un événement et attendre qu'il soit traité"""
        if not await self.publish(event):
            return False

        # Attendre que l'événement soit traité
        while not self._event_queue.empty():
            await asyncio.sleep(0.01)

        return True

    def subscribe(self, event_type: str, handler: EventHandler):
        """S'abonner à un type d'événement"""
        self._registry.register(event_type, handler)

    def subscribe_callback(
        self,
        event_type: str,
        callback: Callable[[Event], None],
        handler_name: Optional[str] = None
    ):
        """S'abonner avec un callback simple"""
        self._registry.register_callback(event_type, callback, handler_name)

    def unsubscribe(self, event_type: str, handler: EventHandler):
        """Se désabonner"""
        self._registry.unregister(event_type, handler)

    async def _process_events(self):
        """Traiter les événements en queue"""
        while self._running:
            try:
                # Récupérer un événement avec timeout
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )

                start_time = time.time()
                await self._handle_event(event)

                # Mesurer le temps de traitement
                processing_time = (time.time() - start_time) * 1000
                event.metadata.processing_time_ms = processing_time

                self._stats["events_processed"] += 1

                # Métriques
                self.metrics.collector.record_histogram(
                    "events.processing_time",
                    processing_time,
                    labels={"event_type": event.event_type}
                )

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in event processing loop: {e}")

    @trace("event_bus.handle_event")
    async def _handle_event(self, event: Event):
        """Traiter un événement avec tous ses handlers"""
        handlers = self._registry.get_handlers(event.event_type)

        if not handlers:
            self.logger.debug(f"No handlers for event type {event.event_type}")
            return

        # Exécuter tous les handlers
        for handler in handlers:
            await self._execute_handler(handler, event)

    async def _execute_handler(self, handler: EventHandler, event: Event):
        """Exécuter un handler avec retry et gestion d'erreur"""
        if not handler.can_handle(event):
            return

        retry_count = 0
        while retry_count <= self._max_retries:
            try:
                await handler.handle(event)

                self._stats["handlers_executed"] += 1

                # Métriques de succès
                self.metrics.collector.increment_counter(
                    "events.handlers.success",
                    labels={
                        "event_type": event.event_type,
                        "handler_name": handler.handler_name
                    }
                )

                self.logger.debug(
                    f"Handler {handler.handler_name} processed event {event.event_type}",
                    event_id=event.event_id,
                    handler_name=handler.handler_name,
                    retry_count=retry_count
                )

                return  # Succès

            except Exception as e:
                retry_count += 1
                self._stats["handlers_failed"] += 1

                # Métriques d'erreur
                self.metrics.collector.increment_counter(
                    "events.handlers.error",
                    labels={
                        "event_type": event.event_type,
                        "handler_name": handler.handler_name,
                        "error_type": type(e).__name__
                    }
                )

                self.logger.error(
                    f"Handler {handler.handler_name} failed to process event {event.event_type} (attempt {retry_count})",
                    error=e,
                    event_id=event.event_id,
                    handler_name=handler.handler_name,
                    retry_count=retry_count
                )

                if retry_count <= self._max_retries:
                    # Attendre avant retry
                    await asyncio.sleep(self._retry_delay_ms * retry_count / 1000)
                else:
                    # Échec définitif
                    self.logger.error(
                        f"Handler {handler.handler_name} permanently failed for event {event.event_type}",
                        event_id=event.event_id,
                        handler_name=handler.handler_name
                    )

    def get_statistics(self) -> Dict[str, Any]:
        """Obtenir les statistiques du bus"""
        registry_stats = self._registry.get_statistics()

        return {
            **self._stats,
            "queue_size": self._event_queue.qsize(),
            "queue_max_size": self._event_queue.maxsize,
            "running": self._running,
            "registry": registry_stats
        }


# Instance globale
_global_event_bus = EventBus()


def get_event_bus() -> EventBus:
    """Obtenir l'instance globale du bus d'événements"""
    return _global_event_bus