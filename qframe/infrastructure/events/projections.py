"""
Infrastructure Layer: Event Projections
=======================================

Système de projections pour maintenir des vues en lecture optimisées
basées sur les événements (CQRS).
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set, Type
import weakref

from ..observability.logging import LoggerFactory
from ..observability.metrics import get_business_metrics
from ..observability.tracing import get_tracer, trace
from .core import Event, get_event_bus
from .event_store import get_event_store, StoredEvent


class ProjectionState(str, Enum):
    """État d'une projection"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    REBUILDING = "rebuilding"


@dataclass
class ProjectionCheckpoint:
    """Point de contrôle d'une projection"""
    projection_name: str
    last_processed_position: int = 0
    last_processed_at: datetime = field(default_factory=datetime.utcnow)
    events_processed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "projection_name": self.projection_name,
            "last_processed_position": self.last_processed_position,
            "last_processed_at": self.last_processed_at.isoformat(),
            "events_processed": self.events_processed
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectionCheckpoint':
        return cls(
            projection_name=data["projection_name"],
            last_processed_position=data["last_processed_position"],
            last_processed_at=datetime.fromisoformat(data["last_processed_at"]),
            events_processed=data["events_processed"]
        )


@dataclass
class ProjectionStatistics:
    """Statistiques d'une projection"""
    events_processed: int = 0
    events_skipped: int = 0
    processing_errors: int = 0
    avg_processing_time_ms: float = 0.0
    last_error: Optional[str] = None
    last_error_at: Optional[datetime] = None

    # Performance
    events_per_second: float = 0.0
    lag_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.last_error_at:
            data["last_error_at"] = self.last_error_at.isoformat()
        return data


class Projection(ABC):
    """
    Classe de base pour les projections.
    Une projection maintient une vue en lecture basée sur les événements.
    """

    def __init__(self, name: str):
        self.name = name
        self.logger = LoggerFactory.get_logger(f"{__name__}.{name}")
        self.metrics = get_business_metrics()

        self._state = ProjectionState.INITIALIZING
        self._checkpoint = ProjectionCheckpoint(projection_name=name)
        self._statistics = ProjectionStatistics()

        # Configuration
        self._batch_size = 100
        self._processing_delay_ms = 10

    @property
    def state(self) -> ProjectionState:
        return self._state

    @property
    def checkpoint(self) -> ProjectionCheckpoint:
        return self._checkpoint

    @property
    def statistics(self) -> ProjectionStatistics:
        return self._statistics

    @abstractmethod
    async def handle_event(self, event: StoredEvent) -> bool:
        """
        Traiter un événement pour mettre à jour la projection.
        Retourne True si l'événement a été traité, False s'il a été ignoré.
        """
        pass

    @abstractmethod
    def supports_event(self, event: StoredEvent) -> bool:
        """Vérifier si cette projection supporte cet événement"""
        pass

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialiser la projection"""
        pass

    @abstractmethod
    async def reset(self) -> bool:
        """Réinitialiser la projection (supprimer toutes les données)"""
        pass

    async def start(self, from_position: Optional[int] = None):
        """Démarrer la projection"""
        self._state = ProjectionState.INITIALIZING

        try:
            # Initialiser
            if not await self.initialize():
                self._state = ProjectionState.ERROR
                return

            # Définir la position de départ
            if from_position is not None:
                self._checkpoint.last_processed_position = from_position

            self._state = ProjectionState.RUNNING
            self.logger.info(f"Projection '{self.name}' started from position {self._checkpoint.last_processed_position}")

        except Exception as e:
            self._state = ProjectionState.ERROR
            self.logger.error(f"Failed to start projection '{self.name}': {e}")

    async def stop(self):
        """Arrêter la projection"""
        self._state = ProjectionState.STOPPED
        self.logger.info(f"Projection '{self.name}' stopped")

    async def rebuild(self):
        """Reconstruire la projection depuis le début"""
        self._state = ProjectionState.REBUILDING
        self.logger.info(f"Rebuilding projection '{self.name}'")

        try:
            # Réinitialiser
            await self.reset()
            self._checkpoint.last_processed_position = 0
            self._checkpoint.events_processed = 0
            self._statistics = ProjectionStatistics()

            # Redémarrer
            await self.start()

        except Exception as e:
            self._state = ProjectionState.ERROR
            self.logger.error(f"Failed to rebuild projection '{self.name}': {e}")

    async def process_event(self, event: StoredEvent) -> bool:
        """Traiter un événement avec mesure de performance"""
        if self._state != ProjectionState.RUNNING:
            return False

        if not self.supports_event(event):
            self._statistics.events_skipped += 1
            return False

        start_time = time.time()

        try:
            handled = await self.handle_event(event)

            if handled:
                # Mettre à jour le checkpoint
                self._checkpoint.last_processed_position = event.global_position
                self._checkpoint.last_processed_at = datetime.utcnow()
                self._checkpoint.events_processed += 1

                # Statistiques
                processing_time = (time.time() - start_time) * 1000
                if self._statistics.events_processed > 0:
                    self._statistics.avg_processing_time_ms = (
                        (self._statistics.avg_processing_time_ms * (self._statistics.events_processed - 1) + processing_time) /
                        self._statistics.events_processed
                    )
                else:
                    self._statistics.avg_processing_time_ms = processing_time

                self._statistics.events_processed += 1

                # Métriques
                self.metrics.collector.increment_counter(
                    "projections.events_processed",
                    labels={"projection_name": self.name, "event_type": event.event.event_type}
                )

                self.metrics.collector.record_histogram(
                    "projections.processing_time",
                    processing_time,
                    labels={"projection_name": self.name}
                )

            return handled

        except Exception as e:
            self._statistics.processing_errors += 1
            self._statistics.last_error = str(e)
            self._statistics.last_error_at = datetime.utcnow()

            self.logger.error(
                f"Error processing event in projection '{self.name}': {e}",
                error=e,
                event_id=event.event.event_id,
                event_type=event.event.event_type
            )

            # Métriques d'erreur
            self.metrics.collector.increment_counter(
                "projections.processing_errors",
                labels={"projection_name": self.name, "error_type": type(e).__name__}
            )

            return False


class ReadModelUpdater(Projection):
    """
    Projection simple qui met à jour des modèles de lecture
    via des callbacks configurables.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self._event_handlers: Dict[str, List[Callable[[StoredEvent], None]]] = defaultdict(list)
        self._supported_events: Set[str] = set()

    def add_event_handler(self, event_type: str, handler: Callable[[StoredEvent], None]):
        """Ajouter un handler pour un type d'événement"""
        self._event_handlers[event_type].append(handler)
        self._supported_events.add(event_type)
        self.logger.debug(f"Added handler for event type '{event_type}'")

    def supports_event(self, event: StoredEvent) -> bool:
        """Vérifier si cet événement est supporté"""
        return event.event.event_type in self._supported_events

    async def handle_event(self, event: StoredEvent) -> bool:
        """Traiter un événement avec tous les handlers configurés"""
        handlers = self._event_handlers.get(event.event.event_type, [])

        if not handlers:
            return False

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                self.logger.error(f"Handler error: {e}", error=e)

        return True

    async def initialize(self) -> bool:
        """Initialiser (rien à faire pour cette implémentation)"""
        return True

    async def reset(self) -> bool:
        """Réinitialiser (rien à faire pour cette implémentation)"""
        return True


class ProjectionManager:
    """
    Gestionnaire des projections pour orchestrer leur exécution
    et maintenir leurs checkpoints.
    """

    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)
        self.metrics = get_business_metrics()
        self.tracer = get_tracer()

        # Projections enregistrées
        self._projections: Dict[str, Projection] = {}

        # Tâches de traitement
        self._processing_tasks: Dict[str, asyncio.Task] = {}
        self._running = False

        # Event store et bus
        self.event_store = get_event_store()

        # Configuration
        self._polling_interval_ms = 100
        self._batch_size = 100

    def register_projection(self, projection: Projection):
        """Enregistrer une projection"""
        self._projections[projection.name] = projection
        self.logger.info(f"Registered projection: {projection.name}")

    def unregister_projection(self, projection_name: str):
        """Désenregistrer une projection"""
        if projection_name in self._projections:
            # Arrêter la tâche de traitement
            if projection_name in self._processing_tasks:
                self._processing_tasks[projection_name].cancel()
                del self._processing_tasks[projection_name]

            del self._projections[projection_name]
            self.logger.info(f"Unregistered projection: {projection_name}")

    async def start_all(self):
        """Démarrer toutes les projections"""
        self._running = True

        for projection in self._projections.values():
            await self._start_projection(projection)

        self.logger.info(f"Started {len(self._projections)} projections")

    async def stop_all(self):
        """Arrêter toutes les projections"""
        self._running = False

        # Arrêter toutes les tâches
        for task in self._processing_tasks.values():
            task.cancel()

        # Attendre que toutes les tâches se terminent
        if self._processing_tasks:
            await asyncio.gather(*self._processing_tasks.values(), return_exceptions=True)

        # Arrêter toutes les projections
        for projection in self._projections.values():
            await projection.stop()

        self._processing_tasks.clear()
        self.logger.info("Stopped all projections")

    async def _start_projection(self, projection: Projection):
        """Démarrer une projection individuelle"""
        await projection.start()

        # Démarrer la tâche de traitement
        task = asyncio.create_task(self._process_projection(projection))
        self._processing_tasks[projection.name] = task

        self.logger.info(f"Started projection processing for '{projection.name}'")

    @trace("projections.process")
    async def _process_projection(self, projection: Projection):
        """Traiter les événements pour une projection"""
        try:
            while self._running and projection.state == ProjectionState.RUNNING:
                # Lire les nouveaux événements
                events = await self.event_store.read_all_events(
                    start_position=projection.checkpoint.last_processed_position,
                    count=self._batch_size
                )

                if not events:
                    # Pas de nouveaux événements, attendre
                    await asyncio.sleep(self._polling_interval_ms / 1000)
                    continue

                # Traiter chaque événement
                for event in events:
                    if not self._running or projection.state != ProjectionState.RUNNING:
                        break

                    await projection.process_event(event)

                # Calculer le lag
                if events:
                    last_event_time = events[-1].stored_at
                    lag = (datetime.utcnow() - last_event_time).total_seconds()
                    projection.statistics.lag_seconds = lag

                    # Calculer les événements par seconde
                    if projection.statistics.events_processed > 0:
                        duration = (datetime.utcnow() - projection.checkpoint.last_processed_at).total_seconds()
                        if duration > 0:
                            projection.statistics.events_per_second = len(events) / duration

                # Pause entre les batches
                await asyncio.sleep(self._processing_delay_ms / 1000)

        except asyncio.CancelledError:
            self.logger.info(f"Projection processing cancelled for '{projection.name}'")
        except Exception as e:
            projection._state = ProjectionState.ERROR
            self.logger.error(f"Projection processing error for '{projection.name}': {e}", error=e)

    async def rebuild_projection(self, projection_name: str):
        """Reconstruire une projection"""
        if projection_name not in self._projections:
            raise ValueError(f"Unknown projection: {projection_name}")

        projection = self._projections[projection_name]

        # Arrêter temporairement la tâche de traitement
        if projection_name in self._processing_tasks:
            self._processing_tasks[projection_name].cancel()
            try:
                await self._processing_tasks[projection_name]
            except asyncio.CancelledError:
                pass

        # Reconstruire
        await projection.rebuild()

        # Redémarrer la tâche de traitement
        if self._running and projection.state == ProjectionState.RUNNING:
            task = asyncio.create_task(self._process_projection(projection))
            self._processing_tasks[projection_name] = task

        self.logger.info(f"Rebuilt projection '{projection_name}'")

    def get_projection(self, name: str) -> Optional[Projection]:
        """Obtenir une projection par nom"""
        return self._projections.get(name)

    def list_projections(self) -> List[str]:
        """Lister les noms des projections"""
        return list(self._projections.keys())

    def get_projection_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Obtenir le statut d'une projection"""
        projection = self._projections.get(name)
        if not projection:
            return None

        return {
            "name": projection.name,
            "state": projection.state.value,
            "checkpoint": projection.checkpoint.to_dict(),
            "statistics": projection.statistics.to_dict()
        }

    def get_global_statistics(self) -> Dict[str, Any]:
        """Obtenir les statistiques globales"""
        total_events = sum(p.statistics.events_processed for p in self._projections.values())
        total_errors = sum(p.statistics.processing_errors for p in self._projections.values())

        running_projections = len([p for p in self._projections.values() if p.state == ProjectionState.RUNNING])
        error_projections = len([p for p in self._projections.values() if p.state == ProjectionState.ERROR])

        avg_lag = 0.0
        if self._projections:
            avg_lag = sum(p.statistics.lag_seconds for p in self._projections.values()) / len(self._projections)

        return {
            "total_projections": len(self._projections),
            "running_projections": running_projections,
            "error_projections": error_projections,
            "total_events_processed": total_events,
            "total_processing_errors": total_errors,
            "average_lag_seconds": avg_lag,
            "manager_running": self._running
        }


# Instance globale
_global_projection_manager = ProjectionManager()


def get_projection_manager() -> ProjectionManager:
    """Obtenir l'instance globale du gestionnaire de projections"""
    return _global_projection_manager