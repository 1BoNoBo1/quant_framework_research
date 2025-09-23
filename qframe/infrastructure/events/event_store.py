"""
Infrastructure Layer: Event Store
=================================

Store pour persister et récupérer les événements.
Support de l'Event Sourcing avec snapshots et streaming.
"""

import asyncio
import json
import pickle
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
from threading import Lock
import gzip

from ..observability.logging import LoggerFactory
from ..observability.metrics import get_business_metrics
from .core import Event, EventMetadata


class EventStoreError(Exception):
    """Erreur de l'event store"""
    pass


class ConcurrencyError(EventStoreError):
    """Erreur de concurrence (version conflict)"""
    pass


@dataclass
class EventStream:
    """Stream d'événements pour un agrégat"""
    stream_id: str
    stream_type: str
    version: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_event_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "stream_type": self.stream_type,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "last_event_at": self.last_event_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventStream':
        return cls(
            stream_id=data["stream_id"],
            stream_type=data["stream_type"],
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_event_at=datetime.fromisoformat(data["last_event_at"])
        )


@dataclass
class EventSnapshot:
    """Snapshot d'un agrégat pour optimiser la reconstruction"""
    stream_id: str
    version: int
    snapshot_data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    compressed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "version": self.version,
            "snapshot_data": self.snapshot_data,
            "created_at": self.created_at.isoformat(),
            "compressed": self.compressed
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventSnapshot':
        return cls(
            stream_id=data["stream_id"],
            version=data["version"],
            snapshot_data=data["snapshot_data"],
            created_at=datetime.fromisoformat(data["created_at"]),
            compressed=data.get("compressed", False)
        )


@dataclass
class StoredEvent:
    """Événement stocké avec métadonnées de persistence"""
    event: Event
    stream_id: str
    stream_version: int
    global_position: int
    stored_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event.to_dict(),
            "stream_id": self.stream_id,
            "stream_version": self.stream_version,
            "global_position": self.global_position,
            "stored_at": self.stored_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StoredEvent':
        event = Event.from_dict(data["event"])
        return cls(
            event=event,
            stream_id=data["stream_id"],
            stream_version=data["stream_version"],
            global_position=data["global_position"],
            stored_at=datetime.fromisoformat(data["stored_at"])
        )


@dataclass
class EventStoreStatistics:
    """Statistiques de l'event store"""
    total_events: int = 0
    total_streams: int = 0
    total_snapshots: int = 0

    # Performance
    events_written_per_second: float = 0.0
    events_read_per_second: float = 0.0
    avg_write_latency_ms: float = 0.0
    avg_read_latency_ms: float = 0.0

    # Storage
    storage_size_bytes: int = 0
    compressed_events: int = 0
    compression_ratio: float = 0.0

    # Streams
    largest_stream_size: int = 0
    avg_stream_size: float = 0.0


class EventStore(ABC):
    """Interface pour l'event store"""

    @abstractmethod
    async def append_events(
        self,
        stream_id: str,
        expected_version: int,
        events: List[Event]
    ) -> int:
        """Ajouter des événements à un stream"""
        pass

    @abstractmethod
    async def read_events(
        self,
        stream_id: str,
        start_version: int = 0,
        count: Optional[int] = None
    ) -> List[StoredEvent]:
        """Lire des événements d'un stream"""
        pass

    @abstractmethod
    async def read_all_events(
        self,
        start_position: int = 0,
        count: Optional[int] = None
    ) -> List[StoredEvent]:
        """Lire tous les événements globalement"""
        pass

    @abstractmethod
    async def get_stream_info(self, stream_id: str) -> Optional[EventStream]:
        """Obtenir les informations d'un stream"""
        pass

    @abstractmethod
    async def create_snapshot(self, snapshot: EventSnapshot) -> bool:
        """Créer un snapshot"""
        pass

    @abstractmethod
    async def get_snapshot(self, stream_id: str) -> Optional[EventSnapshot]:
        """Récupérer le dernier snapshot"""
        pass

    @abstractmethod
    async def stream_events(
        self,
        start_position: int = 0
    ) -> AsyncGenerator[StoredEvent, None]:
        """Stream continu des événements"""
        pass


class InMemoryEventStore(EventStore):
    """
    Implémentation en mémoire de l'event store.
    Pour développement et tests uniquement.
    """

    def __init__(self, enable_compression: bool = False):
        self.logger = LoggerFactory.get_logger(__name__)
        self.metrics = get_business_metrics()

        # Storage
        self._events: List[StoredEvent] = []
        self._streams: Dict[str, EventStream] = {}
        self._snapshots: Dict[str, EventSnapshot] = {}

        # Index pour performance
        self._events_by_stream: Dict[str, List[StoredEvent]] = defaultdict(list)
        self._global_position = 0

        # Configuration
        self._enable_compression = enable_compression
        self._snapshot_frequency = 100  # Snapshot tous les N événements

        # Statistiques
        self._stats = EventStoreStatistics()
        self._write_times: deque = deque(maxlen=1000)
        self._read_times: deque = deque(maxlen=1000)

        # Thread safety
        self._lock = Lock()

    async def append_events(
        self,
        stream_id: str,
        expected_version: int,
        events: List[Event]
    ) -> int:
        """Ajouter des événements à un stream"""
        start_time = time.time()

        try:
            with self._lock:
                # Vérifier la version du stream
                stream = self._streams.get(stream_id)
                if stream is None:
                    if expected_version != 0:
                        raise ConcurrencyError(f"Stream {stream_id} does not exist, expected version 0")

                    # Créer nouveau stream
                    stream = EventStream(
                        stream_id=stream_id,
                        stream_type=self._infer_stream_type(events[0] if events else None)
                    )
                    self._streams[stream_id] = stream
                else:
                    if stream.version != expected_version:
                        raise ConcurrencyError(
                            f"Version conflict: expected {expected_version}, got {stream.version}"
                        )

                # Ajouter les événements
                for event in events:
                    stream.version += 1
                    self._global_position += 1

                    stored_event = StoredEvent(
                        event=event,
                        stream_id=stream_id,
                        stream_version=stream.version,
                        global_position=self._global_position
                    )

                    # Compression optionnelle
                    if self._enable_compression:
                        stored_event = self._compress_event(stored_event)

                    self._events.append(stored_event)
                    self._events_by_stream[stream_id].append(stored_event)

                # Mettre à jour le stream
                stream.last_event_at = datetime.utcnow()

                # Créer snapshot si nécessaire
                if stream.version % self._snapshot_frequency == 0:
                    await self._auto_create_snapshot(stream_id, stream.version)

                # Statistiques
                write_time = (time.time() - start_time) * 1000
                self._write_times.append(write_time)
                self._stats.total_events += len(events)

                if self._write_times:
                    self._stats.avg_write_latency_ms = sum(self._write_times) / len(self._write_times)

                # Métriques
                self.metrics.collector.increment_counter(
                    "event_store.events_appended",
                    value=len(events),
                    labels={"stream_id": stream_id}
                )

                self.logger.debug(
                    f"Appended {len(events)} events to stream {stream_id}",
                    stream_id=stream_id,
                    version=stream.version,
                    event_count=len(events)
                )

                return stream.version

        except Exception as e:
            self.logger.error(f"Error appending events to stream {stream_id}: {e}")
            raise

    async def read_events(
        self,
        stream_id: str,
        start_version: int = 0,
        count: Optional[int] = None
    ) -> List[StoredEvent]:
        """Lire des événements d'un stream"""
        start_time = time.time()

        try:
            with self._lock:
                stream_events = self._events_by_stream.get(stream_id, [])

                # Filtrer par version
                filtered_events = [
                    event for event in stream_events
                    if event.stream_version > start_version
                ]

                # Limiter le nombre
                if count is not None:
                    filtered_events = filtered_events[:count]

                # Décompresser si nécessaire
                result = []
                for event in filtered_events:
                    if event.event.metadata.source == "compressed":
                        result.append(self._decompress_event(event))
                    else:
                        result.append(event)

                # Statistiques
                read_time = (time.time() - start_time) * 1000
                self._read_times.append(read_time)

                if self._read_times:
                    self._stats.avg_read_latency_ms = sum(self._read_times) / len(self._read_times)

                # Métriques
                self.metrics.collector.increment_counter(
                    "event_store.events_read",
                    value=len(result),
                    labels={"stream_id": stream_id}
                )

                return result

        except Exception as e:
            self.logger.error(f"Error reading events from stream {stream_id}: {e}")
            raise

    async def read_all_events(
        self,
        start_position: int = 0,
        count: Optional[int] = None
    ) -> List[StoredEvent]:
        """Lire tous les événements globalement"""
        with self._lock:
            # Filtrer par position globale
            filtered_events = [
                event for event in self._events
                if event.global_position > start_position
            ]

            # Limiter le nombre
            if count is not None:
                filtered_events = filtered_events[:count]

            return filtered_events

    async def get_stream_info(self, stream_id: str) -> Optional[EventStream]:
        """Obtenir les informations d'un stream"""
        with self._lock:
            return self._streams.get(stream_id)

    async def create_snapshot(self, snapshot: EventSnapshot) -> bool:
        """Créer un snapshot"""
        try:
            with self._lock:
                # Compression optionnelle du snapshot
                if self._enable_compression:
                    compressed_data = gzip.compress(
                        json.dumps(snapshot.snapshot_data).encode()
                    )
                    snapshot.snapshot_data = {
                        "_compressed": True,
                        "_data": compressed_data.hex()
                    }
                    snapshot.compressed = True

                self._snapshots[snapshot.stream_id] = snapshot
                self._stats.total_snapshots += 1

                self.logger.debug(
                    f"Created snapshot for stream {snapshot.stream_id}",
                    stream_id=snapshot.stream_id,
                    version=snapshot.version
                )

                return True

        except Exception as e:
            self.logger.error(f"Error creating snapshot: {e}")
            return False

    async def get_snapshot(self, stream_id: str) -> Optional[EventSnapshot]:
        """Récupérer le dernier snapshot"""
        with self._lock:
            snapshot = self._snapshots.get(stream_id)

            if snapshot and snapshot.compressed:
                # Décompresser
                compressed_data = bytes.fromhex(snapshot.snapshot_data["_data"])
                decompressed_data = gzip.decompress(compressed_data)
                snapshot.snapshot_data = json.loads(decompressed_data.decode())
                snapshot.compressed = False

            return snapshot

    async def stream_events(
        self,
        start_position: int = 0
    ) -> AsyncGenerator[StoredEvent, None]:
        """Stream continu des événements"""
        current_position = start_position

        while True:
            # Lire les nouveaux événements
            new_events = await self.read_all_events(current_position, count=100)

            for event in new_events:
                yield event
                current_position = event.global_position

            if not new_events:
                # Pas de nouveaux événements, attendre
                await asyncio.sleep(0.1)

    def _infer_stream_type(self, event: Optional[Event]) -> str:
        """Inférer le type de stream depuis un événement"""
        if not event:
            return "unknown"

        event_type = event.event_type
        if "order" in event_type:
            return "order"
        elif "portfolio" in event_type:
            return "portfolio"
        elif "risk" in event_type:
            return "risk"
        elif "strategy" in event_type:
            return "strategy"
        else:
            return "general"

    def _compress_event(self, stored_event: StoredEvent) -> StoredEvent:
        """Compresser un événement"""
        try:
            # Sérialiser et compresser les données
            event_data = stored_event.event.to_dict()
            compressed_data = gzip.compress(json.dumps(event_data).encode())

            # Créer un nouvel événement avec données compressées
            compressed_event_data = {
                "_compressed": True,
                "_original_size": len(json.dumps(event_data)),
                "_compressed_size": len(compressed_data),
                "_data": compressed_data.hex()
            }

            compressed_event = Event(
                event_type="compressed_event",
                data=compressed_event_data,
                metadata=stored_event.event.metadata
            )
            compressed_event.metadata.source = "compressed"

            stored_event.event = compressed_event
            self._stats.compressed_events += 1

            return stored_event

        except Exception as e:
            self.logger.warning(f"Failed to compress event: {e}")
            return stored_event

    def _decompress_event(self, stored_event: StoredEvent) -> StoredEvent:
        """Décompresser un événement"""
        try:
            if stored_event.event.metadata.source != "compressed":
                return stored_event

            compressed_data = bytes.fromhex(stored_event.event.data["_data"])
            decompressed_data = gzip.decompress(compressed_data)
            original_event_data = json.loads(decompressed_data.decode())

            # Reconstruire l'événement original
            original_event = Event.from_dict(original_event_data)
            stored_event.event = original_event

            return stored_event

        except Exception as e:
            self.logger.error(f"Failed to decompress event: {e}")
            return stored_event

    async def _auto_create_snapshot(self, stream_id: str, version: int):
        """Créer automatiquement un snapshot"""
        try:
            # Pour une implémentation complète, on reconstruirait l'état de l'agrégat
            # Ici on crée un snapshot minimal
            snapshot_data = {
                "auto_generated": True,
                "stream_id": stream_id,
                "version": version,
                "created_at": datetime.utcnow().isoformat()
            }

            snapshot = EventSnapshot(
                stream_id=stream_id,
                version=version,
                snapshot_data=snapshot_data
            )

            await self.create_snapshot(snapshot)

        except Exception as e:
            self.logger.error(f"Error creating auto snapshot: {e}")

    def get_statistics(self) -> EventStoreStatistics:
        """Obtenir les statistiques"""
        with self._lock:
            self._stats.total_streams = len(self._streams)
            self._stats.storage_size_bytes = self._estimate_storage_size()

            if self._stats.total_events > 0:
                self._stats.compression_ratio = self._stats.compressed_events / self._stats.total_events

            # Taille des streams
            stream_sizes = [len(events) for events in self._events_by_stream.values()]
            if stream_sizes:
                self._stats.largest_stream_size = max(stream_sizes)
                self._stats.avg_stream_size = sum(stream_sizes) / len(stream_sizes)

            return self._stats

    def _estimate_storage_size(self) -> int:
        """Estimer la taille de stockage"""
        total_size = 0

        # Taille des événements
        for event in self._events:
            event_json = json.dumps(event.to_dict())
            total_size += len(event_json.encode())

        # Taille des snapshots
        for snapshot in self._snapshots.values():
            snapshot_json = json.dumps(snapshot.to_dict())
            total_size += len(snapshot_json.encode())

        return total_size

    async def clear(self):
        """Vider l'event store (pour tests)"""
        with self._lock:
            self._events.clear()
            self._streams.clear()
            self._snapshots.clear()
            self._events_by_stream.clear()
            self._global_position = 0
            self._stats = EventStoreStatistics()

        self.logger.info("Event store cleared")


# Instance globale
_global_event_store = InMemoryEventStore()


def get_event_store() -> EventStore:
    """Obtenir l'instance globale de l'event store"""
    return _global_event_store