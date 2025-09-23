"""
Infrastructure Layer: WebSocket API Service
===========================================

Service WebSocket pour streaming temps réel des données de marché
et événements du système avec gestion des connexions.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Callable
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
from pydantic import BaseModel, Field

from ..observability.logging import LoggerFactory
from ..observability.metrics import get_business_metrics
from ..observability.tracing import get_tracer, trace

from ..data.market_data_pipeline import MarketDataPoint, get_market_data_pipeline
from ..data.real_time_streaming import get_streaming_service, SubscriptionLevel

from ..events.core import Event, get_event_bus


class MessageType(str, Enum):
    """Types de messages WebSocket"""
    # Client -> Server
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PING = "ping"
    AUTH = "auth"

    # Server -> Client
    DATA = "data"
    ERROR = "error"
    PONG = "pong"
    CONNECTED = "connected"
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"


class WSMessage(BaseModel):
    """Message WebSocket standard"""
    type: MessageType
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class WSSubscribeMessage(BaseModel):
    """Message de subscription"""
    type: MessageType = MessageType.SUBSCRIBE
    channel: str  # "market_data", "events", "orders", etc.
    params: Dict[str, Any] = Field(default_factory=dict)


class WSConnection:
    """Représente une connexion WebSocket active"""

    def __init__(self, websocket: WebSocket, connection_id: str, client_id: Optional[str] = None):
        self.websocket = websocket
        self.connection_id = connection_id
        self.client_id = client_id or str(uuid.uuid4())
        self.connected_at = datetime.utcnow()
        self.last_ping = datetime.utcnow()

        # Subscriptions actives
        self.subscriptions: Dict[str, str] = {}  # channel -> subscription_id
        self.channels: Set[str] = set()

        # Statistiques
        self.messages_sent = 0
        self.messages_received = 0
        self.bytes_sent = 0
        self.bytes_received = 0

        # État
        self.authenticated = False
        self.user_id: Optional[str] = None

    @property
    def is_active(self) -> bool:
        """Vérifier si la connexion est active"""
        return (
            self.websocket.client_state == WebSocketState.CONNECTED and
            self.websocket.application_state == WebSocketState.CONNECTED
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convertir en dictionnaire pour sérialisation"""
        return {
            "connection_id": self.connection_id,
            "client_id": self.client_id,
            "connected_at": self.connected_at.isoformat(),
            "last_ping": self.last_ping.isoformat(),
            "channels": list(self.channels),
            "subscriptions_count": len(self.subscriptions),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "authenticated": self.authenticated,
            "user_id": self.user_id
        }


class ConnectionManager:
    """
    Gestionnaire des connexions WebSocket.
    Gère le cycle de vie des connexions et la distribution des messages.
    """

    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)
        self.metrics = get_business_metrics()

        # Connexions actives
        self._connections: Dict[str, WSConnection] = {}
        self._connections_by_client: Dict[str, List[str]] = {}

        # Statistiques
        self._total_connections = 0
        self._total_disconnections = 0

    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None) -> WSConnection:
        """Accepter une nouvelle connexion"""
        await websocket.accept()

        connection_id = str(uuid.uuid4())
        connection = WSConnection(websocket, connection_id, client_id)

        # Enregistrer la connexion
        self._connections[connection_id] = connection

        if client_id:
            if client_id not in self._connections_by_client:
                self._connections_by_client[client_id] = []
            self._connections_by_client[client_id].append(connection_id)

        self._total_connections += 1

        # Envoyer message de confirmation
        await self.send_to_connection(connection_id, WSMessage(
            type=MessageType.CONNECTED,
            data={
                "connection_id": connection_id,
                "client_id": connection.client_id,
                "server_time": datetime.utcnow().isoformat()
            }
        ))

        self.logger.info(
            f"WebSocket connection established",
            connection_id=connection_id,
            client_id=connection.client_id
        )

        # Métriques
        self.metrics.collector.increment_counter("websocket.connections")

        return connection

    async def disconnect(self, connection_id: str):
        """Fermer une connexion"""
        if connection_id not in self._connections:
            return

        connection = self._connections[connection_id]

        # Nettoyer les subscriptions
        for channel, subscription_id in connection.subscriptions.items():
            await self._unsubscribe_from_channel(connection, channel)

        # Supprimer de la liste des connexions
        del self._connections[connection_id]

        if connection.client_id in self._connections_by_client:
            self._connections_by_client[connection.client_id].remove(connection_id)
            if not self._connections_by_client[connection.client_id]:
                del self._connections_by_client[connection.client_id]

        self._total_disconnections += 1

        self.logger.info(
            f"WebSocket connection closed",
            connection_id=connection_id,
            client_id=connection.client_id,
            duration_seconds=(datetime.utcnow() - connection.connected_at).total_seconds()
        )

        # Métriques
        self.metrics.collector.increment_counter("websocket.disconnections")

    async def send_to_connection(self, connection_id: str, message: WSMessage) -> bool:
        """Envoyer un message à une connexion spécifique"""
        if connection_id not in self._connections:
            return False

        connection = self._connections[connection_id]

        if not connection.is_active:
            await self.disconnect(connection_id)
            return False

        try:
            message_json = message.json()
            await connection.websocket.send_text(message_json)

            # Statistiques
            connection.messages_sent += 1
            connection.bytes_sent += len(message_json.encode())

            return True

        except Exception as e:
            self.logger.error(f"Error sending message to {connection_id}: {e}")
            await self.disconnect(connection_id)
            return False

    async def send_to_client(self, client_id: str, message: WSMessage) -> int:
        """Envoyer un message à toutes les connexions d'un client"""
        if client_id not in self._connections_by_client:
            return 0

        sent_count = 0
        connection_ids = self._connections_by_client[client_id].copy()

        for connection_id in connection_ids:
            if await self.send_to_connection(connection_id, message):
                sent_count += 1

        return sent_count

    async def broadcast_to_channel(self, channel: str, message: WSMessage) -> int:
        """Diffuser un message à toutes les connexions d'un channel"""
        sent_count = 0

        for connection in self._connections.values():
            if channel in connection.channels:
                if await self.send_to_connection(connection.connection_id, message):
                    sent_count += 1

        return sent_count

    async def broadcast_to_all(self, message: WSMessage) -> int:
        """Diffuser un message à toutes les connexions"""
        sent_count = 0

        for connection_id in list(self._connections.keys()):
            if await self.send_to_connection(connection_id, message):
                sent_count += 1

        return sent_count

    def get_connection(self, connection_id: str) -> Optional[WSConnection]:
        """Obtenir une connexion par ID"""
        return self._connections.get(connection_id)

    def get_connections_by_client(self, client_id: str) -> List[WSConnection]:
        """Obtenir toutes les connexions d'un client"""
        if client_id not in self._connections_by_client:
            return []

        return [
            self._connections[conn_id]
            for conn_id in self._connections_by_client[client_id]
            if conn_id in self._connections
        ]

    async def _unsubscribe_from_channel(self, connection: WSConnection, channel: str):
        """Se désabonner d'un channel"""
        if channel not in connection.subscriptions:
            return

        subscription_id = connection.subscriptions[channel]

        # Gérer selon le type de channel
        if channel.startswith("market_data"):
            streaming_service = get_streaming_service()
            await streaming_service.unsubscribe(subscription_id)

        # Nettoyer
        del connection.subscriptions[channel]
        connection.channels.discard(channel)

    def get_statistics(self) -> Dict[str, Any]:
        """Obtenir les statistiques des connexions"""
        active_connections = len(self._connections)
        total_channels = sum(len(conn.channels) for conn in self._connections.values())
        total_subscriptions = sum(len(conn.subscriptions) for conn in self._connections.values())

        authenticated_connections = len([
            conn for conn in self._connections.values() if conn.authenticated
        ])

        return {
            "active_connections": active_connections,
            "total_connections": self._total_connections,
            "total_disconnections": self._total_disconnections,
            "authenticated_connections": authenticated_connections,
            "total_channels": total_channels,
            "total_subscriptions": total_subscriptions,
            "unique_clients": len(self._connections_by_client)
        }


class WebSocketManager:
    """
    Gestionnaire principal des WebSockets.
    Orchestre les connexions et la distribution des données.
    """

    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)
        self.metrics = get_business_metrics()
        self.tracer = get_tracer()

        self.connection_manager = ConnectionManager()

        # Services
        self.streaming_service = get_streaming_service()
        self.market_data_pipeline = get_market_data_pipeline()
        self.event_bus = get_event_bus()

        # Configuration
        self._ping_interval = 30  # seconds
        self._ping_task: Optional[asyncio.Task] = None

    async def start(self):
        """Démarrer le gestionnaire WebSocket"""
        # Démarrer la tâche de ping
        self._ping_task = asyncio.create_task(self._ping_connections())

        self.logger.info("WebSocket manager started")

    async def stop(self):
        """Arrêter le gestionnaire WebSocket"""
        if self._ping_task:
            self._ping_task.cancel()

        # Fermer toutes les connexions
        for connection_id in list(self.connection_manager._connections.keys()):
            await self.connection_manager.disconnect(connection_id)

        self.logger.info("WebSocket manager stopped")

    async def handle_connection(self, websocket: WebSocket, client_id: Optional[str] = None):
        """Gérer une nouvelle connexion WebSocket"""
        connection = await self.connection_manager.connect(websocket, client_id)

        try:
            while connection.is_active:
                # Recevoir message
                data = await websocket.receive_text()
                connection.messages_received += 1
                connection.bytes_received += len(data.encode())

                try:
                    message_data = json.loads(data)
                    await self._handle_message(connection, message_data)
                except json.JSONDecodeError:
                    await self._send_error(connection, "Invalid JSON")
                except Exception as e:
                    await self._send_error(connection, f"Message processing error: {str(e)}")

        except WebSocketDisconnect:
            self.logger.info(f"WebSocket client disconnected: {connection.connection_id}")
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
        finally:
            await self.connection_manager.disconnect(connection.connection_id)

    @trace("websocket.handle_message")
    async def _handle_message(self, connection: WSConnection, message_data: Dict[str, Any]):
        """Traiter un message reçu"""
        message_type = message_data.get("type")

        if message_type == MessageType.PING.value:
            await self._handle_ping(connection)

        elif message_type == MessageType.SUBSCRIBE.value:
            await self._handle_subscribe(connection, message_data)

        elif message_type == MessageType.UNSUBSCRIBE.value:
            await self._handle_unsubscribe(connection, message_data)

        elif message_type == MessageType.AUTH.value:
            await self._handle_auth(connection, message_data)

        else:
            await self._send_error(connection, f"Unknown message type: {message_type}")

    async def _handle_ping(self, connection: WSConnection):
        """Traiter un ping"""
        connection.last_ping = datetime.utcnow()
        await self.connection_manager.send_to_connection(
            connection.connection_id,
            WSMessage(type=MessageType.PONG)
        )

    async def _handle_subscribe(self, connection: WSConnection, message_data: Dict[str, Any]):
        """Traiter une subscription"""
        try:
            channel = message_data.get("channel")
            params = message_data.get("params", {})

            if not channel:
                await self._send_error(connection, "Channel is required")
                return

            if channel in connection.channels:
                await self._send_error(connection, f"Already subscribed to {channel}")
                return

            # Gérer selon le type de channel
            subscription_id = None

            if channel == "market_data":
                subscription_id = await self._subscribe_market_data(connection, params)
            elif channel == "events":
                subscription_id = await self._subscribe_events(connection, params)
            elif channel == "orders":
                subscription_id = await self._subscribe_orders(connection, params)
            else:
                await self._send_error(connection, f"Unknown channel: {channel}")
                return

            if subscription_id:
                connection.subscriptions[channel] = subscription_id
                connection.channels.add(channel)

                await self.connection_manager.send_to_connection(
                    connection.connection_id,
                    WSMessage(
                        type=MessageType.SUBSCRIBED,
                        data={
                            "channel": channel,
                            "subscription_id": subscription_id
                        }
                    )
                )

                self.logger.info(
                    f"WebSocket subscribed to {channel}",
                    connection_id=connection.connection_id,
                    channel=channel,
                    subscription_id=subscription_id
                )

        except Exception as e:
            await self._send_error(connection, f"Subscription error: {str(e)}")

    async def _handle_unsubscribe(self, connection: WSConnection, message_data: Dict[str, Any]):
        """Traiter une unsubscription"""
        channel = message_data.get("channel")

        if not channel or channel not in connection.channels:
            await self._send_error(connection, f"Not subscribed to {channel}")
            return

        await self.connection_manager._unsubscribe_from_channel(connection, channel)

        await self.connection_manager.send_to_connection(
            connection.connection_id,
            WSMessage(
                type=MessageType.UNSUBSCRIBED,
                data={"channel": channel}
            )
        )

    async def _handle_auth(self, connection: WSConnection, message_data: Dict[str, Any]):
        """Traiter l'authentification"""
        # Authentification simplifiée pour l'instant
        token = message_data.get("token")
        if token:  # Validation simplifiée
            connection.authenticated = True
            connection.user_id = "user_" + str(uuid.uuid4())

            await self.connection_manager.send_to_connection(
                connection.connection_id,
                WSMessage(
                    type=MessageType.CONNECTED,
                    data={
                        "authenticated": True,
                        "user_id": connection.user_id
                    }
                )
            )

    async def _subscribe_market_data(self, connection: WSConnection, params: Dict[str, Any]) -> str:
        """S'abonner aux données de marché"""
        level = SubscriptionLevel(params.get("level", "all"))

        subscription_id = self.streaming_service.subscribe_callback(
            client_id=connection.client_id,
            callback=lambda data: asyncio.create_task(
                self._send_market_data(connection.connection_id, data)
            ),
            level=level,
            **params.get("filter_criteria", {})
        )

        return subscription_id

    async def _subscribe_events(self, connection: WSConnection, params: Dict[str, Any]) -> str:
        """S'abonner aux événements"""
        # Créer un handler pour les événements
        def event_handler(event: Event):
            asyncio.create_task(
                self._send_event(connection.connection_id, event)
            )

        # S'abonner aux événements
        event_types = params.get("event_types", ["*"])
        for event_type in event_types:
            self.event_bus.subscribe_callback(
                event_type,
                event_handler,
                f"ws_{connection.connection_id}_{event_type}"
            )

        return f"events_{connection.connection_id}"

    async def _subscribe_orders(self, connection: WSConnection, params: Dict[str, Any]) -> str:
        """S'abonner aux mises à jour d'ordres"""
        # Pour l'instant, subscription simple
        return f"orders_{connection.connection_id}"

    async def _send_market_data(self, connection_id: str, data_point: MarketDataPoint):
        """Envoyer des données de marché via WebSocket"""
        message = WSMessage(
            type=MessageType.DATA,
            data={
                "channel": "market_data",
                "symbol": data_point.symbol,
                "data_type": data_point.data_type.value,
                "timestamp": data_point.timestamp.isoformat(),
                "data": data_point.data,
                "provider": data_point.provider,
                "quality": data_point.quality.value
            }
        )

        await self.connection_manager.send_to_connection(connection_id, message)

    async def _send_event(self, connection_id: str, event: Event):
        """Envoyer un événement via WebSocket"""
        message = WSMessage(
            type=MessageType.DATA,
            data={
                "channel": "events",
                "event_type": event.event_type,
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "data": event.data,
                "correlation_id": event.correlation_id
            }
        )

        await self.connection_manager.send_to_connection(connection_id, message)

    async def _send_error(self, connection: WSConnection, error_message: str):
        """Envoyer un message d'erreur"""
        await self.connection_manager.send_to_connection(
            connection.connection_id,
            WSMessage(
                type=MessageType.ERROR,
                data={"error": error_message}
            )
        )

    async def _ping_connections(self):
        """Envoyer des pings périodiques"""
        while True:
            try:
                await asyncio.sleep(self._ping_interval)

                # Envoyer ping à toutes les connexions
                ping_message = WSMessage(type=MessageType.PING)
                await self.connection_manager.broadcast_to_all(ping_message)

                # Nettoyer les connexions inactives
                now = datetime.utcnow()
                inactive_connections = []

                for connection in self.connection_manager._connections.values():
                    if (now - connection.last_ping).total_seconds() > self._ping_interval * 2:
                        inactive_connections.append(connection.connection_id)

                for connection_id in inactive_connections:
                    await self.connection_manager.disconnect(connection_id)

            except Exception as e:
                self.logger.error(f"Error in ping task: {e}")


# Instance globale
_global_websocket_manager: Optional[WebSocketManager] = None


def get_websocket_manager() -> WebSocketManager:
    """Obtenir l'instance globale du gestionnaire WebSocket"""
    global _global_websocket_manager
    if _global_websocket_manager is None:
        _global_websocket_manager = WebSocketManager()
    return _global_websocket_manager