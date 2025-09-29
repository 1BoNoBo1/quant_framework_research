"""
⚡ Real Time WebSocket Handler
Gestionnaire WebSocket pour les données temps réel
"""

import asyncio
import json
import logging
from typing import Dict, Set, Optional
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime

from qframe.core.container import injectable
from qframe.api.services.real_time_service import RealTimeService, EventType, RealTimeEvent

logger = logging.getLogger(__name__)


@injectable
class RealTimeWebSocketHandler:
    """Gestionnaire WebSocket pour les données temps réel."""

    def __init__(self, real_time_service: RealTimeService):
        self.real_time_service = real_time_service
        self.active_connections: Dict[str, WebSocket] = {}
        self.client_subscriptions: Dict[str, Set[EventType]] = {}

    async def handle_connection(self, websocket: WebSocket):
        """Gère une nouvelle connexion WebSocket."""
        client_id = f"ws_{datetime.now().timestamp()}"

        try:
            # Accepter la connexion
            await websocket.accept()
            self.active_connections[client_id] = websocket

            logger.info(f"WebSocket client {client_id} connected")

            # Envoyer un message de bienvenue
            await self._send_message(websocket, {
                "type": "connection_established",
                "client_id": client_id,
                "timestamp": datetime.now().isoformat()
            })

            # Écouter les messages du client
            await self._listen_to_client(client_id, websocket)

        except WebSocketDisconnect:
            logger.info(f"WebSocket client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Error handling WebSocket connection: {e}")
        finally:
            # Nettoyer la connexion
            await self._cleanup_connection(client_id)

    async def _listen_to_client(self, client_id: str, websocket: WebSocket):
        """Écoute les messages du client."""
        while True:
            try:
                # Recevoir un message
                message = await websocket.receive_text()
                data = json.loads(message)

                # Traiter le message
                await self._handle_client_message(client_id, websocket, data)

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await self._send_error(websocket, "Invalid JSON format")
            except Exception as e:
                logger.error(f"Error handling client message: {e}")
                await self._send_error(websocket, "Internal server error")

    async def _handle_client_message(self, client_id: str, websocket: WebSocket, data: Dict):
        """Traite un message du client."""
        message_type = data.get("type")

        if message_type == "subscribe":
            await self._handle_subscription(client_id, websocket, data)
        elif message_type == "unsubscribe":
            await self._handle_unsubscription(client_id, websocket, data)
        elif message_type == "ping":
            await self._handle_ping(websocket)
        else:
            await self._send_error(websocket, f"Unknown message type: {message_type}")

    async def _handle_subscription(self, client_id: str, websocket: WebSocket, data: Dict):
        """Gère une demande d'abonnement."""
        try:
            event_types_str = data.get("event_types", [])
            symbols = data.get("symbols")

            # Convertir les types d'événements
            event_types = []
            for et_str in event_types_str:
                try:
                    event_types.append(EventType(et_str))
                except ValueError:
                    await self._send_error(websocket, f"Invalid event type: {et_str}")
                    return

            # S'abonner via le service temps réel
            success = await self.real_time_service.subscribe(
                client_id=client_id,
                event_types=event_types,
                symbols=symbols,
                callback=lambda event: asyncio.create_task(self._send_event_to_client(client_id, event))
            )

            if success:
                # Stocker les abonnements
                self.client_subscriptions[client_id] = set(event_types)

                await self._send_message(websocket, {
                    "type": "subscription_success",
                    "event_types": [et.value for et in event_types],
                    "symbols": symbols,
                    "timestamp": datetime.now().isoformat()
                })

                logger.info(f"Client {client_id} subscribed to {len(event_types)} event types")
            else:
                await self._send_error(websocket, "Failed to create subscription")

        except Exception as e:
            logger.error(f"Error handling subscription: {e}")
            await self._send_error(websocket, "Subscription failed")

    async def _handle_unsubscription(self, client_id: str, websocket: WebSocket, data: Dict):
        """Gère une demande de désabonnement."""
        try:
            # Se désabonner via le service temps réel
            success = await self.real_time_service.unsubscribe(client_id)

            if success:
                # Supprimer les abonnements locaux
                if client_id in self.client_subscriptions:
                    del self.client_subscriptions[client_id]

                await self._send_message(websocket, {
                    "type": "unsubscription_success",
                    "timestamp": datetime.now().isoformat()
                })

                logger.info(f"Client {client_id} unsubscribed")
            else:
                await self._send_error(websocket, "Failed to unsubscribe")

        except Exception as e:
            logger.error(f"Error handling unsubscription: {e}")
            await self._send_error(websocket, "Unsubscription failed")

    async def _handle_ping(self, websocket: WebSocket):
        """Gère un ping du client."""
        await self._send_message(websocket, {
            "type": "pong",
            "timestamp": datetime.now().isoformat()
        })

    async def _send_event_to_client(self, client_id: str, event: RealTimeEvent):
        """Envoie un événement à un client spécifique."""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await self._send_message(websocket, {
                    "type": "event",
                    "event_type": event.type.value,
                    "data": event.data,
                    "symbol": event.symbol,
                    "timestamp": event.timestamp.isoformat(),
                    "source": event.source
                })
            except Exception as e:
                logger.error(f"Error sending event to client {client_id}: {e}")

    async def _send_message(self, websocket: WebSocket, message: Dict):
        """Envoie un message via WebSocket."""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            raise

    async def _send_error(self, websocket: WebSocket, error_message: str):
        """Envoie un message d'erreur."""
        await self._send_message(websocket, {
            "type": "error",
            "message": error_message,
            "timestamp": datetime.now().isoformat()
        })

    async def _cleanup_connection(self, client_id: str):
        """Nettoie une connexion fermée."""
        # Supprimer de la liste des connexions actives
        if client_id in self.active_connections:
            del self.active_connections[client_id]

        # Se désabonner des événements
        if client_id in self.client_subscriptions:
            await self.real_time_service.unsubscribe(client_id)
            del self.client_subscriptions[client_id]

        logger.info(f"Cleaned up connection for client {client_id}")

    async def broadcast_event(self, event: RealTimeEvent):
        """Diffuse un événement à tous les clients connectés."""
        disconnected_clients = []

        for client_id, websocket in self.active_connections.items():
            try:
                # Vérifier si le client est abonné à ce type d'événement
                if client_id in self.client_subscriptions:
                    subscribed_types = self.client_subscriptions[client_id]
                    if event.type in subscribed_types:
                        await self._send_event_to_client(client_id, event)

            except Exception as e:
                logger.error(f"Error broadcasting to client {client_id}: {e}")
                disconnected_clients.append(client_id)

        # Nettoyer les clients déconnectés
        for client_id in disconnected_clients:
            await self._cleanup_connection(client_id)

    def get_connection_stats(self) -> Dict:
        """Retourne les statistiques des connexions."""
        return {
            "active_connections": len(self.active_connections),
            "total_subscriptions": len(self.client_subscriptions),
            "connected_clients": list(self.active_connections.keys())
        }