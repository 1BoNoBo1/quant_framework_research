"""
Infrastructure Layer: Order Execution Adapter
============================================

Adaptateur pour l'exécution d'ordres via des courtiers externes.
Interface entre le domaine et les services de courtage.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Protocol
from decimal import Decimal
from datetime import datetime
import logging

from qframe.domain.entities.order import (
    Order, OrderExecution, OrderStatus, OrderSide, OrderType, ExecutionVenue
)
from qframe.domain.services.execution_service import VenueQuote, ExecutionPlan


logger = logging.getLogger(__name__)


class BrokerAdapter(Protocol):
    """Protocol pour les adaptateurs de courtier"""

    async def submit_order(self, order: Order) -> Dict[str, any]:
        """Soumet un ordre au courtier"""
        ...

    async def cancel_order(self, order_id: str, broker_order_id: str) -> bool:
        """Annule un ordre chez le courtier"""
        ...

    async def get_order_status(self, broker_order_id: str) -> Dict[str, any]:
        """Récupère le statut d'un ordre chez le courtier"""
        ...

    async def get_market_data(self, symbol: str) -> VenueQuote:
        """Récupère les données de marché pour un symbole"""
        ...


class OrderExecutionAdapter:
    """
    Adaptateur principal pour l'exécution d'ordres.
    Coordonne l'exécution via multiple courtiers.
    """

    def __init__(self):
        self._brokers: Dict[ExecutionVenue, BrokerAdapter] = {}
        self._order_mappings: Dict[str, Dict[str, str]] = {}  # order_id -> {venue -> broker_order_id}

    def register_broker(self, venue: ExecutionVenue, adapter: BrokerAdapter) -> None:
        """Enregistre un adaptateur de courtier pour une venue"""
        self._brokers[venue] = adapter
        logger.info(f"🏦 Broker adapter registered for venue: {venue.value}")

    async def execute_order_plan(self, order: Order, execution_plan: ExecutionPlan) -> List[OrderExecution]:
        """
        Exécute un plan d'exécution d'ordre via les courtiers appropriés.

        Args:
            order: L'ordre à exécuter
            execution_plan: Le plan d'exécution avec routing et allocations

        Returns:
            Liste des exécutions réalisées
        """
        executions = []

        if order.id not in self._order_mappings:
            self._order_mappings[order.id] = {}

        for allocation in execution_plan.allocations:
            venue = allocation.venue

            if venue not in self._brokers:
                logger.error(f"❌ No broker adapter for venue: {venue.value}")
                continue

            broker = self._brokers[venue]

            try:
                # Créer un ordre partiel pour cette allocation
                partial_order = Order(
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.order_type,
                    quantity=allocation.quantity,
                    price=allocation.price,
                    strategy_id=order.strategy_id,
                    parent_order_id=order.id,
                    tags={**order.tags, "allocation_id": allocation.id, "venue": venue.value}
                )

                # Soumettre l'ordre au courtier
                broker_response = await broker.submit_order(partial_order)
                broker_order_id = broker_response.get("order_id")

                if broker_order_id:
                    self._order_mappings[order.id][venue.value] = broker_order_id

                    # Surveiller l'exécution
                    venue_executions = await self._monitor_execution(
                        partial_order, broker_order_id, broker, venue
                    )
                    executions.extend(venue_executions)

                else:
                    logger.error(f"❌ Failed to submit order to {venue.value}: {broker_response}")

            except Exception as e:
                logger.error(f"❌ Error executing order on {venue.value}: {e}")

        return executions

    async def _monitor_execution(
        self,
        order: Order,
        broker_order_id: str,
        broker: BrokerAdapter,
        venue: ExecutionVenue
    ) -> List[OrderExecution]:
        """
        Surveille l'exécution d'un ordre chez un courtier.

        Args:
            order: L'ordre à surveiller
            broker_order_id: ID de l'ordre chez le courtier
            broker: L'adaptateur du courtier
            venue: La venue d'exécution

        Returns:
            Liste des exécutions détectées
        """
        executions = []

        try:
            # Récupérer le statut de l'ordre
            status_response = await broker.get_order_status(broker_order_id)

            if not status_response:
                logger.warning(f"⚠️ No status response for order {broker_order_id}")
                return executions

            # Parser les exécutions du courtier
            broker_executions = status_response.get("executions", [])

            for exec_data in broker_executions:
                execution = OrderExecution(
                    price=Decimal(str(exec_data.get("price", 0))),
                    quantity=Decimal(str(exec_data.get("quantity", 0))),
                    timestamp=datetime.fromisoformat(exec_data.get("timestamp")) if exec_data.get("timestamp") else datetime.utcnow(),
                    venue=venue,
                    fee=Decimal(str(exec_data.get("fee", 0))) if exec_data.get("fee") else None,
                    commission=Decimal(str(exec_data.get("commission", 0))) if exec_data.get("commission") else None
                )
                executions.append(execution)

        except Exception as e:
            logger.error(f"❌ Error monitoring execution for {broker_order_id}: {e}")

        return executions

    async def cancel_order(self, order_id: str, venue: Optional[ExecutionVenue] = None) -> Dict[str, bool]:
        """
        Annule un ordre sur toutes les venues ou une venue spécifique.

        Args:
            order_id: ID de l'ordre à annuler
            venue: Venue spécifique (None = toutes les venues)

        Returns:
            Dictionnaire venue -> succès de l'annulation
        """
        results = {}

        if order_id not in self._order_mappings:
            logger.warning(f"⚠️ No broker mappings found for order {order_id}")
            return results

        venues_to_cancel = [venue] if venue else list(self._order_mappings[order_id].keys())

        for venue_key in venues_to_cancel:
            try:
                venue_enum = ExecutionVenue(venue_key)
                if venue_enum not in self._brokers:
                    results[venue_key] = False
                    continue

                broker_order_id = self._order_mappings[order_id].get(venue_key)
                if not broker_order_id:
                    results[venue_key] = False
                    continue

                broker = self._brokers[venue_enum]
                success = await broker.cancel_order(order_id, broker_order_id)
                results[venue_key] = success

                if success:
                    logger.info(f"✅ Order {order_id} cancelled on {venue_key}")
                else:
                    logger.warning(f"⚠️ Failed to cancel order {order_id} on {venue_key}")

            except Exception as e:
                logger.error(f"❌ Error cancelling order {order_id} on {venue_key}: {e}")
                results[venue_key] = False

        return results

    async def get_market_data(self, symbol: str, venues: Optional[List[ExecutionVenue]] = None) -> Dict[ExecutionVenue, VenueQuote]:
        """
        Récupère les données de marché pour un symbole sur plusieurs venues.

        Args:
            symbol: Symbole à trader
            venues: Venues spécifiques (None = toutes les venues disponibles)

        Returns:
            Dictionnaire venue -> quote
        """
        quotes = {}
        venues_to_query = venues or list(self._brokers.keys())

        for venue in venues_to_query:
            if venue not in self._brokers:
                continue

            try:
                broker = self._brokers[venue]
                quote = await broker.get_market_data(symbol)
                quotes[venue] = quote

            except Exception as e:
                logger.error(f"❌ Error getting market data for {symbol} on {venue.value}: {e}")

        return quotes

    async def get_order_status_all_venues(self, order_id: str) -> Dict[str, Dict[str, any]]:
        """
        Récupère le statut d'un ordre sur toutes les venues où il a été placé.

        Args:
            order_id: ID de l'ordre

        Returns:
            Dictionnaire venue -> statut de l'ordre
        """
        statuses = {}

        if order_id not in self._order_mappings:
            return statuses

        for venue_key, broker_order_id in self._order_mappings[order_id].items():
            try:
                venue = ExecutionVenue(venue_key)
                if venue not in self._brokers:
                    continue

                broker = self._brokers[venue]
                status = await broker.get_order_status(broker_order_id)
                statuses[venue_key] = status

            except Exception as e:
                logger.error(f"❌ Error getting order status for {order_id} on {venue_key}: {e}")

        return statuses

    def get_supported_venues(self) -> List[ExecutionVenue]:
        """Retourne la liste des venues supportées"""
        return list(self._brokers.keys())

    def is_venue_available(self, venue: ExecutionVenue) -> bool:
        """Vérifie si une venue est disponible"""
        return venue in self._brokers

    async def health_check(self) -> Dict[str, any]:
        """Vérifie la santé de tous les adaptateurs de courtier"""
        health_status = {
            "status": "healthy",
            "total_venues": len(self._brokers),
            "active_orders": len(self._order_mappings),
            "venues": {}
        }

        for venue, broker in self._brokers.items():
            try:
                # Supposer qu'il y a une méthode health_check sur les brokers
                if hasattr(broker, 'health_check'):
                    venue_health = await broker.health_check()
                else:
                    venue_health = {"status": "unknown", "note": "No health check method"}

                health_status["venues"][venue.value] = venue_health

            except Exception as e:
                health_status["venues"][venue.value] = {
                    "status": "error",
                    "error": str(e)
                }

        # Déterminer le statut global
        venue_statuses = [v.get("status") for v in health_status["venues"].values()]
        if any(status == "error" for status in venue_statuses):
            health_status["status"] = "degraded"
        elif any(status != "healthy" for status in venue_statuses):
            health_status["status"] = "warning"

        return health_status