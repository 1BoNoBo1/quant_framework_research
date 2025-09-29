"""
📋 Order Service
Service pour la gestion des ordres de trading
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal

from qframe.core.container import injectable
from qframe.api.services.base_service import BaseService

logger = logging.getLogger(__name__)


@injectable
class OrderService(BaseService):
    """Service de gestion des ordres."""

    def __init__(self):
        super().__init__()
        self._orders: Dict[str, Dict[str, Any]] = {}
        self._order_counter = 0

    async def create_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Crée un nouvel ordre."""
        try:
            self._order_counter += 1
            order_id = f"order_{self._order_counter:06d}"

            order = {
                "id": order_id,
                "symbol": order_data["symbol"],
                "side": order_data["side"],
                "type": order_data["type"],
                "quantity": order_data["quantity"],
                "price": order_data.get("price"),
                "stop_price": order_data.get("stop_price"),
                "status": "PENDING",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "filled_quantity": 0.0,
                "remaining_quantity": order_data["quantity"],
                "average_price": None,
                "commission": 0.0,
                "metadata": order_data.get("metadata", {})
            }

            self._orders[order_id] = order
            logger.info(f"Order created: {order_id}")

            # Simuler l'exécution immédiate pour les ordres market
            if order["type"] == "MARKET":
                await self._simulate_fill(order_id)

            return order

        except Exception as e:
            logger.error(f"Error creating order: {e}")
            raise

    async def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Récupère un ordre par ID."""
        return self._orders.get(order_id)

    async def get_orders(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        page: int = 1,
        per_page: int = 20
    ) -> tuple[List[Dict[str, Any]], int]:
        """Récupère la liste des ordres avec filtres."""
        try:
            orders = list(self._orders.values())

            # Filtrage
            if symbol:
                orders = [o for o in orders if o["symbol"] == symbol]
            if status:
                orders = [o for o in orders if o["status"] == status]

            # Tri par date de création (plus récent en premier)
            orders.sort(key=lambda x: x["created_at"], reverse=True)

            total = len(orders)

            # Pagination
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            paginated_orders = orders[start_idx:end_idx]

            return paginated_orders, total

        except Exception as e:
            logger.error(f"Error fetching orders: {e}")
            raise

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Annule un ordre."""
        try:
            order = self._orders.get(order_id)
            if not order:
                raise ValueError(f"Order {order_id} not found")

            if order["status"] in ["FILLED", "CANCELLED"]:
                raise ValueError(f"Cannot cancel order in status {order['status']}")

            order["status"] = "CANCELLED"
            order["updated_at"] = datetime.utcnow()

            logger.info(f"Order cancelled: {order_id}")
            return order

        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            raise

    async def cancel_all_orders(
        self,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Annule tous les ordres (optionnellement pour un symbole)."""
        try:
            cancelled_orders = []

            for order_id, order in self._orders.items():
                if order["status"] not in ["FILLED", "CANCELLED"]:
                    if symbol is None or order["symbol"] == symbol:
                        order["status"] = "CANCELLED"
                        order["updated_at"] = datetime.utcnow()
                        cancelled_orders.append(order)

            logger.info(f"Cancelled {len(cancelled_orders)} orders")
            return cancelled_orders

        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            raise

    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Récupère l'historique des ordres."""
        try:
            orders = list(self._orders.values())

            # Filtrage
            if symbol:
                orders = [o for o in orders if o["symbol"] == symbol]
            if start_date:
                orders = [o for o in orders if o["created_at"] >= start_date]
            if end_date:
                orders = [o for o in orders if o["created_at"] <= end_date]

            # Tri par date
            orders.sort(key=lambda x: x["created_at"], reverse=True)

            return orders

        except Exception as e:
            logger.error(f"Error fetching order history: {e}")
            raise

    async def _simulate_fill(self, order_id: str) -> None:
        """Simule l'exécution d'un ordre."""
        try:
            order = self._orders[order_id]
            if order["type"] == "MARKET":
                # Simuler un prix d'exécution proche du marché
                simulated_price = 50000.0  # Prix BTC simulé

                order["status"] = "FILLED"
                order["filled_quantity"] = order["quantity"]
                order["remaining_quantity"] = 0.0
                order["average_price"] = simulated_price
                order["commission"] = order["quantity"] * simulated_price * 0.001  # 0.1% commission
                order["updated_at"] = datetime.utcnow()

                logger.info(f"Order filled: {order_id} at {simulated_price}")

        except Exception as e:
            logger.error(f"Error simulating fill for order {order_id}: {e}")

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Récupère les ordres ouverts."""
        try:
            open_orders = [
                order for order in self._orders.values()
                if order["status"] in ["PENDING", "PARTIALLY_FILLED"]
            ]

            if symbol:
                open_orders = [o for o in open_orders if o["symbol"] == symbol]

            return open_orders

        except Exception as e:
            logger.error(f"Error fetching open orders: {e}")
            raise

    async def modify_order(
        self,
        order_id: str,
        new_quantity: Optional[float] = None,
        new_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Modifie un ordre existant."""
        try:
            order = self._orders.get(order_id)
            if not order:
                raise ValueError(f"Order {order_id} not found")

            if order["status"] not in ["PENDING"]:
                raise ValueError(f"Cannot modify order in status {order['status']}")

            if new_quantity is not None:
                order["quantity"] = new_quantity
                order["remaining_quantity"] = new_quantity - order["filled_quantity"]

            if new_price is not None:
                order["price"] = new_price

            order["updated_at"] = datetime.utcnow()

            logger.info(f"Order modified: {order_id}")
            return order

        except Exception as e:
            logger.error(f"Error modifying order {order_id}: {e}")
            raise