"""
Infrastructure Layer: Memory Order Repository
==========================================

Implémentation en mémoire du repository des ordres.
Utilisée pour le développement et les tests.
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional
from collections import defaultdict

from qframe.domain.entities.order import Order, OrderStatus, OrderSide, OrderType
from qframe.domain.repositories.order_repository import OrderRepository


class MemoryOrderRepository(OrderRepository):
    """Repository en mémoire pour les ordres"""

    def __init__(self):
        self._orders: Dict[str, Order] = {}
        self._orders_by_symbol: Dict[str, List[str]] = defaultdict(list)
        self._orders_by_status: Dict[OrderStatus, List[str]] = defaultdict(list)
        self._orders_by_strategy: Dict[str, List[str]] = defaultdict(list)

    async def save(self, order: Order) -> None:
        """Sauvegarde un ordre"""
        old_order = self._orders.get(order.id)

        # Mise à jour des indexes
        if old_order:
            self._remove_from_indexes(old_order)

        self._orders[order.id] = order
        self._add_to_indexes(order)

    async def find_by_id(self, order_id: str) -> Optional[Order]:
        """Trouve un ordre par son ID"""
        return self._orders.get(order_id)

    async def find_by_symbol(self, symbol: str) -> List[Order]:
        """Trouve tous les ordres pour un symbole"""
        order_ids = self._orders_by_symbol.get(symbol, [])
        return [self._orders[order_id] for order_id in order_ids if order_id in self._orders]

    async def find_by_status(self, status: OrderStatus) -> List[Order]:
        """Trouve tous les ordres par statut"""
        order_ids = self._orders_by_status.get(status, [])
        return [self._orders[order_id] for order_id in order_ids if order_id in self._orders]

    async def find_active_orders(self) -> List[Order]:
        """Trouve tous les ordres actifs (PENDING, PARTIAL_FILLED)"""
        active_statuses = [OrderStatus.PENDING, OrderStatus.PARTIAL_FILLED]
        active_orders = []
        for status in active_statuses:
            active_orders.extend(await self.find_by_status(status))
        return active_orders

    async def find_by_strategy_id(self, strategy_id: str) -> List[Order]:
        """Trouve tous les ordres d'une stratégie"""
        order_ids = self._orders_by_strategy.get(strategy_id, [])
        return [self._orders[order_id] for order_id in order_ids if order_id in self._orders]

    async def find_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Order]:
        """Trouve tous les ordres dans une plage de dates"""
        return [
            order for order in self._orders.values()
            if order.created_at and start_date <= order.created_at <= end_date
        ]

    async def find_by_criteria(
        self,
        symbol: Optional[str] = None,
        status: Optional[OrderStatus] = None,
        side: Optional[OrderSide] = None,
        order_type: Optional[OrderType] = None,
        strategy_id: Optional[str] = None,
        min_quantity: Optional[Decimal] = None,
        max_quantity: Optional[Decimal] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Order]:
        """Trouve les ordres selon des critères multiples"""
        orders = list(self._orders.values())

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        if status:
            orders = [o for o in orders if o.status == status]
        if side:
            orders = [o for o in orders if o.side == side]
        if order_type:
            orders = [o for o in orders if o.order_type == order_type]
        if strategy_id:
            orders = [o for o in orders if o.strategy_id == strategy_id]
        if min_quantity:
            orders = [o for o in orders if o.quantity >= min_quantity]
        if max_quantity:
            orders = [o for o in orders if o.quantity <= max_quantity]
        if start_date:
            orders = [o for o in orders if o.created_at and o.created_at >= start_date]
        if end_date:
            orders = [o for o in orders if o.created_at and o.created_at <= end_date]

        return orders

    async def count_by_status(self, status: OrderStatus) -> int:
        """Compte les ordres par statut"""
        return len(self._orders_by_status.get(status, []))

    async def count_by_symbol(self, symbol: str) -> int:
        """Compte les ordres par symbole"""
        return len(self._orders_by_symbol.get(symbol, []))

    async def get_total_volume_by_symbol(self, symbol: str) -> Decimal:
        """Calcule le volume total par symbole"""
        orders = await self.find_by_symbol(symbol)
        return sum(order.quantity for order in orders)

    async def get_average_fill_price(self, order_id: str) -> Optional[Decimal]:
        """Calcule le prix moyen d'exécution d'un ordre"""
        order = await self.find_by_id(order_id)
        if not order or not order.executions:
            return None

        total_value = sum(exec.price * exec.quantity for exec in order.executions)
        total_quantity = sum(exec.quantity for exec in order.executions)

        return total_value / total_quantity if total_quantity > 0 else None

    async def delete(self, order_id: str) -> bool:
        """Supprime un ordre"""
        order = self._orders.get(order_id)
        if not order:
            return False

        self._remove_from_indexes(order)
        del self._orders[order_id]
        return True

    async def exists(self, order_id: str) -> bool:
        """Vérifie si un ordre existe"""
        return order_id in self._orders

    async def count_all(self) -> int:
        """Compte tous les ordres"""
        return len(self._orders)

    def _add_to_indexes(self, order: Order) -> None:
        """Ajoute un ordre aux index"""
        self._orders_by_symbol[order.symbol].append(order.id)
        self._orders_by_status[order.status].append(order.id)
        if order.strategy_id:
            self._orders_by_strategy[order.strategy_id].append(order.id)

    def _remove_from_indexes(self, order: Order) -> None:
        """Supprime un ordre des index"""
        if order.symbol in self._orders_by_symbol:
            try:
                self._orders_by_symbol[order.symbol].remove(order.id)
            except ValueError:
                pass

        if order.status in self._orders_by_status:
            try:
                self._orders_by_status[order.status].remove(order.id)
            except ValueError:
                pass

        if order.strategy_id and order.strategy_id in self._orders_by_strategy:
            try:
                self._orders_by_strategy[order.strategy_id].remove(order.id)
            except ValueError:
                pass

    async def clear_all(self) -> None:
        """Supprime tous les ordres (utile pour les tests)"""
        self._orders.clear()
        self._orders_by_symbol.clear()
        self._orders_by_status.clear()
        self._orders_by_strategy.clear()