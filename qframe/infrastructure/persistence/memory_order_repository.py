"""
Infrastructure Layer: Memory Order Repository
==========================================

Implémentation en mémoire du repository des ordres.
Utilisée pour le développement et les tests.
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
from collections import defaultdict

from qframe.domain.entities.order import Order, OrderStatus, OrderSide, OrderType, TimeInForce, OrderPriority
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
        """Trouve tous les ordres actifs (PENDING, PARTIALLY_FILLED)"""
        active_statuses = [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]
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

    async def count_by_status(self) -> Dict[OrderStatus, int]:
        """Compte les ordres par statut"""
        result = {}
        for status in OrderStatus:
            result[status] = len(self._orders_by_status.get(status, []))
        return result

    async def count_by_symbol(self, limit: Optional[int] = None) -> Dict[str, int]:
        """Compte les ordres par symbole"""
        result = {}
        for symbol, order_ids in self._orders_by_symbol.items():
            result[symbol] = len(order_ids)

        # Si limit spécifié, retourner les top N symboles
        if limit:
            sorted_items = sorted(result.items(), key=lambda x: x[1], reverse=True)
            result = dict(sorted_items[:limit])

        return result

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

    # ===== MÉTHODES ABSTRAITES MANQUANTES =====

    async def find_by_client_order_id(self, client_order_id: str) -> Optional[Order]:
        """Trouve un ordre par son ID client"""
        for order in self._orders.values():
            if order.client_order_id == client_order_id:
                return order
        return None

    async def find_by_broker_order_id(self, broker_order_id: str) -> Optional[Order]:
        """Trouve un ordre par son ID broker"""
        for order in self._orders.values():
            if order.broker_order_id == broker_order_id:
                return order
        return None

    async def find_by_portfolio(self, portfolio_id: str) -> List[Order]:
        """Trouve tous les ordres d'un portfolio"""
        return [order for order in self._orders.values() if order.portfolio_id == portfolio_id]

    async def find_by_strategy(self, strategy_id: str) -> List[Order]:
        """Trouve tous les ordres d'une stratégie"""
        return await self.find_by_strategy_id(strategy_id)

    async def find_by_symbol_and_side(self, symbol: str, side: OrderSide) -> List[Order]:
        """Trouve tous les ordres pour un symbole et un côté"""
        symbol_orders = await self.find_by_symbol(symbol)
        return [order for order in symbol_orders if order.side == side]

    async def find_expired_orders(self, current_time: Optional[datetime] = None) -> List[Order]:
        """Trouve tous les ordres expirés"""
        if current_time is None:
            current_time = datetime.utcnow()

        expired_orders = []
        for order in self._orders.values():
            # Check expire_time
            if order.expire_time and current_time >= order.expire_time:
                expired_orders.append(order)
            # Check good_till_date
            elif order.good_till_date and current_time >= order.good_till_date:
                expired_orders.append(order)
            # Check DAY orders
            elif order.time_in_force == TimeInForce.DAY:
                # Si l'ordre a été créé un autre jour, il est expiré
                order_date = order.created_time.date()
                current_date = current_time.date()
                if current_date > order_date:
                    expired_orders.append(order)

        return expired_orders

    async def find_orders_by_priority(self, priority: OrderPriority) -> List[Order]:
        """Trouve tous les ordres avec une priorité donnée"""
        return [order for order in self._orders.values() if order.priority == priority]

    async def find_orders_by_type(self, order_type: OrderType) -> List[Order]:
        """Trouve tous les ordres d'un type donné"""
        return [order for order in self._orders.values() if order.order_type == order_type]

    async def find_orders_by_time_in_force(self, time_in_force: TimeInForce) -> List[Order]:
        """Trouve tous les ordres avec une durée de validité donnée"""
        return [order for order in self._orders.values() if order.time_in_force == time_in_force]

    async def find_parent_orders(self) -> List[Order]:
        """Trouve tous les ordres parents (sans parent_order_id)"""
        return [order for order in self._orders.values() if order.parent_order_id is None]

    async def find_child_orders(self, parent_order_id: str) -> List[Order]:
        """Trouve tous les ordres enfants d'un ordre parent"""
        return [order for order in self._orders.values() if order.parent_order_id == parent_order_id]

    async def find_orders_with_tag(self, tag_key: str, tag_value: Optional[str] = None) -> List[Order]:
        """Trouve tous les ordres avec un tag donné"""
        result = []
        for order in self._orders.values():
            if hasattr(order, 'tags') and order.tags:
                if tag_key in order.tags:
                    if tag_value is None or order.tags[tag_key] == tag_value:
                        result.append(order)
        return result

    async def update(self, order: Order) -> None:
        """Met à jour un ordre existant"""
        if order.id not in self._orders:
            raise ValueError(f"Order {order.id} not found")

        # Sauvegarder (cela met à jour les indexes automatiquement)
        await self.save(order)

    async def get_order_statistics(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Calcule des statistiques détaillées sur les ordres"""

        # Filtrer les ordres selon les critères
        orders = list(self._orders.values())

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        if start_date:
            orders = [o for o in orders if o.created_time >= start_date]
        if end_date:
            orders = [o for o in orders if o.created_time <= end_date]

        if not orders:
            return {
                "total_orders": 0,
                "avg_order_size": 0,
                "total_volume": 0,
                "fill_rate": 0,
                "avg_fill_time": 0
            }

        total_orders = len(orders)
        total_volume = sum(order.quantity for order in orders)
        avg_order_size = total_volume / total_orders if total_orders > 0 else Decimal("0")

        # Calcul du taux de remplissage
        filled_orders = [o for o in orders if o.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]]
        fill_rate = len(filled_orders) / total_orders if total_orders > 0 else 0

        # Calcul du temps moyen de remplissage
        fill_times = []
        for order in filled_orders:
            if order.submitted_time and order.accepted_time:
                fill_time = (order.accepted_time - order.submitted_time).total_seconds()
                fill_times.append(fill_time)

        avg_fill_time = sum(fill_times) / len(fill_times) if fill_times else 0

        return {
            "total_orders": total_orders,
            "avg_order_size": float(avg_order_size),
            "total_volume": float(total_volume),
            "fill_rate": fill_rate,
            "avg_fill_time": avg_fill_time,
            "orders_by_status": {status.value: len([o for o in orders if o.status == status]) for status in OrderStatus}
        }

    async def get_execution_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Calcule des statistiques sur les exécutions"""

        all_executions = []
        total_execution_value = Decimal("0")
        total_commission = Decimal("0")
        total_fees = Decimal("0")

        for order in self._orders.values():
            # Filtrer par date si spécifié
            order_valid = True
            if start_date and order.created_time < start_date:
                order_valid = False
            if end_date and order.created_time > end_date:
                order_valid = False

            if order_valid and order.executions:
                for execution in order.executions:
                    all_executions.append(execution)
                    total_execution_value += execution.executed_quantity * execution.execution_price
                    total_commission += execution.commission
                    total_fees += execution.fees

        total_executions = len(all_executions)
        avg_execution_size = (sum(exec.executed_quantity for exec in all_executions) / total_executions
                             if total_executions > 0 else Decimal("0"))

        return {
            "total_executions": total_executions,
            "total_execution_value": float(total_execution_value),
            "avg_execution_size": float(avg_execution_size),
            "total_commission": float(total_commission),
            "total_fees": float(total_fees),
            "commission_rate": float(total_commission / total_execution_value) if total_execution_value > 0 else 0
        }

    async def archive_old_orders(
        self,
        cutoff_date: datetime,
        archive_status: Optional[List[OrderStatus]] = None
    ) -> int:
        """Archive les anciens ordres en les marquant comme archivés"""

        if archive_status is None:
            archive_status = [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]

        archived_count = 0
        orders_to_archive = []

        for order in self._orders.values():
            if (order.created_time < cutoff_date and
                order.status in archive_status):
                orders_to_archive.append(order)

        # Marquer comme archivé (ajouter un champ custom)
        for order in orders_to_archive:
            if not hasattr(order, 'archived'):
                order.archived = True
                order.archived_at = datetime.utcnow()
                archived_count += 1

        return archived_count

    async def cleanup_expired_orders(self) -> int:
        """Nettoie les ordres expirés en les marquant comme expirés"""

        expired_orders = await self.find_expired_orders()
        cleaned_count = 0

        for order in expired_orders:
            if order.status not in [OrderStatus.EXPIRED, OrderStatus.FILLED, OrderStatus.CANCELLED]:
                order.status = OrderStatus.EXPIRED
                order.last_update_time = datetime.utcnow()
                await self.update(order)
                cleaned_count += 1

        return cleaned_count