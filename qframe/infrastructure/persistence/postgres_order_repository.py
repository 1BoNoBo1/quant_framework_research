"""
Infrastructure Layer: PostgreSQL Order Repository
===============================================

Implémentation PostgreSQL du repository des ordres.
Utilisée pour la production avec persistance complète.
"""

import json
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
import asyncpg

from qframe.domain.entities.order import Order, OrderStatus, OrderSide, OrderType, OrderExecution
from qframe.domain.repositories.order_repository import OrderRepository


class PostgresOrderRepository(OrderRepository):
    """Repository PostgreSQL pour les ordres"""

    def __init__(self, connection_pool: asyncpg.Pool):
        self.pool = connection_pool

    async def save(self, order: Order) -> None:
        """Sauvegarde un ordre"""
        async with self.pool.acquire() as conn:
            # Vérifier si l'ordre existe déjà
            existing = await conn.fetchrow(
                "SELECT id FROM orders WHERE id = $1", order.id
            )

            executions_json = json.dumps([
                {
                    "id": exec.id,
                    "price": str(exec.price),
                    "quantity": str(exec.quantity),
                    "timestamp": exec.timestamp.isoformat() if exec.timestamp else None,
                    "venue": exec.venue.value if exec.venue else None,
                    "fee": str(exec.fee) if exec.fee else None,
                    "commission": str(exec.commission) if exec.commission else None
                }
                for exec in order.executions
            ])

            if existing:
                # Mise à jour
                await conn.execute("""
                    UPDATE orders SET
                        symbol = $2,
                        side = $3,
                        order_type = $4,
                        quantity = $5,
                        price = $6,
                        stop_price = $7,
                        time_in_force = $8,
                        status = $9,
                        strategy_id = $10,
                        parent_order_id = $11,
                        tags = $12,
                        executions = $13,
                        filled_quantity = $14,
                        remaining_quantity = $15,
                        average_fill_price = $16,
                        commission = $17,
                        fees = $18,
                        created_at = $19,
                        updated_at = $20,
                        submitted_at = $21,
                        filled_at = $22,
                        cancelled_at = $23,
                        rejected_at = $24,
                        rejection_reason = $25
                    WHERE id = $1
                """,
                    order.id,
                    order.symbol,
                    order.side.value,
                    order.order_type.value,
                    str(order.quantity),
                    str(order.price) if order.price else None,
                    str(order.stop_price) if order.stop_price else None,
                    order.time_in_force.value if order.time_in_force else None,
                    order.status.value,
                    order.strategy_id,
                    order.parent_order_id,
                    json.dumps(order.tags) if order.tags else None,
                    executions_json,
                    str(order.filled_quantity),
                    str(order.remaining_quantity),
                    str(order.average_fill_price) if order.average_fill_price else None,
                    str(order.commission) if order.commission else None,
                    str(order.fees) if order.fees else None,
                    order.created_at,
                    order.updated_at,
                    order.submitted_at,
                    order.filled_at,
                    order.cancelled_at,
                    order.rejected_at,
                    order.rejection_reason
                )
            else:
                # Insertion
                await conn.execute("""
                    INSERT INTO orders (
                        id, symbol, side, order_type, quantity, price, stop_price,
                        time_in_force, status, strategy_id, parent_order_id, tags,
                        executions, filled_quantity, remaining_quantity, average_fill_price,
                        commission, fees, created_at, updated_at, submitted_at,
                        filled_at, cancelled_at, rejected_at, rejection_reason
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                        $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25
                    )
                """,
                    order.id,
                    order.symbol,
                    order.side.value,
                    order.order_type.value,
                    str(order.quantity),
                    str(order.price) if order.price else None,
                    str(order.stop_price) if order.stop_price else None,
                    order.time_in_force.value if order.time_in_force else None,
                    order.status.value,
                    order.strategy_id,
                    order.parent_order_id,
                    json.dumps(order.tags) if order.tags else None,
                    executions_json,
                    str(order.filled_quantity),
                    str(order.remaining_quantity),
                    str(order.average_fill_price) if order.average_fill_price else None,
                    str(order.commission) if order.commission else None,
                    str(order.fees) if order.fees else None,
                    order.created_at,
                    order.updated_at,
                    order.submitted_at,
                    order.filled_at,
                    order.cancelled_at,
                    order.rejected_at,
                    order.rejection_reason
                )

    async def find_by_id(self, order_id: str) -> Optional[Order]:
        """Trouve un ordre par son ID"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM orders WHERE id = $1", order_id
            )
            return self._row_to_order(row) if row else None

    async def find_by_symbol(self, symbol: str) -> List[Order]:
        """Trouve tous les ordres pour un symbole"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM orders WHERE symbol = $1 ORDER BY created_at DESC", symbol
            )
            return [self._row_to_order(row) for row in rows]

    async def find_by_status(self, status: OrderStatus) -> List[Order]:
        """Trouve tous les ordres par statut"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM orders WHERE status = $1 ORDER BY created_at DESC", status.value
            )
            return [self._row_to_order(row) for row in rows]

    async def find_active_orders(self) -> List[Order]:
        """Trouve tous les ordres actifs"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM orders
                WHERE status IN ('PENDING', 'PARTIAL_FILLED')
                ORDER BY created_at DESC
            """)
            return [self._row_to_order(row) for row in rows]

    async def find_by_strategy_id(self, strategy_id: str) -> List[Order]:
        """Trouve tous les ordres d'une stratégie"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM orders WHERE strategy_id = $1 ORDER BY created_at DESC", strategy_id
            )
            return [self._row_to_order(row) for row in rows]

    async def find_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Order]:
        """Trouve tous les ordres dans une plage de dates"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM orders
                WHERE created_at >= $1 AND created_at <= $2
                ORDER BY created_at DESC
            """, start_date, end_date)
            return [self._row_to_order(row) for row in rows]

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
        conditions = []
        params = []
        param_count = 0

        if symbol:
            param_count += 1
            conditions.append(f"symbol = ${param_count}")
            params.append(symbol)

        if status:
            param_count += 1
            conditions.append(f"status = ${param_count}")
            params.append(status.value)

        if side:
            param_count += 1
            conditions.append(f"side = ${param_count}")
            params.append(side.value)

        if order_type:
            param_count += 1
            conditions.append(f"order_type = ${param_count}")
            params.append(order_type.value)

        if strategy_id:
            param_count += 1
            conditions.append(f"strategy_id = ${param_count}")
            params.append(strategy_id)

        if min_quantity:
            param_count += 1
            conditions.append(f"quantity >= ${param_count}")
            params.append(str(min_quantity))

        if max_quantity:
            param_count += 1
            conditions.append(f"quantity <= ${param_count}")
            params.append(str(max_quantity))

        if start_date:
            param_count += 1
            conditions.append(f"created_at >= ${param_count}")
            params.append(start_date)

        if end_date:
            param_count += 1
            conditions.append(f"created_at <= ${param_count}")
            params.append(end_date)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        query = f"SELECT * FROM orders WHERE {where_clause} ORDER BY created_at DESC"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [self._row_to_order(row) for row in rows]

    async def count_by_status(self, status: OrderStatus) -> int:
        """Compte les ordres par statut"""
        async with self.pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM orders WHERE status = $1", status.value
            )
            return count or 0

    async def count_by_symbol(self, symbol: str) -> int:
        """Compte les ordres par symbole"""
        async with self.pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM orders WHERE symbol = $1", symbol
            )
            return count or 0

    async def get_total_volume_by_symbol(self, symbol: str) -> Decimal:
        """Calcule le volume total par symbole"""
        async with self.pool.acquire() as conn:
            volume = await conn.fetchval(
                "SELECT SUM(quantity::decimal) FROM orders WHERE symbol = $1", symbol
            )
            return Decimal(str(volume)) if volume else Decimal("0")

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
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM orders WHERE id = $1", order_id
            )
            return result == "DELETE 1"

    async def exists(self, order_id: str) -> bool:
        """Vérifie si un ordre existe"""
        async with self.pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM orders WHERE id = $1", order_id
            )
            return count > 0

    async def count_all(self) -> int:
        """Compte tous les ordres"""
        async with self.pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM orders")
            return count or 0

    def _row_to_order(self, row: asyncpg.Record) -> Order:
        """Convertit une ligne de base de données en Order"""
        executions = []
        if row['executions']:
            executions_data = json.loads(row['executions'])
            for exec_data in executions_data:
                execution = OrderExecution(
                    id=exec_data.get('id', ''),
                    price=Decimal(exec_data['price']) if exec_data.get('price') else Decimal("0"),
                    quantity=Decimal(exec_data['quantity']) if exec_data.get('quantity') else Decimal("0"),
                    timestamp=datetime.fromisoformat(exec_data['timestamp']) if exec_data.get('timestamp') else None,
                    venue=exec_data.get('venue'),
                    fee=Decimal(exec_data['fee']) if exec_data.get('fee') else None,
                    commission=Decimal(exec_data['commission']) if exec_data.get('commission') else None
                )
                executions.append(execution)

        return Order(
            id=row['id'],
            symbol=row['symbol'],
            side=OrderSide(row['side']),
            order_type=OrderType(row['order_type']),
            quantity=Decimal(row['quantity']),
            price=Decimal(row['price']) if row['price'] else None,
            stop_price=Decimal(row['stop_price']) if row['stop_price'] else None,
            time_in_force=row['time_in_force'],
            status=OrderStatus(row['status']),
            strategy_id=row['strategy_id'],
            parent_order_id=row['parent_order_id'],
            tags=json.loads(row['tags']) if row['tags'] else {},
            executions=executions,
            filled_quantity=Decimal(row['filled_quantity']),
            remaining_quantity=Decimal(row['remaining_quantity']),
            average_fill_price=Decimal(row['average_fill_price']) if row['average_fill_price'] else None,
            commission=Decimal(row['commission']) if row['commission'] else None,
            fees=Decimal(row['fees']) if row['fees'] else None,
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            submitted_at=row['submitted_at'],
            filled_at=row['filled_at'],
            cancelled_at=row['cancelled_at'],
            rejected_at=row['rejected_at'],
            rejection_reason=row['rejection_reason']
        )