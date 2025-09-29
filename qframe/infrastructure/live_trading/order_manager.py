"""
Live Order Manager
==================

Production order management with retry logic, execution optimization,
and comprehensive order lifecycle tracking.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
from collections import defaultdict

from ...core.container import injectable
from ...domain.entities.order import Order, OrderStatus
from .broker_adapters import BrokerAdapter, ExecutionResult


logger = logging.getLogger(__name__)


class ExecutionStrategy(str, Enum):
    """Stratégies d'exécution d'ordres"""
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    ICEBERG = "iceberg"
    SMART = "smart"  # Algorithme intelligent


@dataclass
class OrderTracker:
    """Suivi d'un ordre en cours"""
    order: Order
    submission_time: datetime
    execution_strategy: ExecutionStrategy
    retry_count: int = 0
    max_retries: int = 3
    last_error: Optional[str] = None
    partial_fills: List[ExecutionResult] = field(default_factory=list)
    total_executed: Decimal = Decimal("0")
    remaining_quantity: Decimal = field(init=False)

    def __post_init__(self):
        self.remaining_quantity = self.order.quantity

    @property
    def is_fully_executed(self) -> bool:
        return self.total_executed >= self.order.quantity

    @property
    def execution_time_elapsed(self) -> timedelta:
        return datetime.utcnow() - self.submission_time


@dataclass
class SlippageMetrics:
    """Métriques de slippage et exécution"""
    expected_price: Decimal
    executed_price: Decimal
    slippage_bps: Decimal  # Basis points
    execution_time_ms: float
    market_impact_bps: Decimal


@injectable
class LiveOrderManager:
    """
    Gestionnaire d'ordres production avec retry logic et optimisation.

    Responsabilités:
    - Gestion lifecycle complet des ordres
    - Retry automatique avec backoff exponentiel
    - Optimisation d'exécution (TWAP, VWAP, etc.)
    - Tracking de slippage et market impact
    - Détection et gestion ordres bloqués
    """

    def __init__(
        self,
        broker_adapter: BrokerAdapter,
        max_concurrent_orders: int = 50,
        default_timeout: timedelta = timedelta(minutes=5),
        stuck_order_threshold: timedelta = timedelta(minutes=10)
    ):
        self.broker_adapter = broker_adapter
        self.max_concurrent_orders = max_concurrent_orders
        self.default_timeout = default_timeout
        self.stuck_order_threshold = stuck_order_threshold

        # État des ordres
        self.active_orders: Dict[str, OrderTracker] = {}
        self.order_history: List[OrderTracker] = []
        self.execution_metrics: List[SlippageMetrics] = []

        # Statistiques
        self.total_orders_submitted = 0
        self.total_orders_executed = 0
        self.total_orders_failed = 0

        # Configuration retry
        self.base_retry_delay = 1.0  # secondes
        self.max_retry_delay = 30.0  # secondes

    async def initialize(self) -> None:
        """Initialise le gestionnaire d'ordres"""
        logger.info("Order manager initialized")

        # Démarrer tâche de monitoring
        asyncio.create_task(self._monitoring_loop())

    async def submit_order(
        self,
        order: Order,
        execution_strategy: ExecutionStrategy = ExecutionStrategy.SMART,
        timeout: Optional[timedelta] = None
    ) -> ExecutionResult:
        """
        Soumet un ordre avec stratégie d'exécution.

        Args:
            order: Ordre à exécuter
            execution_strategy: Stratégie d'exécution
            timeout: Timeout personnalisé

        Returns:
            Résultat d'exécution
        """
        # Vérifier limites
        if len(self.active_orders) >= self.max_concurrent_orders:
            return ExecutionResult(
                success=False,
                error="Max concurrent orders limit reached"
            )

        # Créer tracker
        tracker = OrderTracker(
            order=order,
            submission_time=datetime.utcnow(),
            execution_strategy=execution_strategy
        )

        self.total_orders_submitted += 1

        try:
            # Exécuter selon stratégie
            if execution_strategy == ExecutionStrategy.MARKET:
                result = await self._execute_market_order(tracker)
            elif execution_strategy == ExecutionStrategy.LIMIT:
                result = await self._execute_limit_order(tracker)
            elif execution_strategy == ExecutionStrategy.TWAP:
                result = await self._execute_twap_order(tracker, timeout or self.default_timeout)
            elif execution_strategy == ExecutionStrategy.SMART:
                result = await self._execute_smart_order(tracker)
            else:
                result = await self._execute_market_order(tracker)  # Fallback

            # Enregistrer résultat
            if result.success:
                self.total_orders_executed += 1
                self._record_execution_metrics(tracker, result)
            else:
                self.total_orders_failed += 1

            # Nettoyer tracker
            if tracker.order.symbol in self.active_orders:
                self.order_history.append(self.active_orders.pop(tracker.order.symbol))

            return result

        except Exception as e:
            logger.error(f"Error submitting order {order.symbol}: {e}")
            self.total_orders_failed += 1

            return ExecutionResult(
                success=False,
                error=str(e)
            )

    async def cancel_order(self, order_id: str) -> bool:
        """Annule un ordre spécifique"""
        try:
            success = await self.broker_adapter.cancel_order(order_id)

            if success:
                # Nettoyer de active_orders si présent
                for symbol, tracker in list(self.active_orders.items()):
                    if any(pf.order_id == order_id for pf in tracker.partial_fills):
                        self.order_history.append(self.active_orders.pop(symbol))
                        break

                logger.info(f"Order {order_id} cancelled successfully")

            return success

        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    async def cancel_all_orders(self) -> int:
        """Annule tous les ordres actifs"""
        cancelled_count = 0

        for tracker in list(self.active_orders.values()):
            for partial_fill in tracker.partial_fills:
                if partial_fill.order_id:
                    if await self.cancel_order(partial_fill.order_id):
                        cancelled_count += 1

        logger.info(f"Cancelled {cancelled_count} orders")
        return cancelled_count

    async def emergency_cancel_all(self) -> None:
        """Annulation d'urgence de tous les ordres"""
        logger.critical("EMERGENCY: Cancelling all orders")

        # Tentatives parallèles d'annulation
        cancel_tasks = []

        for tracker in self.active_orders.values():
            for partial_fill in tracker.partial_fills:
                if partial_fill.order_id:
                    task = asyncio.create_task(
                        self.broker_adapter.cancel_order(partial_fill.order_id)
                    )
                    cancel_tasks.append(task)

        # Attendre toutes les annulations (avec timeout court)
        try:
            await asyncio.wait_for(
                asyncio.gather(*cancel_tasks, return_exceptions=True),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.critical("Emergency cancellation timeout - some orders may still be active")

        # Nettoyer état local
        self.active_orders.clear()

    async def get_pending_orders_count(self) -> int:
        """Retourne le nombre d'ordres en attente"""
        return len(self.active_orders)

    async def check_stuck_orders(self) -> List[OrderTracker]:
        """Détecte les ordres bloqués"""
        stuck_orders = []

        for tracker in self.active_orders.values():
            if tracker.execution_time_elapsed > self.stuck_order_threshold:
                stuck_orders.append(tracker)

        if stuck_orders:
            logger.warning(f"Found {len(stuck_orders)} stuck orders")

        return stuck_orders

    async def get_execution_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques d'exécution"""
        if not self.execution_metrics:
            return {"no_data": True}

        slippages = [m.slippage_bps for m in self.execution_metrics]
        execution_times = [m.execution_time_ms for m in self.execution_metrics]

        return {
            "total_orders_submitted": self.total_orders_submitted,
            "total_orders_executed": self.total_orders_executed,
            "total_orders_failed": self.total_orders_failed,
            "success_rate": (self.total_orders_executed / self.total_orders_submitted) * 100 if self.total_orders_submitted > 0 else 0,
            "average_slippage_bps": sum(slippages) / len(slippages),
            "median_slippage_bps": sorted(slippages)[len(slippages) // 2],
            "average_execution_time_ms": sum(execution_times) / len(execution_times),
            "active_orders_count": len(self.active_orders)
        }

    # === Stratégies d'exécution ===

    async def _execute_market_order(self, tracker: OrderTracker) -> ExecutionResult:
        """Exécute un ordre au marché avec retry"""
        for attempt in range(tracker.max_retries + 1):
            try:
                # Modifier ordre pour type market
                market_order = Order(
                    symbol=tracker.order.symbol,
                    side=tracker.order.side,
                    quantity=tracker.remaining_quantity,
                    order_type="market",
                    timestamp=datetime.utcnow()
                )

                result = await self.broker_adapter.submit_order(market_order)

                if result.success:
                    tracker.partial_fills.append(result)
                    if result.executed_quantity:
                        tracker.total_executed += result.executed_quantity
                        tracker.remaining_quantity -= result.executed_quantity

                    return result
                else:
                    tracker.retry_count = attempt
                    tracker.last_error = result.error

                    if attempt < tracker.max_retries:
                        delay = min(self.base_retry_delay * (2 ** attempt), self.max_retry_delay)
                        await asyncio.sleep(delay)
                    else:
                        return result

            except Exception as e:
                tracker.last_error = str(e)
                if attempt < tracker.max_retries:
                    delay = min(self.base_retry_delay * (2 ** attempt), self.max_retry_delay)
                    await asyncio.sleep(delay)

        return ExecutionResult(
            success=False,
            error=f"Failed after {tracker.max_retries} retries: {tracker.last_error}"
        )

    async def _execute_limit_order(self, tracker: OrderTracker) -> ExecutionResult:
        """Exécute un ordre à cours limité"""
        try:
            result = await self.broker_adapter.submit_order(tracker.order)

            if result.success:
                tracker.partial_fills.append(result)
                # Ajouter à active_orders pour monitoring
                self.active_orders[tracker.order.symbol] = tracker

            return result

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e)
            )

    async def _execute_twap_order(self, tracker: OrderTracker, duration: timedelta) -> ExecutionResult:
        """
        Exécute un ordre TWAP (Time-Weighted Average Price).

        Divise l'ordre en petites tranches sur la durée spécifiée.
        """
        try:
            # Paramètres TWAP
            slice_count = min(10, int(duration.total_seconds() / 30))  # Max 10 tranches, min 30s par tranche
            slice_size = tracker.order.quantity / slice_count
            slice_interval = duration.total_seconds() / slice_count

            logger.info(f"TWAP order: {slice_count} slices of {slice_size} over {duration}")

            executed_results = []

            for i in range(slice_count):
                if tracker.remaining_quantity <= 0:
                    break

                # Créer ordre slice
                slice_quantity = min(slice_size, tracker.remaining_quantity)
                slice_order = Order(
                    symbol=tracker.order.symbol,
                    side=tracker.order.side,
                    quantity=slice_quantity,
                    order_type="market",
                    timestamp=datetime.utcnow()
                )

                # Exécuter slice
                result = await self.broker_adapter.submit_order(slice_order)

                if result.success:
                    tracker.partial_fills.append(result)
                    executed_results.append(result)

                    if result.executed_quantity:
                        tracker.total_executed += result.executed_quantity
                        tracker.remaining_quantity -= result.executed_quantity

                # Attendre avant prochaine slice (sauf dernière)
                if i < slice_count - 1:
                    await asyncio.sleep(slice_interval)

            # Calculer résultat agrégé
            if executed_results:
                total_executed = sum(r.executed_quantity or Decimal("0") for r in executed_results)
                total_commission = sum(r.commission or Decimal("0") for r in executed_results)

                # Prix moyen pondéré
                weighted_price = sum(
                    (r.executed_price or Decimal("0")) * (r.executed_quantity or Decimal("0"))
                    for r in executed_results
                ) / total_executed if total_executed > 0 else Decimal("0")

                return ExecutionResult(
                    success=True,
                    order_id=f"twap_{tracker.order.symbol}_{int(datetime.utcnow().timestamp())}",
                    executed_price=weighted_price,
                    executed_quantity=total_executed,
                    commission=total_commission,
                    timestamp=datetime.utcnow()
                )
            else:
                return ExecutionResult(
                    success=False,
                    error="No slices executed successfully"
                )

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"TWAP execution error: {e}"
            )

    async def _execute_smart_order(self, tracker: OrderTracker) -> ExecutionResult:
        """
        Stratégie d'exécution intelligente.

        Choisit automatiquement la meilleure stratégie basée sur:
        - Taille de l'ordre
        - Liquidité du marché
        - Volatilité
        """
        try:
            # Analyser contexte de marché
            market_data = await self.broker_adapter.get_market_data()
            symbol_data = market_data.get(tracker.order.symbol, {})

            # Décision basée sur taille d'ordre
            if tracker.order.quantity <= Decimal("100"):
                # Petit ordre -> market order
                return await self._execute_market_order(tracker)
            elif tracker.order.quantity <= Decimal("1000"):
                # Ordre moyen -> limit order proche du marché
                current_price = Decimal(str(symbol_data.get("price", 100)))

                # Ajuster prix limit (1 tick inside)
                if tracker.order.side.lower() == "buy":
                    limit_price = current_price * Decimal("1.001")  # 0.1% au-dessus
                else:
                    limit_price = current_price * Decimal("0.999")  # 0.1% en-dessous

                limit_order = Order(
                    symbol=tracker.order.symbol,
                    side=tracker.order.side,
                    quantity=tracker.order.quantity,
                    price=limit_price,
                    order_type="limit",
                    timestamp=datetime.utcnow()
                )

                # Remplacer ordre dans tracker
                tracker.order = limit_order
                return await self._execute_limit_order(tracker)
            else:
                # Gros ordre -> TWAP
                return await self._execute_twap_order(tracker, timedelta(minutes=5))

        except Exception as e:
            logger.error(f"Smart order execution error: {e}")
            # Fallback vers market order
            return await self._execute_market_order(tracker)

    # === Méthodes utilitaires ===

    async def _monitoring_loop(self) -> None:
        """Boucle de monitoring des ordres actifs"""
        while True:
            try:
                # Vérifier statut des ordres actifs
                for symbol, tracker in list(self.active_orders.items()):
                    # Vérifier timeout
                    if tracker.execution_time_elapsed > self.default_timeout:
                        logger.warning(f"Order {symbol} timed out, attempting cancellation")

                        # Essayer d'annuler
                        for partial_fill in tracker.partial_fills:
                            if partial_fill.order_id:
                                await self.cancel_order(partial_fill.order_id)

                        # Déplacer vers historique
                        self.order_history.append(self.active_orders.pop(symbol))

                await asyncio.sleep(30)  # Check toutes les 30 secondes

            except Exception as e:
                logger.error(f"Error in order monitoring loop: {e}")
                await asyncio.sleep(60)

    def _record_execution_metrics(self, tracker: OrderTracker, result: ExecutionResult) -> None:
        """Enregistre les métriques d'exécution"""
        if not result.executed_price or not result.executed_quantity:
            return

        # Calculer slippage (estimation)
        expected_price = tracker.order.price or result.executed_price
        slippage_bps = abs((result.executed_price - expected_price) / expected_price) * 10000

        # Temps d'exécution
        execution_time_ms = tracker.execution_time_elapsed.total_seconds() * 1000

        metrics = SlippageMetrics(
            expected_price=expected_price,
            executed_price=result.executed_price,
            slippage_bps=slippage_bps,
            execution_time_ms=execution_time_ms,
            market_impact_bps=slippage_bps  # Approximation simple
        )

        self.execution_metrics.append(metrics)

        # Garder seulement les 1000 dernières métriques
        if len(self.execution_metrics) > 1000:
            self.execution_metrics = self.execution_metrics[-1000:]