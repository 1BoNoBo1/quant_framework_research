"""
Infrastructure Layer: Mock Broker Adapter
========================================

Adaptateur de courtier fictif pour les tests et le développement.
Implémente le protocole BrokerAdapter avec une simulation complète.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
import logging
import random

from qframe.domain.entities.order import Order, OrderStatus, OrderSide, OrderType, ExecutionVenue
from qframe.domain.services.execution_service import VenueQuote
from qframe.infrastructure.external.order_execution_adapter import BrokerAdapter


logger = logging.getLogger(__name__)


class MockBrokerAdapter(BrokerAdapter):
    """
    Adaptateur de courtier fictif pour simulation et tests.
    Simule le comportement d'un vrai courtier avec latence et variations de prix.
    """

    def __init__(
        self,
        venue: ExecutionVenue,
        base_latency_ms: int = 100,
        fill_probability: float = 0.95,
        price_slippage_bps: float = 5.0
    ):
        self.venue = venue
        self.base_latency_ms = base_latency_ms
        self.fill_probability = fill_probability
        self.price_slippage_bps = price_slippage_bps

        # État interne
        self._orders: Dict[str, Dict[str, Any]] = {}
        self._market_prices: Dict[str, Decimal] = {
            "BTC-USD": Decimal("50000.00"),
            "ETH-USD": Decimal("3000.00"),
            "AAPL": Decimal("150.00"),
            "TSLA": Decimal("200.00"),
            "SPY": Decimal("400.00")
        }
        self._last_price_update = datetime.utcnow()

        # Configuration des frais par venue
        self._fee_rates = {
            ExecutionVenue.BINANCE: Decimal("0.001"),   # 0.1%
            ExecutionVenue.COINBASE: Decimal("0.005"),  # 0.5%
            ExecutionVenue.KRAKEN: Decimal("0.0026"),   # 0.26%
            ExecutionVenue.ALPACA: Decimal("0.0000"),   # Gratuit pour stocks
            ExecutionVenue.INTERACTIVE_BROKERS: Decimal("0.0005")  # 0.05%
        }

        logger.info(f"🤖 MockBrokerAdapter initialized for {venue.value}")

    async def submit_order(self, order: Order) -> Dict[str, Any]:
        """Soumet un ordre au courtier fictif"""
        await self._simulate_latency()

        broker_order_id = f"{self.venue.value}_{uuid.uuid4().hex[:8]}"

        # Simuler parfois un rejet d'ordre
        if random.random() < 0.02:  # 2% de chance de rejet
            return {
                "success": False,
                "error": "Insufficient funds",
                "order_id": None
            }

        # Mettre à jour les prix de marché
        await self._update_market_prices()

        # Déterminer le prix d'exécution
        market_price = self._market_prices.get(order.symbol, Decimal("100.00"))
        execution_price = self._calculate_execution_price(order, market_price)

        # Créer l'ordre chez le courtier
        broker_order = {
            "broker_order_id": broker_order_id,
            "original_order_id": order.id,
            "symbol": order.symbol,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "quantity": str(order.quantity),
            "requested_price": str(order.price) if order.price else None,
            "execution_price": str(execution_price),
            "status": "submitted",
            "filled_quantity": "0",
            "remaining_quantity": str(order.quantity),
            "executions": [],
            "fees": "0",
            "commission": "0",
            "created_at": datetime.utcnow().isoformat(),
            "venue": self.venue.value
        }

        self._orders[broker_order_id] = broker_order

        # Simuler l'exécution asynchrone pour les ordres market
        if order.order_type == OrderType.MARKET:
            asyncio.create_task(self._simulate_market_execution(broker_order_id))
        elif order.order_type == OrderType.LIMIT:
            asyncio.create_task(self._simulate_limit_execution(broker_order_id))

        logger.info(f"📤 Order submitted to {self.venue.value}: {broker_order_id}")

        return {
            "success": True,
            "order_id": broker_order_id,
            "status": "submitted",
            "venue": self.venue.value
        }

    async def cancel_order(self, order_id: str, broker_order_id: str) -> bool:
        """Annule un ordre chez le courtier fictif"""
        await self._simulate_latency()

        if broker_order_id not in self._orders:
            logger.warning(f"⚠️ Order {broker_order_id} not found for cancellation")
            return False

        order = self._orders[broker_order_id]

        # Vérifier si l'ordre peut être annulé
        if order["status"] in ["filled", "cancelled", "rejected"]:
            logger.warning(f"⚠️ Cannot cancel order {broker_order_id} with status {order['status']}")
            return False

        # Simuler parfois un échec d'annulation
        if random.random() < 0.05:  # 5% de chance d'échec
            logger.warning(f"⚠️ Failed to cancel order {broker_order_id} - broker error")
            return False

        # Annuler l'ordre
        order["status"] = "cancelled"
        order["cancelled_at"] = datetime.utcnow().isoformat()

        logger.info(f"❌ Order cancelled: {broker_order_id}")
        return True

    async def get_order_status(self, broker_order_id: str) -> Dict[str, Any]:
        """Récupère le statut d'un ordre chez le courtier fictif"""
        await self._simulate_latency(factor=0.5)  # Plus rapide pour les requêtes de statut

        if broker_order_id not in self._orders:
            return {"error": f"Order {broker_order_id} not found"}

        order = self._orders[broker_order_id].copy()

        # Convertir les exécutions au format attendu
        executions = []
        for exec_data in order["executions"]:
            executions.append({
                "price": exec_data["price"],
                "quantity": exec_data["quantity"],
                "timestamp": exec_data["timestamp"],
                "fee": exec_data.get("fee", "0"),
                "commission": exec_data.get("commission", "0")
            })

        return {
            "broker_order_id": broker_order_id,
            "status": order["status"],
            "filled_quantity": order["filled_quantity"],
            "remaining_quantity": order["remaining_quantity"],
            "executions": executions,
            "total_fees": order["fees"],
            "total_commission": order["commission"]
        }

    async def get_market_data(self, symbol: str) -> VenueQuote:
        """Récupère les données de marché pour un symbole"""
        await self._simulate_latency(factor=0.3)  # Très rapide pour les données de marché
        await self._update_market_prices()

        base_price = self._market_prices.get(symbol, Decimal("100.00"))

        # Simuler un spread bid/ask
        spread_bps = random.uniform(1.0, 10.0)  # 1-10 bps de spread
        spread = base_price * Decimal(str(spread_bps / 10000))

        bid_price = base_price - spread / 2
        ask_price = base_price + spread / 2

        # Simuler des volumes
        base_volume = Decimal(str(random.uniform(100, 10000)))

        return VenueQuote(
            venue=self.venue,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_volume=base_volume * Decimal(str(random.uniform(0.8, 1.2))),
            ask_volume=base_volume * Decimal(str(random.uniform(0.8, 1.2))),
            timestamp=datetime.utcnow()
        )

    async def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé du courtier fictif"""
        await self._simulate_latency(factor=0.2)

        total_orders = len(self._orders)
        active_orders = len([o for o in self._orders.values() if o["status"] in ["submitted", "partial"]])

        return {
            "status": "healthy",
            "venue": self.venue.value,
            "connected": True,
            "total_orders": total_orders,
            "active_orders": active_orders,
            "supported_symbols": list(self._market_prices.keys()),
            "fee_rate": str(self._fee_rates.get(self.venue, Decimal("0.001"))),
            "last_price_update": self._last_price_update.isoformat(),
            "base_latency_ms": self.base_latency_ms
        }

    # Méthodes privées de simulation

    async def _simulate_latency(self, factor: float = 1.0) -> None:
        """Simule la latence réseau"""
        latency = self.base_latency_ms * factor * random.uniform(0.5, 1.5)
        await asyncio.sleep(latency / 1000)

    async def _update_market_prices(self) -> None:
        """Met à jour les prix de marché avec des mouvements simulés"""
        now = datetime.utcnow()
        if now - self._last_price_update < timedelta(seconds=1):
            return

        for symbol in self._market_prices:
            # Mouvement aléatoire de ±0.1%
            movement = random.uniform(-0.001, 0.001)
            self._market_prices[symbol] *= (1 + Decimal(str(movement)))

        self._last_price_update = now

    def _calculate_execution_price(self, order: Order, market_price: Decimal) -> Decimal:
        """Calcule le prix d'exécution avec slippage"""
        if order.order_type == OrderType.LIMIT and order.price:
            return order.price

        # Appliquer le slippage pour les ordres market
        slippage_factor = self.price_slippage_bps / 10000
        if order.side == OrderSide.BUY:
            slippage = Decimal(str(random.uniform(0, slippage_factor)))
            return market_price * (1 + slippage)
        else:
            slippage = Decimal(str(random.uniform(0, slippage_factor)))
            return market_price * (1 - slippage)

    async def _simulate_market_execution(self, broker_order_id: str) -> None:
        """Simule l'exécution d'un ordre market"""
        try:
            # Délai d'exécution
            await asyncio.sleep(random.uniform(0.1, 0.5))

            order = self._orders.get(broker_order_id)
            if not order or order["status"] != "submitted":
                return

            # Vérifier si l'ordre est exécuté
            if random.random() > self.fill_probability:
                order["status"] = "rejected"
                order["rejection_reason"] = "Market conditions"
                return

            # Exécuter complètement l'ordre market
            await self._execute_order_fully(broker_order_id)

        except Exception as e:
            logger.error(f"Error simulating market execution for {broker_order_id}: {e}")

    async def _simulate_limit_execution(self, broker_order_id: str) -> None:
        """Simule l'exécution d'un ordre limite"""
        try:
            order = self._orders.get(broker_order_id)
            if not order or order["status"] != "submitted":
                return

            # Délai avant première vérification
            await asyncio.sleep(random.uniform(1.0, 5.0))

            # Simuler des exécutions partielles ou complètes
            remaining_qty = Decimal(order["remaining_quantity"])

            while remaining_qty > 0 and order["status"] == "submitted":
                # Vérifier si les conditions de marché permettent l'exécution
                if random.random() < 0.3:  # 30% de chance d'exécution à chaque check
                    # Exécution partielle (20-100% du reste)
                    fill_percentage = random.uniform(0.2, 1.0)
                    fill_qty = remaining_qty * Decimal(str(fill_percentage))

                    await self._add_execution(broker_order_id, fill_qty)
                    remaining_qty = Decimal(order["remaining_quantity"])

                # Attendre avant la prochaine vérification
                await asyncio.sleep(random.uniform(2.0, 10.0))

        except Exception as e:
            logger.error(f"Error simulating limit execution for {broker_order_id}: {e}")

    async def _execute_order_fully(self, broker_order_id: str) -> None:
        """Exécute complètement un ordre"""
        order = self._orders[broker_order_id]
        remaining_qty = Decimal(order["remaining_quantity"])
        await self._add_execution(broker_order_id, remaining_qty)

    async def _add_execution(self, broker_order_id: str, quantity: Decimal) -> None:
        """Ajoute une exécution à un ordre"""
        order = self._orders[broker_order_id]

        execution_price = Decimal(order["execution_price"])
        execution_value = quantity * execution_price

        # Calculer les frais
        fee_rate = self._fee_rates.get(self.venue, Decimal("0.001"))
        fee = execution_value * fee_rate

        execution = {
            "price": str(execution_price),
            "quantity": str(quantity),
            "timestamp": datetime.utcnow().isoformat(),
            "fee": str(fee),
            "commission": str(fee)  # Simplifié: commission = fee
        }

        order["executions"].append(execution)

        # Mettre à jour les quantités
        filled_qty = Decimal(order["filled_quantity"]) + quantity
        remaining_qty = Decimal(order["remaining_quantity"]) - quantity

        order["filled_quantity"] = str(filled_qty)
        order["remaining_quantity"] = str(remaining_qty)

        # Mettre à jour les frais totaux
        total_fees = Decimal(order["fees"]) + fee
        order["fees"] = str(total_fees)
        order["commission"] = str(total_fees)

        # Mettre à jour le statut
        if remaining_qty <= 0:
            order["status"] = "filled"
            order["filled_at"] = datetime.utcnow().isoformat()
        else:
            order["status"] = "partial"

        logger.info(f"🎯 Execution added to {broker_order_id}: {quantity} @ {execution_price}")

    def clear_orders(self) -> None:
        """Vide tous les ordres (pour les tests)"""
        self._orders.clear()

    def set_market_price(self, symbol: str, price: Decimal) -> None:
        """Définit manuellement un prix de marché (pour les tests)"""
        self._market_prices[symbol] = price