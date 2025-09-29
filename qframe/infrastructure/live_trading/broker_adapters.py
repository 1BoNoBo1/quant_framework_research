"""
Broker Adapters
===============

Production broker adapters for real trading connections.
"""

from typing import Dict, List, Optional, Any, Protocol
from datetime import datetime
from decimal import Decimal
from dataclasses import dataclass
from abc import ABC, abstractmethod
import asyncio
import logging

from ...core.container import injectable
from ...domain.entities.order import Order, OrderStatus
from ...domain.entities.position import Position
from ...domain.entities.portfolio import Portfolio


logger = logging.getLogger(__name__)


@dataclass
class ConnectionStatus:
    """Status de connexion broker"""
    is_connected: bool
    last_heartbeat: datetime
    connection_quality: str  # "excellent", "good", "poor", "disconnected"
    latency_ms: float
    error_message: Optional[str] = None


@dataclass
class ExecutionResult:
    """Résultat d'exécution d'ordre"""
    success: bool
    order_id: Optional[str] = None
    executed_price: Optional[Decimal] = None
    executed_quantity: Optional[Decimal] = None
    commission: Optional[Decimal] = None
    timestamp: Optional[datetime] = None
    error: Optional[str] = None


class BrokerAdapter(Protocol):
    """Interface pour adaptateurs de brokers"""

    async def connect(self) -> bool:
        """Établit la connexion au broker"""
        ...

    async def disconnect(self) -> None:
        """Ferme la connexion au broker"""
        ...

    async def is_connected(self) -> bool:
        """Vérifie si la connexion est active"""
        ...

    async def get_connection_status(self) -> ConnectionStatus:
        """Retourne le statut détaillé de connexion"""
        ...

    async def submit_order(self, order: Order) -> ExecutionResult:
        """Soumet un ordre au broker"""
        ...

    async def cancel_order(self, order_id: str) -> bool:
        """Annule un ordre"""
        ...

    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Récupère le statut d'un ordre"""
        ...

    async def get_positions(self) -> List[Position]:
        """Récupère toutes les positions"""
        ...

    async def get_portfolio(self) -> Portfolio:
        """Récupère le portfolio complet"""
        ...

    async def get_market_data(self) -> Dict[str, Any]:
        """Récupère les données de marché en temps réel"""
        ...


@injectable
class InteractiveBrokersAdapter:
    """
    Adaptateur pour Interactive Brokers (IB).

    Utilise ib_insync pour la connexion TWS/Gateway.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,  # TWS paper trading port
        client_id: int = 1,
        timeout: int = 30
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout

        self.ib = None  # Will be ib_insync.IB() instance
        self.connection_status = ConnectionStatus(
            is_connected=False,
            last_heartbeat=datetime.utcnow(),
            connection_quality="disconnected",
            latency_ms=0.0
        )

        # Cache pour positions et portfolio
        self._positions_cache: List[Position] = []
        self._portfolio_cache: Optional[Portfolio] = None
        self._cache_timeout = 5  # 5 secondes
        self._last_cache_update = datetime.utcnow()

    async def connect(self) -> bool:
        """Établit la connexion à IB TWS/Gateway"""
        try:
            # Import dynamique pour éviter erreur si ib_insync pas installé
            try:
                from ib_insync import IB, util
                util.startLoop()  # Démarre event loop si nécessaire
            except ImportError:
                logger.error("ib_insync not installed. Install with: pip install ib_insync")
                return False

            self.ib = IB()

            # Connexion avec timeout
            await asyncio.wait_for(
                self.ib.connectAsync(self.host, self.port, clientId=self.client_id),
                timeout=self.timeout
            )

            # Vérifier connexion
            if self.ib.isConnected():
                self.connection_status.is_connected = True
                self.connection_status.connection_quality = "excellent"
                self.connection_status.last_heartbeat = datetime.utcnow()

                logger.info(f"Connected to IB at {self.host}:{self.port}")

                # Setup callbacks pour monitoring
                self.ib.disconnectedEvent += self._on_disconnected

                return True
            else:
                return False

        except asyncio.TimeoutError:
            logger.error(f"Connection timeout to IB at {self.host}:{self.port}")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            self.connection_status.error_message = str(e)
            return False

    async def disconnect(self) -> None:
        """Ferme la connexion IB"""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()

        self.connection_status.is_connected = False
        self.connection_status.connection_quality = "disconnected"
        logger.info("Disconnected from IB")

    async def is_connected(self) -> bool:
        """Vérifie si la connexion IB est active"""
        if not self.ib:
            return False
        return self.ib.isConnected()

    async def get_connection_status(self) -> ConnectionStatus:
        """Retourne le statut détaillé de connexion IB"""
        if self.ib and self.ib.isConnected():
            # Test de latence simple
            start_time = datetime.utcnow()
            try:
                # Ping via reqCurrentTime
                await self.ib.reqCurrentTimeAsync()
                latency = (datetime.utcnow() - start_time).total_seconds() * 1000

                self.connection_status.latency_ms = latency
                self.connection_status.last_heartbeat = datetime.utcnow()

                # Qualité basée sur latence
                if latency < 50:
                    self.connection_status.connection_quality = "excellent"
                elif latency < 100:
                    self.connection_status.connection_quality = "good"
                else:
                    self.connection_status.connection_quality = "poor"

            except Exception as e:
                self.connection_status.connection_quality = "poor"
                self.connection_status.error_message = str(e)

        return self.connection_status

    async def submit_order(self, order: Order) -> ExecutionResult:
        """Soumet un ordre à IB"""
        if not self.ib or not self.ib.isConnected():
            return ExecutionResult(
                success=False,
                error="Not connected to IB"
            )

        try:
            from ib_insync import Stock, MarketOrder, LimitOrder

            # Créer contrat IB
            contract = Stock(order.symbol, 'SMART', 'USD')

            # Créer ordre IB
            if order.order_type.lower() == "market":
                ib_order = MarketOrder(
                    action=order.side.upper(),
                    totalQuantity=float(order.quantity)
                )
            else:  # Limit order
                ib_order = LimitOrder(
                    action=order.side.upper(),
                    totalQuantity=float(order.quantity),
                    lmtPrice=float(order.price or 0)
                )

            # Soumettre ordre
            trade = self.ib.placeOrder(contract, ib_order)

            # Attendre confirmation
            await asyncio.sleep(0.1)  # Court délai pour confirmation

            return ExecutionResult(
                success=True,
                order_id=str(trade.order.orderId),
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Error submitting order to IB: {e}")
            return ExecutionResult(
                success=False,
                error=str(e)
            )

    async def cancel_order(self, order_id: str) -> bool:
        """Annule un ordre IB"""
        if not self.ib or not self.ib.isConnected():
            return False

        try:
            # Trouver trade par order ID
            trades = self.ib.trades()
            target_trade = None

            for trade in trades:
                if str(trade.order.orderId) == order_id:
                    target_trade = trade
                    break

            if target_trade:
                self.ib.cancelOrder(target_trade.order)
                return True
            else:
                logger.warning(f"Order {order_id} not found for cancellation")
                return False

        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Récupère le statut d'un ordre IB"""
        if not self.ib or not self.ib.isConnected():
            return OrderStatus.UNKNOWN

        try:
            trades = self.ib.trades()

            for trade in trades:
                if str(trade.order.orderId) == order_id:
                    # Mapper statut IB vers notre enum
                    ib_status = trade.orderStatus.status

                    if ib_status == "Filled":
                        return OrderStatus.FILLED
                    elif ib_status == "Cancelled":
                        return OrderStatus.CANCELLED
                    elif ib_status in ["Submitted", "PreSubmitted"]:
                        return OrderStatus.PENDING
                    else:
                        return OrderStatus.UNKNOWN

            return OrderStatus.UNKNOWN

        except Exception as e:
            logger.error(f"Error getting order status {order_id}: {e}")
            return OrderStatus.UNKNOWN

    async def get_positions(self) -> List[Position]:
        """Récupère toutes les positions IB"""
        # Vérifier cache
        if self._is_cache_valid():
            return self._positions_cache

        if not self.ib or not self.ib.isConnected():
            return []

        try:
            ib_positions = self.ib.positions()
            positions = []

            for pos in ib_positions:
                if pos.position != 0:  # Seulement positions non-nulles
                    position = Position(
                        symbol=pos.contract.symbol,
                        quantity=Decimal(str(pos.position)),
                        average_price=Decimal(str(pos.avgCost / abs(pos.position) if pos.position != 0 else 0)),
                        market_value=Decimal(str(pos.marketValue)),
                        unrealized_pnl=Decimal(str(pos.unrealizedPNL)),
                        timestamp=datetime.utcnow()
                    )
                    positions.append(position)

            # Mettre à jour cache
            self._positions_cache = positions
            self._last_cache_update = datetime.utcnow()

            return positions

        except Exception as e:
            logger.error(f"Error getting positions from IB: {e}")
            return []

    async def get_portfolio(self) -> Portfolio:
        """Récupère le portfolio complet IB"""
        # Vérifier cache
        if self._is_cache_valid() and self._portfolio_cache:
            return self._portfolio_cache

        if not self.ib or not self.ib.isConnected():
            return Portfolio()

        try:
            # Récupérer account summary
            account_values = self.ib.accountSummary()

            total_value = Decimal("0")
            cash_balance = Decimal("0")

            for item in account_values:
                if item.tag == "NetLiquidation":
                    total_value = Decimal(item.value)
                elif item.tag == "TotalCashValue":
                    cash_balance = Decimal(item.value)

            # Récupérer positions
            positions = await self.get_positions()

            portfolio = Portfolio(
                total_value=total_value,
                cash_balance=cash_balance,
                positions=positions,
                timestamp=datetime.utcnow()
            )

            # Mettre à jour cache
            self._portfolio_cache = portfolio
            self._last_cache_update = datetime.utcnow()

            return portfolio

        except Exception as e:
            logger.error(f"Error getting portfolio from IB: {e}")
            return Portfolio()

    async def get_market_data(self) -> Dict[str, Any]:
        """Récupère les données de marché en temps réel IB"""
        if not self.ib or not self.ib.isConnected():
            return {}

        try:
            # Exemple simple - récupérer données pour quelques symboles
            symbols = ["AAPL", "MSFT", "TSLA"]  # Configurable
            market_data = {}

            from ib_insync import Stock

            for symbol in symbols:
                contract = Stock(symbol, 'SMART', 'USD')
                ticker = self.ib.reqMktData(contract, "", False, False)

                # Attendre données
                await asyncio.sleep(0.1)

                if ticker.last and ticker.last > 0:
                    market_data[symbol] = {
                        "price": float(ticker.last),
                        "bid": float(ticker.bid) if ticker.bid else None,
                        "ask": float(ticker.ask) if ticker.ask else None,
                        "volume": float(ticker.volume) if ticker.volume else None,
                        "timestamp": datetime.utcnow().isoformat()
                    }

            return market_data

        except Exception as e:
            logger.error(f"Error getting market data from IB: {e}")
            return {}

    # === Méthodes privées ===

    def _is_cache_valid(self) -> bool:
        """Vérifie si le cache est encore valide"""
        elapsed = (datetime.utcnow() - self._last_cache_update).total_seconds()
        return elapsed < self._cache_timeout

    def _on_disconnected(self):
        """Callback appelé lors de déconnexion IB"""
        self.connection_status.is_connected = False
        self.connection_status.connection_quality = "disconnected"
        logger.warning("IB connection lost")


@injectable
class PaperTradingAdapter:
    """
    Adaptateur pour paper trading (simulation).

    Simule un broker pour les tests sans risque.
    """

    def __init__(self, initial_balance: Decimal = Decimal("100000")):
        self.initial_balance = initial_balance
        self.cash_balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.next_order_id = 1

        self.connection_status = ConnectionStatus(
            is_connected=True,
            last_heartbeat=datetime.utcnow(),
            connection_quality="excellent",
            latency_ms=1.0  # Simulation très rapide
        )

    async def connect(self) -> bool:
        """Simulation de connexion"""
        self.connection_status.is_connected = True
        logger.info("Paper trading adapter connected")
        return True

    async def disconnect(self) -> None:
        """Simulation de déconnexion"""
        self.connection_status.is_connected = False
        logger.info("Paper trading adapter disconnected")

    async def is_connected(self) -> bool:
        """Toujours connecté en simulation"""
        return self.connection_status.is_connected

    async def get_connection_status(self) -> ConnectionStatus:
        """Statut optimal en simulation"""
        self.connection_status.last_heartbeat = datetime.utcnow()
        return self.connection_status

    async def submit_order(self, order: Order) -> ExecutionResult:
        """Simule l'exécution d'un ordre"""
        order_id = str(self.next_order_id)
        self.next_order_id += 1

        # Simuler prix d'exécution (prix actuel + petit spread)
        executed_price = order.price or Decimal("100")  # Prix par défaut
        executed_quantity = order.quantity

        # Simuler commission
        commission = executed_price * executed_quantity * Decimal("0.001")  # 0.1%

        # Mettre à jour cash et positions
        trade_value = executed_price * executed_quantity

        if order.side.lower() == "buy":
            self.cash_balance -= (trade_value + commission)
            # Ajouter à position
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                new_quantity = pos.quantity + executed_quantity
                new_avg_price = ((pos.average_price * pos.quantity) + (executed_price * executed_quantity)) / new_quantity
                pos.quantity = new_quantity
                pos.average_price = new_avg_price
            else:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=executed_quantity,
                    average_price=executed_price,
                    market_value=trade_value,
                    unrealized_pnl=Decimal("0"),
                    timestamp=datetime.utcnow()
                )
        else:  # sell
            self.cash_balance += (trade_value - commission)
            # Réduire position
            if order.symbol in self.positions:
                self.positions[order.symbol].quantity -= executed_quantity

        return ExecutionResult(
            success=True,
            order_id=order_id,
            executed_price=executed_price,
            executed_quantity=executed_quantity,
            commission=commission,
            timestamp=datetime.utcnow()
        )

    async def cancel_order(self, order_id: str) -> bool:
        """Simulation d'annulation"""
        return True  # Toujours réussi en simulation

    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Tous les ordres sont exécutés instantanément en simulation"""
        return OrderStatus.FILLED

    async def get_positions(self) -> List[Position]:
        """Retourne les positions simulées"""
        return [pos for pos in self.positions.values() if pos.quantity != 0]

    async def get_portfolio(self) -> Portfolio:
        """Retourne le portfolio simulé"""
        positions = await self.get_positions()
        total_position_value = sum(pos.market_value for pos in positions)

        return Portfolio(
            total_value=self.cash_balance + total_position_value,
            cash_balance=self.cash_balance,
            positions=positions,
            timestamp=datetime.utcnow()
        )

    async def get_market_data(self) -> Dict[str, Any]:
        """Simule des données de marché"""
        import random

        symbols = ["AAPL", "MSFT", "TSLA", "BTCUSD"]
        market_data = {}

        for symbol in symbols:
            base_price = 100 + random.uniform(-50, 200)
            market_data[symbol] = {
                "price": round(base_price, 2),
                "bid": round(base_price - 0.01, 2),
                "ask": round(base_price + 0.01, 2),
                "volume": random.randint(1000000, 10000000),
                "timestamp": datetime.utcnow().isoformat()
            }

        return market_data