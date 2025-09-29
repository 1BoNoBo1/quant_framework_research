"""
Live Trading Engine
==================

Production-ready trading engine with real broker connections,
order management, and position reconciliation.
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from ...core.interfaces import Strategy
from ...core.container import injectable
from ...domain.entities.order import Order, OrderStatus
from ...domain.entities.position import Position
from ...domain.value_objects.signal import Signal
from ...strategies.orchestration.multi_strategy_manager import MultiStrategyManager
from .broker_adapters import BrokerAdapter
from .order_manager import LiveOrderManager
from .position_reconciler import PositionReconciler


logger = logging.getLogger(__name__)


class TradingMode(str, Enum):
    """Modes de trading"""
    PAPER = "paper"
    LIVE = "live"
    SIMULATION = "simulation"


@dataclass
class TradingSession:
    """Session de trading active"""
    session_id: str
    mode: TradingMode
    start_time: datetime
    strategies_active: List[str]
    total_pnl: Decimal = Decimal("0")
    trades_executed: int = 0
    is_active: bool = True


@injectable
class LiveTradingEngine:
    """
    Moteur de trading en production.

    Responsabilités:
    - Orchestration trading temps réel
    - Gestion ordres avec retry logic
    - Réconciliation positions automatique
    - Monitoring performance continue
    """

    def __init__(
        self,
        multi_strategy_manager: MultiStrategyManager,
        broker_adapter: BrokerAdapter,
        order_manager: LiveOrderManager,
        position_reconciler: PositionReconciler,
        trading_mode: TradingMode = TradingMode.PAPER
    ):
        self.multi_strategy_manager = multi_strategy_manager
        self.broker_adapter = broker_adapter
        self.order_manager = order_manager
        self.position_reconciler = position_reconciler
        self.trading_mode = trading_mode

        # État de trading
        self.current_session: Optional[TradingSession] = None
        self.market_data_cache: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Any] = {}

        # Configuration
        self.heartbeat_interval = timedelta(seconds=30)
        self.reconciliation_interval = timedelta(minutes=5)
        self.max_retry_attempts = 3

        # Threading pour operations async
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Callbacks
        self.on_trade_executed: Optional[Callable] = None
        self.on_error: Optional[Callable] = None

    async def start_trading_session(
        self,
        session_config: Dict[str, Any]
    ) -> str:
        """
        Démarre une session de trading.

        Args:
            session_config: Configuration de la session

        Returns:
            ID de la session créée
        """
        if self.current_session and self.current_session.is_active:
            raise RuntimeError("Trading session already active")

        # Créer nouvelle session
        session_id = f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        self.current_session = TradingSession(
            session_id=session_id,
            mode=self.trading_mode,
            start_time=datetime.utcnow(),
            strategies_active=session_config.get("strategies", [])
        )

        # Initialiser composants
        await self.broker_adapter.connect()
        await self.order_manager.initialize()

        # Réconciliation initiale
        await self.position_reconciler.reconcile_positions()

        # Démarrer boucles de monitoring
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._reconciliation_loop())
        asyncio.create_task(self._trading_loop())

        logger.info(f"Trading session {session_id} started in {self.trading_mode.value} mode")
        return session_id

    async def stop_trading_session(self) -> None:
        """Arrête la session de trading en cours."""
        if not self.current_session:
            return

        # Annuler ordres en cours
        await self.order_manager.cancel_all_orders()

        # Fermer connexions
        await self.broker_adapter.disconnect()

        # Finaliser session
        self.current_session.is_active = False

        logger.info(f"Trading session {self.current_session.session_id} stopped")

    async def get_session_status(self) -> Dict[str, Any]:
        """Retourne le statut de la session actuelle."""
        if not self.current_session:
            return {"status": "no_active_session"}

        session_duration = datetime.utcnow() - self.current_session.start_time

        return {
            "session_id": self.current_session.session_id,
            "mode": self.current_session.mode.value,
            "duration_minutes": int(session_duration.total_seconds() / 60),
            "strategies_active": self.current_session.strategies_active,
            "total_pnl": float(self.current_session.total_pnl),
            "trades_executed": self.current_session.trades_executed,
            "is_active": self.current_session.is_active,
            "broker_status": await self.broker_adapter.get_connection_status(),
            "pending_orders": await self.order_manager.get_pending_orders_count()
        }

    async def emergency_stop(self, reason: str) -> None:
        """
        Arrêt d'urgence complet.

        Args:
            reason: Raison de l'arrêt d'urgence
        """
        logger.critical(f"EMERGENCY STOP triggered: {reason}")

        try:
            # Annuler tous les ordres immédiatement
            await self.order_manager.emergency_cancel_all()

            # Fermer toutes les positions si configuré
            if self.trading_mode == TradingMode.LIVE:
                await self._emergency_close_positions()

            # Arrêter session
            await self.stop_trading_session()

            # Notifier
            if self.on_error:
                await self.on_error("emergency_stop", reason)

        except Exception as e:
            logger.critical(f"Error during emergency stop: {e}")

    # === Boucles de trading ===

    async def _trading_loop(self) -> None:
        """Boucle principale de génération et exécution de signaux."""
        while self.current_session and self.current_session.is_active:
            try:
                # Obtenir données de marché
                market_data = await self._fetch_market_data()

                # Générer signaux unifiés
                signals = await self.multi_strategy_manager.generate_unified_signals(
                    market_data,
                    await self._get_current_portfolio()
                )

                # Exécuter signaux
                for signal in signals:
                    await self._execute_signal(signal)

                # Attendre prochain cycle
                await asyncio.sleep(1.0)  # 1 seconde entre cycles

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5.0)  # Pause plus longue en cas d'erreur

    async def _heartbeat_loop(self) -> None:
        """Boucle de heartbeat pour vérifier la santé du système."""
        while self.current_session and self.current_session.is_active:
            try:
                # Vérifier connexion broker
                if not await self.broker_adapter.is_connected():
                    logger.warning("Broker connection lost, attempting reconnect")
                    await self.broker_adapter.connect()

                # Vérifier ordres suspendus
                stuck_orders = await self.order_manager.check_stuck_orders()
                if stuck_orders:
                    logger.warning(f"Found {len(stuck_orders)} stuck orders")

                # Mettre à jour métriques
                await self._update_performance_metrics()

                await asyncio.sleep(self.heartbeat_interval.total_seconds())

            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(30.0)

    async def _reconciliation_loop(self) -> None:
        """Boucle de réconciliation des positions."""
        while self.current_session and self.current_session.is_active:
            try:
                await self.position_reconciler.reconcile_positions()
                await asyncio.sleep(self.reconciliation_interval.total_seconds())

            except Exception as e:
                logger.error(f"Error in reconciliation loop: {e}")
                await asyncio.sleep(60.0)

    # === Méthodes privées ===

    async def _execute_signal(self, signal: Signal) -> None:
        """Exécute un signal de trading."""
        try:
            # Convertir signal en ordre
            order = await self._signal_to_order(signal)

            # Soumettre ordre
            execution_result = await self.order_manager.submit_order(order)

            if execution_result.success:
                self.current_session.trades_executed += 1

                if self.on_trade_executed:
                    await self.on_trade_executed(signal, execution_result)

                logger.info(f"Signal executed: {signal.symbol} {signal.action}")
            else:
                logger.warning(f"Signal execution failed: {execution_result.error}")

        except Exception as e:
            logger.error(f"Error executing signal {signal.symbol}: {e}")

    async def _signal_to_order(self, signal: Signal) -> Order:
        """Convertit un signal en ordre."""
        # Logique de conversion simplifiée
        return Order(
            symbol=signal.symbol,
            side="buy" if signal.action == "buy" else "sell",
            quantity=signal.quantity or Decimal("1"),
            order_type="market",
            timestamp=datetime.utcnow()
        )

    async def _fetch_market_data(self) -> Dict[str, Any]:
        """Récupère les données de marché actuelles."""
        # Cache simple pour éviter trop d'appels API
        current_time = datetime.utcnow()
        cache_key = "market_data"

        if (cache_key in self.market_data_cache and
            (current_time - self.market_data_cache[cache_key]['timestamp']).total_seconds() < 5):
            return self.market_data_cache[cache_key]['data']

        # Récupérer nouvelles données
        market_data = await self.broker_adapter.get_market_data()

        self.market_data_cache[cache_key] = {
            'data': market_data,
            'timestamp': current_time
        }

        return market_data

    async def _get_current_portfolio(self):
        """Récupère le portfolio actuel."""
        return await self.broker_adapter.get_portfolio()

    async def _update_performance_metrics(self) -> None:
        """Met à jour les métriques de performance."""
        if not self.current_session:
            return

        portfolio = await self._get_current_portfolio()
        current_value = getattr(portfolio, 'total_value', Decimal("0"))

        # Calculer PnL de session
        # (logique simplifiée)
        self.current_session.total_pnl = current_value

        # Mettre à jour métriques des stratégies
        allocations = await self.multi_strategy_manager.get_strategy_allocations()
        for strategy_id, allocation in allocations.items():
            await self.multi_strategy_manager.update_performance_metrics(
                strategy_id,
                current_value,
                allocation.performance_metrics.total_return
            )

    async def _emergency_close_positions(self) -> None:
        """Ferme toutes les positions en urgence."""
        positions = await self.broker_adapter.get_positions()

        for position in positions:
            if position.quantity != 0:
                # Créer ordre de fermeture
                close_order = Order(
                    symbol=position.symbol,
                    side="sell" if position.quantity > 0 else "buy",
                    quantity=abs(position.quantity),
                    order_type="market",
                    timestamp=datetime.utcnow()
                )

                await self.order_manager.submit_order(close_order)
                logger.warning(f"Emergency close position: {position.symbol}")