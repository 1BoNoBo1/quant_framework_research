"""
📊 Position Service
Service pour la gestion des positions de trading
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal

from qframe.core.container import injectable
from qframe.api.services.base_service import BaseService

logger = logging.getLogger(__name__)


@injectable
class PositionService(BaseService):
    """Service de gestion des positions."""

    def __init__(self):
        super().__init__()
        self._positions: Dict[str, Dict[str, Any]] = {}
        self._position_counter = 0

    async def get_positions(
        self,
        page: int = 1,
        per_page: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> tuple[List[Dict[str, Any]], int]:
        """Récupère la liste des positions avec pagination."""
        try:
            positions = list(self._positions.values())

            # Appliquer les filtres
            if filters:
                if filters.get("symbol"):
                    positions = [p for p in positions if p["symbol"] == filters["symbol"]]
                if filters.get("side"):
                    positions = [p for p in positions if p["side"] == filters["side"]]
                if filters.get("status"):
                    positions = [p for p in positions if p["status"] == filters["status"]]

            total = len(positions)

            # Pagination
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            paginated_positions = positions[start_idx:end_idx]

            return paginated_positions, total

        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            raise

    async def get_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        """Récupère une position spécifique."""
        position = self._positions.get(position_id)
        if position:
            # Mise à jour du PnL en temps réel
            await self._update_position_pnl(position)
        return position

    async def update_stop_loss(self, position_id: str, stop_loss_price: float) -> Dict[str, Any]:
        """Met à jour le stop loss d'une position."""
        try:
            position = self._positions.get(position_id)
            if not position:
                raise ValueError(f"Position {position_id} not found")

            position["stop_loss"] = stop_loss_price
            position["updated_at"] = datetime.utcnow()

            logger.info(f"Stop loss updated for position {position_id}: {stop_loss_price}")
            return position

        except Exception as e:
            logger.error(f"Error updating stop loss: {e}")
            raise

    async def update_take_profit(self, position_id: str, take_profit_price: float) -> Dict[str, Any]:
        """Met à jour le take profit d'une position."""
        try:
            position = self._positions.get(position_id)
            if not position:
                raise ValueError(f"Position {position_id} not found")

            position["take_profit"] = take_profit_price
            position["updated_at"] = datetime.utcnow()

            logger.info(f"Take profit updated for position {position_id}: {take_profit_price}")
            return position

        except Exception as e:
            logger.error(f"Error updating take profit: {e}")
            raise

    async def close_position(self, position_id: str, close_price: Optional[float] = None) -> Dict[str, Any]:
        """Ferme une position."""
        try:
            position = self._positions.get(position_id)
            if not position:
                raise ValueError(f"Position {position_id} not found")

            if position["status"] == "CLOSED":
                raise ValueError(f"Position {position_id} is already closed")

            # Prix de fermeture (market si non spécifié)
            if close_price is None:
                close_price = await self._get_current_market_price(position["symbol"])

            position["close_price"] = close_price
            position["status"] = "CLOSED"
            position["closed_at"] = datetime.utcnow()
            position["updated_at"] = datetime.utcnow()

            # Calcul du PnL final
            await self._calculate_final_pnl(position)

            logger.info(f"Position closed: {position_id} at {close_price}")
            return position

        except Exception as e:
            logger.error(f"Error closing position: {e}")
            raise

    async def close_all_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Ferme toutes les positions."""
        try:
            closed_positions = []

            for position_id, position in self._positions.items():
                if position["status"] == "OPEN":
                    if symbol is None or position["symbol"] == symbol:
                        closed_position = await self.close_position(position_id)
                        closed_positions.append(closed_position)

            logger.info(f"Closed {len(closed_positions)} positions")
            return closed_positions

        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            raise

    async def create_position_from_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Crée une position à partir d'un ordre exécuté."""
        try:
            if order["status"] != "FILLED":
                raise ValueError("Can only create position from filled orders")

            self._position_counter += 1
            position_id = f"pos_{self._position_counter:06d}"

            position = {
                "id": position_id,
                "symbol": order["symbol"],
                "side": "LONG" if order["side"] == "BUY" else "SHORT",
                "quantity": order["filled_quantity"],
                "entry_price": order["average_price"],
                "current_price": order["average_price"],
                "status": "OPEN",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "stop_loss": None,
                "take_profit": None,
                "unrealized_pnl": 0.0,
                "commission": order["commission"],
                "order_id": order["id"]
            }

            self._positions[position_id] = position
            logger.info(f"Position created: {position_id}")

            return position

        except Exception as e:
            logger.error(f"Error creating position from order: {e}")
            raise

    async def _update_position_pnl(self, position: Dict[str, Any]) -> None:
        """Met à jour le PnL d'une position."""
        try:
            if position["status"] != "OPEN":
                return

            current_price = await self._get_current_market_price(position["symbol"])
            position["current_price"] = current_price

            # Calcul PnL
            entry_price = position["entry_price"]
            quantity = position["quantity"]

            if position["side"] == "LONG":
                pnl = (current_price - entry_price) * quantity
            else:  # SHORT
                pnl = (entry_price - current_price) * quantity

            position["unrealized_pnl"] = pnl - position["commission"]
            position["updated_at"] = datetime.utcnow()

        except Exception as e:
            logger.error(f"Error updating position PnL: {e}")

    async def _calculate_final_pnl(self, position: Dict[str, Any]) -> None:
        """Calcule le PnL final d'une position fermée."""
        try:
            entry_price = position["entry_price"]
            close_price = position["close_price"]
            quantity = position["quantity"]

            if position["side"] == "LONG":
                pnl = (close_price - entry_price) * quantity
            else:  # SHORT
                pnl = (entry_price - close_price) * quantity

            position["realized_pnl"] = pnl - position["commission"]
            position["unrealized_pnl"] = 0.0

        except Exception as e:
            logger.error(f"Error calculating final PnL: {e}")

    async def _get_current_market_price(self, symbol: str) -> float:
        """Récupère le prix actuel du marché (simulation)."""
        # Simulation - en réalité, on appellerait un service de market data
        price_simulation = {
            "BTC/USD": 50000.0,
            "ETH/USD": 3000.0,
            "ADA/USD": 0.5,
            "SOL/USD": 100.0
        }
        return price_simulation.get(symbol, 1.0)

    async def get_pnl_analytics(self, period: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Analyse PnL pour une période donnée."""
        try:
            positions = list(self._positions.values())

            if symbol:
                positions = [p for p in positions if p["symbol"] == symbol]

            # Calculs d'analyse
            total_pnl = 0.0
            winning_trades = 0
            losing_trades = 0
            total_trades = len(positions)

            for position in positions:
                if position["status"] == "CLOSED":
                    pnl = position.get("realized_pnl", 0.0)
                    total_pnl += pnl
                    if pnl > 0:
                        winning_trades += 1
                    elif pnl < 0:
                        losing_trades += 1
                else:
                    # Position ouverte
                    await self._update_position_pnl(position)
                    total_pnl += position.get("unrealized_pnl", 0.0)

            win_rate = (winning_trades / max(total_trades, 1)) * 100

            return {
                "period": period,
                "symbol": symbol,
                "total_pnl": total_pnl,
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "average_pnl": total_pnl / max(total_trades, 1)
            }

        except Exception as e:
            logger.error(f"Error calculating PnL analytics: {e}")
            raise