"""
💼 Portfolio Service
Service pour la gestion du portefeuille
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from decimal import Decimal

from qframe.core.container import injectable
from qframe.api.services.base_service import BaseService

logger = logging.getLogger(__name__)


@injectable
class PortfolioService(BaseService):
    """Service de gestion du portefeuille."""

    def __init__(self):
        super().__init__()
        self._initial_balance = 100000.0  # 100k USD initial
        self._current_balance = self._initial_balance
        self._positions = {}
        self._trades_history = []

    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Récupère le résumé du portefeuille."""
        try:
            total_value = await self._calculate_total_portfolio_value()
            total_pnl = total_value - self._initial_balance
            pnl_percentage = (total_pnl / self._initial_balance) * 100

            open_positions_count = len([p for p in self._positions.values() if p["status"] == "OPEN"])
            total_trades = len(self._trades_history)

            summary = {
                "initial_balance": self._initial_balance,
                "current_balance": self._current_balance,
                "total_value": total_value,
                "total_pnl": total_pnl,
                "pnl_percentage": pnl_percentage,
                "open_positions_count": open_positions_count,
                "total_trades": total_trades,
                "last_updated": datetime.utcnow().isoformat()
            }

            return summary

        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            raise

    async def get_portfolio_allocation(self) -> Dict[str, Any]:
        """Récupère l'allocation du portefeuille par asset."""
        try:
            allocations = {}
            total_value = await self._calculate_total_portfolio_value()

            # Cash allocation
            cash_percentage = (self._current_balance / total_value) * 100 if total_value > 0 else 0
            allocations["USD"] = {
                "value": self._current_balance,
                "percentage": cash_percentage,
                "type": "cash"
            }

            # Position allocations
            for position in self._positions.values():
                if position["status"] == "OPEN":
                    symbol = position["symbol"]
                    position_value = position["quantity"] * position["current_price"]
                    percentage = (position_value / total_value) * 100 if total_value > 0 else 0

                    if symbol in allocations:
                        allocations[symbol]["value"] += position_value
                        allocations[symbol]["percentage"] += percentage
                    else:
                        allocations[symbol] = {
                            "value": position_value,
                            "percentage": percentage,
                            "type": "position"
                        }

            return {
                "allocations": allocations,
                "total_value": total_value,
                "last_updated": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting portfolio allocation: {e}")
            raise

    async def get_portfolio_performance(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Récupère les performances du portefeuille."""
        try:
            if start_date is None:
                start_date = datetime.utcnow() - timedelta(days=30)
            if end_date is None:
                end_date = datetime.utcnow()

            # Filtrer les trades par période
            period_trades = [
                trade for trade in self._trades_history
                if start_date <= trade["timestamp"] <= end_date
            ]

            # Calculs de performance
            total_pnl = sum(trade.get("pnl", 0) for trade in period_trades)
            winning_trades = len([t for t in period_trades if t.get("pnl", 0) > 0])
            losing_trades = len([t for t in period_trades if t.get("pnl", 0) < 0])
            total_trades = len(period_trades)

            win_rate = (winning_trades / max(total_trades, 1)) * 100

            # Performance quotidienne simulée
            daily_returns = await self._generate_daily_returns(start_date, end_date)

            performance = {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "total_pnl": total_pnl,
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "average_trade_pnl": total_pnl / max(total_trades, 1),
                "daily_returns": daily_returns,
                "sharpe_ratio": await self._calculate_sharpe_ratio(daily_returns),
                "max_drawdown": await self._calculate_max_drawdown(daily_returns)
            }

            return performance

        except Exception as e:
            logger.error(f"Error getting portfolio performance: {e}")
            raise

    async def update_from_position(self, position: Dict[str, Any]) -> None:
        """Met à jour le portefeuille avec une position."""
        try:
            position_id = position["id"]
            self._positions[position_id] = position

            # Si position fermée, ajouter aux trades history
            if position["status"] == "CLOSED":
                trade = {
                    "id": position_id,
                    "symbol": position["symbol"],
                    "side": position["side"],
                    "quantity": position["quantity"],
                    "entry_price": position["entry_price"],
                    "exit_price": position.get("close_price"),
                    "pnl": position.get("realized_pnl", 0),
                    "commission": position.get("commission", 0),
                    "timestamp": position.get("closed_at", datetime.utcnow())
                }
                self._trades_history.append(trade)

                # Mettre à jour le solde
                self._current_balance += position.get("realized_pnl", 0)

        except Exception as e:
            logger.error(f"Error updating portfolio from position: {e}")
            raise

    async def _calculate_total_portfolio_value(self) -> float:
        """Calcule la valeur totale du portefeuille."""
        try:
            total_value = self._current_balance

            # Ajouter la valeur des positions ouvertes
            for position in self._positions.values():
                if position["status"] == "OPEN":
                    position_value = position["quantity"] * position["current_price"]
                    total_value += position_value

            return total_value

        except Exception as e:
            logger.error(f"Error calculating total portfolio value: {e}")
            return self._current_balance

    async def _generate_daily_returns(self, start_date: datetime, end_date: datetime) -> List[float]:
        """Génère les returns quotidiens (simulation)."""
        try:
            days = (end_date - start_date).days
            # Simulation de returns aléatoires avec une tendance légèrement positive
            import random
            random.seed(42)  # Pour la reproductibilité

            daily_returns = []
            for _ in range(days):
                # Return quotidien entre -2% et +3% avec biais positif
                daily_return = random.uniform(-0.02, 0.03)
                daily_returns.append(daily_return)

            return daily_returns

        except Exception as e:
            logger.error(f"Error generating daily returns: {e}")
            return []

    async def _calculate_sharpe_ratio(self, daily_returns: List[float]) -> float:
        """Calcule le ratio de Sharpe."""
        try:
            if not daily_returns:
                return 0.0

            import statistics

            avg_return = statistics.mean(daily_returns)
            std_return = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0.0

            # Ratio de Sharpe annualisé (252 jours de trading)
            if std_return == 0:
                return 0.0

            risk_free_rate = 0.02 / 252  # 2% annuel
            sharpe_ratio = (avg_return - risk_free_rate) / std_return * (252 ** 0.5)

            return sharpe_ratio

        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    async def _calculate_max_drawdown(self, daily_returns: List[float]) -> float:
        """Calcule le maximum drawdown."""
        try:
            if not daily_returns:
                return 0.0

            cumulative_returns = []
            cumulative = 1.0

            for daily_return in daily_returns:
                cumulative *= (1 + daily_return)
                cumulative_returns.append(cumulative)

            # Calcul du drawdown
            peak = cumulative_returns[0]
            max_drawdown = 0.0

            for value in cumulative_returns:
                if value > peak:
                    peak = value

                drawdown = (peak - value) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

            return max_drawdown

        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0

    async def get_risk_metrics(self) -> Dict[str, Any]:
        """Calcule les métriques de risque du portefeuille."""
        try:
            total_value = await self._calculate_total_portfolio_value()

            # VaR simulation (Value at Risk)
            var_95 = total_value * 0.05  # 5% VaR approximatif
            var_99 = total_value * 0.02  # 2% VaR approximatif

            # Exposition par asset
            asset_exposure = {}
            for position in self._positions.values():
                if position["status"] == "OPEN":
                    symbol = position["symbol"]
                    exposure = position["quantity"] * position["current_price"]
                    asset_exposure[symbol] = asset_exposure.get(symbol, 0) + exposure

            # Concentration risk
            max_exposure = max(asset_exposure.values()) if asset_exposure else 0
            concentration_ratio = (max_exposure / total_value) if total_value > 0 else 0

            metrics = {
                "total_value": total_value,
                "var_95": var_95,
                "var_99": var_99,
                "asset_exposure": asset_exposure,
                "concentration_ratio": concentration_ratio,
                "cash_ratio": self._current_balance / total_value if total_value > 0 else 1.0,
                "last_updated": datetime.utcnow().isoformat()
            }

            return metrics

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            raise