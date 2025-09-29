"""
🎯 Strategy Service
Service pour la gestion des stratégies de trading
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

from qframe.core.container import injectable
from qframe.api.services.base_service import BaseService

logger = logging.getLogger(__name__)


@injectable
class StrategyService(BaseService):
    """Service de gestion des stratégies."""

    def __init__(self):
        super().__init__()
        self._strategies = {}
        self._running_strategies = {}
        self._strategy_counter = 0
        self._init_default_strategies()

    def _init_default_strategies(self):
        """Initialise les stratégies par défaut."""
        default_strategies = [
            {
                "name": "Mean Reversion",
                "description": "Stratégie de retour à la moyenne adaptative",
                "type": "STATISTICAL",
                "parameters": {
                    "lookback_period": 20,
                    "z_threshold": 2.0,
                    "stop_loss": 0.02,
                    "take_profit": 0.03
                },
                "risk_profile": "MEDIUM",
                "expected_return": 0.15,
                "max_drawdown": 0.08
            },
            {
                "name": "Momentum Breakout",
                "description": "Stratégie de momentum avec breakout",
                "type": "TREND",
                "parameters": {
                    "lookback_period": 50,
                    "breakout_threshold": 1.5,
                    "volume_filter": True,
                    "stop_loss": 0.03,
                    "take_profit": 0.05
                },
                "risk_profile": "HIGH",
                "expected_return": 0.25,
                "max_drawdown": 0.15
            },
            {
                "name": "Grid Trading",
                "description": "Stratégie de grid trading automatisé",
                "type": "ARBITRAGE",
                "parameters": {
                    "grid_size": 0.01,
                    "num_levels": 10,
                    "base_order_size": 100,
                    "take_profit": 0.02
                },
                "risk_profile": "LOW",
                "expected_return": 0.12,
                "max_drawdown": 0.05
            }
        ]

        for i, strategy_data in enumerate(default_strategies):
            strategy_id = f"strategy_{i+1:03d}"
            strategy = {
                "id": strategy_id,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "status": "INACTIVE",
                "performance": {
                    "total_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "win_rate": 0.0,
                    "total_trades": 0
                },
                **strategy_data
            }
            self._strategies[strategy_id] = strategy

    async def get_strategies(
        self,
        strategy_type: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Récupère la liste des stratégies."""
        try:
            strategies = list(self._strategies.values())

            # Filtrage
            if strategy_type:
                strategies = [s for s in strategies if s["type"] == strategy_type]
            if status:
                strategies = [s for s in strategies if s["status"] == status]

            return strategies

        except Exception as e:
            logger.error(f"Error getting strategies: {e}")
            raise

    async def get_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Récupère une stratégie spécifique."""
        return self._strategies.get(strategy_id)

    async def create_strategy(self, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Crée une nouvelle stratégie."""
        try:
            self._strategy_counter += 1
            strategy_id = f"custom_{self._strategy_counter:03d}"

            strategy = {
                "id": strategy_id,
                "name": strategy_data["name"],
                "description": strategy_data.get("description", ""),
                "type": strategy_data.get("type", "CUSTOM"),
                "parameters": strategy_data.get("parameters", {}),
                "risk_profile": strategy_data.get("risk_profile", "MEDIUM"),
                "expected_return": strategy_data.get("expected_return", 0.0),
                "max_drawdown": strategy_data.get("max_drawdown", 0.0),
                "status": "INACTIVE",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "performance": {
                    "total_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "win_rate": 0.0,
                    "total_trades": 0
                }
            }

            self._strategies[strategy_id] = strategy
            logger.info(f"Strategy created: {strategy_id}")

            return strategy

        except Exception as e:
            logger.error(f"Error creating strategy: {e}")
            raise

    async def update_strategy(
        self,
        strategy_id: str,
        update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Met à jour une stratégie."""
        try:
            strategy = self._strategies.get(strategy_id)
            if not strategy:
                raise ValueError(f"Strategy {strategy_id} not found")

            if strategy["status"] == "RUNNING":
                raise ValueError("Cannot update running strategy")

            # Mise à jour des champs autorisés
            updatable_fields = ["name", "description", "parameters", "risk_profile"]
            for field in updatable_fields:
                if field in update_data:
                    strategy[field] = update_data[field]

            strategy["updated_at"] = datetime.utcnow()

            logger.info(f"Strategy updated: {strategy_id}")
            return strategy

        except Exception as e:
            logger.error(f"Error updating strategy: {e}")
            raise

    async def start_strategy(self, strategy_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Démarre l'exécution d'une stratégie."""
        try:
            strategy = self._strategies.get(strategy_id)
            if not strategy:
                raise ValueError(f"Strategy {strategy_id} not found")

            if strategy["status"] == "RUNNING":
                raise ValueError("Strategy is already running")

            # Configuration d'exécution
            execution_config = {
                "symbols": config.get("symbols", ["BTC/USD"]),
                "capital_allocation": config.get("capital_allocation", 10000),
                "max_positions": config.get("max_positions", 5),
                "started_at": datetime.utcnow(),
                "auto_restart": config.get("auto_restart", False)
            }

            strategy["status"] = "RUNNING"
            strategy["execution_config"] = execution_config
            strategy["updated_at"] = datetime.utcnow()

            self._running_strategies[strategy_id] = {
                "strategy": strategy,
                "runtime_stats": {
                    "signals_generated": 0,
                    "orders_placed": 0,
                    "current_positions": 0,
                    "uptime": 0.0
                }
            }

            logger.info(f"Strategy started: {strategy_id}")
            return strategy

        except Exception as e:
            logger.error(f"Error starting strategy: {e}")
            raise

    async def stop_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """Arrête l'exécution d'une stratégie."""
        try:
            strategy = self._strategies.get(strategy_id)
            if not strategy:
                raise ValueError(f"Strategy {strategy_id} not found")

            if strategy["status"] != "RUNNING":
                raise ValueError("Strategy is not running")

            strategy["status"] = "STOPPED"
            strategy["updated_at"] = datetime.utcnow()

            # Retirer des stratégies en cours
            if strategy_id in self._running_strategies:
                del self._running_strategies[strategy_id]

            logger.info(f"Strategy stopped: {strategy_id}")
            return strategy

        except Exception as e:
            logger.error(f"Error stopping strategy: {e}")
            raise

    async def get_strategy_performance(
        self,
        strategy_id: str,
        period: Optional[str] = None
    ) -> Dict[str, Any]:
        """Récupère les performances d'une stratégie."""
        try:
            strategy = self._strategies.get(strategy_id)
            if not strategy:
                raise ValueError(f"Strategy {strategy_id} not found")

            # Simulation de données de performance
            base_performance = strategy["performance"].copy()

            # Ajouter des métriques temporelles simulées
            if strategy["status"] == "RUNNING":
                runtime_hours = (datetime.utcnow() - strategy.get("execution_config", {}).get("started_at", datetime.utcnow())).total_seconds() / 3600

                # Simulation de performance basée sur le profil de risque
                risk_multiplier = {"LOW": 0.8, "MEDIUM": 1.0, "HIGH": 1.3}.get(strategy["risk_profile"], 1.0)

                base_performance.update({
                    "total_return": 0.05 * risk_multiplier * (runtime_hours / 24),
                    "sharpe_ratio": 1.2 * risk_multiplier,
                    "win_rate": 65.0 if risk_multiplier > 1 else 55.0,
                    "total_trades": max(1, int(runtime_hours / 2)),
                    "avg_trade_duration": "4.2h",
                    "max_drawdown": strategy["max_drawdown"] * 0.7,
                    "profit_factor": 1.4 * risk_multiplier,
                    "recovery_factor": 2.1,
                    "calmar_ratio": 0.8 * risk_multiplier
                })

            # Ajouter des métriques par période
            period_metrics = await self._calculate_period_metrics(strategy_id, period)
            base_performance.update(period_metrics)

            return {
                "strategy_id": strategy_id,
                "strategy_name": strategy["name"],
                "period": period or "total",
                "performance": base_performance,
                "calculated_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting strategy performance: {e}")
            raise

    async def get_running_strategies(self) -> Dict[str, Any]:
        """Récupère les statistiques des stratégies en cours d'exécution."""
        try:
            running_stats = {}

            for strategy_id, running_data in self._running_strategies.items():
                strategy = running_data["strategy"]
                runtime_stats = running_data["runtime_stats"]

                # Calculer l'uptime
                started_at = strategy.get("execution_config", {}).get("started_at", datetime.utcnow())
                uptime_hours = (datetime.utcnow() - started_at).total_seconds() / 3600

                stats = {
                    "strategy_name": strategy["name"],
                    "status": strategy["status"],
                    "uptime_hours": uptime_hours,
                    "signals_generated": runtime_stats["signals_generated"],
                    "orders_placed": runtime_stats["orders_placed"],
                    "current_positions": runtime_stats["current_positions"],
                    "capital_allocated": strategy.get("execution_config", {}).get("capital_allocation", 0),
                    "current_return": strategy["performance"]["total_return"]
                }

                running_stats[strategy_id] = stats

            return {
                "total_running": len(running_stats),
                "strategies": running_stats,
                "last_updated": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting running strategies: {e}")
            raise

    async def _calculate_period_metrics(
        self,
        strategy_id: str,
        period: Optional[str]
    ) -> Dict[str, Any]:
        """Calcule les métriques pour une période spécifique."""
        try:
            # Simulation de métriques par période
            if not period or period == "total":
                return {}

            period_multipliers = {
                "1d": 1.0 / 365,
                "7d": 7.0 / 365,
                "30d": 30.0 / 365,
                "90d": 90.0 / 365
            }

            multiplier = period_multipliers.get(period, 1.0)

            return {
                "period_return": 0.08 * multiplier,
                "period_volatility": 0.12 * multiplier,
                "period_trades": max(1, int(50 * multiplier)),
                "period_win_rate": 58.0,
                "best_trade": 0.05 * multiplier,
                "worst_trade": -0.02 * multiplier
            }

        except Exception as e:
            logger.error(f"Error calculating period metrics: {e}")
            return {}

    async def delete_strategy(self, strategy_id: str) -> bool:
        """Supprime une stratégie."""
        try:
            strategy = self._strategies.get(strategy_id)
            if not strategy:
                return False

            if strategy["status"] == "RUNNING":
                raise ValueError("Cannot delete running strategy")

            del self._strategies[strategy_id]
            logger.info(f"Strategy deleted: {strategy_id}")

            return True

        except Exception as e:
            logger.error(f"Error deleting strategy: {e}")
            raise