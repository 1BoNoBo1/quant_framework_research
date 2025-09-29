"""
📈 Backtest Service
Service pour les backtests de stratégies
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
from decimal import Decimal

from qframe.core.container import injectable
from qframe.api.services.base_service import BaseService

logger = logging.getLogger(__name__)


@injectable
class BacktestService(BaseService):
    """Service de backtesting."""

    def __init__(self):
        super().__init__()
        self._backtests = {}
        self._backtest_counter = 0

    async def create_backtest(self, backtest_config: Dict[str, Any]) -> Dict[str, Any]:
        """Crée et lance un nouveau backtest."""
        try:
            self._backtest_counter += 1
            backtest_id = f"backtest_{self._backtest_counter:06d}"

            backtest = {
                "id": backtest_id,
                "strategy_id": backtest_config["strategy_id"],
                "strategy_name": backtest_config.get("strategy_name", "Unknown"),
                "config": backtest_config,
                "status": "RUNNING",
                "created_at": datetime.utcnow(),
                "started_at": datetime.utcnow(),
                "completed_at": None,
                "progress": 0.0,
                "results": None,
                "error": None
            }

            self._backtests[backtest_id] = backtest

            # Lancer le backtest de manière asynchrone
            await self._run_backtest(backtest_id)

            logger.info(f"Backtest created and started: {backtest_id}")
            return backtest

        except Exception as e:
            logger.error(f"Error creating backtest: {e}")
            raise

    async def get_backtest(self, backtest_id: str) -> Optional[Dict[str, Any]]:
        """Récupère un backtest spécifique."""
        return self._backtests.get(backtest_id)

    async def get_backtests(
        self,
        strategy_id: Optional[str] = None,
        status: Optional[str] = None,
        page: int = 1,
        per_page: int = 20
    ) -> tuple[List[Dict[str, Any]], int]:
        """Récupère la liste des backtests."""
        try:
            backtests = list(self._backtests.values())

            # Filtrage
            if strategy_id:
                backtests = [b for b in backtests if b["strategy_id"] == strategy_id]
            if status:
                backtests = [b for b in backtests if b["status"] == status]

            # Tri par date de création (plus récent en premier)
            backtests.sort(key=lambda x: x["created_at"], reverse=True)

            total = len(backtests)

            # Pagination
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            paginated_backtests = backtests[start_idx:end_idx]

            return paginated_backtests, total

        except Exception as e:
            logger.error(f"Error getting backtests: {e}")
            raise

    async def cancel_backtest(self, backtest_id: str) -> Dict[str, Any]:
        """Annule un backtest en cours."""
        try:
            backtest = self._backtests.get(backtest_id)
            if not backtest:
                raise ValueError(f"Backtest {backtest_id} not found")

            if backtest["status"] not in ["RUNNING", "PENDING"]:
                raise ValueError(f"Cannot cancel backtest with status {backtest['status']}")

            backtest["status"] = "CANCELLED"
            backtest["completed_at"] = datetime.utcnow()

            logger.info(f"Backtest cancelled: {backtest_id}")
            return backtest

        except Exception as e:
            logger.error(f"Error cancelling backtest: {e}")
            raise

    async def get_backtest_results(self, backtest_id: str) -> Dict[str, Any]:
        """Récupère les résultats détaillés d'un backtest."""
        try:
            backtest = self._backtests.get(backtest_id)
            if not backtest:
                raise ValueError(f"Backtest {backtest_id} not found")

            if backtest["status"] != "COMPLETED":
                raise ValueError(f"Backtest not completed. Status: {backtest['status']}")

            return backtest["results"]

        except Exception as e:
            logger.error(f"Error getting backtest results: {e}")
            raise

    async def compare_backtests(self, backtest_ids: List[str]) -> Dict[str, Any]:
        """Compare les résultats de plusieurs backtests."""
        try:
            comparison = {
                "backtests": [],
                "comparison_metrics": {},
                "generated_at": datetime.utcnow().isoformat()
            }

            for backtest_id in backtest_ids:
                backtest = self._backtests.get(backtest_id)
                if not backtest or backtest["status"] != "COMPLETED":
                    continue

                results = backtest["results"]
                comparison["backtests"].append({
                    "id": backtest_id,
                    "strategy_name": backtest["strategy_name"],
                    "total_return": results["performance"]["total_return"],
                    "sharpe_ratio": results["performance"]["sharpe_ratio"],
                    "max_drawdown": results["performance"]["max_drawdown"],
                    "win_rate": results["performance"]["win_rate"],
                    "total_trades": results["trade_stats"]["total_trades"]
                })

            # Calculer les métriques de comparaison
            if comparison["backtests"]:
                metrics = {}
                for metric in ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"]:
                    values = [b[metric] for b in comparison["backtests"]]
                    metrics[metric] = {
                        "best": max(values) if metric != "max_drawdown" else min(values),
                        "worst": min(values) if metric != "max_drawdown" else max(values),
                        "average": sum(values) / len(values)
                    }

                comparison["comparison_metrics"] = metrics

            return comparison

        except Exception as e:
            logger.error(f"Error comparing backtests: {e}")
            raise

    async def _run_backtest(self, backtest_id: str) -> None:
        """Exécute un backtest (simulation)."""
        try:
            backtest = self._backtests[backtest_id]
            config = backtest["config"]

            # Simulation du processus de backtest
            import asyncio
            import random

            # Paramètres de simulation
            start_date = datetime.fromisoformat(config["start_date"].replace("Z", "+00:00"))
            end_date = datetime.fromisoformat(config["end_date"].replace("Z", "+00:00"))
            initial_capital = config.get("initial_capital", 10000)

            # Simuler le progrès
            for progress in range(0, 101, 10):
                backtest["progress"] = progress
                await asyncio.sleep(0.1)  # Simulation du temps de calcul

            # Générer des résultats simulés
            random.seed(42)  # Pour la reproductibilité

            # Simulation de performance basée sur la stratégie
            strategy_performance_map = {
                "Mean Reversion": {"return": 0.15, "sharpe": 1.2, "drawdown": 0.08, "win_rate": 0.62},
                "Momentum Breakout": {"return": 0.25, "sharpe": 1.5, "drawdown": 0.15, "win_rate": 0.58},
                "Grid Trading": {"return": 0.12, "sharpe": 0.9, "drawdown": 0.05, "win_rate": 0.75}
            }

            strategy_name = backtest["strategy_name"]
            base_perf = strategy_performance_map.get(strategy_name,
                {"return": 0.10, "sharpe": 1.0, "drawdown": 0.10, "win_rate": 0.60})

            # Ajouter un peu de randomness
            noise_factor = random.uniform(0.8, 1.2)

            total_return = base_perf["return"] * noise_factor
            final_capital = initial_capital * (1 + total_return)
            total_trades = random.randint(50, 200)
            winning_trades = int(total_trades * base_perf["win_rate"] * noise_factor)

            results = {
                "performance": {
                    "initial_capital": initial_capital,
                    "final_capital": final_capital,
                    "total_return": total_return,
                    "total_return_percent": total_return * 100,
                    "annualized_return": total_return * (365 / (end_date - start_date).days),
                    "sharpe_ratio": base_perf["sharpe"] * noise_factor,
                    "sortino_ratio": base_perf["sharpe"] * noise_factor * 1.2,
                    "max_drawdown": base_perf["drawdown"] * noise_factor,
                    "max_drawdown_percent": base_perf["drawdown"] * noise_factor * 100,
                    "calmar_ratio": (total_return) / max(base_perf["drawdown"] * noise_factor, 0.01),
                    "win_rate": base_perf["win_rate"] * noise_factor * 100,
                    "profit_factor": 1.4 * noise_factor,
                    "recovery_factor": 2.1 * noise_factor
                },
                "trade_stats": {
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "losing_trades": total_trades - winning_trades,
                    "win_rate": (winning_trades / total_trades) * 100,
                    "average_win": (total_return * initial_capital) / max(winning_trades, 1),
                    "average_loss": -(total_return * initial_capital * 0.3) / max(total_trades - winning_trades, 1),
                    "largest_win": total_return * initial_capital * 0.15,
                    "largest_loss": -total_return * initial_capital * 0.08,
                    "average_trade_duration": f"{random.uniform(2, 48):.1f}h"
                },
                "risk_metrics": {
                    "volatility": base_perf["drawdown"] * 2 * noise_factor,
                    "var_95": initial_capital * 0.05 * noise_factor,
                    "expected_shortfall": initial_capital * 0.08 * noise_factor,
                    "beta": random.uniform(0.8, 1.5),
                    "correlation_to_market": random.uniform(0.2, 0.8)
                },
                "monthly_returns": self._generate_monthly_returns(start_date, end_date, total_return),
                "drawdown_periods": self._generate_drawdown_periods(base_perf["drawdown"] * noise_factor),
                "config": config,
                "execution_time": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "duration_days": (end_date - start_date).days,
                    "backtest_duration_seconds": 1.0  # Simulation rapide
                }
            }

            # Finaliser le backtest
            backtest["status"] = "COMPLETED"
            backtest["completed_at"] = datetime.utcnow()
            backtest["progress"] = 100.0
            backtest["results"] = results

            logger.info(f"Backtest completed: {backtest_id}")

        except Exception as e:
            backtest["status"] = "FAILED"
            backtest["error"] = str(e)
            backtest["completed_at"] = datetime.utcnow()
            logger.error(f"Backtest failed: {backtest_id} - {e}")

    def _generate_monthly_returns(self, start_date: datetime, end_date: datetime, total_return: float) -> List[Dict[str, Any]]:
        """Génère des returns mensuels simulés."""
        import random
        random.seed(42)

        monthly_returns = []
        current_date = start_date.replace(day=1)  # Premier du mois
        months = []

        while current_date < end_date:
            next_month = current_date.replace(month=current_date.month + 1) if current_date.month < 12 else current_date.replace(year=current_date.year + 1, month=1)
            months.append(current_date)
            current_date = next_month

        # Distribuer le return total sur les mois avec variation
        num_months = len(months)
        if num_months == 0:
            return []

        avg_monthly_return = total_return / num_months

        for i, month in enumerate(months):
            # Ajouter de la variabilité
            noise = random.uniform(-0.5, 0.5)
            monthly_return = avg_monthly_return * (1 + noise)

            monthly_returns.append({
                "month": month.strftime("%Y-%m"),
                "return": monthly_return,
                "return_percent": monthly_return * 100
            })

        return monthly_returns

    def _generate_drawdown_periods(self, max_drawdown: float) -> List[Dict[str, Any]]:
        """Génère des périodes de drawdown simulées."""
        import random
        random.seed(42)

        drawdown_periods = []

        # Générer 2-4 périodes de drawdown
        num_periods = random.randint(2, 4)

        for i in range(num_periods):
            duration_days = random.randint(5, 30)
            drawdown_magnitude = max_drawdown * random.uniform(0.3, 1.0)

            start_date = datetime.utcnow() - timedelta(days=random.randint(30, 200))
            end_date = start_date + timedelta(days=duration_days)

            drawdown_periods.append({
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "duration_days": duration_days,
                "max_drawdown": drawdown_magnitude,
                "max_drawdown_percent": drawdown_magnitude * 100
            })

        return drawdown_periods

    async def get_backtest_summary(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """Récupère un résumé des backtests."""
        try:
            backtests = list(self._backtests.values())

            if strategy_id:
                backtests = [b for b in backtests if b["strategy_id"] == strategy_id]

            completed_backtests = [b for b in backtests if b["status"] == "COMPLETED"]

            summary = {
                "total_backtests": len(backtests),
                "completed_backtests": len(completed_backtests),
                "running_backtests": len([b for b in backtests if b["status"] == "RUNNING"]),
                "failed_backtests": len([b for b in backtests if b["status"] == "FAILED"]),
                "strategy_id": strategy_id,
                "generated_at": datetime.utcnow().isoformat()
            }

            if completed_backtests:
                # Statistiques des backtests complétés
                returns = [b["results"]["performance"]["total_return"] for b in completed_backtests]
                sharpe_ratios = [b["results"]["performance"]["sharpe_ratio"] for b in completed_backtests]

                summary["performance_stats"] = {
                    "avg_return": sum(returns) / len(returns),
                    "best_return": max(returns),
                    "worst_return": min(returns),
                    "avg_sharpe": sum(sharpe_ratios) / len(sharpe_ratios),
                    "best_sharpe": max(sharpe_ratios)
                }

            return summary

        except Exception as e:
            logger.error(f"Error getting backtest summary: {e}")
            raise