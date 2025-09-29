"""
⚠️ Risk Service
Service pour la gestion des risques
"""

import logging
import math
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np

from qframe.core.container import injectable
from qframe.api.services.base_service import BaseService

logger = logging.getLogger(__name__)


@injectable
class RiskService(BaseService):
    """Service de gestion des risques."""

    def __init__(self):
        super().__init__()
        self._risk_limits = {
            "max_position_size": 0.1,  # 10% max par position
            "max_total_exposure": 0.8,  # 80% max exposition totale
            "max_daily_loss": 0.05,    # 5% max perte quotidienne
            "max_correlation": 0.7     # Corrélation max entre positions
        }
        self._alerts = []

    async def get_current_risk_metrics(self) -> Dict[str, Any]:
        """Récupère les métriques de risque actuelles."""
        try:
            # Simulation des métriques de risque
            total_value = 100000.0  # Valeur portfolio simulée

            metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "portfolio_value": total_value,
                "var_95_1d": total_value * 0.02,  # VaR 95% 1 jour
                "var_99_1d": total_value * 0.05,  # VaR 99% 1 jour
                "expected_shortfall": total_value * 0.03,
                "beta": 1.2,  # Beta par rapport au marché
                "volatility": 0.15,  # Volatilité annualisée
                "maximum_drawdown": 0.08,
                "sharpe_ratio": 1.5,
                "sortino_ratio": 1.8,
                "exposure_by_asset": {
                    "BTC": 0.3,
                    "ETH": 0.2,
                    "Cash": 0.5
                },
                "concentration_risk": 0.3,  # Concentration maximale
                "correlation_risk": 0.4,
                "leverage_ratio": 1.0,
                "liquidity_risk": "LOW",
                "risk_score": 65  # Score sur 100
            }

            return metrics

        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            raise

    async def calculate_var(
        self,
        confidence_level: float = 0.95,
        time_horizon: int = 1,
        method: str = "monte_carlo"
    ) -> Dict[str, Any]:
        """Calcule la Value at Risk (VaR)."""
        try:
            portfolio_value = 100000.0  # Simulation

            if method == "monte_carlo":
                var_value = await self._calculate_var_monte_carlo(
                    portfolio_value, confidence_level, time_horizon
                )
            elif method == "historical":
                var_value = await self._calculate_var_historical(
                    portfolio_value, confidence_level, time_horizon
                )
            elif method == "parametric":
                var_value = await self._calculate_var_parametric(
                    portfolio_value, confidence_level, time_horizon
                )
            else:
                raise ValueError(f"Unknown VaR method: {method}")

            result = {
                "method": method,
                "confidence_level": confidence_level,
                "time_horizon_days": time_horizon,
                "portfolio_value": portfolio_value,
                "var_absolute": var_value,
                "var_percentage": (var_value / portfolio_value) * 100,
                "expected_shortfall": var_value * 1.3,  # Approximation
                "calculated_at": datetime.utcnow().isoformat()
            }

            return result

        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            raise

    async def stress_test_portfolio(
        self,
        scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Effectue des tests de stress sur le portefeuille."""
        try:
            results = {}
            portfolio_value = 100000.0

            for scenario in scenarios:
                scenario_name = scenario["name"]
                price_shocks = scenario.get("price_shocks", {})

                # Simulation de l'impact du scénario
                scenario_loss = 0.0
                for asset, shock_percentage in price_shocks.items():
                    # Supposer une exposition de 30% à chaque asset principal
                    exposure = portfolio_value * 0.3
                    asset_loss = exposure * abs(shock_percentage) / 100
                    scenario_loss += asset_loss

                results[scenario_name] = {
                    "scenario_loss": scenario_loss,
                    "scenario_loss_percentage": (scenario_loss / portfolio_value) * 100,
                    "remaining_value": portfolio_value - scenario_loss,
                    "price_shocks": price_shocks
                }

            stress_test_result = {
                "portfolio_value": portfolio_value,
                "scenarios": results,
                "worst_case_loss": max(r["scenario_loss"] for r in results.values()),
                "stress_test_date": datetime.utcnow().isoformat()
            }

            return stress_test_result

        except Exception as e:
            logger.error(f"Error in stress testing: {e}")
            raise

    async def check_risk_limits(self, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """Vérifie si une position respecte les limites de risque."""
        try:
            violations = []
            warnings = []

            position_size = position_data.get("size_percentage", 0)
            total_exposure = position_data.get("total_exposure", 0)

            # Vérification des limites
            if position_size > self._risk_limits["max_position_size"]:
                violations.append({
                    "type": "POSITION_SIZE_EXCEEDED",
                    "limit": self._risk_limits["max_position_size"],
                    "current": position_size,
                    "severity": "HIGH"
                })

            if total_exposure > self._risk_limits["max_total_exposure"]:
                violations.append({
                    "type": "TOTAL_EXPOSURE_EXCEEDED",
                    "limit": self._risk_limits["max_total_exposure"],
                    "current": total_exposure,
                    "severity": "CRITICAL"
                })

            # Warnings pour approche des limites
            if position_size > self._risk_limits["max_position_size"] * 0.8:
                warnings.append({
                    "type": "POSITION_SIZE_WARNING",
                    "message": "Approaching position size limit",
                    "current": position_size,
                    "limit": self._risk_limits["max_position_size"]
                })

            result = {
                "position_approved": len(violations) == 0,
                "violations": violations,
                "warnings": warnings,
                "risk_score": self._calculate_position_risk_score(position_data),
                "checked_at": datetime.utcnow().isoformat()
            }

            return result

        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            raise

    async def get_risk_alerts(
        self,
        severity: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Récupère les alertes de risque."""
        try:
            # Simulation d'alertes
            mock_alerts = [
                {
                    "id": "alert_001",
                    "type": "VaR_EXCEEDED",
                    "severity": "HIGH",
                    "message": "Portfolio VaR exceeded daily limit",
                    "current_value": 5200,
                    "threshold": 5000,
                    "created_at": datetime.utcnow() - timedelta(hours=2),
                    "status": "ACTIVE"
                },
                {
                    "id": "alert_002",
                    "type": "CORRELATION_WARNING",
                    "severity": "MEDIUM",
                    "message": "High correlation detected between BTC and ETH positions",
                    "correlation": 0.85,
                    "threshold": 0.7,
                    "created_at": datetime.utcnow() - timedelta(hours=1),
                    "status": "ACTIVE"
                }
            ]

            alerts = mock_alerts
            if severity:
                alerts = [a for a in alerts if a["severity"] == severity]

            return alerts

        except Exception as e:
            logger.error(f"Error getting risk alerts: {e}")
            raise

    async def calculate_liquidity_risk(
        self,
        time_horizon: str = "1d"
    ) -> Dict[str, Any]:
        """Calcule le risque de liquidité."""
        try:
            # Simulation du risque de liquidité
            liquidity_metrics = {
                "time_horizon": time_horizon,
                "liquidation_cost": 0.02,  # 2% coût de liquidation
                "time_to_liquidate": "2h",  # Temps pour liquider 50% du portfolio
                "market_impact": 0.015,    # 1.5% impact marché
                "bid_ask_spread_avg": 0.001,  # 0.1% spread moyen
                "volume_concentration": {
                    "top_3_assets": 0.7,   # 70% de la liquidité dans top 3 assets
                    "diversification_score": 75
                },
                "liquidity_score": 82,     # Score sur 100
                "liquidity_grade": "A-",
                "calculated_at": datetime.utcnow().isoformat()
            }

            return liquidity_metrics

        except Exception as e:
            logger.error(f"Error calculating liquidity risk: {e}")
            raise

    async def _calculate_var_monte_carlo(
        self,
        portfolio_value: float,
        confidence_level: float,
        time_horizon: int
    ) -> float:
        """Calcule VaR par simulation Monte Carlo."""
        try:
            # Simulation simplifiée
            np.random.seed(42)
            num_simulations = 10000

            # Supposer une volatilité annuelle de 20%
            daily_volatility = 0.20 / math.sqrt(252)

            # Générer des returns aléatoires
            random_returns = np.random.normal(0, daily_volatility, num_simulations)

            # Ajuster pour l'horizon temporel
            horizon_returns = random_returns * math.sqrt(time_horizon)

            # Calculer les pertes potentielles
            potential_losses = portfolio_value * np.abs(horizon_returns)

            # VaR au niveau de confiance donné
            var_percentile = (1 - confidence_level) * 100
            var_value = np.percentile(potential_losses, 100 - var_percentile)

            return float(var_value)

        except Exception as e:
            logger.error(f"Error in Monte Carlo VaR calculation: {e}")
            return portfolio_value * 0.05  # Fallback 5%

    async def _calculate_var_historical(
        self,
        portfolio_value: float,
        confidence_level: float,
        time_horizon: int
    ) -> float:
        """Calcule VaR historique."""
        # Simulation - en réalité utiliserait des données historiques
        return portfolio_value * 0.03 * math.sqrt(time_horizon)

    async def _calculate_var_parametric(
        self,
        portfolio_value: float,
        confidence_level: float,
        time_horizon: int
    ) -> float:
        """Calcule VaR paramétrique."""
        # Méthode variance-covariance
        volatility = 0.15  # 15% volatilité annuelle
        daily_vol = volatility / math.sqrt(252)
        horizon_vol = daily_vol * math.sqrt(time_horizon)

        # Z-score pour le niveau de confiance
        z_scores = {0.95: 1.645, 0.99: 2.326, 0.999: 3.090}
        z_score = z_scores.get(confidence_level, 1.645)

        var_value = portfolio_value * z_score * horizon_vol
        return var_value

    def _calculate_position_risk_score(self, position_data: Dict[str, Any]) -> int:
        """Calcule un score de risque pour une position (0-100)."""
        try:
            score = 50  # Score de base

            # Ajustements basés sur les caractéristiques de la position
            size = position_data.get("size_percentage", 0)
            if size > 0.05:  # 5%
                score += min(size * 100, 30)  # Max +30 points

            volatility = position_data.get("volatility", 0.1)
            score += min(volatility * 100, 20)  # Max +20 points

            return min(score, 100)

        except Exception as e:
            logger.error(f"Error calculating position risk score: {e}")
            return 50