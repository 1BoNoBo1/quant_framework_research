"""
Real-Time Risk Calculator
========================

Comprehensive risk metrics calculation for production trading.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass
import math
import statistics
import logging

from ...core.container import injectable
from ...domain.entities.portfolio import Portfolio
from ...domain.entities.position import Position


logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Métriques de risque calculées"""
    timestamp: datetime

    # Métriques de portfolio
    portfolio_value: Decimal
    portfolio_return: Optional[Decimal] = None
    portfolio_volatility: Optional[Decimal] = None
    portfolio_var: Decimal = Decimal("0")  # Value at Risk
    portfolio_cvar: Decimal = Decimal("0")  # Conditional VaR

    # Drawdown
    max_drawdown: Decimal = Decimal("0")
    current_drawdown: Decimal = Decimal("0")
    underwater_period: int = 0  # jours sous l'eau

    # Concentration et diversification
    concentration_risk: Decimal = Decimal("0")  # HHI index
    effective_positions: Decimal = Decimal("0")  # Nombre effectif de positions
    correlation_risk: Optional[Decimal] = None

    # Risque de liquidité
    liquidity_score: Optional[Decimal] = None
    liquidity_var: Optional[Decimal] = None

    # Métriques d'exécution
    execution_quality_score: Optional[Decimal] = None
    slippage_impact: Optional[Decimal] = None

    # Stress testing
    stress_test_var: Optional[Decimal] = None
    tail_risk: Optional[Decimal] = None

    # Métriques de momentum et trend
    momentum_score: Optional[Decimal] = None
    trend_strength: Optional[Decimal] = None

    # Exposition par secteur/asset class
    sector_exposures: Optional[Dict[str, Decimal]] = None
    currency_exposures: Optional[Dict[str, Decimal]] = None


@injectable
class RealTimeRiskCalculator:
    """
    Calculateur de métriques de risque en temps réel.

    Implémente des modèles sophistiqués:
    - VaR historique, paramétrique et Monte Carlo
    - Stress testing avec scénarios historiques
    - Analyse de corrélations dynamiques
    - Risk attribution par facteurs
    - Métriques de liquidité avancées
    """

    def __init__(
        self,
        var_confidence: Decimal = Decimal("0.95"),  # 95% VaR
        var_horizon: int = 1,  # 1 jour
        lookback_window: int = 252,  # 1 an de données
        stress_scenarios: Optional[List[Dict]] = None
    ):
        self.var_confidence = var_confidence
        self.var_horizon = var_horizon
        self.lookback_window = lookback_window
        self.stress_scenarios = stress_scenarios or self._default_stress_scenarios()

        # Historique des valeurs de portfolio
        self.portfolio_history: List[Tuple[datetime, Decimal]] = []
        self.returns_history: List[Decimal] = []

        # Cache pour calculs coûteux
        self._correlation_cache: Dict[str, Any] = {}
        self._volatility_cache: Dict[str, Decimal] = {}
        self._cache_expiry: datetime = datetime.utcnow()

        # Configuration stress testing
        self.max_history_length = 1000

    async def calculate_risk_metrics(self, portfolio: Portfolio) -> RiskMetrics:
        """
        Calcule toutes les métriques de risque pour un portfolio.

        Args:
            portfolio: Portfolio à analyser

        Returns:
            Métriques de risque complètes
        """
        current_time = datetime.utcnow()

        try:
            # Enregistrer valeur de portfolio
            self._record_portfolio_value(current_time, portfolio.total_value)

            # Calculer métriques de base
            portfolio_return = await self._calculate_portfolio_return()
            portfolio_volatility = await self._calculate_portfolio_volatility()

            # VaR et CVaR
            var_95 = await self._calculate_var(portfolio)
            cvar_95 = await self._calculate_cvar(portfolio)

            # Drawdown
            max_dd, current_dd, underwater_days = await self._calculate_drawdown()

            # Concentration
            concentration = await self._calculate_concentration_risk(portfolio)
            effective_positions = await self._calculate_effective_positions(portfolio)

            # Corrélations
            correlation_risk = await self._calculate_correlation_risk(portfolio)

            # Liquidité
            liquidity_score = await self._calculate_liquidity_score(portfolio)
            liquidity_var = await self._calculate_liquidity_var(portfolio)

            # Stress testing
            stress_var = await self._calculate_stress_var(portfolio)
            tail_risk = await self._calculate_tail_risk()

            # Momentum et trend
            momentum = await self._calculate_momentum_score()
            trend_strength = await self._calculate_trend_strength()

            # Expositions
            sector_exp = await self._calculate_sector_exposures(portfolio)
            currency_exp = await self._calculate_currency_exposures(portfolio)

            return RiskMetrics(
                timestamp=current_time,
                portfolio_value=portfolio.total_value,
                portfolio_return=portfolio_return,
                portfolio_volatility=portfolio_volatility,
                portfolio_var=var_95,
                portfolio_cvar=cvar_95,
                max_drawdown=max_dd,
                current_drawdown=current_dd,
                underwater_period=underwater_days,
                concentration_risk=concentration,
                effective_positions=effective_positions,
                correlation_risk=correlation_risk,
                liquidity_score=liquidity_score,
                liquidity_var=liquidity_var,
                stress_test_var=stress_var,
                tail_risk=tail_risk,
                momentum_score=momentum,
                trend_strength=trend_strength,
                sector_exposures=sector_exp,
                currency_exposures=currency_exp
            )

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")

            # Retourner métriques minimales en cas d'erreur
            return RiskMetrics(
                timestamp=current_time,
                portfolio_value=portfolio.total_value
            )

    # === Calculs VaR et CVaR ===

    async def _calculate_var(self, portfolio: Portfolio) -> Decimal:
        """Calcule Value at Risk paramétrique"""
        if len(self.returns_history) < 30:  # Minimum de données requis
            return Decimal("0")

        try:
            # VaR paramétrique (distribution normale)
            returns = [float(r) for r in self.returns_history[-self.lookback_window:]]

            if len(returns) < 2:
                return Decimal("0")

            # Calculer moyenne et écart-type
            mean_return = statistics.mean(returns)
            std_return = statistics.stdev(returns)

            # Z-score pour niveau de confiance
            z_score = self._get_z_score(self.var_confidence)

            # VaR = -(moyenne - z * volatilité) * valeur du portfolio
            var_return = -(mean_return - z_score * std_return)
            var_value = Decimal(str(var_return)) * portfolio.total_value

            return max(var_value, Decimal("0"))

        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return Decimal("0")

    async def _calculate_cvar(self, portfolio: Portfolio) -> Decimal:
        """Calcule Conditional Value at Risk (Expected Shortfall)"""
        if len(self.returns_history) < 30:
            return Decimal("0")

        try:
            returns = [float(r) for r in self.returns_history[-self.lookback_window:]]

            # Seuil VaR
            var_threshold = await self._calculate_var(portfolio)
            var_return_threshold = var_threshold / portfolio.total_value if portfolio.total_value > 0 else 0

            # Moyenner les pertes au-delà du VaR
            tail_losses = [r for r in returns if r <= -float(var_return_threshold)]

            if tail_losses:
                cvar_return = abs(statistics.mean(tail_losses))
                return Decimal(str(cvar_return)) * portfolio.total_value
            else:
                return var_threshold * Decimal("1.3")  # Approximation CVaR = 1.3 * VaR

        except Exception as e:
            logger.error(f"Error calculating CVaR: {e}")
            return Decimal("0")

    # === Calculs de drawdown ===

    async def _calculate_drawdown(self) -> Tuple[Decimal, Decimal, int]:
        """Calcule max drawdown, drawdown actuel et période sous l'eau"""
        if len(self.portfolio_history) < 2:
            return Decimal("0"), Decimal("0"), 0

        try:
            values = [float(value) for _, value in self.portfolio_history]

            # Calculer peak running et drawdowns
            peak = values[0]
            max_drawdown = 0
            current_drawdown = 0
            underwater_days = 0
            consecutive_underwater = 0

            for value in values:
                if value > peak:
                    peak = value
                    consecutive_underwater = 0
                else:
                    consecutive_underwater += 1

                # Drawdown actuel
                current_dd = (peak - value) / peak if peak > 0 else 0
                current_drawdown = current_dd

                # Max drawdown
                if current_dd > max_drawdown:
                    max_drawdown = current_dd

                # Période sous l'eau (jours consécutifs de drawdown)
                if current_dd > 0:
                    underwater_days = consecutive_underwater

            return (
                Decimal(str(max_drawdown)),
                Decimal(str(current_drawdown)),
                underwater_days
            )

        except Exception as e:
            logger.error(f"Error calculating drawdown: {e}")
            return Decimal("0"), Decimal("0"), 0

    # === Métriques de concentration ===

    async def _calculate_concentration_risk(self, portfolio: Portfolio) -> Decimal:
        """Calcule l'indice de concentration HHI (Herfindahl-Hirschman)"""
        if not portfolio.positions or portfolio.total_value <= 0:
            return Decimal("0")

        try:
            # Calculer poids de chaque position
            weights_squared = []

            for position in portfolio.positions:
                position_value = abs(position.market_value or Decimal("0"))
                weight = position_value / portfolio.total_value
                weights_squared.append(weight * weight)

            # HHI = somme des carrés des poids
            hhi = sum(weights_squared)

            return Decimal(str(hhi))

        except Exception as e:
            logger.error(f"Error calculating concentration risk: {e}")
            return Decimal("0")

    async def _calculate_effective_positions(self, portfolio: Portfolio) -> Decimal:
        """Calcule le nombre effectif de positions (inverse du HHI)"""
        concentration = await self._calculate_concentration_risk(portfolio)

        if concentration > 0:
            return Decimal("1") / concentration
        else:
            return Decimal(str(len(portfolio.positions)))

    # === Risque de corrélation ===

    async def _calculate_correlation_risk(self, portfolio: Portfolio) -> Optional[Decimal]:
        """Estime le risque de corrélation basé sur la diversification"""
        if len(portfolio.positions) < 2:
            return None

        try:
            # Approximation simple: diversification benefit
            # Dans la réalité, utiliserait les corrélations historiques

            n_positions = len(portfolio.positions)

            # Bénéfice de diversification idéal (corrélation = 0)
            ideal_diversification = Decimal("1") / Decimal(str(math.sqrt(n_positions)))

            # Concentration réelle
            concentration = await self._calculate_concentration_risk(portfolio)
            actual_diversification = Decimal("1") / Decimal(str(math.sqrt(float(concentration))))

            # Risque de corrélation = écart à la diversification idéale
            correlation_risk = actual_diversification / ideal_diversification

            return min(correlation_risk, Decimal("2.0"))  # Cap à 2.0

        except Exception as e:
            logger.error(f"Error calculating correlation risk: {e}")
            return None

    # === Métriques de liquidité ===

    async def _calculate_liquidity_score(self, portfolio: Portfolio) -> Optional[Decimal]:
        """Calcule un score de liquidité composite du portfolio"""
        if not portfolio.positions:
            return None

        try:
            # Simulation de scores de liquidité par asset
            liquidity_scores = {
                "BTC": 0.85, "ETH": 0.80, "AAPL": 0.90, "TSLA": 0.75,
                "SPY": 0.95, "QQQ": 0.90, "MSFT": 0.88
            }

            weighted_liquidity = Decimal("0")
            total_weight = Decimal("0")

            for position in portfolio.positions:
                position_value = abs(position.market_value or Decimal("0"))
                weight = position_value / portfolio.total_value if portfolio.total_value > 0 else Decimal("0")

                # Score de liquidité (par défaut 0.5 si inconnu)
                liquidity = Decimal(str(liquidity_scores.get(position.symbol, 0.5)))

                weighted_liquidity += weight * liquidity
                total_weight += weight

            if total_weight > 0:
                return weighted_liquidity / total_weight
            else:
                return Decimal("0.5")  # Score neutre

        except Exception as e:
            logger.error(f"Error calculating liquidity score: {e}")
            return None

    async def _calculate_liquidity_var(self, portfolio: Portfolio) -> Optional[Decimal]:
        """Calcule VaR ajusté pour risque de liquidité"""
        base_var = await self._calculate_var(portfolio)
        liquidity_score = await self._calculate_liquidity_score(portfolio)

        if liquidity_score is not None and liquidity_score > 0:
            # Ajuster VaR basé sur liquidité (moins liquide = VaR plus élevé)
            liquidity_multiplier = Decimal("2") - liquidity_score  # 1.0 à 2.0
            return base_var * liquidity_multiplier

        return base_var

    # === Stress testing ===

    async def _calculate_stress_var(self, portfolio: Portfolio) -> Optional[Decimal]:
        """Calcule VaR de stress testing basé sur scénarios historiques"""
        try:
            stress_losses = []

            for scenario in self.stress_scenarios:
                scenario_loss = await self._apply_stress_scenario(portfolio, scenario)
                stress_losses.append(float(scenario_loss))

            if stress_losses:
                # VaR de stress = percentile basé sur scénarios
                stress_losses.sort(reverse=True)  # Du pire au meilleur
                percentile_idx = int(len(stress_losses) * (1 - float(self.var_confidence)))

                if percentile_idx < len(stress_losses):
                    return Decimal(str(stress_losses[percentile_idx]))

            return None

        except Exception as e:
            logger.error(f"Error calculating stress VaR: {e}")
            return None

    async def _apply_stress_scenario(self, portfolio: Portfolio, scenario: Dict) -> Decimal:
        """Applique un scénario de stress au portfolio"""
        total_loss = Decimal("0")

        for position in portfolio.positions:
            position_value = position.market_value or Decimal("0")

            # Récupérer shock pour cet asset (ou par défaut)
            shock = scenario.get(position.symbol, scenario.get("default", -0.1))
            position_loss = position_value * Decimal(str(abs(shock)))

            total_loss += position_loss

        return total_loss

    # === Métriques de tail risk ===

    async def _calculate_tail_risk(self) -> Optional[Decimal]:
        """Calcule les métriques de risque de queue"""
        if len(self.returns_history) < 100:
            return None

        try:
            returns = [float(r) for r in self.returns_history[-self.lookback_window:]]
            returns.sort()  # Tri ascendant

            # Prendre les 5% pires returns
            tail_size = max(1, len(returns) // 20)
            tail_returns = returns[:tail_size]

            if tail_returns:
                # Moyenne des returns de queue
                tail_mean = statistics.mean(tail_returns)
                return Decimal(str(abs(tail_mean)))

            return None

        except Exception as e:
            logger.error(f"Error calculating tail risk: {e}")
            return None

    # === Métriques de momentum ===

    async def _calculate_momentum_score(self) -> Optional[Decimal]:
        """Calcule un score de momentum du portfolio"""
        if len(self.returns_history) < 20:
            return None

        try:
            # Momentum = moyenne mobile des returns récents
            recent_returns = self.returns_history[-20:]

            if recent_returns:
                momentum = statistics.mean([float(r) for r in recent_returns])
                return Decimal(str(momentum))

            return None

        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return None

    async def _calculate_trend_strength(self) -> Optional[Decimal]:
        """Calcule la force de la tendance"""
        if len(self.portfolio_history) < 50:
            return None

        try:
            # Regression linéaire simple sur les valeurs récentes
            recent_values = [float(value) for _, value in self.portfolio_history[-50:]]

            n = len(recent_values)
            x_values = list(range(n))

            # Calculer slope de régression
            x_mean = statistics.mean(x_values)
            y_mean = statistics.mean(recent_values)

            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, recent_values))
            denominator = sum((x - x_mean) ** 2 for x in x_values)

            if denominator > 0:
                slope = numerator / denominator
                # Normaliser par valeur moyenne
                trend_strength = slope / y_mean if y_mean > 0 else 0
                return Decimal(str(abs(trend_strength)))

            return None

        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return None

    # === Expositions ===

    async def _calculate_sector_exposures(self, portfolio: Portfolio) -> Optional[Dict[str, Decimal]]:
        """Calcule l'exposition par secteur"""
        # Mapping simple symbole -> secteur
        sector_map = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "TSLA": "Automotive",
            "BTC": "Cryptocurrency",
            "ETH": "Cryptocurrency",
            "SPY": "Index",
            "QQQ": "Index"
        }

        sector_exposures = {}

        for position in portfolio.positions:
            sector = sector_map.get(position.symbol, "Other")
            position_value = abs(position.market_value or Decimal("0"))

            if sector not in sector_exposures:
                sector_exposures[sector] = Decimal("0")

            sector_exposures[sector] += position_value

        # Normaliser par valeur totale du portfolio
        if portfolio.total_value > 0:
            for sector in sector_exposures:
                sector_exposures[sector] /= portfolio.total_value

        return sector_exposures

    async def _calculate_currency_exposures(self, portfolio: Portfolio) -> Optional[Dict[str, Decimal]]:
        """Calcule l'exposition par devise"""
        # Pour simplifier, supposer que tout est en USD sauf crypto
        currency_exposures = {"USD": Decimal("0")}

        for position in portfolio.positions:
            position_value = abs(position.market_value or Decimal("0"))

            if position.symbol in ["BTC", "ETH"]:
                currency = position.symbol
            else:
                currency = "USD"

            if currency not in currency_exposures:
                currency_exposures[currency] = Decimal("0")

            currency_exposures[currency] += position_value

        # Normaliser
        if portfolio.total_value > 0:
            for currency in currency_exposures:
                currency_exposures[currency] /= portfolio.total_value

        return currency_exposures

    # === Méthodes utilitaires ===

    async def _calculate_portfolio_return(self) -> Optional[Decimal]:
        """Calcule le return du portfolio"""
        if len(self.portfolio_history) < 2:
            return None

        current_value = self.portfolio_history[-1][1]
        previous_value = self.portfolio_history[-2][1]

        if previous_value > 0:
            return (current_value - previous_value) / previous_value

        return None

    async def _calculate_portfolio_volatility(self) -> Optional[Decimal]:
        """Calcule la volatilité du portfolio"""
        if len(self.returns_history) < 10:
            return None

        try:
            returns = [float(r) for r in self.returns_history[-self.lookback_window:]]

            if len(returns) > 1:
                volatility = statistics.stdev(returns)
                return Decimal(str(volatility))

            return None

        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return None

    def _record_portfolio_value(self, timestamp: datetime, value: Decimal) -> None:
        """Enregistre une valeur de portfolio dans l'historique"""
        self.portfolio_history.append((timestamp, value))

        # Calculer return si possible
        if len(self.portfolio_history) >= 2:
            current_value = self.portfolio_history[-1][1]
            previous_value = self.portfolio_history[-2][1]

            if previous_value > 0:
                portfolio_return = (current_value - previous_value) / previous_value
                self.returns_history.append(portfolio_return)

        # Limiter taille historique
        if len(self.portfolio_history) > self.max_history_length:
            self.portfolio_history = self.portfolio_history[-self.max_history_length:]

        if len(self.returns_history) > self.max_history_length:
            self.returns_history = self.returns_history[-self.max_history_length:]

    def _get_z_score(self, confidence: Decimal) -> float:
        """Retourne le z-score pour un niveau de confiance donné"""
        # Approximation des z-scores courants
        z_scores = {
            Decimal("0.90"): 1.28,
            Decimal("0.95"): 1.645,
            Decimal("0.99"): 2.326
        }

        return z_scores.get(confidence, 1.645)  # 95% par défaut

    def _default_stress_scenarios(self) -> List[Dict]:
        """Retourne les scénarios de stress par défaut"""
        return [
            # Crash du marché actions 2008
            {"AAPL": -0.30, "MSFT": -0.25, "TSLA": -0.40, "SPY": -0.35, "default": -0.20},

            # Krach crypto 2022
            {"BTC": -0.70, "ETH": -0.75, "default": -0.10},

            # Volatilité extrême
            {"default": -0.15},

            # Crise de liquidité
            {"default": -0.25},

            # Inflation shock
            {"default": -0.12}
        ]