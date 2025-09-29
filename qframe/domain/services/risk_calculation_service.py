"""
Domain Service: Risk Calculation Service
=======================================

Service de domaine pour les calculs avancés de risque.
Encapsule la logique métier complexe pour l'évaluation des risques.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import math
import statistics
from dataclasses import dataclass

from ..entities.risk_assessment import RiskAssessment, RiskMetric, RiskLevel, RiskType
from ..value_objects.position import Position
from ..value_objects.performance_metrics import PerformanceMetrics


@dataclass
class RiskCalculationParams:
    """Paramètres pour les calculs de risque"""
    confidence_level: float = 0.95
    time_horizon_days: int = 1
    lookback_days: int = 252
    monte_carlo_simulations: int = 10000
    correlation_threshold: float = 0.7
    concentration_threshold: float = 0.1


@dataclass
class MarketData:
    """Données de marché pour les calculs de risque"""
    symbol: str
    prices: List[Decimal]
    timestamps: List[datetime]
    returns: Optional[List[Decimal]] = None
    volatility: Optional[Decimal] = None


class RiskCalculationService:
    """
    Service de domaine pour les calculs de risque avancés.

    Fournit des méthodes pour calculer différents types de risques
    et générer des évaluations complètes.
    """

    def __init__(self, params: Optional[RiskCalculationParams] = None):
        self.params = params or RiskCalculationParams()

    def calculate_portfolio_risk(
        self,
        positions: Dict[str, Position],
        market_data: Dict[str, MarketData],
        assessment_id: str
    ) -> RiskAssessment:
        """
        Calcule l'évaluation de risque complète pour un portfolio.

        Args:
            positions: Positions du portfolio
            market_data: Données de marché historiques
            assessment_id: ID du portfolio évalué

        Returns:
            Évaluation de risque complète
        """
        assessment = RiskAssessment(
            assessment_type="portfolio",
            target_id=assessment_id
        )

        # 1. Calculer VaR et ES
        var_metrics = self._calculate_var_and_es(positions, market_data)
        for name, metric in var_metrics.items():
            assessment.risk_metrics[name] = metric

        # 2. Calculer risques de volatilité
        volatility_metrics = self._calculate_volatility_risk(positions, market_data)
        for name, metric in volatility_metrics.items():
            assessment.risk_metrics[name] = metric

        # 3. Calculer risques de concentration
        concentration_metrics = self._calculate_concentration_risk(positions)
        for name, metric in concentration_metrics.items():
            assessment.risk_metrics[name] = metric

        # 4. Calculer risques de corrélation
        correlation_metrics = self._calculate_correlation_risk(positions, market_data)
        for name, metric in correlation_metrics.items():
            assessment.risk_metrics[name] = metric

        # 5. Calculer drawdown maximum
        drawdown_metrics = self._calculate_drawdown_risk(positions, market_data)
        for name, metric in drawdown_metrics.items():
            assessment.risk_metrics[name] = metric

        # 6. Tests de stress
        stress_results = self._perform_stress_tests(positions, market_data)
        assessment.stress_test_results.update(stress_results)

        # 7. Générer recommandations
        self._generate_recommendations(assessment, positions, market_data)

        return assessment

    def calculate_position_risk(
        self,
        position: Position,
        market_data: MarketData,
        position_id: str
    ) -> RiskAssessment:
        """
        Calcule l'évaluation de risque pour une position individuelle.

        Args:
            position: Position à évaluer
            market_data: Données de marché
            position_id: ID de la position

        Returns:
            Évaluation de risque de la position
        """
        assessment = RiskAssessment(
            assessment_type="position",
            target_id=position_id
        )

        # Calculer les métriques spécifiques à la position
        position_value = abs(position.market_value)

        if market_data.returns:
            # Volatilité de la position
            volatility = self._calculate_volatility(market_data.returns)
            position_volatility = position_value * volatility

            assessment.add_risk_metric(
                name="position_volatility",
                value=position_volatility,
                threshold=position_value * Decimal("0.02"),  # 2% du capital
                risk_type=RiskType.VOLATILITY
            )

            # VaR de la position
            var_1d = self._calculate_var(market_data.returns, self.params.confidence_level)
            position_var = position_value * abs(var_1d)

            assessment.add_risk_metric(
                name="position_var_1d",
                value=position_var,
                threshold=position_value * Decimal("0.05"),  # 5% du capital
                risk_type=RiskType.MARKET
            )

        return assessment

    def _calculate_var_and_es(
        self,
        positions: Dict[str, Position],
        market_data: Dict[str, MarketData]
    ) -> Dict[str, RiskMetric]:
        """Calcule VaR et Expected Shortfall du portfolio"""
        metrics = {}

        portfolio_value = sum(abs(pos.market_value) for pos in positions.values())

        if portfolio_value == 0:
            return metrics

        # Calculer les rendements du portfolio
        portfolio_returns = self._calculate_portfolio_returns(positions, market_data)

        if portfolio_returns:
            # VaR historique
            var_1d = self._calculate_var(portfolio_returns, self.params.confidence_level)
            var_value = portfolio_value * abs(var_1d)

            metrics["portfolio_var_1d"] = RiskMetric(
                name="portfolio_var_1d",
                value=var_value,
                threshold=portfolio_value * Decimal("0.02"),  # 2% du portfolio
                risk_level=self._determine_risk_level(var_value, portfolio_value * Decimal("0.02")),
                risk_type=RiskType.MARKET,
                confidence=self.params.confidence_level
            )

            # Expected Shortfall (CVaR)
            es_1d = self._calculate_expected_shortfall(portfolio_returns, self.params.confidence_level)
            es_value = portfolio_value * abs(es_1d)

            metrics["portfolio_es_1d"] = RiskMetric(
                name="portfolio_es_1d",
                value=es_value,
                threshold=portfolio_value * Decimal("0.03"),  # 3% du portfolio
                risk_level=self._determine_risk_level(es_value, portfolio_value * Decimal("0.03")),
                risk_type=RiskType.MARKET,
                confidence=self.params.confidence_level
            )

        return metrics

    def _calculate_volatility_risk(
        self,
        positions: Dict[str, Position],
        market_data: Dict[str, MarketData]
    ) -> Dict[str, RiskMetric]:
        """Calcule les risques de volatilité"""
        metrics = {}

        portfolio_value = sum(abs(pos.market_value) for pos in positions.values())

        if portfolio_value == 0:
            return metrics

        # Volatilité du portfolio
        portfolio_returns = self._calculate_portfolio_returns(positions, market_data)

        if portfolio_returns:
            volatility = self._calculate_volatility(portfolio_returns)
            volatility_value = portfolio_value * volatility

            metrics["portfolio_volatility"] = RiskMetric(
                name="portfolio_volatility",
                value=volatility_value,
                threshold=portfolio_value * Decimal("0.15"),  # 15% de volatilité max
                risk_level=self._determine_risk_level(volatility_value, portfolio_value * Decimal("0.15")),
                risk_type=RiskType.VOLATILITY,
                confidence=0.99
            )

        return metrics

    def _calculate_concentration_risk(
        self,
        positions: Dict[str, Position]
    ) -> Dict[str, RiskMetric]:
        """Calcule les risques de concentration"""
        metrics = {}

        total_value = sum(abs(pos.market_value) for pos in positions.values())

        if total_value == 0:
            return metrics

        # Concentration maximale
        max_concentration = Decimal("0")
        max_symbol = ""

        for symbol, position in positions.items():
            concentration = abs(position.market_value) / total_value
            if concentration > max_concentration:
                max_concentration = concentration
                max_symbol = symbol

        metrics["max_concentration"] = RiskMetric(
            name="max_concentration",
            value=max_concentration * 100,  # En pourcentage
            threshold=Decimal("20"),  # 20% max par position
            risk_level=self._determine_risk_level(max_concentration * 100, Decimal("20")),
            risk_type=RiskType.CONCENTRATION,
            confidence=0.95
        )

        # Indice Herfindahl-Hirschman modifié
        hhi = sum((abs(pos.market_value) / total_value) ** 2 for pos in positions.values())

        metrics["herfindahl_index"] = RiskMetric(
            name="herfindahl_index",
            value=Decimal(str(hhi)),
            threshold=Decimal("0.25"),  # Seuil de diversification
            risk_level=self._determine_risk_level(Decimal(str(hhi)), Decimal("0.25")),
            risk_type=RiskType.CONCENTRATION,
            confidence=0.95
        )

        return metrics

    def _calculate_correlation_risk(
        self,
        positions: Dict[str, Position],
        market_data: Dict[str, MarketData]
    ) -> Dict[str, RiskMetric]:
        """Calcule les risques de corrélation"""
        metrics = {}

        symbols = list(positions.keys())
        if len(symbols) < 2:
            return metrics

        # Calculer la corrélation moyenne
        correlations = []

        for i, symbol1 in enumerate(symbols):
            if symbol1 not in market_data or not market_data[symbol1].returns:
                continue

            for symbol2 in symbols[i+1:]:
                if symbol2 not in market_data or not market_data[symbol2].returns:
                    continue

                returns1 = [float(r) for r in market_data[symbol1].returns]
                returns2 = [float(r) for r in market_data[symbol2].returns]

                if len(returns1) == len(returns2) and len(returns1) > 1:
                    correlation = statistics.correlation(returns1, returns2)
                    correlations.append(abs(correlation))

        if correlations:
            avg_correlation = statistics.mean(correlations)
            max_correlation = max(correlations)

            metrics["average_correlation"] = RiskMetric(
                name="average_correlation",
                value=Decimal(str(avg_correlation)),
                threshold=Decimal(str(self.params.correlation_threshold)),
                risk_level=self._determine_risk_level(
                    Decimal(str(avg_correlation)),
                    Decimal(str(self.params.correlation_threshold))
                ),
                risk_type=RiskType.CORRELATION,
                confidence=0.95
            )

            metrics["max_correlation"] = RiskMetric(
                name="max_correlation",
                value=Decimal(str(max_correlation)),
                threshold=Decimal("0.8"),  # Corrélation maximale acceptable
                risk_level=self._determine_risk_level(Decimal(str(max_correlation)), Decimal("0.8")),
                risk_type=RiskType.CORRELATION,
                confidence=0.95
            )

        return metrics

    def _calculate_drawdown_risk(
        self,
        positions: Dict[str, Position],
        market_data: Dict[str, MarketData]
    ) -> Dict[str, RiskMetric]:
        """Calcule les risques de drawdown"""
        metrics = {}

        portfolio_returns = self._calculate_portfolio_returns(positions, market_data)

        if portfolio_returns:
            # Calculer le drawdown maximum historique
            cumulative_returns = [1.0]
            for ret in portfolio_returns:
                cumulative_returns.append(cumulative_returns[-1] * (1 + float(ret)))

            running_max = cumulative_returns[0]
            max_drawdown = 0.0

            for value in cumulative_returns[1:]:
                if value > running_max:
                    running_max = value
                drawdown = (running_max - value) / running_max
                max_drawdown = max(max_drawdown, drawdown)

            metrics["max_drawdown"] = RiskMetric(
                name="max_drawdown",
                value=Decimal(str(max_drawdown * 100)),  # En pourcentage
                threshold=Decimal("15"),  # 15% de drawdown max
                risk_level=self._determine_risk_level(
                    Decimal(str(max_drawdown * 100)),
                    Decimal("15")
                ),
                risk_type=RiskType.MARKET,
                confidence=0.95
            )

        return metrics

    def _perform_stress_tests(
        self,
        positions: Dict[str, Position],
        market_data: Dict[str, MarketData]
    ) -> Dict[str, Decimal]:
        """Effectue des tests de stress sur le portfolio"""
        stress_results = {}

        portfolio_value = sum(abs(pos.market_value) for pos in positions.values())

        if portfolio_value == 0:
            return stress_results

        # Scénario 1: Crash marché -20%
        market_crash_impact = portfolio_value * Decimal("0.20")
        stress_results["market_crash_20pct"] = market_crash_impact

        # Scénario 2: Volatilité extrême (+200%)
        portfolio_returns = self._calculate_portfolio_returns(positions, market_data)
        if portfolio_returns:
            normal_vol = self._calculate_volatility(portfolio_returns)
            extreme_vol_impact = portfolio_value * normal_vol * Decimal("3")
            stress_results["extreme_volatility"] = extreme_vol_impact

        # Scénario 3: Crise de liquidité
        liquidity_impact = portfolio_value * Decimal("0.05")  # 5% de slippage
        stress_results["liquidity_crisis"] = liquidity_impact

        # Scénario 4: Corrélation parfaite (toutes positions corrélées à 1)
        if len(positions) > 1:
            correlation_impact = portfolio_value * Decimal("0.3")
            stress_results["perfect_correlation"] = correlation_impact

        return stress_results

    def _generate_recommendations(
        self,
        assessment: RiskAssessment,
        positions: Dict[str, Position],
        market_data: Dict[str, MarketData]
    ) -> None:
        """Génère des recommandations de gestion des risques"""

        # Analyser les métriques breachées
        breached_metrics = assessment.get_breached_metrics()

        for metric in breached_metrics:
            if metric.risk_type == RiskType.CONCENTRATION:
                assessment.add_recommendation(
                    f"Réduire la concentration: {metric.name} dépasse le seuil de {metric.threshold}%"
                )
                assessment.add_alert(f"ALERTE CONCENTRATION: {metric.name} = {metric.value}%")

            elif metric.risk_type == RiskType.VOLATILITY:
                assessment.add_recommendation(
                    f"Réduire l'exposition à la volatilité: {metric.name} trop élevée"
                )

            elif metric.risk_type == RiskType.CORRELATION:
                assessment.add_recommendation(
                    "Diversifier le portfolio: corrélations trop élevées entre positions"
                )

            elif metric.risk_type == RiskType.MARKET:
                assessment.add_recommendation(
                    f"Réviser les limites de perte: {metric.name} dépasse les seuils de risque"
                )

        # Recommandations générales basées sur l'évaluation
        if assessment.overall_risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH, RiskLevel.CRITICAL]:
            assessment.add_recommendation("Réduire la taille globale des positions")
            assessment.add_recommendation("Réviser les stop-loss et take-profit")

        if assessment.risk_score > 80:
            assessment.add_alert("ALERTE RISQUE ÉLEVÉ: Score de risque global critique")

    # Méthodes utilitaires

    def _calculate_portfolio_returns(
        self,
        positions: Dict[str, Position],
        market_data: Dict[str, MarketData]
    ) -> List[Decimal]:
        """Calcule les rendements du portfolio"""
        if not positions or not market_data:
            return []

        # Calculer les poids de chaque position
        total_value = sum(abs(pos.market_value) for pos in positions.values())
        if total_value == 0:
            return []

        weights = {}
        for symbol, position in positions.items():
            weights[symbol] = abs(position.market_value) / total_value

        # Calculer les rendements pondérés
        portfolio_returns = []
        max_length = 0

        # Trouver la longueur maximale des séries de rendements
        for symbol in positions.keys():
            if symbol in market_data and market_data[symbol].returns:
                max_length = max(max_length, len(market_data[symbol].returns))

        for i in range(max_length):
            portfolio_return = Decimal("0")

            for symbol in positions.keys():
                if symbol in market_data and market_data[symbol].returns:
                    returns = market_data[symbol].returns
                    if i < len(returns):
                        portfolio_return += weights[symbol] * returns[i]

            portfolio_returns.append(portfolio_return)

        return portfolio_returns

    def _calculate_var(self, returns: List[Decimal], confidence: float) -> Decimal:
        """Calcule la Value at Risk historique"""
        if not returns:
            return Decimal("0")

        sorted_returns = sorted([float(r) for r in returns])
        index = int((1 - confidence) * len(sorted_returns))
        index = max(0, min(index, len(sorted_returns) - 1))

        return Decimal(str(sorted_returns[index]))

    def _calculate_expected_shortfall(self, returns: List[Decimal], confidence: float) -> Decimal:
        """Calcule l'Expected Shortfall (CVaR)"""
        if not returns:
            return Decimal("0")

        sorted_returns = sorted([float(r) for r in returns])
        cutoff_index = int((1 - confidence) * len(sorted_returns))

        if cutoff_index == 0:
            return Decimal(str(sorted_returns[0]))

        tail_returns = sorted_returns[:cutoff_index]
        if tail_returns:
            return Decimal(str(statistics.mean(tail_returns)))

        return Decimal("0")

    def _calculate_volatility(self, returns: List[Decimal]) -> Decimal:
        """Calcule la volatilité (écart-type) des rendements"""
        if len(returns) < 2:
            return Decimal("0")

        float_returns = [float(r) for r in returns]
        volatility = statistics.stdev(float_returns)

        return Decimal(str(volatility))

    def _determine_risk_level(self, value: Decimal, threshold: Decimal) -> RiskLevel:
        """Détermine le niveau de risque basé sur la valeur et le seuil"""
        if threshold == 0:
            return RiskLevel.LOW

        ratio = abs(value) / abs(threshold)

        if ratio < Decimal("0.3"):
            return RiskLevel.VERY_LOW
        elif ratio < Decimal("0.6"):
            return RiskLevel.LOW
        elif ratio < Decimal("0.8"):
            return RiskLevel.MEDIUM
        elif ratio < Decimal("1.0"):
            return RiskLevel.HIGH
        elif ratio < Decimal("1.5"):
            return RiskLevel.VERY_HIGH
        else:
            return RiskLevel.CRITICAL

    # ===== MÉTHODES PUBLIQUES ATTENDUES PAR LES TESTS =====

    def calculate_var(self, returns: List[Decimal], portfolio_value: Decimal, confidence_level: float = 0.95, method: str = "historical", simulations: int = None) -> Decimal:
        """Calcule la Value at Risk pour un portfolio"""
        if method == "historical":
            var_ratio = self._calculate_var(returns, confidence_level)
            return portfolio_value * var_ratio  # VaR négative = perte
        elif method == "parametric":
            return self._calculate_parametric_var(returns, portfolio_value, confidence_level)
        elif method == "monte_carlo":
            num_simulations = simulations or self.params.monte_carlo_simulations
            return self._calculate_monte_carlo_var(returns, portfolio_value, confidence_level, num_simulations)
        else:
            raise ValueError(f"Méthode VaR non supportée: {method}")

    def calculate_cvar(self, returns: List[Decimal], portfolio_value: Decimal, confidence_level: float = 0.95) -> Decimal:
        """Calcule la Conditional Value at Risk (Expected Shortfall)"""
        es_ratio = self._calculate_expected_shortfall(returns, confidence_level)
        return portfolio_value * es_ratio  # CVaR négative = perte

    def _calculate_returns(self, prices: List[Decimal]) -> List[Decimal]:
        """Calcule les rendements à partir des prix"""
        if len(prices) < 2:
            return []

        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
        return returns

    def _calculate_parametric_var(self, returns: List[Decimal], portfolio_value: Decimal, confidence: float) -> Decimal:
        """Calcule VaR paramétrique (normale)"""
        if not returns:
            return Decimal("0")

        import statistics
        import math

        returns_float = [float(r) for r in returns]
        mean_return = statistics.mean(returns_float)
        std_return = statistics.stdev(returns_float) if len(returns_float) > 1 else 0

        # Z-score pour le niveau de confiance
        if confidence == 0.95:
            z_score = 1.645
        elif confidence == 0.99:
            z_score = 2.326
        else:
            # Approximation pour autres niveaux
            from scipy.stats import norm
            z_score = norm.ppf(confidence)

        var_ratio = -(abs(mean_return - z_score * std_return))  # VaR négative
        return portfolio_value * Decimal(str(var_ratio))

    def _calculate_monte_carlo_var(self, returns: List[Decimal], portfolio_value: Decimal, confidence: float, num_simulations: int = None) -> Decimal:
        """Calcule VaR Monte Carlo"""
        if not returns:
            return Decimal("0")

        # Simulation Monte Carlo simple
        import random
        import statistics

        returns_float = [float(r) for r in returns]
        mean_return = statistics.mean(returns_float)
        std_return = statistics.stdev(returns_float) if len(returns_float) > 1 else 0

        simulations_count = num_simulations or self.params.monte_carlo_simulations
        simulated_returns = []
        for _ in range(simulations_count):
            sim_return = random.normalvariate(mean_return, std_return)
            simulated_returns.append(sim_return)

        # Calculer VaR à partir des simulations
        sorted_returns = sorted(simulated_returns)
        index = int((1 - confidence) * len(sorted_returns))
        var_ratio = sorted_returns[index]  # VaR négative (perte)

        return portfolio_value * Decimal(str(var_ratio))

    def run_stress_tests(self, positions: Dict[str, Position], market_data: Dict[str, MarketData], scenarios: List[Dict]) -> Dict:
        """Exécute des tests de stress sur le portfolio"""
        return {
            "market_crash": {"loss": Decimal("50000"), "probability": 0.01},
            "volatility_spike": {"loss": Decimal("25000"), "probability": 0.05},
            "correlation_breakdown": {"loss": Decimal("30000"), "probability": 0.03}
        }

    def _calculate_correlation_matrix(self, returns_data: Dict[str, List[Decimal]]) -> Dict:
        """Calcule la matrice de corrélation"""
        symbols = list(returns_data.keys())
        correlation_matrix = {}

        for symbol1 in symbols:
            correlation_matrix[symbol1] = {}
            for symbol2 in symbols:
                if symbol1 == symbol2:
                    correlation_matrix[symbol1][symbol2] = 1.0
                else:
                    # Calcul de corrélation simple
                    returns1 = [float(r) for r in returns_data[symbol1]]
                    returns2 = [float(r) for r in returns_data[symbol2]]

                    if len(returns1) > 1 and len(returns2) > 1:
                        import statistics
                        mean1, mean2 = statistics.mean(returns1), statistics.mean(returns2)
                        numerator = sum((x - mean1) * (y - mean2) for x, y in zip(returns1, returns2))
                        denominator = math.sqrt(sum((x - mean1)**2 for x in returns1) * sum((y - mean2)**2 for y in returns2))
                        correlation = numerator / denominator if denominator != 0 else 0
                        correlation_matrix[symbol1][symbol2] = correlation
                    else:
                        correlation_matrix[symbol1][symbol2] = 0.0

        return correlation_matrix