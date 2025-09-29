"""
Domain Service: Portfolio Service
================================

Service de domaine pour la logique métier complexe des portfolios.
Gère l'optimisation d'allocation, le rééquilibrage, et les calculs de performance.
"""

from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass
import math
import statistics

from ..entities.portfolio import Portfolio, PortfolioSnapshot, RebalancingFrequency
from ..entities.position import Position
from ..value_objects.performance_metrics import PerformanceMetrics


@dataclass
class RebalancingPlan:
    """Plan de rééquilibrage d'un portfolio"""
    portfolio_id: str
    timestamp: datetime
    target_allocations: Dict[str, Decimal]
    current_allocations: Dict[str, Decimal]
    trades_required: Dict[str, Decimal]  # symbol -> montant (+ acheter, - vendre)
    estimated_cost: Decimal
    reason: str

    def get_trade_value(self) -> Decimal:
        """Retourne la valeur totale des trades requis"""
        return sum(abs(amount) for amount in self.trades_required.values())

    def get_symbols_to_buy(self) -> List[str]:
        """Retourne les symboles à acheter"""
        return [symbol for symbol, amount in self.trades_required.items() if amount > 0]

    def get_symbols_to_sell(self) -> List[str]:
        """Retourne les symboles à vendre"""
        return [symbol for symbol, amount in self.trades_required.items() if amount < 0]


@dataclass
class AllocationOptimization:
    """Résultat d'optimisation d'allocation"""
    original_allocations: Dict[str, Decimal]
    optimized_allocations: Dict[str, Decimal]
    expected_return: Decimal
    expected_risk: Decimal
    sharpe_ratio: Decimal
    optimization_method: str
    constraints_applied: List[str]

    def get_allocation_changes(self) -> Dict[str, Decimal]:
        """Retourne les changements d'allocation"""
        changes = {}
        all_symbols = set(self.original_allocations.keys()) | set(self.optimized_allocations.keys())

        for symbol in all_symbols:
            original = self.original_allocations.get(symbol, Decimal("0"))
            optimized = self.optimized_allocations.get(symbol, Decimal("0"))
            changes[symbol] = optimized - original

        return changes


@dataclass
class PortfolioPerformanceAnalysis:
    """Analyse de performance complète d'un portfolio"""
    portfolio_id: str
    analysis_period_days: int
    total_return: Decimal
    annualized_return: Decimal
    volatility: Decimal
    sharpe_ratio: Decimal
    max_drawdown: Decimal
    win_rate: Optional[Decimal]
    best_day: Optional[Decimal]
    worst_day: Optional[Decimal]
    var_95: Optional[Decimal]  # Value at Risk 95%
    benchmark_return: Optional[Decimal]
    alpha: Optional[Decimal]
    beta: Optional[Decimal]
    information_ratio: Optional[Decimal]

    def to_dict(self) -> Dict[str, any]:
        return {
            "portfolio_id": self.portfolio_id,
            "analysis_period_days": self.analysis_period_days,
            "total_return": float(self.total_return),
            "annualized_return": float(self.annualized_return),
            "volatility": float(self.volatility),
            "sharpe_ratio": float(self.sharpe_ratio),
            "max_drawdown": float(self.max_drawdown),
            "win_rate": float(self.win_rate) if self.win_rate else None,
            "best_day": float(self.best_day) if self.best_day else None,
            "worst_day": float(self.worst_day) if self.worst_day else None,
            "var_95": float(self.var_95) if self.var_95 else None,
            "benchmark_return": float(self.benchmark_return) if self.benchmark_return else None,
            "alpha": float(self.alpha) if self.alpha else None,
            "beta": float(self.beta) if self.beta else None,
            "information_ratio": float(self.information_ratio) if self.information_ratio else None
        }


class PortfolioService:
    """
    Service de domaine pour la gestion avancée des portfolios.

    Fournit des méthodes pour l'optimisation d'allocation, le rééquilibrage,
    l'analyse de performance et la gestion des risques.
    """

    def __init__(self, risk_free_rate: Decimal = Decimal("0.02")):
        self.risk_free_rate = risk_free_rate

    # === Rééquilibrage ===

    def create_rebalancing_plan(
        self,
        portfolio: Portfolio,
        target_allocations: Optional[Dict[str, Decimal]] = None,
        rebalancing_threshold: Decimal = Decimal("0.05"),
        transaction_cost_rate: Decimal = Decimal("0.001")
    ) -> Optional[RebalancingPlan]:
        """
        Crée un plan de rééquilibrage pour un portfolio.

        Args:
            portfolio: Portfolio à rééquilibrer
            target_allocations: Allocations cibles (utilise celles du portfolio si None)
            rebalancing_threshold: Seuil de déclenchement du rééquilibrage
            transaction_cost_rate: Taux de coût de transaction

        Returns:
            Plan de rééquilibrage ou None si pas nécessaire
        """
        # Utiliser les allocations cibles du portfolio si non spécifiées
        targets = target_allocations or portfolio.target_allocations

        if not targets:
            return None

        # Calculer les allocations actuelles
        current_allocations = self._calculate_current_allocations(portfolio)

        # Vérifier si le rééquilibrage est nécessaire
        needs_rebalancing = False
        for symbol, target in targets.items():
            current = current_allocations.get(symbol, Decimal("0"))
            if abs(current - target) > rebalancing_threshold:
                needs_rebalancing = True
                break

        if not needs_rebalancing:
            return None

        # Calculer les trades requis
        trades_required = {}
        for symbol, target_weight in targets.items():
            current_weight = current_allocations.get(symbol, Decimal("0"))
            weight_diff = target_weight - current_weight
            trade_amount = weight_diff * portfolio.total_value

            # Seuil minimum pour éviter les micro-trades
            if abs(trade_amount) > Decimal("10"):  # 10 USD minimum
                trades_required[symbol] = trade_amount

        # Calculer le coût estimé
        trade_value = sum(abs(amount) for amount in trades_required.values())
        estimated_cost = trade_value * transaction_cost_rate

        # Déterminer la raison du rééquilibrage
        reason = self._determine_rebalancing_reason(portfolio, current_allocations, targets)

        return RebalancingPlan(
            portfolio_id=portfolio.id,
            timestamp=datetime.utcnow(),
            target_allocations=targets.copy(),
            current_allocations=current_allocations.copy(),
            trades_required=trades_required,
            estimated_cost=estimated_cost,
            reason=reason
        )

    def should_rebalance_by_frequency(
        self,
        portfolio: Portfolio,
        frequency: Optional[RebalancingFrequency] = None
    ) -> bool:
        """
        Vérifie si le portfolio doit être rééquilibré selon sa fréquence.

        Args:
            portfolio: Portfolio à vérifier
            frequency: Fréquence de rééquilibrage (utilise celle du portfolio si None)

        Returns:
            True si le rééquilibrage est dû
        """
        freq = frequency or portfolio.constraints.rebalancing_frequency

        if freq == RebalancingFrequency.MANUAL:
            return False

        if not portfolio.last_rebalanced_at:
            return True

        now = datetime.utcnow()
        time_since_last = now - portfolio.last_rebalanced_at

        if freq == RebalancingFrequency.DAILY:
            return time_since_last >= timedelta(days=1)
        elif freq == RebalancingFrequency.WEEKLY:
            return time_since_last >= timedelta(weeks=1)
        elif freq == RebalancingFrequency.MONTHLY:
            return time_since_last >= timedelta(days=30)
        elif freq == RebalancingFrequency.QUARTERLY:
            return time_since_last >= timedelta(days=90)

        return False

    def execute_rebalancing_plan(
        self,
        portfolio: Portfolio,
        plan: RebalancingPlan
    ) -> List[str]:
        """
        Simule l'exécution d'un plan de rééquilibrage.

        Args:
            portfolio: Portfolio à rééquilibrer
            plan: Plan de rééquilibrage

        Returns:
            Liste des messages d'exécution

        Note:
            Cette méthode simule l'exécution. Dans un vrai système,
            elle déléguerait à un service d'exécution.
        """
        execution_log = []

        # Vérifier que le plan est pour ce portfolio
        if plan.portfolio_id != portfolio.id:
            raise ValueError("Rebalancing plan portfolio ID mismatch")

        # Exécuter les trades (simulation)
        for symbol, trade_amount in plan.trades_required.items():
            if trade_amount > 0:
                # Achat
                execution_log.append(f"BUY {symbol}: ${float(trade_amount):.2f}")
            else:
                # Vente
                execution_log.append(f"SELL {symbol}: ${float(abs(trade_amount)):.2f}")

        # Mettre à jour la date de dernier rééquilibrage
        portfolio.last_rebalanced_at = datetime.utcnow()
        portfolio.updated_at = datetime.utcnow()

        execution_log.append(f"Rebalancing completed. Cost: ${float(plan.estimated_cost):.2f}")

        return execution_log

    # === Optimisation d'allocation ===

    def optimize_allocation_equal_weight(
        self,
        symbols: List[str],
        constraints: Optional[Dict[str, Tuple[Decimal, Decimal]]] = None
    ) -> AllocationOptimization:
        """
        Optimisation d'allocation équi-pondérée.

        Args:
            symbols: Liste des symboles
            constraints: Contraintes min/max par symbole

        Returns:
            Allocation optimisée
        """
        original_allocations = {symbol: Decimal("0") for symbol in symbols}

        # Allocation équi-pondérée de base
        base_weight = Decimal("1") / len(symbols)
        optimized_allocations = {symbol: base_weight for symbol in symbols}

        # Appliquer les contraintes si spécifiées
        constraints_applied = []
        if constraints:
            for symbol, (min_weight, max_weight) in constraints.items():
                if symbol in optimized_allocations:
                    current_weight = optimized_allocations[symbol]
                    if current_weight < min_weight:
                        optimized_allocations[symbol] = min_weight
                        constraints_applied.append(f"{symbol} min weight constraint")
                    elif current_weight > max_weight:
                        optimized_allocations[symbol] = max_weight
                        constraints_applied.append(f"{symbol} max weight constraint")

            # Renormaliser pour que la somme fasse 1
            total_weight = sum(optimized_allocations.values())
            if total_weight != Decimal("1"):
                for symbol in optimized_allocations:
                    optimized_allocations[symbol] /= total_weight

        return AllocationOptimization(
            original_allocations=original_allocations,
            optimized_allocations=optimized_allocations,
            expected_return=Decimal("0.08"),  # Placeholder
            expected_risk=Decimal("0.15"),    # Placeholder
            sharpe_ratio=Decimal("0.53"),     # Placeholder
            optimization_method="equal_weight",
            constraints_applied=constraints_applied
        )

    def optimize_allocation_risk_parity(
        self,
        symbols: List[str],
        risk_estimates: Dict[str, Decimal]
    ) -> AllocationOptimization:
        """
        Optimisation d'allocation par parité de risque.

        Args:
            symbols: Liste des symboles
            risk_estimates: Estimations de risque par symbole

        Returns:
            Allocation optimisée
        """
        original_allocations = {symbol: Decimal("0") for symbol in symbols}

        # Calculer les poids inversement proportionnels au risque
        inverse_risks = {symbol: Decimal("1") / risk_estimates.get(symbol, Decimal("0.1"))
                        for symbol in symbols}

        total_inverse_risk = sum(inverse_risks.values())
        optimized_allocations = {symbol: inverse_risk / total_inverse_risk
                               for symbol, inverse_risk in inverse_risks.items()}

        # Estimer les métriques de performance
        avg_risk = sum(risk_estimates.get(symbol, Decimal("0.1")) for symbol in symbols) / len(symbols)
        expected_return = Decimal("0.07")  # Placeholder
        expected_risk = avg_risk * Decimal("0.8")  # Risque réduit par diversification

        return AllocationOptimization(
            original_allocations=original_allocations,
            optimized_allocations=optimized_allocations,
            expected_return=expected_return,
            expected_risk=expected_risk,
            sharpe_ratio=(expected_return - self.risk_free_rate) / expected_risk,
            optimization_method="risk_parity",
            constraints_applied=[]
        )

    def optimize_allocation_momentum(
        self,
        symbols: List[str],
        momentum_scores: Dict[str, Decimal],
        lookback_period: int = 12
    ) -> AllocationOptimization:
        """
        Optimisation d'allocation basée sur le momentum.

        Args:
            symbols: Liste des symboles
            momentum_scores: Scores de momentum par symbole
            lookback_period: Période de lookback en mois

        Returns:
            Allocation optimisée
        """
        original_allocations = {symbol: Decimal("0") for symbol in symbols}

        # Trier par score de momentum (descendant)
        sorted_symbols = sorted(symbols, key=lambda s: momentum_scores.get(s, Decimal("0")), reverse=True)

        # Allocation pondérée par momentum
        total_momentum = sum(max(momentum_scores.get(symbol, Decimal("0")), Decimal("0")) for symbol in symbols)

        optimized_allocations = {}
        if total_momentum > 0:
            for symbol in symbols:
                momentum = max(momentum_scores.get(symbol, Decimal("0")), Decimal("0"))
                optimized_allocations[symbol] = momentum / total_momentum
        else:
            # Fallback vers allocation équi-pondérée si pas de momentum positif
            equal_weight = Decimal("1") / len(symbols)
            optimized_allocations = {symbol: equal_weight for symbol in symbols}

        return AllocationOptimization(
            original_allocations=original_allocations,
            optimized_allocations=optimized_allocations,
            expected_return=Decimal("0.10"),  # Placeholder - momentum tend à donner plus de rendement
            expected_risk=Decimal("0.18"),    # Placeholder - mais avec plus de risque
            sharpe_ratio=Decimal("0.44"),     # Placeholder
            optimization_method="momentum",
            constraints_applied=[]
        )

    # === Analyse de performance ===

    def analyze_portfolio_performance(
        self,
        portfolio: Portfolio,
        analysis_period_days: int = 252,
        benchmark_returns: Optional[List[Decimal]] = None
    ) -> PortfolioPerformanceAnalysis:
        """
        Analyse complète de la performance d'un portfolio.

        Args:
            portfolio: Portfolio à analyser
            analysis_period_days: Période d'analyse en jours
            benchmark_returns: Rendements du benchmark (optionnel)

        Returns:
            Analyse de performance complète
        """
        # Récupérer les snapshots pour la période
        snapshots = self._get_relevant_snapshots(portfolio, analysis_period_days)

        if len(snapshots) < 2:
            # Pas assez de données pour l'analyse
            return self._create_default_analysis(portfolio, analysis_period_days)

        # Calculer les rendements journaliers
        returns = self._calculate_daily_returns(snapshots)

        # Calculer les métriques de base
        total_return = self._calculate_total_return(snapshots)
        annualized_return = self._annualize_return(total_return, len(snapshots))
        volatility = self._calculate_volatility(returns)
        sharpe_ratio = self._calculate_sharpe_ratio(annualized_return, volatility)
        max_drawdown = self._calculate_max_drawdown(snapshots)

        # Métriques supplémentaires
        win_rate = self._calculate_win_rate(returns)
        best_day = max(returns) if returns else None
        worst_day = min(returns) if returns else None
        var_95 = self._calculate_var(returns, Decimal("0.95")) if returns else None

        # Métriques relatives au benchmark
        benchmark_return = None
        alpha = None
        beta = None
        information_ratio = None

        if benchmark_returns and len(benchmark_returns) == len(returns):
            benchmark_return = self._calculate_total_return_from_returns(benchmark_returns)
            beta = self._calculate_beta(returns, benchmark_returns)
            alpha = annualized_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
            information_ratio = self._calculate_information_ratio(returns, benchmark_returns)

        return PortfolioPerformanceAnalysis(
            portfolio_id=portfolio.id,
            analysis_period_days=analysis_period_days,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            best_day=best_day,
            worst_day=worst_day,
            var_95=var_95,
            benchmark_return=benchmark_return,
            alpha=alpha,
            beta=beta,
            information_ratio=information_ratio
        )

    def compare_portfolios(
        self,
        portfolios: List[Portfolio],
        metric: str = "sharpe_ratio"
    ) -> List[Tuple[Portfolio, PortfolioPerformanceAnalysis]]:
        """
        Compare plusieurs portfolios selon une métrique.

        Args:
            portfolios: Liste des portfolios à comparer
            metric: Métrique de comparaison

        Returns:
            Liste triée des portfolios avec leurs analyses
        """
        portfolio_analyses = []

        for portfolio in portfolios:
            analysis = self.analyze_portfolio_performance(portfolio)
            portfolio_analyses.append((portfolio, analysis))

        # Trier selon la métrique choisie
        def get_metric_value(item):
            _, analysis = item
            return getattr(analysis, metric, Decimal("0"))

        return sorted(portfolio_analyses, key=get_metric_value, reverse=True)

    # === Méthodes utilitaires ===

    def _calculate_current_allocations(self, portfolio: Portfolio) -> Dict[str, Decimal]:
        """Calcule les allocations actuelles du portfolio"""
        if portfolio.total_value == 0:
            return {}

        allocations = {}
        for symbol, position in portfolio.positions.items():
            allocations[symbol] = abs(position.market_value) / portfolio.total_value

        return allocations

    def _determine_rebalancing_reason(
        self,
        portfolio: Portfolio,
        current_allocations: Dict[str, Decimal],
        target_allocations: Dict[str, Decimal]
    ) -> str:
        """Détermine la raison du rééquilibrage"""
        max_drift = Decimal("0")
        max_drift_symbol = ""

        for symbol, target in target_allocations.items():
            current = current_allocations.get(symbol, Decimal("0"))
            drift = abs(current - target)
            if drift > max_drift:
                max_drift = drift
                max_drift_symbol = symbol

        if max_drift > Decimal("0.1"):
            return f"Large allocation drift detected: {max_drift_symbol} ({float(max_drift):.1%})"
        elif self.should_rebalance_by_frequency(portfolio):
            return f"Scheduled rebalancing ({portfolio.constraints.rebalancing_frequency.value})"
        else:
            return "Threshold-based rebalancing"

    def _get_relevant_snapshots(self, portfolio: Portfolio, days: int) -> List[PortfolioSnapshot]:
        """Récupère les snapshots pertinents pour l'analyse"""
        if not portfolio.snapshots:
            return []

        cutoff_date = datetime.utcnow() - timedelta(days=days)
        return [s for s in portfolio.snapshots if s.timestamp >= cutoff_date]

    def _create_default_analysis(self, portfolio: Portfolio, period_days: int) -> PortfolioPerformanceAnalysis:
        """Crée une analyse par défaut quand pas assez de données"""
        return PortfolioPerformanceAnalysis(
            portfolio_id=portfolio.id,
            analysis_period_days=period_days,
            total_return=Decimal("0"),
            annualized_return=Decimal("0"),
            volatility=Decimal("0"),
            sharpe_ratio=Decimal("0"),
            max_drawdown=Decimal("0"),
            win_rate=None,
            best_day=None,
            worst_day=None,
            var_95=None,
            benchmark_return=None,
            alpha=None,
            beta=None,
            information_ratio=None
        )

    def _calculate_daily_returns(self, snapshots: List[PortfolioSnapshot]) -> List[Decimal]:
        """Calcule les rendements journaliers"""
        if len(snapshots) < 2:
            return []

        returns = []
        for i in range(1, len(snapshots)):
            prev_value = snapshots[i-1].total_value
            curr_value = snapshots[i].total_value

            if prev_value > 0:
                return_pct = (curr_value - prev_value) / prev_value
                returns.append(return_pct)

        return returns

    def _calculate_total_return(self, snapshots: List[PortfolioSnapshot]) -> Decimal:
        """Calcule le rendement total"""
        if len(snapshots) < 2:
            return Decimal("0")

        initial_value = snapshots[0].total_value
        final_value = snapshots[-1].total_value

        if initial_value > 0:
            return (final_value - initial_value) / initial_value
        return Decimal("0")

    def _calculate_total_return_from_returns(self, returns: List[Decimal]) -> Decimal:
        """Calcule le rendement total à partir des rendements journaliers"""
        if not returns:
            return Decimal("0")

        cumulative = Decimal("1")
        for ret in returns:
            cumulative *= (1 + ret)

        return cumulative - 1

    def _annualize_return(self, total_return: Decimal, days: int) -> Decimal:
        """Annualise un rendement"""
        if days <= 0:
            return Decimal("0")

        years = Decimal(str(days)) / Decimal("252")  # 252 jours de trading par an
        if years > 0:
            return ((1 + total_return) ** (1 / years)) - 1
        return total_return

    def _calculate_volatility(self, returns: List[Decimal]) -> Decimal:
        """Calcule la volatilité annualisée"""
        if len(returns) < 2:
            return Decimal("0")

        float_returns = [float(r) for r in returns]
        daily_vol = Decimal(str(statistics.stdev(float_returns)))
        return daily_vol * Decimal(str(math.sqrt(252)))  # Annualiser

    def _calculate_sharpe_ratio(self, annual_return: Decimal, volatility: Decimal) -> Decimal:
        """Calcule le ratio de Sharpe"""
        if volatility == 0:
            return Decimal("0")
        return (annual_return - self.risk_free_rate) / volatility

    def _calculate_max_drawdown(self, snapshots: List[PortfolioSnapshot]) -> Decimal:
        """Calcule le drawdown maximum"""
        if len(snapshots) < 2:
            return Decimal("0")

        peak = snapshots[0].total_value
        max_drawdown = Decimal("0")

        for snapshot in snapshots[1:]:
            if snapshot.total_value > peak:
                peak = snapshot.total_value

            drawdown = (peak - snapshot.total_value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def _calculate_win_rate(self, returns: List[Decimal]) -> Optional[Decimal]:
        """Calcule le taux de réussite (jours positifs)"""
        if not returns:
            return None

        positive_days = sum(1 for r in returns if r > 0)
        return Decimal(str(positive_days)) / Decimal(str(len(returns)))

    def _calculate_var(self, returns: List[Decimal], confidence: Decimal) -> Decimal:
        """Calcule la Value at Risk"""
        if not returns:
            return Decimal("0")

        sorted_returns = sorted([float(r) for r in returns])
        index = int((1 - float(confidence)) * len(sorted_returns))
        index = max(0, min(index, len(sorted_returns) - 1))

        return Decimal(str(abs(sorted_returns[index])))

    def _calculate_beta(self, portfolio_returns: List[Decimal], benchmark_returns: List[Decimal]) -> Decimal:
        """Calcule le beta du portfolio"""
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
            return Decimal("1")

        port_floats = [float(r) for r in portfolio_returns]
        bench_floats = [float(r) for r in benchmark_returns]

        # Calculer la covariance et la variance
        port_mean = statistics.mean(port_floats)
        bench_mean = statistics.mean(bench_floats)

        covariance = sum((p - port_mean) * (b - bench_mean) for p, b in zip(port_floats, bench_floats)) / len(port_floats)
        bench_variance = statistics.variance(bench_floats)

        if bench_variance > 0:
            return Decimal(str(covariance / bench_variance))
        return Decimal("1")

    def _calculate_information_ratio(self, portfolio_returns: List[Decimal], benchmark_returns: List[Decimal]) -> Decimal:
        """Calcule le ratio d'information"""
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
            return Decimal("0")

        # Calculer les rendements excédentaires
        excess_returns = [p - b for p, b in zip(portfolio_returns, benchmark_returns)]
        excess_floats = [float(r) for r in excess_returns]

        if len(excess_floats) < 2:
            return Decimal("0")

        mean_excess = Decimal(str(statistics.mean(excess_floats)))
        std_excess = Decimal(str(statistics.stdev(excess_floats)))

        if std_excess > 0:
            return mean_excess / std_excess
        return Decimal("0")

    # === Méthodes d'alias pour compatibilité ===

    def calculate_rebalancing_plan(
        self,
        portfolio: Portfolio,
        target_allocations: Optional[Dict[str, Decimal]] = None,
        rebalancing_threshold: Decimal = Decimal("0.05"),
        transaction_cost_rate: Decimal = Decimal("0.001")
    ) -> Optional[RebalancingPlan]:
        """
        Alias pour create_rebalancing_plan pour compatibilité avec les tests.
        """
        return self.create_rebalancing_plan(
            portfolio, target_allocations, rebalancing_threshold, transaction_cost_rate
        )

    def calculate_risk_metrics(self, portfolio: Portfolio) -> Optional[Dict[str, Any]]:
        """
        Calcule les métriques de risque pour un portfolio.

        Args:
            portfolio: Portfolio à analyser

        Returns:
            Dictionnaire des métriques de risque ou None si pas assez de données
        """
        if len(portfolio.snapshots) < 2:
            return None

        try:
            # Calculer métriques de base
            returns = self._calculate_daily_returns(portfolio.snapshots)

            if len(returns) < 2:
                return None

            return_floats = [float(r) for r in returns]
            volatility = Decimal(str(statistics.stdev(return_floats)))

            # VaR 95%
            var_95 = self._calculate_var(returns, Decimal("0.95"))

            # Maximum Drawdown
            max_dd = self._calculate_max_drawdown(portfolio.snapshots)

            # Sharpe ratio
            avg_return = Decimal(str(statistics.mean(return_floats)))
            annualized_return = self._annualize_return(avg_return, len(returns))
            sharpe = self._calculate_sharpe_ratio(annualized_return, volatility)

            return {
                "volatility": float(volatility),
                "var_95": float(var_95),
                "max_drawdown": float(max_dd),
                "sharpe_ratio": float(sharpe),
                "avg_daily_return": float(avg_return),
                "observation_count": len(returns)
            }

        except Exception as e:
            # En cas d'erreur, retourner métriques simplifiées
            return {
                "volatility": 0.0,
                "var_95": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "avg_daily_return": 0.0,
                "observation_count": 0,
                "error": str(e)
            }

    def calculate_concentration_risk(self, portfolio: Portfolio) -> Decimal:
        """
        Calcule le risque de concentration du portfolio.

        Args:
            portfolio: Portfolio à analyser

        Returns:
            Score de concentration (0 = très diversifié, 1 = très concentré)
        """
        if not portfolio.positions or portfolio.total_value == 0:
            return Decimal("0")

        # Calculer l'indice Herfindahl-Hirschman (HHI)
        allocations = portfolio.get_allocation_percentages()

        # Exclure le cash du calcul de concentration
        position_allocations = {k: v for k, v in allocations.items() if k != "CASH"}

        if not position_allocations:
            return Decimal("0")

        # HHI = somme des carrés des poids
        hhi = sum(Decimal(str(weight)) ** 2 for weight in position_allocations.values())

        # Normaliser: HHI varie de 1/n (très diversifié) à 1 (très concentré)
        # où n est le nombre de positions
        n_positions = len(position_allocations)
        min_hhi = Decimal("1") / Decimal(str(n_positions))

        if min_hhi >= 1:
            return Decimal("0")

        # Score normalisé entre 0 et 1
        concentration_score = (hhi - min_hhi) / (Decimal("1") - min_hhi)

        return min(concentration_score, Decimal("1"))

    def estimate_correlation_risk(self, portfolio: Portfolio) -> Decimal:
        """
        Estime le risque de corrélation du portfolio.

        Args:
            portfolio: Portfolio à analyser

        Returns:
            Score de corrélation estimé (0 = faible corrélation, 1 = forte corrélation)
        """
        if not portfolio.positions or len(portfolio.positions) < 2:
            return Decimal("0")

        # Estimation simplifiée basée sur les secteurs/types d'actifs
        symbols = list(portfolio.positions.keys())

        # Heuristiques simples pour estimer la corrélation
        crypto_count = sum(1 for symbol in symbols if any(
            crypto in symbol.upper() for crypto in ["BTC", "ETH", "ADA", "DOT", "SOL", "USDT", "USDC"]
        ))

        stock_count = len(symbols) - crypto_count

        # Si tout est crypto ou tout est stock -> haute corrélation
        if crypto_count == len(symbols) or stock_count == len(symbols):
            correlation_risk = Decimal("0.8")  # 80% de risque de corrélation
        else:
            # Mix crypto/stock -> corrélation modérée
            correlation_risk = Decimal("0.4")  # 40% de risque de corrélation

        # Ajuster selon le nombre de positions
        # Plus de positions = généralement moins de corrélation
        position_factor = max(Decimal("0.5"), Decimal("1") - Decimal(str(len(symbols))) / Decimal("20"))

        return correlation_risk * position_factor