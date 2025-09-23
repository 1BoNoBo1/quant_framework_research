"""
Infrastructure Layer: Performance Calculator
==========================================

Calculateur de métriques de performance pour le backtesting.
Implémente tous les calculs de métriques financières et de risque.
"""

import numpy as np
import pandas as pd
from decimal import Decimal
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import scipy.stats as stats

from qframe.domain.entities.backtest import BacktestMetrics, TradeExecution


class PerformanceCalculator:
    """
    Calculateur de performance complet pour les backtests.
    Implémente toutes les métriques financières standard et avancées.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate  # Taux sans risque annuel

    def calculate_comprehensive_metrics(
        self,
        returns: pd.Series,
        portfolio_values: pd.Series,
        trades: List[TradeExecution],
        benchmark_returns: Optional[pd.Series] = None,
        initial_capital: Decimal = Decimal("100000")
    ) -> BacktestMetrics:
        """
        Calcule toutes les métriques de performance de façon complète.

        Args:
            returns: Série des returns du portfolio
            portfolio_values: Série des valeurs du portfolio
            trades: Liste des trades exécutés
            benchmark_returns: Returns du benchmark (optionnel)
            initial_capital: Capital initial

        Returns:
            BacktestMetrics complètes
        """
        if len(returns) == 0:
            return BacktestMetrics()

        # Métriques de base
        basic_metrics = self._calculate_basic_metrics(returns, initial_capital)

        # Métriques de risque
        risk_metrics = self._calculate_risk_metrics(returns, portfolio_values)

        # Métriques de trading
        trading_metrics = self._calculate_trading_metrics(trades, returns)

        # Métriques vs benchmark
        benchmark_metrics = {}
        if benchmark_returns is not None:
            benchmark_metrics = self._calculate_benchmark_metrics(returns, benchmark_returns)

        # Métriques avancées
        advanced_metrics = self._calculate_advanced_metrics(returns)

        # Métriques de stabilité
        stability_metrics = self._calculate_stability_metrics(returns)

        # Combiner toutes les métriques
        metrics = BacktestMetrics(
            # Métriques de base
            **basic_metrics,
            # Métriques de risque
            **risk_metrics,
            # Métriques de trading
            **trading_metrics,
            # Métriques vs benchmark
            **benchmark_metrics,
            # Métriques avancées
            **advanced_metrics,
            # Métriques de stabilité
            **stability_metrics
        )

        return metrics

    def _calculate_basic_metrics(
        self,
        returns: pd.Series,
        initial_capital: Decimal
    ) -> Dict[str, Any]:
        """Calcule les métriques de performance de base"""

        # Returns cumulatifs
        cumulative_returns = (1 + returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1

        # Returns annualisés
        periods_per_year = self._infer_periods_per_year(returns)
        num_periods = len(returns)
        annualized_return = (1 + total_return) ** (periods_per_year / num_periods) - 1

        # Volatilité
        volatility = returns.std() * np.sqrt(periods_per_year)

        # Sharpe ratio
        excess_return = annualized_return - self.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0

        return {
            "total_return": Decimal(str(total_return)),
            "annualized_return": Decimal(str(annualized_return)),
            "volatility": Decimal(str(volatility)),
            "sharpe_ratio": Decimal(str(sharpe_ratio))
        }

    def _calculate_risk_metrics(
        self,
        returns: pd.Series,
        portfolio_values: pd.Series
    ) -> Dict[str, Any]:
        """Calcule les métriques de risque"""

        # Maximum Drawdown
        max_drawdown, max_dd_duration = self._calculate_max_drawdown(portfolio_values)

        # Value at Risk et Expected Shortfall
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        expected_shortfall_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0

        # Sortino Ratio
        sortino_ratio = self._calculate_sortino_ratio(returns)

        # Calmar Ratio
        annualized_return = float(self._calculate_basic_metrics(returns, Decimal("100000"))["annualized_return"])
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            "max_drawdown": Decimal(str(max_drawdown)),
            "max_drawdown_duration": max_dd_duration,
            "value_at_risk_95": Decimal(str(var_95)),
            "expected_shortfall_95": Decimal(str(expected_shortfall_95)),
            "sortino_ratio": Decimal(str(sortino_ratio)),
            "calmar_ratio": Decimal(str(calmar_ratio))
        }

    def _calculate_trading_metrics(
        self,
        trades: List[TradeExecution],
        returns: pd.Series
    ) -> Dict[str, Any]:
        """Calcule les métriques de trading"""

        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": Decimal("0"),
                "profit_factor": Decimal("0"),
                "average_trade_return": Decimal("0"),
                "average_win": Decimal("0"),
                "average_loss": Decimal("0")
            }

        # Calculer les P&L des trades
        trade_pnls = [float(trade.value - trade.commission - trade.slippage) for trade in trades]

        # Séparer gains et pertes
        wins = [pnl for pnl in trade_pnls if pnl > 0]
        losses = [pnl for pnl in trade_pnls if pnl < 0]

        # Métriques de base
        total_trades = len(trades)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Profit Factor
        total_profits = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_profits / total_losses if total_losses > 0 else float('inf')

        # Moyennes
        avg_trade_return = sum(trade_pnls) / len(trade_pnls) if trade_pnls else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": Decimal(str(win_rate)),
            "profit_factor": Decimal(str(profit_factor)),
            "average_trade_return": Decimal(str(avg_trade_return)),
            "average_win": Decimal(str(avg_win)),
            "average_loss": Decimal(str(avg_loss))
        }

    def _calculate_benchmark_metrics(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, Any]:
        """Calcule les métriques par rapport au benchmark"""

        # Aligner les séries temporelles
        aligned_returns = portfolio_returns.align(benchmark_returns, join='inner')
        port_ret = aligned_returns[0].dropna()
        bench_ret = aligned_returns[1].dropna()

        if len(port_ret) == 0 or len(bench_ret) == 0:
            return {}

        # Excess returns
        excess_returns = port_ret - bench_ret

        # Alpha et Beta (régression linéaire)
        if len(port_ret) > 1:
            beta, alpha, r_value, p_value, std_err = stats.linregress(bench_ret, port_ret)
        else:
            beta = alpha = 0

        # Information Ratio
        tracking_error = excess_returns.std() * np.sqrt(self._infer_periods_per_year(excess_returns))
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(self._infer_periods_per_year(excess_returns)) if excess_returns.std() > 0 else 0

        return {
            "alpha": Decimal(str(alpha)) if alpha else None,
            "beta": Decimal(str(beta)) if beta else None,
            "information_ratio": Decimal(str(information_ratio)) if information_ratio else None,
            "tracking_error": Decimal(str(tracking_error)) if tracking_error else None
        }

    def _calculate_advanced_metrics(
        self,
        returns: pd.Series
    ) -> Dict[str, Any]:
        """Calcule les métriques avancées"""

        # Tail Ratio
        tail_ratio = abs(returns.quantile(0.95)) / abs(returns.quantile(0.05)) if returns.quantile(0.05) != 0 else 0

        # Skewness et Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # Séquences consécutives
        max_consecutive_wins = self._calculate_max_consecutive_sequence(returns, positive=True)
        max_consecutive_losses = self._calculate_max_consecutive_sequence(returns, positive=False)

        return {
            "tail_ratio": Decimal(str(tail_ratio)),
            "skewness": Decimal(str(skewness)),
            "kurtosis": Decimal(str(kurtosis)),
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses
        }

    def _calculate_stability_metrics(
        self,
        returns: pd.Series
    ) -> Dict[str, Any]:
        """Calcule les métriques de stabilité"""

        # Consistance des returns (% de périodes positives)
        return_consistency = (returns > 0).mean()

        # Stabilité du Sharpe ratio (rolling window)
        window_size = min(252, len(returns) // 4)  # 1 an ou 1/4 de la période
        if window_size >= 30:  # Minimum pour calculer
            rolling_sharpe = returns.rolling(window_size).apply(
                lambda x: x.mean() / x.std() * np.sqrt(self._infer_periods_per_year(returns)) if x.std() > 0 else 0
            )
            rolling_sharpe_std = rolling_sharpe.std()
        else:
            rolling_sharpe_std = 0

        return {
            "return_consistency": Decimal(str(return_consistency)),
            "rolling_sharpe_std": Decimal(str(rolling_sharpe_std))
        }

    def _calculate_max_drawdown(
        self,
        portfolio_values: pd.Series
    ) -> Tuple[float, int]:
        """
        Calcule le maximum drawdown et sa durée.

        Returns:
            Tuple (max_drawdown, duration_days)
        """
        if len(portfolio_values) == 0:
            return 0.0, 0

        # Calculer les pics historiques
        peak = portfolio_values.expanding().max()

        # Calculer le drawdown
        drawdown = (portfolio_values - peak) / peak

        # Maximum drawdown
        max_drawdown = drawdown.min()

        # Calculer la durée du maximum drawdown
        max_dd_duration = 0
        current_duration = 0
        max_dd_start = None

        for i, dd in enumerate(drawdown):
            if dd < 0:  # En drawdown
                if current_duration == 0:
                    max_dd_start = i
                current_duration += 1
            else:  # Nouveau pic
                if current_duration > max_dd_duration:
                    max_dd_duration = current_duration
                current_duration = 0

        # Vérifier la dernière séquence
        if current_duration > max_dd_duration:
            max_dd_duration = current_duration

        return float(max_drawdown), max_dd_duration

    def _calculate_sortino_ratio(
        self,
        returns: pd.Series
    ) -> float:
        """Calcule le ratio de Sortino"""
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            return float('inf')

        downside_std = downside_returns.std()
        if downside_std == 0:
            return float('inf')

        periods_per_year = self._infer_periods_per_year(returns)
        excess_return = returns.mean() - self.risk_free_rate / periods_per_year

        return excess_return / downside_std * np.sqrt(periods_per_year)

    def _calculate_max_consecutive_sequence(
        self,
        returns: pd.Series,
        positive: bool = True
    ) -> int:
        """Calcule la séquence consécutive maximale de gains ou pertes"""
        if positive:
            sequence = returns > 0
        else:
            sequence = returns < 0

        max_consecutive = 0
        current_consecutive = 0

        for value in sequence:
            if value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _infer_periods_per_year(self, returns: pd.Series) -> int:
        """Infère le nombre de périodes par an à partir de l'index"""
        if len(returns) < 2:
            return 252  # Défaut: daily

        # Calculer la fréquence moyenne
        time_diff = returns.index[-1] - returns.index[0]
        avg_period = time_diff / (len(returns) - 1)

        # Mapper vers des fréquences standard
        if avg_period.days >= 360:
            return 1  # Annuel
        elif avg_period.days >= 80:
            return 4  # Trimestriel
        elif avg_period.days >= 25:
            return 12  # Mensuel
        elif avg_period.days >= 6:
            return 52  # Hebdomadaire
        elif avg_period.total_seconds() >= 20 * 3600:  # Plus de 20 heures
            return 252  # Daily
        else:
            return 252 * 24  # Hourly

    def calculate_rolling_metrics(
        self,
        returns: pd.Series,
        window_size: int = 252
    ) -> pd.DataFrame:
        """
        Calcule des métriques sur une fenêtre glissante.

        Args:
            returns: Série des returns
            window_size: Taille de la fenêtre (défaut: 252 jours)

        Returns:
            DataFrame avec les métriques rolling
        """
        if len(returns) < window_size:
            window_size = len(returns)

        periods_per_year = self._infer_periods_per_year(returns)

        rolling_metrics = pd.DataFrame(index=returns.index)

        # Return rolling
        rolling_metrics['rolling_return'] = returns.rolling(window_size).sum()

        # Volatilité rolling
        rolling_metrics['rolling_volatility'] = returns.rolling(window_size).std() * np.sqrt(periods_per_year)

        # Sharpe rolling
        rolling_metrics['rolling_sharpe'] = returns.rolling(window_size).apply(
            lambda x: (x.mean() * periods_per_year - self.risk_free_rate) / (x.std() * np.sqrt(periods_per_year)) if x.std() > 0 else 0
        )

        # Max drawdown rolling
        portfolio_values = (1 + returns).cumprod()
        rolling_metrics['rolling_max_dd'] = portfolio_values.rolling(window_size).apply(
            lambda x: self._calculate_max_drawdown(x)[0]
        )

        return rolling_metrics.dropna()

    def calculate_monte_carlo_confidence_intervals(
        self,
        simulation_results: List[Dict[str, float]],
        confidence_levels: List[float] = [0.05, 0.25, 0.75, 0.95]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calcule les intervalles de confiance à partir des résultats Monte Carlo.

        Args:
            simulation_results: Liste des résultats de simulation
            confidence_levels: Niveaux de confiance à calculer

        Returns:
            Dictionnaire des intervalles de confiance par métrique
        """
        if not simulation_results:
            return {}

        # Extraire les métriques
        metrics_data = {}
        for result in simulation_results:
            for metric, value in result.items():
                if metric not in metrics_data:
                    metrics_data[metric] = []
                metrics_data[metric].append(value)

        # Calculer les intervalles de confiance
        confidence_intervals = {}
        for metric, values in metrics_data.items():
            if values:
                confidence_intervals[metric] = {}
                for cl in confidence_levels:
                    confidence_intervals[metric][f'p{int(cl*100)}'] = np.percentile(values, cl*100)

                # Ajouter des statistiques supplémentaires
                confidence_intervals[metric]['mean'] = np.mean(values)
                confidence_intervals[metric]['std'] = np.std(values)
                confidence_intervals[metric]['min'] = np.min(values)
                confidence_intervals[metric]['max'] = np.max(values)

        return confidence_intervals

    def calculate_strategy_comparison_metrics(
        self,
        strategy_results: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """
        Compare les performances de plusieurs stratégies.

        Args:
            strategy_results: Dictionnaire {strategy_name: returns_series}

        Returns:
            Métriques de comparaison
        """
        comparison = {}

        for strategy_name, returns in strategy_results.items():
            if len(returns) > 0:
                basic_metrics = self._calculate_basic_metrics(returns, Decimal("100000"))
                risk_metrics = self._calculate_risk_metrics(returns, (1 + returns).cumprod())

                comparison[strategy_name] = {
                    "total_return": float(basic_metrics["total_return"]),
                    "annualized_return": float(basic_metrics["annualized_return"]),
                    "volatility": float(basic_metrics["volatility"]),
                    "sharpe_ratio": float(basic_metrics["sharpe_ratio"]),
                    "max_drawdown": float(risk_metrics["max_drawdown"]),
                    "sortino_ratio": float(risk_metrics["sortino_ratio"])
                }

        # Calculer des métriques de comparaison
        if len(comparison) > 1:
            metrics_names = ['sharpe_ratio', 'total_return', 'max_drawdown', 'volatility']
            for metric in metrics_names:
                values = [strategy[metric] for strategy in comparison.values() if metric in strategy]
                if values:
                    best_strategy = max(comparison.keys(),
                                      key=lambda s: comparison[s][metric] if metric != 'max_drawdown' else -comparison[s][metric])
                    comparison[f'best_{metric}'] = best_strategy

        return comparison

    def calculate_market_regime_performance(
        self,
        returns: pd.Series,
        market_returns: pd.Series,
        volatility_threshold: float = 0.02
    ) -> Dict[str, Any]:
        """
        Analyse la performance selon les régimes de marché.

        Args:
            returns: Returns de la stratégie
            market_returns: Returns du marché
            volatility_threshold: Seuil de volatilité pour définir les régimes

        Returns:
            Performance par régime de marché
        """
        # Aligner les séries
        aligned = returns.align(market_returns, join='inner')
        strategy_ret = aligned[0].dropna()
        market_ret = aligned[1].dropna()

        if len(strategy_ret) == 0:
            return {}

        # Identifier les régimes de marché
        market_volatility = market_ret.rolling(21).std()  # Volatilité sur 21 jours

        # Bull market: returns positifs et volatilité faible
        bull_market = (market_ret > 0) & (market_volatility < volatility_threshold)

        # Bear market: returns négatifs
        bear_market = market_ret < 0

        # High volatility: volatilité élevée
        high_vol = market_volatility > volatility_threshold

        regimes = {
            'bull_market': strategy_ret[bull_market],
            'bear_market': strategy_ret[bear_market],
            'high_volatility': strategy_ret[high_vol],
            'normal': strategy_ret[~(bull_market | bear_market | high_vol)]
        }

        regime_performance = {}
        for regime_name, regime_returns in regimes.items():
            if len(regime_returns) > 0:
                regime_performance[regime_name] = {
                    'periods': len(regime_returns),
                    'total_return': float((1 + regime_returns).prod() - 1),
                    'annualized_return': float((1 + regime_returns.mean()) ** self._infer_periods_per_year(regime_returns) - 1),
                    'volatility': float(regime_returns.std() * np.sqrt(self._infer_periods_per_year(regime_returns))),
                    'sharpe_ratio': float(regime_returns.mean() / regime_returns.std() * np.sqrt(self._infer_periods_per_year(regime_returns))) if regime_returns.std() > 0 else 0,
                    'win_rate': float((regime_returns > 0).mean())
                }

        return regime_performance