"""
Backtest Results
===============

Comprehensive backtest results with analysis and visualization.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class BacktestResults:
    """Résultats complets d'un backtesting"""

    # Métriques de performance
    performance_metrics: Any  # PerformanceMetrics

    # Séries temporelles
    portfolio_values: pd.Series
    positions: pd.Series
    trades: List[Dict[str, Any]]

    # Configuration et métadonnées
    config: Any  # BacktestConfig
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Métriques dérivées (calculées à la demande)
    _drawdown_series: Optional[pd.Series] = None
    _rolling_metrics: Optional[Dict[str, pd.Series]] = None
    _trade_analysis: Optional[Dict[str, Any]] = None

    def get_summary_report(self) -> Dict[str, Any]:
        """Retourne un rapport de synthèse"""
        return {
            "test_period": {
                "start": self.portfolio_values.index[0].isoformat(),
                "end": self.portfolio_values.index[-1].isoformat(),
                "duration_days": (self.portfolio_values.index[-1] - self.portfolio_values.index[0]).days
            },
            "performance": {
                "total_return": float(self.performance_metrics.total_return),
                "annualized_return": float(self.performance_metrics.annualized_return),
                "volatility": float(self.performance_metrics.volatility),
                "sharpe_ratio": float(self.performance_metrics.sharpe_ratio),
                "max_drawdown": float(self.performance_metrics.max_drawdown),
                "calmar_ratio": float(self.performance_metrics.calmar_ratio)
            },
            "trading": {
                "total_trades": self.performance_metrics.total_trades,
                "win_rate": float(self.performance_metrics.win_rate),
                "profit_factor": float(self.performance_metrics.profit_factor),
                "avg_trade_return": float(self.performance_metrics.avg_trade_return)
            },
            "execution": {
                "backtest_time": self.execution_time,
                "trades_per_second": self.performance_metrics.total_trades / self.execution_time if self.execution_time > 0 else 0
            }
        }

    def get_drawdown_series(self) -> pd.Series:
        """Calcule et retourne la série de drawdowns"""
        if self._drawdown_series is None:
            rolling_max = self.portfolio_values.expanding().max()
            self._drawdown_series = (self.portfolio_values - rolling_max) / rolling_max
        return self._drawdown_series

    def get_rolling_metrics(self, window: int = 252) -> Dict[str, pd.Series]:
        """Calcule les métriques sur fenêtre glissante"""
        if self._rolling_metrics is None or window != getattr(self, '_rolling_window', None):
            returns = self.portfolio_values.pct_change().dropna()

            self._rolling_metrics = {
                "rolling_return": returns.rolling(window).mean() * 252,
                "rolling_volatility": returns.rolling(window).std() * np.sqrt(252),
                "rolling_sharpe": (returns.rolling(window).mean() * 252) / (returns.rolling(window).std() * np.sqrt(252)),
                "rolling_drawdown": self.get_drawdown_series().rolling(window).min()
            }
            self._rolling_window = window

        return self._rolling_metrics

    def get_trade_analysis(self) -> Dict[str, Any]:
        """Analyse détaillée des trades"""
        if self._trade_analysis is None and self.trades:
            trade_returns = [t.get("pnl", 0) for t in self.trades]
            winning_trades = [r for r in trade_returns if r > 0]
            losing_trades = [r for r in trade_returns if r < 0]

            # Séries de trades
            consecutive_wins = 0
            consecutive_losses = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            current_streak_wins = 0
            current_streak_losses = 0

            for trade_return in trade_returns:
                if trade_return > 0:
                    current_streak_wins += 1
                    current_streak_losses = 0
                    max_consecutive_wins = max(max_consecutive_wins, current_streak_wins)
                elif trade_return < 0:
                    current_streak_losses += 1
                    current_streak_wins = 0
                    max_consecutive_losses = max(max_consecutive_losses, current_streak_losses)

            self._trade_analysis = {
                "total_trades": len(self.trades),
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate": len(winning_trades) / len(self.trades) if self.trades else 0,
                "avg_win": np.mean(winning_trades) if winning_trades else 0,
                "avg_loss": np.mean(losing_trades) if losing_trades else 0,
                "largest_win": max(winning_trades) if winning_trades else 0,
                "largest_loss": min(losing_trades) if losing_trades else 0,
                "profit_factor": sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else float('inf'),
                "max_consecutive_wins": max_consecutive_wins,
                "max_consecutive_losses": max_consecutive_losses,
                "trades_per_day": len(self.trades) / ((self.portfolio_values.index[-1] - self.portfolio_values.index[0]).days or 1)
            }

        return self._trade_analysis or {}

    def generate_performance_chart_data(self) -> Dict[str, Any]:
        """Génère les données pour graphiques de performance"""

        # Série de performance
        performance_data = [{
            "date": idx.isoformat(),
            "portfolio_value": float(value),
            "cumulative_return": float((value / self.portfolio_values.iloc[0]) - 1)
        } for idx, value in self.portfolio_values.items()]

        # Série de drawdown
        drawdown_series = self.get_drawdown_series()
        drawdown_data = [{
            "date": idx.isoformat(),
            "drawdown": float(dd)
        } for idx, dd in drawdown_series.items()]

        # Distribution des returns
        returns = self.portfolio_values.pct_change().dropna()
        return_distribution = {
            "daily_returns": [float(r) for r in returns],
            "return_histogram": np.histogram(returns, bins=50)[0].tolist(),
            "return_bins": np.histogram(returns, bins=50)[1].tolist()
        }

        return {
            "performance_series": performance_data,
            "drawdown_series": drawdown_data,
            "return_distribution": return_distribution,
            "trades": [
                {
                    "date": trade.get("timestamp", datetime.now()).isoformat() if hasattr(trade.get("timestamp", ""), "isoformat") else str(trade.get("timestamp", "")),
                    "symbol": trade.get("symbol", ""),
                    "action": trade.get("action", ""),
                    "size": trade.get("size", 0),
                    "price": trade.get("price", 0),
                    "pnl": trade.get("pnl", 0)
                }
                for trade in self.trades
            ]
        }

    def export_to_excel(self, filename: str) -> None:
        """Exporte les résultats vers Excel"""
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Feuille de synthèse
                summary_df = pd.DataFrame([self.get_summary_report()])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

                # Série de performance
                performance_df = pd.DataFrame({
                    'Date': self.portfolio_values.index,
                    'Portfolio_Value': self.portfolio_values.values,
                    'Drawdown': self.get_drawdown_series().values
                })
                performance_df.to_excel(writer, sheet_name='Performance', index=False)

                # Trades
                if self.trades:
                    trades_df = pd.DataFrame(self.trades)
                    trades_df.to_excel(writer, sheet_name='Trades', index=False)

                # Métriques glissantes
                rolling_metrics = self.get_rolling_metrics()
                if rolling_metrics:
                    rolling_df = pd.DataFrame(rolling_metrics)
                    rolling_df.to_excel(writer, sheet_name='Rolling_Metrics', index=False)

                logger.info(f"Results exported to {filename}")

        except Exception as e:
            logger.error(f"Failed to export results to Excel: {e}")

    def compare_with_benchmark(self, benchmark_returns: pd.Series) -> Dict[str, Any]:
        """Compare la performance avec un benchmark"""

        # Aligner les dates
        strategy_returns = self.portfolio_values.pct_change().dropna()

        # Réindexer le benchmark sur les mêmes dates
        aligned_benchmark = benchmark_returns.reindex(strategy_returns.index, method='ffill').dropna()
        aligned_strategy = strategy_returns.reindex(aligned_benchmark.index).dropna()

        if len(aligned_strategy) == 0 or len(aligned_benchmark) == 0:
            return {"error": "No overlapping dates with benchmark"}

        # Calculer métriques comparatives
        excess_returns = aligned_strategy - aligned_benchmark

        # Beta et Alpha (régression simple)
        covariance = np.cov(aligned_strategy, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

        strategy_mean = np.mean(aligned_strategy) * 252
        benchmark_mean = np.mean(aligned_benchmark) * 252
        alpha = strategy_mean - beta * benchmark_mean

        # Information ratio
        tracking_error = np.std(excess_returns) * np.sqrt(252)
        information_ratio = np.mean(excess_returns) * 252 / tracking_error if tracking_error > 0 else 0

        return {
            "alpha_annualized": alpha,
            "beta": beta,
            "information_ratio": information_ratio,
            "tracking_error": tracking_error,
            "correlation": np.corrcoef(aligned_strategy, aligned_benchmark)[0, 1],
            "excess_return_annualized": np.mean(excess_returns) * 252,
            "periods_compared": len(aligned_strategy)
        }

    def get_risk_metrics(self) -> Dict[str, float]:
        """Calcule des métriques de risque avancées"""

        returns = self.portfolio_values.pct_change().dropna()

        if len(returns) == 0:
            return {}

        # VaR et CVaR
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)

        # CVaR (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
        cvar_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else 0

        # Skewness et Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # Downside deviation
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() if len(negative_returns) > 0 else 0

        # Ulcer Index
        drawdown_series = self.get_drawdown_series()
        ulcer_index = np.sqrt(np.mean(drawdown_series ** 2))

        return {
            "var_95_daily": float(var_95),
            "var_99_daily": float(var_99),
            "cvar_95_daily": float(cvar_95),
            "cvar_99_daily": float(cvar_99),
            "var_95_annualized": float(var_95 * np.sqrt(252)),
            "cvar_95_annualized": float(cvar_95 * np.sqrt(252)),
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
            "downside_deviation_annualized": float(downside_deviation * np.sqrt(252)),
            "ulcer_index": float(ulcer_index)
        }

    def validate_results(self) -> Dict[str, Any]:
        """Valide la cohérence des résultats"""

        validation_results = {
            "is_valid": True,
            "warnings": [],
            "errors": []
        }

        # Vérifier cohérence des données
        if len(self.portfolio_values) == 0:
            validation_results["errors"].append("Empty portfolio values series")
            validation_results["is_valid"] = False

        if len(self.trades) != self.performance_metrics.total_trades:
            validation_results["warnings"].append(f"Trade count mismatch: {len(self.trades)} vs {self.performance_metrics.total_trades}")

        # Vérifier cohérence des métriques
        calculated_total_return = (self.portfolio_values.iloc[-1] / self.portfolio_values.iloc[0]) - 1 if len(self.portfolio_values) > 0 else 0
        reported_total_return = float(self.performance_metrics.total_return)

        if abs(calculated_total_return - reported_total_return) > 0.01:  # 1% tolerance
            validation_results["warnings"].append(f"Total return discrepancy: calculated {calculated_total_return:.4f} vs reported {reported_total_return:.4f}")

        # Vérifier drawdown
        max_dd_calculated = abs(self.get_drawdown_series().min()) if len(self.portfolio_values) > 0 else 0
        max_dd_reported = float(self.performance_metrics.max_drawdown)

        if abs(max_dd_calculated - max_dd_reported) > 0.01:
            validation_results["warnings"].append(f"Max drawdown discrepancy: calculated {max_dd_calculated:.4f} vs reported {max_dd_reported:.4f}")

        # Vérifier valeurs aberrantes
        if len(self.portfolio_values) > 0:
            portfolio_changes = self.portfolio_values.pct_change().dropna()
            extreme_changes = portfolio_changes[abs(portfolio_changes) > 0.5]  # Changes > 50%

            if len(extreme_changes) > 0:
                validation_results["warnings"].append(f"Found {len(extreme_changes)} extreme portfolio changes (>50%)")

        return validation_results

    def __str__(self) -> str:
        """Représentation string des résultats"""
        summary = self.get_summary_report()

        return f"""
Backtest Results Summary
========================
Period: {summary['test_period']['start']} to {summary['test_period']['end']} ({summary['test_period']['duration_days']} days)

Performance:
- Total Return: {summary['performance']['total_return']:.2%}
- Annualized Return: {summary['performance']['annualized_return']:.2%}
- Volatility: {summary['performance']['volatility']:.2%}
- Sharpe Ratio: {summary['performance']['sharpe_ratio']:.3f}
- Max Drawdown: {summary['performance']['max_drawdown']:.2%}
- Calmar Ratio: {summary['performance']['calmar_ratio']:.3f}

Trading:
- Total Trades: {summary['trading']['total_trades']}
- Win Rate: {summary['trading']['win_rate']:.2%}
- Profit Factor: {summary['trading']['profit_factor']:.3f}
- Avg Trade Return: {summary['trading']['avg_trade_return']:.4f}

Execution:
- Backtest Time: {summary['execution']['backtest_time']:.3f}s
- Trades/Second: {summary['execution']['trades_per_second']:.1f}
        """