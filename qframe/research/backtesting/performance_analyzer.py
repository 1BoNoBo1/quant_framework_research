"""
📊 Advanced Performance Analyzer

Enhanced performance metrics and visualization for QFrame backtesting results.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from qframe.domain.entities.backtest import BacktestResult


class AdvancedPerformanceAnalyzer:
    """
    🔬 Advanced performance analysis for backtesting results

    Provides comprehensive metrics beyond basic Sharpe ratio:
    - Risk-adjusted returns (Sortino, Calmar)
    - Rolling performance metrics
    - Drawdown analysis
    - Factor exposure analysis
    - Performance attribution
    """

    def __init__(self):
        self.results_cache = {}

    def analyze_comprehensive(
        self,
        backtest_result: BacktestResult,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive performance analysis

        Args:
            backtest_result: Backtest results to analyze
            benchmark_returns: Optional benchmark for comparison

        Returns:
            Dictionary with all performance metrics
        """
        if not hasattr(backtest_result, 'portfolio_values') or backtest_result.portfolio_values is None:
            # Create basic equity curve from total return
            total_return = float(backtest_result.metrics.total_return) if backtest_result.metrics else 0.0
            equity_curve = pd.Series([100000, 100000 * (1 + total_return)])
            returns = equity_curve.pct_change().dropna()
        else:
            returns = backtest_result.portfolio_values.pct_change().dropna()

        analysis = {
            'basic_metrics': self._calculate_basic_metrics(backtest_result, returns),
            'risk_metrics': self._calculate_risk_metrics(returns),
            'drawdown_analysis': self._analyze_drawdowns(returns),
            'rolling_metrics': self._calculate_rolling_metrics(returns),
        }

        if benchmark_returns is not None:
            analysis['relative_performance'] = self._analyze_relative_performance(
                returns, benchmark_returns
            )

        return analysis

    def _calculate_basic_metrics(
        self,
        backtest_result: BacktestResult,
        returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate basic performance metrics"""
        metrics = backtest_result.metrics
        if not metrics:
            return {}

        return {
            'total_return': float(metrics.total_return),
            'annualized_return': self._annualize_return(float(metrics.total_return), len(returns)),
            'volatility': float(metrics.volatility),
            'sharpe_ratio': float(metrics.sharpe_ratio),
            'max_drawdown': float(metrics.max_drawdown),
            'win_rate': float(metrics.win_rate),
            'profit_factor': self._calculate_profit_factor(returns),
            'avg_win': returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0,
            'avg_loss': returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0,
        }

    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate advanced risk metrics"""
        downside_returns = returns[returns < 0]

        return {
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'calmar_ratio': self._calculate_calmar_ratio(returns),
            'max_consecutive_losses': self._max_consecutive_losses(returns),
            'var_95': np.percentile(returns, 5) if len(returns) > 0 else 0,
            'cvar_95': returns[returns <= np.percentile(returns, 5)].mean() if len(returns) > 0 else 0,
            'skewness': returns.skew() if len(returns) > 0 else 0,
            'kurtosis': returns.kurtosis() if len(returns) > 0 else 0,
            'downside_deviation': downside_returns.std() if len(downside_returns) > 0 else 0,
        }

    def _analyze_drawdowns(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze drawdown characteristics"""
        if len(returns) == 0:
            return {
                'max_drawdown': 0,
                'avg_drawdown': 0,
                'drawdown_duration_avg': 0,
                'drawdown_recovery_avg': 0,
                'num_drawdowns': 0
            }

        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max

        # Find drawdown periods
        in_drawdown = drawdowns < 0
        drawdown_starts = in_drawdown & ~in_drawdown.shift(1, fill_value=False)
        drawdown_ends = ~in_drawdown & in_drawdown.shift(1, fill_value=False)

        return {
            'max_drawdown': drawdowns.min(),
            'avg_drawdown': drawdowns[drawdowns < 0].mean() if len(drawdowns[drawdowns < 0]) > 0 else 0,
            'drawdown_duration_avg': float(drawdown_starts.sum()) if drawdown_starts.sum() > 0 else 0,
            'num_drawdowns': int(drawdown_starts.sum()),
            'time_in_drawdown': float(in_drawdown.mean()),
        }

    def _calculate_rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 30
    ) -> Dict[str, pd.Series]:
        """Calculate rolling performance metrics"""
        if len(returns) < window:
            return {
                'rolling_sharpe': pd.Series(dtype=float),
                'rolling_volatility': pd.Series(dtype=float),
                'rolling_return': pd.Series(dtype=float)
            }

        rolling_return = returns.rolling(window).mean() * 252  # Annualized
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)  # Annualized
        rolling_sharpe = rolling_return / rolling_vol

        return {
            'rolling_sharpe': rolling_sharpe,
            'rolling_volatility': rolling_vol,
            'rolling_return': rolling_return,
        }

    def _analyze_relative_performance(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        """Analyze performance relative to benchmark"""
        # Align series
        common_index = strategy_returns.index.intersection(benchmark_returns.index)
        strat_aligned = strategy_returns.loc[common_index]
        bench_aligned = benchmark_returns.loc[common_index]

        excess_returns = strat_aligned - bench_aligned

        return {
            'alpha': excess_returns.mean() * 252,  # Annualized
            'beta': self._calculate_beta(strat_aligned, bench_aligned),
            'information_ratio': excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0,
            'tracking_error': excess_returns.std() * np.sqrt(252),
            'correlation': strat_aligned.corr(bench_aligned),
        }

    def _calculate_sortino_ratio(self, returns: pd.Series, target_return: float = 0) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - target_return
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        return excess_returns.mean() / downside_returns.std() * np.sqrt(252)

    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio"""
        if len(returns) == 0:
            return 0.0

        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_dd = abs(drawdowns.min())

        if max_dd == 0:
            return 0.0

        annual_return = returns.mean() * 252
        return annual_return / max_dd

    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor"""
        wins = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())

        return wins / losses if losses != 0 else float('inf')

    def _max_consecutive_losses(self, returns: pd.Series) -> int:
        """Calculate maximum consecutive losses"""
        if len(returns) == 0:
            return 0

        losses = returns < 0
        groups = (losses != losses.shift()).cumsum()
        consecutive_losses = losses.groupby(groups).sum()

        return int(consecutive_losses.max()) if len(consecutive_losses) > 0 else 0

    def _calculate_beta(self, strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta relative to benchmark"""
        if len(strategy_returns) < 2 or len(benchmark_returns) < 2:
            return 0.0

        covariance = strategy_returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()

        return covariance / benchmark_variance if benchmark_variance != 0 else 0.0

    def _annualize_return(self, total_return: float, periods: int) -> float:
        """Annualize a total return"""
        if periods <= 0:
            return 0.0

        # Assuming daily data
        years = periods / 252
        return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    def generate_performance_report(
        self,
        backtest_result: BacktestResult,
        save_path: Optional[str] = None
    ) -> str:
        """Generate comprehensive performance report"""
        analysis = self.analyze_comprehensive(backtest_result)

        report = []
        report.append("📊 QFrame Advanced Performance Report")
        report.append("=" * 50)

        # Basic metrics
        basic = analysis['basic_metrics']
        report.append(f"\n🎯 Basic Performance:")
        report.append(f"  Total Return: {basic['total_return']:.2%}")
        report.append(f"  Annualized Return: {basic['annualized_return']:.2%}")
        report.append(f"  Volatility: {basic['volatility']:.2%}")
        report.append(f"  Sharpe Ratio: {basic['sharpe_ratio']:.3f}")
        report.append(f"  Max Drawdown: {basic['max_drawdown']:.2%}")
        report.append(f"  Win Rate: {basic['win_rate']:.1%}")

        # Risk metrics
        risk = analysis['risk_metrics']
        report.append(f"\n⚠️ Risk Metrics:")
        report.append(f"  Sortino Ratio: {risk['sortino_ratio']:.3f}")
        report.append(f"  Calmar Ratio: {risk['calmar_ratio']:.3f}")
        report.append(f"  VaR (95%): {risk['var_95']:.2%}")
        report.append(f"  CVaR (95%): {risk['cvar_95']:.2%}")
        report.append(f"  Skewness: {risk['skewness']:.3f}")
        report.append(f"  Kurtosis: {risk['kurtosis']:.3f}")

        # Drawdown analysis
        dd = analysis['drawdown_analysis']
        report.append(f"\n📉 Drawdown Analysis:")
        report.append(f"  Max Drawdown: {dd['max_drawdown']:.2%}")
        report.append(f"  Avg Drawdown: {dd['avg_drawdown']:.2%}")
        report.append(f"  Number of Drawdowns: {dd['num_drawdowns']}")
        report.append(f"  Time in Drawdown: {dd['time_in_drawdown']:.1%}")

        report_text = "\n".join(report)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)

        return report_text

    def create_performance_dashboard(
        self,
        backtest_result: BacktestResult
    ) -> Dict[str, Any]:
        """Create interactive performance dashboard (if Plotly available)"""
        if not PLOTLY_AVAILABLE:
            return {"error": "Plotly not available for interactive dashboard"}

        analysis = self.analyze_comprehensive(backtest_result)

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Equity Curve', 'Rolling Sharpe', 'Drawdown', 'Returns Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # This would create interactive plots but requires actual data
        # For now, return the analysis structure
        return {
            "analysis": analysis,
            "charts_available": PLOTLY_AVAILABLE,
            "dashboard_type": "plotly" if PLOTLY_AVAILABLE else "basic"
        }