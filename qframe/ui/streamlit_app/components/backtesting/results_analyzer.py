"""
Results Analyzer - Composant pour analyser les résultats de backtests
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
from scipy import stats

class ResultsAnalyzer:
    """Analyseur de résultats de backtests avec métriques avancées."""

    def __init__(self):
        self.risk_free_rate = 0.03  # 3% par défaut

    def calculate_all_metrics(self, returns: np.ndarray, prices: np.ndarray = None,
                             benchmark_returns: np.ndarray = None) -> Dict:
        """Calcule toutes les métriques de performance."""
        if len(returns) == 0:
            return {}

        # Métriques de base
        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)

        # Ratios risk-adjusted
        excess_returns = returns - self.risk_free_rate / 252
        sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        # Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        downside_deviation = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0

        # Drawdown analysis
        if prices is not None:
            cumulative = prices
        else:
            cumulative = np.cumprod(1 + returns)

        rolling_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns)

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # VaR et CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else var_95

        # Trade statistics (simulées)
        win_rate = np.mean(returns > 0)
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0
        avg_loss = np.mean(negative_returns) if len(negative_returns) > 0 else 0
        profit_factor = abs(np.sum(positive_returns) / np.sum(negative_returns)) if np.sum(negative_returns) != 0 else np.inf

        # Statistical metrics
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)

        # Information ratio (vs benchmark)
        information_ratio = 0
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            active_returns = returns - benchmark_returns
            tracking_error = np.std(active_returns) * np.sqrt(252)
            information_ratio = np.mean(active_returns) * 252 / tracking_error if tracking_error > 0 else 0

        # Maximum consecutive wins/losses
        win_streak, loss_streak = self._calculate_streaks(returns)

        # Drawdown duration analysis
        dd_durations = self._calculate_drawdown_durations(drawdowns)

        return {
            # Returns
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'best_day': np.max(returns),
            'worst_day': np.min(returns),

            # Risk-adjusted metrics
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,

            # Drawdown metrics
            'max_drawdown': max_drawdown,
            'avg_drawdown': np.mean(drawdowns[drawdowns < 0]) if np.any(drawdowns < 0) else 0,
            'max_drawdown_duration': max(dd_durations) if dd_durations else 0,
            'avg_drawdown_duration': np.mean(dd_durations) if dd_durations else 0,

            # Risk metrics
            'var_95': var_95,
            'cvar_95': cvar_95,
            'downside_deviation': downside_deviation,

            # Trade metrics
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': abs(avg_win / avg_loss) if avg_loss != 0 else np.inf,

            # Statistical metrics
            'skewness': skewness,
            'kurtosis': kurtosis,
            'tail_ratio': abs(np.percentile(returns, 95) / np.percentile(returns, 5)) if np.percentile(returns, 5) != 0 else np.inf,

            # Streak analysis
            'max_win_streak': win_streak,
            'max_loss_streak': loss_streak,

            # Other
            'total_trades': len(returns),
            'trading_days': len(returns)
        }

    def _calculate_streaks(self, returns: np.ndarray) -> Tuple[int, int]:
        """Calcule les streaks de gains/pertes consécutifs."""
        if len(returns) == 0:
            return 0, 0

        wins = returns > 0
        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0

        for win in wins:
            if win:
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            else:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)

        return max_win_streak, max_loss_streak

    def _calculate_drawdown_durations(self, drawdowns: np.ndarray) -> List[int]:
        """Calcule les durées des drawdowns."""
        durations = []
        in_drawdown = False
        current_duration = 0

        for dd in drawdowns:
            if dd < 0:
                if not in_drawdown:
                    in_drawdown = True
                    current_duration = 1
                else:
                    current_duration += 1
            else:
                if in_drawdown:
                    durations.append(current_duration)
                    in_drawdown = False
                    current_duration = 0

        # Si on finit en drawdown
        if in_drawdown:
            durations.append(current_duration)

        return durations

    def render_performance_overview(self, metrics: Dict):
        """Rendu de l'aperçu de performance."""
        st.subheader("📈 Performance Overview")

        # Métriques principales
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            total_return = metrics.get('total_return', 0)
            st.metric(
                "Total Return",
                f"{total_return:.1%}",
                delta=f"{total_return:.1%}",
                delta_color="normal"
            )

        with col2:
            sharpe = metrics.get('sharpe_ratio', 0)
            st.metric(
                "Sharpe Ratio",
                f"{sharpe:.2f}",
                help="Risk-adjusted return metric"
            )

        with col3:
            max_dd = metrics.get('max_drawdown', 0)
            st.metric(
                "Max Drawdown",
                f"{max_dd:.1%}",
                delta=f"{max_dd:.1%}",
                delta_color="inverse"
            )

        with col4:
            win_rate = metrics.get('win_rate', 0)
            st.metric(
                "Win Rate",
                f"{win_rate:.1%}",
                help="Percentage of winning trades"
            )

        with col5:
            volatility = metrics.get('volatility', 0)
            st.metric(
                "Volatility",
                f"{volatility:.1%}",
                help="Annualized volatility"
            )

        # Classification de performance
        self._render_performance_classification(metrics)

    def _render_performance_classification(self, metrics: Dict):
        """Rendu de la classification de performance."""
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = metrics.get('max_drawdown', 0)
        total_return = metrics.get('total_return', 0)

        # Déterminer la classe de performance
        if sharpe > 2.0 and max_dd > -0.15 and total_return > 0.15:
            performance_class = "🏆 Excellent"
            class_color = "success"
        elif sharpe > 1.5 and max_dd > -0.25 and total_return > 0.10:
            performance_class = "🥈 Very Good"
            class_color = "success"
        elif sharpe > 1.0 and max_dd > -0.35 and total_return > 0.05:
            performance_class = "🥉 Good"
            class_color = "info"
        elif sharpe > 0.5 and max_dd > -0.50:
            performance_class = "⚠️ Acceptable"
            class_color = "warning"
        else:
            performance_class = "❌ Needs Improvement"
            class_color = "error"

        st.markdown(f"""
        **Overall Performance Rating:** {performance_class}
        """)

    def render_detailed_metrics(self, metrics: Dict):
        """Rendu des métriques détaillées."""
        st.subheader("📋 Detailed Performance Metrics")

        col_returns, col_risk, col_trades = st.columns(3)

        with col_returns:
            st.markdown("### 📈 Returns Metrics")
            returns_metrics = [
                ("Total Return", f"{metrics.get('total_return', 0):.2%}"),
                ("Annualized Return", f"{metrics.get('annualized_return', 0):.2%}"),
                ("Best Day", f"{metrics.get('best_day', 0):.2%}"),
                ("Worst Day", f"{metrics.get('worst_day', 0):.2%}"),
                ("Volatility", f"{metrics.get('volatility', 0):.2%}")
            ]

            for metric, value in returns_metrics:
                st.markdown(f"**{metric}:** {value}")

        with col_risk:
            st.markdown("### ⚠️ Risk Metrics")
            risk_metrics = [
                ("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}"),
                ("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.3f}"),
                ("Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.3f}"),
                ("VaR (95%)", f"{metrics.get('var_95', 0):.2%}"),
                ("CVaR (95%)", f"{metrics.get('cvar_95', 0):.2%}")
            ]

            for metric, value in risk_metrics:
                st.markdown(f"**{metric}:** {value}")

        with col_trades:
            st.markdown("### 💼 Trading Metrics")
            trade_metrics = [
                ("Win Rate", f"{metrics.get('win_rate', 0):.1%}"),
                ("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}"),
                ("Avg Win", f"{metrics.get('avg_win', 0):.2%}"),
                ("Avg Loss", f"{metrics.get('avg_loss', 0):.2%}"),
                ("Win/Loss Ratio", f"{metrics.get('win_loss_ratio', 0):.2f}")
            ]

            for metric, value in trade_metrics:
                st.markdown(f"**{metric}:** {value}")

    def render_drawdown_analysis(self, drawdown_series: np.ndarray, dates: List[datetime] = None):
        """Rendu de l'analyse des drawdowns."""
        st.subheader("📉 Drawdown Analysis")

        if dates is None:
            dates = pd.date_range(start='2023-01-01', periods=len(drawdown_series), freq='D')

        # Graphique de drawdown
        fig_dd = go.Figure()

        fig_dd.add_trace(go.Scatter(
            x=dates,
            y=drawdown_series * 100,
            mode='lines',
            name='Drawdown',
            line=dict(color='#ff6b6b', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.3)'
        ))

        # Ligne de drawdown maximum
        max_dd = np.min(drawdown_series)
        fig_dd.add_hline(
            y=max_dd * 100,
            line_dash="dash",
            line_color="#ff6b6b",
            annotation_text=f"Max DD: {max_dd:.1%}"
        )

        fig_dd.update_layout(
            title="Drawdown Over Time",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template='plotly_dark',
            height=400
        )

        st.plotly_chart(fig_dd, use_container_width=True)

        # Statistiques de drawdown
        col_dd1, col_dd2, col_dd3, col_dd4 = st.columns(4)

        durations = self._calculate_drawdown_durations(drawdown_series)

        with col_dd1:
            st.metric("Max Drawdown", f"{max_dd:.1%}")

        with col_dd2:
            avg_dd = np.mean(drawdown_series[drawdown_series < 0]) if np.any(drawdown_series < 0) else 0
            st.metric("Avg Drawdown", f"{avg_dd:.1%}")

        with col_dd3:
            max_duration = max(durations) if durations else 0
            st.metric("Max DD Duration", f"{max_duration} days")

        with col_dd4:
            avg_duration = np.mean(durations) if durations else 0
            st.metric("Avg DD Duration", f"{avg_duration:.0f} days")

    def render_returns_distribution(self, returns: np.ndarray):
        """Rendu de la distribution des returns."""
        st.subheader("📊 Returns Distribution")

        col_dist1, col_dist2 = st.columns(2)

        with col_dist1:
            # Histogramme
            fig_hist = go.Figure()

            fig_hist.add_trace(go.Histogram(
                x=returns * 100,
                nbinsx=50,
                name='Daily Returns',
                marker_color='#00ff88',
                opacity=0.7
            ))

            # Lignes de percentiles
            percentiles = [5, 25, 50, 75, 95]
            colors = ['#ff6b6b', '#ffa500', '#00ff88', '#ffa500', '#ff6b6b']

            for p, color in zip(percentiles, colors):
                value = np.percentile(returns * 100, p)
                fig_hist.add_vline(
                    x=value,
                    line_dash="dash",
                    line_color=color,
                    annotation_text=f"P{p}: {value:.2f}%"
                )

            fig_hist.update_layout(
                title="Daily Returns Distribution",
                xaxis_title="Daily Return (%)",
                yaxis_title="Frequency",
                template='plotly_dark',
                height=400
            )

            st.plotly_chart(fig_hist, use_container_width=True)

        with col_dist2:
            # QQ plot pour normalité
            from scipy.stats import probplot

            theoretical_quantiles, sample_quantiles = probplot(returns, dist="norm")

            fig_qq = go.Figure()

            fig_qq.add_trace(go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode='markers',
                name='Returns',
                marker=dict(color='#00ff88', size=4)
            ))

            # Ligne de référence normale
            min_val, max_val = min(theoretical_quantiles), max(theoretical_quantiles)
            fig_qq.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Normal Distribution',
                line=dict(color='#ff6b6b', dash='dash')
            ))

            fig_qq.update_layout(
                title="Q-Q Plot vs Normal Distribution",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles",
                template='plotly_dark',
                height=400
            )

            st.plotly_chart(fig_qq, use_container_width=True)

        # Statistiques de distribution
        st.markdown("### Distribution Statistics")

        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

        with col_stat1:
            skewness = stats.skew(returns)
            st.metric("Skewness", f"{skewness:.3f}")

        with col_stat2:
            kurtosis = stats.kurtosis(returns)
            st.metric("Kurtosis", f"{kurtosis:.3f}")

        with col_stat3:
            jarque_bera_stat, jarque_bera_p = stats.jarque_bera(returns)
            st.metric("Jarque-Bera p-value", f"{jarque_bera_p:.4f}")

        with col_stat4:
            # Test de normalité Shapiro-Wilk (sur échantillon)
            sample_size = min(5000, len(returns))
            sample_returns = np.random.choice(returns, sample_size, replace=False)
            shapiro_stat, shapiro_p = stats.shapiro(sample_returns)
            st.metric("Shapiro p-value", f"{shapiro_p:.4f}")

    def render_rolling_metrics(self, returns: np.ndarray, window: int = 30,
                              dates: List[datetime] = None):
        """Rendu des métriques mobiles."""
        st.subheader("📈 Rolling Performance Metrics")

        if dates is None:
            dates = pd.date_range(start='2023-01-01', periods=len(returns), freq='D')

        # Calcul des métriques mobiles
        rolling_returns = pd.Series(returns).rolling(window=window)
        rolling_sharpe = rolling_returns.apply(
            lambda x: np.mean(x) / np.std(x) * np.sqrt(252) if np.std(x) > 0 else 0
        )
        rolling_volatility = rolling_returns.std() * np.sqrt(252)
        rolling_win_rate = rolling_returns.apply(lambda x: np.mean(x > 0))

        # Graphique des métriques mobiles
        fig_rolling = go.Figure()

        # Sharpe ratio mobile
        fig_rolling.add_trace(go.Scatter(
            x=dates[window:],
            y=rolling_sharpe.dropna(),
            mode='lines',
            name=f'{window}d Rolling Sharpe',
            line=dict(color='#00ff88', width=2),
            yaxis='y'
        ))

        # Volatilité mobile
        fig_rolling.add_trace(go.Scatter(
            x=dates[window:],
            y=rolling_volatility.dropna() * 100,
            mode='lines',
            name=f'{window}d Rolling Volatility (%)',
            line=dict(color='#ff6b6b', width=2),
            yaxis='y2'
        ))

        # Win rate mobile
        fig_rolling.add_trace(go.Scatter(
            x=dates[window:],
            y=rolling_win_rate.dropna() * 100,
            mode='lines',
            name=f'{window}d Rolling Win Rate (%)',
            line=dict(color='#6b88ff', width=2),
            yaxis='y3'
        ))

        fig_rolling.update_layout(
            title=f"Rolling Performance Metrics ({window}-day window)",
            xaxis_title="Date",
            yaxis=dict(
                title="Sharpe Ratio",
                titlefont=dict(color="#00ff88"),
                tickfont=dict(color="#00ff88"),
                side="left"
            ),
            yaxis2=dict(
                title="Volatility (%)",
                titlefont=dict(color="#ff6b6b"),
                tickfont=dict(color="#ff6b6b"),
                anchor="x",
                overlaying="y",
                side="right"
            ),
            yaxis3=dict(
                title="Win Rate (%)",
                titlefont=dict(color="#6b88ff"),
                tickfont=dict(color="#6b88ff"),
                anchor="free",
                overlaying="y",
                side="right",
                position=0.05
            ),
            template='plotly_dark',
            height=500,
            hovermode='x unified'
        )

        st.plotly_chart(fig_rolling, use_container_width=True)

    def render_benchmark_comparison(self, strategy_metrics: Dict, benchmark_metrics: Dict,
                                   strategy_name: str = "Strategy", benchmark_name: str = "Benchmark"):
        """Rendu de la comparaison avec benchmark."""
        st.subheader("🏆 Benchmark Comparison")

        # Tableau de comparaison
        comparison_data = {
            'Metric': [
                'Total Return',
                'Annualized Return',
                'Volatility',
                'Sharpe Ratio',
                'Max Drawdown',
                'Win Rate',
                'Sortino Ratio',
                'Calmar Ratio'
            ],
            strategy_name: [
                f"{strategy_metrics.get('total_return', 0):.2%}",
                f"{strategy_metrics.get('annualized_return', 0):.2%}",
                f"{strategy_metrics.get('volatility', 0):.2%}",
                f"{strategy_metrics.get('sharpe_ratio', 0):.3f}",
                f"{strategy_metrics.get('max_drawdown', 0):.2%}",
                f"{strategy_metrics.get('win_rate', 0):.1%}",
                f"{strategy_metrics.get('sortino_ratio', 0):.3f}",
                f"{strategy_metrics.get('calmar_ratio', 0):.3f}"
            ],
            benchmark_name: [
                f"{benchmark_metrics.get('total_return', 0):.2%}",
                f"{benchmark_metrics.get('annualized_return', 0):.2%}",
                f"{benchmark_metrics.get('volatility', 0):.2%}",
                f"{benchmark_metrics.get('sharpe_ratio', 0):.3f}",
                f"{benchmark_metrics.get('max_drawdown', 0):.2%}",
                f"{benchmark_metrics.get('win_rate', 0):.1%}",
                f"{benchmark_metrics.get('sortino_ratio', 0):.3f}",
                f"{benchmark_metrics.get('calmar_ratio', 0):.3f}"
            ]
        }

        # Calcul des différences
        differences = []
        for i, metric in enumerate([
            'total_return', 'annualized_return', 'volatility', 'sharpe_ratio',
            'max_drawdown', 'win_rate', 'sortino_ratio', 'calmar_ratio'
        ]):
            strategy_val = strategy_metrics.get(metric, 0)
            benchmark_val = benchmark_metrics.get(metric, 0)

            if metric in ['volatility', 'max_drawdown']:
                # Pour ces métriques, moins c'est mieux
                diff = benchmark_val - strategy_val
            else:
                diff = strategy_val - benchmark_val

            if metric in ['total_return', 'annualized_return', 'volatility', 'max_drawdown', 'win_rate']:
                differences.append(f"{diff:+.2%}")
            else:
                differences.append(f"{diff:+.3f}")

        comparison_data['Difference'] = differences

        comparison_df = pd.DataFrame(comparison_data)

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        # Analyse de la performance relative
        self._render_relative_performance_analysis(strategy_metrics, benchmark_metrics)

    def _render_relative_performance_analysis(self, strategy_metrics: Dict, benchmark_metrics: Dict):
        """Analyse de la performance relative."""
        st.markdown("### Relative Performance Analysis")

        # Calcul de l'information ratio et autres métriques relatives
        strategy_return = strategy_metrics.get('total_return', 0)
        benchmark_return = benchmark_metrics.get('total_return', 0)
        strategy_sharpe = strategy_metrics.get('sharpe_ratio', 0)
        benchmark_sharpe = benchmark_metrics.get('sharpe_ratio', 0)

        alpha = strategy_return - benchmark_return
        sharpe_improvement = strategy_sharpe - benchmark_sharpe

        col_rel1, col_rel2, col_rel3 = st.columns(3)

        with col_rel1:
            st.metric("Alpha", f"{alpha:+.2%}", help="Excess return over benchmark")

        with col_rel2:
            st.metric("Sharpe Improvement", f"{sharpe_improvement:+.3f}", help="Sharpe ratio improvement")

        with col_rel3:
            # Risk-adjusted alpha
            strategy_vol = strategy_metrics.get('volatility', 0)
            risk_adjusted_alpha = alpha / strategy_vol if strategy_vol > 0 else 0
            st.metric("Risk-Adjusted Alpha", f"{risk_adjusted_alpha:.3f}")

        # Recommandation
        if alpha > 0.05 and sharpe_improvement > 0.5:
            st.success("🏆 **Strong Outperformance**: Strategy significantly outperforms benchmark")
        elif alpha > 0.02 and sharpe_improvement > 0.2:
            st.info("👍 **Moderate Outperformance**: Strategy shows improvement over benchmark")
        elif alpha > -0.02 and sharpe_improvement > -0.2:
            st.warning("⚖️ **Similar Performance**: Strategy performs similarly to benchmark")
        else:
            st.error("📉 **Underperformance**: Strategy underperforms benchmark")

    def export_analysis_report(self, metrics: Dict, strategy_name: str) -> str:
        """Exporte un rapport d'analyse complet."""
        report = f"""
# Performance Analysis Report

## Strategy: {strategy_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- **Total Return:** {metrics.get('total_return', 0):.2%}
- **Sharpe Ratio:** {metrics.get('sharpe_ratio', 0):.3f}
- **Maximum Drawdown:** {metrics.get('max_drawdown', 0):.2%}
- **Win Rate:** {metrics.get('win_rate', 0):.1%}

## Detailed Metrics
"""

        # Ajouter toutes les métriques
        for key, value in metrics.items():
            if isinstance(value, float):
                if key in ['total_return', 'annualized_return', 'volatility', 'max_drawdown', 'win_rate']:
                    report += f"- **{key.replace('_', ' ').title()}:** {value:.2%}\n"
                else:
                    report += f"- **{key.replace('_', ' ').title()}:** {value:.3f}\n"
            else:
                report += f"- **{key.replace('_', ' ').title()}:** {value}\n"

        return report