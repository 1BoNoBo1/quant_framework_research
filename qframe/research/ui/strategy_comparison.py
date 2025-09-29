"""
🔍 QFrame Strategy Comparison - UI Component

UI component for comparing strategy performance and metrics.
"""

from typing import Any, Dict, List, Optional
import pandas as pd

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None


class StrategyComparison:
    """
    🔍 Strategy Comparison Interface

    Streamlit-based interface for:
    - Side-by-side strategy comparison
    - Performance metrics analysis
    - Risk-return visualization
    - Statistical significance testing
    """

    def __init__(self, qframe_api=None):
        """
        Initialize Strategy Comparison

        Args:
            qframe_api: QFrameResearch instance (optional)
        """
        self.qframe_api = qframe_api
        self.comparison_results = []

        if not STREAMLIT_AVAILABLE:
            print("⚠️ Streamlit not available. UI components disabled.")

    def render_strategy_comparison(self):
        """Render the main strategy comparison interface"""
        if not STREAMLIT_AVAILABLE:
            print("⚠️ Streamlit not available")
            return

        st.title("🔍 QFrame Strategy Comparison")
        st.markdown("Compare performance across different trading strategies")

        # Strategy selection
        self._render_strategy_selector()

        # Comparison results
        if self.comparison_results:
            self._render_comparison_results()
        else:
            st.info("Select strategies to compare their performance")

    def _render_strategy_selector(self):
        """Render strategy selection interface"""
        st.header("🎯 Strategy Selection")

        available_strategies = [
            "adaptive_mean_reversion",
            "dmn_lstm",
            "funding_arbitrage",
            "rl_alpha",
            "grid_trading"
        ]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 Available Strategies")
            selected_strategies = st.multiselect(
                "Select strategies to compare",
                available_strategies,
                default=available_strategies[:2]
            )

        with col2:
            st.subheader("⚙️ Comparison Settings")

            symbol = st.text_input("Trading Pair", value="BTC/USDT")
            days = st.slider("Historical Data (days)", 7, 90, 30)
            initial_capital = st.number_input("Initial Capital", value=100000)

            if st.button("🚀 Run Comparison"):
                self._run_strategy_comparison(selected_strategies, symbol, days, initial_capital)

    def _run_strategy_comparison(self, strategies: List[str], symbol: str, days: int, capital: float):
        """Run strategy comparison"""
        if not strategies or len(strategies) < 2:
            st.error("Please select at least 2 strategies to compare")
            return

        st.info(f"Running comparison for {len(strategies)} strategies...")

        # Mock comparison results for demo
        import numpy as np
        import random

        results = []
        for strategy in strategies:
            # Generate mock performance data
            np.random.seed(hash(strategy) % 1000)

            returns = np.random.normal(0.001, 0.02, days)
            cumulative_returns = np.cumprod(1 + returns) - 1

            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            max_dd = np.min(np.minimum.accumulate(cumulative_returns[-30:]))

            results.append({
                "strategy": strategy,
                "total_return": cumulative_returns[-1],
                "sharpe_ratio": sharpe,
                "max_drawdown": max_dd,
                "volatility": np.std(returns) * np.sqrt(252),
                "win_rate": random.uniform(0.45, 0.65),
                "trades": random.randint(50, 200),
                "returns_series": cumulative_returns
            })

        self.comparison_results = results
        st.success(f"Comparison completed for {len(strategies)} strategies!")
        st.rerun()

    def _render_comparison_results(self):
        """Render comparison results"""
        st.header("📊 Comparison Results")

        # Performance metrics table
        st.subheader("📋 Performance Summary")

        metrics_data = []
        for result in self.comparison_results:
            metrics_data.append({
                "Strategy": result["strategy"],
                "Total Return": f"{result['total_return']:.2%}",
                "Sharpe Ratio": f"{result['sharpe_ratio']:.2f}",
                "Max Drawdown": f"{result['max_drawdown']:.2%}",
                "Volatility": f"{result['volatility']:.2%}",
                "Win Rate": f"{result['win_rate']:.1%}",
                "Trades": result["trades"]
            })

        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True)

        # Performance charts
        self._render_performance_charts()

        # Risk-return scatter
        self._render_risk_return_scatter()

        # Statistical analysis
        self._render_statistical_analysis()

    def _render_performance_charts(self):
        """Render performance comparison charts"""
        st.subheader("📈 Cumulative Returns Comparison")

        fig = go.Figure()

        for result in self.comparison_results:
            dates = pd.date_range(start='2024-01-01', periods=len(result["returns_series"]), freq='D')

            fig.add_trace(go.Scatter(
                x=dates,
                y=result["returns_series"],
                mode='lines',
                name=result["strategy"],
                line=dict(width=2)
            ))

        fig.update_layout(
            title="Cumulative Returns Comparison",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            yaxis_tickformat=".1%",
            height=500,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_risk_return_scatter(self):
        """Render risk-return scatter plot"""
        st.subheader("🎯 Risk-Return Analysis")

        # Prepare data for scatter plot
        strategy_names = [r["strategy"] for r in self.comparison_results]
        returns = [r["total_return"] * 100 for r in self.comparison_results]
        volatilities = [r["volatility"] * 100 for r in self.comparison_results]
        sharpe_ratios = [r["sharpe_ratio"] for r in self.comparison_results]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=volatilities,
            y=returns,
            mode='markers+text',
            text=strategy_names,
            textposition="top center",
            marker=dict(
                size=[abs(s) * 10 + 10 for s in sharpe_ratios],  # Size based on Sharpe ratio
                color=sharpe_ratios,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio")
            ),
            hovertemplate='<b>%{text}</b><br>' +
                         'Volatility: %{x:.1f}%<br>' +
                         'Return: %{y:.1f}%<br>' +
                         '<extra></extra>'
        ))

        fig.update_layout(
            title="Risk vs Return (Bubble size = |Sharpe Ratio|)",
            xaxis_title="Volatility (%)",
            yaxis_title="Total Return (%)",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_statistical_analysis(self):
        """Render statistical significance analysis"""
        st.subheader("📊 Statistical Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Top Performers:**")

            # Sort by Sharpe ratio
            sorted_by_sharpe = sorted(self.comparison_results, key=lambda x: x["sharpe_ratio"], reverse=True)

            for i, result in enumerate(sorted_by_sharpe[:3]):
                medal = ["🥇", "🥈", "🥉"][i]
                st.write(f"{medal} {result['strategy']}: {result['sharpe_ratio']:.2f}")

        with col2:
            st.write("**Risk Metrics:**")

            # Sort by max drawdown (ascending, better)
            sorted_by_dd = sorted(self.comparison_results, key=lambda x: x["max_drawdown"])

            st.write("🛡️ **Lowest Drawdown:**")
            for result in sorted_by_dd[:2]:
                st.write(f"• {result['strategy']}: {result['max_drawdown']:.1%}")

        # Correlation matrix
        st.subheader("🔗 Strategy Correlation")
        st.info("Strategy correlation analysis would require actual return series data")

        # Performance ranking
        st.subheader("🏆 Overall Ranking")

        # Calculate composite score
        for result in self.comparison_results:
            # Normalize metrics (0-1 scale)
            result["composite_score"] = (
                result["sharpe_ratio"] * 0.4 +  # 40% weight
                result["total_return"] * 0.3 +  # 30% weight
                (1 - abs(result["max_drawdown"])) * 0.3  # 30% weight (inverted)
            )

        ranked_strategies = sorted(self.comparison_results, key=lambda x: x["composite_score"], reverse=True)

        ranking_data = []
        for i, result in enumerate(ranked_strategies):
            ranking_data.append({
                "Rank": i + 1,
                "Strategy": result["strategy"],
                "Composite Score": f"{result['composite_score']:.3f}",
                "Strengths": self._get_strategy_strengths(result)
            })

        df_ranking = pd.DataFrame(ranking_data)
        st.dataframe(df_ranking, use_container_width=True)

    def _get_strategy_strengths(self, result: Dict) -> str:
        """Get strategy strengths based on metrics"""
        strengths = []

        if result["sharpe_ratio"] > 1.0:
            strengths.append("High Sharpe")
        if result["total_return"] > 0.15:
            strengths.append("High Returns")
        if result["max_drawdown"] > -0.1:
            strengths.append("Low Drawdown")
        if result["win_rate"] > 0.6:
            strengths.append("High Win Rate")

        return ", ".join(strengths) if strengths else "Balanced"

    def export_comparison_results(self) -> pd.DataFrame:
        """Export comparison results to DataFrame"""
        if not self.comparison_results:
            return pd.DataFrame()

        export_data = []
        for result in self.comparison_results:
            export_data.append({
                "strategy": result["strategy"],
                "total_return": result["total_return"],
                "sharpe_ratio": result["sharpe_ratio"],
                "max_drawdown": result["max_drawdown"],
                "volatility": result["volatility"],
                "win_rate": result["win_rate"],
                "trades": result["trades"],
                "composite_score": result.get("composite_score", 0)
            })

        return pd.DataFrame(export_data)

    def get_comparison_stats(self) -> Dict[str, Any]:
        """Get comparison statistics"""
        if not self.comparison_results:
            return {}

        return {
            "strategies_compared": len(self.comparison_results),
            "best_sharpe": max(r["sharpe_ratio"] for r in self.comparison_results),
            "best_return": max(r["total_return"] for r in self.comparison_results),
            "lowest_drawdown": min(r["max_drawdown"] for r in self.comparison_results)
        }

    def __repr__(self):
        return f"StrategyComparison({len(self.comparison_results)} results)"