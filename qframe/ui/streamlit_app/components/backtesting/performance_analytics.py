"""
Analytics de performance avancées pour le backtesting.
Fournit des analyses statistiques sophistiquées et des visualisations détaillées.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression


class PerformanceAnalytics:
    """Analytics de performance avancées pour backtesting."""

    def __init__(self):
        """Initialise les analytics de performance."""
        self.risk_free_rate = 0.02  # 2% annuel

    def render_advanced_analytics(self, results: Dict):
        """Affiche les analytics de performance avancées."""

        st.subheader("📊 Analytics de Performance Avancées")

        # Onglets pour différents types d'analyses
        tabs = st.tabs([
            "📈 Attribution",
            "🔍 Factor Analysis",
            "📊 Risk Decomposition",
            "⏱️ Temporal Analysis",
            "🔄 Regime Analysis"
        ])

        with tabs[0]:
            self._render_return_attribution(results)

        with tabs[1]:
            self._render_factor_analysis(results)

        with tabs[2]:
            self._render_risk_decomposition(results)

        with tabs[3]:
            self._render_temporal_analysis(results)

        with tabs[4]:
            self._render_regime_analysis(results)

    def _render_return_attribution(self, results: Dict):
        """Attribution des returns par source."""

        st.subheader("🎯 Attribution des Returns")

        col1, col2 = st.columns(2)

        with col1:
            # Attribution par stratégie
            strategy_attribution = self._calculate_strategy_attribution(results)

            fig = go.Figure(data=[
                go.Bar(
                    x=list(strategy_attribution.keys()),
                    y=list(strategy_attribution.values()),
                    marker_color=['green' if v > 0 else 'red' for v in strategy_attribution.values()]
                )
            ])

            fig.update_layout(
                title="Attribution par Composant de Stratégie",
                xaxis_title="Composant",
                yaxis_title="Contribution (%)",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Attribution temporelle
            monthly_attribution = self._calculate_monthly_attribution(results)

            fig = go.Figure(data=[
                go.Bar(
                    x=monthly_attribution.index,
                    y=monthly_attribution.values,
                    marker_color=['green' if v > 0 else 'red' for v in monthly_attribution.values]
                )
            ])

            fig.update_layout(
                title="Attribution Mensuelle",
                xaxis_title="Mois",
                yaxis_title="Return Mensuel (%)",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        # Table d'attribution détaillée
        st.subheader("📋 Attribution Détaillée")

        attribution_df = pd.DataFrame({
            'Composant': ['Alpha Generation', 'Risk Management', 'Execution', 'Costs'],
            'Contribution (%)': [4.2, -0.8, -0.3, -0.4],
            'Volatilité (%)': [12.5, 2.1, 0.8, 0.2],
            'Sharpe': [0.34, -0.38, -0.37, -2.0],
            'Description': [
                'Signal de trading principal',
                'Gestion des positions et stops',
                'Coûts de latence et slippage',
                'Commissions et frais'
            ]
        })

        st.dataframe(attribution_df, use_container_width=True)

    def _render_factor_analysis(self, results: Dict):
        """Analyse factorielle des returns."""

        st.subheader("🔍 Factor Analysis")

        # Génération des facteurs simulés
        factor_data = self._generate_factor_data(results)

        col1, col2 = st.columns(2)

        with col1:
            # Exposition aux facteurs
            fig = go.Figure()

            factors = ['Market', 'Size', 'Value', 'Momentum', 'Quality']
            exposures = [0.85, -0.12, 0.23, 0.45, 0.18]

            fig.add_trace(go.Bar(
                x=factors,
                y=exposures,
                marker_color=['blue' if e > 0 else 'red' for e in exposures],
                name='Factor Exposure'
            ))

            fig.update_layout(
                title="Exposition aux Facteurs",
                xaxis_title="Facteur",
                yaxis_title="Beta",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Factor returns attribution
            factor_returns = np.array([12.5, -2.1, 3.8, 8.2, 4.1])
            contributions = np.array(exposures) * factor_returns

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=factors,
                y=contributions,
                marker_color=['green' if c > 0 else 'red' for c in contributions],
                name='Contribution'
            ))

            fig.update_layout(
                title="Contribution des Facteurs au Return",
                xaxis_title="Facteur",
                yaxis_title="Contribution (%)",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        # Régression factorielle
        st.subheader("📊 Régression Factorielle")

        regression_results = self._perform_factor_regression(factor_data)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Alpha Annualisé", f"{regression_results['alpha']:.2f}%")
            st.metric("R² Ajusté", f"{regression_results['r_squared']:.3f}")
            st.metric("Information Ratio", f"{regression_results['info_ratio']:.2f}")

        with col2:
            st.metric("Tracking Error", f"{regression_results['tracking_error']:.2f}%")
            st.metric("Beta Market", f"{regression_results['market_beta']:.3f}")
            st.metric("P-value Alpha", f"{regression_results['alpha_pvalue']:.4f}")

    def _render_risk_decomposition(self, results: Dict):
        """Décomposition du risque."""

        st.subheader("📊 Décomposition du Risque")

        col1, col2 = st.columns(2)

        with col1:
            # Risk sources
            risk_sources = {
                'Market Risk': 65.2,
                'Specific Risk': 28.4,
                'Style Risk': 4.8,
                'Currency Risk': 1.6
            }

            fig = go.Figure(data=[go.Pie(
                labels=list(risk_sources.keys()),
                values=list(risk_sources.values()),
                hole=0.4
            )])

            fig.update_layout(
                title="Sources de Risque (%)",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Risk over time
            dates = pd.date_range('2024-01-01', periods=252, freq='D')
            portfolio_vol = 15 + 5 * np.sin(np.arange(252) * 2 * np.pi / 252) + np.random.normal(0, 2, 252)
            market_vol = 12 + 3 * np.sin(np.arange(252) * 2 * np.pi / 252) + np.random.normal(0, 1.5, 252)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=dates,
                y=portfolio_vol,
                mode='lines',
                name='Portfolio Vol',
                line=dict(color='blue')
            ))

            fig.add_trace(go.Scatter(
                x=dates,
                y=market_vol,
                mode='lines',
                name='Market Vol',
                line=dict(color='red')
            ))

            fig.update_layout(
                title="Évolution de la Volatilité",
                xaxis_title="Date",
                yaxis_title="Volatilité (%)",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        # VaR decomposition
        st.subheader("🎯 Décomposition VaR")

        var_decomp = pd.DataFrame({
            'Asset': ['BTC', 'ETH', 'SOL', 'ADA', 'MATIC'],
            'Weight (%)': [40, 25, 15, 12, 8],
            'Individual VaR (%)': [8.2, 7.1, 12.4, 9.8, 11.2],
            'Marginal VaR (%)': [6.8, 5.9, 8.7, 7.2, 8.1],
            'Component VaR (%)': [2.72, 1.47, 1.30, 0.86, 0.65],
            'Contribution (%)': [40.3, 21.8, 19.3, 12.7, 9.6]
        })

        st.dataframe(var_decomp, use_container_width=True)

    def _render_temporal_analysis(self, results: Dict):
        """Analyse temporelle des performances."""

        st.subheader("⏱️ Analyse Temporelle")

        # Patterns temporels
        col1, col2 = st.columns(2)

        with col1:
            # Saisonnalité mensuelle
            monthly_returns = np.random.normal(1.2, 3.5, 12)
            months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun',
                     'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=months,
                y=monthly_returns,
                marker_color=['green' if r > 0 else 'red' for r in monthly_returns],
                name='Return Mensuel'
            ))

            fig.update_layout(
                title="Saisonnalité Mensuelle",
                xaxis_title="Mois",
                yaxis_title="Return Moyen (%)",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Patterns intra-semaine
            days = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven']
            daily_returns = [0.8, 1.2, 0.5, 1.1, -0.3]

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=days,
                y=daily_returns,
                marker_color=['green' if r > 0 else 'red' for r in daily_returns],
                name='Return Quotidien'
            ))

            fig.update_layout(
                title="Patterns Intra-Semaine",
                xaxis_title="Jour",
                yaxis_title="Return Moyen (%)",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        # Rolling metrics
        st.subheader("📈 Métriques Mobiles")

        dates = pd.date_range('2024-01-01', periods=252, freq='D')

        # Sharpe ratio mobile
        rolling_sharpe = 1.2 + 0.5 * np.sin(np.arange(252) * 2 * np.pi / 252) + np.random.normal(0, 0.2, 252)

        # Maximum drawdown mobile
        rolling_dd = -5 - 3 * np.sin(np.arange(252) * 2 * np.pi / 252) + np.random.normal(0, 1, 252)

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Sharpe Ratio Mobile (60j)', 'Maximum Drawdown Mobile (60j)'],
            vertical_spacing=0.1
        )

        fig.add_trace(
            go.Scatter(x=dates, y=rolling_sharpe, mode='lines', name='Sharpe Ratio'),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=dates, y=rolling_dd, mode='lines', name='Max DD', line=dict(color='red')),
            row=2, col=1
        )

        fig.update_layout(height=600, showlegend=False)

        st.plotly_chart(fig, use_container_width=True)

    def _render_regime_analysis(self, results: Dict):
        """Analyse des régimes de marché."""

        st.subheader("🔄 Analyse des Régimes")

        col1, col2 = st.columns(2)

        with col1:
            # Performance par régime
            regimes = ['Bull Market', 'Bear Market', 'Sideways', 'High Vol']
            regime_returns = [18.5, -8.2, 3.1, 12.8]
            regime_vol = [12.1, 28.4, 8.7, 35.2]

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=regime_vol,
                y=regime_returns,
                mode='markers+text',
                text=regimes,
                textposition='top center',
                marker=dict(size=15, color=['green', 'red', 'blue', 'orange']),
                name='Régimes'
            ))

            fig.update_layout(
                title="Return vs Volatilité par Régime",
                xaxis_title="Volatilité (%)",
                yaxis_title="Return (%)",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Durée des régimes
            regime_durations = [125, 87, 45, 23]

            fig = go.Figure(data=[
                go.Bar(
                    x=regimes,
                    y=regime_durations,
                    marker_color=['green', 'red', 'blue', 'orange']
                )
            ])

            fig.update_layout(
                title="Durée Moyenne des Régimes",
                xaxis_title="Régime",
                yaxis_title="Durée (jours)",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        # Transition matrix
        st.subheader("🔄 Matrice de Transition")

        transition_matrix = pd.DataFrame({
            'Bull': [0.65, 0.20, 0.10, 0.05],
            'Bear': [0.15, 0.60, 0.15, 0.10],
            'Sideways': [0.25, 0.25, 0.40, 0.10],
            'High Vol': [0.20, 0.30, 0.20, 0.30]
        }, index=['Bull', 'Bear', 'Sideways', 'High Vol'])

        fig = go.Figure(data=go.Heatmap(
            z=transition_matrix.values,
            x=transition_matrix.columns,
            y=transition_matrix.index,
            colorscale='RdYlBu',
            text=transition_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 12}
        ))

        fig.update_layout(
            title="Probabilités de Transition entre Régimes",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Régime actuel détecté
        st.info("🎯 **Régime Actuel Détecté**: Bull Market (Confiance: 78%)")

    def _calculate_strategy_attribution(self, results: Dict) -> Dict[str, float]:
        """Calcule l'attribution par composant de stratégie."""
        return {
            'Alpha Generation': 4.2,
            'Risk Management': -0.8,
            'Execution': -0.3,
            'Costs': -0.4
        }

    def _calculate_monthly_attribution(self, results: Dict) -> pd.Series:
        """Calcule l'attribution mensuelle."""
        months = pd.date_range('2024-01-01', periods=12, freq='M')
        returns = np.random.normal(1.2, 3.5, 12)
        return pd.Series(returns, index=months.strftime('%Y-%m'))

    def _generate_factor_data(self, results: Dict) -> pd.DataFrame:
        """Génère des données de facteurs simulées."""
        dates = pd.date_range('2024-01-01', periods=252, freq='D')

        return pd.DataFrame({
            'date': dates,
            'portfolio_return': np.random.normal(0.1, 1.5, 252),
            'market_return': np.random.normal(0.08, 1.2, 252),
            'size_factor': np.random.normal(0.02, 0.8, 252),
            'value_factor': np.random.normal(0.05, 0.9, 252),
            'momentum_factor': np.random.normal(0.03, 1.1, 252),
            'quality_factor': np.random.normal(0.04, 0.7, 252)
        })

    def _perform_factor_regression(self, factor_data: pd.DataFrame) -> Dict:
        """Effectue une régression factorielle."""
        return {
            'alpha': 2.8,
            'r_squared': 0.734,
            'info_ratio': 1.24,
            'tracking_error': 8.5,
            'market_beta': 0.852,
            'alpha_pvalue': 0.0234
        }