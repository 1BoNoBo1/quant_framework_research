"""
Composants pour les rapports scientifiques QFrame
================================================

Composants réutilisables pour la génération et l'affichage
de rapports scientifiques de qualité institutionnelle.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import base64
import io

class ScientificReportComponents:
    """Composants de base pour rapports scientifiques"""

    @staticmethod
    def render_performance_summary_table(metrics: Dict[str, Any]) -> None:
        """Tableau de résumé des métriques de performance"""

        st.subheader("📊 Résumé des Métriques")

        # Organisation des métriques en catégories
        performance_metrics = {
            'Retour Total (%)': metrics.get('total_return', 0),
            'Retour Annualisé (%)': metrics.get('annualized_return', 0),
            'Volatilité (%)': metrics.get('volatility', 0),
            'Ratio Sharpe': metrics.get('sharpe_ratio', 0),
        }

        risk_metrics = {
            'Max Drawdown (%)': metrics.get('max_drawdown', 0),
            'VaR 95% (%)': metrics.get('var_95', 0),
            'CVaR 95% (%)': metrics.get('cvar_95', 0),
            'Ratio Sortino': metrics.get('sortino_ratio', 0),
        }

        trading_metrics = {
            'Total Trades': metrics.get('total_trades', 0),
            'Taux de Gain (%)': metrics.get('win_rate', 0),
            'Facteur Profit': metrics.get('profit_factor', 0),
            'Trade Moyen (%)': metrics.get('avg_trade_return', 0),
        }

        # Affichage en colonnes
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🎯 Performance**")
            for metric, value in performance_metrics.items():
                if isinstance(value, float):
                    st.metric(metric, f"{value:.3f}")
                else:
                    st.metric(metric, str(value))

        with col2:
            st.markdown("**⚠️ Risque**")
            for metric, value in risk_metrics.items():
                if isinstance(value, float):
                    st.metric(metric, f"{value:.3f}")
                else:
                    st.metric(metric, str(value))

        with col3:
            st.markdown("**💼 Trading**")
            for metric, value in trading_metrics.items():
                if isinstance(value, float):
                    st.metric(metric, f"{value:.3f}")
                else:
                    st.metric(metric, str(value))

    @staticmethod
    def render_performance_chart(returns_data: pd.DataFrame, benchmark_data: Optional[pd.DataFrame] = None) -> None:
        """Graphique de performance avec benchmark optionnel"""

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Performance Cumulative', 'Drawdown'),
            row_width=[0.7, 0.3]
        )

        # Performance cumulative
        fig.add_trace(
            go.Scatter(
                x=returns_data.index,
                y=(1 + returns_data['returns']).cumprod() * 100 - 100,
                mode='lines',
                name='Stratégie',
                line=dict(color='#1f77b4', width=2)
            ),
            row=1, col=1
        )

        if benchmark_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=benchmark_data.index,
                    y=(1 + benchmark_data['returns']).cumprod() * 100 - 100,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='#ff7f0e', width=2, dash='dash')
                ),
                row=1, col=1
            )

        # Calcul du drawdown
        cumulative = (1 + returns_data['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = ((cumulative - running_max) / running_max) * 100

        fig.add_trace(
            go.Scatter(
                x=returns_data.index,
                y=drawdown,
                mode='lines',
                name='Drawdown',
                fill='tonexty',
                line=dict(color='red', width=1),
                fillcolor='rgba(255,0,0,0.3)'
            ),
            row=2, col=1
        )

        fig.update_layout(
            height=600,
            title_text="Analyse de Performance et Drawdown",
            showlegend=True
        )

        fig.update_yaxes(title_text="Retour Cumulatif (%)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_returns_distribution(returns: pd.Series) -> None:
        """Distribution des retours avec statistiques"""

        col1, col2 = st.columns([2, 1])

        with col1:
            # Histogramme des retours
            fig = px.histogram(
                x=returns * 100,
                nbins=50,
                title="Distribution des Retours",
                labels={'x': 'Retours (%)', 'y': 'Fréquence'}
            )

            # Ajout de la courbe normale théorique
            mean_return = returns.mean() * 100
            std_return = returns.std() * 100
            x_normal = np.linspace(returns.min() * 100, returns.max() * 100, 100)
            y_normal = (1/(std_return * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_normal - mean_return) / std_return) ** 2)
            y_normal = y_normal * len(returns) * (returns.max() - returns.min()) / 50  # Normalisation

            fig.add_trace(
                go.Scatter(
                    x=x_normal,
                    y=y_normal,
                    mode='lines',
                    name='Distribution Normale',
                    line=dict(color='red', dash='dash')
                )
            )

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Statistiques descriptives
            st.markdown("**📊 Statistiques**")

            stats = {
                'Moyenne (%)': returns.mean() * 100,
                'Médiane (%)': returns.median() * 100,
                'Écart-type (%)': returns.std() * 100,
                'Asymétrie': returns.skew(),
                'Kurtosis': returns.kurtosis(),
                'Min (%)': returns.min() * 100,
                'Max (%)': returns.max() * 100,
            }

            for stat, value in stats.items():
                st.metric(stat, f"{value:.3f}")

    @staticmethod
    def render_rolling_metrics(returns_data: pd.DataFrame, window: int = 252) -> None:
        """Métriques roulantes (Sharpe, volatilité, etc.)"""

        st.subheader(f"📈 Métriques Roulantes ({window} jours)")

        # Calcul des métriques roulantes
        rolling_returns = returns_data['returns'].rolling(window)
        rolling_sharpe = rolling_returns.mean() / rolling_returns.std() * np.sqrt(252)
        rolling_vol = rolling_returns.std() * np.sqrt(252) * 100

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Ratio Sharpe Roulant', 'Volatilité Roulante (%)'),
        )

        # Sharpe roulant
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe,
                mode='lines',
                name='Sharpe Roulant',
                line=dict(color='green', width=2)
            ),
            row=1, col=1
        )

        # Volatilité roulante
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol,
                mode='lines',
                name='Volatilité Roulante',
                line=dict(color='orange', width=2)
            ),
            row=2, col=1
        )

        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_risk_analysis(returns: pd.Series, confidence_levels: List[float] = [0.95, 0.99]) -> None:
        """Analyse de risque détaillée (VaR, CVaR)"""

        st.subheader("⚠️ Analyse de Risque")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📊 Value at Risk (VaR)**")

            for confidence in confidence_levels:
                var_value = np.percentile(returns, (1 - confidence) * 100) * 100
                st.metric(f"VaR {confidence*100:.0f}%", f"{var_value:.2f}%")

        with col2:
            st.markdown("**📉 Conditional VaR (CVaR)**")

            for confidence in confidence_levels:
                var_threshold = np.percentile(returns, (1 - confidence) * 100)
                cvar_value = returns[returns <= var_threshold].mean() * 100
                st.metric(f"CVaR {confidence*100:.0f}%", f"{cvar_value:.2f}%")

        # Graphique de queue de distribution
        st.markdown("**📈 Analyse des Queues de Distribution**")

        # Focus sur les pertes extrêmes (5% pires retours)
        worst_returns = returns.sort_values().head(int(len(returns) * 0.05)) * 100

        fig = px.histogram(
            x=worst_returns,
            nbins=20,
            title="Distribution des 5% Pires Retours",
            labels={'x': 'Retours (%)', 'y': 'Fréquence'}
        )

        # Lignes VaR
        for confidence in confidence_levels:
            var_value = np.percentile(returns, (1 - confidence) * 100) * 100
            fig.add_vline(
                x=var_value,
                line_dash="dash",
                annotation_text=f"VaR {confidence*100:.0f}%",
                line_color="red" if confidence == 0.95 else "darkred"
            )

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_correlation_analysis(features_data: pd.DataFrame, target_returns: pd.Series) -> None:
        """Analyse de corrélation entre features et retours"""

        st.subheader("🔗 Analyse de Corrélation")

        # Calcul de la matrice de corrélation
        correlation_data = features_data.copy()
        correlation_data['returns'] = target_returns

        corr_matrix = correlation_data.corr()

        # Heatmap de corrélation
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Matrice de Corrélation Features-Retours",
            color_continuous_scale="RdBu_r"
        )

        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Top corrélations avec les retours
        returns_corr = corr_matrix['returns'].drop('returns').abs().sort_values(ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🔝 Top Corrélations Positives**")
            positive_corr = corr_matrix['returns'].drop('returns').sort_values(ascending=False).head(5)
            for feature, corr in positive_corr.items():
                st.metric(feature, f"{corr:.3f}")

        with col2:
            st.markdown("**🔻 Top Corrélations Négatives**")
            negative_corr = corr_matrix['returns'].drop('returns').sort_values(ascending=True).head(5)
            for feature, corr in negative_corr.items():
                st.metric(feature, f"{corr:.3f}")

    @staticmethod
    def render_report_download_section(report_data: Dict[str, Any]) -> None:
        """Section de téléchargement de rapport"""

        st.subheader("📄 Téléchargement du Rapport")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📋 Générer PDF", use_container_width=True):
                # Simulation de génération PDF
                pdf_content = f"Rapport scientifique généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                b64_pdf = base64.b64encode(pdf_content.encode()).decode()

                href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="scientific_report.pdf">Télécharger PDF</a>'
                st.markdown(href, unsafe_allow_html=True)

        with col2:
            if st.button("📊 Exporter Excel", use_container_width=True):
                # Génération fichier Excel avec métriques
                buffer = io.BytesIO()

                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    # Feuille des métriques
                    metrics_df = pd.DataFrame.from_dict(report_data, orient='index', columns=['Valeur'])
                    metrics_df.to_excel(writer, sheet_name='Métriques')

                buffer.seek(0)
                b64_excel = base64.b64encode(buffer.read()).decode()

                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" download="metrics_report.xlsx">Télécharger Excel</a>'
                st.markdown(href, unsafe_allow_html=True)

        with col3:
            if st.button("🔗 Partager Lien", use_container_width=True):
                # Génération d'un lien de partage (simulation)
                share_link = f"https://qframe.app/reports/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.code(share_link)
                st.success("Lien de partage généré!")

    @staticmethod
    def render_validation_badges(validation_scores: Dict[str, float]) -> None:
        """Badges de validation scientifique"""

        st.subheader("🏆 Badges de Validation")

        # Configuration des badges
        badge_config = {
            'Qualité Données': {'threshold': 95, 'icon': '💎', 'color': 'blue'},
            'Tests Overfitting': {'threshold': 85, 'icon': '🛡️', 'color': 'green'},
            'Signification Statistique': {'threshold': 95, 'icon': '📊', 'color': 'purple'},
            'Robustesse': {'threshold': 80, 'icon': '💪', 'color': 'orange'}
        }

        cols = st.columns(len(badge_config))

        for i, (metric, config) in enumerate(badge_config.items()):
            with cols[i]:
                score = validation_scores.get(metric.lower().replace(' ', '_'), 0)
                passed = score >= config['threshold']

                badge_color = config['color'] if passed else 'gray'
                status = "VALIDÉ" if passed else "ÉCHEC"

                st.markdown(f"""
                <div style="
                    border: 2px solid {badge_color};
                    border-radius: 10px;
                    padding: 10px;
                    text-align: center;
                    margin: 5px;
                    background-color: rgba(0,0,0,0.1);
                ">
                    <h3>{config['icon']}</h3>
                    <h4>{metric}</h4>
                    <p><strong>{score:.1f}/100</strong></p>
                    <p style="color: {badge_color}"><strong>{status}</strong></p>
                </div>
                """, unsafe_allow_html=True)

# Fonctions utilitaires pour la génération de données de test
def generate_sample_returns(n_periods: int = 1000, annual_return: float = 0.12, annual_vol: float = 0.15) -> pd.Series:
    """Génère une série de retours d'exemple"""

    daily_return = annual_return / 252
    daily_vol = annual_vol / np.sqrt(252)

    returns = np.random.normal(daily_return, daily_vol, n_periods)
    dates = pd.date_range(start='2022-01-01', periods=n_periods, freq='D')

    return pd.Series(returns, index=dates, name='returns')

def generate_sample_features(returns: pd.Series, n_features: int = 10) -> pd.DataFrame:
    """Génère des features d'exemple corrélées aux retours"""

    np.random.seed(42)  # Pour la reproductibilité

    features_data = {}

    for i in range(n_features):
        # Génération de features avec différents niveaux de corrélation
        if i < 3:  # Features fortement corrélées
            correlation = np.random.uniform(0.3, 0.7)
            noise_level = 0.3
        elif i < 6:  # Features modérément corrélées
            correlation = np.random.uniform(0.1, 0.3)
            noise_level = 0.5
        else:  # Features faiblement corrélées
            correlation = np.random.uniform(-0.1, 0.1)
            noise_level = 0.8

        # Génération de la feature
        signal = returns * correlation
        noise = np.random.normal(0, noise_level, len(returns))
        feature = signal + noise

        features_data[f'feature_{i+1}'] = feature

    return pd.DataFrame(features_data, index=returns.index)