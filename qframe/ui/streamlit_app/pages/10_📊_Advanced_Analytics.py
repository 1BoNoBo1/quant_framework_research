"""
Page d'analytics avancé pour QFrame
===================================

Interface pour analyses approfondies, visualisations interactives
et génération d'insights à partir des données de trading.
"""

import streamlit as st
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Ajouter le chemin du projet
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from qframe.ui.streamlit_app.components.scientific_reports import (
    ScientificReportComponents,
    generate_sample_returns,
    generate_sample_features
)

# Configuration de la page
st.set_page_config(
    page_title="Analytics Avancé - QFrame",
    page_icon="📊",
    layout="wide"
)

def generate_advanced_analytics_data():
    """Génère des données d'analytics avancées"""

    # Données de base
    returns = generate_sample_returns(n_periods=500, annual_return=0.15, annual_vol=0.20)
    features = generate_sample_features(returns, n_features=15)

    # Données de trading simulées
    n_trades = 200
    trade_data = {
        'trade_id': range(1, n_trades + 1),
        'timestamp': pd.date_range(start=returns.index[0], end=returns.index[-1], periods=n_trades),
        'symbol': np.random.choice(['BTC/USD', 'ETH/USD', 'ADA/USD', 'SOL/USD'], n_trades),
        'side': np.random.choice(['BUY', 'SELL'], n_trades),
        'quantity': np.random.uniform(0.1, 2.0, n_trades),
        'price': np.random.uniform(30000, 70000, n_trades),
        'pnl': np.random.normal(50, 200, n_trades),
        'strategy': np.random.choice(['AdaptiveMeanReversion', 'DMN_LSTM', 'RL_Alpha'], n_trades)
    }
    trades_df = pd.DataFrame(trade_data)

    # Métriques de performance par stratégie
    strategy_metrics = {
        'AdaptiveMeanReversion': {
            'total_return': 23.5,
            'sharpe_ratio': 1.85,
            'max_drawdown': -8.2,
            'win_rate': 62.5,
            'avg_trade_duration': 4.2,
            'profit_factor': 1.45
        },
        'DMN_LSTM': {
            'total_return': 31.2,
            'sharpe_ratio': 2.12,
            'max_drawdown': -12.1,
            'win_rate': 58.3,
            'avg_trade_duration': 6.8,
            'profit_factor': 1.72
        },
        'RL_Alpha': {
            'total_return': 18.7,
            'sharpe_ratio': 1.56,
            'max_drawdown': -6.4,
            'win_rate': 65.1,
            'avg_trade_duration': 3.5,
            'profit_factor': 1.38
        }
    }

    return returns, features, trades_df, strategy_metrics

def render_pca_analysis(features: pd.DataFrame, returns: pd.Series):
    """Analyse en Composantes Principales des features"""

    st.header("🔍 Analyse PCA des Features")

    # Standardisation des données
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # PCA
    pca = PCA()
    pca_result = pca.fit_transform(features_scaled)

    col1, col2 = st.columns([1, 1])

    with col1:
        # Graphique de variance expliquée
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=list(range(1, len(explained_variance) + 1)),
            y=explained_variance * 100,
            name='Variance Expliquée',
            marker_color='lightblue'
        ))

        fig.add_trace(go.Scatter(
            x=list(range(1, len(cumulative_variance) + 1)),
            y=cumulative_variance * 100,
            mode='lines+markers',
            name='Variance Cumulative',
            yaxis='y2',
            line=dict(color='red', width=3)
        ))

        fig.update_layout(
            title="Variance Expliquée par Composante PCA",
            xaxis_title="Composante",
            yaxis_title="Variance Expliquée (%)",
            yaxis2=dict(title="Variance Cumulative (%)", overlaying='y', side='right'),
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Métriques PCA
        st.subheader("📊 Métriques PCA")

        n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

        st.metric("Composantes pour 90%", f"{n_components_90}/{len(features.columns)}")
        st.metric("Composantes pour 95%", f"{n_components_95}/{len(features.columns)}")
        st.metric("Variance PC1", f"{explained_variance[0]*100:.1f}%")
        st.metric("Variance PC2", f"{explained_variance[1]*100:.1f}%")

        # Top features contributoires
        st.subheader("🔝 Top Contributions PC1")
        pc1_contributions = abs(pca.components_[0])
        top_features_pc1 = sorted(zip(features.columns, pc1_contributions),
                                 key=lambda x: x[1], reverse=True)[:5]

        for feature, contribution in top_features_pc1:
            st.metric(feature, f"{contribution:.3f}")

    # Graphique 2D PCA avec corrélation aux retours
    st.subheader("🎯 Projection PCA 2D")

    # Calcul des corrélations avec les retours
    correlations = features.corrwith(returns).abs()

    fig = px.scatter(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        color=returns,
        title="Projection PCA avec Retours",
        labels={'x': f'PC1 ({explained_variance[0]*100:.1f}%)',
                'y': f'PC2 ({explained_variance[1]*100:.1f}%)',
                'color': 'Retours'},
        color_continuous_scale='RdYlBu_r'
    )

    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def render_strategy_comparison(strategy_metrics: dict, trades_df: pd.DataFrame):
    """Comparaison détaillée des stratégies"""

    st.header("⚔️ Comparaison des Stratégies")

    # Tableau de comparaison
    comparison_df = pd.DataFrame(strategy_metrics).T

    st.subheader("📊 Tableau Comparatif")

    # Style du tableau avec highlighting
    styled_df = comparison_df.style.background_gradient(cmap='RdYlGn', axis=0)
    st.dataframe(styled_df, use_container_width=True)

    # Graphiques de comparaison
    col1, col2 = st.columns(2)

    with col1:
        # Graphique radar des performances
        strategies = list(strategy_metrics.keys())
        metrics = ['total_return', 'sharpe_ratio', 'win_rate', 'profit_factor']
        metric_labels = ['Retour Total', 'Sharpe Ratio', 'Taux Gain', 'Facteur Profit']

        fig = go.Figure()

        for strategy in strategies:
            values = [strategy_metrics[strategy][metric] for metric in metrics]
            # Normalisation pour le graphique radar
            normalized_values = []
            for i, value in enumerate(values):
                if metrics[i] == 'total_return':
                    normalized_values.append(value / 50 * 100)  # Normalise sur 50%
                elif metrics[i] == 'sharpe_ratio':
                    normalized_values.append(value / 3 * 100)  # Normalise sur 3
                elif metrics[i] == 'win_rate':
                    normalized_values.append(value)  # Déjà en %
                elif metrics[i] == 'profit_factor':
                    normalized_values.append(value / 2 * 100)  # Normalise sur 2

            fig.add_trace(go.Scatterpolar(
                r=normalized_values,
                theta=metric_labels,
                fill='toself',
                name=strategy
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Comparaison Performance (Radar)",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Distribution des PnL par stratégie
        fig = go.Figure()

        for strategy in strategies:
            strategy_trades = trades_df[trades_df['strategy'] == strategy]

            fig.add_trace(go.Box(
                y=strategy_trades['pnl'],
                name=strategy,
                boxpoints='outliers'
            ))

        fig.update_layout(
            title="Distribution PnL par Stratégie",
            yaxis_title="PnL ($)",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    # Analyse des trades par stratégie
    st.subheader("📈 Analyse Temporelle des Trades")

    # PnL cumulatif par stratégie
    trades_df['cumulative_pnl'] = trades_df.groupby('strategy')['pnl'].cumsum()

    fig = px.line(
        trades_df,
        x='timestamp',
        y='cumulative_pnl',
        color='strategy',
        title="PnL Cumulatif par Stratégie",
        labels={'cumulative_pnl': 'PnL Cumulatif ($)', 'timestamp': 'Date'}
    )

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def render_correlation_heatmap(features: pd.DataFrame, returns: pd.Series):
    """Heatmap de corrélation avancée"""

    st.header("🔥 Matrice de Corrélation Avancée")

    # Calcul de la matrice de corrélation
    all_data = features.copy()
    all_data['returns'] = returns
    corr_matrix = all_data.corr()

    # Masque pour la moitié supérieure
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    corr_matrix_masked = corr_matrix.mask(mask)

    # Heatmap interactive
    fig = px.imshow(
        corr_matrix_masked,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        title="Matrice de Corrélation (Triangle Inférieur)"
    )

    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Analyse des corrélations fortes
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔗 Corrélations Fortes (|r| > 0.7)")

        # Extraire les corrélations fortes
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_correlations.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Corrélation': corr_value
                    })

        if strong_correlations:
            strong_df = pd.DataFrame(strong_correlations)
            st.dataframe(strong_df.sort_values('Corrélation', key=abs, ascending=False))
        else:
            st.info("Aucune corrélation forte détectée")

    with col2:
        st.subheader("🎯 Corrélations avec Retours")

        returns_corr = corr_matrix['returns'].drop('returns').sort_values(key=abs, ascending=False)

        fig = px.bar(
            x=returns_corr.values,
            y=returns_corr.index,
            orientation='h',
            title="Corrélation Features-Retours",
            labels={'x': 'Corrélation', 'y': 'Features'},
            color=returns_corr.values,
            color_continuous_scale='RdBu_r'
        )

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def render_regime_analysis(returns: pd.Series):
    """Analyse des régimes de marché"""

    st.header("📊 Analyse des Régimes de Marché")

    # Calcul de la volatilité roulante
    rolling_vol = returns.rolling(30).std() * np.sqrt(252) * 100

    # Classification des régimes basée sur la volatilité
    vol_quantiles = rolling_vol.quantile([0.33, 0.67])
    regimes = pd.cut(rolling_vol,
                    bins=[-np.inf, vol_quantiles.iloc[0], vol_quantiles.iloc[1], np.inf],
                    labels=['Faible Vol', 'Vol Normale', 'Forte Vol'])

    col1, col2 = st.columns([2, 1])

    with col1:
        # Graphique des régimes
        fig = go.Figure()

        # Retours
        fig.add_trace(go.Scatter(
            x=returns.index,
            y=returns.cumsum() * 100,
            mode='lines',
            name='Retours Cumulés',
            line=dict(color='blue', width=2)
        ))

        # Zones de régimes
        colors = {'Faible Vol': 'green', 'Vol Normale': 'yellow', 'Forte Vol': 'red'}

        for regime in regimes.cat.categories:
            regime_mask = regimes == regime
            if regime_mask.any():
                regime_periods = returns.index[regime_mask]

                for period in regime_periods:
                    fig.add_vrect(
                        x0=period - timedelta(days=1),
                        x1=period + timedelta(days=1),
                        fillcolor=colors[regime],
                        opacity=0.2,
                        layer="below",
                        line_width=0,
                    )

        fig.update_layout(
            title="Régimes de Volatilité et Performance",
            xaxis_title="Date",
            yaxis_title="Retours Cumulés (%)",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Statistiques par Régime")

        for regime in regimes.cat.categories:
            regime_returns = returns[regimes == regime]

            if len(regime_returns) > 0:
                st.markdown(f"**{regime}**")
                st.metric("Nombre de périodes", len(regime_returns))
                st.metric("Retour moyen", f"{regime_returns.mean()*100:.3f}%")
                st.metric("Volatilité", f"{regime_returns.std()*100:.3f}%")
                st.metric("Sharpe", f"{regime_returns.mean()/regime_returns.std():.3f}")
                st.markdown("---")

def render_statistical_tests(returns: pd.Series):
    """Tests statistiques sur les retours"""

    st.header("🧪 Tests Statistiques")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Tests de Normalité")

        # Test de Shapiro-Wilk
        shapiro_stat, shapiro_p = stats.shapiro(returns)
        st.metric("Shapiro-Wilk Statistic", f"{shapiro_stat:.4f}")
        st.metric("P-value", f"{shapiro_p:.6f}")

        normalite = "✅ Normale" if shapiro_p > 0.05 else "❌ Non-normale"
        st.info(f"Distribution: {normalite}")

        # Test de Jarque-Bera
        jb_stat, jb_p = stats.jarque_bera(returns)
        st.metric("Jarque-Bera Statistic", f"{jb_stat:.4f}")
        st.metric("P-value", f"{jb_p:.6f}")

    with col2:
        st.subheader("🔄 Tests d'Autocorrélation")

        # Test de Ljung-Box
        from scipy.stats import diagnostic

        # Autocorrélation à différents lags
        lags = [1, 5, 10, 20]
        autocorrs = [returns.autocorr(lag) for lag in lags]

        for lag, autocorr in zip(lags, autocorrs):
            st.metric(f"Autocorr Lag {lag}", f"{autocorr:.4f}")

        # Test de stationnarité (Dickey-Fuller)
        from statsmodels.tsa.stattools import adfuller

        adf_result = adfuller(returns.dropna())
        st.metric("ADF Statistic", f"{adf_result[0]:.4f}")
        st.metric("P-value", f"{adf_result[1]:.6f}")

        stationnarite = "✅ Stationnaire" if adf_result[1] < 0.05 else "❌ Non-stationnaire"
        st.info(f"Série: {stationnarite}")

def main():
    """Point d'entrée principal"""

    st.title("📊 Analytics Avancé QFrame")
    st.markdown("Interface d'analyse approfondie pour insights quantitatifs et visualisations interactives.")

    # Sidebar pour sélection d'analyse
    with st.sidebar:
        st.header("📋 Types d'Analyse")

        analysis_type = st.selectbox(
            "Choisir l'analyse",
            [
                "Analyse PCA",
                "Comparaison Stratégies",
                "Corrélations Avancées",
                "Régimes de Marché",
                "Tests Statistiques"
            ]
        )

        # Paramètres d'analyse
        st.header("⚙️ Paramètres")

        n_periods = st.slider("Nombre de périodes", 100, 1000, 500)
        annual_return = st.slider("Retour annuel (%)", 5, 30, 15) / 100
        annual_vol = st.slider("Volatilité annuelle (%)", 10, 40, 20) / 100

    # Génération des données
    try:
        returns, features, trades_df, strategy_metrics = generate_advanced_analytics_data()

        # Affichage selon le type d'analyse
        if analysis_type == "Analyse PCA":
            render_pca_analysis(features, returns)

        elif analysis_type == "Comparaison Stratégies":
            render_strategy_comparison(strategy_metrics, trades_df)

        elif analysis_type == "Corrélations Avancées":
            render_correlation_heatmap(features, returns)

        elif analysis_type == "Régimes de Marché":
            render_regime_analysis(returns)

        elif analysis_type == "Tests Statistiques":
            render_statistical_tests(returns)

    except Exception as e:
        st.error(f"Erreur lors du chargement des analyses: {e}")
        st.code(str(e))

if __name__ == "__main__":
    main()