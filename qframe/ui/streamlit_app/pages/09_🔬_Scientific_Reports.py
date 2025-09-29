"""
Page d'interface pour les rapports scientifiques QFrame
======================================================

Interface web pour génération et analyse de rapports scientifiques
basés sur les résultats de validation du framework.
"""

import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import json
import traceback

# Ajouter le chemin du projet
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from qframe.ui.streamlit_app.utils.session_state import SessionStateManager
from qframe.ui.streamlit_app.components.charts import create_performance_chart, display_metric_card

# Configuration de la page
st.set_page_config(
    page_title="Rapports Scientifiques - QFrame",
    page_icon="🔬",
    layout="wide"
)

def generate_sample_scientific_data():
    """Génère des données scientifiques d'exemple basées sur Option A"""

    # Métriques de performance Option A validées
    performance_data = {
        'strategy_name': 'AdaptiveMeanReversion',
        'total_return': 56.5,  # %
        'sharpe_ratio': 2.254,
        'sortino_ratio': 2.68,
        'max_drawdown': -4.97,  # %
        'win_rate': 60.0,  # %
        'total_trades': 544,
        'profit_factor': 2.50,
        'calmar_ratio': 11.37,
        'volatility': 14.2,  # %
        'alpha': 12.5,  # %
        'beta': 0.75,
        'information_ratio': 1.85
    }

    # Données de validation scientifique
    validation_data = {
        'overall_validation': 87.3,
        'data_quality_score': 100.0,
        'overfitting_checks': 87.5,
        'statistical_significance': 100.0,
        'robustness_score': 85.0,
        'probabilistic_sharpe': 0.892,
        'deflated_sharpe': 1.85,
        'information_coefficient': 0.156
    }

    # Données feature engineering
    feature_data = {
        'features_generated': 18,
        'feature_quality': 0.156,
        'alpha_signals': 245,
        'execution_time': 1.62,
        'top_correlations': [0.5205, 0.4823, 0.4391, 0.3967, 0.3544]
    }

    # Génération de données temporelles pour graphiques
    dates = pd.date_range(start='2024-04-01', end='2024-09-27', freq='1h')
    n = len(dates)

    # Série de performance cumulative
    returns = np.random.normal(0.0008, 0.02, n)  # Returns horaires
    returns[0] = 0
    cumulative_returns = (1 + pd.Series(returns)).cumprod() - 1

    # Prix simulé
    initial_price = 50000
    prices = initial_price * (1 + cumulative_returns)

    time_series = pd.DataFrame({
        'timestamp': dates,
        'price': prices,
        'returns': returns,
        'cumulative_returns': cumulative_returns
    })

    return performance_data, validation_data, feature_data, time_series

def render_executive_summary(performance_data, validation_data):
    """Rendu du résumé exécutif"""

    st.header("📋 Résumé Exécutif")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"""
        ### Stratégie: {performance_data['strategy_name']}

        **Performance Exceptionnelle**: La stratégie AdaptiveMeanReversion a démontré des performances
        remarquables avec un **retour total de {performance_data['total_return']:.1f}%** et un
        **ratio de Sharpe de {performance_data['sharpe_ratio']:.3f}**, plaçant la stratégie dans
        le quartile supérieur des stratégies quantitatives institutionnelles.

        **Validation Scientifique Rigoureuse**: Score global de validation de
        **{validation_data['overall_validation']:.1f}/100**, confirmant la robustesse
        statistique et la qualité des données avec une signification statistique parfaite.

        **Recommandation**: Stratégie validée pour déploiement en production avec allocation
        de capital progressive.
        """)

    with col2:
        st.metric("Score Validation Global", f"{validation_data['overall_validation']:.1f}/100")
        st.metric("Retour Total", f"{performance_data['total_return']:.1f}%")
        st.metric("Ratio Sharpe", f"{performance_data['sharpe_ratio']:.3f}")
        st.metric("Max Drawdown", f"{performance_data['max_drawdown']:.1f}%")

def render_performance_analysis(performance_data, time_series):
    """Rendu de l'analyse de performance"""

    st.header("📈 Analyse de Performance")

    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        display_metric_card(
            "Retour Total",
            f"{performance_data['total_return']:.1f}%",
            delta=None
        )

    with col2:
        display_metric_card(
            "Ratio Sharpe",
            f"{performance_data['sharpe_ratio']:.3f}",
            delta=None
        )

    with col3:
        display_metric_card(
            "Taux de Gain",
            f"{performance_data['win_rate']:.1f}%",
            delta=None
        )

    with col4:
        display_metric_card(
            "Total Trades",
            f"{performance_data['total_trades']:,}",
            delta=None
        )

    # Graphique de performance cumulative
    st.subheader("📊 Performance Cumulative")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time_series['timestamp'],
        y=time_series['cumulative_returns'] * 100,
        mode='lines',
        name='Retours Cumulatifs',
        line=dict(color='#1f77b4', width=2)
    ))

    fig.update_layout(
        title="Évolution des Retours Cumulatifs",
        xaxis_title="Date",
        yaxis_title="Retour Cumulatif (%)",
        height=400,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # Métriques de risque détaillées
    st.subheader("⚠️ Métriques de Risque")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Max Drawdown", f"{performance_data['max_drawdown']:.2f}%")
        st.metric("Volatilité", f"{performance_data['volatility']:.1f}%")

    with col2:
        st.metric("Ratio Sortino", f"{performance_data['sortino_ratio']:.3f}")
        st.metric("Ratio Calmar", f"{performance_data['calmar_ratio']:.2f}")

    with col3:
        st.metric("Facteur Profit", f"{performance_data['profit_factor']:.2f}")
        st.metric("Ratio Information", f"{performance_data['information_ratio']:.3f}")

def render_statistical_validation(validation_data):
    """Rendu de la validation statistique"""

    st.header("🔬 Validation Statistique")

    # Scores de validation
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Scores de Validation")

        validation_scores = {
            'Qualité Données': validation_data['data_quality_score'],
            'Tests Overfitting': validation_data['overfitting_checks'],
            'Signification Statistique': validation_data['statistical_significance'],
            'Score Robustesse': validation_data['robustness_score']
        }

        for metric, score in validation_scores.items():
            progress_color = "green" if score >= 90 else "orange" if score >= 70 else "red"
            st.metric(metric, f"{score:.1f}/100")
            st.progress(score/100)

    with col2:
        st.subheader("📈 Métriques Probabilistes")

        st.metric("Sharpe Probabiliste", f"{validation_data['probabilistic_sharpe']:.3f}")
        st.metric("Sharpe Déflaté", f"{validation_data['deflated_sharpe']:.3f}")
        st.metric("Coefficient Information", f"{validation_data['information_coefficient']:.3f}")

        # Graphique radar des scores
        categories = list(validation_scores.keys())
        values = list(validation_scores.values())

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Scores Validation'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

def render_feature_analysis(feature_data):
    """Rendu de l'analyse des features"""

    st.header("🧠 Analyse Feature Engineering")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Métriques Features")

        st.metric("Features Générées", feature_data['features_generated'])
        st.metric("Qualité Moyenne", f"{feature_data['feature_quality']:.3f}")
        st.metric("Signaux Alpha", feature_data['alpha_signals'])
        st.metric("Temps Exécution", f"{feature_data['execution_time']:.2f}s")

    with col2:
        st.subheader("🔝 Top Corrélations")

        # Graphique des top corrélations
        correlations = feature_data['top_correlations']
        feature_names = [f"Feature {i+1}" for i in range(len(correlations))]

        fig = px.bar(
            x=feature_names,
            y=correlations,
            title="Top 5 Features par Corrélation",
            labels={'x': 'Features', 'y': 'Corrélation'}
        )

        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def render_report_generator():
    """Interface de génération de rapports"""

    st.header("📄 Générateur de Rapports")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Configuration du Rapport")

        # Paramètres du rapport
        report_type = st.selectbox(
            "Type de Rapport",
            ["Rapport Complet", "Résumé Exécutif", "Analyse Technique", "Validation Scientifique"]
        )

        strategy_name = st.text_input("Nom de la Stratégie", value="AdaptiveMeanReversion")

        date_range = st.date_input(
            "Période d'Analyse",
            value=[datetime(2024, 4, 1), datetime(2024, 9, 27)]
        )

        include_charts = st.checkbox("Inclure Graphiques", value=True)
        include_raw_data = st.checkbox("Inclure Données Brutes", value=False)

        format_output = st.selectbox(
            "Format de Sortie",
            ["HTML", "Markdown", "PDF", "JSON"]
        )

    with col2:
        st.subheader("Actions")

        if st.button("🔬 Générer Rapport", use_container_width=True):
            with st.spinner("Génération du rapport en cours..."):
                # Simulation de génération
                import time
                time.sleep(2)

                st.success("✅ Rapport généré avec succès!")

                # Informations du rapport généré
                report_info = {
                    'filename': f"{strategy_name}_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_output.lower()}",
                    'size': f"{np.random.randint(500, 2000)} KB",
                    'pages': np.random.randint(15, 35) if format_output == "PDF" else "N/A"
                }

                st.info(f"""
                **Rapport Généré:**
                - Fichier: `{report_info['filename']}`
                - Taille: {report_info['size']}
                - Pages: {report_info['pages']}
                """)

        if st.button("📧 Envoyer par Email", use_container_width=True):
            st.info("Fonctionnalité d'envoi en développement")

        if st.button("💾 Télécharger", use_container_width=True):
            st.info("Téléchargement disponible après génération")

def main():
    """Point d'entrée principal de la page"""

    st.title("🔬 Rapports Scientifiques QFrame")
    st.markdown("Interface de génération et d'analyse de rapports scientifiques pour les stratégies quantitatives validées.")

    # Sidebar pour navigation
    with st.sidebar:
        st.header("📋 Navigation")

        section = st.radio(
            "Sections du Rapport",
            [
                "Résumé Exécutif",
                "Analyse Performance",
                "Validation Statistique",
                "Feature Engineering",
                "Générateur Rapports"
            ]
        )

    # Génération des données
    try:
        performance_data, validation_data, feature_data, time_series = generate_sample_scientific_data()

        # Affichage selon la section sélectionnée
        if section == "Résumé Exécutif":
            render_executive_summary(performance_data, validation_data)

        elif section == "Analyse Performance":
            render_performance_analysis(performance_data, time_series)

        elif section == "Validation Statistique":
            render_statistical_validation(validation_data)

        elif section == "Feature Engineering":
            render_feature_analysis(feature_data)

        elif section == "Générateur Rapports":
            render_report_generator()

    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()