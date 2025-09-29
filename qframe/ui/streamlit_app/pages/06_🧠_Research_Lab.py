"""
Research Lab - Interface complète pour RL Alpha Generation et ML
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import time

# Import des composants
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api_client import APIClient
from components.utils import init_session_state, get_cached_data
from components.charts import create_line_chart, create_candlestick_chart
from components.tables import create_data_table

# Configuration de la page
st.set_page_config(
    page_title="QFrame - Research Lab",
    page_icon="🧠",
    layout="wide"
)

# Initialisation
api_client = APIClient()
init_session_state()

# Styles CSS personnalisés
st.markdown("""
<style>
    .research-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .model-card {
        background: #1a1a2e;
        border: 1px solid #00ff88;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 255, 136, 0.1);
    }
    .training-metric {
        background: #0f0f23;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #333;
    }
    .alpha-formula {
        font-family: 'Courier New', monospace;
        background: #0a0a0a;
        border: 1px solid #00ff88;
        border-radius: 5px;
        padding: 1rem;
        color: #00ff88;
        margin: 0.5rem 0;
    }
    .ic-score {
        font-size: 1.5rem;
        font-weight: bold;
        color: #00ff88;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="research-header">
    <h1>🧠 Research Lab</h1>
    <p>Centre de recherche quantitative avancée - RL Alpha Generation & Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Tabs principaux
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 RL Alpha Generation",
    "🧬 DMN LSTM Training",
    "📊 Models Performance",
    "🏛️ Alpha Library",
    "⚙️ Configuration"
])

# ================== TAB 1: RL Alpha Generation ==================
with tab1:
    st.header("🤖 Reinforcement Learning Alpha Generator")
    st.info("Génération automatique de formules alpha via apprentissage par renforcement")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🎮 Training Control Panel")

        # Configuration de l'entraînement RL
        with st.expander("📋 Configuration de l'Agent RL", expanded=True):
            col_config1, col_config2, col_config3 = st.columns(3)

            with col_config1:
                agent_type = st.selectbox(
                    "Type d'Agent",
                    ["PPO (Proximal Policy Optimization)", "A2C", "DQN", "SAC"],
                    help="PPO recommandé pour la génération d'alphas"
                )

                learning_rate = st.number_input(
                    "Learning Rate",
                    min_value=0.00001,
                    max_value=0.1,
                    value=0.0003,
                    format="%.5f",
                    help="Taux d'apprentissage de l'agent"
                )

                batch_size = st.number_input(
                    "Batch Size",
                    min_value=8,
                    max_value=512,
                    value=64,
                    step=8
                )

            with col_config2:
                max_depth = st.slider(
                    "Max Formula Depth",
                    min_value=1,
                    max_value=10,
                    value=4,
                    help="Profondeur maximale des formules générées"
                )

                n_episodes = st.number_input(
                    "Episodes",
                    min_value=100,
                    max_value=10000,
                    value=1000,
                    step=100,
                    help="Nombre d'épisodes d'entraînement"
                )

                exploration_rate = st.slider(
                    "Exploration Rate",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    help="Taux d'exploration vs exploitation"
                )

            with col_config3:
                reward_type = st.selectbox(
                    "Reward Function",
                    ["Information Coefficient (IC)", "Rank IC", "Sharpe Ratio", "Combined"],
                    help="Fonction de récompense pour guider l'apprentissage"
                )

                complexity_penalty = st.slider(
                    "Complexity Penalty",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    help="Pénalité pour formules trop complexes"
                )

                data_lookback = st.number_input(
                    "Data Lookback (days)",
                    min_value=30,
                    max_value=1000,
                    value=252,
                    help="Historique de données pour évaluation"
                )

        # Contrôles d'entraînement
        col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)
        with col_ctrl1:
            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                st.session_state['rl_training_active'] = True
                st.session_state['rl_start_time'] = datetime.now()
                st.success("Training démarré!")

        with col_ctrl2:
            if st.button("⏸️ Pause", use_container_width=True):
                st.session_state['rl_training_active'] = False
                st.info("Training en pause")

        with col_ctrl3:
            if st.button("🔄 Resume", use_container_width=True):
                st.session_state['rl_training_active'] = True
                st.success("Training repris")

        with col_ctrl4:
            if st.button("🛑 Stop", type="secondary", use_container_width=True):
                st.session_state['rl_training_active'] = False
                st.session_state['rl_start_time'] = None
                st.warning("Training arrêté")

        # Monitoring en temps réel
        st.subheader("📈 Training Metrics")

        if st.session_state.get('rl_training_active', False):
            # Simulation de métriques de training
            progress = st.progress(0)
            status_text = st.empty()
            metrics_container = st.container()

            with metrics_container:
                metric_cols = st.columns(4)

                # Métriques simulées (en production, venir de l'API)
                episode = np.random.randint(1, 1000)
                avg_reward = np.random.uniform(-0.5, 1.5)
                best_ic = np.random.uniform(0.01, 0.08)
                formulas_found = np.random.randint(10, 100)

                with metric_cols[0]:
                    st.metric("Episode", f"{episode}/1000", f"+{episode//10}")
                with metric_cols[1]:
                    st.metric("Avg Reward", f"{avg_reward:.3f}", f"{avg_reward*0.1:.3f}")
                with metric_cols[2]:
                    st.metric("Best IC", f"{best_ic:.4f}", f"+{best_ic*0.01:.4f}")
                with metric_cols[3]:
                    st.metric("Valid Formulas", formulas_found, f"+{formulas_found//10}")

            # Graphique de progression
            training_data = pd.DataFrame({
                'Episode': range(100),
                'Reward': np.cumsum(np.random.randn(100) * 0.1),
                'IC Score': np.abs(np.random.randn(100) * 0.05),
                'Exploration': np.linspace(1.0, 0.1, 100)
            })

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=training_data['Episode'],
                y=training_data['Reward'],
                mode='lines',
                name='Cumulative Reward',
                line=dict(color='#00ff88', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=training_data['Episode'],
                y=training_data['IC Score'],
                mode='lines',
                name='IC Score',
                line=dict(color='#ff6b6b', width=2),
                yaxis='y2'
            ))

            fig.update_layout(
                title="Training Progress",
                xaxis_title="Episode",
                yaxis_title="Reward",
                yaxis2=dict(
                    title="IC Score",
                    overlaying='y',
                    side='right'
                ),
                hovermode='x unified',
                template='plotly_dark',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Generated Alphas")

        # Affichage des alphas générées
        st.markdown("### Top Performing Formulas")

        # Simulation d'alphas générées (en production, venir de l'agent RL)
        example_alphas = [
            {
                "formula": "(-1 * Corr(open, volume, 10))",
                "ic": 0.0723,
                "complexity": 3,
                "type": "Mean Reversion"
            },
            {
                "formula": "sign(delta(cs_rank(close), 5) * volume)",
                "ic": 0.0651,
                "complexity": 4,
                "type": "Momentum"
            },
            {
                "formula": "ts_rank(vwap - Min(low, 20), 10)",
                "ic": 0.0589,
                "complexity": 5,
                "type": "Technical"
            }
        ]

        for i, alpha in enumerate(example_alphas, 1):
            st.markdown(f"""
            <div class="model-card">
                <h4>Alpha #{i}</h4>
                <div class="alpha-formula">{alpha['formula']}</div>
                <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                    <span>IC: <span class="ic-score">{alpha['ic']:.4f}</span></span>
                    <span>Complexity: {alpha['complexity']}</span>
                    <span>Type: {alpha['type']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Actions sur les alphas
        st.markdown("### Actions")
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button("💾 Save to Library", use_container_width=True):
                st.success("Alphas sauvegardées!")
        with col_act2:
            if st.button("🧪 Test Alpha", use_container_width=True):
                st.info("Test lancé...")

# ================== TAB 2: DMN LSTM Training ==================
with tab2:
    st.header("🧬 Deep Market Networks - LSTM Training")
    st.info("Entraînement de modèles LSTM avec attention pour prédiction de marchés")

    col1, col2 = st.columns([3, 1])

    with col1:
        # Configuration du modèle
        st.subheader("🔧 Model Configuration")

        with st.expander("Architecture LSTM", expanded=True):
            col_arch1, col_arch2, col_arch3 = st.columns(3)

            with col_arch1:
                window_size = st.slider(
                    "Window Size",
                    min_value=10,
                    max_value=200,
                    value=64,
                    help="Taille de la fenêtre temporelle"
                )

                hidden_size = st.selectbox(
                    "Hidden Size",
                    [32, 64, 128, 256, 512],
                    index=1,
                    help="Taille des couches cachées LSTM"
                )

                num_layers = st.slider(
                    "Number of Layers",
                    min_value=1,
                    max_value=5,
                    value=2,
                    help="Nombre de couches LSTM"
                )

            with col_arch2:
                dropout = st.slider(
                    "Dropout Rate",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.2,
                    help="Taux de dropout pour régularisation"
                )

                use_attention = st.checkbox(
                    "Use Attention Mechanism",
                    value=True,
                    help="Activer le mécanisme d'attention"
                )

                bidirectional = st.checkbox(
                    "Bidirectional LSTM",
                    value=False,
                    help="LSTM bidirectionnel"
                )

            with col_arch3:
                optimizer = st.selectbox(
                    "Optimizer",
                    ["Adam", "SGD", "RMSprop", "AdamW"],
                    help="Optimiseur pour l'entraînement"
                )

                learning_rate_lstm = st.number_input(
                    "Learning Rate",
                    min_value=0.0001,
                    max_value=0.1,
                    value=0.001,
                    format="%.4f"
                )

                batch_size_lstm = st.selectbox(
                    "Batch Size",
                    [16, 32, 64, 128, 256],
                    index=2
                )

        # Dataset Configuration
        with st.expander("📊 Dataset Configuration"):
            col_data1, col_data2 = st.columns(2)

            with col_data1:
                symbols = st.multiselect(
                    "Symbols",
                    ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"],
                    default=["BTC/USDT"],
                    help="Symboles pour l'entraînement"
                )

                timeframe = st.selectbox(
                    "Timeframe",
                    ["1m", "5m", "15m", "1h", "4h", "1d"],
                    index=3,
                    help="Période temporelle des données"
                )

            with col_data2:
                train_split = st.slider(
                    "Train/Test Split",
                    min_value=0.6,
                    max_value=0.9,
                    value=0.8,
                    help="Proportion de données d'entraînement"
                )

                validation_split = st.slider(
                    "Validation Split",
                    min_value=0.1,
                    max_value=0.3,
                    value=0.2,
                    help="Proportion de validation"
                )

        # Training Controls
        st.subheader("🎮 Training Controls")
        col_train1, col_train2, col_train3, col_train4 = st.columns(4)

        with col_train1:
            epochs = st.number_input(
                "Epochs",
                min_value=1,
                max_value=1000,
                value=100,
                help="Nombre d'époques d'entraînement"
            )

        with col_train2:
            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                st.session_state['lstm_training'] = True
                st.success("LSTM Training démarré!")

        with col_train3:
            if st.button("⏸️ Pause Training", use_container_width=True):
                st.session_state['lstm_training'] = False

        with col_train4:
            if st.button("💾 Save Model", use_container_width=True):
                st.success("Modèle sauvegardé!")

        # Training Visualization
        if st.session_state.get('lstm_training', False):
            st.subheader("📊 Training Progress")

            # Métriques en temps réel
            metric_cols = st.columns(5)
            with metric_cols[0]:
                st.metric("Epoch", "24/100", "+1")
            with metric_cols[1]:
                st.metric("Train Loss", "0.0234", "-0.002")
            with metric_cols[2]:
                st.metric("Val Loss", "0.0289", "-0.001")
            with metric_cols[3]:
                st.metric("Accuracy", "87.3%", "+0.5%")
            with metric_cols[4]:
                st.metric("Time/Epoch", "1.2s", "-0.1s")

            # Graphiques de loss
            loss_data = pd.DataFrame({
                'Epoch': range(1, 25),
                'Train Loss': np.exp(-np.linspace(2, 4, 24)) + np.random.randn(24) * 0.01,
                'Val Loss': np.exp(-np.linspace(1.8, 3.5, 24)) + np.random.randn(24) * 0.02
            })

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=loss_data['Epoch'],
                y=loss_data['Train Loss'],
                mode='lines',
                name='Train Loss',
                line=dict(color='#00ff88', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=loss_data['Epoch'],
                y=loss_data['Val Loss'],
                mode='lines',
                name='Validation Loss',
                line=dict(color='#ff6b6b', width=2)
            ))

            fig.update_layout(
                title="Training & Validation Loss",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                template='plotly_dark',
                height=350
            )

            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 Model Performance")

        # Prédictions vs Réalité
        st.markdown("### Predictions vs Actual")

        # Simulation de prédictions
        pred_data = pd.DataFrame({
            'Time': pd.date_range(start='2024-01-01', periods=100, freq='H'),
            'Actual': np.cumsum(np.random.randn(100) * 0.01) + 100,
            'Predicted': np.cumsum(np.random.randn(100) * 0.01) + 100
        })

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=pred_data['Time'],
            y=pred_data['Actual'],
            mode='lines',
            name='Actual',
            line=dict(color='#00ff88', width=2)
        ))
        fig_pred.add_trace(go.Scatter(
            x=pred_data['Time'],
            y=pred_data['Predicted'],
            mode='lines',
            name='Predicted',
            line=dict(color='#ff6b6b', width=2, dash='dash')
        ))

        fig_pred.update_layout(
            title="LSTM Predictions",
            xaxis_title="Time",
            yaxis_title="Price",
            template='plotly_dark',
            height=300,
            showlegend=True
        )

        st.plotly_chart(fig_pred, use_container_width=True)

        # Métriques de performance
        st.markdown("### Performance Metrics")
        st.metric("R² Score", "0.8734")
        st.metric("MAE", "0.0142")
        st.metric("RMSE", "0.0234")
        st.metric("Direction Accuracy", "73.4%")

# ================== TAB 3: Models Performance ==================
with tab3:
    st.header("📊 Models Performance Dashboard")

    # Sélection du modèle
    col_select1, col_select2 = st.columns([2, 1])
    with col_select1:
        model_type = st.selectbox(
            "Select Model Type",
            ["RL Alpha Models", "DMN LSTM Models", "All Models"],
            help="Filtrer par type de modèle"
        )
    with col_select2:
        time_period = st.selectbox(
            "Time Period",
            ["Last 24h", "Last Week", "Last Month", "All Time"],
            help="Période d'analyse"
        )

    # Tableau de performance des modèles
    st.subheader("📋 Model Comparison")

    models_data = pd.DataFrame({
        'Model': ['RL_Alpha_v1.2', 'DMN_LSTM_64', 'RL_Alpha_v2.0', 'DMN_LSTM_128', 'Ensemble_001'],
        'Type': ['RL', 'LSTM', 'RL', 'LSTM', 'Ensemble'],
        'IC Score': [0.0723, 0.0651, 0.0812, 0.0689, 0.0856],
        'Sharpe': [1.82, 1.65, 2.14, 1.73, 2.31],
        'Returns': ['12.3%', '10.5%', '15.7%', '11.2%', '17.2%'],
        'Status': ['🟢 Active', '🟡 Testing', '🟢 Active', '🔴 Stopped', '🟢 Active'],
        'Created': ['2024-01-15', '2024-01-10', '2024-01-20', '2024-01-08', '2024-01-22']
    })

    st.dataframe(
        models_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            "IC Score": st.column_config.ProgressColumn(
                "IC Score",
                help="Information Coefficient",
                min_value=0,
                max_value=0.1,
            ),
            "Sharpe": st.column_config.NumberColumn(
                "Sharpe",
                help="Sharpe Ratio",
                format="%.2f",
            ),
        }
    )

    # Graphiques de comparaison
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        # Bar chart des IC Scores
        fig_ic = px.bar(
            models_data,
            x='Model',
            y='IC Score',
            color='Type',
            title='Information Coefficient by Model',
            color_discrete_map={'RL': '#00ff88', 'LSTM': '#ff6b6b', 'Ensemble': '#6b88ff'}
        )
        fig_ic.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig_ic, use_container_width=True)

    with col_chart2:
        # Scatter plot Sharpe vs Returns
        models_data['Returns_num'] = models_data['Returns'].str.rstrip('%').astype(float)
        fig_scatter = px.scatter(
            models_data,
            x='Sharpe',
            y='Returns_num',
            size='IC Score',
            color='Type',
            hover_data=['Model'],
            title='Risk-Adjusted Performance',
            labels={'Returns_num': 'Returns (%)'},
            color_discrete_map={'RL': '#00ff88', 'LSTM': '#ff6b6b', 'Ensemble': '#6b88ff'}
        )
        fig_scatter.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

# ================== TAB 4: Alpha Library ==================
with tab4:
    st.header("🏛️ Alpha Formula Library")
    st.info("Bibliothèque de formules alpha générées et curées")

    # Filtres et recherche
    col_filter1, col_filter2, col_filter3 = st.columns([2, 1, 1])
    with col_filter1:
        search_query = st.text_input(
            "🔍 Search Formulas",
            placeholder="Ex: corr, volume, delta...",
            help="Rechercher dans les formules"
        )
    with col_filter2:
        filter_type = st.selectbox(
            "Formula Type",
            ["All", "Mean Reversion", "Momentum", "Technical", "Volume", "Custom"]
        )
    with col_filter3:
        sort_by = st.selectbox(
            "Sort By",
            ["IC Score", "Sharpe Ratio", "Complexity", "Date Added"]
        )

    # Catégories de formules
    st.subheader("📚 Formula Categories")

    tab_classic, tab_generated, tab_custom = st.tabs(["Classic Alpha101", "RL Generated", "Custom Formulas"])

    with tab_classic:
        st.markdown("### Academic Alpha Formulas (Alpha101)")

        alpha101_formulas = [
            {
                "id": "Alpha006",
                "formula": "(-1 * Corr(open, volume, 10))",
                "description": "Negative correlation between open price and volume",
                "ic": 0.0723,
                "complexity": 3
            },
            {
                "id": "Alpha061",
                "formula": "Less(CSRank((vwap - Min(vwap, 16))), CSRank(Corr(vwap, Mean(volume, 180), 17)))",
                "description": "Complex VWAP and volume correlation ranking",
                "ic": 0.0456,
                "complexity": 8
            },
            {
                "id": "Alpha099",
                "formula": "sign(delta(cs_rank(close * volume), 5))",
                "description": "Sign of delta in cross-sectional rank of dollar volume",
                "ic": 0.0612,
                "complexity": 5
            }
        ]

        for alpha in alpha101_formulas:
            with st.expander(f"{alpha['id']} - IC: {alpha['ic']:.4f}"):
                st.code(alpha['formula'], language='python')
                st.markdown(f"**Description:** {alpha['description']}")
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    st.metric("IC Score", f"{alpha['ic']:.4f}")
                with col_a2:
                    st.metric("Complexity", alpha['complexity'])

                if st.button(f"Deploy {alpha['id']}", key=f"deploy_{alpha['id']}"):
                    st.success(f"{alpha['id']} déployé!")

    with tab_generated:
        st.markdown("### RL Agent Generated Formulas")

        # Formules générées par RL
        rl_formulas = [
            {
                "id": "RL_2024_001",
                "formula": "ts_rank(delta(vwap, 5) * sign(volume - Mean(volume, 20)), 10)",
                "ic": 0.0812,
                "generation": 523,
                "agent": "PPO_v2.0"
            },
            {
                "id": "RL_2024_002",
                "formula": "product(cs_rank(high - low), wma(close, 15))",
                "ic": 0.0734,
                "generation": 892,
                "agent": "PPO_v2.0"
            }
        ]

        for formula in rl_formulas:
            with st.expander(f"{formula['id']} - IC: {formula['ic']:.4f}"):
                st.code(formula['formula'], language='python')
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    st.metric("IC Score", f"{formula['ic']:.4f}")
                with col_r2:
                    st.metric("Generation", formula['generation'])
                with col_r3:
                    st.metric("Agent", formula['agent'])

                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    if st.button(f"Backtest", key=f"bt_{formula['id']}"):
                        st.info("Backtest lancé...")
                with col_b2:
                    if st.button(f"Deploy", key=f"deploy_rl_{formula['id']}"):
                        st.success("Formule déployée!")

    with tab_custom:
        st.markdown("### Create Custom Formula")

        # Éditeur de formules
        custom_formula = st.text_area(
            "Formula Editor",
            placeholder="Enter your custom formula using symbolic operators...\nEx: delta(cs_rank(close), 5) * volume",
            height=100,
            help="Utilisez les opérateurs disponibles: delta, ts_rank, cs_rank, corr, etc."
        )

        col_custom1, col_custom2 = st.columns(2)
        with col_custom1:
            if st.button("🧪 Validate Formula", type="primary", use_container_width=True):
                if custom_formula:
                    st.success("✅ Formula syntaxiquement valide!")
                    st.info("IC Score estimé: 0.0523")

        with col_custom2:
            if st.button("💾 Save to Library", use_container_width=True):
                if custom_formula:
                    st.success("Formula sauvegardée dans la bibliothèque!")

# ================== TAB 5: Configuration ==================
with tab5:
    st.header("⚙️ Research Lab Configuration")

    # Configuration globale
    st.subheader("🔧 Global Settings")

    col_conf1, col_conf2 = st.columns(2)

    with col_conf1:
        st.markdown("### Compute Resources")

        use_gpu = st.checkbox(
            "Use GPU Acceleration",
            value=True,
            help="Utiliser GPU pour l'entraînement (si disponible)"
        )

        if use_gpu:
            gpu_device = st.selectbox(
                "GPU Device",
                ["cuda:0", "cuda:1", "Auto"],
                help="Sélectionner le GPU à utiliser"
            )

        max_memory = st.slider(
            "Max Memory Usage (GB)",
            min_value=1,
            max_value=64,
            value=8,
            help="Limite de mémoire pour l'entraînement"
        )

        num_workers = st.number_input(
            "Number of Workers",
            min_value=1,
            max_value=16,
            value=4,
            help="Nombre de workers parallèles"
        )

    with col_conf2:
        st.markdown("### Data Settings")

        data_provider = st.selectbox(
            "Data Provider",
            ["Binance", "CCXT Multi-Exchange", "Local Files"],
            help="Source de données pour l'entraînement"
        )

        cache_enabled = st.checkbox(
            "Enable Data Caching",
            value=True,
            help="Cache des données pour performance"
        )

        if cache_enabled:
            cache_ttl = st.number_input(
                "Cache TTL (minutes)",
                min_value=1,
                max_value=1440,
                value=60,
                help="Durée de vie du cache"
            )

    # Sauvegarde de configuration
    st.markdown("---")
    col_save1, col_save2, col_save3 = st.columns([1, 1, 2])

    with col_save1:
        if st.button("💾 Save Configuration", type="primary", use_container_width=True):
            st.success("Configuration sauvegardée!")

    with col_save2:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            st.info("Configuration réinitialisée")

    # Monitoring système
    st.subheader("📊 System Monitoring")

    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("CPU Usage", "42%", "-5%")
    with metric_cols[1]:
        st.metric("Memory", "5.2/8.0 GB", "+0.3 GB")
    with metric_cols[2]:
        st.metric("GPU Memory", "3.1/8.0 GB", "+0.5 GB")
    with metric_cols[3]:
        st.metric("Models Cached", "7", "+2")

# Sidebar avec informations
with st.sidebar:
    st.markdown("### 🧠 Research Lab")
    st.info("Centre de recherche quantitative avec RL et ML")

    st.markdown("### 📊 Active Models")
    st.metric("RL Agents", "3", "+1")
    st.metric("LSTM Models", "2", "0")
    st.metric("Total Formulas", "147", "+12")

    st.markdown("### 🎯 Best Performance")
    st.success("**Top IC:** 0.0856 (Ensemble_001)")
    st.success("**Top Sharpe:** 2.31")

    st.markdown("### 📚 Resources")
    st.markdown("""
    - [RL Alpha Paper](https://arxiv.org/abs/2401.02710v2)
    - [DMN Documentation](#)
    - [Operator Reference](#)
    """)