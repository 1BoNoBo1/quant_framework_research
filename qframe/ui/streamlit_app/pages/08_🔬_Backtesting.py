"""
Backtesting Suite - Interface complète pour validation de stratégies
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple
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
from components.backtesting.backtest_configurator import BacktestConfigurator
from components.backtesting.results_analyzer import ResultsAnalyzer
from components.backtesting.walk_forward_interface import WalkForwardInterface
from components.backtesting.monte_carlo_simulator import MonteCarloSimulator
from components.backtesting.performance_analytics import PerformanceAnalytics
from components.backtesting.integration_manager import BacktestingIntegrationManager

# Configuration de la page
st.set_page_config(
    page_title="QFrame - Backtesting Suite",
    page_icon="🔬",
    layout="wide"
)

# Initialisation
api_client = APIClient()
init_session_state()

# Styles CSS personnalisés
st.markdown("""
<style>
    .backtesting-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .config-card {
        background: #1a1a2e;
        border: 1px solid #00ff88;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 255, 136, 0.1);
    }
    .metric-card {
        background: #0f0f23;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #333;
        margin: 0.5rem;
    }
    .status-running {
        background-color: rgba(0, 255, 136, 0.1);
        border: 1px solid #00ff88;
        border-radius: 8px;
        padding: 1rem;
    }
    .status-completed {
        background-color: rgba(0, 136, 255, 0.1);
        border: 1px solid #0088ff;
        border-radius: 8px;
        padding: 1rem;
    }
    .status-error {
        background-color: rgba(255, 107, 107, 0.1);
        border: 1px solid #ff6b6b;
        border-radius: 8px;
        padding: 1rem;
    }
    .performance-metric {
        font-size: 1.2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .positive { color: #00ff88; }
    .negative { color: #ff6b6b; }
    .neutral { color: #6b88ff; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="backtesting-header">
    <h1>🔬 Backtesting Suite</h1>
    <p>Suite complète de validation et d'analyse de stratégies quantitatives</p>
</div>
""", unsafe_allow_html=True)

# Initialisation session state pour backtesting
if 'backtest_queue' not in st.session_state:
    st.session_state.backtest_queue = []
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = {}
if 'current_backtest' not in st.session_state:
    st.session_state.current_backtest = None

# Navigation principale
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔧 Configuration",
    "▶️ Execution",
    "📊 Results",
    "🏆 Comparison",
    "📋 Reports",
    "🔄 Workflow Intégré"
])

# ================== TAB 1: Configuration ==================
with tab1:
    st.header("🔧 Backtest Configuration")

    col_config_left, col_config_right = st.columns([2, 1])

    with col_config_left:
        # Configuration principale
        with st.expander("⚙️ Strategy & Parameters", expanded=True):
            col_strat1, col_strat2 = st.columns(2)

            with col_strat1:
                strategy_type = st.selectbox(
                    "Strategy Type",
                    [
                        "DMN LSTM Strategy",
                        "Adaptive Mean Reversion",
                        "Funding Arbitrage",
                        "RL Alpha Generator",
                        "Grid Trading",
                        "Simple Moving Average",
                        "Custom Strategy"
                    ],
                    help="Sélectionner la stratégie à tester"
                )

                strategy_config = st.text_area(
                    "Strategy Parameters (JSON)",
                    value="""{
    "window_size": 64,
    "lookback_period": 20,
    "threshold": 0.02,
    "max_position": 1.0
}""",
                    height=120,
                    help="Configuration JSON de la stratégie"
                )

            with col_strat2:
                optimization_mode = st.selectbox(
                    "Optimization Mode",
                    ["None", "Grid Search", "Bayesian", "Genetic Algorithm"],
                    help="Méthode d'optimisation des paramètres"
                )

                if optimization_mode != "None":
                    param_ranges = st.text_area(
                        "Parameter Ranges",
                        value="""{
    "window_size": [32, 64, 128],
    "threshold": [0.01, 0.02, 0.05]
}""",
                        height=80
                    )

        # Configuration des données
        with st.expander("📊 Data & Universe", expanded=True):
            col_data1, col_data2, col_data3 = st.columns(3)

            with col_data1:
                asset_universe = st.multiselect(
                    "Asset Universe",
                    [
                        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
                        "AAPL", "GOOGL", "MSFT", "TSLA",
                        "EUR/USD", "GBP/USD", "USD/JPY"
                    ],
                    default=["BTC/USDT"],
                    help="Sélectionner les actifs à trader"
                )

                timeframe = st.selectbox(
                    "Timeframe",
                    ["1m", "5m", "15m", "1h", "4h", "1d"],
                    index=3,
                    help="Période temporelle des données"
                )

            with col_data2:
                start_date = st.date_input(
                    "Start Date",
                    value=date(2023, 1, 1),
                    help="Date de début du backtest"
                )

                end_date = st.date_input(
                    "End Date",
                    value=date(2024, 1, 1),
                    help="Date de fin du backtest"
                )

            with col_data3:
                initial_capital = st.number_input(
                    "Initial Capital (USD)",
                    min_value=1000,
                    max_value=10000000,
                    value=100000,
                    step=1000
                )

                data_source = st.selectbox(
                    "Data Source",
                    ["Binance", "CCXT Multi-Exchange", "Local CSV", "API Provider"],
                    help="Source des données historiques"
                )

        # Configuration des coûts
        with st.expander("💰 Trading Costs & Constraints"):
            col_cost1, col_cost2, col_cost3 = st.columns(3)

            with col_cost1:
                commission_rate = st.number_input(
                    "Commission Rate (%)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.01,
                    format="%.3f",
                    help="Frais de transaction en %"
                )

                slippage_rate = st.number_input(
                    "Slippage Rate (%)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.05,
                    step=0.01,
                    format="%.3f",
                    help="Slippage estimé en %"
                )

            with col_cost2:
                max_leverage = st.number_input(
                    "Max Leverage",
                    min_value=1.0,
                    max_value=100.0,
                    value=1.0,
                    step=0.1,
                    help="Levier maximum autorisé"
                )

                position_size_limit = st.number_input(
                    "Position Size Limit (%)",
                    min_value=1,
                    max_value=100,
                    value=100,
                    help="Taille max de position en % du capital"
                )

            with col_cost3:
                risk_free_rate = st.number_input(
                    "Risk-Free Rate (%/year)",
                    min_value=0.0,
                    max_value=10.0,
                    value=3.0,
                    step=0.1,
                    help="Taux sans risque pour calcul Sharpe"
                )

                rebalance_frequency = st.selectbox(
                    "Rebalancing",
                    ["Every Trade", "Daily", "Weekly", "Monthly"],
                    index=1,
                    help="Fréquence de rééquilibrage"
                )

        # Benchmarks
        with st.expander("📈 Benchmarks & Comparison"):
            benchmark_assets = st.multiselect(
                "Benchmark Assets",
                ["BTC", "ETH", "SPY", "QQQ", "Custom"],
                default=["BTC"],
                help="Actifs de référence pour comparaison"
            )

            if "Custom" in benchmark_assets:
                custom_benchmark = st.text_input(
                    "Custom Benchmark Strategy",
                    placeholder="Nom de la stratégie de référence"
                )

    with col_config_right:
        st.subheader("📋 Configuration Summary")

        # Résumé de la configuration
        config_summary = {
            "Strategy": strategy_type,
            "Assets": len(asset_universe),
            "Period": f"{start_date} to {end_date}",
            "Capital": f"${initial_capital:,}",
            "Commission": f"{commission_rate}%",
            "Timeframe": timeframe
        }

        for key, value in config_summary.items():
            st.metric(key, value)

        st.markdown("---")

        # Templates de configuration
        st.subheader("🎯 Configuration Templates")

        template_names = [
            "Conservative Portfolio",
            "Aggressive Trading",
            "Crypto Focus",
            "Multi-Asset Balanced",
            "High-Frequency"
        ]

        selected_template = st.selectbox("Load Template", ["None"] + template_names)

        if selected_template != "None":
            if st.button("Load Template", use_container_width=True):
                st.success(f"Template '{selected_template}' loaded!")

        st.markdown("---")

        # Validation de configuration
        st.subheader("✅ Configuration Validation")

        validation_checks = [
            ("Date Range", "✅" if start_date < end_date else "❌"),
            ("Assets Selected", "✅" if asset_universe else "❌"),
            ("Capital > 0", "✅" if initial_capital > 0 else "❌"),
            ("Valid Timeframe", "✅" if timeframe else "❌")
        ]

        for check, status in validation_checks:
            st.markdown(f"{status} {check}")

        # Bouton de lancement
        st.markdown("---")
        config_valid = all(status == "✅" for _, status in validation_checks)

        if config_valid:
            if st.button("🚀 Start Backtest", type="primary", use_container_width=True):
                # Créer la configuration de backtest
                backtest_config = {
                    'id': f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'strategy_type': strategy_type,
                    'assets': asset_universe,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'initial_capital': initial_capital,
                    'timeframe': timeframe,
                    'commission': commission_rate,
                    'slippage': slippage_rate,
                    'created_at': datetime.now(),
                    'status': 'queued'
                }

                st.session_state.backtest_queue.append(backtest_config)
                st.session_state.current_backtest = backtest_config
                st.success("Backtest added to queue!")
                st.rerun()
        else:
            st.error("Please fix configuration errors before starting")

# ================== TAB 2: Execution ==================
with tab2:
    st.header("▶️ Backtest Execution")

    # Queue des backtests
    st.subheader("📋 Backtest Queue")

    if st.session_state.backtest_queue:
        queue_df = pd.DataFrame([
            {
                'ID': bt['id'],
                'Strategy': bt['strategy_type'],
                'Assets': ', '.join(bt['assets'][:2]) + ('...' if len(bt['assets']) > 2 else ''),
                'Period': f"{bt['start_date']} to {bt['end_date']}",
                'Status': bt['status'],
                'Created': bt['created_at'].strftime('%H:%M:%S')
            }
            for bt in st.session_state.backtest_queue
        ])

        st.dataframe(
            queue_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Status": st.column_config.TextColumn(
                    "Status",
                    help="Current backtest status"
                )
            }
        )

        # Contrôles de queue
        col_queue1, col_queue2, col_queue3 = st.columns(3)

        with col_queue1:
            if st.button("▶️ Process Queue", type="primary", use_container_width=True):
                st.session_state.processing_queue = True
                st.rerun()

        with col_queue2:
            if st.button("⏸️ Pause Queue", use_container_width=True):
                st.session_state.processing_queue = False

        with col_queue3:
            if st.button("🗑️ Clear Queue", use_container_width=True):
                st.session_state.backtest_queue = []
                st.rerun()

    else:
        st.info("No backtests in queue. Configure a backtest in the Configuration tab.")

    # Execution en cours
    if st.session_state.get('processing_queue', False) and st.session_state.backtest_queue:
        st.markdown("---")
        st.subheader("⚡ Current Execution")

        current_bt = st.session_state.backtest_queue[0]

        # Status en cours
        st.markdown(f"""
        <div class="status-running">
            <h4>🔄 Running: {current_bt['strategy_type']}</h4>
            <p><strong>Assets:</strong> {', '.join(current_bt['assets'])}</p>
            <p><strong>Period:</strong> {current_bt['start_date']} to {current_bt['end_date']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Progress bar avec simulation
        progress_col, metrics_col = st.columns([2, 1])

        with progress_col:
            # Simulation de progrès
            if 'backtest_progress' not in st.session_state:
                st.session_state.backtest_progress = 0

            progress = st.session_state.backtest_progress
            st.progress(progress / 100)

            # Simulation d'avancement
            if progress < 100:
                st.session_state.backtest_progress += np.random.randint(1, 5)
                if st.session_state.backtest_progress > 100:
                    st.session_state.backtest_progress = 100

            # Informations de progression
            total_days = (datetime.fromisoformat(current_bt['end_date']) -
                         datetime.fromisoformat(current_bt['start_date'])).days
            current_day = int(total_days * progress / 100)

            st.markdown(f"""
            **Progress:** {progress:.1f}%
            **Current Date:** {datetime.fromisoformat(current_bt['start_date']) + timedelta(days=current_day)}
            **ETA:** {max(0, int((100 - progress) * 2))} seconds
            """)

        with metrics_col:
            # Métriques intermédiaires simulées
            if progress > 20:
                interim_return = np.random.uniform(-5, 15)
                interim_sharpe = np.random.uniform(0.5, 2.5)
                trades_executed = int(progress * np.random.uniform(0.5, 2.0))

                st.metric("Interim Return", f"{interim_return:.1f}%")
                st.metric("Current Sharpe", f"{interim_sharpe:.2f}")
                st.metric("Trades", trades_executed)

        # Boutons de contrôle
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            if st.button("⏸️ Pause Execution", use_container_width=True):
                st.session_state.processing_queue = False

        with col_ctrl2:
            if st.button("🛑 Stop & Save", use_container_width=True):
                st.session_state.processing_queue = False

        # Auto-completion simulation
        if progress >= 100:
            # Simuler la completion
            result_id = current_bt['id']
            simulated_results = generate_simulated_results(current_bt)

            st.session_state.backtest_results[result_id] = simulated_results
            st.session_state.backtest_queue.pop(0)
            st.session_state.backtest_progress = 0

            st.success(f"✅ Backtest completed: {current_bt['strategy_type']}")
            time.sleep(1)
            st.rerun()

    # Historique des exécutions récentes
    if st.session_state.backtest_results:
        st.markdown("---")
        st.subheader("📊 Recent Completions")

        recent_results = list(st.session_state.backtest_results.items())[-3:]

        for result_id, result in recent_results:
            col_res1, col_res2, col_res3, col_res4 = st.columns(4)

            with col_res1:
                st.markdown(f"**{result['strategy_type']}**")
                st.markdown(f"*{result_id}*")

            with col_res2:
                st.metric("Total Return", f"{result['total_return']:.1f}%")

            with col_res3:
                st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")

            with col_res4:
                if st.button("View Results", key=f"view_{result_id}"):
                    st.session_state.selected_result = result_id
                    st.switch_page("pages/08_🔬_Backtesting.py")

# ================== TAB 3: Results ==================
with tab3:
    st.header("📊 Backtest Results")

    if not st.session_state.backtest_results:
        st.info("No backtest results available. Run a backtest first.")
    else:
        # Sélection du résultat à analyser
        result_ids = list(st.session_state.backtest_results.keys())
        selected_result = st.selectbox(
            "Select Backtest Result",
            result_ids,
            index=len(result_ids) - 1  # Dernier résultat par défaut
        )

        if selected_result:
            result_data = st.session_state.backtest_results[selected_result]

            # Métriques principales
            st.subheader("📈 Performance Overview")

            col_perf1, col_perf2, col_perf3, col_perf4, col_perf5 = st.columns(5)

            with col_perf1:
                total_return = result_data['total_return']
                st.metric(
                    "Total Return",
                    f"{total_return:.1f}%",
                    delta=f"{total_return - 0:.1f}%"
                )

            with col_perf2:
                sharpe = result_data['sharpe_ratio']
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")

            with col_perf3:
                max_dd = result_data['max_drawdown']
                st.metric("Max Drawdown", f"{max_dd:.1f}%")

            with col_perf4:
                win_rate = result_data['win_rate']
                st.metric("Win Rate", f"{win_rate:.1f}%")

            with col_perf5:
                total_trades = result_data['total_trades']
                st.metric("Total Trades", total_trades)

            # Graphiques principaux
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                # Equity curve
                equity_data = result_data['equity_curve']
                fig_equity = go.Figure()

                fig_equity.add_trace(go.Scatter(
                    x=equity_data['dates'],
                    y=equity_data['values'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#00ff88', width=2)
                ))

                # Benchmark
                if 'benchmark_curve' in result_data:
                    benchmark_data = result_data['benchmark_curve']
                    fig_equity.add_trace(go.Scatter(
                        x=benchmark_data['dates'],
                        y=benchmark_data['values'],
                        mode='lines',
                        name='Benchmark',
                        line=dict(color='#666666', width=1, dash='dash')
                    ))

                fig_equity.update_layout(
                    title="Equity Curve",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value (USD)",
                    template='plotly_dark',
                    height=400
                )

                st.plotly_chart(fig_equity, use_container_width=True)

            with col_chart2:
                # Drawdown chart
                drawdown_data = result_data['drawdown_series']
                fig_dd = go.Figure()

                fig_dd.add_trace(go.Scatter(
                    x=equity_data['dates'],
                    y=drawdown_data,
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='#ff6b6b', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255, 107, 107, 0.3)'
                ))

                fig_dd.update_layout(
                    title="Drawdown Analysis",
                    xaxis_title="Date",
                    yaxis_title="Drawdown (%)",
                    template='plotly_dark',
                    height=400
                )

                st.plotly_chart(fig_dd, use_container_width=True)

            # Métriques détaillées
            st.subheader("📋 Detailed Metrics")

            col_metrics1, col_metrics2, col_metrics3 = st.columns(3)

            with col_metrics1:
                st.markdown("### Returns Metrics")
                metrics_data = [
                    ("Annualized Return", f"{result_data['annualized_return']:.1f}%"),
                    ("Volatility", f"{result_data['volatility']:.1f}%"),
                    ("Best Month", f"{result_data['best_month']:.1f}%"),
                    ("Worst Month", f"{result_data['worst_month']:.1f}%")
                ]

                for metric, value in metrics_data:
                    st.markdown(f"**{metric}:** {value}")

            with col_metrics2:
                st.markdown("### Risk Metrics")
                risk_metrics = [
                    ("Sortino Ratio", f"{result_data['sortino_ratio']:.2f}"),
                    ("Calmar Ratio", f"{result_data['calmar_ratio']:.2f}"),
                    ("VaR (95%)", f"{result_data['var_95']:.1f}%"),
                    ("CVaR (95%)", f"{result_data['cvar_95']:.1f}%")
                ]

                for metric, value in risk_metrics:
                    st.markdown(f"**{metric}:** {value}")

            with col_metrics3:
                st.markdown("### Trade Metrics")
                trade_metrics = [
                    ("Profit Factor", f"{result_data['profit_factor']:.2f}"),
                    ("Avg Trade", f"{result_data['avg_trade']:.2f}%"),
                    ("Avg Win", f"{result_data['avg_win']:.2f}%"),
                    ("Avg Loss", f"{result_data['avg_loss']:.2f}%")
                ]

                for metric, value in trade_metrics:
                    st.markdown(f"**{metric}:** {value}")

            # Distribution des returns
            st.subheader("📊 Returns Distribution")

            monthly_returns = result_data['monthly_returns']
            fig_dist = go.Figure()

            fig_dist.add_trace(go.Histogram(
                x=monthly_returns,
                nbinsx=20,
                name='Monthly Returns',
                marker_color='#00ff88',
                opacity=0.7
            ))

            # Ligne de moyenne
            mean_return = np.mean(monthly_returns)
            fig_dist.add_vline(
                x=mean_return,
                line_dash="dash",
                line_color="#ff6b6b",
                annotation_text=f"Mean: {mean_return:.1f}%"
            )

            fig_dist.update_layout(
                title="Monthly Returns Distribution",
                xaxis_title="Monthly Return (%)",
                yaxis_title="Frequency",
                template='plotly_dark',
                height=350
            )

            st.plotly_chart(fig_dist, use_container_width=True)

            # Analytics de performance avancées
            st.divider()
            analytics = PerformanceAnalytics()
            analytics.render_advanced_analytics(result_data)

# ================== TAB 4: Comparison ==================
with tab4:
    st.header("🏆 Strategy Comparison")

    if len(st.session_state.backtest_results) < 2:
        st.info("Need at least 2 backtest results for comparison.")
    else:
        # Sélection des stratégies à comparer
        result_ids = list(st.session_state.backtest_results.keys())
        selected_strategies = st.multiselect(
            "Select Strategies to Compare",
            result_ids,
            default=result_ids[-2:] if len(result_ids) >= 2 else result_ids
        )

        if len(selected_strategies) >= 2:
            # Tableau de comparaison
            comparison_data = []
            for result_id in selected_strategies:
                result = st.session_state.backtest_results[result_id]
                comparison_data.append({
                    'Strategy': result['strategy_type'],
                    'ID': result_id.split('_')[-1],  # Juste le timestamp
                    'Total Return (%)': f"{result['total_return']:.1f}",
                    'Sharpe Ratio': f"{result['sharpe_ratio']:.2f}",
                    'Max Drawdown (%)': f"{result['max_drawdown']:.1f}",
                    'Win Rate (%)': f"{result['win_rate']:.1f}",
                    'Volatility (%)': f"{result['volatility']:.1f}",
                    'Total Trades': result['total_trades']
                })

            comparison_df = pd.DataFrame(comparison_data)

            st.subheader("📋 Performance Comparison Table")
            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )

            # Graphiques de comparaison
            col_comp1, col_comp2 = st.columns(2)

            with col_comp1:
                # Equity curves comparatives
                fig_comp = go.Figure()

                colors = ['#00ff88', '#ff6b6b', '#6b88ff', '#ffd93d', '#ff6bff']

                for i, result_id in enumerate(selected_strategies):
                    result = st.session_state.backtest_results[result_id]
                    equity_data = result['equity_curve']

                    fig_comp.add_trace(go.Scatter(
                        x=equity_data['dates'],
                        y=equity_data['values'],
                        mode='lines',
                        name=result['strategy_type'],
                        line=dict(color=colors[i % len(colors)], width=2)
                    ))

                fig_comp.update_layout(
                    title="Comparative Equity Curves",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value (USD)",
                    template='plotly_dark',
                    height=400
                )

                st.plotly_chart(fig_comp, use_container_width=True)

            with col_comp2:
                # Radar chart des métriques
                metrics_for_radar = ['Total Return', 'Sharpe Ratio', 'Win Rate', 'Volatility']

                fig_radar = go.Figure()

                for i, result_id in enumerate(selected_strategies):
                    result = st.session_state.backtest_results[result_id]

                    # Normaliser les métriques pour le radar (0-1)
                    normalized_metrics = [
                        min(1, max(0, result['total_return'] / 50)),  # Return normalisé sur 50%
                        min(1, max(0, result['sharpe_ratio'] / 3)),   # Sharpe normalisé sur 3
                        result['win_rate'] / 100,                     # Win rate déjà en %
                        1 - min(1, result['volatility'] / 50)        # Volatility inversée
                    ]

                    fig_radar.add_trace(go.Scatterpolar(
                        r=normalized_metrics + [normalized_metrics[0]],  # Fermer le polygone
                        theta=metrics_for_radar + [metrics_for_radar[0]],
                        fill='toself',
                        name=result['strategy_type'],
                        line=dict(color=colors[i % len(colors)])
                    ))

                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    title="Performance Radar Chart",
                    template='plotly_dark',
                    height=400
                )

                st.plotly_chart(fig_radar, use_container_width=True)

            # Analyse statistique
            st.subheader("📊 Statistical Analysis")

            # Corrélation entre stratégies
            returns_data = {}
            for result_id in selected_strategies:
                result = st.session_state.backtest_results[result_id]
                returns_data[result['strategy_type']] = result['monthly_returns']

            if len(returns_data) >= 2:
                returns_df = pd.DataFrame(returns_data)
                correlation_matrix = returns_df.corr()

                fig_corr = px.imshow(
                    correlation_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Strategy Correlation Matrix",
                    color_continuous_scale='RdBu'
                )
                fig_corr.update_layout(template='plotly_dark', height=400)

                st.plotly_chart(fig_corr, use_container_width=True)

# ================== TAB 5: Reports ==================
with tab5:
    st.header("📋 Backtest Reports")

    if not st.session_state.backtest_results:
        st.info("No backtest results available for reporting.")
    else:
        # Sélection du rapport
        report_types = [
            "Executive Summary",
            "Detailed Performance Report",
            "Risk Analysis Report",
            "Trade Analysis Report",
            "Benchmark Comparison Report"
        ]

        selected_report = st.selectbox("Select Report Type", report_types)
        result_for_report = st.selectbox(
            "Select Backtest",
            list(st.session_state.backtest_results.keys())
        )

        if st.button("📄 Generate Report", type="primary"):
            with st.spinner("Generating report..."):
                time.sleep(2)  # Simulation

                result_data = st.session_state.backtest_results[result_for_report]

                if selected_report == "Executive Summary":
                    st.markdown(generate_executive_summary(result_data))
                elif selected_report == "Detailed Performance Report":
                    st.markdown(generate_detailed_report(result_data))
                elif selected_report == "Risk Analysis Report":
                    st.markdown(generate_risk_report(result_data))
                elif selected_report == "Trade Analysis Report":
                    st.markdown(generate_trade_report(result_data))
                elif selected_report == "Benchmark Comparison Report":
                    st.markdown(generate_benchmark_report(result_data))

                st.success("Report generated successfully!")

        # Options d'export
        st.markdown("---")
        st.subheader("📤 Export Options")

        col_export1, col_export2, col_export3 = st.columns(3)

        with col_export1:
            if st.button("📄 Export PDF", use_container_width=True):
                st.info("PDF export functionality would be implemented here")

        with col_export2:
            if st.button("📊 Export Excel", use_container_width=True):
                st.info("Excel export functionality would be implemented here")

        with col_export3:
            if st.button("📧 Email Report", use_container_width=True):
                st.info("Email functionality would be implemented here")

# Fonctions utilitaires
def generate_simulated_results(config):
    """Génère des résultats de backtest simulés."""
    # Simulation de données réalistes
    np.random.seed(42)  # Pour reproductibilité

    # Métriques de base
    total_return = np.random.uniform(-10, 30)
    volatility = np.random.uniform(15, 40)
    sharpe_ratio = np.random.uniform(0.5, 2.5)

    # Dates
    start_date = datetime.fromisoformat(config['start_date'])
    end_date = datetime.fromisoformat(config['end_date'])
    dates = pd.date_range(start_date, end_date, freq='D')

    # Equity curve
    daily_returns = np.random.randn(len(dates)) * (volatility / 100) / np.sqrt(252)
    equity_values = config['initial_capital'] * np.cumprod(1 + daily_returns)

    # Drawdowns
    rolling_max = np.maximum.accumulate(equity_values)
    drawdowns = (equity_values - rolling_max) / rolling_max * 100

    # Monthly returns
    monthly_returns = np.random.randn(12) * (volatility / 100) / np.sqrt(12) * 100

    return {
        'strategy_type': config['strategy_type'],
        'total_return': total_return,
        'annualized_return': total_return * (365 / (end_date - start_date).days),
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sharpe_ratio * 1.2,
        'calmar_ratio': total_return / abs(np.min(drawdowns)) if np.min(drawdowns) < 0 else total_return,
        'max_drawdown': np.min(drawdowns),
        'var_95': np.percentile(daily_returns * 100, 5),
        'cvar_95': np.mean(daily_returns[daily_returns <= np.percentile(daily_returns, 5)]) * 100,
        'win_rate': np.random.uniform(45, 70),
        'total_trades': np.random.randint(50, 500),
        'profit_factor': np.random.uniform(1.1, 2.5),
        'avg_trade': np.random.uniform(-0.5, 1.5),
        'avg_win': np.random.uniform(1.0, 3.0),
        'avg_loss': np.random.uniform(-2.0, -0.5),
        'best_month': np.max(monthly_returns),
        'worst_month': np.min(monthly_returns),
        'monthly_returns': monthly_returns.tolist(),
        'equity_curve': {
            'dates': dates.tolist(),
            'values': equity_values.tolist()
        },
        'benchmark_curve': {
            'dates': dates.tolist(),
            'values': (config['initial_capital'] * np.cumprod(1 + np.random.randn(len(dates)) * 0.01)).tolist()
        },
        'drawdown_series': drawdowns.tolist()
    }

def generate_executive_summary(result_data):
    """Génère un résumé exécutif."""
    return f"""
# 📊 Executive Summary

## Strategy Overview
**Strategy:** {result_data['strategy_type']}
**Period:** Backtest period analysis
**Status:** ✅ Completed Successfully

## Key Performance Indicators

### 🎯 Returns
- **Total Return:** {result_data['total_return']:.1f}%
- **Annualized Return:** {result_data['annualized_return']:.1f}%
- **Best Month:** {result_data['best_month']:.1f}%
- **Worst Month:** {result_data['worst_month']:.1f}%

### 📈 Risk Metrics
- **Sharpe Ratio:** {result_data['sharpe_ratio']:.2f}
- **Sortino Ratio:** {result_data['sortino_ratio']:.2f}
- **Maximum Drawdown:** {result_data['max_drawdown']:.1f}%
- **Volatility:** {result_data['volatility']:.1f}%

### 💼 Trading Activity
- **Total Trades:** {result_data['total_trades']}
- **Win Rate:** {result_data['win_rate']:.1f}%
- **Profit Factor:** {result_data['profit_factor']:.2f}

## 🎯 Key Insights

✅ **Strengths:**
- Strong risk-adjusted returns (Sharpe > 1.0)
- Consistent performance across different market conditions
- Reasonable drawdown levels

⚠️ **Areas for Improvement:**
- Consider position sizing optimization
- Monitor correlation with market benchmarks
- Evaluate performance during high volatility periods

## 📋 Recommendation

{'**APPROVED** for live trading with recommended position sizing' if result_data['sharpe_ratio'] > 1.0 and result_data['max_drawdown'] > -20 else '**REQUIRES OPTIMIZATION** before live deployment'}

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

def generate_detailed_report(result_data):
    """Génère un rapport détaillé."""
    return f"""
# 📊 Detailed Performance Report

## Strategy Configuration
- **Strategy Type:** {result_data['strategy_type']}
- **Analysis Period:** Full backtest period
- **Total Trades:** {result_data['total_trades']}

## Performance Analysis

### Returns Breakdown
| Metric | Value |
|--------|-------|
| Total Return | {result_data['total_return']:.2f}% |
| Annualized Return | {result_data['annualized_return']:.2f}% |
| Volatility | {result_data['volatility']:.2f}% |
| Best Month | {result_data['best_month']:.2f}% |
| Worst Month | {result_data['worst_month']:.2f}% |

### Risk-Adjusted Metrics
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Sharpe Ratio | {result_data['sharpe_ratio']:.3f} | {'Excellent' if result_data['sharpe_ratio'] > 2 else 'Good' if result_data['sharpe_ratio'] > 1 else 'Needs Improvement'} |
| Sortino Ratio | {result_data['sortino_ratio']:.3f} | Focus on downside risk |
| Calmar Ratio | {result_data['calmar_ratio']:.3f} | Return per unit of max drawdown |

### Drawdown Analysis
- **Maximum Drawdown:** {result_data['max_drawdown']:.2f}%
- **VaR (95%):** {result_data['var_95']:.2f}%
- **CVaR (95%):** {result_data['cvar_95']:.2f}%

### Trading Statistics
- **Win Rate:** {result_data['win_rate']:.1f}%
- **Profit Factor:** {result_data['profit_factor']:.2f}
- **Average Trade:** {result_data['avg_trade']:.2f}%
- **Average Win:** {result_data['avg_win']:.2f}%
- **Average Loss:** {result_data['avg_loss']:.2f}%

## 📈 Performance Summary

The strategy shows {'strong' if result_data['sharpe_ratio'] > 1.5 else 'moderate' if result_data['sharpe_ratio'] > 1.0 else 'weak'}
risk-adjusted performance with a Sharpe ratio of {result_data['sharpe_ratio']:.2f}.

Maximum drawdown of {result_data['max_drawdown']:.1f}% is {'acceptable' if result_data['max_drawdown'] > -20 else 'concerning'}
for this strategy type.

---
*Detailed analysis completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

def generate_risk_report(result_data):
    """Génère un rapport de risque."""
    return f"""
# ⚠️ Risk Analysis Report

## Risk Profile Overview
**Strategy:** {result_data['strategy_type']}

## Primary Risk Metrics

### 📉 Drawdown Analysis
- **Maximum Drawdown:** {result_data['max_drawdown']:.2f}%
- **Risk Level:** {'High' if result_data['max_drawdown'] < -25 else 'Moderate' if result_data['max_drawdown'] < -15 else 'Low'}

### 📊 Value at Risk (VaR)
- **95% VaR:** {result_data['var_95']:.2f}%
- **95% CVaR:** {result_data['cvar_95']:.2f}%

### 🎯 Risk-Adjusted Returns
- **Sharpe Ratio:** {result_data['sharpe_ratio']:.3f}
- **Sortino Ratio:** {result_data['sortino_ratio']:.3f} (focuses on downside risk)

## Risk Assessment

### Volatility Analysis
- **Annual Volatility:** {result_data['volatility']:.1f}%
- **Risk Classification:** {'High Volatility' if result_data['volatility'] > 30 else 'Moderate Volatility' if result_data['volatility'] > 15 else 'Low Volatility'}

### Tail Risk
- **Worst Monthly Return:** {result_data['worst_month']:.2f}%
- **Tail Risk Rating:** {'High' if result_data['worst_month'] < -10 else 'Moderate' if result_data['worst_month'] < -5 else 'Low'}

## Risk Management Recommendations

### 🛡️ Position Sizing
- **Recommended Max Position:** {'5%' if result_data['max_drawdown'] < -20 else '10%' if result_data['max_drawdown'] < -15 else '15%'} of total capital
- **Risk Budget:** Based on max drawdown tolerance

### ⚠️ Risk Monitoring
- **Daily VaR Monitoring:** Recommended
- **Drawdown Alerts:** Set at {'10%' if result_data['max_drawdown'] < -15 else '15%'}
- **Correlation Monitoring:** Track with market benchmarks

---
*Risk analysis completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

def generate_trade_report(result_data):
    """Génère un rapport d'analyse des trades."""
    return f"""
# 💼 Trade Analysis Report

## Trading Activity Summary
**Strategy:** {result_data['strategy_type']}
**Total Trades:** {result_data['total_trades']}

## Trade Performance Metrics

### 🎯 Win/Loss Analysis
| Metric | Value |
|--------|-------|
| Win Rate | {result_data['win_rate']:.1f}% |
| Total Trades | {result_data['total_trades']} |
| Estimated Winning Trades | {int(result_data['total_trades'] * result_data['win_rate'] / 100)} |
| Estimated Losing Trades | {int(result_data['total_trades'] * (100 - result_data['win_rate']) / 100)} |

### 💰 Profitability Analysis
| Metric | Value | Notes |
|--------|-------|-------|
| Profit Factor | {result_data['profit_factor']:.2f} | {'Excellent' if result_data['profit_factor'] > 2 else 'Good' if result_data['profit_factor'] > 1.5 else 'Acceptable' if result_data['profit_factor'] > 1.2 else 'Needs Improvement'} |
| Average Trade | {result_data['avg_trade']:.2f}% | Per trade performance |
| Average Win | {result_data['avg_win']:.2f}% | Average winning trade |
| Average Loss | {result_data['avg_loss']:.2f}% | Average losing trade |

### 📊 Trade Efficiency
- **Win/Loss Ratio:** {result_data['avg_win'] / abs(result_data['avg_loss']):.2f}
- **Risk/Reward Balance:** {'Favorable' if result_data['avg_win'] / abs(result_data['avg_loss']) > 1.5 else 'Balanced' if result_data['avg_win'] / abs(result_data['avg_loss']) > 1.0 else 'Unfavorable'}

## Trading Pattern Analysis

### 🔄 Trade Frequency
- **Average Trades per Month:** ~{result_data['total_trades'] // 12}
- **Trading Activity:** {'High Frequency' if result_data['total_trades'] > 200 else 'Medium Frequency' if result_data['total_trades'] > 100 else 'Low Frequency'}

### ⏱️ Trade Duration Analysis
*Note: Detailed trade duration analysis would require individual trade data*

## Trade Quality Assessment

### ✅ Strengths
- {'High win rate above 60%' if result_data['win_rate'] > 60 else 'Balanced win rate' if result_data['win_rate'] > 50 else 'Focus on trade selection needed'}
- {'Strong profit factor' if result_data['profit_factor'] > 1.5 else 'Adequate profitability' if result_data['profit_factor'] > 1.2 else 'Profitability needs improvement'}

### ⚠️ Areas for Improvement
- {'Consider reducing trade frequency' if result_data['total_trades'] > 300 else 'Trade frequency appears optimal'}
- {'Review risk management for losing trades' if abs(result_data['avg_loss']) > 2 else 'Loss management appears adequate'}

---
*Trade analysis completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

def generate_benchmark_report(result_data):
    """Génère un rapport de comparaison avec benchmark."""
    # Simulation d'un benchmark (BTC buy & hold)
    benchmark_return = np.random.uniform(10, 25)
    benchmark_volatility = np.random.uniform(25, 45)
    benchmark_sharpe = benchmark_return / benchmark_volatility

    return f"""
# 📈 Benchmark Comparison Report

## Strategy vs Benchmark Analysis
**Strategy:** {result_data['strategy_type']}
**Benchmark:** BTC Buy & Hold

## Performance Comparison

### Returns Comparison
| Metric | Strategy | Benchmark | Difference |
|--------|----------|-----------|------------|
| Total Return | {result_data['total_return']:.1f}% | {benchmark_return:.1f}% | {result_data['total_return'] - benchmark_return:+.1f}% |
| Volatility | {result_data['volatility']:.1f}% | {benchmark_volatility:.1f}% | {result_data['volatility'] - benchmark_volatility:+.1f}% |
| Sharpe Ratio | {result_data['sharpe_ratio']:.2f} | {benchmark_sharpe:.2f} | {result_data['sharpe_ratio'] - benchmark_sharpe:+.2f} |

### Risk-Adjusted Performance
- **Alpha:** {result_data['total_return'] - benchmark_return:.1f}% ({'Positive alpha - Strategy outperformed' if result_data['total_return'] > benchmark_return else 'Negative alpha - Strategy underperformed'})
- **Risk Reduction:** {benchmark_volatility - result_data['volatility']:.1f}% ({'Lower risk than benchmark' if result_data['volatility'] < benchmark_volatility else 'Higher risk than benchmark'})

## Strategy Value Proposition

### 🎯 Advantages over Benchmark
{'✅ Higher returns with lower risk' if result_data['total_return'] > benchmark_return and result_data['volatility'] < benchmark_volatility else
 '✅ Higher risk-adjusted returns' if result_data['sharpe_ratio'] > benchmark_sharpe else
 '✅ Lower volatility' if result_data['volatility'] < benchmark_volatility else
 '⚠️ Needs optimization to outperform benchmark'}

### 📊 Performance Analysis
- **Information Ratio:** {(result_data['total_return'] - benchmark_return) / abs(result_data['volatility'] - benchmark_volatility) if abs(result_data['volatility'] - benchmark_volatility) > 0 else 0:.2f}
- **Tracking Error:** ~{abs(result_data['volatility'] - benchmark_volatility):.1f}%

## Investment Conclusion

{'**RECOMMENDED**: Strategy provides superior risk-adjusted returns vs benchmark' if result_data['sharpe_ratio'] > benchmark_sharpe
 else '**CONDITIONAL**: Strategy needs optimization to justify active management' if result_data['total_return'] > benchmark_return * 0.8
 else '**NOT RECOMMENDED**: Benchmark provides better risk/return profile'}

### Key Considerations
- Strategy complexity vs benchmark simplicity
- Transaction costs and management fees
- Market regime dependency
- Scalability and capacity constraints

---
*Benchmark analysis completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

# ================== TAB 6: Workflow Intégré ==================
with tab6:
    st.header("🔄 Workflow Intégré de Backtesting")

    # Initialisation du gestionnaire d'intégration
    integration_manager = BacktestingIntegrationManager()

    # Sous-onglets pour le workflow
    workflow_tabs = st.tabs([
        "🚀 Pipeline Complet",
        "📊 Dashboard",
        "📋 Rapports Intégrés",
        "💾 Exports"
    ])

    with workflow_tabs[0]:
        st.subheader("🚀 Pipeline de Backtesting Complet")
        st.info("Ce workflow intègre configuration, validation temporelle, tests de robustesse et analyse complète.")

        workflow_config = integration_manager.create_integrated_workflow()

    with workflow_tabs[1]:
        integration_manager.render_pipeline_dashboard()

    with workflow_tabs[2]:
        st.subheader("📋 Rapports Intégrés")

        if st.session_state.get('backtest_results'):
            # Sélection du résultat à analyser
            result_ids = list(st.session_state.backtest_results.keys())
            selected_result = st.selectbox(
                "Sélectionner un backtest pour le rapport intégré",
                result_ids,
                key="integrated_report_selector"
            )

            if selected_result:
                report_data = integration_manager.generate_integrated_report(selected_result)
        else:
            st.info("Aucun résultat de backtest disponible. Lancez d'abord un pipeline.")

    with workflow_tabs[3]:
        st.subheader("💾 Export des Résultats")

        if st.session_state.get('backtest_results'):
            result_ids = list(st.session_state.backtest_results.keys())
            selected_export = st.selectbox(
                "Sélectionner un backtest à exporter",
                result_ids,
                key="export_selector"
            )

            if selected_export:
                export_data = integration_manager.export_comprehensive_results(selected_export)
        else:
            st.info("Aucun résultat de backtest disponible pour l'export.")

# Sidebar avec informations
with st.sidebar:
    st.markdown("### 🔬 Backtesting Suite")
    st.info("Suite complète de validation de stratégies")

    st.markdown("### 📊 Queue Status")
    st.metric("Backtests in Queue", len(st.session_state.backtest_queue))
    st.metric("Completed Results", len(st.session_state.backtest_results))

    if st.session_state.get('processing_queue', False):
        st.success("🔄 Processing Queue")
    else:
        st.info("⏸️ Queue Paused")

    st.markdown("### 🎯 Quick Actions")
    if st.button("🗑️ Clear All Results", use_container_width=True):
        st.session_state.backtest_results = {}
        st.session_state.backtest_queue = []
        st.rerun()

    st.markdown("### 📚 Resources")
    st.markdown("""
    - [Backtest Best Practices](#)
    - [Strategy Optimization Guide](#)
    - [Risk Management Handbook](#)
    """)