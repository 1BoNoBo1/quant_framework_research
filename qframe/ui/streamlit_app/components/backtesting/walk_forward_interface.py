"""
Walk-Forward Interface - Composant pour analyse walk-forward
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple
import json

class WalkForwardInterface:
    """Interface pour analyse walk-forward (validation temporelle)."""

    def __init__(self):
        self.default_config = {
            'train_window': 252,  # 1 an
            'test_window': 63,    # 3 mois
            'step_size': 21,      # 1 mois
            'min_train_size': 100,
            'reoptimize': True,
            'purge_days': 1
        }

    def render_walk_forward_config(self):
        """Rendu de la configuration walk-forward."""
        st.subheader("🔄 Walk-Forward Analysis Configuration")

        with st.expander("Walk-Forward Settings", expanded=True):
            col_wf1, col_wf2, col_wf3 = st.columns(3)

            with col_wf1:
                st.markdown("### Window Configuration")

                train_window = st.number_input(
                    "Training Window (days)",
                    min_value=50,
                    max_value=1000,
                    value=self.default_config['train_window'],
                    step=10,
                    help="Taille de la fenêtre d'entraînement"
                )

                test_window = st.number_input(
                    "Testing Window (days)",
                    min_value=10,
                    max_value=365,
                    value=self.default_config['test_window'],
                    step=5,
                    help="Taille de la fenêtre de test"
                )

                step_size = st.number_input(
                    "Step Size (days)",
                    min_value=1,
                    max_value=test_window,
                    value=self.default_config['step_size'],
                    step=1,
                    help="Taille du pas entre fenêtres"
                )

            with col_wf2:
                st.markdown("### Advanced Options")

                min_train_size = st.number_input(
                    "Minimum Training Size",
                    min_value=50,
                    max_value=train_window,
                    value=self.default_config['min_train_size'],
                    help="Taille minimale d'entraînement"
                )

                purge_days = st.number_input(
                    "Purge Days",
                    min_value=0,
                    max_value=10,
                    value=self.default_config['purge_days'],
                    help="Jours de purge entre train/test"
                )

                reoptimize = st.checkbox(
                    "Reoptimize Parameters",
                    value=self.default_config['reoptimize'],
                    help="Réoptimiser les paramètres à chaque fenêtre"
                )

            with col_wf3:
                st.markdown("### Validation Methods")

                validation_methods = st.multiselect(
                    "Validation Methods",
                    [
                        "Anchored Walk-Forward",
                        "Rolling Walk-Forward",
                        "Time Series Split",
                        "Purged Cross-Validation"
                    ],
                    default=["Rolling Walk-Forward"],
                    help="Méthodes de validation temporelle"
                )

                optimization_metric = st.selectbox(
                    "Optimization Metric",
                    ["Sharpe Ratio", "Total Return", "Calmar Ratio", "Sortino Ratio"],
                    help="Métrique à optimiser en walk-forward"
                )

        return {
            'train_window': train_window,
            'test_window': test_window,
            'step_size': step_size,
            'min_train_size': min_train_size,
            'purge_days': purge_days,
            'reoptimize': reoptimize,
            'validation_methods': validation_methods,
            'optimization_metric': optimization_metric
        }

    def render_walk_forward_visualization(self, config: Dict, total_days: int = 1000):
        """Visualisation du schéma walk-forward."""
        st.subheader("📊 Walk-Forward Scheme Visualization")

        # Calcul des fenêtres
        windows = self._calculate_windows(config, total_days)

        if not windows:
            st.warning("Configuration invalide - aucune fenêtre générée")
            return

        # Graphique de visualisation
        fig = go.Figure()

        colors = ['#00ff88', '#ff6b6b', '#6b88ff', '#ffd93d']

        for i, window in enumerate(windows):
            color = colors[i % len(colors)]

            # Fenêtre d'entraînement
            fig.add_trace(go.Scatter(
                x=[window['train_start'], window['train_end']],
                y=[i, i],
                mode='lines',
                line=dict(color=color, width=8),
                name=f'Train {i+1}' if i == 0 else None,
                legendgroup='train',
                showlegend=i == 0,
                hovertemplate=f"<b>Train Window {i+1}</b><br>" +
                             f"Start: Day {window['train_start']}<br>" +
                             f"End: Day {window['train_end']}<br>" +
                             f"Duration: {window['train_end'] - window['train_start']} days<extra></extra>"
            ))

            # Gap de purge
            if config['purge_days'] > 0:
                fig.add_trace(go.Scatter(
                    x=[window['train_end'], window['test_start']],
                    y=[i, i],
                    mode='lines',
                    line=dict(color='gray', width=4, dash='dash'),
                    name='Purge' if i == 0 else None,
                    legendgroup='purge',
                    showlegend=i == 0,
                    hovertemplate=f"<b>Purge Period</b><br>" +
                                 f"Duration: {config['purge_days']} days<extra></extra>"
                ))

            # Fenêtre de test
            fig.add_trace(go.Scatter(
                x=[window['test_start'], window['test_end']],
                y=[i, i],
                mode='lines',
                line=dict(color=color, width=8, dash='dot'),
                name=f'Test {i+1}' if i == 0 else None,
                legendgroup='test',
                showlegend=i == 0,
                hovertemplate=f"<b>Test Window {i+1}</b><br>" +
                             f"Start: Day {window['test_start']}<br>" +
                             f"End: Day {window['test_end']}<br>" +
                             f"Duration: {window['test_end'] - window['test_start']} days<extra></extra>"
            ))

        fig.update_layout(
            title="Walk-Forward Analysis Scheme",
            xaxis_title="Trading Days",
            yaxis_title="Window Number",
            template='plotly_dark',
            height=400,
            yaxis=dict(tickmode='linear', tick0=0, dtick=1),
            hovermode='closest'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Statistiques du schéma
        st.markdown("### Walk-Forward Statistics")

        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

        with col_stat1:
            st.metric("Total Windows", len(windows))

        with col_stat2:
            total_test_days = sum(w['test_end'] - w['test_start'] for w in windows)
            st.metric("Total Test Days", total_test_days)

        with col_stat3:
            coverage = total_test_days / total_days * 100
            st.metric("Test Coverage", f"{coverage:.1f}%")

        with col_stat4:
            avg_train_size = np.mean([w['train_end'] - w['train_start'] for w in windows])
            st.metric("Avg Train Size", f"{avg_train_size:.0f} days")

    def _calculate_windows(self, config: Dict, total_days: int) -> List[Dict]:
        """Calcule les fenêtres walk-forward."""
        windows = []
        train_window = config['train_window']
        test_window = config['test_window']
        step_size = config['step_size']
        purge_days = config['purge_days']
        min_train_size = config['min_train_size']

        current_pos = 0

        while current_pos + train_window + purge_days + test_window <= total_days:
            train_start = current_pos
            train_end = current_pos + train_window
            test_start = train_end + purge_days
            test_end = test_start + test_window

            # Vérifier la taille minimale d'entraînement
            if train_end - train_start >= min_train_size:
                windows.append({
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end
                })

            current_pos += step_size

        return windows

    def render_walk_forward_execution(self, config: Dict, strategy_params: Dict):
        """Rendu de l'exécution walk-forward."""
        st.subheader("▶️ Walk-Forward Execution")

        if st.button("🚀 Start Walk-Forward Analysis", type="primary"):
            st.session_state['wf_running'] = True
            st.session_state['wf_progress'] = 0
            st.session_state['wf_results'] = []

        if st.session_state.get('wf_running', False):
            self._render_execution_progress(config)

    def _render_execution_progress(self, config: Dict):
        """Rendu du progrès d'exécution."""
        # Simulation du progrès
        if 'wf_progress' not in st.session_state:
            st.session_state.wf_progress = 0

        progress = st.session_state.wf_progress
        total_windows = 10  # Simulé

        # Progress bar
        progress_bar = st.progress(progress / 100)
        current_window = int(progress / 100 * total_windows) + 1

        status_col, metrics_col = st.columns([2, 1])

        with status_col:
            st.markdown(f"""
            **Status:** Running Walk-Forward Analysis
            **Current Window:** {current_window} / {total_windows}
            **Progress:** {progress:.1f}%
            **ETA:** {max(0, int((100 - progress) * 3))} seconds
            """)

        with metrics_col:
            if progress > 20:
                # Métriques intermédiaires simulées
                avg_oos_sharpe = np.random.uniform(0.5, 2.0)
                consistency = np.random.uniform(60, 90)
                degradation = np.random.uniform(-20, 5)

                st.metric("Avg OOS Sharpe", f"{avg_oos_sharpe:.2f}")
                st.metric("Consistency", f"{consistency:.0f}%")
                st.metric("IS vs OOS", f"{degradation:+.1f}%")

        # Simulation d'avancement
        if progress < 100:
            st.session_state.wf_progress += np.random.randint(2, 8)
            if st.session_state.wf_progress > 100:
                st.session_state.wf_progress = 100

        # Completion
        if progress >= 100:
            st.success("✅ Walk-Forward Analysis Completed!")
            st.session_state.wf_running = False

            # Générer des résultats simulés
            if 'wf_results' not in st.session_state:
                st.session_state.wf_results = self._generate_wf_results()

    def _generate_wf_results(self) -> Dict:
        """Génère des résultats walk-forward simulés."""
        num_windows = 10

        # Métriques par fenêtre
        is_sharpes = np.random.uniform(1.0, 3.0, num_windows)  # In-sample
        oos_sharpes = is_sharpes * np.random.uniform(0.6, 0.9, num_windows)  # Out-of-sample

        is_returns = np.random.uniform(0.05, 0.25, num_windows)
        oos_returns = is_returns * np.random.uniform(0.7, 1.1, num_windows)

        # Dates
        dates = pd.date_range(start='2023-01-01', periods=num_windows, freq='M')

        return {
            'num_windows': num_windows,
            'dates': dates.tolist(),
            'is_sharpes': is_sharpes.tolist(),
            'oos_sharpes': oos_sharpes.tolist(),
            'is_returns': is_returns.tolist(),
            'oos_returns': oos_returns.tolist(),
            'avg_is_sharpe': np.mean(is_sharpes),
            'avg_oos_sharpe': np.mean(oos_sharpes),
            'degradation': (np.mean(oos_sharpes) - np.mean(is_sharpes)) / np.mean(is_sharpes) * 100,
            'consistency': np.mean(oos_sharpes > 0) * 100,
            'best_window': np.argmax(oos_sharpes) + 1,
            'worst_window': np.argmin(oos_sharpes) + 1
        }

    def render_walk_forward_results(self, results: Dict):
        """Rendu des résultats walk-forward."""
        if not results:
            st.info("No walk-forward results available. Run analysis first.")
            return

        st.subheader("📊 Walk-Forward Results")

        # Métriques de résumé
        col_summary1, col_summary2, col_summary3, col_summary4 = st.columns(4)

        with col_summary1:
            st.metric("Avg OOS Sharpe", f"{results['avg_oos_sharpe']:.2f}")

        with col_summary2:
            st.metric("IS vs OOS", f"{results['degradation']:+.1f}%")

        with col_summary3:
            st.metric("Consistency", f"{results['consistency']:.0f}%")

        with col_summary4:
            st.metric("Total Windows", results['num_windows'])

        # Graphiques de performance
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            # Comparaison IS vs OOS Sharpe
            fig_sharpe = go.Figure()

            fig_sharpe.add_trace(go.Scatter(
                x=list(range(1, results['num_windows'] + 1)),
                y=results['is_sharpes'],
                mode='lines+markers',
                name='In-Sample Sharpe',
                line=dict(color='#00ff88', width=3),
                marker=dict(size=8)
            ))

            fig_sharpe.add_trace(go.Scatter(
                x=list(range(1, results['num_windows'] + 1)),
                y=results['oos_sharpes'],
                mode='lines+markers',
                name='Out-of-Sample Sharpe',
                line=dict(color='#ff6b6b', width=3),
                marker=dict(size=8)
            ))

            fig_sharpe.update_layout(
                title="In-Sample vs Out-of-Sample Sharpe Ratio",
                xaxis_title="Window",
                yaxis_title="Sharpe Ratio",
                template='plotly_dark',
                height=400
            )

            st.plotly_chart(fig_sharpe, use_container_width=True)

        with col_chart2:
            # Distribution des performances OOS
            fig_dist = go.Figure()

            fig_dist.add_trace(go.Histogram(
                x=results['oos_sharpes'],
                nbinsx=15,
                name='OOS Sharpe Distribution',
                marker_color='#6b88ff',
                opacity=0.7
            ))

            fig_dist.add_vline(
                x=results['avg_oos_sharpe'],
                line_dash="dash",
                line_color="#ff6b6b",
                annotation_text=f"Mean: {results['avg_oos_sharpe']:.2f}"
            )

            fig_dist.update_layout(
                title="Out-of-Sample Performance Distribution",
                xaxis_title="Sharpe Ratio",
                yaxis_title="Frequency",
                template='plotly_dark',
                height=400
            )

            st.plotly_chart(fig_dist, use_container_width=True)

        # Tableau détaillé des résultats
        st.subheader("📋 Detailed Window Results")

        window_data = []
        for i in range(results['num_windows']):
            window_data.append({
                'Window': i + 1,
                'Date': results['dates'][i].strftime('%Y-%m'),
                'IS Sharpe': f"{results['is_sharpes'][i]:.3f}",
                'OOS Sharpe': f"{results['oos_sharpes'][i]:.3f}",
                'IS Return': f"{results['is_returns'][i]:.1%}",
                'OOS Return': f"{results['oos_returns'][i]:.1%}",
                'Degradation': f"{(results['oos_sharpes'][i] - results['is_sharpes'][i]) / results['is_sharpes'][i] * 100:+.1f}%"
            })

        results_df = pd.DataFrame(window_data)

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "OOS Sharpe": st.column_config.NumberColumn(
                    "OOS Sharpe",
                    help="Out-of-sample Sharpe ratio",
                    format="%.3f",
                ),
                "Degradation": st.column_config.TextColumn(
                    "Degradation",
                    help="Performance degradation from IS to OOS"
                )
            }
        )

        # Analyse de robustesse
        self._render_robustness_analysis(results)

    def _render_robustness_analysis(self, results: Dict):
        """Rendu de l'analyse de robustesse."""
        st.subheader("🔍 Robustness Analysis")

        col_robust1, col_robust2, col_robust3 = st.columns(3)

        with col_robust1:
            st.markdown("### Performance Stability")

            # Coefficient de variation
            cv_oos = np.std(results['oos_sharpes']) / np.mean(results['oos_sharpes'])
            st.metric("Coefficient of Variation", f"{cv_oos:.3f}")

            # Pourcentage de fenêtres positives
            positive_windows = np.mean(np.array(results['oos_sharpes']) > 0) * 100
            st.metric("Positive Windows", f"{positive_windows:.0f}%")

        with col_robust2:
            st.markdown("### Overfitting Detection")

            # Dégradation moyenne
            degradation = results['degradation']
            st.metric("Avg Degradation", f"{degradation:.1f}%")

            # Test de significativité (simulé)
            t_stat = np.random.uniform(1.5, 3.5)
            st.metric("T-statistic", f"{t_stat:.2f}")

        with col_robust3:
            st.markdown("### Risk Assessment")

            # Worst case scenario
            worst_sharpe = min(results['oos_sharpes'])
            st.metric("Worst OOS Sharpe", f"{worst_sharpe:.3f}")

            # Drawdown periods (simulé)
            dd_periods = np.random.randint(2, 5)
            st.metric("Drawdown Periods", dd_periods)

        # Recommandations basées sur les résultats
        st.markdown("### 📋 Recommendations")

        if results['degradation'] > -10 and results['consistency'] > 70:
            st.success("✅ **ROBUST STRATEGY**: Low overfitting, consistent performance")
        elif results['degradation'] > -20 and results['consistency'] > 60:
            st.info("👍 **ACCEPTABLE**: Moderate degradation, reasonable consistency")
        elif results['degradation'] > -30:
            st.warning("⚠️ **NEEDS OPTIMIZATION**: Significant performance degradation")
        else:
            st.error("❌ **HIGH OVERFITTING RISK**: Strategy not recommended for live trading")

        # Actions recommandées
        recommendations = []

        if results['degradation'] < -15:
            recommendations.append("Consider reducing strategy complexity")
        if results['consistency'] < 60:
            recommendations.append("Improve parameter stability")
        if min(results['oos_sharpes']) < 0:
            recommendations.append("Add risk management constraints")

        if recommendations:
            st.markdown("**Suggested Improvements:**")
            for rec in recommendations:
                st.markdown(f"• {rec}")

    def render_sensitivity_analysis(self, base_config: Dict):
        """Rendu de l'analyse de sensibilité des paramètres WF."""
        st.subheader("🎛️ Parameter Sensitivity Analysis")

        # Paramètres à tester
        param_ranges = {
            'train_window': [180, 252, 365, 500],
            'test_window': [30, 63, 90, 126],
            'step_size': [7, 21, 30, 63]
        }

        # Sélection du paramètre à analyser
        selected_param = st.selectbox(
            "Parameter to Analyze",
            list(param_ranges.keys()),
            format_func=lambda x: x.replace('_', ' ').title()
        )

        if st.button("Run Sensitivity Analysis"):
            # Simulation de l'analyse de sensibilité
            param_values = param_ranges[selected_param]
            oos_sharpes = []
            degradations = []

            for value in param_values:
                # Simuler des résultats pour chaque valeur
                oos_sharpe = np.random.uniform(0.8, 2.2)
                degradation = np.random.uniform(-30, -5)

                oos_sharpes.append(oos_sharpe)
                degradations.append(degradation)

            # Graphique de sensibilité
            fig_sens = go.Figure()

            fig_sens.add_trace(go.Scatter(
                x=param_values,
                y=oos_sharpes,
                mode='lines+markers',
                name='OOS Sharpe Ratio',
                line=dict(color='#00ff88', width=3),
                marker=dict(size=10)
            ))

            fig_sens.update_layout(
                title=f"Sensitivity to {selected_param.replace('_', ' ').title()}",
                xaxis_title=selected_param.replace('_', ' ').title(),
                yaxis_title="Out-of-Sample Sharpe Ratio",
                template='plotly_dark',
                height=400
            )

            st.plotly_chart(fig_sens, use_container_width=True)

            # Optimal value
            optimal_idx = np.argmax(oos_sharpes)
            optimal_value = param_values[optimal_idx]

            st.success(f"Optimal {selected_param}: {optimal_value}")

    def export_walk_forward_report(self, results: Dict, config: Dict) -> str:
        """Exporte un rapport walk-forward."""
        report = f"""
# Walk-Forward Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration
- Training Window: {config['train_window']} days
- Testing Window: {config['test_window']} days
- Step Size: {config['step_size']} days
- Purge Days: {config['purge_days']} days

## Summary Results
- Average OOS Sharpe Ratio: {results['avg_oos_sharpe']:.3f}
- Performance Degradation: {results['degradation']:.1f}%
- Consistency: {results['consistency']:.0f}%
- Total Windows: {results['num_windows']}

## Robustness Assessment
- Coefficient of Variation: {np.std(results['oos_sharpes']) / np.mean(results['oos_sharpes']):.3f}
- Positive Windows: {np.mean(np.array(results['oos_sharpes']) > 0) * 100:.0f}%
- Worst OOS Sharpe: {min(results['oos_sharpes']):.3f}

## Recommendation
"""

        if results['degradation'] > -10 and results['consistency'] > 70:
            report += "ROBUST STRATEGY - Recommended for live trading"
        elif results['degradation'] > -20:
            report += "ACCEPTABLE - Consider minor optimizations"
        else:
            report += "HIGH OVERFITTING RISK - Not recommended for live trading"

        return report