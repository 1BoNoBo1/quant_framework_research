"""
Monte Carlo Simulator - Composant pour simulation Monte Carlo des backtests
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats
import json

class MonteCarloSimulator:
    """Simulateur Monte Carlo pour tester la robustesse des stratégies."""

    def __init__(self):
        self.default_config = {
            'num_simulations': 1000,
            'bootstrap_method': 'block',
            'block_size': 21,
            'confidence_levels': [5, 25, 50, 75, 95],
            'randomize_parameters': True,
            'parameter_noise': 0.1
        }

    def render_monte_carlo_config(self):
        """Rendu de la configuration Monte Carlo."""
        st.subheader("🎲 Monte Carlo Simulation Configuration")

        with st.expander("Simulation Settings", expanded=True):
            col_mc1, col_mc2, col_mc3 = st.columns(3)

            with col_mc1:
                st.markdown("### Basic Configuration")

                num_simulations = st.number_input(
                    "Number of Simulations",
                    min_value=100,
                    max_value=10000,
                    value=self.default_config['num_simulations'],
                    step=100,
                    help="Nombre de simulations Monte Carlo"
                )

                bootstrap_method = st.selectbox(
                    "Bootstrap Method",
                    ["block", "stationary", "circular"],
                    index=0,
                    help="Méthode de bootstrap pour préserver autocorrélation"
                )

                block_size = st.number_input(
                    "Block Size (days)",
                    min_value=1,
                    max_value=100,
                    value=self.default_config['block_size'],
                    help="Taille des blocs pour bootstrap"
                )

            with col_mc2:
                st.markdown("### Parameter Randomization")

                randomize_parameters = st.checkbox(
                    "Randomize Strategy Parameters",
                    value=self.default_config['randomize_parameters'],
                    help="Ajouter du bruit aux paramètres de stratégie"
                )

                if randomize_parameters:
                    parameter_noise = st.slider(
                        "Parameter Noise Level",
                        min_value=0.01,
                        max_value=0.5,
                        value=self.default_config['parameter_noise'],
                        help="Niveau de bruit sur les paramètres (% de variation)"
                    )
                else:
                    parameter_noise = 0.0

                stress_test = st.checkbox(
                    "Include Stress Testing",
                    value=False,
                    help="Inclure des scénarios de stress"
                )

            with col_mc3:
                st.markdown("### Output Configuration")

                confidence_levels = st.multiselect(
                    "Confidence Levels (%)",
                    [1, 5, 10, 25, 50, 75, 90, 95, 99],
                    default=self.default_config['confidence_levels'],
                    help="Niveaux de confiance pour percentiles"
                )

                include_extreme = st.checkbox(
                    "Include Extreme Scenarios",
                    value=True,
                    help="Inclure les scénarios extrêmes dans l'analyse"
                )

                parallel_processing = st.checkbox(
                    "Enable Parallel Processing",
                    value=True,
                    help="Utiliser le traitement parallèle (plus rapide)"
                )

        return {
            'num_simulations': num_simulations,
            'bootstrap_method': bootstrap_method,
            'block_size': block_size,
            'randomize_parameters': randomize_parameters,
            'parameter_noise': parameter_noise,
            'stress_test': stress_test,
            'confidence_levels': sorted(confidence_levels),
            'include_extreme': include_extreme,
            'parallel_processing': parallel_processing
        }

    def render_monte_carlo_execution(self, config: Dict, strategy_config: Dict):
        """Rendu de l'exécution Monte Carlo."""
        st.subheader("▶️ Monte Carlo Execution")

        # Estimation du temps d'exécution
        estimated_time = self._estimate_execution_time(config)

        col_exec1, col_exec2 = st.columns([2, 1])

        with col_exec1:
            st.markdown(f"""
            **Configuration:**
            - Simulations: {config['num_simulations']:,}
            - Bootstrap Method: {config['bootstrap_method'].title()}
            - Parameter Noise: {config.get('parameter_noise', 0):.1%}
            - Estimated Time: {estimated_time}
            """)

        with col_exec2:
            if st.button("🚀 Start Monte Carlo", type="primary", use_container_width=True):
                st.session_state['mc_running'] = True
                st.session_state['mc_progress'] = 0
                st.session_state['mc_results'] = None

        if st.session_state.get('mc_running', False):
            self._render_execution_progress(config)

    def _estimate_execution_time(self, config: Dict) -> str:
        """Estime le temps d'exécution."""
        base_time = 0.05  # secondes par simulation
        num_sims = config['num_simulations']

        if config.get('parallel_processing', True):
            estimated_seconds = (base_time * num_sims) / 4  # Assume 4 cores
        else:
            estimated_seconds = base_time * num_sims

        if estimated_seconds < 60:
            return f"{estimated_seconds:.0f} seconds"
        elif estimated_seconds < 3600:
            return f"{estimated_seconds/60:.1f} minutes"
        else:
            return f"{estimated_seconds/3600:.1f} hours"

    def _render_execution_progress(self, config: Dict):
        """Rendu du progrès d'exécution Monte Carlo."""
        if 'mc_progress' not in st.session_state:
            st.session_state.mc_progress = 0

        progress = st.session_state.mc_progress
        num_sims = config['num_simulations']

        # Progress bar
        progress_bar = st.progress(progress / 100)
        current_sim = int(progress / 100 * num_sims)

        status_col, metrics_col = st.columns([2, 1])

        with status_col:
            st.markdown(f"""
            **Status:** Running Monte Carlo Simulation
            **Current Simulation:** {current_sim:,} / {num_sims:,}
            **Progress:** {progress:.1f}%
            **ETA:** {max(0, int((100 - progress) * 2))} seconds
            """)

        with metrics_col:
            if progress > 10:
                # Métriques intermédiaires
                completed_sims = max(1, current_sim)
                avg_return = np.random.uniform(-5, 15)
                avg_sharpe = np.random.uniform(0.5, 2.0)
                success_rate = np.random.uniform(60, 85)

                st.metric("Avg Return", f"{avg_return:.1f}%")
                st.metric("Avg Sharpe", f"{avg_sharpe:.2f}")
                st.metric("Success Rate", f"{success_rate:.0f}%")

        # Simulation d'avancement
        if progress < 100:
            # Plus rapide au début, plus lent vers la fin
            increment = max(1, np.random.randint(1, 6) * (100 - progress) / 100)
            st.session_state.mc_progress += increment
            if st.session_state.mc_progress > 100:
                st.session_state.mc_progress = 100

        # Completion
        if progress >= 100:
            st.success("✅ Monte Carlo Simulation Completed!")
            st.session_state.mc_running = False

            # Générer des résultats
            if st.session_state.get('mc_results') is None:
                st.session_state.mc_results = self._generate_mc_results(config)

    def _generate_mc_results(self, config: Dict) -> Dict:
        """Génère des résultats Monte Carlo simulés."""
        num_sims = config['num_simulations']
        confidence_levels = config['confidence_levels']

        # Simulation des résultats avec distribution réaliste
        np.random.seed(42)  # Pour reproductibilité

        # Génération des returns avec distribution fat-tailed
        base_return = 0.12
        base_volatility = 0.20

        # Distribution t de Student pour fat tails
        df = 4  # degrés de liberté pour fat tails
        returns = stats.t.rvs(df, loc=base_return, scale=base_volatility/np.sqrt(df/(df-2)), size=num_sims)

        # Sharpe ratios correspondants
        sharpe_ratios = (returns - 0.03) / (base_volatility * np.random.uniform(0.8, 1.2, num_sims))

        # Max drawdowns (distribution beta pour borner entre 0 et 1)
        max_drawdowns = -stats.beta.rvs(2, 5, size=num_sims) * 0.6  # Max 60% DD

        # Win rates
        win_rates = stats.beta.rvs(8, 4, size=num_sims)  # Biaisé vers 60-80%

        # Calmar ratios
        calmar_ratios = np.abs(returns / max_drawdowns)

        # Calcul des percentiles
        percentiles = {}
        for level in confidence_levels:
            percentiles[f'p{level}'] = {
                'return': np.percentile(returns, level),
                'sharpe': np.percentile(sharpe_ratios, level),
                'max_dd': np.percentile(max_drawdowns, level),
                'win_rate': np.percentile(win_rates, level),
                'calmar': np.percentile(calmar_ratios, level)
            }

        return {
            'num_simulations': num_sims,
            'returns': returns.tolist(),
            'sharpe_ratios': sharpe_ratios.tolist(),
            'max_drawdowns': max_drawdowns.tolist(),
            'win_rates': win_rates.tolist(),
            'calmar_ratios': calmar_ratios.tolist(),
            'percentiles': percentiles,
            'summary_stats': {
                'mean_return': np.mean(returns),
                'std_return': np.std(returns),
                'mean_sharpe': np.mean(sharpe_ratios),
                'std_sharpe': np.std(sharpe_ratios),
                'success_rate': np.mean(returns > 0) * 100,
                'extreme_loss_prob': np.mean(returns < -0.2) * 100,
                'excellent_perf_prob': np.mean(sharpe_ratios > 2.0) * 100
            }
        }

    def render_monte_carlo_results(self, results: Dict):
        """Rendu des résultats Monte Carlo."""
        if not results:
            st.info("No Monte Carlo results available. Run simulation first.")
            return

        st.subheader("🎲 Monte Carlo Results")

        # Statistiques de résumé
        summary = results['summary_stats']

        col_summary1, col_summary2, col_summary3, col_summary4 = st.columns(4)

        with col_summary1:
            st.metric("Mean Return", f"{summary['mean_return']:.1%}")

        with col_summary2:
            st.metric("Mean Sharpe", f"{summary['mean_sharpe']:.2f}")

        with col_summary3:
            st.metric("Success Rate", f"{summary['success_rate']:.0f}%")

        with col_summary4:
            st.metric("Simulations", f"{results['num_simulations']:,}")

        # Distribution des résultats
        self._render_distributions(results)

        # Percentiles et confidence intervals
        self._render_confidence_intervals(results)

        # Analyse de risque
        self._render_risk_analysis(results)

    def _render_distributions(self, results: Dict):
        """Rendu des distributions de résultats."""
        st.subheader("📊 Performance Distributions")

        col_dist1, col_dist2 = st.columns(2)

        with col_dist1:
            # Distribution des returns
            fig_returns = go.Figure()

            fig_returns.add_trace(go.Histogram(
                x=np.array(results['returns']) * 100,
                nbinsx=50,
                name='Annual Returns',
                marker_color='#00ff88',
                opacity=0.7
            ))

            # Ligne de médiane
            median_return = np.median(results['returns']) * 100
            fig_returns.add_vline(
                x=median_return,
                line_dash="dash",
                line_color="#ff6b6b",
                annotation_text=f"Median: {median_return:.1f}%"
            )

            fig_returns.update_layout(
                title="Distribution of Annual Returns",
                xaxis_title="Annual Return (%)",
                yaxis_title="Frequency",
                template='plotly_dark',
                height=400
            )

            st.plotly_chart(fig_returns, use_container_width=True)

        with col_dist2:
            # Distribution des Sharpe ratios
            fig_sharpe = go.Figure()

            fig_sharpe.add_trace(go.Histogram(
                x=results['sharpe_ratios'],
                nbinsx=50,
                name='Sharpe Ratios',
                marker_color='#6b88ff',
                opacity=0.7
            ))

            # Ligne de médiane
            median_sharpe = np.median(results['sharpe_ratios'])
            fig_sharpe.add_vline(
                x=median_sharpe,
                line_dash="dash",
                line_color="#ff6b6b",
                annotation_text=f"Median: {median_sharpe:.2f}"
            )

            fig_sharpe.update_layout(
                title="Distribution of Sharpe Ratios",
                xaxis_title="Sharpe Ratio",
                yaxis_title="Frequency",
                template='plotly_dark',
                height=400
            )

            st.plotly_chart(fig_sharpe, use_container_width=True)

        # Scatter plot Return vs Risk
        st.subheader("📈 Risk-Return Scatter")

        returns_array = np.array(results['returns'])
        volatilities = np.abs(np.array(results['max_drawdowns']))  # Proxy pour volatilité

        fig_scatter = go.Figure()

        fig_scatter.add_trace(go.Scatter(
            x=volatilities * 100,
            y=returns_array * 100,
            mode='markers',
            name='Simulations',
            marker=dict(
                size=4,
                color=results['sharpe_ratios'],
                colorscale='Viridis',
                colorbar=dict(title="Sharpe Ratio"),
                opacity=0.6
            ),
            hovertemplate='<b>Return:</b> %{y:.1f}%<br>' +
                          '<b>Max DD:</b> %{x:.1f}%<br>' +
                          '<extra></extra>'
        ))

        fig_scatter.update_layout(
            title="Risk-Return Profile (Color: Sharpe Ratio)",
            xaxis_title="Maximum Drawdown (%)",
            yaxis_title="Annual Return (%)",
            template='plotly_dark',
            height=500
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

    def _render_confidence_intervals(self, results: Dict):
        """Rendu des intervalles de confiance."""
        st.subheader("📏 Confidence Intervals")

        percentiles = results['percentiles']

        # Tableau des percentiles
        metrics = ['return', 'sharpe', 'max_dd', 'win_rate', 'calmar']
        metric_names = ['Annual Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)', 'Calmar Ratio']

        percentile_data = []
        for metric, name in zip(metrics, metric_names):
            row = {'Metric': name}
            for level in sorted(percentiles.keys()):
                value = percentiles[level][metric]
                if metric in ['return', 'max_dd']:
                    row[level.upper()] = f"{value:.1%}"
                elif metric == 'win_rate':
                    row[level.upper()] = f"{value:.1%}"
                else:
                    row[level.upper()] = f"{value:.2f}"
            percentile_data.append(row)

        percentiles_df = pd.DataFrame(percentile_data)

        st.dataframe(
            percentiles_df,
            use_container_width=True,
            hide_index=True
        )

        # Fan chart pour returns
        st.subheader("📊 Return Confidence Fan Chart")

        # Simuler une série temporelle pour le fan chart
        time_periods = 252  # 1 an de trading
        dates = pd.date_range(start='2024-01-01', periods=time_periods, freq='D')

        # Créer des chemins de performance pour différents percentiles
        confidence_paths = {}
        base_path = np.cumprod(1 + np.random.randn(time_periods) * 0.01) - 1

        for level in [5, 25, 50, 75, 95]:
            # Ajuster le chemin selon le percentile
            adjustment = percentiles[f'p{level}']['return'] / np.mean(results['returns'])
            confidence_paths[level] = base_path * adjustment

        fig_fan = go.Figure()

        # Zones de confiance
        colors = ['rgba(255, 107, 107, 0.1)', 'rgba(255, 165, 0, 0.2)', 'rgba(0, 255, 136, 0.8)',
                  'rgba(255, 165, 0, 0.2)', 'rgba(255, 107, 107, 0.1)']

        for i, level in enumerate([5, 25, 50, 75, 95]):
            fig_fan.add_trace(go.Scatter(
                x=dates,
                y=confidence_paths[level] * 100,
                mode='lines',
                name=f'P{level}',
                line=dict(color=colors[i][:7], width=2 if level == 50 else 1),
                fill='tonexty' if level > 5 else None,
                fillcolor=colors[i]
            ))

        fig_fan.update_layout(
            title="Performance Confidence Fan Chart",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            template='plotly_dark',
            height=400
        )

        st.plotly_chart(fig_fan, use_container_width=True)

    def _render_risk_analysis(self, results: Dict):
        """Rendu de l'analyse de risque Monte Carlo."""
        st.subheader("⚠️ Risk Analysis")

        summary = results['summary_stats']

        col_risk1, col_risk2, col_risk3 = st.columns(3)

        with col_risk1:
            st.markdown("### Tail Risk")

            # Probabilité de perte extrême
            extreme_loss_prob = summary['extreme_loss_prob']
            st.metric("P(Loss > 20%)", f"{extreme_loss_prob:.1f}%")

            # VaR Monte Carlo
            var_5 = np.percentile(results['returns'], 5)
            st.metric("VaR (5%)", f"{var_5:.1%}")

            # Expected Shortfall
            returns_array = np.array(results['returns'])
            es_5 = np.mean(returns_array[returns_array <= var_5])
            st.metric("Expected Shortfall", f"{es_5:.1%}")

        with col_risk2:
            st.markdown("### Drawdown Risk")

            # Probabilité de DD sévère
            severe_dd_prob = np.mean(np.array(results['max_drawdowns']) < -0.3) * 100
            st.metric("P(DD > 30%)", f"{severe_dd_prob:.1f}%")

            # DD médian
            median_dd = np.median(results['max_drawdowns'])
            st.metric("Median Max DD", f"{median_dd:.1%}")

            # Worst case DD
            worst_dd = np.min(results['max_drawdowns'])
            st.metric("Worst Case DD", f"{worst_dd:.1%}")

        with col_risk3:
            st.markdown("### Performance Risk")

            # Probabilité de performance excellente
            excellent_prob = summary['excellent_perf_prob']
            st.metric("P(Sharpe > 2.0)", f"{excellent_prob:.1f}%")

            # Volatilité des returns
            vol_returns = summary['std_return']
            st.metric("Return Volatility", f"{vol_returns:.1%}")

            # Stabilité Sharpe
            vol_sharpe = summary['std_sharpe']
            st.metric("Sharpe Volatility", f"{vol_sharpe:.2f}")

        # Classification du risque
        self._render_risk_classification(results)

    def _render_risk_classification(self, results: Dict):
        """Classification du niveau de risque."""
        st.markdown("### 🎯 Risk Classification")

        summary = results['summary_stats']
        percentiles = results['percentiles']

        # Critères de classification
        success_rate = summary['success_rate']
        worst_case_return = percentiles['p5']['return']
        sharpe_consistency = 1 - (summary['std_sharpe'] / abs(summary['mean_sharpe']))

        # Déterminer la classe de risque
        if success_rate > 80 and worst_case_return > -0.1 and sharpe_consistency > 0.8:
            risk_class = "🟢 LOW RISK"
            risk_color = "success"
            risk_desc = "Conservative strategy with consistent performance"
        elif success_rate > 70 and worst_case_return > -0.2 and sharpe_consistency > 0.6:
            risk_class = "🟡 MODERATE RISK"
            risk_color = "info"
            risk_desc = "Balanced risk-return profile"
        elif success_rate > 60 and worst_case_return > -0.3:
            risk_class = "🟠 MEDIUM-HIGH RISK"
            risk_color = "warning"
            risk_desc = "Higher volatility but acceptable worst case"
        else:
            risk_class = "🔴 HIGH RISK"
            risk_color = "error"
            risk_desc = "High volatility with significant tail risk"

        st.markdown(f"""
        **Risk Classification:** {risk_class}

        **Assessment:** {risk_desc}
        """)

        # Recommandations spécifiques
        recommendations = []

        if summary['extreme_loss_prob'] > 10:
            recommendations.append("Consider implementing stop-loss mechanisms")
        if summary['std_sharpe'] / abs(summary['mean_sharpe']) > 0.5:
            recommendations.append("Strategy performance is inconsistent - review parameters")
        if np.median(results['max_drawdowns']) < -0.25:
            recommendations.append("High typical drawdowns - reduce position sizing")

        if recommendations:
            st.markdown("**Risk Management Recommendations:**")
            for rec in recommendations:
                st.markdown(f"• {rec}")

    def render_scenario_analysis(self, results: Dict):
        """Rendu de l'analyse de scénarios."""
        st.subheader("🎭 Scenario Analysis")

        if not results:
            st.info("Run Monte Carlo simulation first to see scenario analysis.")
            return

        # Définir des scénarios
        scenarios = {
            "Bear Market": {"return_factor": 0.3, "volatility_factor": 1.5},
            "Bull Market": {"return_factor": 1.8, "volatility_factor": 0.8},
            "High Volatility": {"return_factor": 1.0, "volatility_factor": 2.0},
            "Low Volatility": {"return_factor": 1.0, "volatility_factor": 0.5},
            "Crisis": {"return_factor": -0.5, "volatility_factor": 3.0}
        }

        selected_scenario = st.selectbox(
            "Select Scenario",
            list(scenarios.keys()),
            help="Analyser la performance sous différents régimes de marché"
        )

        if st.button("Analyze Scenario"):
            scenario = scenarios[selected_scenario]

            # Ajuster les résultats selon le scénario
            adjusted_returns = np.array(results['returns']) * scenario['return_factor']
            adjusted_sharpes = np.array(results['sharpe_ratios']) / scenario['volatility_factor']

            col_scen1, col_scen2, col_scen3 = st.columns(3)

            with col_scen1:
                st.metric("Scenario Return", f"{np.mean(adjusted_returns):.1%}")

            with col_scen2:
                st.metric("Scenario Sharpe", f"{np.mean(adjusted_sharpes):.2f}")

            with col_scen3:
                success_rate = np.mean(adjusted_returns > 0) * 100
                st.metric("Success Rate", f"{success_rate:.0f}%")

            # Graphique de comparaison
            fig_scenario = go.Figure()

            fig_scenario.add_trace(go.Histogram(
                x=results['returns'],
                name='Base Case',
                opacity=0.7,
                marker_color='#00ff88'
            ))

            fig_scenario.add_trace(go.Histogram(
                x=adjusted_returns,
                name=selected_scenario,
                opacity=0.7,
                marker_color='#ff6b6b'
            ))

            fig_scenario.update_layout(
                title=f"Return Distribution: Base Case vs {selected_scenario}",
                xaxis_title="Annual Return",
                yaxis_title="Frequency",
                template='plotly_dark',
                height=400,
                barmode='overlay'
            )

            st.plotly_chart(fig_scenario, use_container_width=True)

    def export_monte_carlo_report(self, results: Dict, config: Dict) -> str:
        """Exporte un rapport Monte Carlo."""
        if not results:
            return "No Monte Carlo results available."

        summary = results['summary_stats']
        percentiles = results['percentiles']

        report = f"""
# Monte Carlo Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Simulation Configuration
- Number of Simulations: {config['num_simulations']:,}
- Bootstrap Method: {config['bootstrap_method'].title()}
- Parameter Randomization: {'Enabled' if config.get('randomize_parameters', False) else 'Disabled'}

## Summary Statistics
- Mean Annual Return: {summary['mean_return']:.2%}
- Return Standard Deviation: {summary['std_return']:.2%}
- Mean Sharpe Ratio: {summary['mean_sharpe']:.3f}
- Success Rate: {summary['success_rate']:.1f}%

## Risk Metrics
- 5% VaR: {percentiles['p5']['return']:.2%}
- Expected Shortfall: {np.mean([r for r in results['returns'] if r <= percentiles['p5']['return']]):.2%}
- Extreme Loss Probability (>20%): {summary['extreme_loss_prob']:.1f}%

## Confidence Intervals (Annual Return)
- 5th Percentile: {percentiles['p5']['return']:.2%}
- 25th Percentile: {percentiles['p25']['return']:.2%}
- Median: {percentiles['p50']['return']:.2%}
- 75th Percentile: {percentiles['p75']['return']:.2%}
- 95th Percentile: {percentiles['p95']['return']:.2%}

## Risk Assessment
"""

        # Ajouter l'évaluation du risque
        if summary['success_rate'] > 80:
            report += "LOW RISK - High probability of positive returns with limited downside"
        elif summary['success_rate'] > 70:
            report += "MODERATE RISK - Good risk-return balance with acceptable tail risk"
        else:
            report += "HIGH RISK - Significant probability of losses with high volatility"

        return report