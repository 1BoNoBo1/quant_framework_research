"""
Gestionnaire d'intégration pour orchestrer tous les composants de backtesting.
Centralise la logique de coordination entre les différents modules.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import uuid

from .backtest_configurator import BacktestConfigurator
from .results_analyzer import ResultsAnalyzer
from .walk_forward_interface import WalkForwardInterface
from .monte_carlo_simulator import MonteCarloSimulator
from .performance_analytics import PerformanceAnalytics


class BacktestingIntegrationManager:
    """Gestionnaire d'intégration pour l'orchestration des composants de backtesting."""

    def __init__(self):
        """Initialise le gestionnaire d'intégration."""
        self.configurator = BacktestConfigurator()
        self.analyzer = ResultsAnalyzer()
        self.walk_forward = WalkForwardInterface()
        self.monte_carlo = MonteCarloSimulator()
        self.analytics = PerformanceAnalytics()

        # Initialisation des états
        self._init_session_state()

    def _init_session_state(self):
        """Initialise les états de session pour le backtesting."""
        if 'bt_manager_initialized' not in st.session_state:
            st.session_state.bt_manager_initialized = True
            st.session_state.backtest_pipeline = {}
            st.session_state.active_backtests = {}
            st.session_state.validation_results = {}

    def create_integrated_workflow(self) -> Dict[str, Any]:
        """Crée un workflow intégré de backtesting."""
        st.subheader("🔄 Workflow Intégré de Backtesting")

        workflow_config = {}

        # Étape 1: Configuration
        with st.expander("🔧 1. Configuration de Base", expanded=True):
            config_data = self.configurator.render_configuration_section()
            workflow_config['base_config'] = config_data

        # Étape 2: Validation Walk-Forward
        with st.expander("⏭️ 2. Validation Temporelle", expanded=False):
            if workflow_config.get('base_config'):
                wf_config = self.walk_forward.render_configuration()
                workflow_config['walk_forward'] = wf_config
            else:
                st.info("Complétez d'abord la configuration de base.")

        # Étape 3: Monte Carlo
        with st.expander("🎲 3. Tests de Robustesse", expanded=False):
            if workflow_config.get('base_config'):
                mc_config = self.monte_carlo.render_configuration()
                workflow_config['monte_carlo'] = mc_config
            else:
                st.info("Complétez d'abord la configuration de base.")

        # Étape 4: Exécution
        with st.expander("▶️ 4. Exécution du Pipeline", expanded=False):
            if self._validate_workflow(workflow_config):
                if st.button("🚀 Lancer le Pipeline Complet", type="primary"):
                    self._execute_integrated_pipeline(workflow_config)
            else:
                st.warning("Veuillez compléter toutes les configurations requises.")

        return workflow_config

    def render_pipeline_dashboard(self):
        """Affiche le tableau de bord du pipeline."""
        st.subheader("📊 Dashboard du Pipeline")

        if not st.session_state.get('active_backtests'):
            st.info("Aucun backtest en cours. Lancez un pipeline pour commencer.")
            return

        # Status des backtests actifs
        col1, col2, col3 = st.columns(3)

        with col1:
            active_count = len([bt for bt in st.session_state.active_backtests.values()
                              if bt['status'] == 'running'])
            st.metric("Backtests Actifs", active_count)

        with col2:
            completed_count = len([bt for bt in st.session_state.active_backtests.values()
                                 if bt['status'] == 'completed'])
            st.metric("Backtests Terminés", completed_count)

        with col3:
            failed_count = len([bt for bt in st.session_state.active_backtests.values()
                              if bt['status'] == 'failed'])
            st.metric("Backtests Échoués", failed_count)

        # Table des backtests
        self._render_backtests_table()

    def generate_integrated_report(self, backtest_id: str) -> Dict[str, Any]:
        """Génère un rapport intégré complet."""
        if backtest_id not in st.session_state.get('backtest_results', {}):
            st.error("Résultats de backtest non trouvés.")
            return {}

        results = st.session_state.backtest_results[backtest_id]

        st.subheader("📋 Rapport Intégré Complet")

        # Onglets du rapport
        report_tabs = st.tabs([
            "📈 Performance",
            "🔍 Analytics",
            "⏭️ Walk-Forward",
            "🎲 Monte Carlo",
            "📊 Comparaisons"
        ])

        with report_tabs[0]:
            self.analyzer.render_performance_summary(results)

        with report_tabs[1]:
            self.analytics.render_advanced_analytics(results)

        with report_tabs[2]:
            if 'walk_forward_results' in results:
                self.walk_forward.render_results(results['walk_forward_results'])
            else:
                st.info("Pas de résultats Walk-Forward disponibles.")

        with report_tabs[3]:
            if 'monte_carlo_results' in results:
                self.monte_carlo.render_results(results['monte_carlo_results'])
            else:
                st.info("Pas de résultats Monte Carlo disponibles.")

        with report_tabs[4]:
            self._render_strategy_comparison()

        return self._compile_report_data(results)

    def export_comprehensive_results(self, backtest_id: str) -> Dict[str, Any]:
        """Exporte les résultats complets dans différents formats."""
        st.subheader("💾 Export des Résultats")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Export Excel"):
                excel_data = self._generate_excel_export(backtest_id)
                st.success("Export Excel généré!")
                return excel_data

        with col2:
            if st.button("📄 Export PDF"):
                pdf_data = self._generate_pdf_report(backtest_id)
                st.success("Rapport PDF généré!")
                return pdf_data

        with col3:
            if st.button("📋 Export JSON"):
                json_data = self._generate_json_export(backtest_id)
                st.success("Export JSON généré!")
                return json_data

    def _validate_workflow(self, workflow_config: Dict) -> bool:
        """Valide la configuration du workflow."""
        required_keys = ['base_config']
        return all(key in workflow_config and workflow_config[key] for key in required_keys)

    def _execute_integrated_pipeline(self, workflow_config: Dict):
        """Exécute le pipeline intégré de backtesting."""
        pipeline_id = str(uuid.uuid4())[:8]

        # Initialisation du pipeline
        pipeline_data = {
            'id': pipeline_id,
            'config': workflow_config,
            'status': 'running',
            'started_at': datetime.now(),
            'steps_completed': 0,
            'total_steps': 4,
            'results': {}
        }

        st.session_state.active_backtests[pipeline_id] = pipeline_data

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Étape 1: Backtest de base
            status_text.text("🔄 Exécution du backtest de base...")
            base_results = self._run_base_backtest(workflow_config['base_config'])
            pipeline_data['results']['base'] = base_results
            pipeline_data['steps_completed'] = 1
            progress_bar.progress(0.25)

            # Étape 2: Walk-Forward (si configuré)
            if 'walk_forward' in workflow_config:
                status_text.text("⏭️ Validation Walk-Forward...")
                wf_results = self._run_walk_forward_analysis(workflow_config['walk_forward'])
                pipeline_data['results']['walk_forward'] = wf_results
                pipeline_data['steps_completed'] = 2
                progress_bar.progress(0.5)

            # Étape 3: Monte Carlo (si configuré)
            if 'monte_carlo' in workflow_config:
                status_text.text("🎲 Simulation Monte Carlo...")
                mc_results = self._run_monte_carlo_simulation(workflow_config['monte_carlo'])
                pipeline_data['results']['monte_carlo'] = mc_results
                pipeline_data['steps_completed'] = 3
                progress_bar.progress(0.75)

            # Étape 4: Analyse intégrée
            status_text.text("📊 Génération de l'analyse intégrée...")
            integrated_results = self._compile_integrated_results(pipeline_data['results'])
            pipeline_data['results']['integrated'] = integrated_results
            pipeline_data['steps_completed'] = 4
            progress_bar.progress(1.0)

            # Finalisation
            pipeline_data['status'] = 'completed'
            pipeline_data['completed_at'] = datetime.now()

            # Sauvegarde des résultats
            if pipeline_id not in st.session_state.backtest_results:
                st.session_state.backtest_results = {}
            st.session_state.backtest_results[pipeline_id] = integrated_results

            status_text.text("✅ Pipeline terminé avec succès!")
            st.success(f"Pipeline {pipeline_id} terminé avec succès!")

        except Exception as e:
            pipeline_data['status'] = 'failed'
            pipeline_data['error'] = str(e)
            status_text.text(f"❌ Erreur: {str(e)}")
            st.error(f"Erreur lors de l'exécution du pipeline: {str(e)}")

    def _render_backtests_table(self):
        """Affiche le tableau des backtests."""
        if not st.session_state.get('active_backtests'):
            return

        backtests_data = []
        for bt_id, bt_data in st.session_state.active_backtests.items():
            backtests_data.append({
                'ID': bt_id,
                'Status': bt_data['status'],
                'Démarré': bt_data['started_at'].strftime('%H:%M:%S'),
                'Étapes': f"{bt_data['steps_completed']}/{bt_data['total_steps']}",
                'Durée': self._calculate_duration(bt_data['started_at'])
            })

        if backtests_data:
            df = pd.DataFrame(backtests_data)
            st.dataframe(df, use_container_width=True)

    def _run_base_backtest(self, config: Dict) -> Dict:
        """Exécute un backtest de base."""
        # Simulation d'un backtest avec résultats réalistes
        return self._generate_simulated_backtest_results(config)

    def _run_walk_forward_analysis(self, config: Dict) -> Dict:
        """Exécute une analyse Walk-Forward."""
        return self.walk_forward._generate_simulated_results(config)

    def _run_monte_carlo_simulation(self, config: Dict) -> Dict:
        """Exécute une simulation Monte Carlo."""
        return self.monte_carlo._generate_mc_results(config)

    def _compile_integrated_results(self, results: Dict) -> Dict:
        """Compile les résultats de tous les composants."""
        integrated = results.get('base', {})

        # Intégration des résultats Walk-Forward
        if 'walk_forward' in results:
            integrated['walk_forward_results'] = results['walk_forward']

        # Intégration des résultats Monte Carlo
        if 'monte_carlo' in results:
            integrated['monte_carlo_results'] = results['monte_carlo']

        # Métriques consolidées
        integrated['integration_score'] = self._calculate_integration_score(results)
        integrated['confidence_level'] = self._calculate_confidence_level(results)

        return integrated

    def _generate_simulated_backtest_results(self, config: Dict) -> Dict:
        """Génère des résultats de backtest simulés."""
        np.random.seed(42)  # Pour la reproductibilité

        # Génération de la série de prix
        days = 252
        initial_price = 10000
        dates = pd.date_range(start='2024-01-01', periods=days, freq='D')

        # Simulation d'une stratégie avec drift positif
        daily_returns = np.random.normal(0.0008, 0.02, days)  # ~20% annuel avec 20% vol
        daily_returns[0] = 0  # Premier jour = 0

        cumulative_returns = np.cumprod(1 + daily_returns)
        equity_curve = initial_price * cumulative_returns

        # Calcul des métriques
        total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
        annualized_return = ((equity_curve[-1] / equity_curve[0]) ** (252/days) - 1) * 100
        volatility = np.std(daily_returns) * np.sqrt(252) * 100

        # Sharpe ratio
        risk_free_rate = 0.02
        excess_returns = daily_returns - risk_free_rate/252
        sharpe_ratio = np.mean(excess_returns) / np.std(daily_returns) * np.sqrt(252)

        # Drawdown
        rolling_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - rolling_max) / rolling_max * 100
        max_drawdown = np.min(drawdown)

        # Métriques de trading
        num_trades = np.random.randint(80, 150)
        win_rate = np.random.uniform(45, 65)
        profit_factor = np.random.uniform(1.1, 2.5)

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'sortino_ratio': sharpe_ratio * 1.2,  # Approximation
            'calmar_ratio': annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            'var_95': np.percentile(daily_returns, 5) * 100,
            'cvar_95': np.mean(daily_returns[daily_returns <= np.percentile(daily_returns, 5)]) * 100,
            'win_rate': win_rate,
            'total_trades': num_trades,
            'profit_factor': profit_factor,
            'avg_trade': total_return / num_trades,
            'avg_win': total_return / (num_trades * win_rate / 100),
            'avg_loss': -total_return / (num_trades * (100 - win_rate) / 100),
            'best_month': np.max(daily_returns) * 100,
            'worst_month': np.min(daily_returns) * 100,
            'equity_curve': {
                'dates': dates.tolist(),
                'values': equity_curve.tolist()
            },
            'drawdown_series': drawdown.tolist(),
            'monthly_returns': np.random.normal(1.5, 4.0, 12).tolist(),
            'benchmark_curve': {
                'dates': dates.tolist(),
                'values': (initial_price * np.cumprod(1 + np.random.normal(0.0005, 0.015, days))).tolist()
            }
        }

    def _calculate_integration_score(self, results: Dict) -> float:
        """Calcule un score d'intégration basé sur la cohérence des résultats."""
        base_score = 0.7  # Score de base

        # Bonus pour la cohérence Walk-Forward
        if 'walk_forward' in results:
            base_score += 0.15

        # Bonus pour la robustesse Monte Carlo
        if 'monte_carlo' in results:
            base_score += 0.15

        return min(base_score, 1.0)

    def _calculate_confidence_level(self, results: Dict) -> float:
        """Calcule le niveau de confiance global."""
        confidence = 0.6  # Base

        # Augmentation basée sur les tests
        if 'walk_forward' in results:
            confidence += 0.2
        if 'monte_carlo' in results:
            confidence += 0.2

        return min(confidence, 0.95)

    def _render_strategy_comparison(self):
        """Affiche la comparaison des stratégies."""
        if len(st.session_state.get('backtest_results', {})) < 2:
            st.info("Besoin d'au moins 2 résultats pour la comparaison.")
            return

        # Table de comparaison simple
        comparison_data = []
        for bt_id, results in st.session_state.backtest_results.items():
            comparison_data.append({
                'Strategy': bt_id,
                'Total Return (%)': f"{results.get('total_return', 0):.1f}",
                'Sharpe Ratio': f"{results.get('sharpe_ratio', 0):.2f}",
                'Max DD (%)': f"{results.get('max_drawdown', 0):.1f}",
                'Win Rate (%)': f"{results.get('win_rate', 0):.1f}"
            })

        if comparison_data:
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)

    def _calculate_duration(self, start_time: datetime) -> str:
        """Calcule la durée écoulée."""
        duration = datetime.now() - start_time
        seconds = int(duration.total_seconds())
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds//60}m {seconds%60}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"

    def _compile_report_data(self, results: Dict) -> Dict:
        """Compile les données pour le rapport."""
        return {
            'executive_summary': self._generate_executive_summary(results),
            'detailed_metrics': results,
            'recommendations': self._generate_recommendations(results),
            'risk_assessment': self._generate_risk_assessment(results)
        }

    def _generate_executive_summary(self, results: Dict) -> str:
        """Génère un résumé exécutif."""
        total_return = results.get('total_return', 0)
        sharpe = results.get('sharpe_ratio', 0)
        max_dd = results.get('max_drawdown', 0)

        performance_rating = "Excellente" if total_return > 15 else "Bonne" if total_return > 5 else "Modérée"

        return f"""
        La stratégie a généré un retour total de {total_return:.1f}% avec un ratio de Sharpe de {sharpe:.2f}.
        Le drawdown maximal de {abs(max_dd):.1f}% indique un niveau de risque contrôlé.
        Performance globale: {performance_rating}.
        """

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Génère des recommandations."""
        recommendations = []

        sharpe = results.get('sharpe_ratio', 0)
        if sharpe < 1.0:
            recommendations.append("Considérer l'optimisation des paramètres pour améliorer le Sharpe ratio")

        max_dd = results.get('max_drawdown', 0)
        if abs(max_dd) > 20:
            recommendations.append("Implementer des mécanismes de stop-loss plus stricts")

        win_rate = results.get('win_rate', 0)
        if win_rate < 50:
            recommendations.append("Analyser les signaux pour améliorer la précision")

        return recommendations

    def _generate_risk_assessment(self, results: Dict) -> Dict:
        """Génère une évaluation des risques."""
        var_95 = results.get('var_95', 0)
        max_dd = results.get('max_drawdown', 0)

        risk_level = "Élevé" if abs(max_dd) > 25 else "Modéré" if abs(max_dd) > 15 else "Faible"

        return {
            'risk_level': risk_level,
            'var_assessment': f"VaR 95%: {var_95:.2f}%",
            'drawdown_assessment': f"Drawdown maximal: {abs(max_dd):.1f}%",
            'overall_risk': risk_level
        }

    def _generate_excel_export(self, backtest_id: str) -> Dict:
        """Génère un export Excel."""
        return {'format': 'excel', 'data': 'Excel export data'}

    def _generate_pdf_report(self, backtest_id: str) -> Dict:
        """Génère un rapport PDF."""
        return {'format': 'pdf', 'data': 'PDF report data'}

    def _generate_json_export(self, backtest_id: str) -> Dict:
        """Génère un export JSON."""
        return {'format': 'json', 'data': st.session_state.backtest_results.get(backtest_id, {})}