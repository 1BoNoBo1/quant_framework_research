"""
Alpha Formula Visualizer - Composant pour visualiser et analyser les formules alpha
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import re

class AlphaFormulaVisualizer:
    """Visualiseur et analyseur de formules alpha."""

    def __init__(self):
        self.alpha_library = self._load_alpha_library()

    def _load_alpha_library(self):
        """Charge la bibliothèque de formules alpha."""
        return {
            "classic": [
                {
                    "id": "Alpha006",
                    "formula": "(-1 * corr(open, volume, 10))",
                    "description": "Negative correlation between open price and volume",
                    "ic": 0.0723,
                    "sharpe": 1.82,
                    "complexity": 3,
                    "category": "Mean Reversion"
                },
                {
                    "id": "Alpha061",
                    "formula": "Less(CSRank((vwap - Min(vwap, 16))), CSRank(Corr(vwap, Mean(volume, 180), 17)))",
                    "description": "Complex VWAP and volume correlation ranking",
                    "ic": 0.0456,
                    "sharpe": 1.34,
                    "complexity": 8,
                    "category": "Technical"
                },
                {
                    "id": "Alpha099",
                    "formula": "sign(delta(cs_rank(close * volume), 5))",
                    "description": "Sign of delta in cross-sectional rank of dollar volume",
                    "ic": 0.0612,
                    "sharpe": 1.67,
                    "complexity": 5,
                    "category": "Momentum"
                }
            ],
            "rl_generated": [
                {
                    "id": "RL_2024_001",
                    "formula": "ts_rank(delta(vwap, 5) * sign(volume - mean(volume, 20)), 10)",
                    "ic": 0.0812,
                    "sharpe": 2.14,
                    "complexity": 6,
                    "generation": 523,
                    "agent": "PPO_v2.0",
                    "category": "Volume"
                },
                {
                    "id": "RL_2024_002",
                    "formula": "product(cs_rank(high - low), wma(close, 15))",
                    "ic": 0.0734,
                    "sharpe": 1.89,
                    "complexity": 5,
                    "generation": 892,
                    "agent": "PPO_v2.0",
                    "category": "Volatility"
                }
            ],
            "custom": []
        }

    def render_alpha_library_view(self):
        """Rendu de la vue bibliothèque d'alphas."""
        st.subheader("🏛️ Alpha Formula Library")

        # Filtres
        col_filter1, col_filter2, col_filter3 = st.columns(3)

        with col_filter1:
            library_type = st.selectbox(
                "Library Type",
                ["All", "Classic Alpha101", "RL Generated", "Custom"],
                help="Filtrer par type de bibliothèque"
            )

        with col_filter2:
            category_filter = st.selectbox(
                "Category",
                ["All", "Mean Reversion", "Momentum", "Technical", "Volume", "Volatility"],
                help="Filtrer par catégorie"
            )

        with col_filter3:
            sort_by = st.selectbox(
                "Sort By",
                ["IC Score", "Sharpe Ratio", "Complexity", "Alphabetical"],
                help="Critère de tri"
            )

        # Recherche
        search_query = st.text_input(
            "🔍 Search Formulas",
            placeholder="Ex: corr, volume, delta, vwap...",
            help="Rechercher dans les formules et descriptions"
        )

        # Affichage des formules par catégorie
        if library_type in ["All", "Classic Alpha101"]:
            self._render_formula_category("📚 Classic Alpha101", self.alpha_library["classic"],
                                        category_filter, search_query, sort_by)

        if library_type in ["All", "RL Generated"]:
            self._render_formula_category("🤖 RL Generated", self.alpha_library["rl_generated"],
                                        category_filter, search_query, sort_by)

        if library_type in ["All", "Custom"]:
            self._render_formula_category("✏️ Custom Formulas", self.alpha_library["custom"],
                                        category_filter, search_query, sort_by)

    def _render_formula_category(self, title: str, formulas: List[Dict],
                                category_filter: str, search_query: str, sort_by: str):
        """Rendu d'une catégorie de formules."""
        if not formulas:
            return

        # Filtrage
        filtered_formulas = self._filter_formulas(formulas, category_filter, search_query)

        # Tri
        sorted_formulas = self._sort_formulas(filtered_formulas, sort_by)

        if not sorted_formulas:
            return

        st.markdown(f"### {title}")

        for formula in sorted_formulas:
            with st.expander(f"{formula['id']} - IC: {formula['ic']:.4f}", expanded=False):
                self._render_single_formula(formula)

    def _filter_formulas(self, formulas: List[Dict], category_filter: str, search_query: str) -> List[Dict]:
        """Filtre les formules selon les critères."""
        filtered = formulas

        # Filtre par catégorie
        if category_filter != "All":
            filtered = [f for f in filtered if f.get('category') == category_filter]

        # Filtre par recherche
        if search_query:
            query_lower = search_query.lower()
            filtered = [f for f in filtered if
                       query_lower in f['formula'].lower() or
                       query_lower in f.get('description', '').lower() or
                       query_lower in f['id'].lower()]

        return filtered

    def _sort_formulas(self, formulas: List[Dict], sort_by: str) -> List[Dict]:
        """Trie les formulas selon le critère."""
        if sort_by == "IC Score":
            return sorted(formulas, key=lambda x: x['ic'], reverse=True)
        elif sort_by == "Sharpe Ratio":
            return sorted(formulas, key=lambda x: x['sharpe'], reverse=True)
        elif sort_by == "Complexity":
            return sorted(formulas, key=lambda x: x['complexity'])
        else:  # Alphabetical
            return sorted(formulas, key=lambda x: x['id'])

    def _render_single_formula(self, formula: Dict):
        """Rendu d'une formule individuelle."""
        # Header avec métriques
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

        with col1:
            st.markdown(f"**{formula['id']}**")
            if 'description' in formula:
                st.markdown(f"*{formula['description']}*")

        with col2:
            st.metric("IC Score", f"{formula['ic']:.4f}")

        with col3:
            st.metric("Sharpe", f"{formula['sharpe']:.2f}")

        with col4:
            st.metric("Complexity", formula['complexity'])

        # Formule
        st.code(formula['formula'], language='python')

        # Informations supplémentaires
        if 'generation' in formula:
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.markdown(f"**Generation:** {formula['generation']}")
            with col_info2:
                st.markdown(f"**Agent:** {formula['agent']}")

        # Actions
        col_action1, col_action2, col_action3, col_action4 = st.columns(4)

        with col_action1:
            if st.button("📊 Analyze", key=f"analyze_{formula['id']}", use_container_width=True):
                self._show_formula_analysis(formula)

        with col_action2:
            if st.button("🧪 Backtest", key=f"backtest_{formula['id']}", use_container_width=True):
                self._run_backtest(formula)

        with col_action3:
            if st.button("🚀 Deploy", key=f"deploy_{formula['id']}", use_container_width=True):
                st.success(f"Formule {formula['id']} déployée!")

        with col_action4:
            if st.button("📋 Copy", key=f"copy_{formula['id']}", use_container_width=True):
                st.success("Formule copiée dans le presse-papier!")

    def _show_formula_analysis(self, formula: Dict):
        """Affiche l'analyse détaillée d'une formule."""
        st.subheader(f"📊 Analysis: {formula['id']}")

        # Analyse de la structure
        col_struct1, col_struct2 = st.columns(2)

        with col_struct1:
            st.markdown("### Structure Analysis")
            structure = self._analyze_formula_structure(formula['formula'])
            st.json(structure)

        with col_struct2:
            st.markdown("### Performance Metrics")
            metrics = self._generate_performance_metrics(formula)

            for metric, value in metrics.items():
                st.metric(metric, value)

        # Graphiques de performance
        self._render_performance_charts(formula)

    def _analyze_formula_structure(self, formula: str) -> Dict:
        """Analyse la structure d'une formule."""
        # Extraction basique des composants
        operators = re.findall(r'\b(corr|delta|ts_rank|cs_rank|sign|abs|mean|std|min|max|wma|ema)\b', formula)
        features = re.findall(r'\b(open|high|low|close|volume|vwap)\b', formula)
        numbers = re.findall(r'\b\d+\b', formula)

        return {
            "operators_used": list(set(operators)),
            "features_used": list(set(features)),
            "time_periods": list(set(numbers)),
            "nesting_depth": formula.count('('),
            "total_length": len(formula),
            "complexity_score": len(set(operators)) + len(set(features)) + formula.count('(')
        }

    def _generate_performance_metrics(self, formula: Dict) -> Dict:
        """Génère des métriques de performance simulées."""
        return {
            "Annualized Return": f"{np.random.uniform(8, 25):.1f}%",
            "Volatility": f"{np.random.uniform(12, 30):.1f}%",
            "Max Drawdown": f"{np.random.uniform(-15, -5):.1f}%",
            "Win Rate": f"{np.random.uniform(45, 65):.1f}%",
            "Calmar Ratio": f"{np.random.uniform(0.8, 2.5):.2f}",
            "Sortino Ratio": f"{np.random.uniform(1.2, 3.0):.2f}"
        }

    def _render_performance_charts(self, formula: Dict):
        """Rendu des graphiques de performance."""
        st.markdown("### Performance Charts")

        # Données simulées
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        returns = np.random.randn(len(dates)) * 0.02
        cumulative_returns = np.cumprod(1 + returns) - 1
        benchmark_returns = np.cumprod(1 + np.random.randn(len(dates)) * 0.015) - 1

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            # Graphique de performance cumulative
            fig_perf = go.Figure()

            fig_perf.add_trace(go.Scatter(
                x=dates,
                y=cumulative_returns * 100,
                mode='lines',
                name=f'Alpha {formula["id"]}',
                line=dict(color='#00ff88', width=2)
            ))

            fig_perf.add_trace(go.Scatter(
                x=dates,
                y=benchmark_returns * 100,
                mode='lines',
                name='Benchmark',
                line=dict(color='#666666', width=1, dash='dash')
            ))

            fig_perf.update_layout(
                title="Cumulative Performance",
                xaxis_title="Date",
                yaxis_title="Return (%)",
                template='plotly_dark',
                height=400
            )

            st.plotly_chart(fig_perf, use_container_width=True)

        with col_chart2:
            # Distribution des returns
            fig_dist = go.Figure()

            fig_dist.add_trace(go.Histogram(
                x=returns * 100,
                nbinsx=30,
                name='Daily Returns',
                marker_color='#00ff88',
                opacity=0.7
            ))

            fig_dist.update_layout(
                title="Returns Distribution",
                xaxis_title="Daily Return (%)",
                yaxis_title="Frequency",
                template='plotly_dark',
                height=400
            )

            st.plotly_chart(fig_dist, use_container_width=True)

    def _run_backtest(self, formula: Dict):
        """Lance un backtest de la formule."""
        with st.spinner("Running backtest..."):
            # Simulation de backtest
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)

            # Résultats simulés
            results = {
                "Total Return": f"{np.random.uniform(15, 45):.1f}%",
                "Annualized Return": f"{np.random.uniform(8, 25):.1f}%",
                "Sharpe Ratio": f"{np.random.uniform(1.2, 2.8):.2f}",
                "Max Drawdown": f"{np.random.uniform(-20, -5):.1f}%",
                "Win Rate": f"{np.random.uniform(45, 65):.1f}%",
                "Profit Factor": f"{np.random.uniform(1.2, 2.5):.2f}",
                "IC Score": f"{formula['ic']:.4f}",
                "Calmar Ratio": f"{np.random.uniform(0.8, 2.2):.2f}"
            }

            st.success("✅ Backtest completed!")

            # Affichage des résultats
            col_res1, col_res2, col_res3, col_res4 = st.columns(4)

            metrics_list = list(results.items())
            for i, (metric, value) in enumerate(metrics_list):
                col = [col_res1, col_res2, col_res3, col_res4][i % 4]
                with col:
                    st.metric(metric, value)

    def render_formula_comparison(self, formulas: List[Dict]):
        """Rendu de la comparaison entre formules."""
        st.subheader("🏆 Formula Comparison")

        if len(formulas) < 2:
            st.warning("Select at least 2 formulas to compare")
            return

        # Tableau de comparaison
        comparison_df = pd.DataFrame(formulas)[['id', 'ic', 'sharpe', 'complexity', 'category']]

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ic": st.column_config.ProgressColumn(
                    "IC Score",
                    help="Information Coefficient",
                    min_value=0,
                    max_value=0.1,
                ),
                "sharpe": st.column_config.NumberColumn(
                    "Sharpe Ratio",
                    help="Risk-adjusted return",
                    format="%.2f",
                ),
            }
        )

        # Graphiques de comparaison
        col_comp1, col_comp2 = st.columns(2)

        with col_comp1:
            # IC vs Complexity
            fig_scatter = px.scatter(
                comparison_df,
                x='complexity',
                y='ic',
                size='sharpe',
                color='category',
                hover_data=['id'],
                title='IC Score vs Complexity',
                labels={'complexity': 'Complexity', 'ic': 'IC Score'}
            )
            fig_scatter.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_comp2:
            # Radar chart
            fig_radar = go.Figure()

            for formula in formulas[:3]:  # Limite à 3 pour lisibilité
                # Normaliser les métriques pour le radar
                normalized_metrics = [
                    formula['ic'] / 0.1,  # Normaliser IC sur 0.1
                    formula['sharpe'] / 3.0,  # Normaliser Sharpe sur 3.0
                    1 - (formula['complexity'] / 10)  # Complexité inversée et normalisée
                ]

                fig_radar.add_trace(go.Scatterpolar(
                    r=normalized_metrics + [normalized_metrics[0]],  # Fermer le polygone
                    theta=['IC Score', 'Sharpe Ratio', 'Simplicity', 'IC Score'],
                    fill='toself',
                    name=formula['id']
                ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                title="Formula Performance Radar",
                template='plotly_dark',
                height=400
            )

            st.plotly_chart(fig_radar, use_container_width=True)

    def render_alpha_performance_summary(self):
        """Rendu du résumé de performance des alphas."""
        st.subheader("📈 Alpha Performance Summary")

        # Statistiques globales
        all_formulas = (self.alpha_library["classic"] +
                       self.alpha_library["rl_generated"] +
                       self.alpha_library["custom"])

        if not all_formulas:
            st.info("No formulas in library")
            return

        # Métriques de résumé
        avg_ic = np.mean([f['ic'] for f in all_formulas])
        avg_sharpe = np.mean([f['sharpe'] for f in all_formulas])
        avg_complexity = np.mean([f['complexity'] for f in all_formulas])

        col_summary1, col_summary2, col_summary3, col_summary4 = st.columns(4)

        with col_summary1:
            st.metric("Total Formulas", len(all_formulas))
        with col_summary2:
            st.metric("Avg IC Score", f"{avg_ic:.4f}")
        with col_summary3:
            st.metric("Avg Sharpe", f"{avg_sharpe:.2f}")
        with col_summary4:
            st.metric("Avg Complexity", f"{avg_complexity:.1f}")

        # Distribution des métriques
        col_dist1, col_dist2 = st.columns(2)

        with col_dist1:
            # Distribution IC scores
            ic_scores = [f['ic'] for f in all_formulas]
            fig_ic_dist = px.histogram(
                x=ic_scores,
                nbins=20,
                title="IC Score Distribution",
                labels={'x': 'IC Score', 'y': 'Count'}
            )
            fig_ic_dist.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig_ic_dist, use_container_width=True)

        with col_dist2:
            # Performance par catégorie
            categories = [f.get('category', 'Unknown') for f in all_formulas]
            category_ic = {}

            for cat in set(categories):
                cat_formulas = [f for f in all_formulas if f.get('category') == cat]
                category_ic[cat] = np.mean([f['ic'] for f in cat_formulas])

            fig_cat = px.bar(
                x=list(category_ic.keys()),
                y=list(category_ic.values()),
                title="Average IC by Category",
                labels={'x': 'Category', 'y': 'Average IC Score'}
            )
            fig_cat.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig_cat, use_container_width=True)