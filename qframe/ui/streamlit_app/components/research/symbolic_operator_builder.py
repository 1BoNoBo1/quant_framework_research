"""
Symbolic Operator Builder - Constructeur interactif d'opérateurs symboliques
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import json
import re

class SymbolicOperatorBuilder:
    """Constructeur interactif pour assembler des opérateurs symboliques."""

    def __init__(self):
        self.operators = self._load_operators()
        self.market_features = ["open", "high", "low", "close", "volume", "vwap"]
        self.constants = ["-2.0", "-1.0", "-0.5", "0.5", "1.0", "2.0", "5.0", "10.0"]
        self.time_deltas = ["5", "10", "20", "30", "40", "50", "60", "120"]

    def _load_operators(self) -> Dict:
        """Charge les opérateurs symboliques disponibles."""
        return {
            "temporal": {
                "delta": {
                    "syntax": "delta(x, t)",
                    "description": "Différence avec t périodes passées",
                    "params": ["expression", "periods"],
                    "example": "delta(close, 5)",
                    "complexity": 2
                },
                "ts_rank": {
                    "syntax": "ts_rank(x, t)",
                    "description": "Rang temporel sur t périodes",
                    "params": ["expression", "periods"],
                    "example": "ts_rank(volume, 20)",
                    "complexity": 3
                },
                "argmax": {
                    "syntax": "argmax(x, t)",
                    "description": "Index du maximum sur t périodes",
                    "params": ["expression", "periods"],
                    "example": "argmax(high, 10)",
                    "complexity": 2
                },
                "argmin": {
                    "syntax": "argmin(x, t)",
                    "description": "Index du minimum sur t périodes",
                    "params": ["expression", "periods"],
                    "example": "argmin(low, 10)",
                    "complexity": 2
                }
            },
            "statistical": {
                "mean": {
                    "syntax": "mean(x, t)",
                    "description": "Moyenne mobile sur t périodes",
                    "params": ["expression", "periods"],
                    "example": "mean(volume, 20)",
                    "complexity": 1
                },
                "std": {
                    "syntax": "std(x, t)",
                    "description": "Écart-type sur t périodes",
                    "params": ["expression", "periods"],
                    "example": "std(close, 10)",
                    "complexity": 2
                },
                "skew": {
                    "syntax": "skew(x, t)",
                    "description": "Asymétrie de distribution",
                    "params": ["expression", "periods"],
                    "example": "skew(returns, 30)",
                    "complexity": 3
                },
                "kurt": {
                    "syntax": "kurt(x, t)",
                    "description": "Kurtosis (peakedness)",
                    "params": ["expression", "periods"],
                    "example": "kurt(returns, 30)",
                    "complexity": 3
                },
                "mad": {
                    "syntax": "mad(x, t)",
                    "description": "Mean Absolute Deviation",
                    "params": ["expression", "periods"],
                    "example": "mad(close, 15)",
                    "complexity": 2
                }
            },
            "cross_sectional": {
                "cs_rank": {
                    "syntax": "cs_rank(x)",
                    "description": "Rang cross-sectionnel",
                    "params": ["expression"],
                    "example": "cs_rank(volume)",
                    "complexity": 2
                },
                "scale": {
                    "syntax": "scale(x)",
                    "description": "Normalisation par somme absolue",
                    "params": ["expression"],
                    "example": "scale(returns)",
                    "complexity": 1
                }
            },
            "mathematical": {
                "sign": {
                    "syntax": "sign(x)",
                    "description": "Signe de l'expression",
                    "params": ["expression"],
                    "example": "sign(delta(close, 1))",
                    "complexity": 1
                },
                "abs": {
                    "syntax": "abs(x)",
                    "description": "Valeur absolue",
                    "params": ["expression"],
                    "example": "abs(returns)",
                    "complexity": 1
                },
                "pow": {
                    "syntax": "pow(x, n)",
                    "description": "Puissance n de x",
                    "params": ["expression", "power"],
                    "example": "pow(returns, 2)",
                    "complexity": 1
                },
                "log": {
                    "syntax": "log(x)",
                    "description": "Logarithme naturel",
                    "params": ["expression"],
                    "example": "log(volume + 1)",
                    "complexity": 2
                }
            },
            "correlation": {
                "corr": {
                    "syntax": "corr(x, y, t)",
                    "description": "Corrélation sur t périodes",
                    "params": ["expression1", "expression2", "periods"],
                    "example": "corr(open, volume, 10)",
                    "complexity": 4
                },
                "cov": {
                    "syntax": "cov(x, y, t)",
                    "description": "Covariance sur t périodes",
                    "params": ["expression1", "expression2", "periods"],
                    "example": "cov(high, low, 15)",
                    "complexity": 4
                }
            },
            "moving_averages": {
                "wma": {
                    "syntax": "wma(x, t)",
                    "description": "Moyenne mobile pondérée",
                    "params": ["expression", "periods"],
                    "example": "wma(close, 20)",
                    "complexity": 2
                },
                "ema": {
                    "syntax": "ema(x, t)",
                    "description": "Moyenne mobile exponentielle",
                    "params": ["expression", "periods"],
                    "example": "ema(close, 12)",
                    "complexity": 2
                }
            },
            "conditional": {
                "cond": {
                    "syntax": "cond(condition, true_val, false_val)",
                    "description": "Condition ternaire",
                    "params": ["condition", "true_value", "false_value"],
                    "example": "cond(close > open, 1, -1)",
                    "complexity": 3
                },
                "max": {
                    "syntax": "max(x, y)",
                    "description": "Maximum de deux expressions",
                    "params": ["expression1", "expression2"],
                    "example": "max(high, close)",
                    "complexity": 1
                },
                "min": {
                    "syntax": "min(x, y)",
                    "description": "Minimum de deux expressions",
                    "params": ["expression1", "expression2"],
                    "example": "min(low, open)",
                    "complexity": 1
                }
            }
        }

    def render_operator_palette(self):
        """Rendu de la palette d'opérateurs."""
        st.subheader("🎨 Operator Palette")

        # Recherche et filtres
        col_search, col_filter = st.columns([2, 1])

        with col_search:
            search_query = st.text_input(
                "🔍 Search Operators",
                placeholder="Ex: corr, delta, rank...",
                help="Rechercher dans les opérateurs"
            )

        with col_filter:
            complexity_filter = st.selectbox(
                "Complexity Level",
                ["All", "Simple (1-2)", "Medium (3-4)", "Complex (5+)"],
                help="Filtrer par niveau de complexité"
            )

        # Affichage par catégorie
        categories = list(self.operators.keys())
        selected_categories = st.multiselect(
            "Select Categories",
            categories,
            default=categories,
            help="Choisir les catégories d'opérateurs à afficher"
        )

        for category_key in selected_categories:
            category = self.operators[category_key]
            category_name = category_key.replace('_', ' ').title()

            with st.expander(f"📁 {category_name}", expanded=True):
                # Grille d'opérateurs
                operators_list = list(category.items())
                cols_per_row = 2

                for i in range(0, len(operators_list), cols_per_row):
                    cols = st.columns(cols_per_row)

                    for j, col in enumerate(cols):
                        if i + j < len(operators_list):
                            op_name, op_info = operators_list[i + j]

                            # Filtrage
                            if self._should_show_operator(op_name, op_info, search_query, complexity_filter):
                                with col:
                                    self._render_operator_card(category_key, op_name, op_info)

    def _should_show_operator(self, op_name: str, op_info: Dict, search_query: str, complexity_filter: str) -> bool:
        """Détermine si un opérateur doit être affiché selon les filtres."""
        # Filtre de recherche
        if search_query:
            query_lower = search_query.lower()
            if not (query_lower in op_name.lower() or
                   query_lower in op_info.get('description', '').lower() or
                   query_lower in op_info.get('syntax', '').lower()):
                return False

        # Filtre de complexité
        complexity = op_info.get('complexity', 1)
        if complexity_filter == "Simple (1-2)" and complexity > 2:
            return False
        elif complexity_filter == "Medium (3-4)" and (complexity < 3 or complexity > 4):
            return False
        elif complexity_filter == "Complex (5+)" and complexity < 5:
            return False

        return True

    def _render_operator_card(self, category: str, op_name: str, op_info: Dict):
        """Rendu d'une carte d'opérateur."""
        # Carte avec style
        st.markdown(f"""
        <div style="
            border: 1px solid #00ff88;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            background: #1a1a2e;
            transition: all 0.3s ease;
        ">
            <h4 style="color: #00ff88; margin: 0 0 0.5rem 0;">{op_name}</h4>
            <code style="background: #0a0a0a; padding: 0.2rem; border-radius: 3px;">{op_info['syntax']}</code>
            <p style="margin: 0.5rem 0; font-size: 0.9rem; color: #ccc;">{op_info['description']}</p>
            <small style="color: #666;">Complexity: {op_info['complexity']} | Example: {op_info['example']}</small>
        </div>
        """, unsafe_allow_html=True)

        # Bouton d'ajout
        if st.button(f"Add {op_name}", key=f"add_op_{category}_{op_name}", use_container_width=True):
            if 'formula_components' not in st.session_state:
                st.session_state.formula_components = []

            st.session_state.formula_components.append({
                'type': 'operator',
                'category': category,
                'name': op_name,
                'info': op_info
            })
            st.success(f"Added {op_name}!")
            st.rerun()

    def render_feature_palette(self):
        """Rendu de la palette de features et constantes."""
        st.subheader("📊 Features & Constants")

        col_features, col_constants, col_times = st.columns(3)

        with col_features:
            st.markdown("### Market Features")
            for feature in self.market_features:
                if st.button(feature, key=f"feature_{feature}", use_container_width=True):
                    if 'formula_components' not in st.session_state:
                        st.session_state.formula_components = []

                    st.session_state.formula_components.append({
                        'type': 'feature',
                        'value': feature
                    })
                    st.rerun()

        with col_constants:
            st.markdown("### Constants")
            for const in self.constants:
                if st.button(const, key=f"const_{const}", use_container_width=True):
                    if 'formula_components' not in st.session_state:
                        st.session_state.formula_components = []

                    st.session_state.formula_components.append({
                        'type': 'constant',
                        'value': const
                    })
                    st.rerun()

        with col_times:
            st.markdown("### Time Periods")
            for delta in self.time_deltas:
                if st.button(f"{delta}d", key=f"time_{delta}", use_container_width=True):
                    if 'formula_components' not in st.session_state:
                        st.session_state.formula_components = []

                    st.session_state.formula_components.append({
                        'type': 'time_delta',
                        'value': delta
                    })
                    st.rerun()

    def render_formula_builder(self):
        """Rendu du constructeur de formules."""
        st.subheader("🏗️ Formula Builder")

        # Affichage des composants actuels
        if 'formula_components' not in st.session_state:
            st.session_state.formula_components = []

        if st.session_state.formula_components:
            st.markdown("### Current Components")

            # Affichage en grille
            cols_per_row = 5
            components = st.session_state.formula_components

            for i in range(0, len(components), cols_per_row):
                cols = st.columns(cols_per_row)

                for j, col in enumerate(cols):
                    if i + j < len(components):
                        component = components[i + j]

                        with col:
                            self._render_component_badge(component, i + j)

            # Actions sur les composants
            col_action1, col_action2, col_action3 = st.columns(3)

            with col_action1:
                if st.button("🗑️ Clear All", use_container_width=True):
                    st.session_state.formula_components = []
                    if 'current_formula' in st.session_state:
                        st.session_state.current_formula = ""
                    st.rerun()

            with col_action2:
                if st.button("🔗 Auto-Build", type="primary", use_container_width=True):
                    formula = self.auto_build_formula(components)
                    st.session_state.current_formula = formula
                    st.rerun()

            with col_action3:
                if st.button("💾 Save to History", use_container_width=True):
                    self._save_to_history()

        else:
            # Zone de drop vide
            st.markdown("""
            <div style="
                border: 2px dashed #00ff88;
                border-radius: 10px;
                padding: 3rem;
                text-align: center;
                margin: 2rem 0;
                background: rgba(0, 255, 136, 0.05);
            ">
                <h3>🎯 Start Building Your Alpha Formula</h3>
                <p>Select operators and features from the palette above</p>
                <p><code>Example: corr(open, volume, 10) or delta(cs_rank(close), 5)</code></p>
            </div>
            """, unsafe_allow_html=True)

    def _render_component_badge(self, component: Dict, index: int):
        """Rendu d'un badge de composant."""
        comp_type = component['type']

        if comp_type == 'operator':
            icon = "🔧"
            name = component['name']
            color = "#00ff88"
        elif comp_type == 'feature':
            icon = "📊"
            name = component['value']
            color = "#6b88ff"
        elif comp_type == 'constant':
            icon = "🔢"
            name = component['value']
            color = "#ff6b6b"
        elif comp_type == 'time_delta':
            icon = "⏰"
            name = f"{component['value']}d"
            color = "#ffd93d"
        else:
            icon = "❓"
            name = str(component.get('value', ''))
            color = "#666666"

        # Badge avec bouton de suppression
        st.markdown(f"""
        <div style="
            background: {color}20;
            border: 1px solid {color};
            border-radius: 20px;
            padding: 0.5rem 1rem;
            margin: 0.2rem 0;
            text-align: center;
            font-size: 0.9rem;
        ">
            {icon} <strong>{name}</strong>
        </div>
        """, unsafe_allow_html=True)

        if st.button("❌", key=f"remove_comp_{index}", help="Remove component"):
            st.session_state.formula_components.pop(index)
            st.rerun()

    def auto_build_formula(self, components: List[Dict]) -> str:
        """Construction automatique d'une formule."""
        try:
            operators = [c for c in components if c['type'] == 'operator']
            features = [c for c in components if c['type'] == 'feature']
            constants = [c for c in components if c['type'] == 'constant']
            time_deltas = [c for c in components if c['type'] == 'time_delta']

            if not operators and not features:
                return "Need operators and features to build formula"

            if not operators:
                # Formule simple avec features
                if len(features) >= 2:
                    return f"({features[0]['value']} - {features[1]['value']})"
                else:
                    return features[0]['value']

            # Sélectionner le premier opérateur
            operator = operators[0]
            op_name = operator['name']
            op_info = operator['info']
            params = op_info['params']

            # Construire selon le type d'opérateur
            if len(params) == 1 and params[0] == 'expression':
                # Opérateur unaire
                if features:
                    return f"{op_name}({features[0]['value']})"
                else:
                    return f"{op_name}(close)"

            elif len(params) == 2 and 'periods' in params:
                # Opérateur temporel
                feature = features[0]['value'] if features else 'close'
                period = time_deltas[0]['value'] if time_deltas else '10'
                return f"{op_name}({feature}, {period})"

            elif len(params) == 3 and op_name == 'corr':
                # Corrélation
                feat1 = features[0]['value'] if len(features) > 0 else 'open'
                feat2 = features[1]['value'] if len(features) > 1 else 'volume'
                period = time_deltas[0]['value'] if time_deltas else '10'
                return f"corr({feat1}, {feat2}, {period})"

            elif len(params) == 3 and op_name == 'cond':
                # Condition
                feat1 = features[0]['value'] if len(features) > 0 else 'close'
                feat2 = features[1]['value'] if len(features) > 1 else 'open'
                return f"cond({feat1} > {feat2}, 1, -1)"

            else:
                # Cas générique
                if features:
                    return f"{op_name}({features[0]['value']})"
                else:
                    return f"{op_name}(close)"

        except Exception as e:
            return f"Auto-build error: {str(e)}"

    def _save_to_history(self):
        """Sauvegarde la formule actuelle dans l'historique."""
        if 'formula_history' not in st.session_state:
            st.session_state.formula_history = []

        if 'current_formula' in st.session_state and st.session_state.current_formula:
            st.session_state.formula_history.append({
                'formula': st.session_state.current_formula,
                'components': st.session_state.formula_components.copy(),
                'timestamp': datetime.now()
            })
            st.success("Formula saved to history!")

    def render_formula_templates(self):
        """Rendu des templates de formules populaires."""
        st.subheader("🌟 Formula Templates")

        templates = [
            {
                "name": "Classic Alpha006",
                "formula": "(-1 * corr(open, volume, 10))",
                "description": "Negative correlation between price and volume",
                "category": "Mean Reversion"
            },
            {
                "name": "Momentum Signal",
                "formula": "sign(delta(cs_rank(close), 5))",
                "description": "Change in cross-sectional ranking",
                "category": "Momentum"
            },
            {
                "name": "Volume Reversal",
                "formula": "ts_rank(volume, 20) * sign(delta(close, 1))",
                "description": "Volume rank with price direction",
                "category": "Volume"
            },
            {
                "name": "Volatility Breakout",
                "formula": "cond(std(close, 20) > mean(std(close, 20), 60), 1, 0)",
                "description": "High volatility periods detection",
                "category": "Volatility"
            }
        ]

        for template in templates:
            with st.expander(f"📖 {template['name']} - {template['category']}"):
                st.code(template['formula'], language='python')
                st.markdown(f"**Description:** {template['description']}")

                col_temp1, col_temp2 = st.columns(2)
                with col_temp1:
                    if st.button("Load Template", key=f"load_template_{template['name']}"):
                        st.session_state.current_formula = template['formula']
                        st.success(f"Loaded {template['name']}!")
                        st.rerun()

                with col_temp2:
                    if st.button("Analyze Template", key=f"analyze_template_{template['name']}"):
                        st.info("Analysis would show complexity, operators used, etc.")

    def validate_formula(self, formula: str) -> Dict:
        """Validation d'une formule."""
        try:
            if not formula.strip():
                return {"valid": False, "error": "Empty formula"}

            # Validation basique de syntaxe
            if formula.count('(') != formula.count(')'):
                return {"valid": False, "error": "Mismatched parentheses"}

            # Extraire les opérateurs utilisés
            all_operators = []
            for category in self.operators.values():
                all_operators.extend(category.keys())

            used_operators = [op for op in all_operators if op in formula]
            used_features = [feat for feat in self.market_features if feat in formula]

            # Calcul de complexité
            complexity = (len(used_operators) * 2 +
                         len(used_features) +
                         formula.count('('))

            return {
                "valid": True,
                "complexity": complexity,
                "operators_used": used_operators,
                "features_used": used_features,
                "operators_count": len(used_operators),
                "features_count": len(used_features)
            }

        except Exception as e:
            return {"valid": False, "error": str(e)}

    def render_operator_documentation(self):
        """Rendu de la documentation des opérateurs."""
        st.subheader("📚 Operator Documentation")

        # Recherche dans la documentation
        doc_search = st.text_input("🔍 Search Documentation", placeholder="Ex: correlation, ranking...")

        for category_name, category in self.operators.items():
            category_title = category_name.replace('_', ' ').title()

            with st.expander(f"📖 {category_title} Operators"):
                for op_name, op_info in category.items():
                    if not doc_search or doc_search.lower() in op_name.lower() or doc_search.lower() in op_info['description'].lower():
                        st.markdown(f"### `{op_info['syntax']}`")
                        st.markdown(f"**Description:** {op_info['description']}")
                        st.markdown(f"**Parameters:** {', '.join(op_info['params'])}")
                        st.markdown(f"**Complexity:** {op_info['complexity']}")
                        st.code(op_info['example'], language='python')
                        st.markdown("---")

    def get_component_stats(self) -> Dict:
        """Retourne les statistiques des composants sélectionnés."""
        if 'formula_components' not in st.session_state:
            return {}

        components = st.session_state.formula_components

        stats = {
            'total_components': len(components),
            'operators': len([c for c in components if c['type'] == 'operator']),
            'features': len([c for c in components if c['type'] == 'feature']),
            'constants': len([c for c in components if c['type'] == 'constant']),
            'time_deltas': len([c for c in components if c['type'] == 'time_delta'])
        }

        # Estimation de complexité
        if components:
            operator_complexity = sum([
                c['info']['complexity'] for c in components
                if c['type'] == 'operator'
            ])
            stats['estimated_complexity'] = operator_complexity + len(components)

        return stats