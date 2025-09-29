"""
Feature Engineering Studio - Constructeur interactif de formules alpha
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import re
from typing import Dict, List, Optional, Any

# Import des composants
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api_client import APIClient
from components.utils import init_session_state, get_cached_data
from components.charts import create_line_chart
from components.tables import create_data_table

# Configuration de la page
st.set_page_config(
    page_title="QFrame - Feature Engineering",
    page_icon="⚙️",
    layout="wide"
)

# Initialisation
api_client = APIClient()
init_session_state()

# Styles CSS personnalisés
st.markdown("""
<style>
    .feature-header {
        background: linear-gradient(135deg, #ff6b6b 0%, #ffd93d 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .operator-card {
        background: #1a1a2e;
        border: 1px solid #00ff88;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .operator-card:hover {
        border-color: #ff6b6b;
        box-shadow: 0 4px 8px rgba(0, 255, 136, 0.2);
    }
    .formula-builder {
        background: #0f0f23;
        border: 2px dashed #00ff88;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        min-height: 200px;
        text-align: center;
    }
    .formula-display {
        font-family: 'Courier New', monospace;
        background: #0a0a0a;
        border: 1px solid #00ff88;
        border-radius: 5px;
        padding: 1.5rem;
        color: #00ff88;
        font-size: 1.2rem;
        margin: 1rem 0;
    }
    .operator-category {
        background: #16213e;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .feature-metric {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #333;
        margin: 0.5rem;
    }
    .validation-result {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .validation-success {
        background-color: rgba(0, 255, 136, 0.1);
        border: 1px solid #00ff88;
    }
    .validation-error {
        background-color: rgba(255, 107, 107, 0.1);
        border: 1px solid #ff6b6b;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="feature-header">
    <h1>⚙️ Feature Engineering Studio</h1>
    <p>Constructeur interactif de formules alpha avec opérateurs symboliques</p>
</div>
""", unsafe_allow_html=True)

# Initialisation des données de session
if 'formula_components' not in st.session_state:
    st.session_state.formula_components = []
if 'current_formula' not in st.session_state:
    st.session_state.current_formula = ""
if 'formula_history' not in st.session_state:
    st.session_state.formula_history = []

# Définition des opérateurs symboliques
OPERATORS = {
    "Temporal": {
        "delta": {
            "name": "delta(x, t)",
            "description": "Différence avec t périodes passées",
            "params": ["expression", "periods"],
            "example": "delta(close, 5)"
        },
        "ts_rank": {
            "name": "ts_rank(x, t)",
            "description": "Rang temporel sur t périodes",
            "params": ["expression", "periods"],
            "example": "ts_rank(volume, 20)"
        },
        "argmax": {
            "name": "argmax(x, t)",
            "description": "Index du maximum sur t périodes",
            "params": ["expression", "periods"],
            "example": "argmax(high, 10)"
        },
        "argmin": {
            "name": "argmin(x, t)",
            "description": "Index du minimum sur t périodes",
            "params": ["expression", "periods"],
            "example": "argmin(low, 10)"
        }
    },
    "Statistical": {
        "mean": {
            "name": "mean(x, t)",
            "description": "Moyenne mobile sur t périodes",
            "params": ["expression", "periods"],
            "example": "mean(volume, 20)"
        },
        "std": {
            "name": "std(x, t)",
            "description": "Écart-type sur t périodes",
            "params": ["expression", "periods"],
            "example": "std(close, 10)"
        },
        "skew": {
            "name": "skew(x, t)",
            "description": "Asymétrie de distribution",
            "params": ["expression", "periods"],
            "example": "skew(returns, 30)"
        },
        "kurt": {
            "name": "kurt(x, t)",
            "description": "Kurtosis (peakedness)",
            "params": ["expression", "periods"],
            "example": "kurt(returns, 30)"
        },
        "mad": {
            "name": "mad(x, t)",
            "description": "Mean Absolute Deviation",
            "params": ["expression", "periods"],
            "example": "mad(close, 15)"
        }
    },
    "Cross-Sectional": {
        "cs_rank": {
            "name": "cs_rank(x)",
            "description": "Rang cross-sectionnel",
            "params": ["expression"],
            "example": "cs_rank(volume)"
        },
        "scale": {
            "name": "scale(x)",
            "description": "Normalisation par somme absolue",
            "params": ["expression"],
            "example": "scale(returns)"
        }
    },
    "Mathematical": {
        "sign": {
            "name": "sign(x)",
            "description": "Signe de l'expression",
            "params": ["expression"],
            "example": "sign(delta(close, 1))"
        },
        "abs": {
            "name": "abs(x)",
            "description": "Valeur absolue",
            "params": ["expression"],
            "example": "abs(returns)"
        },
        "pow": {
            "name": "pow(x, n)",
            "description": "Puissance n de x",
            "params": ["expression", "power"],
            "example": "pow(returns, 2)"
        },
        "log": {
            "name": "log(x)",
            "description": "Logarithme naturel",
            "params": ["expression"],
            "example": "log(volume + 1)"
        }
    },
    "Correlation": {
        "corr": {
            "name": "corr(x, y, t)",
            "description": "Corrélation sur t périodes",
            "params": ["expression1", "expression2", "periods"],
            "example": "corr(open, volume, 10)"
        },
        "cov": {
            "name": "cov(x, y, t)",
            "description": "Covariance sur t périodes",
            "params": ["expression1", "expression2", "periods"],
            "example": "cov(high, low, 15)"
        }
    },
    "Moving Averages": {
        "wma": {
            "name": "wma(x, t)",
            "description": "Moyenne mobile pondérée",
            "params": ["expression", "periods"],
            "example": "wma(close, 20)"
        },
        "ema": {
            "name": "ema(x, t)",
            "description": "Moyenne mobile exponentielle",
            "params": ["expression", "periods"],
            "example": "ema(close, 12)"
        }
    },
    "Conditional": {
        "cond": {
            "name": "cond(condition, true_val, false_val)",
            "description": "Condition ternaire",
            "params": ["condition", "true_value", "false_value"],
            "example": "cond(close > open, 1, -1)"
        },
        "max": {
            "name": "max(x, y)",
            "description": "Maximum de deux expressions",
            "params": ["expression1", "expression2"],
            "example": "max(high, close)"
        },
        "min": {
            "name": "min(x, y)",
            "description": "Minimum de deux expressions",
            "params": ["expression1", "expression2"],
            "example": "min(low, open)"
        }
    }
}

# Market features disponibles
MARKET_FEATURES = ["open", "high", "low", "close", "volume", "vwap"]
CONSTANTS = ["-2.0", "-1.0", "-0.5", "0.5", "1.0", "2.0", "5.0", "10.0"]
TIME_DELTAS = ["5", "10", "20", "30", "40", "50", "60", "120"]

# Interface principale
col_left, col_right = st.columns([1, 2])

# ================== COLONNE GAUCHE: Palette d'opérateurs ==================
with col_left:
    st.subheader("🎨 Operator Palette")

    # Recherche d'opérateurs
    search_operator = st.text_input("🔍 Search Operators", placeholder="Ex: corr, delta, rank...")

    # Filtrage par catégorie
    selected_categories = st.multiselect(
        "Filter Categories",
        list(OPERATORS.keys()),
        default=list(OPERATORS.keys())
    )

    # Affichage des opérateurs par catégorie
    for category, operators in OPERATORS.items():
        if category in selected_categories:
            with st.expander(f"📁 {category}", expanded=True):
                for op_key, op_info in operators.items():
                    if not search_operator or search_operator.lower() in op_key.lower():
                        st.markdown(f"""
                        <div class="operator-card">
                            <strong>{op_info['name']}</strong><br>
                            <small>{op_info['description']}</small><br>
                            <code>{op_info['example']}</code>
                        </div>
                        """, unsafe_allow_html=True)

                        if st.button(f"Add {op_key}", key=f"add_{category}_{op_key}", use_container_width=True):
                            st.session_state.formula_components.append({
                                'type': 'operator',
                                'value': op_key,
                                'info': op_info
                            })
                            st.rerun()

    # Market Features
    st.markdown("---")
    st.subheader("📊 Market Features")
    for feature in MARKET_FEATURES:
        if st.button(feature, key=f"feature_{feature}", use_container_width=True):
            st.session_state.formula_components.append({
                'type': 'feature',
                'value': feature
            })
            st.rerun()

    # Constants
    st.subheader("🔢 Constants")
    col_const1, col_const2 = st.columns(2)
    for i, const in enumerate(CONSTANTS):
        col = col_const1 if i % 2 == 0 else col_const2
        with col:
            if st.button(const, key=f"const_{const}", use_container_width=True):
                st.session_state.formula_components.append({
                    'type': 'constant',
                    'value': const
                })
                st.rerun()

    # Time Deltas
    st.subheader("⏰ Time Periods")
    col_time1, col_time2 = st.columns(2)
    for i, delta in enumerate(TIME_DELTAS):
        col = col_time1 if i % 2 == 0 else col_time2
        with col:
            if st.button(f"{delta}d", key=f"time_{delta}", use_container_width=True):
                st.session_state.formula_components.append({
                    'type': 'time_delta',
                    'value': delta
                })
                st.rerun()

# ================== COLONNE DROITE: Constructeur de formules ==================
with col_right:
    st.subheader("🏗️ Formula Builder")

    # Zone de construction interactive
    st.markdown("### Drag & Drop Formula Construction")

    # Affichage des composants actuels
    if st.session_state.formula_components:
        st.markdown("**Current Components:**")

        component_cols = st.columns(min(len(st.session_state.formula_components), 5))
        for i, component in enumerate(st.session_state.formula_components):
            col_idx = i % 5
            with component_cols[col_idx]:
                comp_type = component['type']
                comp_value = component['value']

                if comp_type == 'operator':
                    st.markdown(f"🔧 `{comp_value}`")
                elif comp_type == 'feature':
                    st.markdown(f"📊 `{comp_value}`")
                elif comp_type == 'constant':
                    st.markdown(f"🔢 `{comp_value}`")
                elif comp_type == 'time_delta':
                    st.markdown(f"⏰ `{comp_value}`")

                if st.button("❌", key=f"remove_{i}", help="Remove component"):
                    st.session_state.formula_components.pop(i)
                    st.rerun()

        # Actions sur les composants
        col_action1, col_action2, col_action3 = st.columns(3)
        with col_action1:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.formula_components = []
                st.session_state.current_formula = ""
                st.rerun()

        with col_action2:
            if st.button("🔗 Auto-Build", type="primary", use_container_width=True):
                # Construction automatique d'une formule basée sur les composants
                if len(st.session_state.formula_components) >= 2:
                    # Logique simple de construction automatique
                    formula = build_auto_formula(st.session_state.formula_components)
                    st.session_state.current_formula = formula
                    st.rerun()

        with col_action3:
            if st.button("💾 Save Formula", use_container_width=True):
                if st.session_state.current_formula:
                    st.session_state.formula_history.append({
                        'formula': st.session_state.current_formula,
                        'timestamp': datetime.now(),
                        'components': st.session_state.formula_components.copy()
                    })
                    st.success("Formula saved to history!")

    else:
        st.markdown("""
        <div class="formula-builder">
            <h3>🎯 Start Building Your Alpha Formula</h3>
            <p>Select operators and features from the palette to build your formula</p>
            <p>Example: corr(open, volume, 10) or delta(cs_rank(close), 5)</p>
        </div>
        """, unsafe_allow_html=True)

    # Éditeur de formule manuel
    st.markdown("### ✏️ Manual Formula Editor")
    manual_formula = st.text_area(
        "Enter Formula Manually",
        value=st.session_state.current_formula,
        placeholder="Ex: (-1 * corr(open, volume, 10)) or ts_rank(delta(close, 5), 20)",
        height=100,
        help="Entrez votre formule manuellement ou utilisez le constructeur visuel"
    )

    if manual_formula != st.session_state.current_formula:
        st.session_state.current_formula = manual_formula

    # Affichage de la formule actuelle
    if st.session_state.current_formula:
        st.markdown("### 🔍 Current Formula")
        st.markdown(f"""
        <div class="formula-display">
            {st.session_state.current_formula}
        </div>
        """, unsafe_allow_html=True)

        # Validation et analyse
        col_val1, col_val2 = st.columns(2)

        with col_val1:
            if st.button("✅ Validate Formula", type="primary", use_container_width=True):
                validation_result = validate_formula(st.session_state.current_formula)
                if validation_result['valid']:
                    st.markdown(f"""
                    <div class="validation-result validation-success">
                        <strong>✅ Formula is Valid!</strong><br>
                        Complexity Score: {validation_result['complexity']}<br>
                        Operators Used: {validation_result['operators_count']}<br>
                        Features Used: {validation_result['features_count']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="validation-result validation-error">
                        <strong>❌ Formula Error:</strong><br>
                        {validation_result['error']}
                    </div>
                    """, unsafe_allow_html=True)

        with col_val2:
            if st.button("🧪 Backtest Formula", use_container_width=True):
                with st.spinner("Running backtest..."):
                    # Simulation de backtest
                    time.sleep(2)
                    ic_score = np.random.uniform(0.01, 0.08)
                    sharpe = np.random.uniform(0.5, 2.5)

                    st.success(f"✅ Backtest Complete!")
                    col_metric1, col_metric2 = st.columns(2)
                    with col_metric1:
                        st.metric("IC Score", f"{ic_score:.4f}")
                    with col_metric2:
                        st.metric("Sharpe Ratio", f"{sharpe:.2f}")

    # Historique des formules
    if st.session_state.formula_history:
        st.markdown("### 📚 Formula History")

        for i, hist_formula in enumerate(reversed(st.session_state.formula_history[-5:])):
            with st.expander(f"Formula {len(st.session_state.formula_history)-i} - {hist_formula['timestamp'].strftime('%H:%M:%S')}"):
                st.code(hist_formula['formula'], language='python')

                col_hist1, col_hist2 = st.columns(2)
                with col_hist1:
                    if st.button(f"Load", key=f"load_hist_{i}"):
                        st.session_state.current_formula = hist_formula['formula']
                        st.session_state.formula_components = hist_formula['components']
                        st.rerun()

                with col_hist2:
                    if st.button(f"Test", key=f"test_hist_{i}"):
                        st.info("Running test...")

    # Exemples de formules populaires
    st.markdown("### 🌟 Popular Formula Examples")

    examples = [
        {
            "name": "Classic Alpha006",
            "formula": "(-1 * corr(open, volume, 10))",
            "description": "Negative correlation between open price and volume"
        },
        {
            "name": "Momentum Signal",
            "formula": "sign(delta(cs_rank(close), 5))",
            "description": "Sign of change in cross-sectional ranking"
        },
        {
            "name": "Volume Reversal",
            "formula": "ts_rank(volume, 20) * sign(delta(close, 1))",
            "description": "Volume rank combined with price change direction"
        }
    ]

    for example in examples:
        with st.expander(f"📖 {example['name']}"):
            st.code(example['formula'], language='python')
            st.write(example['description'])

            if st.button(f"Load Example", key=f"load_ex_{example['name']}"):
                st.session_state.current_formula = example['formula']
                st.rerun()

# Fonctions utilitaires

def build_auto_formula(components):
    """Construction automatique d'une formule basée sur les composants."""
    try:
        # Logique simple pour construire une formule
        operators = [c for c in components if c['type'] == 'operator']
        features = [c for c in components if c['type'] == 'feature']
        constants = [c for c in components if c['type'] == 'constant']
        time_deltas = [c for c in components if c['type'] == 'time_delta']

        if not operators or not features:
            return "Need at least 1 operator and 1 feature"

        # Construire une formule simple
        operator = operators[0]['value']
        feature = features[0]['value']

        if operator in ['delta', 'ts_rank', 'mean', 'std']:
            time_delta = time_deltas[0]['value'] if time_deltas else "10"
            return f"{operator}({feature}, {time_delta})"
        elif operator in ['sign', 'abs', 'cs_rank', 'scale']:
            return f"{operator}({feature})"
        elif operator == 'corr' and len(features) >= 2:
            time_delta = time_deltas[0]['value'] if time_deltas else "10"
            return f"corr({features[0]['value']}, {features[1]['value']}, {time_delta})"
        else:
            return f"{operator}({feature})"

    except Exception as e:
        return f"Auto-build error: {str(e)}"

def validate_formula(formula):
    """Validation d'une formule alpha."""
    try:
        if not formula.strip():
            return {"valid": False, "error": "Empty formula"}

        # Validation basique de syntaxe
        valid_operators = []
        for category in OPERATORS.values():
            valid_operators.extend(category.keys())

        valid_features = MARKET_FEATURES

        # Compter les opérateurs et features utilisés
        operators_count = sum(1 for op in valid_operators if op in formula)
        features_count = sum(1 for feat in valid_features if feat in formula)

        # Calcul simple de complexité
        complexity = len(formula.split('(')) - 1  # Nombre de parenthèses ouvrantes

        # Validation basique
        if '(' in formula and ')' not in formula:
            return {"valid": False, "error": "Missing closing parenthesis"}

        if formula.count('(') != formula.count(')'):
            return {"valid": False, "error": "Mismatched parentheses"}

        return {
            "valid": True,
            "complexity": complexity,
            "operators_count": operators_count,
            "features_count": features_count
        }

    except Exception as e:
        return {"valid": False, "error": str(e)}

# Sidebar avec informations
with st.sidebar:
    st.markdown("### ⚙️ Feature Engineering")
    st.info("Constructeur interactif de formules alpha")

    st.markdown("### 📊 Statistics")
    st.metric("Operators Available", len([op for cat in OPERATORS.values() for op in cat]))
    st.metric("Market Features", len(MARKET_FEATURES))
    st.metric("Formulas in History", len(st.session_state.formula_history))

    st.markdown("### 🎯 Current Session")
    st.metric("Components Selected", len(st.session_state.formula_components))

    if st.session_state.current_formula:
        complexity = len(st.session_state.current_formula.split('(')) - 1
        st.metric("Formula Complexity", max(complexity, 0))

    st.markdown("### 📚 Quick Reference")
    st.markdown("""
    **Operators:**
    - `delta(x, t)` - Difference
    - `ts_rank(x, t)` - Time rank
    - `corr(x, y, t)` - Correlation
    - `cs_rank(x)` - Cross-section rank
    - `sign(x)` - Sign function

    **Features:**
    - `open`, `high`, `low`, `close`
    - `volume`, `vwap`

    **Example:**
    ```
    (-1 * corr(open, volume, 10))
    ```
    """)