"""
ML Utils - Utilitaires pour machine learning et recherche quantitative
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json

class MLUtils:
    """Utilitaires pour machine learning et visualisations."""

    @staticmethod
    def generate_sample_ohlcv(periods: int = 1000, start_price: float = 50000) -> pd.DataFrame:
        """Génère des données OHLCV simulées pour tests."""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='H')

        # Simulation de prix avec random walk
        returns = np.random.randn(periods) * 0.02
        prices = start_price * np.cumprod(1 + returns)

        # Génération OHLCV réaliste
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            volatility = np.random.uniform(0.005, 0.03)
            high = price * (1 + volatility * np.random.uniform(0, 1))
            low = price * (1 - volatility * np.random.uniform(0, 1))
            open_price = prices[i-1] if i > 0 else price
            close = price

            volume = np.random.uniform(100, 1000) * (1 + abs(returns[i]) * 10)
            vwap = (high + low + close) / 3

            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'vwap': vwap
            })

        return pd.DataFrame(data)

    @staticmethod
    def calculate_alpha_metrics(returns: np.ndarray, benchmark_returns: Optional[np.ndarray] = None) -> Dict:
        """Calcule les métriques de performance d'une alpha."""
        if benchmark_returns is None:
            benchmark_returns = np.random.randn(len(returns)) * 0.01

        # Métriques de base
        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # Information Coefficient
        ic = np.corrcoef(returns[1:], benchmark_returns[1:])[0, 1] if len(returns) > 1 else 0

        # Drawdown
        cumulative = np.cumprod(1 + returns)
        rolling_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns)

        # Win rate
        win_rate = np.mean(returns > 0)

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Sortino ratio
        negative_returns = returns[returns < 0]
        downside_volatility = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'ic': ic,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio
        }

    @staticmethod
    def create_performance_chart(returns: np.ndarray, title: str = "Performance",
                               benchmark_returns: Optional[np.ndarray] = None) -> go.Figure:
        """Crée un graphique de performance cumulative."""
        dates = pd.date_range(start='2023-01-01', periods=len(returns), freq='D')
        cumulative_returns = np.cumprod(1 + returns) - 1

        fig = go.Figure()

        # Performance de l'alpha
        fig.add_trace(go.Scatter(
            x=dates,
            y=cumulative_returns * 100,
            mode='lines',
            name='Alpha Performance',
            line=dict(color='#00ff88', width=2)
        ))

        # Benchmark si fourni
        if benchmark_returns is not None:
            benchmark_cumulative = np.cumprod(1 + benchmark_returns) - 1
            fig.add_trace(go.Scatter(
                x=dates,
                y=benchmark_cumulative * 100,
                mode='lines',
                name='Benchmark',
                line=dict(color='#666666', width=1, dash='dash')
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            template='plotly_dark',
            height=400,
            hovermode='x unified'
        )

        return fig

    @staticmethod
    def create_drawdown_chart(returns: np.ndarray) -> go.Figure:
        """Crée un graphique de drawdown."""
        dates = pd.date_range(start='2023-01-01', periods=len(returns), freq='D')
        cumulative = np.cumprod(1 + returns)
        rolling_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - rolling_max) / rolling_max * 100

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dates,
            y=drawdowns,
            mode='lines',
            name='Drawdown',
            line=dict(color='#ff6b6b', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.3)'
        ))

        fig.update_layout(
            title="Drawdown Analysis",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template='plotly_dark',
            height=300
        )

        return fig

    @staticmethod
    def create_returns_distribution(returns: np.ndarray) -> go.Figure:
        """Crée un histogramme de distribution des returns."""
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=returns * 100,
            nbinsx=50,
            name='Returns Distribution',
            marker_color='#00ff88',
            opacity=0.7
        ))

        # Ligne de moyenne
        mean_return = np.mean(returns) * 100
        fig.add_vline(
            x=mean_return,
            line_dash="dash",
            line_color="#ff6b6b",
            annotation_text=f"Mean: {mean_return:.2f}%"
        )

        fig.update_layout(
            title="Returns Distribution",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            template='plotly_dark',
            height=350
        )

        return fig

    @staticmethod
    def create_rolling_metrics_chart(returns: np.ndarray, window: int = 30) -> go.Figure:
        """Crée un graphique des métriques mobiles."""
        dates = pd.date_range(start='2023-01-01', periods=len(returns), freq='D')

        # Calcul des métriques mobiles
        rolling_sharpe = []
        rolling_volatility = []

        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            vol = np.std(window_returns) * np.sqrt(252)
            sharpe = (np.mean(window_returns) * 252) / vol if vol > 0 else 0

            rolling_sharpe.append(sharpe)
            rolling_volatility.append(vol * 100)

        rolling_dates = dates[window:]

        fig = go.Figure()

        # Sharpe ratio mobile
        fig.add_trace(go.Scatter(
            x=rolling_dates,
            y=rolling_sharpe,
            mode='lines',
            name='Rolling Sharpe Ratio',
            line=dict(color='#00ff88', width=2)
        ))

        # Volatilité mobile (axe secondaire)
        fig.add_trace(go.Scatter(
            x=rolling_dates,
            y=rolling_volatility,
            mode='lines',
            name='Rolling Volatility (%)',
            line=dict(color='#ff6b6b', width=2),
            yaxis='y2'
        ))

        fig.update_layout(
            title=f"Rolling Metrics ({window}-day window)",
            xaxis_title="Date",
            yaxis=dict(
                title="Sharpe Ratio",
                titlefont=dict(color="#00ff88"),
                tickfont=dict(color="#00ff88")
            ),
            yaxis2=dict(
                title="Volatility (%)",
                titlefont=dict(color="#ff6b6b"),
                tickfont=dict(color="#ff6b6b"),
                anchor="x",
                overlaying="y",
                side="right"
            ),
            template='plotly_dark',
            height=400,
            hovermode='x unified'
        )

        return fig

    @staticmethod
    def format_performance_metrics(metrics: Dict) -> Dict[str, str]:
        """Formate les métriques de performance pour affichage."""
        return {
            "Total Return": f"{metrics['total_return']:.1%}",
            "Annualized Return": f"{metrics['annualized_return']:.1%}",
            "Volatility": f"{metrics['volatility']:.1%}",
            "Sharpe Ratio": f"{metrics['sharpe_ratio']:.2f}",
            "IC Score": f"{metrics['ic']:.4f}",
            "Max Drawdown": f"{metrics['max_drawdown']:.1%}",
            "Win Rate": f"{metrics['win_rate']:.1%}",
            "Calmar Ratio": f"{metrics['calmar_ratio']:.2f}",
            "Sortino Ratio": f"{metrics['sortino_ratio']:.2f}"
        }

    @staticmethod
    def create_correlation_heatmap(data: pd.DataFrame, features: List[str]) -> go.Figure:
        """Crée une heatmap de corrélation."""
        if not all(feat in data.columns for feat in features):
            features = [feat for feat in features if feat in data.columns]

        if len(features) < 2:
            return go.Figure().add_annotation(text="Not enough features for correlation")

        corr_matrix = data[features].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10}
        ))

        fig.update_layout(
            title="Feature Correlation Matrix",
            template='plotly_dark',
            height=400
        )

        return fig

    @staticmethod
    def simulate_rl_training_data(episodes: int = 1000) -> Dict:
        """Simule des données d'entraînement RL."""
        episodes_range = range(episodes)

        # Simulation réaliste de training RL
        base_reward = -0.5
        rewards = []
        ic_scores = []
        exploration_rates = []

        for ep in episodes_range:
            # Amélioration progressive avec du bruit
            progress = ep / episodes
            reward = base_reward + progress * 2.0 + np.random.randn() * 0.3
            ic = max(0, progress * 0.08 + np.random.randn() * 0.02)
            exploration = max(0.01, 1.0 - progress * 0.9 + np.random.randn() * 0.05)

            rewards.append(reward)
            ic_scores.append(ic)
            exploration_rates.append(exploration)

        return {
            'episodes': list(episodes_range),
            'rewards': rewards,
            'cumulative_rewards': np.cumsum(rewards),
            'ic_scores': ic_scores,
            'exploration_rates': exploration_rates,
            'best_ic': max(ic_scores),
            'final_reward': rewards[-1],
            'convergence_episode': int(episodes * 0.7)
        }

    @staticmethod
    def simulate_lstm_training_data(epochs: int = 100) -> Dict:
        """Simule des données d'entraînement LSTM."""
        epochs_range = range(1, epochs + 1)

        # Simulation réaliste de training LSTM
        train_losses = []
        val_losses = []
        accuracies = []

        initial_loss = 1.0
        for epoch in epochs_range:
            # Décroissance exponentielle avec plateaux
            progress = epoch / epochs
            train_loss = initial_loss * np.exp(-3 * progress) + np.random.randn() * 0.01
            val_loss = train_loss * 1.1 + np.random.randn() * 0.02

            # Accuracy croissante
            accuracy = 1 - np.exp(-2 * progress) + np.random.randn() * 0.02
            accuracy = max(0.5, min(0.95, accuracy))

            train_losses.append(max(0.001, train_loss))
            val_losses.append(max(0.001, val_loss))
            accuracies.append(accuracy)

        return {
            'epochs': list(epochs_range),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'accuracies': accuracies,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'final_accuracy': accuracies[-1],
            'best_val_loss': min(val_losses),
            'overfitting_start': epochs // 2 if val_losses[-1] > min(val_losses) * 1.1 else None
        }

    @staticmethod
    def create_feature_importance_chart(features: List[str], importance_scores: List[float]) -> go.Figure:
        """Crée un graphique d'importance des features."""
        # Trier par importance
        sorted_data = sorted(zip(features, importance_scores), key=lambda x: x[1], reverse=True)
        sorted_features, sorted_scores = zip(*sorted_data)

        fig = go.Figure(go.Bar(
            x=list(sorted_scores),
            y=list(sorted_features),
            orientation='h',
            marker_color='#00ff88',
            text=[f'{score:.3f}' for score in sorted_scores],
            textposition='auto'
        ))

        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            template='plotly_dark',
            height=400
        )

        return fig

    @staticmethod
    def validate_alpha_formula(formula: str) -> Dict:
        """Valide une formule alpha et retourne des métriques."""
        try:
            # Validation basique
            if not formula.strip():
                return {"valid": False, "error": "Empty formula"}

            if formula.count('(') != formula.count(')'):
                return {"valid": False, "error": "Mismatched parentheses"}

            # Analyse de complexité
            operators = ['corr', 'delta', 'ts_rank', 'cs_rank', 'sign', 'abs', 'mean', 'std']
            features = ['open', 'high', 'low', 'close', 'volume', 'vwap']

            used_operators = [op for op in operators if op in formula]
            used_features = [feat for feat in features if feat in formula]

            complexity = len(used_operators) * 2 + len(used_features) + formula.count('(')

            # Estimation de performance (simulée)
            estimated_ic = min(0.1, max(0.01, complexity * 0.01 + np.random.uniform(-0.02, 0.02)))

            return {
                "valid": True,
                "complexity": complexity,
                "operators_count": len(used_operators),
                "features_count": len(used_features),
                "estimated_ic": estimated_ic,
                "operators_used": used_operators,
                "features_used": used_features
            }

        except Exception as e:
            return {"valid": False, "error": str(e)}

class ResearchStateManager:
    """Gestionnaire d'état pour la recherche ML."""

    @staticmethod
    def init_research_session():
        """Initialise les variables de session pour la recherche."""
        defaults = {
            'rl_training_active': False,
            'lstm_training_active': False,
            'formula_components': [],
            'current_formula': "",
            'formula_history': [],
            'training_metrics': {},
            'model_library': [],
            'alpha_library': []
        }

        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    @staticmethod
    def get_training_status(model_type: str) -> Dict:
        """Retourne le statut d'entraînement d'un modèle."""
        key = f"{model_type}_training_active"
        start_key = f"{model_type}_start_time"

        if not st.session_state.get(key, False):
            return {"status": "stopped", "active": False}

        start_time = st.session_state.get(start_key)
        if not start_time:
            return {"status": "error", "active": False}

        elapsed = datetime.now() - start_time
        return {
            "status": "running",
            "active": True,
            "elapsed_time": elapsed,
            "start_time": start_time
        }

    @staticmethod
    def save_model_to_library(model_config: Dict, model_type: str):
        """Sauvegarde un modèle dans la bibliothèque."""
        if 'model_library' not in st.session_state:
            st.session_state.model_library = []

        model_entry = {
            **model_config,
            'type': model_type,
            'created_at': datetime.now(),
            'id': f"{model_type}_{len(st.session_state.model_library) + 1}"
        }

        st.session_state.model_library.append(model_entry)

    @staticmethod
    def get_model_library(model_type: Optional[str] = None) -> List[Dict]:
        """Retourne la bibliothèque de modèles."""
        library = st.session_state.get('model_library', [])

        if model_type:
            return [model for model in library if model.get('type') == model_type]

        return library