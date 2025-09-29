"""
Configuration et fixtures pour les tests de l'interface utilisateur.
Fournit des mocks pour Streamlit et des données de test réalistes.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import sys
from typing import Dict, Any, List
import json


# Mock Streamlit pour permettre l'import des modules UI
class MockStreamlit:
    """Mock complet de Streamlit pour les tests."""

    def __init__(self):
        self.session_state = {}
        self._components = []

    def __getattr__(self, name):
        """Retourne un mock pour tout attribut Streamlit."""
        if name == 'session_state':
            return self.session_state
        return MagicMock()

    def set_page_config(self, **kwargs):
        return MagicMock()

    def tabs(self, tab_names):
        return [MagicMock() for _ in tab_names]

    def columns(self, cols):
        return [MagicMock() for _ in range(cols)]

    def selectbox(self, label, options, **kwargs):
        return options[0] if options else None

    def multiselect(self, label, options, **kwargs):
        default = kwargs.get('default', [])
        return default if default else options[:1] if options else []

    def text_area(self, label, **kwargs):
        return kwargs.get('value', '')

    def button(self, label, **kwargs):
        return False

    def metric(self, label, value, **kwargs):
        return MagicMock()

    def plotly_chart(self, fig, **kwargs):
        return MagicMock()

    def dataframe(self, df, **kwargs):
        return MagicMock()


@pytest.fixture(scope="session", autouse=True)
def mock_streamlit():
    """Mock Streamlit globalement pour tous les tests UI."""
    mock_st = MockStreamlit()

    # Mock les imports Streamlit
    with patch.dict('sys.modules', {
        'streamlit': mock_st,
        'plotly.graph_objects': MagicMock(),
        'plotly.express': MagicMock(),
        'plotly.subplots': MagicMock()
    }):
        yield mock_st


@pytest.fixture
def sample_backtest_config():
    """Configuration de backtest d'exemple pour les tests."""
    return {
        'strategy_type': 'DMN LSTM Strategy',
        'parameters': {
            'window_size': 64,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'signal_threshold': 0.1
        },
        'data_config': {
            'symbols': ['BTC/USDT'],
            'timeframe': '1h',
            'start_date': '2024-01-01',
            'end_date': '2024-12-31'
        },
        'risk_config': {
            'max_position_size': 1.0,
            'stop_loss': 0.05,
            'take_profit': 0.10
        }
    }


@pytest.fixture
def sample_backtest_results():
    """Résultats de backtest réalistes pour les tests."""
    np.random.seed(42)  # Pour reproductibilité

    # Génération de données de performance réalistes
    days = 252
    dates = pd.date_range('2024-01-01', periods=days, freq='D')

    # Simulation returns avec drift positif
    daily_returns = np.random.normal(0.0008, 0.02, days)
    daily_returns[0] = 0

    # Equity curve
    cumulative_returns = np.cumprod(1 + daily_returns)
    equity_curve = 10000 * cumulative_returns

    # Drawdown calculation
    rolling_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - rolling_max) / rolling_max * 100

    # Performance metrics
    total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
    annualized_return = ((equity_curve[-1] / equity_curve[0]) ** (252/days) - 1) * 100
    volatility = np.std(daily_returns) * np.sqrt(252) * 100

    # Sharpe ratio
    risk_free_rate = 0.02
    excess_returns = daily_returns - risk_free_rate/252
    sharpe_ratio = np.mean(excess_returns) / np.std(daily_returns) * np.sqrt(252)

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': np.min(drawdown),
        'sortino_ratio': sharpe_ratio * 1.2,
        'calmar_ratio': annualized_return / abs(np.min(drawdown)) if np.min(drawdown) != 0 else 0,
        'var_95': np.percentile(daily_returns, 5) * 100,
        'cvar_95': np.mean(daily_returns[daily_returns <= np.percentile(daily_returns, 5)]) * 100,
        'win_rate': 55.0,
        'total_trades': 120,
        'profit_factor': 1.45,
        'avg_trade': total_return / 120,
        'avg_win': 2.1,
        'avg_loss': -1.8,
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
            'values': (10000 * np.cumprod(1 + np.random.normal(0.0005, 0.015, days))).tolist()
        },
        'trades': [
            {
                'entry_time': '2024-01-15 10:30:00',
                'exit_time': '2024-01-15 14:20:00',
                'symbol': 'BTC/USDT',
                'side': 'long',
                'entry_price': 42000.0,
                'exit_price': 42850.0,
                'quantity': 0.1,
                'pnl': 85.0,
                'pnl_pct': 2.02
            },
            {
                'entry_time': '2024-01-16 09:15:00',
                'exit_time': '2024-01-16 16:45:00',
                'symbol': 'BTC/USDT',
                'side': 'short',
                'entry_price': 41800.0,
                'exit_price': 41200.0,
                'quantity': 0.15,
                'pnl': 90.0,
                'pnl_pct': 1.44
            }
        ]
    }


@pytest.fixture
def crypto_market_data():
    """Données de marché crypto réalistes pour les tests."""
    np.random.seed(42)

    days = 365
    dates = pd.date_range('2024-01-01', periods=days, freq='D')

    # Simulation prix BTC avec volatilité réaliste
    initial_price = 42000.0
    daily_returns = np.random.normal(0.001, 0.04, days)  # Volatilité crypto réaliste
    prices = initial_price * np.cumprod(1 + daily_returns)

    # OHLCV data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # High/Low basé sur volatilité intraday
        daily_vol = abs(daily_returns[i]) * price
        high = price + np.random.uniform(0, daily_vol)
        low = price - np.random.uniform(0, daily_vol)
        open_price = prices[i-1] if i > 0 else price
        close_price = price
        volume = np.random.uniform(1000, 5000)  # Volume en BTC

        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })

    return pd.DataFrame(data)


@pytest.fixture
def extreme_market_scenarios():
    """Scénarios de marché extrêmes pour stress testing."""
    scenarios = {
        'flash_crash': {
            'description': 'Crash de 30% en 1 heure',
            'returns': [-0.30] + [0.01] * 23,  # 30% crash puis récupération
            'volume_spike': 10.0  # 10x volume normal
        },
        'pump_dump': {
            'description': 'Pump 50% puis dump 40%',
            'returns': [0.50, -0.40] + [0.005] * 22,
            'volume_spike': 8.0
        },
        'low_liquidity': {
            'description': 'Période de faible liquidité',
            'returns': [0.001] * 24,  # Très faible mouvement
            'volume_factor': 0.1  # 10% du volume normal
        },
        'high_volatility': {
            'description': 'Volatilité extrême bidirectionnelle',
            'returns': [0.15, -0.12, 0.18, -0.14] * 6,
            'volume_spike': 5.0
        }
    }

    return scenarios


@pytest.fixture
def mock_session_state():
    """Mock session state Streamlit avec données de test."""
    return {
        'backtest_queue': [],
        'backtest_results': {},
        'current_backtest': None,
        'bt_manager_initialized': False,
        'active_backtests': {},
        'validation_results': {}
    }


@pytest.fixture
def strategy_templates():
    """Templates de stratégies pour les tests de configuration."""
    return {
        "DMN LSTM Strategy": {
            "default_params": {
                "window_size": 64,
                "hidden_size": 128,
                "num_layers": 2,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "signal_threshold": 0.1
            },
            "param_ranges": {
                "window_size": [32, 64, 128],
                "hidden_size": [64, 128, 256],
                "learning_rate": [0.0001, 0.001, 0.01],
                "signal_threshold": [0.05, 0.1, 0.2]
            },
            "description": "Deep Market Networks avec LSTM pour prédiction de trends"
        },
        "Adaptive Mean Reversion": {
            "default_params": {
                "lookback_short": 10,
                "lookback_long": 50,
                "z_entry_base": 1.0,
                "z_exit_base": 0.2,
                "regime_window": 252
            },
            "param_ranges": {
                "lookback_short": [5, 10, 20],
                "lookback_long": [30, 50, 100],
                "z_entry_base": [0.5, 1.0, 2.0],
                "z_exit_base": [0.1, 0.2, 0.5]
            },
            "description": "Mean reversion adaptatif avec détection de régimes"
        },
        "RL Alpha Generator": {
            "default_params": {
                "state_dim": 50,
                "action_dim": 42,
                "learning_rate": 0.0003,
                "epsilon": 0.1,
                "batch_size": 32
            },
            "param_ranges": {
                "learning_rate": [0.0001, 0.0003, 0.001],
                "epsilon": [0.05, 0.1, 0.2],
                "batch_size": [16, 32, 64]
            },
            "description": "Génération d'alphas via Reinforcement Learning"
        }
    }


@pytest.fixture
def walk_forward_config():
    """Configuration Walk-Forward pour les tests."""
    return {
        'train_window': 180,  # 6 mois
        'test_window': 30,    # 1 mois
        'purge_days': 5,      # 5 jours de purge
        'step_size': 15,      # Avancement 15 jours
        'min_periods': 60,    # Minimum 2 mois de données
        'start_date': '2024-01-01',
        'end_date': '2024-12-31'
    }


@pytest.fixture
def monte_carlo_config():
    """Configuration Monte Carlo pour les tests."""
    return {
        'num_simulations': 1000,
        'confidence_levels': [0.90, 0.95, 0.99],
        'bootstrap_method': 'standard',
        'stress_scenarios': ['normal', 'high_vol', 'crash'],
        'random_seed': 42
    }


@pytest.fixture
def performance_metrics_expected():
    """Métriques de performance attendues pour validation."""
    return {
        'return_metrics': [
            'total_return', 'annualized_return', 'volatility',
            'best_month', 'worst_month'
        ],
        'risk_metrics': [
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'max_drawdown', 'var_95', 'cvar_95'
        ],
        'trade_metrics': [
            'total_trades', 'win_rate', 'profit_factor',
            'avg_trade', 'avg_win', 'avg_loss'
        ],
        'advanced_metrics': [
            'information_ratio', 'tracking_error', 'beta',
            'alpha', 'skewness', 'kurtosis'
        ]
    }


@pytest.fixture
def ui_component_mocks():
    """Mocks spécialisés pour les composants UI."""

    class ComponentMocks:
        def __init__(self):
            self.configurator = MagicMock()
            self.analyzer = MagicMock()
            self.walk_forward = MagicMock()
            self.monte_carlo = MagicMock()
            self.analytics = MagicMock()
            self.integration_manager = MagicMock()

        def setup_configurator_mocks(self, config_data):
            """Configure les mocks pour le configurateur."""
            self.configurator.render_configuration_section.return_value = config_data
            self.configurator._load_strategy_templates.return_value = {
                'DMN LSTM Strategy': {'default_params': {'window_size': 64}}
            }

        def setup_analyzer_mocks(self, results_data):
            """Configure les mocks pour l'analyseur."""
            self.analyzer.calculate_all_metrics.return_value = results_data
            self.analyzer.render_performance_summary.return_value = None

        def setup_integration_mocks(self, pipeline_data):
            """Configure les mocks pour le gestionnaire d'intégration."""
            self.integration_manager.create_integrated_workflow.return_value = pipeline_data
            self.integration_manager._execute_integrated_pipeline.return_value = None

    return ComponentMocks()


# Utilitaires de test pour l'UI
class UITestUtils:
    """Utilitaires pour faciliter les tests UI."""

    @staticmethod
    def assert_component_rendered(mock_streamlit, component_name):
        """Vérifie qu'un composant a été rendu."""
        # Cette fonction peut être étendue pour vérifier des appels spécifiques
        assert hasattr(mock_streamlit, component_name)

    @staticmethod
    def assert_metrics_displayed(mock_streamlit, expected_metrics):
        """Vérifie que les métriques attendues sont affichées."""
        # Simulation de vérification des métriques
        return True

    @staticmethod
    def simulate_user_interaction(mock_streamlit, action, **kwargs):
        """Simule une interaction utilisateur."""
        if action == 'select_strategy':
            return kwargs.get('strategy', 'DMN LSTM Strategy')
        elif action == 'click_button':
            return True
        elif action == 'upload_data':
            return kwargs.get('data', None)
        return None

    @staticmethod
    def validate_chart_data(chart_data, expected_fields):
        """Valide les données de graphique."""
        if not chart_data:
            return False

        for field in expected_fields:
            if field not in chart_data:
                return False
        return True


@pytest.fixture
def ui_test_utils():
    """Fixture pour les utilitaires de test UI."""
    return UITestUtils()