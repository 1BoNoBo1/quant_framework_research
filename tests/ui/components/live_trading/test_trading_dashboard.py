"""
Tests pour TradingDashboard component
"""

import pytest
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Ajouter le chemin du projet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from qframe.ui.streamlit_app.components.live_trading.trading_dashboard import TradingDashboard


@pytest.fixture
def clean_session_state():
    """Nettoie le session state avant chaque test."""
    if hasattr(st, 'session_state'):
        for key in list(st.session_state.keys()):
            if key.startswith(('dashboard_', 'price_', 'pnl_', 'active_')):
                delattr(st.session_state, key)


@pytest.fixture
def mock_streamlit():
    """Mock des fonctions Streamlit."""
    with patch.multiple(
        'streamlit',
        header=Mock(),
        button=Mock(return_value=False),
        columns=Mock(return_value=[Mock(), Mock(), Mock(), Mock(), Mock()]),
        metric=Mock(),
        caption=Mock(),
        subheader=Mock(),
        info=Mock(),
        plotly_chart=Mock(),
        dataframe=Mock(),
        rerun=Mock()
    ) as mocks:
        yield mocks


@pytest.fixture
def sample_dashboard_data():
    """Données d'exemple pour le dashboard."""
    return {
        'account_balance': 100000.0,
        'total_pnl': 1250.0,
        'daily_pnl': 340.0,
        'open_positions': 3,
        'active_orders': 2,
        'last_update': datetime.now()
    }


@pytest.fixture
def sample_price_data():
    """Données de prix d'exemple."""
    base_time = datetime.now() - timedelta(hours=1)
    return {
        'BTC/USDT': {
            'current_price': 43250.0,
            'history': [
                {
                    'timestamp': base_time + timedelta(minutes=i),
                    'price': 43000.0 + i * 10,
                    'volume': 1000000 + i * 50000,
                    'change_pct': 0.1 * i
                }
                for i in range(60)
            ],
            'change_24h': 2.5,
            'volume_24h': 500000000
        },
        'ETH/USDT': {
            'current_price': 2650.0,
            'history': [
                {
                    'timestamp': base_time + timedelta(minutes=i),
                    'price': 2600.0 + i * 2,
                    'volume': 800000 + i * 40000,
                    'change_pct': 0.08 * i
                }
                for i in range(60)
            ],
            'change_24h': -1.2,
            'volume_24h': 300000000
        }
    }


@pytest.fixture
def sample_pnl_history():
    """Historique PnL d'exemple."""
    base_time = datetime.now() - timedelta(hours=24)
    history = []

    for i in range(288):  # 24h en intervalles de 5 minutes
        timestamp = base_time + timedelta(minutes=5*i)
        pnl = np.random.normal(0, 50) * i / 100
        cumulative_pnl = sum([h.get('cumulative_pnl', 0) for h in history[:i]]) + pnl

        history.append({
            'timestamp': timestamp,
            'pnl': pnl,
            'cumulative_pnl': cumulative_pnl
        })

    return history


class TestTradingDashboard:
    """Tests pour la classe TradingDashboard."""

    def test_init(self, clean_session_state):
        """Test de l'initialisation."""
        dashboard = TradingDashboard()
        assert dashboard.refresh_interval == 3

    def test_init_session_state(self, clean_session_state):
        """Test de l'initialisation du session state."""
        dashboard = TradingDashboard()
        dashboard.init_session_state()

        # Vérifier que les clés nécessaires existent
        assert 'dashboard_data' in st.session_state
        assert 'price_data' in st.session_state
        assert 'pnl_history' in st.session_state

        # Vérifier la structure des données
        dashboard_data = st.session_state.dashboard_data
        assert 'account_balance' in dashboard_data
        assert 'total_pnl' in dashboard_data
        assert 'last_update' in dashboard_data

        # Vérifier que l'historique PnL est généré
        assert len(st.session_state.pnl_history) == 288  # 24h * 12 (5min intervals)

    def test_generate_initial_price_data(self):
        """Test de la génération des données de prix initiales."""
        dashboard = TradingDashboard()
        price_data = dashboard._generate_initial_price_data()

        # Vérifier que les symboles attendus sont présents
        expected_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT']
        for symbol in expected_symbols:
            assert symbol in price_data

        # Vérifier la structure des données pour BTC
        btc_data = price_data['BTC/USDT']
        assert 'current_price' in btc_data
        assert 'history' in btc_data
        assert 'change_24h' in btc_data
        assert 'volume_24h' in btc_data

        # Vérifier que l'historique a la bonne longueur
        assert len(btc_data['history']) == 60  # 1h de données

        # Vérifier la structure d'un point d'historique
        history_point = btc_data['history'][0]
        assert 'timestamp' in history_point
        assert 'price' in history_point
        assert 'volume' in history_point
        assert 'change_pct' in history_point

    @patch('streamlit.rerun')
    @patch('time.sleep')
    def test_render(self, mock_sleep, mock_rerun, mock_streamlit, clean_session_state, sample_dashboard_data):
        """Test du rendu principal."""
        # Setup
        st.session_state.dashboard_data = sample_dashboard_data
        st.session_state.price_data = {}
        st.session_state.pnl_history = []

        dashboard = TradingDashboard()

        # Mock button pour éviter la mise à jour
        with patch('streamlit.button', return_value=False):
            dashboard.render()

        # Vérifier que les fonctions Streamlit sont appelées
        mock_streamlit['header'].assert_called_with("🎯 Trading Dashboard")
        mock_streamlit['columns'].assert_called()

    def test_render_key_metrics(self, mock_streamlit, sample_dashboard_data):
        """Test du rendu des métriques clés."""
        st.session_state.dashboard_data = sample_dashboard_data

        dashboard = TradingDashboard()
        dashboard._render_key_metrics()

        # Vérifier que metric est appelé 5 fois (5 métriques)
        assert mock_streamlit['metric'].call_count >= 5

    def test_render_pnl_chart_empty(self, mock_streamlit):
        """Test du graphique PnL avec données vides."""
        st.session_state.pnl_history = []

        dashboard = TradingDashboard()
        dashboard._render_pnl_chart()

        mock_streamlit['info'].assert_called_with("Aucune donnée PnL disponible")

    def test_render_pnl_chart_with_data(self, mock_streamlit, sample_pnl_history):
        """Test du graphique PnL avec données."""
        st.session_state.pnl_history = sample_pnl_history

        dashboard = TradingDashboard()
        dashboard._render_pnl_chart()

        # Vérifier que plotly_chart est appelé
        mock_streamlit['plotly_chart'].assert_called()

    def test_render_market_overview(self, mock_streamlit, sample_price_data):
        """Test de l'aperçu marché."""
        st.session_state.price_data = sample_price_data

        dashboard = TradingDashboard()
        dashboard._render_market_overview()

        # Vérifier que dataframe est appelé
        mock_streamlit['dataframe'].assert_called()

    def test_render_active_positions_empty(self, mock_streamlit):
        """Test des positions actives quand aucune position."""
        st.session_state.active_positions = []

        dashboard = TradingDashboard()
        dashboard._render_active_positions()

        mock_streamlit['info'].assert_called_with("Aucune position ouverte")

    def test_generate_sample_positions(self, sample_price_data):
        """Test de la génération de positions d'exemple."""
        st.session_state.price_data = sample_price_data

        dashboard = TradingDashboard()
        positions = dashboard._generate_sample_positions()

        # Vérifier que des positions sont générées (probabiliste)
        assert isinstance(positions, list)

        # Si des positions sont générées, vérifier la structure
        if positions:
            position = positions[0]
            required_fields = ['Symbol', 'Side', 'Size', 'Entry Price', 'Current Price', 'PnL (%)', 'PnL ($)']
            for field in required_fields:
                assert field in position

    def test_update_real_time_data(self, sample_price_data, sample_pnl_history):
        """Test de la mise à jour des données temps réel."""
        st.session_state.price_data = sample_price_data
        st.session_state.pnl_history = sample_pnl_history
        st.session_state.active_positions = []

        dashboard = TradingDashboard()
        initial_btc_price = st.session_state.price_data['BTC/USDT']['current_price']

        dashboard._update_real_time_data()

        # Vérifier que le prix a été mis à jour (peut être égal par hasard)
        updated_btc_price = st.session_state.price_data['BTC/USDT']['current_price']
        assert isinstance(updated_btc_price, float)

        # Vérifier que l'historique des prix a été mis à jour
        btc_history = st.session_state.price_data['BTC/USDT']['history']
        assert len(btc_history) <= 60  # Limite à 60 minutes

        # Vérifier que last_update est mis à jour
        assert 'last_update' in st.session_state.dashboard_data

    def test_get_dashboard_summary(self, sample_dashboard_data):
        """Test du résumé dashboard."""
        st.session_state.dashboard_data = sample_dashboard_data

        dashboard = TradingDashboard()
        summary = dashboard.get_dashboard_summary()

        expected_keys = ['total_balance', 'total_pnl', 'daily_pnl', 'open_positions', 'active_orders', 'last_update']
        for key in expected_keys:
            assert key in summary

        assert summary['total_balance'] == sample_dashboard_data['account_balance']
        assert summary['total_pnl'] == sample_dashboard_data['total_pnl']


class TestTradingDashboardIntegration:
    """Tests d'intégration pour TradingDashboard."""

    def test_full_initialization_flow(self, clean_session_state):
        """Test du flux complet d'initialisation."""
        dashboard = TradingDashboard()

        # Vérifier que toutes les données sont initialisées
        assert 'dashboard_data' in st.session_state
        assert 'price_data' in st.session_state
        assert 'pnl_history' in st.session_state

        # Vérifier que les données ont du contenu
        assert len(st.session_state.price_data) > 0
        assert len(st.session_state.pnl_history) == 288

    def test_real_time_update_consistency(self, clean_session_state):
        """Test de la cohérence des mises à jour temps réel."""
        dashboard = TradingDashboard()

        # Capturer l'état initial
        initial_price_count = len(st.session_state.price_data)
        initial_pnl_count = len(st.session_state.pnl_history)

        # Effectuer une mise à jour
        dashboard._update_real_time_data()

        # Vérifier que la structure est préservée
        assert len(st.session_state.price_data) == initial_price_count

        # Vérifier que l'historique PnL est maintenu dans les limites
        assert len(st.session_state.pnl_history) <= 288  # Maximum 24h

    def test_price_data_structure_consistency(self, clean_session_state):
        """Test de la cohérence de la structure des données de prix."""
        dashboard = TradingDashboard()

        for symbol, data in st.session_state.price_data.items():
            # Vérifier la structure de base
            assert 'current_price' in data
            assert 'history' in data
            assert 'change_24h' in data
            assert 'volume_24h' in data

            # Vérifier que current_price est numérique et positif
            assert isinstance(data['current_price'], (int, float))
            assert data['current_price'] > 0

            # Vérifier la structure de l'historique
            for history_point in data['history']:
                assert 'timestamp' in history_point
                assert 'price' in history_point
                assert 'volume' in history_point
                assert 'change_pct' in history_point

                assert isinstance(history_point['price'], (int, float))
                assert history_point['price'] > 0

    @patch('streamlit.button', return_value=True)  # Simule click sur refresh
    def test_manual_refresh_trigger(self, mock_button, clean_session_state):
        """Test du déclenchement manuel de refresh."""
        dashboard = TradingDashboard()

        initial_update_time = st.session_state.dashboard_data['last_update']

        # Simuler un délai
        import time
        time.sleep(0.01)

        # Le render() devrait déclencher une mise à jour
        with patch.multiple(
            'streamlit',
            header=Mock(),
            columns=Mock(return_value=[Mock()] * 5),
            metric=Mock(),
            caption=Mock(),
            rerun=Mock()
        ):
            with patch('time.sleep'):  # Éviter le sleep dans le test
                with patch.object(dashboard, '_update_real_time_data') as mock_update:
                    dashboard.render()
                    mock_update.assert_called_once()


# Tests de performance
class TestTradingDashboardPerformance:
    """Tests de performance pour TradingDashboard."""

    def test_large_history_performance(self, clean_session_state):
        """Test des performances avec un gros historique."""
        dashboard = TradingDashboard()

        # Générer un gros historique
        large_history = []
        base_time = datetime.now() - timedelta(days=7)  # 7 jours

        for i in range(2016):  # 7 jours * 24h * 12 (5min intervals)
            large_history.append({
                'timestamp': base_time + timedelta(minutes=5*i),
                'pnl': np.random.normal(0, 50),
                'cumulative_pnl': i * 10
            })

        st.session_state.pnl_history = large_history

        # Tester que le rendu PnL ne plante pas
        with patch('streamlit.plotly_chart'):
            dashboard._render_pnl_chart()

    def test_many_positions_performance(self, clean_session_state, sample_price_data):
        """Test des performances avec beaucoup de positions."""
        st.session_state.price_data = sample_price_data

        # Créer beaucoup de positions simulées
        many_positions = []
        symbols = list(sample_price_data.keys())

        for i in range(50):  # 50 positions
            symbol = symbols[i % len(symbols)]
            many_positions.append({
                'Symbol': symbol,
                'Side': 'Long' if i % 2 == 0 else 'Short',
                'Size': f"{0.1 * (i + 1):.3f}",
                'Entry Price': f"${1000 + i * 10:,.2f}",
                'Current Price': f"${1020 + i * 10:,.2f}",
                'PnL (%)': f"{np.random.uniform(-5, 5):+.2f}%",
                'PnL ($)': f"{np.random.uniform(-100, 100):+.2f}"
            })

        st.session_state.active_positions = many_positions

        dashboard = TradingDashboard()

        # Tester que le rendu des positions ne plante pas
        with patch.multiple(
            'streamlit',
            dataframe=Mock(),
            button=Mock(return_value=False),
            info=Mock()
        ):
            dashboard._render_active_positions()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])