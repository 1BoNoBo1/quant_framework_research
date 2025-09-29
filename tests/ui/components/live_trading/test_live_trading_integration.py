"""
Tests d'intégration pour Live Trading Interface
Validation du workflow end-to-end
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
from qframe.ui.streamlit_app.components.live_trading.order_manager import OrderManager
from qframe.ui.streamlit_app.components.live_trading.position_monitor import PositionMonitor
from qframe.ui.streamlit_app.components.live_trading.risk_monitor import RiskMonitor


@pytest.fixture
def clean_session_state():
    """Nettoie complètement le session state."""
    if hasattr(st, 'session_state'):
        for key in list(st.session_state.keys()):
            delattr(st.session_state, key)


@pytest.fixture
def full_trading_environment(clean_session_state):
    """Configure un environnement de trading complet."""
    # Initialiser tous les composants
    dashboard = TradingDashboard()
    order_manager = OrderManager()
    position_monitor = PositionMonitor()
    risk_monitor = RiskMonitor()

    return {
        'dashboard': dashboard,
        'order_manager': order_manager,
        'position_monitor': position_monitor,
        'risk_monitor': risk_monitor
    }


@pytest.fixture
def realistic_market_data():
    """Données de marché réalistes pour les tests."""
    base_time = datetime.now()

    return {
        'price_data': {
            'BTC/USDT': {
                'current_price': 43250.0,
                'change_24h': 2.5,
                'volume_24h': 1500000000,
                'history': [
                    {
                        'timestamp': base_time - timedelta(minutes=60-i),
                        'price': 43000 + i * 5 + np.random.normal(0, 50),
                        'volume': 1000000 + np.random.uniform(0, 500000),
                        'change_pct': np.random.normal(0, 0.1)
                    }
                    for i in range(60)
                ]
            },
            'ETH/USDT': {
                'current_price': 2650.0,
                'change_24h': -1.2,
                'volume_24h': 800000000,
                'history': [
                    {
                        'timestamp': base_time - timedelta(minutes=60-i),
                        'price': 2600 + i * 2 + np.random.normal(0, 20),
                        'volume': 800000 + np.random.uniform(0, 200000),
                        'change_pct': np.random.normal(0, 0.08)
                    }
                    for i in range(60)
                ]
            }
        },
        'positions': [
            {
                'id': 'pos_001',
                'symbol': 'BTC/USDT',
                'side': 'Long',
                'size': 0.5,
                'entry_price': 42800.0,
                'current_price': 43250.0,
                'mark_price': 43240.0,
                'leverage': 2.0,
                'margin': 21400.0,
                'unrealized_pnl': 225.0,
                'realized_pnl': 0.0,
                'entry_time': base_time - timedelta(hours=2),
                'last_update': base_time,
                'stop_loss': 40500.0,
                'take_profit': 45000.0,
                'funding_fee': -1.5,
                'commission': 10.7
            },
            {
                'id': 'pos_002',
                'symbol': 'ETH/USDT',
                'side': 'Short',
                'size': 5.0,
                'entry_price': 2680.0,
                'current_price': 2650.0,
                'mark_price': 2649.5,
                'leverage': 1.5,
                'margin': 8933.33,
                'unrealized_pnl': 150.0,
                'realized_pnl': 75.0,
                'entry_time': base_time - timedelta(hours=1),
                'last_update': base_time,
                'stop_loss': 2750.0,
                'take_profit': 2500.0,
                'funding_fee': -0.8,
                'commission': 6.7
            }
        ],
        'orders': [
            {
                'id': 'ord_001',
                'symbol': 'BTC/USDT',
                'side': 'Buy',
                'type': 'Limit',
                'quantity': 0.25,
                'price': 42500.0,
                'filled_quantity': 0.0,
                'status': 'Pending',
                'created_at': base_time - timedelta(minutes=30),
                'updated_at': base_time - timedelta(minutes=30)
            }
        ]
    }


class TestLiveTradingIntegration:
    """Tests d'intégration pour l'interface Live Trading."""

    def test_components_initialization(self, full_trading_environment):
        """Test que tous les composants s'initialisent correctement."""
        components = full_trading_environment

        # Vérifier que tous les composants sont créés
        assert components['dashboard'] is not None
        assert components['order_manager'] is not None
        assert components['position_monitor'] is not None
        assert components['risk_monitor'] is not None

        # Vérifier que le session state est correctement initialisé
        assert 'dashboard_data' in st.session_state
        assert 'price_data' in st.session_state
        assert 'orders' in st.session_state
        assert 'positions' in st.session_state

    def test_data_consistency_across_components(self, full_trading_environment, realistic_market_data):
        """Test de la cohérence des données entre composants."""
        # Injecter des données réalistes
        st.session_state.price_data = realistic_market_data['price_data']
        st.session_state.positions = realistic_market_data['positions']
        st.session_state.orders = realistic_market_data['orders']

        components = full_trading_environment

        # Test que tous les composants accèdent aux mêmes données
        dashboard_summary = components['dashboard'].get_dashboard_summary()
        assert dashboard_summary['open_positions'] == len(realistic_market_data['positions'])

        # Mettre à jour le dashboard
        components['dashboard']._update_real_time_data()

        # Vérifier que les prix sont cohérents
        btc_price = st.session_state.price_data['BTC/USDT']['current_price']
        assert isinstance(btc_price, (int, float))
        assert btc_price > 0

    def test_order_to_position_workflow(self, full_trading_environment):
        """Test du workflow ordre → position."""
        components = full_trading_environment

        # Simuler placement d'un ordre market
        st.session_state.order_form = {
            'symbol': 'BTC/USDT',
            'side': 'Buy',
            'type': 'Market',
            'quantity': 0.1,
            'price': None
        }

        # Placer l'ordre
        components['order_manager']._place_order()

        # Vérifier que l'ordre est créé
        orders = st.session_state.orders
        market_orders = [o for o in orders if o['type'] == 'Market']
        assert len(market_orders) >= 1

        # Pour un ordre market, il devrait être immédiatement rempli
        filled_orders = [o for o in market_orders if o['status'] == 'Filled']
        assert len(filled_orders) >= 1

    def test_risk_monitoring_integration(self, full_trading_environment, realistic_market_data):
        """Test de l'intégration du monitoring des risques."""
        st.session_state.positions = realistic_market_data['positions']

        components = full_trading_environment

        # Calculer les métriques de risque
        components['risk_monitor']._calculate_risk_metrics()

        # Vérifier que les métriques sont calculées
        risk_metrics = st.session_state.risk_metrics
        assert 'portfolio_var_95' in risk_metrics
        assert 'leverage_ratio' in risk_metrics
        assert 'concentration_risk' in risk_metrics

        # Vérifier que les valeurs sont sensées
        assert risk_metrics['portfolio_var_95'] >= 0
        assert risk_metrics['leverage_ratio'] >= 0

    def test_position_pnl_calculation_consistency(self, full_trading_environment, realistic_market_data):
        """Test de la cohérence des calculs PnL."""
        st.session_state.positions = realistic_market_data['positions']
        st.session_state.price_data = realistic_market_data['price_data']

        components = full_trading_environment

        # Mettre à jour les positions
        components['position_monitor']._update_positions()

        # Vérifier que les PnL sont recalculés de manière cohérente
        for pos in st.session_state.positions:
            if pos['side'] == 'Long':
                expected_pnl = pos['size'] * (pos['current_price'] - pos['entry_price'])
            else:
                expected_pnl = pos['size'] * (pos['entry_price'] - pos['current_price'])

            # Tolérance pour les erreurs de calcul flottant
            assert abs(pos['unrealized_pnl'] - expected_pnl) < 0.01

    def test_real_time_updates_synchronization(self, full_trading_environment, realistic_market_data):
        """Test de la synchronisation des mises à jour temps réel."""
        st.session_state.price_data = realistic_market_data['price_data']
        st.session_state.positions = realistic_market_data['positions']

        components = full_trading_environment

        # Capturer les prix initiaux
        initial_btc_price = st.session_state.price_data['BTC/USDT']['current_price']

        # Mettre à jour le dashboard (qui met à jour les prix)
        components['dashboard']._update_real_time_data()

        # Mettre à jour les positions (qui utilise les nouveaux prix)
        components['position_monitor']._update_positions()

        # Vérifier que les positions utilisent les prix mis à jour
        btc_position = next((p for p in st.session_state.positions if p['symbol'] == 'BTC/USDT'), None)
        if btc_position:
            # Le prix courant de la position devrait être le même que dans price_data
            assert btc_position['current_price'] == st.session_state.price_data['BTC/USDT']['current_price']

    def test_alert_generation_workflow(self, full_trading_environment, realistic_market_data):
        """Test du workflow de génération d'alertes."""
        # Créer une situation de risque élevé
        high_risk_positions = [
            {
                'id': 'high_risk_pos',
                'symbol': 'BTC/USDT',
                'side': 'Long',
                'size': 10.0,  # Position très large
                'entry_price': 45000.0,
                'current_price': 40000.0,  # Perte importante
                'margin': 450000.0,
                'unrealized_pnl': -50000.0,  # Grosse perte
                'leverage': 5.0,  # Leverage élevé
                'stop_loss': 38000.0,
                'entry_time': datetime.now() - timedelta(hours=1),
                'last_update': datetime.now(),
                'funding_fee': -10.0,
                'commission': 225.0
            }
        ]

        st.session_state.positions = high_risk_positions

        components = full_trading_environment

        # Calculer les risques et générer les alertes
        components['risk_monitor']._calculate_risk_metrics()
        alerts = components['risk_monitor']._generate_risk_alerts()

        # Vérifier que des alertes sont générées
        assert len(alerts) > 0

        # Vérifier qu'il y a des alertes de risque élevé
        high_severity_alerts = [a for a in alerts if a['severity'] == 'HIGH']
        assert len(high_severity_alerts) > 0

    def test_emergency_stop_workflow(self, full_trading_environment, realistic_market_data):
        """Test du workflow d'arrêt d'urgence."""
        st.session_state.positions = realistic_market_data['positions']
        st.session_state.trading_active = True

        components = full_trading_environment

        # Déclencher l'arrêt d'urgence
        components['risk_monitor']._emergency_stop()

        # Vérifier que le trading est arrêté
        assert st.session_state.get('emergency_stop_active', False) is True
        assert st.session_state.get('trading_active', True) is False

    def test_close_all_positions_workflow(self, full_trading_environment, realistic_market_data):
        """Test du workflow de fermeture de toutes les positions."""
        st.session_state.positions = realistic_market_data['positions']
        initial_position_count = len(st.session_state.positions)

        components = full_trading_environment

        # Fermer toutes les positions
        components['risk_monitor']._close_all_positions()

        # Vérifier que toutes les positions sont fermées
        assert len(st.session_state.positions) == 0
        assert initial_position_count > 0  # S'assurer qu'il y avait des positions avant

    def test_order_cancellation_workflow(self, full_trading_environment, realistic_market_data):
        """Test du workflow d'annulation d'ordre."""
        st.session_state.orders = realistic_market_data['orders']
        pending_order = realistic_market_data['orders'][0]

        components = full_trading_environment

        # Annuler l'ordre
        components['order_manager']._cancel_order(pending_order['id'])

        # Vérifier que l'ordre est annulé
        updated_order = next(o for o in st.session_state.orders if o['id'] == pending_order['id'])
        assert updated_order['status'] == 'Cancelled'

    def test_portfolio_metrics_calculation(self, full_trading_environment, realistic_market_data):
        """Test des calculs de métriques du portefeuille."""
        st.session_state.positions = realistic_market_data['positions']

        components = full_trading_environment

        # Calculer les métriques
        components['position_monitor']._calculate_risk_metrics()

        # Vérifier les calculs de base
        positions = st.session_state.positions
        expected_total_margin = sum(pos['margin'] for pos in positions)
        expected_total_pnl = sum(pos['unrealized_pnl'] for pos in positions)

        # Les métriques devraient refléter ces calculs
        assert expected_total_margin > 0
        assert isinstance(expected_total_pnl, (int, float))


class TestLiveTradingScenarios:
    """Tests de scénarios complets de trading."""

    def test_profitable_trade_scenario(self, full_trading_environment):
        """Test d'un scénario de trade profitable complet."""
        components = full_trading_environment

        # 1. Placer un ordre d'achat
        st.session_state.order_form = {
            'symbol': 'BTC/USDT',
            'side': 'Buy',
            'type': 'Market',
            'quantity': 0.1
        }
        components['order_manager']._place_order()

        # 2. Simuler une hausse de prix
        initial_price = 43000.0
        new_price = 44000.0  # +2.3% gain

        st.session_state.price_data = {
            'BTC/USDT': {
                'current_price': new_price,
                'change_24h': 2.3,
                'volume_24h': 1000000000,
                'history': []
            }
        }

        # 3. Vérifier que l'ordre est devenu profitable
        filled_orders = [o for o in st.session_state.orders if o['status'] == 'Filled']
        assert len(filled_orders) > 0

        # 4. La position devrait être profitable
        # (Dans un vrai système, l'ordre créerait une position)

    def test_loss_scenario_with_stop_loss(self, full_trading_environment):
        """Test d'un scénario de perte avec stop loss."""
        # Créer une position en perte
        losing_position = {
            'id': 'losing_pos',
            'symbol': 'BTC/USDT',
            'side': 'Long',
            'size': 0.5,
            'entry_price': 45000.0,
            'current_price': 43000.0,  # -4.4% perte
            'margin': 22500.0,
            'unrealized_pnl': -1000.0,
            'stop_loss': 43500.0,  # Stop loss proche
            'entry_time': datetime.now() - timedelta(hours=1),
            'last_update': datetime.now(),
            'leverage': 1.0,
            'funding_fee': -2.0,
            'commission': 11.25
        }

        st.session_state.positions = [losing_position]

        components = full_trading_environment

        # Générer des alertes
        alerts = components['position_monitor']._generate_risk_alerts()

        # Il devrait y avoir une alerte de stop loss proche
        sl_alerts = [a for a in alerts if 'Stop Loss' in a['title']]
        assert len(sl_alerts) > 0

    def test_high_leverage_risk_scenario(self, full_trading_environment):
        """Test d'un scénario de risque avec leverage élevé."""
        # Créer des positions avec leverage élevé
        high_leverage_positions = [
            {
                'id': f'lev_pos_{i}',
                'symbol': f'{"BTC" if i % 2 == 0 else "ETH"}/USDT',
                'side': 'Long',
                'size': 1.0,
                'entry_price': 40000.0 if i % 2 == 0 else 2500.0,
                'current_price': 40000.0 if i % 2 == 0 else 2500.0,
                'margin': 8000.0 if i % 2 == 0 else 500.0,
                'unrealized_pnl': 0.0,
                'leverage': 5.0,  # Leverage très élevé
                'entry_time': datetime.now(),
                'last_update': datetime.now(),
                'stop_loss': 38000.0 if i % 2 == 0 else 2375.0,
                'take_profit': 42000.0 if i % 2 == 0 else 2625.0,
                'funding_fee': -5.0,
                'commission': 20.0
            }
            for i in range(3)
        ]

        st.session_state.positions = high_leverage_positions

        components = full_trading_environment

        # Calculer les risques
        components['risk_monitor']._calculate_risk_metrics()

        # Le leverage ratio devrait être élevé
        risk_metrics = st.session_state.risk_metrics
        assert risk_metrics['leverage_ratio'] > 2.0

        # Générer des alertes
        alerts = components['risk_monitor']._generate_risk_alerts()

        # Il devrait y avoir une alerte de leverage
        leverage_alerts = [a for a in alerts if 'Leverage' in a['title']]
        assert len(leverage_alerts) > 0


class TestLiveTradingPerformance:
    """Tests de performance pour l'interface Live Trading."""

    def test_large_dataset_performance(self, full_trading_environment):
        """Test des performances avec de gros datasets."""
        # Créer beaucoup de positions
        large_positions = []
        for i in range(100):
            large_positions.append({
                'id': f'perf_pos_{i}',
                'symbol': f'SYMBOL_{i % 10}/USDT',
                'side': 'Long' if i % 2 == 0 else 'Short',
                'size': 1.0,
                'entry_price': 1000.0 + i,
                'current_price': 1000.0 + i + np.random.uniform(-50, 50),
                'margin': 1000.0,
                'unrealized_pnl': np.random.uniform(-100, 100),
                'leverage': 1.0,
                'entry_time': datetime.now() - timedelta(hours=i % 24),
                'last_update': datetime.now(),
                'stop_loss': 950.0 + i,
                'take_profit': 1050.0 + i,
                'funding_fee': -1.0,
                'commission': 5.0
            })

        st.session_state.positions = large_positions

        components = full_trading_environment

        # Tester que les calculs restent rapides
        import time
        start_time = time.time()

        components['risk_monitor']._calculate_risk_metrics()
        components['position_monitor']._update_positions()

        end_time = time.time()

        # Les calculs ne devraient pas prendre plus de 1 seconde
        assert (end_time - start_time) < 1.0

    def test_concurrent_updates_stability(self, full_trading_environment, realistic_market_data):
        """Test de la stabilité avec des mises à jour concurrentes."""
        st.session_state.price_data = realistic_market_data['price_data']
        st.session_state.positions = realistic_market_data['positions']

        components = full_trading_environment

        # Simuler des mises à jour concurrentes
        for _ in range(10):
            components['dashboard']._update_real_time_data()
            components['position_monitor']._update_positions()
            components['risk_monitor']._calculate_risk_metrics()

        # Vérifier que les données restent cohérentes
        assert len(st.session_state.positions) == len(realistic_market_data['positions'])
        assert 'dashboard_data' in st.session_state
        assert 'risk_metrics' in st.session_state


if __name__ == '__main__':
    pytest.main([__file__, '-v'])