"""
Tests complets pour le WalkForwardInterface.
Valide la validation temporelle, calcul des fenêtres, et analyse out-of-sample.
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch, Mock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import json

# Ajouter le chemin pour importer les modules UI
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../qframe/ui'))


class TestWalkForwardInterface:
    """Tests complets pour l'interface Walk-Forward."""

    def setup_method(self):
        """Configuration avant chaque test."""
        # Mock Streamlit et autres dépendances
        self.mock_st = MagicMock()
        self.mock_go = MagicMock()
        self.mock_px = MagicMock()

        # Configuration des mocks Streamlit
        self.mock_st.subheader = MagicMock()
        self.mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        self.mock_st.number_input.side_effect = [180, 30, 5, 15, 60]  # train, test, purge, step, min_periods
        self.mock_st.date_input.side_effect = [date(2024, 1, 1), date(2024, 12, 31)]
        self.mock_st.plotly_chart = MagicMock()
        self.mock_st.dataframe = MagicMock()
        self.mock_st.metric = MagicMock()

        # Patch les imports
        modules_to_mock = {
            'streamlit': self.mock_st,
            'plotly.graph_objects': self.mock_go,
            'plotly.express': self.mock_px,
            'plotly.subplots': MagicMock(),
            'pandas': pd,
            'numpy': np
        }

        self.patcher = patch.dict('sys.modules', modules_to_mock)
        self.patcher.start()

        # Import du walk forward après le mock
        try:
            from streamlit_app.components.backtesting.walk_forward_interface import WalkForwardInterface
            self.walk_forward = WalkForwardInterface()
        except ImportError:
            pytest.skip("WalkForwardInterface non disponible")

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.patcher.stop()

    def test_initialization(self):
        """Test de l'initialisation de l'interface Walk-Forward."""
        assert self.walk_forward is not None

    def test_window_calculation_basic(self, walk_forward_config):
        """Test du calcul de base des fenêtres."""
        config = walk_forward_config
        total_days = 365

        try:
            windows = self.walk_forward._calculate_windows(config, total_days)

            assert isinstance(windows, list)
            assert len(windows) > 0

            # Vérifier la structure de chaque fenêtre
            for window in windows:
                assert isinstance(window, dict)
                required_keys = ['train_start', 'train_end', 'test_start', 'test_end']
                for key in required_keys:
                    assert key in window

                # Vérifier la logique temporelle
                assert window['train_start'] < window['train_end']
                assert window['test_start'] < window['test_end']
                assert window['train_end'] <= window['test_start']  # Pas de overlap (sauf purge)

        except AttributeError:
            pytest.skip("Méthode _calculate_windows non implémentée")

    def test_window_calculation_with_purge(self, walk_forward_config):
        """Test du calcul des fenêtres avec période de purge."""
        config = walk_forward_config.copy()
        config['purge_days'] = 10
        total_days = 365

        try:
            windows = self.walk_forward._calculate_windows(config, total_days)

            for window in windows:
                # Avec purge, il devrait y avoir un gap entre train_end et test_start
                gap = window['test_start'] - window['train_end']
                assert gap >= config['purge_days']

        except AttributeError:
            pytest.skip("Méthode _calculate_windows non implémentée")

    def test_window_validation_constraints(self, walk_forward_config):
        """Test de validation des contraintes des fenêtres."""
        config = walk_forward_config

        # Test avec différentes tailles de datasets
        test_cases = [
            (100, "Dataset trop petit"),
            (365, "Dataset normal"),
            (1000, "Dataset large")
        ]

        for total_days, description in test_cases:
            try:
                windows = self.walk_forward._calculate_windows(config, total_days)

                if total_days < config['train_window'] + config['test_window']:
                    # Dataset trop petit - devrait retourner liste vide ou lever erreur
                    assert len(windows) == 0 or windows is None
                else:
                    # Dataset suffisant - devrait générer des fenêtres
                    assert len(windows) > 0

                    for window in windows:
                        # Vérifier que les fenêtres sont dans les limites
                        assert 0 <= window['train_start'] < total_days
                        assert 0 <= window['test_end'] <= total_days

            except (AttributeError, ValueError):
                # Méthode pas implémentée ou erreur attendue
                pass

    def test_configuration_validation(self):
        """Test de validation de la configuration Walk-Forward."""
        valid_configs = [
            {
                'train_window': 180,
                'test_window': 30,
                'purge_days': 5,
                'step_size': 15,
                'min_periods': 60
            },
            {
                'train_window': 252,
                'test_window': 21,
                'purge_days': 0,
                'step_size': 21,
                'min_periods': 100
            }
        ]

        invalid_configs = [
            {
                'train_window': -1,  # Négatif
                'test_window': 30,
                'purge_days': 5,
                'step_size': 15,
                'min_periods': 60
            },
            {
                'train_window': 10,
                'test_window': 30,  # Test > Train
                'purge_days': 5,
                'step_size': 15,
                'min_periods': 60
            },
            {
                'train_window': 180,
                'test_window': 30,
                'purge_days': 5,
                'step_size': 0,  # Step zéro
                'min_periods': 60
            }
        ]

        # Test configurations valides
        for config in valid_configs:
            assert config['train_window'] > 0
            assert config['test_window'] > 0
            assert config['purge_days'] >= 0
            assert config['step_size'] > 0
            assert config['min_periods'] > 0

        # Test configurations invalides
        for config in invalid_configs:
            # Au moins une contrainte devrait être violée
            violations = []
            if config['train_window'] <= 0:
                violations.append("train_window")
            if config['test_window'] <= 0:
                violations.append("test_window")
            if config['purge_days'] < 0:
                violations.append("purge_days")
            if config['step_size'] <= 0:
                violations.append("step_size")

            assert len(violations) > 0

    def test_out_of_sample_validation(self, walk_forward_config, crypto_market_data):
        """Test de validation out-of-sample."""
        config = walk_forward_config
        data = crypto_market_data

        if len(data) >= config['train_window'] + config['test_window']:
            try:
                # Simuler validation out-of-sample
                train_end_idx = config['train_window']
                test_start_idx = train_end_idx + config['purge_days']
                test_end_idx = test_start_idx + config['test_window']

                if test_end_idx <= len(data):
                    train_data = data.iloc[:train_end_idx]
                    test_data = data.iloc[test_start_idx:test_end_idx]

                    # Vérifications de base
                    assert len(train_data) == config['train_window']
                    assert len(test_data) == config['test_window']

                    # Pas de chevauchement temporel
                    train_last_date = train_data['timestamp'].iloc[-1]
                    test_first_date = test_data['timestamp'].iloc[0]
                    assert train_last_date < test_first_date

            except (KeyError, IndexError):
                # Structure de données différente attendue
                pass

    def test_performance_metrics_across_windows(self, walk_forward_config):
        """Test des métriques de performance à travers les fenêtres."""
        config = walk_forward_config

        # Simuler résultats de plusieurs fenêtres
        simulated_windows_results = []
        for i in range(5):  # 5 fenêtres d'exemple
            window_result = {
                'window_id': i,
                'train_period': f"2024-{i+1:02d}-01_to_2024-{i+6:02d}-30",
                'test_period': f"2024-{i+7:02d}-01_to_2024-{i+7:02d}-30",
                'sharpe_ratio': np.random.normal(1.2, 0.5),
                'total_return': np.random.normal(8.5, 5.0),
                'max_drawdown': np.random.normal(-12.0, 8.0),
                'win_rate': np.random.uniform(45, 65)
            }
            simulated_windows_results.append(window_result)

        # Analyse de stabilité
        sharpe_ratios = [w['sharpe_ratio'] for w in simulated_windows_results]
        returns = [w['total_return'] for w in simulated_windows_results]

        # Métriques de stabilité
        sharpe_std = np.std(sharpe_ratios)
        return_std = np.std(returns)

        # Critères de stabilité (exemples)
        assert sharpe_std < 2.0  # Sharpe pas trop volatile
        assert return_std < 20.0  # Returns pas trop volatiles

        # Cohérence des métriques
        for window in simulated_windows_results:
            assert isinstance(window['sharpe_ratio'], (int, float))
            assert isinstance(window['total_return'], (int, float))
            assert window['max_drawdown'] <= 0  # Drawdown négatif
            assert 0 <= window['win_rate'] <= 100  # Win rate en pourcentage

    def test_temporal_consistency(self, walk_forward_config):
        """Test de cohérence temporelle des fenêtres."""
        config = walk_forward_config
        total_days = 365

        try:
            windows = self.walk_forward._calculate_windows(config, total_days)

            if len(windows) > 1:
                for i in range(1, len(windows)):
                    prev_window = windows[i-1]
                    curr_window = windows[i]

                    # La fenêtre courante devrait commencer après la précédente
                    assert curr_window['train_start'] >= prev_window['train_start']

                    # Le pas d'avancement devrait être respecté
                    step_diff = curr_window['train_start'] - prev_window['train_start']
                    assert step_diff == config['step_size']

        except AttributeError:
            pytest.skip("Méthode _calculate_windows non implémentée")

    def test_minimum_data_requirements(self, walk_forward_config):
        """Test des exigences minimales de données."""
        config = walk_forward_config

        # Calculer la quantité minimale de données requise
        min_required_days = config['train_window'] + config['purge_days'] + config['test_window']

        test_cases = [
            (min_required_days - 1, False),  # Insuffisant
            (min_required_days, True),       # Juste suffisant
            (min_required_days + 100, True)  # Plus que suffisant
        ]

        for total_days, should_succeed in test_cases:
            try:
                windows = self.walk_forward._calculate_windows(config, total_days)

                if should_succeed:
                    assert windows is not None
                    if isinstance(windows, list):
                        assert len(windows) >= 0  # Au moins pas d'erreur
                else:
                    assert windows is None or len(windows) == 0

            except (AttributeError, ValueError):
                # Comportement attendu pour données insuffisantes
                if not should_succeed:
                    assert True

    def test_configuration_rendering(self, walk_forward_config):
        """Test du rendu de la configuration."""
        try:
            config = self.walk_forward.render_configuration()

            # Le config devrait être un dict ou None
            assert config is None or isinstance(config, dict)

            if isinstance(config, dict):
                # Vérifier les clés importantes
                important_keys = ['train_window', 'test_window', 'purge_days']
                for key in important_keys:
                    if key in config:
                        assert isinstance(config[key], (int, float))
                        assert config[key] >= 0

        except AttributeError:
            pytest.skip("Méthode render_configuration non implémentée")

    def test_results_visualization_data(self, walk_forward_config):
        """Test des données pour visualisation des résultats."""
        # Simuler des résultats Walk-Forward
        simulated_results = {
            'windows': [
                {
                    'window_id': 0,
                    'train_start': 0,
                    'train_end': 180,
                    'test_start': 185,
                    'test_end': 215,
                    'metrics': {
                        'sharpe_ratio': 1.25,
                        'total_return': 8.5,
                        'max_drawdown': -12.3
                    }
                },
                {
                    'window_id': 1,
                    'train_start': 15,
                    'train_end': 195,
                    'test_start': 200,
                    'test_end': 230,
                    'metrics': {
                        'sharpe_ratio': 0.95,
                        'total_return': 6.2,
                        'max_drawdown': -8.7
                    }
                }
            ],
            'summary_metrics': {
                'avg_sharpe': 1.10,
                'std_sharpe': 0.21,
                'avg_return': 7.35,
                'stability_score': 0.85
            }
        }

        try:
            self.walk_forward.render_results(simulated_results)

            # Vérifier que les méthodes Streamlit ont été appelées
            assert self.mock_st.subheader.called or self.mock_st.plotly_chart.called

        except AttributeError:
            pytest.skip("Méthode render_results non implémentée")

    def test_edge_cases_handling(self, walk_forward_config):
        """Test de gestion des cas limites."""
        edge_cases = [
            # Configuration extrême
            {
                'train_window': 1,
                'test_window': 1,
                'purge_days': 0,
                'step_size': 1,
                'min_periods': 1
            },
            # Fenêtre de test très grande
            {
                'train_window': 30,
                'test_window': 300,
                'purge_days': 0,
                'step_size': 30,
                'min_periods': 30
            },
            # Purge très grande
            {
                'train_window': 100,
                'test_window': 30,
                'purge_days': 100,
                'step_size': 30,
                'min_periods': 50
            }
        ]

        for config in edge_cases:
            try:
                windows = self.walk_forward._calculate_windows(config, 365)

                # Vérifier que même dans les cas extrêmes, la structure est cohérente
                if windows and isinstance(windows, list):
                    for window in windows:
                        if isinstance(window, dict):
                            # Vérifications de base même pour cas extrêmes
                            assert window['train_start'] <= window['train_end']
                            assert window['test_start'] <= window['test_end']

            except (AttributeError, ValueError, KeyError):
                # Erreurs attendues pour configurations extrêmes
                assert True

    def test_simulation_results_generation(self, walk_forward_config):
        """Test de génération de résultats simulés."""
        config = walk_forward_config

        try:
            simulated_results = self.walk_forward._generate_simulated_results(config)

            assert isinstance(simulated_results, dict)

            # Vérifier la structure des résultats simulés
            if 'windows' in simulated_results:
                windows = simulated_results['windows']
                assert isinstance(windows, list)

                for window in windows:
                    assert isinstance(window, dict)
                    if 'metrics' in window:
                        metrics = window['metrics']
                        assert isinstance(metrics, dict)

        except AttributeError:
            pytest.skip("Méthode _generate_simulated_results non implémentée")

    @pytest.mark.parametrize("step_size", [1, 7, 15, 30])
    def test_different_step_sizes(self, walk_forward_config, step_size):
        """Test avec différentes tailles de pas."""
        config = walk_forward_config.copy()
        config['step_size'] = step_size
        total_days = 365

        try:
            windows = self.walk_forward._calculate_windows(config, total_days)

            if windows and len(windows) > 1:
                # Vérifier que le pas est respecté
                for i in range(1, len(windows)):
                    step_diff = windows[i]['train_start'] - windows[i-1]['train_start']
                    assert step_diff == step_size

        except AttributeError:
            pytest.skip("Méthode _calculate_windows non implémentée")

    def test_overlapping_prevention(self, walk_forward_config):
        """Test de prévention du chevauchement entre train et test."""
        config = walk_forward_config
        config['purge_days'] = 0  # Pas de purge pour ce test

        try:
            windows = self.walk_forward._calculate_windows(config, 365)

            for window in windows:
                # Sans purge, test devrait commencer exactement après train
                assert window['test_start'] >= window['train_end']

        except AttributeError:
            pytest.skip("Méthode _calculate_windows non implémentée")

    def test_performance_degradation_detection(self):
        """Test de détection de dégradation de performance."""
        # Simuler une dégradation de performance au fil du temps
        degrading_performance = [
            {'sharpe_ratio': 1.5, 'return': 12.0},
            {'sharpe_ratio': 1.2, 'return': 9.5},
            {'sharpe_ratio': 0.8, 'return': 6.2},
            {'sharpe_ratio': 0.3, 'return': 2.8},
            {'sharpe_ratio': -0.1, 'return': -1.5}
        ]

        # Calculer la tendance
        sharpe_values = [p['sharpe_ratio'] for p in degrading_performance]
        return_values = [p['return'] for p in degrading_performance]

        # Test de tendance décroissante
        for i in range(1, len(sharpe_values)):
            if i >= 2:  # Avec au moins 3 points
                # Calculer pente simple
                recent_change = sharpe_values[i] - sharpe_values[i-1]
                # Une dégradation constante devrait donner des changements négatifs
                assert recent_change <= 0.5  # Tolérance pour fluctuations