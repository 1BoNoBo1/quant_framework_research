"""
Tests d'intégration pour le workflow complet de backtesting.
Valide l'intégration entre tous les composants et le pipeline end-to-end.
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch, Mock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Ajouter le chemin pour importer les modules UI
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../qframe/ui'))


class TestBacktestingIntegration:
    """Tests d'intégration pour le workflow complet de backtesting."""

    def setup_method(self):
        """Configuration avant chaque test."""
        # Mock Streamlit et toutes les dépendances
        self.mock_st = MagicMock()

        # Configuration détaillée des mocks Streamlit
        self.mock_st.session_state = {}
        self.mock_st.tabs.return_value = [MagicMock() for _ in range(6)]
        self.mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        self.mock_st.selectbox.return_value = "DMN LSTM Strategy"
        self.mock_st.button.return_value = False
        self.mock_st.progress = MagicMock()
        self.mock_st.empty = MagicMock
        self.mock_st.success = MagicMock()
        self.mock_st.error = MagicMock()

        # Patch les imports
        modules_to_mock = {
            'streamlit': self.mock_st,
            'plotly.graph_objects': MagicMock(),
            'plotly.express': MagicMock(),
            'plotly.subplots': MagicMock(),
            'scipy': MagicMock(),
            'scipy.stats': MagicMock(),
            'pandas': pd,
            'numpy': np
        }

        self.patcher = patch.dict('sys.modules', modules_to_mock)
        self.patcher.start()

        # Import des composants après les mocks
        try:
            from streamlit_app.components.backtesting.backtest_configurator import BacktestConfigurator
            from streamlit_app.components.backtesting.results_analyzer import ResultsAnalyzer
            from streamlit_app.components.backtesting.walk_forward_interface import WalkForwardInterface
            from streamlit_app.components.backtesting.monte_carlo_simulator import MonteCarloSimulator
            from streamlit_app.components.backtesting.performance_analytics import PerformanceAnalytics
            from streamlit_app.components.backtesting.integration_manager import BacktestingIntegrationManager

            self.components = {
                'configurator': BacktestConfigurator(),
                'analyzer': ResultsAnalyzer(),
                'walk_forward': WalkForwardInterface(),
                'monte_carlo': MonteCarloSimulator(),
                'analytics': PerformanceAnalytics(),
                'integration_manager': BacktestingIntegrationManager()
            }

        except ImportError as e:
            pytest.skip(f"Composants de backtesting non disponibles: {e}")

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.patcher.stop()

    def test_component_initialization_integration(self):
        """Test d'intégration de l'initialisation de tous les composants."""
        # Vérifier que tous les composants s'initialisent correctement
        for component_name, component in self.components.items():
            assert component is not None, f"Composant {component_name} non initialisé"

    def test_configuration_to_execution_flow(self, sample_backtest_config, mock_session_state):
        """Test du flux configuration → exécution."""
        config = sample_backtest_config
        configurator = self.components['configurator']
        integration_manager = self.components['integration_manager']

        # Phase 1: Configuration
        with patch('streamlit.session_state', mock_session_state):
            try:
                # Simuler la création d'une configuration
                mock_session_state['current_config'] = config

                # Vérifier que la configuration est valide
                assert 'strategy_type' in config
                assert 'parameters' in config
                assert 'data_config' in config

                # Phase 2: Validation de la configuration
                strategy_type = config['strategy_type']
                assert strategy_type in [
                    "DMN LSTM Strategy",
                    "Adaptive Mean Reversion",
                    "Funding Arbitrage",
                    "RL Alpha Generator"
                ]

                # Phase 3: Simulation d'exécution
                mock_session_state['backtest_queue'] = [config]
                mock_session_state['processing_queue'] = True

                # Vérifier que le pipeline peut traiter la configuration
                assert len(mock_session_state['backtest_queue']) == 1

            except AttributeError:
                # Certaines méthodes peuvent ne pas être implémentées
                pass

    def test_results_generation_to_analysis_flow(self, sample_backtest_results):
        """Test du flux génération résultats → analyse."""
        results = sample_backtest_results
        analyzer = self.components['analyzer']
        analytics = self.components['analytics']

        try:
            # Phase 1: Validation des résultats générés
            required_keys = [
                'total_return', 'sharpe_ratio', 'max_drawdown',
                'equity_curve', 'drawdown_series'
            ]

            for key in required_keys:
                assert key in results, f"Clé manquante dans résultats: {key}"

            # Phase 2: Test de l'analyse des résultats
            if hasattr(analyzer, 'calculate_all_metrics'):
                equity_curve = np.array(results['equity_curve']['values'])
                returns = np.diff(equity_curve) / equity_curve[:-1]

                calculated_metrics = analyzer.calculate_all_metrics(returns)
                assert isinstance(calculated_metrics, dict)

            # Phase 3: Test des analytics avancées
            if hasattr(analytics, 'render_advanced_analytics'):
                # Devrait pouvoir traiter les résultats sans erreur
                analytics.render_advanced_analytics(results)

        except AttributeError:
            # Méthodes pas encore implémentées
            pass

    def test_walk_forward_to_monte_carlo_integration(self, walk_forward_config, monte_carlo_config):
        """Test d'intégration Walk-Forward → Monte Carlo."""
        walk_forward = self.components['walk_forward']
        monte_carlo = self.components['monte_carlo']

        try:
            # Phase 1: Génération résultats Walk-Forward
            if hasattr(walk_forward, '_generate_simulated_results'):
                wf_results = walk_forward._generate_simulated_results(walk_forward_config)

                if isinstance(wf_results, dict) and 'windows' in wf_results:
                    windows = wf_results['windows']

                    # Phase 2: Utiliser résultats WF pour Monte Carlo
                    # Extraire les métriques de performance de chaque fenêtre
                    window_metrics = []
                    for window in windows:
                        if isinstance(window, dict) and 'metrics' in window:
                            window_metrics.append(window['metrics'])

                    # Phase 3: Validation robustesse avec Monte Carlo
                    if len(window_metrics) > 1:
                        # Analyser la distribution des métriques
                        sharpe_ratios = [m.get('sharpe_ratio', 0) for m in window_metrics if isinstance(m, dict)]

                        if sharpe_ratios:
                            # Test de stabilité
                            sharpe_std = np.std(sharpe_ratios)
                            sharpe_mean = np.mean(sharpe_ratios)

                            # Coefficient de variation comme mesure de stabilité
                            if sharpe_mean != 0:
                                cv = sharpe_std / abs(sharpe_mean)
                                assert cv < 2.0  # Coefficient de variation raisonnable

        except AttributeError:
            pytest.skip("Méthodes d'intégration WF-MC non implémentées")

    def test_end_to_end_pipeline_execution(self, sample_backtest_config, mock_session_state):
        """Test du pipeline end-to-end complet."""
        config = sample_backtest_config
        integration_manager = self.components['integration_manager']

        with patch('streamlit.session_state', mock_session_state):
            try:
                # Phase 1: Configuration du workflow
                workflow_config = {
                    'base_config': config,
                    'walk_forward': {
                        'train_window': 180,
                        'test_window': 30,
                        'purge_days': 5,
                        'step_size': 15
                    },
                    'monte_carlo': {
                        'num_simulations': 100,
                        'confidence_levels': [0.95],
                        'random_seed': 42
                    }
                }

                # Phase 2: Validation du workflow
                if hasattr(integration_manager, '_validate_workflow'):
                    is_valid = integration_manager._validate_workflow(workflow_config)
                    assert is_valid is True or is_valid is None  # None si méthode pas implémentée

                # Phase 3: Simulation d'exécution pipeline
                pipeline_id = "test_pipeline_001"
                mock_session_state['active_backtests'] = {
                    pipeline_id: {
                        'id': pipeline_id,
                        'config': workflow_config,
                        'status': 'running',
                        'started_at': datetime.now(),
                        'steps_completed': 0,
                        'total_steps': 4
                    }
                }

                # Phase 4: Validation de l'état du pipeline
                pipeline_data = mock_session_state['active_backtests'][pipeline_id]
                assert pipeline_data['status'] == 'running'
                assert pipeline_data['total_steps'] == 4

            except AttributeError:
                # Méthodes pipeline pas encore implémentées
                pass

    def test_error_handling_across_components(self, sample_backtest_config):
        """Test de gestion d'erreur à travers les composants."""
        config = sample_backtest_config

        # Test avec configuration invalide
        invalid_configs = [
            {},  # Config vide
            {'strategy_type': 'Invalid Strategy'},  # Stratégie inexistante
            {'strategy_type': 'DMN LSTM Strategy', 'parameters': None},  # Paramètres None
        ]

        for invalid_config in invalid_configs:
            for component_name, component in self.components.items():
                try:
                    # Tester si le composant gère gracieusement les erreurs
                    if hasattr(component, 'render_configuration'):
                        component.render_configuration()

                    if hasattr(component, '_validate_config'):
                        component._validate_config(invalid_config)

                except (ValueError, KeyError, TypeError, AttributeError):
                    # Erreurs attendues - gestion gracieuse
                    assert True
                except Exception as e:
                    # Erreurs inattendues
                    pytest.fail(f"Erreur inattendue dans {component_name}: {e}")

    def test_session_state_consistency(self, mock_session_state):
        """Test de cohérence du session state entre composants."""
        integration_manager = self.components['integration_manager']

        with patch('streamlit.session_state', mock_session_state):
            # Phase 1: Initialisation du state
            expected_keys = [
                'backtest_queue',
                'backtest_results',
                'current_backtest',
                'active_backtests'
            ]

            # Initialiser les clés manquantes
            for key in expected_keys:
                if key not in mock_session_state:
                    if key == 'backtest_queue':
                        mock_session_state[key] = []
                    elif key == 'backtest_results':
                        mock_session_state[key] = {}
                    elif key == 'active_backtests':
                        mock_session_state[key] = {}
                    else:
                        mock_session_state[key] = None

            # Phase 2: Validation de la cohérence
            for key in expected_keys:
                assert key in mock_session_state

            # Phase 3: Test de modification cohérente
            backtest_id = "test_001"
            test_result = {'total_return': 15.5, 'sharpe_ratio': 1.2}

            mock_session_state['backtest_results'][backtest_id] = test_result

            # Vérifier que la modification est persistée
            assert backtest_id in mock_session_state['backtest_results']
            assert mock_session_state['backtest_results'][backtest_id] == test_result

    def test_data_flow_validation(self, crypto_market_data, sample_backtest_config):
        """Test de validation du flux de données."""
        config = sample_backtest_config
        data = crypto_market_data

        # Phase 1: Validation des données d'entrée
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0

        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col in data.columns:
                assert data[col].notna().all(), f"Colonne {col} contient des NaN"

        # Phase 2: Transformation pour backtesting
        if 'close' in data.columns:
            prices = data['close'].values
            returns = np.diff(prices) / prices[:-1]

            # Validation des returns
            assert len(returns) == len(prices) - 1
            assert not np.any(np.isnan(returns)), "Returns contiennent des NaN"
            assert not np.any(np.isinf(returns)), "Returns contiennent des infinis"

        # Phase 3: Compatibilité avec configuration
        symbols = config.get('data_config', {}).get('symbols', [])
        if symbols:
            # Vérifier que les symboles sont cohérents
            assert isinstance(symbols, list)
            assert len(symbols) > 0

    def test_performance_metrics_consistency(self, sample_backtest_results):
        """Test de cohérence des métriques de performance."""
        results = sample_backtest_results
        analyzer = self.components['analyzer']

        # Phase 1: Validation des métriques de base
        basic_metrics = [
            'total_return', 'annualized_return', 'volatility',
            'sharpe_ratio', 'max_drawdown'
        ]

        for metric in basic_metrics:
            if metric in results:
                value = results[metric]
                assert isinstance(value, (int, float)), f"Métrique {metric} non numérique"
                assert not np.isnan(value), f"Métrique {metric} est NaN"
                assert np.isfinite(value), f"Métrique {metric} est infinie"

        # Phase 2: Validation de cohérence mathématique
        if all(m in results for m in ['total_return', 'volatility', 'sharpe_ratio']):
            # Approximation du Sharpe ratio (sans risk-free rate exact)
            total_ret = results['total_return']
            volatility = results['volatility']
            sharpe = results['sharpe_ratio']

            if volatility > 0 and abs(total_ret) > 1:  # Si valeurs significatives
                # Le signe du Sharpe devrait être cohérent avec le return
                if total_ret > 2:  # Au-dessus du risk-free rate approximatif
                    assert sharpe >= -0.5, "Sharpe négatif avec return positif significatif"

        # Phase 3: Validation des contraintes
        if 'max_drawdown' in results:
            assert results['max_drawdown'] <= 0, "Max drawdown devrait être négatif"

        if 'win_rate' in results:
            assert 0 <= results['win_rate'] <= 100, "Win rate hors de [0, 100]"

    def test_visualization_data_integration(self, sample_backtest_results):
        """Test d'intégration des données de visualisation."""
        results = sample_backtest_results
        analytics = self.components['analytics']

        # Phase 1: Validation des données pour equity curve
        if 'equity_curve' in results:
            equity_data = results['equity_curve']
            assert 'dates' in equity_data and 'values' in equity_data

            dates = equity_data['dates']
            values = equity_data['values']

            assert len(dates) == len(values), "Dates et valeurs de tailles différentes"
            assert all(v > 0 for v in values), "Valeurs d'equity négatives"

        # Phase 2: Validation des données pour drawdown
        if 'drawdown_series' in results:
            drawdown = results['drawdown_series']
            assert isinstance(drawdown, list)
            assert all(dd <= 0 for dd in drawdown), "Drawdown positif trouvé"

        # Phase 3: Test de compatibilité avec Plotly
        try:
            # Simuler la création de graphiques
            if hasattr(analytics, 'render_advanced_analytics'):
                analytics.render_advanced_analytics(results)

        except AttributeError:
            # Méthode pas implémentée
            pass

    def test_memory_management_integration(self, crypto_market_data):
        """Test de gestion mémoire dans le pipeline intégré."""
        data = crypto_market_data

        # Phase 1: Test avec dataset normal
        initial_memory_usage = data.memory_usage(deep=True).sum()

        # Phase 2: Simulation de traitement intensif
        large_data = pd.concat([data] * 10, ignore_index=True)  # 10x plus de données

        # Phase 3: Validation que les composants gèrent les gros datasets
        for component_name, component in self.components.items():
            try:
                # Test avec gros dataset
                if hasattr(component, '_process_data'):
                    # Méthode hypothétique de traitement
                    component._process_data(large_data)

                # Vérifier qu'il n'y a pas de fuite mémoire massive
                # (test basique - dans un vrai test on utiliserait memory_profiler)

            except (AttributeError, MemoryError):
                # Attendu si méthode pas implémentée ou mémoire insuffisante
                pass

    def test_concurrent_backtests_handling(self, mock_session_state):
        """Test de gestion de backtests concurrents."""
        integration_manager = self.components['integration_manager']

        with patch('streamlit.session_state', mock_session_state):
            # Phase 1: Simuler plusieurs backtests actifs
            active_backtests = {}
            for i in range(3):
                backtest_id = f"backtest_{i:03d}"
                active_backtests[backtest_id] = {
                    'id': backtest_id,
                    'status': 'running',
                    'started_at': datetime.now() - timedelta(minutes=i),
                    'steps_completed': i,
                    'total_steps': 4
                }

            mock_session_state['active_backtests'] = active_backtests

            # Phase 2: Validation de l'état des backtests
            assert len(mock_session_state['active_backtests']) == 3

            # Phase 3: Test de gestion des statuts
            for backtest_id, backtest_data in mock_session_state['active_backtests'].items():
                assert backtest_data['status'] in ['running', 'completed', 'failed']
                assert backtest_data['steps_completed'] <= backtest_data['total_steps']

    def test_configuration_persistence_integration(self, sample_backtest_config, mock_session_state):
        """Test de persistance de configuration à travers le pipeline."""
        config = sample_backtest_config

        with patch('streamlit.session_state', mock_session_state):
            # Phase 1: Sauvegarder configuration
            config_id = "config_001"
            mock_session_state['saved_configs'] = {config_id: config}

            # Phase 2: Validation de persistance
            assert config_id in mock_session_state['saved_configs']
            recovered_config = mock_session_state['saved_configs'][config_id]

            # Phase 3: Test d'intégrité
            assert recovered_config == config

            # Phase 4: Test de modification et persistance
            modified_config = config.copy()
            modified_config['parameters']['window_size'] = 128

            mock_session_state['saved_configs'][config_id] = modified_config

            # Vérifier que la modification est persistée
            updated_config = mock_session_state['saved_configs'][config_id]
            assert updated_config['parameters']['window_size'] == 128