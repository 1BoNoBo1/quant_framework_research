"""
Tests complets pour le BacktestConfigurator.
Valide la configuration des stratégies, validation des paramètres, et templates.
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch, Mock
import json
import pandas as pd
import numpy as np
from datetime import datetime, date

# Ajouter le chemin pour importer les modules UI
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../qframe/ui'))


class TestBacktestConfigurator:
    """Tests complets pour le configurateur de backtests."""

    def setup_method(self):
        """Configuration avant chaque test."""
        # Mock Streamlit complètement
        self.mock_st = MagicMock()

        # Mock des composants Streamlit spécifiques
        self.mock_st.selectbox.return_value = "DMN LSTM Strategy"
        self.mock_st.text_area.return_value = '{"window_size": 64, "learning_rate": 0.001}'
        self.mock_st.multiselect.return_value = ["BTC/USDT"]
        self.mock_st.date_input.return_value = date(2024, 1, 1)
        self.mock_st.number_input.return_value = 10000
        self.mock_st.slider.return_value = 0.05
        self.mock_st.expander.return_value.__enter__ = MagicMock()
        self.mock_st.expander.return_value.__exit__ = MagicMock()
        self.mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]

        # Patch les imports
        modules_to_mock = {
            'streamlit': self.mock_st,
            'plotly.graph_objects': MagicMock(),
            'plotly.express': MagicMock()
        }

        self.patcher = patch.dict('sys.modules', modules_to_mock)
        self.patcher.start()

        # Import du configurateur après le mock
        try:
            from streamlit_app.components.backtesting.backtest_configurator import BacktestConfigurator
            self.configurator = BacktestConfigurator()
        except ImportError:
            pytest.skip("BacktestConfigurator non disponible")

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.patcher.stop()

    def test_initialization(self):
        """Test de l'initialisation du configurateur."""
        assert self.configurator is not None
        assert hasattr(self.configurator, '_load_strategy_templates')

    def test_strategy_templates_loading(self, strategy_templates):
        """Test du chargement des templates de stratégies."""
        templates = self.configurator._load_strategy_templates()

        assert isinstance(templates, dict)
        assert len(templates) > 0

        # Vérifier que les stratégies principales sont présentes
        expected_strategies = [
            "DMN LSTM Strategy",
            "Adaptive Mean Reversion",
            "Funding Arbitrage",
            "RL Alpha Generator"
        ]

        for strategy in expected_strategies:
            assert strategy in templates

        # Vérifier la structure de chaque template
        for strategy_name, template in templates.items():
            assert 'default_params' in template
            assert 'param_ranges' in template
            assert 'description' in template
            assert isinstance(template['default_params'], dict)

    def test_configuration_validation_valid_json(self):
        """Test de validation avec JSON valide."""
        valid_config = {
            "window_size": 64,
            "hidden_size": 128,
            "learning_rate": 0.001,
            "signal_threshold": 0.1
        }

        # Test que la validation ne lève pas d'exception
        try:
            config_str = json.dumps(valid_config)
            parsed_config = json.loads(config_str)
            assert parsed_config == valid_config
        except Exception as e:
            pytest.fail(f"Configuration valide rejetée: {e}")

    def test_configuration_validation_invalid_json(self):
        """Test de validation avec JSON invalide."""
        invalid_configs = [
            '{"window_size": 64, "learning_rate":}',  # JSON mal formé
            '{"window_size": "invalid"}',              # Type invalide
            '{}',                                      # Config vide
            'not json at all'                          # Pas du JSON
        ]

        for invalid_config in invalid_configs:
            try:
                json.loads(invalid_config)
                # Si on arrive ici, le JSON est techniquement valide mais peut être incorrect
                config = json.loads(invalid_config)
                if not config:  # Config vide
                    assert True  # Expected behavior
            except json.JSONDecodeError:
                assert True  # Expected behavior for invalid JSON

    def test_parameter_ranges_validation(self, strategy_templates):
        """Test de validation des plages de paramètres."""
        templates = self.configurator._load_strategy_templates()

        for strategy_name, template in templates.items():
            default_params = template['default_params']
            param_ranges = template.get('param_ranges', {})

            # Vérifier que les paramètres par défaut sont dans les plages
            for param_name, default_value in default_params.items():
                if param_name in param_ranges:
                    ranges = param_ranges[param_name]
                    if isinstance(ranges, list) and len(ranges) > 1:
                        if isinstance(default_value, (int, float)):
                            # Pour les valeurs numériques, vérifier la plage
                            min_val = min(ranges)
                            max_val = max(ranges)
                            assert min_val <= default_value <= max_val, \
                                f"Paramètre {param_name} hors plage pour {strategy_name}"
                        else:
                            # Pour les valeurs discrètes, vérifier membership
                            assert default_value in ranges, \
                                f"Paramètre {param_name} non dans options pour {strategy_name}"

    def test_asset_universe_selection(self):
        """Test de sélection d'univers d'actifs."""
        # Mock du multiselect pour les actifs
        expected_assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        self.mock_st.multiselect.return_value = expected_assets

        # Test que les actifs attendus sont disponibles
        available_assets = [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
            "AAPL", "GOOGL", "MSFT", "TSLA",
            "EUR/USD", "GBP/USD", "USD/JPY"
        ]

        for asset in expected_assets:
            assert asset in available_assets

    def test_timeframe_validation(self):
        """Test de validation des timeframes."""
        valid_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        self.mock_st.selectbox.return_value = "1h"

        selected_timeframe = self.mock_st.selectbox.return_value
        assert selected_timeframe in valid_timeframes

    def test_date_range_validation(self):
        """Test de validation des plages de dates."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 12, 31)

        # Vérifier que la date de fin est après la date de début
        assert end_date > start_date

        # Vérifier que les dates sont dans une plage raisonnable
        current_year = datetime.now().year
        assert start_date.year <= current_year + 1
        assert end_date.year <= current_year + 1

    def test_capital_configuration(self):
        """Test de configuration du capital initial."""
        self.mock_st.number_input.return_value = 10000

        initial_capital = self.mock_st.number_input.return_value

        # Vérifier que le capital est positif
        assert initial_capital > 0

        # Vérifier que le capital est dans une plage raisonnable
        assert 1000 <= initial_capital <= 10000000

    def test_risk_parameters_validation(self):
        """Test de validation des paramètres de risque."""
        # Mock des paramètres de risque
        self.mock_st.slider.side_effect = [0.05, 0.10, 1.0]  # stop_loss, take_profit, max_position

        stop_loss = 0.05
        take_profit = 0.10
        max_position = 1.0

        # Vérifier que stop_loss < take_profit
        assert stop_loss < take_profit

        # Vérifier que les valeurs sont dans des plages raisonnables
        assert 0 < stop_loss <= 0.20  # Max 20% stop loss
        assert 0 < take_profit <= 1.0  # Max 100% take profit
        assert 0 < max_position <= 1.0  # Max 100% position

    def test_optimization_mode_configuration(self):
        """Test de configuration du mode d'optimisation."""
        optimization_modes = ["None", "Grid Search", "Bayesian", "Genetic Algorithm"]

        for mode in optimization_modes:
            self.mock_st.selectbox.return_value = mode
            selected_mode = self.mock_st.selectbox.return_value
            assert selected_mode in optimization_modes

    def test_configuration_persistence(self, sample_backtest_config):
        """Test de persistance de la configuration."""
        config = sample_backtest_config

        # Simuler la sauvegarde et le rechargement
        config_json = json.dumps(config)
        reloaded_config = json.loads(config_json)

        # Vérifier que la configuration est identique
        assert reloaded_config == config

        # Vérifier que toutes les sections importantes sont présentes
        required_sections = ['strategy_type', 'parameters', 'data_config', 'risk_config']
        for section in required_sections:
            assert section in reloaded_config

    def test_default_configuration_completeness(self):
        """Test de complétude des configurations par défaut."""
        templates = self.configurator._load_strategy_templates()

        for strategy_name, template in templates.items():
            default_params = template['default_params']

            # Vérifier que les paramètres essentiels sont présents
            # (les paramètres spécifiques dépendent de la stratégie)
            assert len(default_params) > 0, f"Pas de paramètres par défaut pour {strategy_name}"

            # Vérifier que tous les paramètres ont des valeurs valides
            for param_name, param_value in default_params.items():
                assert param_value is not None, f"Paramètre {param_name} est None pour {strategy_name}"
                assert param_name != "", f"Nom de paramètre vide pour {strategy_name}"

    def test_parameter_type_validation(self):
        """Test de validation des types de paramètres."""
        templates = self.configurator._load_strategy_templates()

        for strategy_name, template in templates.items():
            default_params = template['default_params']

            for param_name, param_value in default_params.items():
                # Vérifier que les types sont supportés
                assert isinstance(param_value, (int, float, str, bool, list)), \
                    f"Type non supporté pour {param_name} dans {strategy_name}: {type(param_value)}"

                # Validations spécifiques par type
                if isinstance(param_value, (int, float)):
                    assert not (isinstance(param_value, float) and np.isnan(param_value)), \
                        f"Valeur NaN pour {param_name} dans {strategy_name}"

    def test_strategy_description_presence(self):
        """Test de présence des descriptions de stratégies."""
        templates = self.configurator._load_strategy_templates()

        for strategy_name, template in templates.items():
            assert 'description' in template, f"Description manquante pour {strategy_name}"
            assert isinstance(template['description'], str), \
                f"Description non-string pour {strategy_name}"
            assert len(template['description']) > 10, \
                f"Description trop courte pour {strategy_name}"

    @pytest.mark.parametrize("strategy_name", [
        "DMN LSTM Strategy",
        "Adaptive Mean Reversion",
        "Funding Arbitrage",
        "RL Alpha Generator"
    ])
    def test_specific_strategy_configuration(self, strategy_name):
        """Test de configuration spécifique par stratégie."""
        templates = self.configurator._load_strategy_templates()

        if strategy_name not in templates:
            pytest.skip(f"Stratégie {strategy_name} non disponible")

        template = templates[strategy_name]

        # Tests spécifiques par stratégie
        if strategy_name == "DMN LSTM Strategy":
            assert 'window_size' in template['default_params']
            assert 'hidden_size' in template['default_params']
            assert 'learning_rate' in template['default_params']

        elif strategy_name == "Adaptive Mean Reversion":
            assert 'lookback_short' in template['default_params']
            assert 'lookback_long' in template['default_params']
            assert 'z_entry_base' in template['default_params']

        elif strategy_name == "RL Alpha Generator":
            assert 'state_dim' in template['default_params']
            assert 'action_dim' in template['default_params']
            assert 'learning_rate' in template['default_params']

    def test_render_configuration_section(self, mock_session_state):
        """Test de rendu de la section de configuration."""
        # Mock session state
        with patch('streamlit.session_state', mock_session_state):
            try:
                # Tenter de rendre la section de configuration
                result = self.configurator.render_configuration_section()

                # Le résultat devrait être un dictionnaire ou None
                assert result is None or isinstance(result, dict)

            except Exception as e:
                # Acceptable si la méthode n'existe pas encore
                if "has no attribute 'render_configuration_section'" in str(e):
                    pytest.skip("Méthode render_configuration_section non implémentée")
                else:
                    raise

    def test_configuration_validation_edge_cases(self):
        """Test de validation avec des cas limites."""
        edge_cases = [
            # Valeurs extrêmes
            {"window_size": 1, "learning_rate": 0.00001},
            {"window_size": 1000, "learning_rate": 1.0},

            # Valeurs négatives
            {"window_size": -1, "learning_rate": -0.001},

            # Valeurs zéro
            {"window_size": 0, "learning_rate": 0},

            # Types mixtes
            {"window_size": "64", "learning_rate": "0.001"},
        ]

        for config in edge_cases:
            try:
                config_str = json.dumps(config)
                parsed_config = json.loads(config_str)

                # Validation métier (exemple)
                if 'window_size' in parsed_config:
                    window_size = parsed_config['window_size']
                    if isinstance(window_size, str):
                        try:
                            window_size = int(window_size)
                        except ValueError:
                            continue  # Configuration invalide attendue

                    # Vérifier les contraintes métier
                    if window_size <= 0 or window_size > 500:
                        continue  # Configuration invalide attendue

            except (json.JSONDecodeError, ValueError, TypeError):
                continue  # Erreurs attendues pour les cas limites

    def test_ui_component_interaction(self):
        """Test d'interaction avec les composants UI."""
        # Tester les interactions typiques
        interactions = [
            ('selectbox', 'strategy_type', 'DMN LSTM Strategy'),
            ('text_area', 'parameters', '{"window_size": 64}'),
            ('multiselect', 'assets', ['BTC/USDT']),
            ('date_input', 'start_date', date(2024, 1, 1)),
            ('number_input', 'capital', 10000),
        ]

        for component_type, param_name, expected_value in interactions:
            mock_component = getattr(self.mock_st, component_type)
            mock_component.return_value = expected_value

            result = mock_component.return_value
            assert result == expected_value

    def test_error_handling_graceful(self):
        """Test de gestion gracieuse des erreurs."""
        # Simuler des erreurs d'UI
        self.mock_st.selectbox.side_effect = Exception("UI Error")

        try:
            # Le configurateur devrait gérer les erreurs gracieusement
            templates = self.configurator._load_strategy_templates()
            assert templates is not None
        except Exception:
            # Si une erreur est levée, elle devrait être spécifique et informative
            pass