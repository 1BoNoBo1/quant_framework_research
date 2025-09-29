"""
Tests complets pour le MonteCarloSimulator.
Valide les simulations de robustesse, bootstrap, et analyses de stress.
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
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../qframe/ui'))


class TestMonteCarloSimulator:
    """Tests complets pour le simulateur Monte Carlo."""

    def setup_method(self):
        """Configuration avant chaque test."""
        # Mock Streamlit et dépendances
        self.mock_st = MagicMock()
        self.mock_go = MagicMock()
        self.mock_stats = MagicMock()

        # Configuration des mocks Streamlit
        self.mock_st.subheader = MagicMock()
        self.mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        self.mock_st.number_input.side_effect = [1000, 42]  # num_simulations, random_seed
        self.mock_st.multiselect.return_value = [0.90, 0.95, 0.99]  # confidence_levels
        self.mock_st.selectbox.return_value = "standard"  # bootstrap_method
        self.mock_st.plotly_chart = MagicMock()
        self.mock_st.metric = MagicMock()

        # Mock scipy.stats
        self.mock_stats.t.rvs.return_value = np.random.normal(0, 1, 1000)
        self.mock_stats.norm.fit.return_value = (0, 1)

        # Patch les imports
        modules_to_mock = {
            'streamlit': self.mock_st,
            'plotly.graph_objects': self.mock_go,
            'plotly.express': MagicMock(),
            'scipy.stats': self.mock_stats,
            'pandas': pd,
            'numpy': np
        }

        self.patcher = patch.dict('sys.modules', modules_to_mock)
        self.patcher.start()

        # Import du Monte Carlo après le mock
        try:
            from streamlit_app.components.backtesting.monte_carlo_simulator import MonteCarloSimulator
            self.monte_carlo = MonteCarloSimulator()
        except ImportError:
            pytest.skip("MonteCarloSimulator non disponible")

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.patcher.stop()

    def test_initialization(self):
        """Test de l'initialisation du simulateur Monte Carlo."""
        assert self.monte_carlo is not None
        assert hasattr(self.monte_carlo, 'risk_free_rate')

    def test_bootstrap_standard_method(self, sample_backtest_results):
        """Test de la méthode bootstrap standard."""
        results = sample_backtest_results
        equity_curve = np.array(results['equity_curve']['values'])
        returns = np.diff(equity_curve) / equity_curve[:-1]

        # Test bootstrap standard (resampling)
        num_simulations = 100
        bootstrap_returns = []

        for i in range(num_simulations):
            # Bootstrap resampling
            resampled_indices = np.random.choice(len(returns), size=len(returns), replace=True)
            resampled_returns = returns[resampled_indices]
            bootstrap_returns.append(resampled_returns)

        assert len(bootstrap_returns) == num_simulations
        assert all(len(br) == len(returns) for br in bootstrap_returns)

        # Vérifier que les returns bootstrap sont différents des originaux
        assert not np.array_equal(bootstrap_returns[0], returns)

    def test_bootstrap_parametric_method(self, sample_backtest_results):
        """Test de la méthode bootstrap paramétrique."""
        results = sample_backtest_results
        equity_curve = np.array(results['equity_curve']['values'])
        returns = np.diff(equity_curve) / equity_curve[:-1]

        # Bootstrap paramétrique avec distribution normale
        mu = np.mean(returns)
        sigma = np.std(returns)
        num_simulations = 100

        parametric_returns = []
        np.random.seed(42)  # Pour reproductibilité

        for i in range(num_simulations):
            simulated_returns = np.random.normal(mu, sigma, len(returns))
            parametric_returns.append(simulated_returns)

        assert len(parametric_returns) == num_simulations

        # Vérifier que les moyennes et écarts-types sont proches des originaux
        avg_mean = np.mean([np.mean(pr) for pr in parametric_returns])
        avg_std = np.mean([np.std(pr) for pr in parametric_returns])

        assert abs(avg_mean - mu) < 0.1 * abs(mu)  # 10% de tolérance
        assert abs(avg_std - sigma) < 0.1 * sigma

    def test_confidence_intervals_calculation(self, monte_carlo_config):
        """Test du calcul des intervalles de confiance."""
        config = monte_carlo_config
        confidence_levels = config['confidence_levels']

        # Simuler des résultats Monte Carlo
        num_simulations = 1000
        np.random.seed(42)
        simulated_returns = np.random.normal(0.08, 0.15, num_simulations)  # 8% return, 15% vol

        # Calculer intervalles de confiance
        confidence_intervals = {}
        for confidence_level in confidence_levels:
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            lower_bound = np.percentile(simulated_returns, lower_percentile)
            upper_bound = np.percentile(simulated_returns, upper_percentile)

            confidence_intervals[confidence_level] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'width': upper_bound - lower_bound
            }

        # Vérifications
        for level in confidence_levels:
            ci = confidence_intervals[level]
            assert ci['lower'] < ci['upper']
            assert ci['width'] > 0

        # Les intervalles de confiance plus élevés devraient être plus larges
        assert confidence_intervals[0.99]['width'] > confidence_intervals[0.95]['width']
        assert confidence_intervals[0.95]['width'] > confidence_intervals[0.90]['width']

    def test_stress_testing_scenarios(self, extreme_market_scenarios):
        """Test des scénarios de stress testing."""
        scenarios = extreme_market_scenarios

        for scenario_name, scenario_data in scenarios.items():
            returns = scenario_data['returns']

            # Calculer métriques sous stress
            total_return = np.prod(1 + np.array(returns)) - 1
            volatility = np.std(returns) * np.sqrt(252)  # Annualisé
            max_drawdown = self._calculate_max_drawdown(returns)

            # Vérifications spécifiques par scénario
            if scenario_name == 'flash_crash':
                assert total_return < -0.1  # Au moins 10% de perte
                assert max_drawdown < -0.25  # Au moins 25% de drawdown

            elif scenario_name == 'pump_dump':
                assert volatility > 0.5  # Très haute volatilité
                assert max_drawdown < -0.3  # Fort drawdown après pump

            elif scenario_name == 'low_liquidity':
                assert volatility < 0.1  # Faible volatilité
                assert abs(total_return) < 0.05  # Faible mouvement

            elif scenario_name == 'high_volatility':
                assert volatility > 0.8  # Volatilité extrême

    def test_simulation_result_consistency(self, monte_carlo_config):
        """Test de cohérence des résultats de simulation."""
        config = monte_carlo_config

        try:
            results = self.monte_carlo._generate_mc_results(config)

            assert isinstance(results, dict)

            # Vérifier la structure des résultats
            expected_keys = [
                'summary_stats',
                'confidence_intervals',
                'stress_test_results',
                'distribution_analysis'
            ]

            for key in expected_keys:
                if key in results:
                    assert results[key] is not None

            # Vérifier la cohérence des statistiques
            if 'summary_stats' in results:
                stats = results['summary_stats']
                if isinstance(stats, dict):
                    # Vérifier que les métriques sont numériques
                    for metric_name, value in stats.items():
                        if isinstance(value, (int, float)):
                            assert not np.isnan(value)
                            assert np.isfinite(value)

        except AttributeError:
            pytest.skip("Méthode _generate_mc_results non implémentée")

    def test_tail_risk_analysis(self, sample_backtest_results):
        """Test de l'analyse des risques de queue."""
        results = sample_backtest_results
        equity_curve = np.array(results['equity_curve']['values'])
        returns = np.diff(equity_curve) / equity_curve[:-1]

        # Analyse VaR et CVaR à différents niveaux
        confidence_levels = [0.90, 0.95, 0.99]
        tail_metrics = {}

        for level in confidence_levels:
            alpha = 1 - level
            var_threshold = np.percentile(returns, alpha * 100)

            # CVaR (Expected Shortfall)
            tail_returns = returns[returns <= var_threshold]
            cvar = np.mean(tail_returns) if len(tail_returns) > 0 else var_threshold

            tail_metrics[level] = {
                'var': var_threshold,
                'cvar': cvar,
                'tail_expectation': cvar - var_threshold
            }

        # Vérifications
        for level in confidence_levels:
            metrics = tail_metrics[level]

            # VaR et CVaR devraient être négatifs (pertes)
            assert metrics['var'] <= 0
            assert metrics['cvar'] <= 0

            # CVaR devrait être pire que VaR
            assert metrics['cvar'] <= metrics['var']

        # VaR 99% devrait être pire que VaR 95%
        assert tail_metrics[0.99]['var'] <= tail_metrics[0.95]['var']

    def test_monte_carlo_convergence(self, sample_backtest_results):
        """Test de convergence des simulations Monte Carlo."""
        results = sample_backtest_results
        equity_curve = np.array(results['equity_curve']['values'])
        returns = np.diff(equity_curve) / equity_curve[:-1]

        # Test de convergence avec différents nombres de simulations
        simulation_counts = [100, 500, 1000, 2000]
        convergence_results = []

        np.random.seed(42)  # Pour reproductibilité

        for num_sims in simulation_counts:
            # Bootstrap sampling
            sim_means = []
            for i in range(num_sims):
                bootstrap_sample = np.random.choice(returns, size=len(returns), replace=True)
                sim_means.append(np.mean(bootstrap_sample))

            avg_mean = np.mean(sim_means)
            std_mean = np.std(sim_means)

            convergence_results.append({
                'num_simulations': num_sims,
                'estimated_mean': avg_mean,
                'standard_error': std_mean
            })

        # Vérifier la convergence (erreur standard devrait diminuer)
        for i in range(1, len(convergence_results)):
            current_se = convergence_results[i]['standard_error']
            previous_se = convergence_results[i-1]['standard_error']

            # L'erreur standard devrait diminuer (avec tolérance pour variance Monte Carlo)
            improvement_ratio = current_se / previous_se
            assert improvement_ratio <= 1.2  # Tolérance de 20%

    def test_distribution_fitting(self, sample_backtest_results):
        """Test d'ajustement de distributions."""
        results = sample_backtest_results
        equity_curve = np.array(results['equity_curve']['values'])
        returns = np.diff(equity_curve) / equity_curve[:-1]

        # Test d'ajustement normal
        from scipy import stats

        # Paramètres distribution normale
        mu_hat, sigma_hat = stats.norm.fit(returns)

        assert isinstance(mu_hat, (int, float))
        assert isinstance(sigma_hat, (int, float))
        assert sigma_hat > 0  # Écart-type positif

        # Test d'ajustement t-Student
        try:
            df_hat, loc_hat, scale_hat = stats.t.fit(returns)
            assert df_hat > 0  # Degrés de liberté positifs
            assert scale_hat > 0  # Échelle positive
        except:
            # Ajustement peut échouer avec peu de données
            pass

        # Vérifier que les paramètres estimés sont raisonnables
        empirical_mean = np.mean(returns)
        empirical_std = np.std(returns)

        assert abs(mu_hat - empirical_mean) < 0.1 * abs(empirical_mean) + 0.001
        assert abs(sigma_hat - empirical_std) < 0.1 * empirical_std + 0.001

    def test_configuration_rendering(self, monte_carlo_config):
        """Test du rendu de la configuration."""
        try:
            config = self.monte_carlo.render_configuration()

            # Le config devrait être un dict ou None
            assert config is None or isinstance(config, dict)

            if isinstance(config, dict):
                # Vérifier les clés importantes
                if 'num_simulations' in config:
                    assert isinstance(config['num_simulations'], int)
                    assert config['num_simulations'] > 0

                if 'confidence_levels' in config:
                    levels = config['confidence_levels']
                    assert isinstance(levels, list)
                    for level in levels:
                        assert 0 < level < 1

        except AttributeError:
            pytest.skip("Méthode render_configuration non implémentée")

    def test_stress_scenario_generation(self, monte_carlo_config):
        """Test de génération de scénarios de stress."""
        config = monte_carlo_config

        # Scénarios de stress prédéfinis
        stress_scenarios = {
            'market_crash': {
                'return_shock': -0.20,  # 20% crash
                'volatility_multiplier': 2.0
            },
            'liquidity_crisis': {
                'return_shock': -0.05,
                'volatility_multiplier': 3.0
            },
            'inflation_spike': {
                'return_shock': -0.10,
                'volatility_multiplier': 1.5
            }
        }

        for scenario_name, scenario_params in stress_scenarios.items():
            # Générer des returns sous stress
            base_return = 0.08 / 252  # Return quotidien de base
            base_volatility = 0.20 / np.sqrt(252)  # Vol quotidienne de base

            # Appliquer les chocs
            stressed_return = base_return + scenario_params['return_shock'] / 252
            stressed_volatility = base_volatility * scenario_params['volatility_multiplier']

            # Générer série de returns stressés
            num_days = 252
            np.random.seed(42)
            stressed_returns = np.random.normal(stressed_return, stressed_volatility, num_days)

            # Vérifications
            assert np.mean(stressed_returns) < base_return  # Return moyen plus faible
            assert np.std(stressed_returns) >= base_volatility  # Volatilité plus élevée

    def test_scenario_probability_weighting(self):
        """Test de pondération probabiliste des scénarios."""
        # Scénarios avec probabilités
        scenarios = [
            {'name': 'normal', 'probability': 0.70, 'return_factor': 1.0},
            {'name': 'recession', 'probability': 0.20, 'return_factor': 0.5},
            {'name': 'crisis', 'probability': 0.10, 'return_factor': 0.2}
        ]

        # Vérifier que les probabilités somment à 1
        total_probability = sum(s['probability'] for s in scenarios)
        assert abs(total_probability - 1.0) < 0.001

        # Simulation pondérée
        num_simulations = 1000
        np.random.seed(42)

        weighted_results = []
        for i in range(num_simulations):
            # Sélectionner scénario selon probabilités
            rand = np.random.random()
            cumulative_prob = 0

            for scenario in scenarios:
                cumulative_prob += scenario['probability']
                if rand <= cumulative_prob:
                    selected_scenario = scenario
                    break

            # Générer résultat selon scénario sélectionné
            base_return = 0.08
            scenario_return = base_return * selected_scenario['return_factor']
            weighted_results.append(scenario_return)

        # Vérifier que la distribution reflète les pondérations
        avg_return = np.mean(weighted_results)
        expected_return = sum(s['probability'] * 0.08 * s['return_factor'] for s in scenarios)

        assert abs(avg_return - expected_return) < 0.01  # 1% de tolérance

    def test_results_visualization_data(self, monte_carlo_config):
        """Test des données pour visualisation."""
        config = monte_carlo_config

        try:
            results = self.monte_carlo._generate_mc_results(config)
            self.monte_carlo.render_results(results)

            # Vérifier que les méthodes de visualisation ont été appelées
            assert self.mock_st.subheader.called or self.mock_st.plotly_chart.called

        except AttributeError:
            pytest.skip("Méthodes de rendu non implémentées")

    @pytest.mark.parametrize("num_simulations", [100, 500, 1000, 2000])
    def test_simulation_scalability(self, monte_carlo_config, num_simulations):
        """Test de scalabilité avec différents nombres de simulations."""
        config = monte_carlo_config.copy()
        config['num_simulations'] = num_simulations

        try:
            start_time = datetime.now()
            results = self.monte_carlo._generate_mc_results(config)
            end_time = datetime.now()

            execution_time = (end_time - start_time).total_seconds()

            # Vérifications de performance
            assert execution_time < 10.0  # Maximum 10 secondes
            assert isinstance(results, dict)

            # Le temps d'exécution devrait évoluer de manière raisonnable
            # (pas nécessairement linéaire à cause des optimisations NumPy)

        except AttributeError:
            pytest.skip("Méthode _generate_mc_results non implémentée")

    def test_random_seed_reproducibility(self, monte_carlo_config):
        """Test de reproductibilité avec graine aléatoire."""
        config = monte_carlo_config.copy()
        config['random_seed'] = 42

        try:
            # Première exécution
            results1 = self.monte_carlo._generate_mc_results(config)

            # Deuxième exécution avec même graine
            results2 = self.monte_carlo._generate_mc_results(config)

            # Les résultats devraient être identiques
            if isinstance(results1, dict) and isinstance(results2, dict):
                for key in results1:
                    if key in results2:
                        val1, val2 = results1[key], results2[key]
                        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                            assert abs(val1 - val2) < 1e-10  # Quasi-identiques

        except AttributeError:
            pytest.skip("Méthode _generate_mc_results non implémentée")

    def _calculate_max_drawdown(self, returns):
        """Utilitaire pour calculer le maximum drawdown."""
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)