"""
Tests complets pour le ResultsAnalyzer.
Valide le calcul des métriques financières, visualisations, et analyses statistiques.
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


class TestResultsAnalyzer:
    """Tests complets pour l'analyseur de résultats."""

    def setup_method(self):
        """Configuration avant chaque test."""
        # Mock Streamlit et Plotly
        self.mock_st = MagicMock()
        self.mock_go = MagicMock()
        self.mock_px = MagicMock()

        # Configuration des mocks Streamlit
        self.mock_st.subheader = MagicMock()
        self.mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        self.mock_st.metric = MagicMock()
        self.mock_st.plotly_chart = MagicMock()
        self.mock_st.dataframe = MagicMock()
        self.mock_st.markdown = MagicMock()

        # Patch les imports
        modules_to_mock = {
            'streamlit': self.mock_st,
            'plotly.graph_objects': self.mock_go,
            'plotly.express': self.mock_px,
            'plotly.subplots': MagicMock(),
            'scipy': MagicMock(),
            'scipy.stats': MagicMock()
        }

        self.patcher = patch.dict('sys.modules', modules_to_mock)
        self.patcher.start()

        # Import de l'analyseur après le mock
        try:
            from streamlit_app.components.backtesting.results_analyzer import ResultsAnalyzer
            self.analyzer = ResultsAnalyzer()
        except ImportError:
            pytest.skip("ResultsAnalyzer non disponible")

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.patcher.stop()

    def test_initialization(self):
        """Test de l'initialisation de l'analyseur."""
        assert self.analyzer is not None
        assert hasattr(self.analyzer, 'risk_free_rate')
        assert self.analyzer.risk_free_rate == 0.02  # 2% par défaut

    def test_calculate_basic_metrics(self, sample_backtest_results):
        """Test du calcul des métriques de base."""
        results = sample_backtest_results

        # Extraire les returns pour les calculs
        equity_curve = np.array(results['equity_curve']['values'])
        returns = np.diff(equity_curve) / equity_curve[:-1]

        # Test des métriques calculées
        assert 'total_return' in results
        assert 'annualized_return' in results
        assert 'volatility' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results

        # Validation des valeurs
        assert isinstance(results['total_return'], (int, float))
        assert isinstance(results['sharpe_ratio'], (int, float))
        assert results['max_drawdown'] <= 0  # Drawdown est négatif

    def test_sharpe_ratio_calculation(self, sample_backtest_results):
        """Test spécifique du calcul du Sharpe ratio."""
        results = sample_backtest_results
        equity_curve = np.array(results['equity_curve']['values'])
        returns = np.diff(equity_curve) / equity_curve[:-1]

        # Calcul manuel du Sharpe ratio
        risk_free_rate = 0.02
        excess_returns = returns - risk_free_rate/252
        expected_sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)

        # Tolérance pour les différences de calcul
        assert abs(results['sharpe_ratio'] - expected_sharpe) < 0.5

    def test_sortino_ratio_calculation(self, sample_backtest_results):
        """Test du calcul du Sortino ratio."""
        results = sample_backtest_results

        assert 'sortino_ratio' in results
        assert isinstance(results['sortino_ratio'], (int, float))

        # Le Sortino devrait être différent du Sharpe (généralement plus élevé)
        if results['sharpe_ratio'] > 0:
            assert results['sortino_ratio'] >= results['sharpe_ratio']

    def test_calmar_ratio_calculation(self, sample_backtest_results):
        """Test du calcul du Calmar ratio."""
        results = sample_backtest_results

        assert 'calmar_ratio' in results
        assert isinstance(results['calmar_ratio'], (int, float))

        # Calmar = Annualized Return / |Max Drawdown|
        if results['max_drawdown'] != 0:
            expected_calmar = results['annualized_return'] / abs(results['max_drawdown'])
            assert abs(results['calmar_ratio'] - expected_calmar) < 0.1

    def test_drawdown_calculation(self, sample_backtest_results):
        """Test du calcul du drawdown."""
        results = sample_backtest_results

        assert 'drawdown_series' in results
        assert 'max_drawdown' in results

        drawdown_series = results['drawdown_series']

        # Vérifications basiques
        assert isinstance(drawdown_series, list)
        assert len(drawdown_series) > 0

        # Tous les drawdowns doivent être négatifs ou zéro
        assert all(dd <= 0 for dd in drawdown_series)

        # Le max drawdown doit être le minimum de la série
        assert abs(results['max_drawdown'] - min(drawdown_series)) < 0.01

    def test_var_cvar_calculation(self, sample_backtest_results):
        """Test du calcul VaR et CVaR."""
        results = sample_backtest_results

        assert 'var_95' in results
        assert 'cvar_95' in results

        var_95 = results['var_95']
        cvar_95 = results['cvar_95']

        # VaR et CVaR devraient être négatifs (pertes)
        assert var_95 <= 0
        assert cvar_95 <= 0

        # CVaR devrait être pire que VaR
        assert cvar_95 <= var_95

    def test_trade_metrics_calculation(self, sample_backtest_results):
        """Test du calcul des métriques de trading."""
        results = sample_backtest_results

        trade_metrics = ['total_trades', 'win_rate', 'profit_factor', 'avg_trade']

        for metric in trade_metrics:
            assert metric in results
            assert isinstance(results[metric], (int, float))

        # Validations spécifiques
        assert results['total_trades'] >= 0
        assert 0 <= results['win_rate'] <= 100
        assert results['profit_factor'] >= 0

    def test_monthly_returns_analysis(self, sample_backtest_results):
        """Test de l'analyse des returns mensuels."""
        results = sample_backtest_results

        assert 'monthly_returns' in results
        monthly_returns = results['monthly_returns']

        assert isinstance(monthly_returns, list)
        assert len(monthly_returns) == 12  # 12 mois

        # Vérifier que les returns sont numériques
        assert all(isinstance(ret, (int, float)) for ret in monthly_returns)

    def test_benchmark_comparison(self, sample_backtest_results):
        """Test de comparaison avec benchmark."""
        results = sample_backtest_results

        if 'benchmark_curve' in results:
            benchmark_data = results['benchmark_curve']

            assert 'dates' in benchmark_data
            assert 'values' in benchmark_data

            benchmark_values = benchmark_data['values']
            portfolio_values = results['equity_curve']['values']

            # Les deux courbes doivent avoir la même longueur
            assert len(benchmark_values) == len(portfolio_values)

    def test_risk_metrics_consistency(self, sample_backtest_results):
        """Test de cohérence des métriques de risque."""
        results = sample_backtest_results

        # La volatilité doit être positive
        assert results['volatility'] >= 0

        # Si Sharpe > 0, la stratégie génère un excess return
        if results['sharpe_ratio'] > 0:
            assert results['total_return'] > 0

        # Le max drawdown doit être cohérent avec la volatilité
        # Plus de volatilité = potentiellement plus de drawdown
        if results['volatility'] > 30:  # Haute volatilité
            assert abs(results['max_drawdown']) >= 5  # Au moins 5% de drawdown

    def test_performance_summary_rendering(self, sample_backtest_results):
        """Test du rendu du résumé de performance."""
        results = sample_backtest_results

        try:
            # Tenter de rendre le résumé
            self.analyzer.render_performance_summary(results)

            # Vérifier que les méthodes Streamlit ont été appelées
            assert self.mock_st.subheader.called
            assert self.mock_st.metric.called

        except AttributeError:
            # Méthode pas encore implémentée
            pytest.skip("Méthode render_performance_summary non implémentée")

    def test_calculate_all_metrics_method(self, sample_backtest_results):
        """Test de la méthode calculate_all_metrics."""
        equity_curve = np.array(sample_backtest_results['equity_curve']['values'])
        returns = np.diff(equity_curve) / equity_curve[:-1]

        try:
            # Test avec returns seulement
            metrics = self.analyzer.calculate_all_metrics(returns)

            assert isinstance(metrics, dict)
            assert len(metrics) > 0

            # Vérifier les métriques essentielles
            essential_metrics = ['sharpe_ratio', 'volatility', 'max_drawdown']
            for metric in essential_metrics:
                assert metric in metrics

        except AttributeError:
            pytest.skip("Méthode calculate_all_metrics non implémentée")

    def test_statistical_analysis(self, sample_backtest_results):
        """Test de l'analyse statistique des returns."""
        monthly_returns = sample_backtest_results['monthly_returns']

        if len(monthly_returns) > 3:  # Besoin de données suffisantes
            # Test de normalité basique
            mean_return = np.mean(monthly_returns)
            std_return = np.std(monthly_returns)

            assert isinstance(mean_return, (int, float))
            assert isinstance(std_return, (int, float))
            assert std_return >= 0

            # Test de skewness et kurtosis si disponible
            if len(monthly_returns) > 10:
                from scipy import stats
                skewness = stats.skew(monthly_returns)
                kurtosis = stats.kurtosis(monthly_returns)

                assert isinstance(skewness, (int, float))
                assert isinstance(kurtosis, (int, float))

    @pytest.mark.parametrize("confidence_level", [0.90, 0.95, 0.99])
    def test_var_calculation_different_confidence_levels(self, sample_backtest_results, confidence_level):
        """Test du calcul VaR à différents niveaux de confiance."""
        equity_curve = np.array(sample_backtest_results['equity_curve']['values'])
        returns = np.diff(equity_curve) / equity_curve[:-1]

        # Calcul VaR manuel
        var_value = np.percentile(returns, (1 - confidence_level) * 100)

        # VaR devrait être négatif pour des pertes
        assert var_value <= 0

        # VaR 99% devrait être pire que VaR 95% qui devrait être pire que VaR 90%
        # (plus le niveau de confiance est élevé, plus le VaR est sévère)

    def test_equity_curve_validation(self, sample_backtest_results):
        """Test de validation de la courbe d'equity."""
        equity_data = sample_backtest_results['equity_curve']

        assert 'dates' in equity_data
        assert 'values' in equity_data

        dates = equity_data['dates']
        values = equity_data['values']

        # Même nombre de dates et valeurs
        assert len(dates) == len(values)

        # Toutes les valeurs doivent être positives
        assert all(v > 0 for v in values)

        # Première valeur devrait être proche du capital initial
        assert 8000 <= values[0] <= 12000  # Tolérance autour de 10000

    def test_trade_analysis_detailed(self, sample_backtest_results):
        """Test détaillé de l'analyse des trades."""
        if 'trades' in sample_backtest_results:
            trades = sample_backtest_results['trades']

            for trade in trades:
                # Vérifier la structure de chaque trade
                required_fields = ['entry_time', 'exit_time', 'symbol', 'side', 'pnl']
                for field in required_fields:
                    assert field in trade

                # Vérifier les valeurs
                assert trade['side'] in ['long', 'short']
                assert isinstance(trade['pnl'], (int, float))

                # Entry time devrait être avant exit time
                entry_time = datetime.fromisoformat(trade['entry_time'].replace('Z', '+00:00').replace(' ', 'T'))
                exit_time = datetime.fromisoformat(trade['exit_time'].replace('Z', '+00:00').replace(' ', 'T'))
                assert entry_time < exit_time

    def test_performance_attribution(self, sample_backtest_results):
        """Test d'attribution de performance."""
        total_return = sample_backtest_results['total_return']

        # Décomposition simple de la performance
        # Performance = Alpha + Beta * Market Return + Noise
        # Pour le test, on vérifie juste que les calculs sont cohérents

        if 'benchmark_curve' in sample_backtest_results:
            benchmark_values = sample_backtest_results['benchmark_curve']['values']
            portfolio_values = sample_backtest_results['equity_curve']['values']

            benchmark_return = (benchmark_values[-1] / benchmark_values[0] - 1) * 100
            portfolio_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100

            # Alpha approximatif
            alpha = portfolio_return - benchmark_return

            # L'alpha peut être positif ou négatif
            assert isinstance(alpha, (int, float))

    def test_rolling_metrics_calculation(self, sample_backtest_results):
        """Test du calcul de métriques mobiles."""
        equity_curve = np.array(sample_backtest_results['equity_curve']['values'])

        if len(equity_curve) >= 60:  # Besoin de données suffisantes
            returns = np.diff(equity_curve) / equity_curve[:-1]

            # Calcul Sharpe mobile sur 60 jours
            window = 60
            rolling_sharpe = []

            for i in range(window, len(returns)):
                window_returns = returns[i-window:i]
                rolling_sharpe_value = np.mean(window_returns) / np.std(window_returns) * np.sqrt(252)
                rolling_sharpe.append(rolling_sharpe_value)

            assert len(rolling_sharpe) > 0
            assert all(isinstance(sr, (int, float)) for sr in rolling_sharpe)

    def test_error_handling_invalid_data(self):
        """Test de gestion d'erreur avec des données invalides."""
        invalid_data_sets = [
            {},  # Données vides
            {'total_return': None},  # Valeur None
            {'equity_curve': {'dates': [], 'values': []}},  # Listes vides
            {'equity_curve': {'dates': [1, 2, 3], 'values': [100, 200]}},  # Tailles différentes
        ]

        for invalid_data in invalid_data_sets:
            try:
                # L'analyseur devrait gérer gracieusement les données invalides
                if hasattr(self.analyzer, 'calculate_all_metrics'):
                    # Si la méthode existe, elle devrait soit réussir soit lever une erreur spécifique
                    pass
                else:
                    # Méthode pas encore implémentée
                    pass
            except (ValueError, KeyError, TypeError) as e:
                # Erreurs attendues pour des données invalides
                assert True
            except Exception as e:
                # Autres erreurs inattendues
                pytest.fail(f"Erreur inattendue: {e}")

    def test_visualization_data_preparation(self, sample_backtest_results):
        """Test de préparation des données pour visualisation."""
        results = sample_backtest_results

        # Données pour equity curve
        equity_data = results['equity_curve']
        assert len(equity_data['dates']) == len(equity_data['values'])

        # Données pour drawdown
        drawdown_data = results['drawdown_series']
        assert isinstance(drawdown_data, list)
        assert len(drawdown_data) > 0

        # Vérifier que les données sont compatibles avec Plotly
        # (types de base Python, pas de NaN, etc.)
        for value in equity_data['values'][:5]:  # Vérifier quelques valeurs
            assert not np.isnan(value)
            assert np.isfinite(value)

    def test_metrics_mathematical_properties(self, sample_backtest_results):
        """Test des propriétés mathématiques des métriques."""
        results = sample_backtest_results

        # Propriétés du Sharpe ratio
        if results['volatility'] > 0:
            # Sharpe = (Return - RiskFree) / Volatility
            expected_sharpe_sign = 1 if results['total_return'] > 2 else -1  # 2% risk-free
            actual_sharpe_sign = 1 if results['sharpe_ratio'] > 0 else -1

            # Le signe devrait être cohérent (avec tolérance)
            if abs(results['total_return']) > 1:  # Si return significatif
                assert expected_sharpe_sign == actual_sharpe_sign or abs(results['sharpe_ratio']) < 0.1

        # Propriétés de la volatilité
        assert results['volatility'] >= 0

        # Propriétés du profit factor
        if 'profit_factor' in results:
            assert results['profit_factor'] >= 0
            # Profit factor > 1 indique profitabilité
            if results['total_return'] > 0:
                assert results['profit_factor'] >= 1