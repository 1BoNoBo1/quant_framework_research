"""
Tests pour Performance Analyzer
===============================

Tests pour l'analyseur de performance avancée.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from qframe.research.reports.performance_analyzer import PerformanceAnalyzer


class TestPerformanceAnalyzer:
    """Tests pour PerformanceAnalyzer."""

    def setup_method(self):
        """Setup pour chaque test."""
        self.analyzer = PerformanceAnalyzer()

        # Génération de données de test
        np.random.seed(42)
        n_periods = 252  # 1 an de données journalières
        dates = pd.date_range('2024-01-01', periods=n_periods, freq='D')

        # Returns simulés avec drift positif
        returns = np.random.normal(0.0008, 0.02, n_periods)  # ~20% annuel, 20% vol
        self.sample_returns = pd.Series(returns, index=dates, name='returns')

    def test_analyzer_initialization(self):
        """Test initialisation de l'analyseur."""
        assert self.analyzer is not None
        assert hasattr(self.analyzer, 'calculate_basic_metrics')
        assert self.analyzer.risk_free_rate == 0.02

    def test_basic_metrics_calculation(self):
        """Test calcul des métriques de base."""
        metrics = self.analyzer.calculate_basic_metrics(self.sample_returns)

        assert 'total_return' in metrics
        assert 'annualized_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics

        # Vérifications de base
        assert isinstance(metrics['total_return'], (int, float))
        assert isinstance(metrics['sharpe_ratio'], (int, float))
        assert metrics['max_drawdown'] <= 0  # Drawdown devrait être négatif

    def test_risk_metrics_calculation(self):
        """Test calcul des métriques de risque."""
        risk_metrics = self.analyzer.calculate_risk_metrics(self.sample_returns)

        assert 'var_95' in risk_metrics
        assert 'var_99' in risk_metrics
        assert 'cvar_95' in risk_metrics
        assert 'cvar_99' in risk_metrics
        assert 'sortino_ratio' in risk_metrics
        assert 'downside_deviation' in risk_metrics

        # VaR 99% devrait être plus extrême que VaR 95%
        assert risk_metrics['var_99'] <= risk_metrics['var_95']

    def test_trading_metrics_with_trades(self):
        """Test métriques de trading avec données de trades."""
        # Simulation de données de trades
        trades_data = {
            'timestamp': pd.date_range('2024-01-01', periods=50, freq='D'),
            'pnl': np.random.normal(10, 50, 50),  # PnL variable
            'symbol': ['BTC/USD'] * 50
        }
        trades_df = pd.DataFrame(trades_data)

        trading_metrics = self.analyzer.calculate_trading_metrics(trades_df)

        assert 'total_trades' in trading_metrics
        assert 'win_rate' in trading_metrics
        assert 'profit_factor' in trading_metrics

        assert trading_metrics['total_trades'] == 50
        assert 0 <= trading_metrics['win_rate'] <= 1

    def test_trading_metrics_empty_trades(self):
        """Test métriques de trading avec données vides."""
        empty_trades = pd.DataFrame()
        trading_metrics = self.analyzer.calculate_trading_metrics(empty_trades)

        assert trading_metrics['total_trades'] == 0

    def test_rolling_metrics_calculation(self):
        """Test calcul des métriques roulantes."""
        rolling_metrics = self.analyzer.calculate_rolling_metrics(self.sample_returns)

        assert isinstance(rolling_metrics, pd.DataFrame)
        assert 'rolling_sharpe' in rolling_metrics.columns
        assert 'rolling_volatility' in rolling_metrics.columns
        assert 'rolling_drawdown' in rolling_metrics.columns

        # Vérifier que nous avons des données (au moins quelques points)
        assert len(rolling_metrics) > 0

    def test_rolling_metrics_insufficient_data(self):
        """Test métriques roulantes avec données insuffisantes."""
        short_returns = self.sample_returns.head(50)  # Moins que window=252
        rolling_metrics = self.analyzer.calculate_rolling_metrics(short_returns)

        assert isinstance(rolling_metrics, pd.DataFrame)
        assert len(rolling_metrics) == 0  # Pas assez de données

    def test_comprehensive_analysis(self):
        """Test analyse complète."""
        analysis = self.analyzer.analyze_comprehensive(self.sample_returns)

        assert 'basic_metrics' in analysis
        assert 'risk_metrics' in analysis
        assert 'descriptive_stats' in analysis
        assert 'period_info' in analysis

        # Vérifier les stats descriptives
        stats = analysis['descriptive_stats']
        assert 'mean' in stats
        assert 'std' in stats
        assert 'skewness' in stats
        assert 'kurtosis' in stats

        # Vérifier les infos de période
        period_info = analysis['period_info']
        assert 'start_date' in period_info
        assert 'end_date' in period_info
        assert 'total_periods' in period_info

    def test_comprehensive_analysis_with_trades(self):
        """Test analyse complète avec données de trades."""
        trades_data = {
            'timestamp': pd.date_range('2024-01-01', periods=30, freq='D'),
            'pnl': np.random.normal(15, 40, 30)
        }
        trades_df = pd.DataFrame(trades_data)

        analysis = self.analyzer.analyze_comprehensive(
            self.sample_returns,
            trades_df=trades_df
        )

        assert 'trading_metrics' in analysis
        assert analysis['trading_metrics']['total_trades'] == 30

    def test_strategy_comparison(self):
        """Test comparaison de stratégies."""
        # Créer plusieurs séries de returns
        strategy_returns = {
            'Strategy A': self.sample_returns,
            'Strategy B': self.sample_returns * 1.2,  # Returns amplifiés
            'Strategy C': self.sample_returns * 0.8   # Returns réduits
        }

        comparison = self.analyzer.compare_strategies(strategy_returns)

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 3  # 3 stratégies
        assert 'total_return' in comparison.columns
        assert 'sharpe_ratio' in comparison.columns

        # Vérifier les rankings
        rank_columns = [col for col in comparison.columns if '_rank' in col]
        assert len(rank_columns) > 0

    def test_strategy_comparison_empty(self):
        """Test comparaison avec dictionnaire vide."""
        comparison = self.analyzer.compare_strategies({})
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 0

    def test_performance_summary_generation(self):
        """Test génération de résumé de performance."""
        summary = self.analyzer.generate_performance_summary(
            self.sample_returns,
            strategy_name="Test Strategy"
        )

        assert isinstance(summary, str)
        assert "Test Strategy" in summary
        assert "Performance Metrics" in summary
        assert "Risk Metrics" in summary
        assert "%" in summary  # Pourcentages présents

    def test_edge_cases_empty_returns(self):
        """Test cas limites avec returns vides."""
        empty_returns = pd.Series([], dtype=float)

        basic_metrics = self.analyzer.calculate_basic_metrics(empty_returns)
        assert basic_metrics == {}

        risk_metrics = self.analyzer.calculate_risk_metrics(empty_returns)
        assert risk_metrics == {}

    def test_edge_cases_single_return(self):
        """Test cas limites avec un seul return."""
        single_return = pd.Series([0.01], index=[datetime.now()])

        basic_metrics = self.analyzer.calculate_basic_metrics(single_return)
        assert 'total_return' in basic_metrics
        # Volatilité devrait être 0 ou NaN avec un seul point
        vol = basic_metrics['volatility']
        assert vol == 0 or np.isnan(vol)

    def test_negative_returns_handling(self):
        """Test gestion des returns négatifs."""
        negative_returns = pd.Series([-0.01, -0.02, -0.015, -0.005],
                                   index=pd.date_range('2024-01-01', periods=4))

        metrics = self.analyzer.calculate_basic_metrics(negative_returns)

        assert metrics['total_return'] < 0
        assert metrics['sharpe_ratio'] < 0  # Sharpe négatif avec returns négatifs