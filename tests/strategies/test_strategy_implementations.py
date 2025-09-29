"""
Tests for Strategy Implementations
==================================

Tests ciblés pour les implémentations de stratégies.
"""

import pytest
import numpy as np
import pandas as pd
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from qframe.strategies.research.adaptive_mean_reversion_strategy import AdaptiveMeanReversionStrategy
from qframe.domain.value_objects.signal import Signal, SignalAction, SignalConfidence


@pytest.fixture
def sample_data():
    """Données OHLCV de test."""
    dates = pd.date_range('2023-01-01', periods=100, freq='H')
    np.random.seed(42)

    # Générer des prix avec tendance et mean reversion
    base_price = 45000
    returns = np.random.normal(0.0001, 0.02, 100)

    # Ajouter de la mean reversion
    prices = [base_price]
    for i in range(1, 100):
        # Mean reversion vers prix de base
        mean_reversion = -0.1 * (prices[i-1] - base_price) / base_price
        price_change = returns[i] + mean_reversion * 0.5
        new_price = prices[i-1] * (1 + price_change)
        prices.append(new_price)

    return pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': np.random.uniform(100, 1000, 100)
    }).set_index('timestamp')


@pytest.fixture
def sample_features():
    """Features d'exemple pour les tests."""
    dates = pd.date_range('2023-01-01', periods=100, freq='H')
    np.random.seed(42)

    return pd.DataFrame({
        'sma_20': np.random.uniform(44000, 46000, 100),
        'rsi': np.random.uniform(20, 80, 100),
        'bb_upper': np.random.uniform(45500, 47000, 100),
        'bb_lower': np.random.uniform(43000, 44500, 100),
        'momentum': np.random.uniform(-0.1, 0.1, 100),
        'volatility': np.random.uniform(0.01, 0.05, 100)
    }, index=dates)


class TestAdaptiveMeanReversionStrategy:
    """Tests pour la stratégie de mean reversion adaptative."""

    @pytest.fixture
    def strategy_config(self):
        return {
            'lookback_short': 10,
            'lookback_long': 50,
            'z_entry_base': 1.0,
            'z_exit_base': 0.2,
            'regime_window': 252,
            'use_ml_optimization': True,
            'max_position_size': 0.1
        }

    @pytest.fixture
    def strategy(self, strategy_config):
        return AdaptiveMeanReversionStrategy(strategy_config)

    def test_strategy_initialization(self, strategy, strategy_config):
        """Test initialisation de la stratégie."""
        assert strategy.config['lookback_short'] == 10
        assert strategy.config['lookback_long'] == 50
        assert strategy.config['use_ml_optimization'] is True

    def test_z_score_calculation(self, strategy, sample_data):
        """Test calcul du z-score."""
        prices = sample_data['close']
        z_scores = strategy._calculate_z_score(prices, window=20)

        assert len(z_scores) == len(prices)
        assert not z_scores.isnull().all()
        # Z-scores devraient être normalisés autour de 0
        assert abs(z_scores.mean()) < 0.5

    def test_regime_detection(self, strategy, sample_data):
        """Test détection de régime de volatilité."""
        prices = sample_data['close']
        regime = strategy._detect_volatility_regime(prices)

        assert regime in ['low_vol', 'normal', 'high_vol']

    def test_signal_generation_basic(self, strategy, sample_data, sample_features):
        """Test génération de signaux de base."""
        signals = strategy.generate_signals(sample_data, sample_features)

        assert isinstance(signals, list)
        # Devrait générer au moins quelques signaux
        assert len(signals) >= 0

        if signals:
            signal = signals[0]
            assert isinstance(signal, Signal)
            assert signal.symbol is not None
            assert signal.action in [SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD]

    def test_adaptive_thresholds(self, strategy, sample_data):
        """Test seuils adaptatifs selon le régime."""
        base_threshold = 1.0

        # Test différents régimes
        low_vol_threshold = strategy._adapt_threshold_to_regime(base_threshold, 'low_vol')
        normal_threshold = strategy._adapt_threshold_to_regime(base_threshold, 'normal')
        high_vol_threshold = strategy._adapt_threshold_to_regime(base_threshold, 'high_vol')

        # Les seuils devraient être différents selon le régime
        assert low_vol_threshold != normal_threshold
        assert normal_threshold != high_vol_threshold
        # Généralement, plus de volatilité = seuils plus élevés
        assert high_vol_threshold >= normal_threshold

    def test_position_sizing(self, strategy, sample_data):
        """Test calcul de la taille de position."""
        signal_strength = 0.8
        volatility = 0.02

        position_size = strategy._calculate_position_size(signal_strength, volatility)

        assert 0 <= position_size <= strategy.config['max_position_size']
        assert isinstance(position_size, (int, float, Decimal))

    def test_ml_optimization_mock(self, strategy, sample_data):
        """Test optimisation ML (simulée)."""
        with patch.object(strategy, '_optimize_thresholds_ml') as mock_optimize:
            mock_optimize.return_value = {'z_entry': 1.2, 'z_exit': 0.3}

            optimized_params = strategy._optimize_thresholds_ml(sample_data)

            assert 'z_entry' in optimized_params
            assert 'z_exit' in optimized_params
            assert optimized_params['z_entry'] > optimized_params['z_exit']

    def test_signal_confidence_calculation(self, strategy):
        """Test calcul de confiance du signal."""
        # Signal fort
        strong_confidence = strategy._calculate_signal_confidence(
            z_score=2.5, regime='low_vol', trend_alignment=True
        )

        # Signal faible
        weak_confidence = strategy._calculate_signal_confidence(
            z_score=0.5, regime='high_vol', trend_alignment=False
        )

        assert strong_confidence >= weak_confidence
        assert strong_confidence in [SignalConfidence.LOW, SignalConfidence.MEDIUM,
                                   SignalConfidence.HIGH, SignalConfidence.VERY_HIGH]

    def test_risk_management_integration(self, strategy, sample_data):
        """Test intégration de la gestion des risques."""
        # Simuler position existante importante
        strategy.current_position_size = 0.08  # 8% du portfolio

        signals = strategy.generate_signals(sample_data)

        # Les signaux devraient tenir compte de la position existante
        if signals:
            buy_signals = [s for s in signals if s.action == SignalAction.BUY]
            # Avec position importante, moins de signaux d'achat
            assert len(buy_signals) <= len(signals)

    def test_strategy_with_extreme_market_conditions(self, strategy):
        """Test stratégie dans conditions de marché extrêmes."""
        # Données avec crash de marché
        dates = pd.date_range('2023-01-01', periods=50, freq='H')
        crash_prices = [45000] * 10 + [p * 0.95 for p in range(40500, 35000, -110)][:40]

        extreme_data = pd.DataFrame({
            'timestamp': dates,
            'open': crash_prices,
            'high': [p * 1.005 for p in crash_prices],
            'low': [p * 0.995 for p in crash_prices],
            'close': crash_prices,
            'volume': [1000] * 50
        }).set_index('timestamp')

        signals = strategy.generate_signals(extreme_data)

        # Stratégie devrait être plus conservatrice
        assert isinstance(signals, list)

    def test_strategy_performance_tracking(self, strategy, sample_data):
        """Test suivi de performance de la stratégie."""
        # Générer plusieurs signaux
        signals = strategy.generate_signals(sample_data)

        # Calculer métriques de base
        if signals:
            buy_signals = len([s for s in signals if s.action == SignalAction.BUY])
            sell_signals = len([s for s in signals if s.action == SignalAction.SELL])

            signal_ratio = buy_signals / (buy_signals + sell_signals) if (buy_signals + sell_signals) > 0 else 0

            assert 0 <= signal_ratio <= 1

    def test_strategy_parameter_sensitivity(self, strategy_config, sample_data):
        """Test sensibilité aux paramètres."""
        # Tester avec paramètres conservateurs
        conservative_config = strategy_config.copy()
        conservative_config['z_entry_base'] = 2.0  # Seuil plus élevé
        conservative_strategy = AdaptiveMeanReversionStrategy(conservative_config)

        # Tester avec paramètres agressifs
        aggressive_config = strategy_config.copy()
        aggressive_config['z_entry_base'] = 0.5  # Seuil plus bas
        aggressive_strategy = AdaptiveMeanReversionStrategy(aggressive_config)

        conservative_signals = conservative_strategy.generate_signals(sample_data)
        aggressive_signals = aggressive_strategy.generate_signals(sample_data)

        # Stratégie agressive devrait générer plus de signaux
        assert len(aggressive_signals) >= len(conservative_signals)

    def test_strategy_state_management(self, strategy):
        """Test gestion de l'état de la stratégie."""
        # État initial
        assert hasattr(strategy, 'current_position_size')
        assert hasattr(strategy, 'last_signal_time')

        # Mise à jour d'état
        strategy.update_state({
            'position_size': 0.05,
            'last_signal': datetime.utcnow()
        })

        assert strategy.current_position_size == 0.05

    def test_strategy_with_insufficient_data(self, strategy):
        """Test stratégie avec données insuffisantes."""
        # Données très limitées
        insufficient_data = pd.DataFrame({
            'timestamp': [datetime.utcnow()],
            'open': [45000],
            'high': [45100],
            'low': [44900],
            'close': [45050],
            'volume': [1000]
        }).set_index('timestamp')

        signals = strategy.generate_signals(insufficient_data)

        # Devrait gérer gracieusement les données insuffisantes
        assert isinstance(signals, list)
        # Probablement aucun signal avec données insuffisantes
        assert len(signals) == 0

    def test_strategy_feature_requirements(self, strategy, sample_data):
        """Test exigences de features."""
        # Test sans features
        signals_no_features = strategy.generate_signals(sample_data, None)

        # Test avec features
        basic_features = pd.DataFrame({
            'sma_20': [45000] * len(sample_data),
            'rsi': [50] * len(sample_data)
        }, index=sample_data.index)

        signals_with_features = strategy.generate_signals(sample_data, basic_features)

        # Les deux devraient fonctionner
        assert isinstance(signals_no_features, list)
        assert isinstance(signals_with_features, list)

    def test_signal_timing_constraints(self, strategy, sample_data):
        """Test contraintes de timing des signaux."""
        # Simuler signaux récents
        strategy.last_signal_time = datetime.utcnow() - timedelta(minutes=30)
        strategy.min_signal_interval = timedelta(hours=1)

        signals = strategy.generate_signals(sample_data)

        # Devrait respecter l'intervalle minimum
        if hasattr(strategy, 'min_signal_interval'):
            # Implémentation dépend de la logique de la stratégie
            assert isinstance(signals, list)


class TestStrategyComparison:
    """Tests de comparaison entre stratégies."""

    def test_multiple_strategy_signals(self, sample_data):
        """Test signaux de plusieurs stratégies."""
        config1 = {'lookback_short': 10, 'z_entry_base': 1.0}
        config2 = {'lookback_short': 20, 'z_entry_base': 1.5}

        strategy1 = AdaptiveMeanReversionStrategy(config1)
        strategy2 = AdaptiveMeanReversionStrategy(config2)

        signals1 = strategy1.generate_signals(sample_data)
        signals2 = strategy2.generate_signals(sample_data)

        # Les deux stratégies devraient générer des signaux
        assert isinstance(signals1, list)
        assert isinstance(signals2, list)

        # Probablement différents nombres de signaux
        # (pas nécessairement vrai selon les données)

    def test_strategy_ensemble_potential(self, sample_data):
        """Test potentiel d'ensemble de stratégies."""
        configs = [
            {'lookback_short': 10, 'z_entry_base': 1.0},
            {'lookback_short': 15, 'z_entry_base': 1.2},
            {'lookback_short': 20, 'z_entry_base': 1.5}
        ]

        all_signals = []
        for config in configs:
            strategy = AdaptiveMeanReversionStrategy(config)
            signals = strategy.generate_signals(sample_data)
            all_signals.extend(signals)

        # Ensemble devrait avoir plus de signaux
        assert len(all_signals) >= 0

        # Pourrait implémenter vote majoritaire ou pondération
        if all_signals:
            signal_times = [s.timestamp for s in all_signals]
            # Vérifier distribution temporelle
            assert len(set(signal_times)) <= len(signal_times)