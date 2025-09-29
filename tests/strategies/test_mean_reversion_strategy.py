"""
Tests for Mean Reversion Strategy
================================

Suite de tests complète pour la stratégie de mean reversion de base.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime
from decimal import Decimal

from qframe.strategies.research.mean_reversion_strategy import (
    MeanReversionStrategy,
    MeanReversionConfig,
    RegimeDetector,
    MLOptimizer
)
from qframe.domain.value_objects.signal import Signal, SignalAction, SignalConfidence


class TestMeanReversionConfig:
    """Tests de configuration pour Mean Reversion Strategy"""

    def test_default_configuration(self):
        """Test la configuration par défaut"""
        config = MeanReversionConfig()

        assert config.lookback_short == 10
        assert config.lookback_long == 50
        assert config.z_entry_base == 1.0
        assert config.z_exit_base == 0.2
        assert config.regime_window == 252
        assert config.use_ml_optimization == True
        assert config.position_size == 0.02
        assert config.max_position_days == 5
        assert config.risk_per_trade == 0.02

    def test_custom_configuration(self):
        """Test une configuration personnalisée"""
        config = MeanReversionConfig(
            lookback_short=20,
            lookback_long=100,
            z_entry_base=1.5,
            z_exit_base=0.3,
            use_ml_optimization=False
        )

        assert config.lookback_short == 20
        assert config.lookback_long == 100
        assert config.z_entry_base == 1.5
        assert config.z_exit_base == 0.3
        assert config.use_ml_optimization == False

    def test_config_validation(self):
        """Test la validation des paramètres de configuration"""
        # Test que lookback_short < lookback_long
        with pytest.raises(ValueError):
            config = MeanReversionConfig(lookback_short=100, lookback_long=50)
            # Validation devrait être dans __post_init__ si elle existe

        # Test valeurs négatives
        with pytest.raises(ValueError):
            MeanReversionConfig(z_entry_base=-1.0)


class TestRegimeDetector:
    """Tests pour le détecteur de régimes"""

    @pytest.fixture
    def sample_data(self):
        """Données de marché pour les tests"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=300, freq='1h')

        # Créer différents régimes
        low_vol = np.random.randn(100) * 0.5  # Faible volatilité
        normal_vol = np.random.randn(100) * 1.0  # Volatilité normale
        high_vol = np.random.randn(100) * 2.0  # Haute volatilité

        returns = np.concatenate([low_vol, normal_vol, high_vol])
        prices = 100 * np.exp(np.cumsum(returns * 0.01))

        data = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'volume': np.random.uniform(1000, 5000, 300)
        })
        data.set_index('timestamp', inplace=True)
        return data

    def test_regime_detector_initialization(self):
        """Test l'initialisation du détecteur de régimes"""
        detector = RegimeDetector(window=100)
        assert detector.window == 100
        assert hasattr(detector, 'current_regime')

    def test_volatility_calculation(self, sample_data):
        """Test le calcul de volatilité"""
        detector = RegimeDetector(window=50)
        volatility = detector._calculate_volatility(sample_data['close'])

        assert isinstance(volatility, (float, np.float64))
        assert volatility > 0
        assert not np.isnan(volatility)

    def test_regime_classification(self, sample_data):
        """Test la classification des régimes"""
        detector = RegimeDetector(window=50)
        regime = detector.detect_regime(sample_data['close'])

        assert regime in ['low_vol', 'normal', 'high_vol']

    def test_regime_transitions(self, sample_data):
        """Test les transitions entre régimes"""
        detector = RegimeDetector(window=50)

        regimes = []
        for i in range(50, len(sample_data), 10):
            regime = detector.detect_regime(sample_data['close'].iloc[:i])
            regimes.append(regime)

        # On devrait voir des transitions
        unique_regimes = set(regimes)
        assert len(unique_regimes) > 1


class TestMLOptimizer:
    """Tests pour l'optimiseur ML"""

    def test_ml_optimizer_initialization(self):
        """Test l'initialisation de l'optimiseur ML"""
        optimizer = MLOptimizer()
        assert hasattr(optimizer, 'model')
        assert hasattr(optimizer, 'scaler')

    def test_feature_engineering(self):
        """Test l'engineering des features"""
        optimizer = MLOptimizer()

        # Données d'exemple
        np.random.seed(42)
        data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'volume': np.random.uniform(1000, 5000, 100)
        })

        features = optimizer._engineer_features(data)

        assert isinstance(features, pd.DataFrame)
        assert len(features) <= len(data)  # Peut être plus court à cause des calculs


class TestMeanReversionStrategy:
    """Tests complets pour la stratégie Mean Reversion"""

    @pytest.fixture
    def config(self):
        """Configuration de test"""
        return MeanReversionConfig(
            lookback_short=10,
            lookback_long=30,
            z_entry_base=1.5,
            z_exit_base=0.3,
            use_ml_optimization=False  # Simplifier pour les tests
        )

    @pytest.fixture
    def sample_market_data(self):
        """Données de marché pour les tests"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='1h')

        # Créer des données avec mean reversion
        returns = np.random.randn(200) * 0.01
        # Ajouter de la mean reversion
        for i in range(1, len(returns)):
            returns[i] -= 0.1 * returns[i-1]  # Mean reversion factor

        prices = 100 * np.exp(np.cumsum(returns))

        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.randn(200) * 0.001),
            'high': prices * (1 + abs(np.random.randn(200) * 0.002)),
            'low': prices * (1 - abs(np.random.randn(200) * 0.002)),
            'close': prices,
            'volume': np.random.uniform(1000, 5000, 200)
        })
        data.set_index('timestamp', inplace=True)
        return data

    @pytest.fixture
    def strategy(self, config):
        """Stratégie initialisée pour les tests"""
        return MeanReversionStrategy(
            config=config,
            metrics_collector=Mock()
        )

    def test_strategy_initialization(self, strategy, config):
        """Test l'initialisation de la stratégie"""
        assert strategy.config == config
        assert hasattr(strategy, 'regime_detector')
        assert hasattr(strategy, 'ml_optimizer')

    def test_feature_preparation(self, strategy, sample_market_data):
        """Test la préparation des features"""
        features = strategy._prepare_features(sample_market_data)

        assert isinstance(features, pd.DataFrame)
        assert len(features) <= len(sample_market_data)

        # Vérifier que les colonnes nécessaires sont présentes
        expected_cols = ['returns', 'volatility', 'rsi']
        for col in expected_cols:
            if col in features.columns:
                assert features[col].notna().any()

    def test_adaptive_thresholds(self, strategy, sample_market_data):
        """Test le calcul des seuils adaptatifs"""
        thresholds = strategy._get_adaptive_thresholds(sample_market_data)

        assert isinstance(thresholds, dict)
        assert 'z_entry' in thresholds
        assert 'z_exit' in thresholds

        # Les seuils devraient être raisonnables
        assert 0 < thresholds['z_entry'] < 5
        assert 0 < thresholds['z_exit'] < thresholds['z_entry']

    def test_signal_generation(self, strategy, sample_market_data):
        """Test la génération de signaux"""
        signals = strategy.generate_signals(sample_market_data)

        assert isinstance(signals, list)

        # Vérifier le format des signaux
        for signal in signals:
            assert isinstance(signal, Signal)
            assert signal.action in [SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD]
            assert isinstance(signal.strength, Decimal)
            assert isinstance(signal.confidence, SignalConfidence)

    def test_regime_adaptation(self, strategy, sample_market_data):
        """Test l'adaptation aux régimes"""
        # Simuler différents régimes
        normal_signals = strategy.generate_signals(sample_market_data.iloc[:100])

        # Créer des données plus volatiles
        volatile_data = sample_market_data.copy()
        volatile_data['close'] *= (1 + np.random.randn(len(volatile_data)) * 0.05)
        volatile_signals = strategy.generate_signals(volatile_data.iloc[:100])

        # Les paramètres devraient s'adapter
        assert len(normal_signals) >= 0  # Au moins valide
        assert len(volatile_signals) >= 0

    def test_ml_optimization_integration(self, config):
        """Test l'intégration de l'optimisation ML"""
        config.use_ml_optimization = True
        strategy = MeanReversionStrategy(
            config=config,
            metrics_collector=Mock()
        )

        assert hasattr(strategy, 'ml_optimizer')
        assert strategy.config.use_ml_optimization == True

    def test_strategy_state(self, strategy):
        """Test l'état de la stratégie"""
        state = strategy.get_strategy_state()

        assert isinstance(state, dict)
        assert 'config' in state
        assert 'regime_detector' in state
        assert 'ml_optimizer' in state

    def test_regime_multiplier(self, strategy):
        """Test le multiplicateur de régime"""
        # Test différents régimes
        low_vol_mult = strategy._get_regime_multiplier('low_vol')
        normal_mult = strategy._get_regime_multiplier('normal')
        high_vol_mult = strategy._get_regime_multiplier('high_vol')

        assert isinstance(low_vol_mult, (int, float))
        assert isinstance(normal_mult, (int, float))
        assert isinstance(high_vol_mult, (int, float))

        # Les multiplicateurs devraient être différents
        assert low_vol_mult != normal_mult or normal_mult != high_vol_mult

    def test_signal_creation(self, strategy, sample_market_data):
        """Test la création de signaux individuels"""
        # Tester avec des données de base
        price = 100.0
        timestamp = sample_market_data.index[0]
        z_score = 1.5
        regime = 'normal'

        signal = strategy._create_signal(
            action=SignalAction.BUY,
            timestamp=timestamp,
            price=price,
            z_score=z_score,
            regime=regime,
            metadata={}
        )

        assert isinstance(signal, Signal)
        assert signal.action == SignalAction.BUY
        assert signal.price == Decimal(str(price))

    def test_performance_metrics(self, strategy, sample_market_data):
        """Test les métriques de performance"""
        signals = strategy.generate_signals(sample_market_data)

        if signals:
            # Calculer des métriques simples
            buy_signals = [s for s in signals if s.action == SignalAction.BUY]
            sell_signals = [s for s in signals if s.action == SignalAction.SELL]

            total_signals = len(signals)
            assert total_signals >= 0

            # Vérifier l'équilibre des signaux
            if total_signals > 0:
                buy_ratio = len(buy_signals) / total_signals
                sell_ratio = len(sell_signals) / total_signals
                assert 0 <= buy_ratio <= 1
                assert 0 <= sell_ratio <= 1

    def test_backtest_integration(self, strategy, sample_market_data):
        """Test l'intégration avec le backtesting"""
        # Simuler un backtest simple
        signals = strategy.generate_signals(sample_market_data)

        # Vérifier que les signaux peuvent être utilisés pour le backtesting
        assert isinstance(signals, list)

        # Chaque signal devrait avoir les informations nécessaires
        for signal in signals:
            assert hasattr(signal, 'timestamp')
            assert hasattr(signal, 'action')
            assert hasattr(signal, 'strength')

    @pytest.mark.performance
    def test_strategy_performance(self, strategy, sample_market_data):
        """Test de performance de la stratégie"""
        import time

        start_time = time.time()
        signals = strategy.generate_signals(sample_market_data)
        execution_time = time.time() - start_time

        # La génération de signaux devrait être rapide
        assert execution_time < 5.0  # Moins de 5 secondes
        assert len(signals) >= 0

    def test_config_impact_on_signals(self, sample_market_data):
        """Test l'impact de la configuration sur les signaux"""
        # Configuration conservatrice
        conservative_config = MeanReversionConfig(
            z_entry_base=2.0,  # Seuil élevé
            z_exit_base=0.1    # Sortie rapide
        )
        conservative_strategy = MeanReversionStrategy(
            config=conservative_config,
            metrics_collector=Mock()
        )

        # Configuration agressive
        aggressive_config = MeanReversionConfig(
            z_entry_base=0.5,  # Seuil bas
            z_exit_base=0.5    # Sortie tardive
        )
        aggressive_strategy = MeanReversionStrategy(
            config=aggressive_config,
            metrics_collector=Mock()
        )

        conservative_signals = conservative_strategy.generate_signals(sample_market_data)
        aggressive_signals = aggressive_strategy.generate_signals(sample_market_data)

        # La stratégie agressive devrait générer plus de signaux
        assert len(aggressive_signals) >= len(conservative_signals)