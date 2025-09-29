"""
Tests for Core Research Strategies
=================================

Tests ciblés pour les stratégies de recherche essentielles.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from decimal import Decimal

from qframe.strategies.research.dmn_lstm_strategy import DMNLSTMStrategy, DMNConfig
from qframe.strategies.research.adaptive_mean_reversion_strategy import AdaptiveMeanReversionStrategy
from qframe.domain.value_objects.signal import Signal, SignalAction


class TestDMNLSTMStrategy:
    """Tests pour DMNLSTMStrategy."""

    @pytest.fixture
    def dmn_config(self):
        return DMNConfig(
            window_size=20,
            hidden_size=32,
            num_layers=1,
            dropout=0.1,
            learning_rate=0.001,
            signal_threshold=0.1
        )

    @pytest.fixture
    def dmn_strategy(self, dmn_config):
        return DMNLSTMStrategy(config=dmn_config)

    @pytest.fixture
    def sample_data(self):
        """Données OHLCV de test."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
        np.random.seed(42)

        # Générer des données réalistes
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

        return pd.DataFrame({
            'timestamp': dates,
            'open': prices + np.random.randn(100) * 0.1,
            'high': prices + np.abs(np.random.randn(100) * 0.3),
            'low': prices - np.abs(np.random.randn(100) * 0.3),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        }).set_index('timestamp')

    def test_strategy_initialization(self, dmn_strategy, dmn_config):
        """Test initialisation de la stratégie."""
        assert dmn_strategy.config == dmn_config
        assert dmn_strategy.config.window_size == 20
        assert dmn_strategy.config.hidden_size == 32

    def test_prepare_features(self, dmn_strategy, sample_data):
        """Test préparation des features."""
        features = dmn_strategy._prepare_features(sample_data)

        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        assert not features.empty

    def test_create_sequences(self, dmn_strategy, sample_data):
        """Test création de séquences temporelles."""
        features = dmn_strategy._prepare_features(sample_data)
        sequences, targets = dmn_strategy._create_sequences(features)

        assert len(sequences) > 0
        assert len(targets) > 0
        assert len(sequences) == len(targets)

    def test_build_model(self, dmn_strategy):
        """Test construction du modèle."""
        model = dmn_strategy._build_model(input_size=5)

        assert model is not None
        # Vérifier que le modèle a les bonnes couches
        assert hasattr(model, 'lstm')
        assert hasattr(model, 'dropout')
        assert hasattr(model, 'fc')

    @patch('torch.save')
    def test_train_model(self, mock_save, dmn_strategy, sample_data):
        """Test entraînement du modèle."""
        # Mock du processus d'entraînement
        dmn_strategy.model = Mock()
        dmn_strategy.model.train = Mock()

        result = dmn_strategy.train(sample_data)

        assert result is not None
        dmn_strategy.model.train.assert_called()

    def test_generate_signals_basic(self, dmn_strategy, sample_data):
        """Test génération de signaux basique."""
        # Mock du modèle entraîné
        dmn_strategy.model = Mock()
        dmn_strategy.model.eval = Mock()
        dmn_strategy.model.forward = Mock(return_value=Mock(detach=Mock(return_value=Mock(numpy=Mock(return_value=np.array([0.15]))))))
        dmn_strategy._scaler = Mock()
        dmn_strategy._scaler.transform = Mock(return_value=np.random.randn(1, 5))

        signals = dmn_strategy.generate_signals(sample_data)

        assert isinstance(signals, list)

    def test_get_strategy_info(self, dmn_strategy):
        """Test informations de la stratégie."""
        info = dmn_strategy.get_strategy_info()

        assert isinstance(info, dict)
        assert "name" in info
        assert "description" in info
        assert "parameters" in info

    def test_update_config(self, dmn_strategy):
        """Test mise à jour de configuration."""
        new_config = DMNConfig(
            window_size=30,
            hidden_size=64,
            num_layers=2
        )

        dmn_strategy.update_config(new_config)

        assert dmn_strategy.config.window_size == 30
        assert dmn_strategy.config.hidden_size == 64

    def test_predict_returns(self, dmn_strategy, sample_data):
        """Test prédiction de returns."""
        # Mock du modèle
        dmn_strategy.model = Mock()
        dmn_strategy.model.eval = Mock()
        dmn_strategy._scaler = Mock()
        dmn_strategy._scaler.transform = Mock(return_value=np.random.randn(1, 5))

        with patch.object(dmn_strategy, '_prepare_features') as mock_features:
            mock_features.return_value = pd.DataFrame(np.random.randn(50, 5))

            predictions = dmn_strategy.predict_returns(sample_data)

            assert predictions is not None

    def test_calculate_confidence(self, dmn_strategy):
        """Test calcul de confiance."""
        prediction = 0.15
        confidence = dmn_strategy._calculate_confidence(prediction)

        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1


class TestAdaptiveMeanReversionStrategy:
    """Tests pour AdaptiveMeanReversionStrategy."""

    @pytest.fixture
    def mean_reversion_strategy(self):
        # Mock dependencies
        mock_data_provider = Mock()
        mock_risk_manager = Mock()
        mock_config = Mock()
        mock_config.dict.return_value = {}

        return AdaptiveMeanReversionStrategy(
            data_provider=mock_data_provider,
            risk_manager=mock_risk_manager,
            config=mock_config
        )

    @pytest.fixture
    def sample_data(self):
        """Données de test pour mean reversion."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
        np.random.seed(42)

        # Créer des données avec mean reversion
        base_price = 100
        prices = []
        price = base_price

        for i in range(100):
            # Mean reversion vers base_price
            mean_revert = (base_price - price) * 0.1
            random_shock = np.random.randn() * 0.5
            price += mean_revert + random_shock
            prices.append(price)

        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + abs(np.random.randn() * 0.2) for p in prices],
            'low': [p - abs(np.random.randn() * 0.2) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        }).set_index('timestamp')

    def test_strategy_initialization(self, mean_reversion_strategy):
        """Test initialisation de la stratégie."""
        assert mean_reversion_strategy is not None
        assert hasattr(mean_reversion_strategy, 'config')

    def test_detect_regime(self, mean_reversion_strategy, sample_data):
        """Test détection de régime."""
        regime = mean_reversion_strategy._detect_regime(sample_data)

        assert regime in ['trending', 'ranging', 'volatile']

    def test_calculate_z_score(self, mean_reversion_strategy, sample_data):
        """Test calcul de z-score."""
        z_score = mean_reversion_strategy._calculate_z_score(sample_data['close'], window=20)

        assert isinstance(z_score, (pd.Series, float, np.ndarray))

    def test_generate_signals_basic(self, mean_reversion_strategy, sample_data):
        """Test génération de signaux."""
        signals = mean_reversion_strategy.generate_signals(sample_data)

        assert isinstance(signals, list)

    def test_adaptive_thresholds(self, mean_reversion_strategy, sample_data):
        """Test seuils adaptatifs."""
        regime = "ranging"
        thresholds = mean_reversion_strategy._get_adaptive_thresholds(regime)

        assert isinstance(thresholds, dict)
        assert "entry" in thresholds
        assert "exit" in thresholds

    def test_calculate_position_size(self, mean_reversion_strategy):
        """Test calcul de taille de position."""
        signal_strength = 0.8
        portfolio_value = Decimal("10000")

        position_size = mean_reversion_strategy._calculate_position_size(
            signal_strength, portfolio_value
        )

        assert isinstance(position_size, (Decimal, float))
        assert position_size > 0

    def test_feature_engineering(self, mean_reversion_strategy, sample_data):
        """Test feature engineering."""
        features = mean_reversion_strategy._engineer_features(sample_data)

        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0

    def test_volatility_adjustment(self, mean_reversion_strategy, sample_data):
        """Test ajustement de volatilité."""
        vol_adj = mean_reversion_strategy._calculate_volatility_adjustment(sample_data['close'])

        assert isinstance(vol_adj, (float, np.float64))
        assert vol_adj > 0

    def test_regime_parameters(self, mean_reversion_strategy):
        """Test paramètres de régime."""
        for regime in ['trending', 'ranging', 'volatile']:
            params = mean_reversion_strategy._get_regime_parameters(regime)

            assert isinstance(params, dict)
            assert "lookback_multiplier" in params

    def test_performance_metrics(self, mean_reversion_strategy, sample_data):
        """Test métriques de performance."""
        signals = [
            Signal(symbol="BTC/USD", action=SignalAction.BUY, confidence=0.8, timestamp=datetime.utcnow())
        ]

        metrics = mean_reversion_strategy._calculate_performance_metrics(signals, sample_data)

        assert isinstance(metrics, dict)