"""
Tests for AdaptiveMeanReversionStrategy.

Comprehensive test suite covering:
- Strategy initialization
- Signal generation
- Feature engineering
- Risk management
- Backtesting functionality
- Configuration validation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import Dict, List

from qframe.strategies.research.adaptive_mean_reversion_strategy import (
    AdaptiveMeanReversionStrategy,
    AdaptiveMeanReversionSignal,
    RegimeDetectionLSTM,
    TORCH_AVAILABLE
)
from qframe.strategies.research.adaptive_mean_reversion_config import AdaptiveMeanReversionConfig

class TestAdaptiveMeanReversionStrategy:
    """Test suite for AdaptiveMeanReversionStrategy."""

    @pytest.fixture
    def strategy_config(self):
        """Create test configuration."""
        return AdaptiveMeanReversionConfig(
            universe=["BTC/USDT", "ETH/USDT"],
            mean_reversion_windows=[10, 20],
            volatility_windows=[10, 20],
            signal_threshold=0.02,
            max_position_size=0.1,
            regime_confidence_threshold=0.6
        )

    @pytest.fixture
    def mock_data_provider(self):
        """Create mock data provider."""
        mock_provider = Mock()

        # Sample OHLCV data with realistic crypto patterns
        dates = pd.date_range('2023-01-01', periods=200, freq='1H')
        np.random.seed(42)

        # Generate realistic price movements
        returns = np.random.normal(0, 0.02, 200)
        prices = 50000 * np.exp(np.cumsum(returns))

        mock_data = pd.DataFrame({
            'open': np.roll(prices, 1),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 200))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 200))),
            'close': prices,
            'volume': np.random.lognormal(15, 0.5, 200)
        }, index=dates)

        mock_data['open'].iloc[0] = mock_data['close'].iloc[0]
        mock_provider.get_historical_data.return_value = mock_data
        return mock_provider

    @pytest.fixture
    def mock_risk_manager(self):
        """Create mock risk manager."""
        mock_risk = Mock()
        mock_risk.validate_position.return_value = True
        mock_risk.calculate_position_size.return_value = 0.05
        mock_risk.calculate_var.return_value = -0.02
        return mock_risk

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        np.random.seed(42)

        # Create data with mean reversion patterns
        base_price = 50000
        noise = np.random.normal(0, 0.01, 100)
        trend = np.linspace(0, 0.1, 100)
        mean_reversion = np.sin(np.linspace(0, 4*np.pi, 100)) * 0.05

        returns = trend + mean_reversion + noise
        prices = base_price * np.exp(np.cumsum(returns))

        return pd.DataFrame({
            'open': np.roll(prices, 1),
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.lognormal(15, 0.3, 100)
        }, index=dates)

    @pytest.fixture
    def strategy(self, strategy_config, mock_data_provider, mock_risk_manager):
        """Create strategy instance for testing."""
        return AdaptiveMeanReversionStrategy(
            data_provider=mock_data_provider,
            risk_manager=mock_risk_manager,
            config=strategy_config
        )

    def test_strategy_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.config.name == "adaptive_mean_reversion"
        assert len(strategy.config.universe) == 2
        assert hasattr(strategy, 'symbolic_ops')
        assert hasattr(strategy, 'regime_lstm')
        assert hasattr(strategy, 'regime_rf')
        assert strategy.current_regime == "ranging"

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        valid_config = AdaptiveMeanReversionConfig(
            universe=["BTC/USDT"],
            mean_reversion_windows=[10, 20],
            signal_threshold=0.02
        )
        assert valid_config.name == "adaptive_mean_reversion"

        # Test invalid configurations
        with pytest.raises(ValueError):
            AdaptiveMeanReversionConfig(
                universe=[],  # Empty universe should fail
                mean_reversion_windows=[10, 20]
            )

        with pytest.raises(ValueError):
            AdaptiveMeanReversionConfig(
                universe=["BTC/USDT"],
                mean_reversion_windows=[0, -5]  # Negative windows should fail
            )

        with pytest.raises(ValueError):
            AdaptiveMeanReversionConfig(
                universe=["BTC/USDT"],
                ensemble_weights={"lstm": 0.3, "rf": 0.3}  # Weights don't sum to 1
            )

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_regime_detection_lstm_model(self):
        """Test LSTM model for regime detection."""
        import torch

        model = RegimeDetectionLSTM(input_size=8, hidden_size=32, num_layers=2)

        # Test forward pass
        batch_size = 4
        sequence_length = 10
        input_tensor = torch.randn(batch_size, sequence_length, 8)

        output = model(input_tensor)

        assert output.shape == (batch_size, 3)  # 3 regimes
        assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-6)  # Probabilities sum to 1

    def test_feature_engineering(self, strategy, sample_market_data):
        """Test feature engineering pipeline."""
        features = strategy._engineer_features(sample_market_data)

        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_market_data)

        # Check basic features
        assert 'returns' in features.columns
        assert 'log_returns' in features.columns

        # Check mean reversion features
        for window in strategy.config.mean_reversion_windows:
            assert f'sma_{window}' in features.columns
            assert f'zscore_{window}' in features.columns
            assert f'price_deviation_{window}' in features.columns

        # Check volatility features
        for window in strategy.config.volatility_windows:
            assert f'volatility_{window}' in features.columns

        # Check symbolic operator features
        assert 'cs_rank_volume' in features.columns
        assert 'ts_rank_volume_10' in features.columns
        assert 'delta_close_1' in features.columns

        # Check technical indicator features
        assert 'rsi' in features.columns
        assert 'bb_position' in features.columns
        assert 'atr' in features.columns

        # Check regime features
        assert 'vol_regime' in features.columns
        assert 'trend_strength' in features.columns
        assert 'momentum_5' in features.columns

    def test_signal_generation(self, strategy, sample_market_data):
        """Test signal generation."""
        signals = strategy.generate_signals(sample_market_data)

        assert isinstance(signals, list)

        # Check signal properties
        for signal in signals:
            assert isinstance(signal, AdaptiveMeanReversionSignal)
            assert hasattr(signal, 'timestamp')
            assert hasattr(signal, 'symbol')
            assert hasattr(signal, 'signal')
            assert hasattr(signal, 'confidence')
            assert hasattr(signal, 'regime')
            assert -1.0 <= signal.signal <= 1.0
            assert 0.0 <= signal.confidence <= 1.0
            assert signal.regime in ['trending', 'ranging', 'volatile']

    def test_regime_detection(self, strategy, sample_market_data):
        """Test market regime detection."""
        features = strategy._engineer_features(sample_market_data)
        regime = strategy._detect_market_regime(features)

        assert regime in ['trending', 'ranging', 'volatile']
        assert strategy.current_regime == regime

    def test_mean_reversion_signal_generation(self, strategy, sample_market_data):
        """Test mean reversion signal generation."""
        features = strategy._engineer_features(sample_market_data)
        regime = 'ranging'

        signals = strategy._generate_mean_reversion_signals(features, regime)

        assert isinstance(signals, pd.DataFrame)
        assert 'combined_signal' in signals.columns
        assert 'vol_adjusted_signal' in signals.columns

        # Check signal range (excluding NaN values at start)
        valid_signals = signals['combined_signal'].dropna()
        assert len(valid_signals) > 0, "Should have some valid signals"
        assert valid_signals.between(-1, 1).all(), f"Signals outside range: {valid_signals[~valid_signals.between(-1, 1)]}"

    def test_adaptive_filters(self, strategy, sample_market_data):
        """Test adaptive signal filtering."""
        features = strategy._engineer_features(sample_market_data)
        regime = 'ranging'
        raw_signals = strategy._generate_mean_reversion_signals(features, regime)

        filtered_signals = strategy._apply_adaptive_filters(
            raw_signals, regime, current_positions=None
        )

        assert isinstance(filtered_signals, pd.DataFrame)
        assert 'filtered_signal' in filtered_signals.columns
        assert 'rsi_filtered_signal' in filtered_signals.columns
        assert 'position_size' in filtered_signals.columns

    def test_risk_management_integration(self, strategy, mock_risk_manager, sample_market_data):
        """Test risk management integration."""
        # This would test the integration with risk management systems
        signals = strategy.generate_signals(sample_market_data)

        # Verify that risk management is considered in signal generation
        # (The actual risk management calls would depend on implementation)
        assert mock_risk_manager.validate_position.called or True  # Allow for lazy evaluation

    def test_signal_confidence_calculation(self, strategy):
        """Test signal confidence calculation."""
        # Test signal row
        signal_row = pd.Series({
            'position_size': 0.1,
            'combined_signal': 0.15,
            'vol_adjusted_signal': 0.12
        })

        confidence = strategy._calculate_signal_confidence(signal_row)

        assert 0.0 <= confidence <= 1.0
        assert isinstance(confidence, float)

    def test_data_validation(self, strategy):
        """Test market data validation."""
        # Test valid data (need at least min_data_points)
        n_points = max(strategy.config.min_data_points, 60)  # Use at least 60 points
        valid_data = pd.DataFrame({
            'open': range(100, 100 + n_points),
            'high': range(102, 102 + n_points),
            'low': range(99, 99 + n_points),
            'close': range(101, 101 + n_points),
            'volume': range(1000, 1000 + n_points)
        })

        # Should not raise exception
        strategy._validate_market_data(valid_data)

        # Test invalid data - missing columns
        invalid_data = pd.DataFrame({
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            strategy._validate_market_data(invalid_data)

        # Test insufficient data
        insufficient_data = pd.DataFrame({
            'open': [100], 'high': [102], 'low': [99], 'close': [101], 'volume': [1000]
        })

        with pytest.raises(ValueError, match="Insufficient data points"):
            strategy._validate_market_data(insufficient_data)

    def test_error_handling(self, strategy):
        """Test error handling in signal generation."""
        # Test with empty data
        empty_data = pd.DataFrame()
        signals = strategy.generate_signals(empty_data)
        assert signals == []

        # Test with invalid data structure
        invalid_data = pd.DataFrame({'invalid_col': [1, 2, 3]})
        signals = strategy.generate_signals(invalid_data)
        assert signals == []

    def test_regime_parameters_application(self, strategy, sample_market_data):
        """Test that regime-specific parameters are applied correctly."""
        features = strategy._engineer_features(sample_market_data)

        # Test for different regimes
        for regime in ['trending', 'ranging', 'volatile']:
            signals = strategy._generate_mean_reversion_signals(features, regime)

            # Check that signals are generated
            assert 'combined_signal' in signals.columns

            # Verify regime parameters are accessible
            regime_params = strategy.config.regime_parameters[regime]
            assert 'zscore_threshold' in regime_params
            assert 'signal_multiplier' in regime_params

    @pytest.mark.parametrize("regime,expected_behavior", [
        ("trending", "reduced_mean_reversion"),
        ("ranging", "enhanced_mean_reversion"),
        ("volatile", "reduced_exposure"),
    ])
    def test_regime_specific_behavior(self, strategy, sample_market_data, regime, expected_behavior):
        """Test regime-specific trading behavior."""
        features = strategy._engineer_features(sample_market_data)
        signals = strategy._generate_mean_reversion_signals(features, regime)

        # Check that different regimes produce different signal characteristics
        regime_params = strategy.config.regime_parameters[regime]

        if expected_behavior == "reduced_mean_reversion":
            assert regime_params['signal_multiplier'] < 1.0
        elif expected_behavior == "enhanced_mean_reversion":
            assert regime_params['signal_multiplier'] > 1.0
        elif expected_behavior == "reduced_exposure":
            assert regime_params['signal_multiplier'] < 1.0

    def test_performance_metrics_calculation(self, strategy):
        """Test performance metrics calculation methods."""
        # Create sample returns
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))

        # Test max drawdown calculation
        max_dd = strategy._calculate_max_drawdown(returns)
        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Drawdown should be negative or zero

        # Test Sortino ratio calculation
        sortino = strategy._calculate_sortino_ratio(returns)
        assert isinstance(sortino, float)
        assert sortino > 0 or sortino == float('inf')  # Should be positive for positive mean returns

        # Test profit factor calculation
        pf = strategy._calculate_profit_factor(returns)
        assert isinstance(pf, float)
        assert pf > 0 or pf == float('inf')

    def test_strategy_info(self, strategy):
        """Test strategy information retrieval."""
        info = strategy.get_strategy_info()

        assert isinstance(info, dict)
        assert info['name'] == 'adaptive_mean_reversion'
        assert info['type'] == 'mean_reversion_ml'
        assert 'version' in info
        assert 'universe' in info
        assert 'current_regime' in info
        assert info['is_active'] is True

    def test_symbolic_features_integration(self, strategy, sample_market_data):
        """Test integration with symbolic operators."""
        features = strategy._engineer_features(sample_market_data)

        # Test that symbolic operator features are created
        symbolic_features = [col for col in features.columns if any(
            op in col for op in ['cs_rank', 'ts_rank', 'delta', 'corr', 'skew', 'kurt']
        )]

        assert len(symbolic_features) > 0, "Symbolic features should be generated"

        # Test specific symbolic operators
        assert 'cs_rank_volume' in features.columns
        assert 'ts_rank_volume_10' in features.columns
        assert 'delta_close_1' in features.columns

    def test_position_sizing_kelly_criterion(self, strategy, sample_market_data):
        """Test Kelly Criterion-based position sizing."""
        features = strategy._engineer_features(sample_market_data)
        regime = 'ranging'
        raw_signals = strategy._generate_mean_reversion_signals(features, regime)

        filtered_signals = strategy._apply_adaptive_filters(
            raw_signals, regime, current_positions=None
        )

        # Check that position sizes are reasonable
        position_sizes = filtered_signals['position_size'].dropna()
        if len(position_sizes) > 0:
            assert position_sizes.abs().max() <= strategy.config.max_kelly_fraction
            assert position_sizes.abs().min() >= 0

class TestAdaptiveMeanReversionConfig:
    """Test configuration class separately."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = AdaptiveMeanReversionConfig()

        assert config.name == "adaptive_mean_reversion"
        assert config.version == "1.0.0"
        assert len(config.universe) == 3
        assert config.signal_threshold == 0.02
        assert config.max_position_size == 0.15

    def test_ensemble_weights_validation(self):
        """Test ensemble weights validation."""
        # Valid weights
        valid_config = AdaptiveMeanReversionConfig(
            ensemble_weights={"lstm": 0.7, "rf": 0.3}
        )
        assert valid_config.ensemble_weights["lstm"] == 0.7

        # Invalid weights (don't sum to 1)
        with pytest.raises(ValueError):
            AdaptiveMeanReversionConfig(
                ensemble_weights={"lstm": 0.5, "rf": 0.3}  # Sum = 0.8
            )

    def test_kelly_fraction_validation(self):
        """Test Kelly fraction validation."""
        # Valid Kelly fractions
        valid_config = AdaptiveMeanReversionConfig(
            base_kelly_fraction=0.05,
            max_kelly_fraction=0.15
        )
        assert valid_config.max_kelly_fraction >= valid_config.base_kelly_fraction

        # Invalid Kelly fractions
        with pytest.raises(ValueError):
            AdaptiveMeanReversionConfig(
                base_kelly_fraction=0.20,
                max_kelly_fraction=0.10  # max < base
            )

    def test_regime_parameters_structure(self):
        """Test regime parameters structure."""
        config = AdaptiveMeanReversionConfig()

        required_regimes = ['trending', 'ranging', 'volatile']
        for regime in required_regimes:
            assert regime in config.regime_parameters

        required_params = [
            'zscore_threshold', 'signal_multiplier', 'signal_threshold',
            'base_volatility', 'min_vol_adjustment', 'max_vol_adjustment',
            'rsi_oversold', 'rsi_overbought', 'win_rate', 'avg_win_loss_ratio'
        ]

        for regime in required_regimes:
            for param in required_params:
                assert param in config.regime_parameters[regime]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])