"""
Configuration for AdaptiveMeanReversionStrategy.

This configuration supports adaptive mean reversion with ML-based regime detection.
The strategy adapts its parameters based on detected market regimes.
"""

from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field, validator
from decimal import Decimal

class AdaptiveMeanReversionConfig(BaseModel):
    """Configuration parameters for Adaptive Mean Reversion Strategy."""

    # Strategy identification
    name: str = "adaptive_mean_reversion"
    version: str = "1.0.0"
    description: str = "Mean reversion adaptative avec détection ML des régimes de marché"

    # Universe and timeframe
    universe: List[str] = Field(
        default=["BTC/USDT", "ETH/USDT", "BNB/USDT"],
        description="Trading universe symbols"
    )
    timeframe: str = Field(
        default="1h",
        description="Strategy timeframe"
    )

    # Signal generation parameters
    signal_threshold: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Minimum signal strength threshold"
    )
    min_signal_strength: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Minimum signal strength to generate trade"
    )

    # Mean reversion parameters
    mean_reversion_windows: List[int] = Field(
        default=[10, 20, 50],
        description="Lookback windows for mean reversion calculation"
    )
    volatility_windows: List[int] = Field(
        default=[10, 20, 50],
        description="Windows for volatility calculations"
    )
    correlation_window: int = Field(
        default=60,
        gt=0,
        description="Window for correlation calculations"
    )

    # ML Model parameters for regime detection
    regime_features_window: int = Field(
        default=60,
        gt=0,
        description="Window size for regime detection features"
    )
    regime_confidence_threshold: float = Field(
        default=0.6,
        ge=0.5,
        le=1.0,
        description="Minimum confidence to change regime"
    )

    # LSTM model parameters
    lstm_input_size: int = Field(
        default=8,
        gt=0,
        description="Input size for LSTM regime detection model"
    )
    lstm_hidden_size: int = Field(
        default=64,
        gt=0,
        description="Hidden size for LSTM model"
    )
    lstm_num_layers: int = Field(
        default=2,
        gt=0,
        description="Number of LSTM layers"
    )

    # Random Forest parameters
    rf_n_estimators: int = Field(
        default=100,
        gt=0,
        description="Number of trees in Random Forest"
    )
    rf_max_depth: int = Field(
        default=10,
        gt=0,
        description="Maximum depth of Random Forest trees"
    )

    # Ensemble model weights
    ensemble_weights: Dict[str, float] = Field(
        default={"lstm": 0.6, "rf": 0.4},
        description="Weights for ensemble regime prediction"
    )

    # Risk management parameters
    max_position_size: float = Field(
        default=0.15,
        gt=0.0,
        le=1.0,
        description="Maximum position size as fraction of portfolio"
    )
    stop_loss: Optional[float] = Field(
        default=0.03,
        gt=0.0,
        description="Stop loss threshold"
    )
    take_profit: Optional[float] = Field(
        default=0.06,
        gt=0.0,
        description="Take profit threshold"
    )
    max_drawdown: float = Field(
        default=0.12,
        gt=0.0,
        le=1.0,
        description="Maximum allowed drawdown"
    )
    volatility_scalar: float = Field(
        default=2.0,
        gt=0.0,
        description="Volatility scaling factor for position sizing"
    )

    # Kelly Criterion parameters
    base_kelly_fraction: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description="Base Kelly fraction for position sizing"
    )
    max_kelly_fraction: float = Field(
        default=0.25,
        gt=0.0,
        le=1.0,
        description="Maximum Kelly fraction allowed"
    )

    # Regime-specific parameters
    regime_parameters: Dict[str, Dict[str, Union[float, int]]] = Field(
        default={
            "trending": {
                "zscore_threshold": 2.5,
                "signal_multiplier": 0.5,  # Reduce mean reversion in trending markets
                "signal_threshold": 0.03,
                "base_volatility": 0.02,
                "min_vol_adjustment": 0.3,
                "max_vol_adjustment": 2.0,
                "rsi_oversold": 25,
                "rsi_overbought": 75,
                "win_rate": 0.45,
                "avg_win_loss_ratio": 1.8
            },
            "ranging": {
                "zscore_threshold": 1.8,
                "signal_multiplier": 1.2,  # Enhance mean reversion in ranging markets
                "signal_threshold": 0.02,
                "base_volatility": 0.015,
                "min_vol_adjustment": 0.5,
                "max_vol_adjustment": 1.5,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "win_rate": 0.62,
                "avg_win_loss_ratio": 1.3
            },
            "volatile": {
                "zscore_threshold": 3.0,
                "signal_multiplier": 0.7,  # Reduce exposure in volatile markets
                "signal_threshold": 0.04,
                "base_volatility": 0.03,
                "min_vol_adjustment": 0.2,
                "max_vol_adjustment": 1.2,
                "rsi_oversold": 20,
                "rsi_overbought": 80,
                "win_rate": 0.52,
                "avg_win_loss_ratio": 1.5
            }
        },
        description="Regime-specific trading parameters"
    )

    # Data requirements
    min_data_points: int = Field(
        default=50,
        gt=0,
        description="Minimum data points required for signal generation"
    )

    # Backtesting parameters
    transaction_cost: float = Field(
        default=0.001,
        ge=0.0,
        description="Transaction cost per trade"
    )
    slippage: float = Field(
        default=0.0005,
        ge=0.0,
        description="Expected slippage"
    )

    # Performance targets
    target_sharpe_ratio: float = Field(
        default=1.8,
        gt=0.0,
        description="Target Sharpe ratio"
    )
    target_max_drawdown: float = Field(
        default=0.10,
        gt=0.0,
        le=1.0,
        description="Target maximum drawdown"
    )
    target_win_rate: float = Field(
        default=0.58,
        ge=0.0,
        le=1.0,
        description="Target win rate"
    )

    @validator('universe')
    def validate_universe(cls, v):
        """Validate trading universe."""
        if not v or len(v) == 0:
            raise ValueError("Universe cannot be empty")
        return v

    @validator('mean_reversion_windows')
    def validate_mr_windows(cls, v):
        """Validate mean reversion windows."""
        if not all(w > 0 for w in v):
            raise ValueError("All mean reversion windows must be positive")
        return sorted(set(v))  # Remove duplicates and sort

    @validator('volatility_windows')
    def validate_vol_windows(cls, v):
        """Validate volatility windows."""
        if not all(w > 0 for w in v):
            raise ValueError("All volatility windows must be positive")
        return sorted(set(v))  # Remove duplicates and sort

    @validator('ensemble_weights')
    def validate_ensemble_weights(cls, v):
        """Validate ensemble weights sum to 1."""
        total_weight = sum(v.values())
        if not (0.99 <= total_weight <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Ensemble weights must sum to 1.0, got {total_weight}")
        return v

    @validator('regime_parameters')
    def validate_regime_parameters(cls, v):
        """Validate regime parameters structure."""
        required_regimes = ['trending', 'ranging', 'volatile']
        required_params = [
            'zscore_threshold', 'signal_multiplier', 'signal_threshold',
            'base_volatility', 'min_vol_adjustment', 'max_vol_adjustment',
            'rsi_oversold', 'rsi_overbought', 'win_rate', 'avg_win_loss_ratio'
        ]

        for regime in required_regimes:
            if regime not in v:
                raise ValueError(f"Missing regime parameters for: {regime}")

            for param in required_params:
                if param not in v[regime]:
                    raise ValueError(f"Missing parameter '{param}' for regime '{regime}'")

        return v

    @validator('max_kelly_fraction')
    def validate_kelly_fractions(cls, v, values):
        """Validate Kelly fraction constraints."""
        base_kelly = values.get('base_kelly_fraction', 0.1)
        if v < base_kelly:
            raise ValueError("max_kelly_fraction must be >= base_kelly_fraction")
        return v

    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "forbid"
        json_schema_extra = {
            "example": {
                "name": "adaptive_mean_reversion",
                "universe": ["BTC/USDT", "ETH/USDT"],
                "mean_reversion_windows": [10, 20, 50],
                "regime_confidence_threshold": 0.6,
                "max_position_size": 0.15,
                "target_sharpe_ratio": 1.8
            }
        }