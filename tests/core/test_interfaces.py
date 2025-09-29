"""
Tests for Core Interfaces
=========================

Tests rapides pour les interfaces Protocol du framework.
"""

import pytest
from datetime import datetime
from decimal import Decimal
from typing import List
import pandas as pd

from qframe.core.interfaces import (
    DataProvider, Strategy, FeatureProcessor,
    MetricsCollector, TimeFrame
)
from qframe.domain.value_objects.signal import Signal, SignalAction


class MockDataProvider:
    """Implémentation mock de DataProvider."""

    async def fetch_ohlcv(self, symbol: str, timeframe: TimeFrame, limit: int = 100) -> pd.DataFrame:
        return pd.DataFrame({
            'timestamp': [datetime.now()],
            'open': [100.0],
            'high': [101.0],
            'low': [99.0],
            'close': [100.5],
            'volume': [1000.0]
        })

    async def get_current_price(self, symbol: str) -> dict:
        return {
            'symbol': symbol,
            'price': 100.5,
            'timestamp': datetime.now()
        }


class MockStrategy:
    """Implémentation mock de Strategy."""

    def generate_signals(self, data: pd.DataFrame, features: pd.DataFrame = None) -> List[Signal]:
        return [
            Signal(
                symbol="BTC/USD",
                action=SignalAction.BUY,
                timestamp=datetime.now(),
                strength=Decimal("0.8")
            )
        ]


class MockFeatureProcessor:
    """Implémentation mock de FeatureProcessor."""

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.copy()

    def get_feature_names(self) -> List[str]:
        return ['sma_20', 'rsi', 'bollinger_bands']


class MockMetricsCollector:
    """Implémentation mock de MetricsCollector."""

    def __init__(self):
        self.metrics = {}

    def increment(self, name: str, value: int = 1, tags: dict = None):
        self.metrics[name] = self.metrics.get(name, 0) + value

    def gauge(self, name: str, value: float, tags: dict = None):
        self.metrics[name] = value

    def histogram(self, name: str, value: float, tags: dict = None):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)


class TestInterfaces:
    """Tests des interfaces Protocol."""

    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'timestamp': [datetime.now()],
            'open': [100.0],
            'high': [101.0],
            'low': [99.0],
            'close': [100.5],
            'volume': [1000.0]
        })

    def test_data_provider_interface(self, sample_data):
        """Test de l'interface DataProvider."""
        provider = MockDataProvider()

        # Test typing
        assert hasattr(provider, 'fetch_ohlcv')
        assert hasattr(provider, 'get_current_price')

    def test_strategy_interface(self, sample_data):
        """Test de l'interface Strategy."""
        strategy = MockStrategy()

        signals = strategy.generate_signals(sample_data)

        assert isinstance(signals, list)
        assert len(signals) > 0
        assert isinstance(signals[0], Signal)

    def test_feature_processor_interface(self, sample_data):
        """Test de l'interface FeatureProcessor."""
        processor = MockFeatureProcessor()

        features = processor.process(sample_data)
        feature_names = processor.get_feature_names()

        assert isinstance(features, pd.DataFrame)
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0

    def test_metrics_collector_interface(self):
        """Test de l'interface MetricsCollector."""
        collector = MockMetricsCollector()

        collector.increment('orders.created')
        collector.gauge('portfolio.value', 100000.0)
        collector.histogram('latency', 50.0)

        assert collector.metrics['orders.created'] == 1
        assert collector.metrics['portfolio.value'] == 100000.0
        assert len(collector.metrics['latency']) == 1

    def test_timeframe_enum(self):
        """Test de l'enum TimeFrame."""
        assert TimeFrame.ONE_MINUTE
        assert TimeFrame.FIVE_MINUTES
        assert TimeFrame.ONE_HOUR
        assert TimeFrame.ONE_DAY