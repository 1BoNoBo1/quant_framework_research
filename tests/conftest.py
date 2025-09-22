"""
Configuration des tests avec fixtures partagées
==============================================

Fixtures pytest pour les tests unitaires et d'intégration.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

from qframe.core.container import DIContainer, MockContainer
from qframe.core.config import TestingConfig
from qframe.core.interfaces import MetricsCollector


@pytest.fixture
def test_config():
    """Configuration de test"""
    return TestingConfig()


@pytest.fixture
def mock_container():
    """Container DI mocké pour les tests"""
    return MockContainer()


@pytest.fixture
def sample_ohlcv_data():
    """Données OHLCV de test"""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="1h")

    # Simulation d'un random walk pour les prix
    base_price = 50000
    returns = np.random.normal(0, 0.02, 100)
    prices = base_price * np.cumprod(1 + returns)

    data = pd.DataFrame({
        "timestamp": dates,
        "open": prices * (1 + np.random.normal(0, 0.001, 100)),
        "high": prices * (1 + np.random.uniform(0, 0.02, 100)),
        "low": prices * (1 - np.random.uniform(0, 0.02, 100)),
        "close": prices,
        "volume": np.random.randint(1000, 10000, 100),
        "symbol": "BTCUSDT"
    }, index=dates)

    # Ajouter VWAP et returns
    data["vwap"] = (data["high"] + data["low"] + data["close"]) / 3
    data["returns"] = data["close"].pct_change().fillna(0)

    return data


@pytest.fixture
def sample_features_data(sample_ohlcv_data):
    """Features de test basées sur les données OHLCV"""
    data = sample_ohlcv_data.copy()

    # Ajouter quelques features techniques simples
    data["sma_10"] = data["close"].rolling(10).mean()
    data["sma_20"] = data["close"].rolling(20).mean()
    data["rsi_14"] = 50 + np.random.normal(0, 15, len(data))  # RSI simulé
    data["volume_sma"] = data["volume"].rolling(10).mean()

    return data


@pytest.fixture
def mock_metrics_collector():
    """Mock MetricsCollector pour les tests"""

    class MockMetricsCollector:
        def __init__(self):
            self.metrics = []
            self.counters = {}
            self.histograms = []

        def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
            self.metrics.append({
                "name": name,
                "value": value,
                "tags": tags or {},
                "timestamp": datetime.now()
            })

        def increment_counter(self, name: str, tags: Dict[str, str] = None):
            key = f"{name}:{tags or {}}"
            self.counters[key] = self.counters.get(key, 0) + 1

        def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
            self.histograms.append({
                "name": name,
                "value": value,
                "tags": tags or {},
                "timestamp": datetime.now()
            })

    return MockMetricsCollector()


@pytest.fixture
def temp_data_dir(tmp_path):
    """Répertoire temporaire pour les données de test"""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def strategy_configs():
    """Configurations de stratégies pour les tests"""
    return {
        "dmn_lstm": {
            "window_size": 32,
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.1,
            "signal_threshold": 0.1,
            "position_size": 0.01
        },
        "mean_reversion": {
            "lookback_short": 5,
            "lookback_long": 20,
            "z_entry_base": 1.0,
            "z_exit_base": 0.2,
            "position_size": 0.01
        },
        "rl_alpha": {
            "episodes_per_batch": 3,
            "max_complexity": 5,
            "signal_threshold": 0.05,
            "position_size": 0.01
        }
    }


@pytest.fixture
def mock_data_provider():
    """Mock DataProvider pour les tests"""

    class MockDataProvider:
        def __init__(self, data: pd.DataFrame):
            self.data = data

        async def fetch_ohlcv(self, symbol, timeframe, limit=1000, start_time=None, end_time=None):
            return self.data.copy()

        async def fetch_latest_price(self, symbol):
            return float(self.data["close"].iloc[-1])

        async def get_available_symbols(self):
            return ["BTCUSDT", "ETHUSDT", "ADAUSDT"]

    return MockDataProvider


@pytest.fixture(scope="session")
def test_database():
    """Base de données de test en mémoire"""
    # Pour les tests nécessitant une vraie base de données
    # On utiliserait SQLite en mémoire
    pass


@pytest.fixture
def sample_signals():
    """Signaux de test"""
    from qframe.core.interfaces import Signal, SignalAction

    return [
        Signal(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            action=SignalAction.BUY,
            strength=0.8,
            price=50000.0,
            size=0.02,
            metadata={"strategy": "test_strategy"}
        ),
        Signal(
            timestamp=datetime.now() + timedelta(hours=1),
            symbol="BTCUSDT",
            action=SignalAction.SELL,
            strength=0.6,
            price=51000.0,
            size=0.02,
            metadata={"strategy": "test_strategy"}
        )
    ]


@pytest.fixture
def sample_positions():
    """Positions de test"""
    from qframe.core.interfaces import Position

    return [
        Position(
            symbol="BTCUSDT",
            size=0.02,
            entry_price=50000.0,
            current_price=51000.0,
            timestamp=datetime.now(),
            unrealized_pnl=20.0,
            metadata={"strategy": "test_strategy"}
        )
    ]


# Markers pour les tests
def pytest_configure(config):
    """Configuration des markers pytest"""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )


# Fixtures paramétrées pour différents timeframes
@pytest.fixture(params=["1h", "4h", "1d"])
def timeframe(request):
    """Différents timeframes pour les tests"""
    return request.param


@pytest.fixture(params=["BTCUSDT", "ETHUSDT"])
def symbol(request):
    """Différents symboles pour les tests"""
    return request.param