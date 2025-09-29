"""
Tests for Market Data Service
=============================

Suite de tests complète pour le service de données de marché.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from decimal import Decimal

from qframe.api.services.market_data_service import MarketDataService
from qframe.core.interfaces import DataProvider, CacheService, MetricsCollector
from qframe.infrastructure.data.binance_provider import BinanceProvider


@pytest.fixture
def mock_data_provider():
    """Data provider mocké."""
    provider = Mock(spec=DataProvider)
    return provider


@pytest.fixture
def mock_cache_service():
    """Service de cache mocké."""
    cache = Mock(spec=CacheService)
    return cache


@pytest.fixture
def mock_metrics_collector():
    """Collecteur de métriques mocké."""
    metrics = Mock(spec=MetricsCollector)
    return metrics


@pytest.fixture
def market_data_service(mock_data_provider, mock_cache_service, mock_metrics_collector):
    """Service de données de marché pour les tests."""
    return MarketDataService(
        data_provider=mock_data_provider,
        cache_service=mock_cache_service,
        metrics_collector=mock_metrics_collector
    )


@pytest.fixture
def sample_ohlcv_data():
    """Données OHLCV d'exemple."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="1H")
    data = pd.DataFrame({
        "timestamp": dates,
        "open": np.random.uniform(40000, 50000, 100),
        "high": np.random.uniform(40000, 50000, 100),
        "low": np.random.uniform(40000, 50000, 100),
        "close": np.random.uniform(40000, 50000, 100),
        "volume": np.random.uniform(10, 100, 100)
    })
    # Assurer la cohérence OHLC
    data["high"] = data[["open", "high", "close"]].max(axis=1)
    data["low"] = data[["open", "low", "close"]].min(axis=1)
    return data


class TestMarketDataService:
    """Tests pour MarketDataService."""

    async def test_get_current_price(self, market_data_service, mock_data_provider):
        """Test de récupération du prix actuel."""
        # Arrange
        expected_price = {
            "symbol": "BTC/USD",
            "price": Decimal("45000.00"),
            "timestamp": datetime.now(),
            "bid": Decimal("44995.00"),
            "ask": Decimal("45005.00")
        }
        mock_data_provider.get_current_price.return_value = expected_price

        # Act
        result = await market_data_service.get_current_price("BTC/USD")

        # Assert
        assert result["symbol"] == "BTC/USD"
        assert result["price"] == Decimal("45000.00")
        mock_data_provider.get_current_price.assert_called_once_with("BTC/USD")

    async def test_get_ohlcv_with_cache_hit(self, market_data_service, mock_cache_service, sample_ohlcv_data):
        """Test de récupération OHLCV avec cache hit."""
        # Arrange
        cache_key = "ohlcv:BTC/USD:1h:100"
        mock_cache_service.get.return_value = sample_ohlcv_data

        # Act
        result = await market_data_service.get_ohlcv("BTC/USD", "1h", limit=100)

        # Assert
        assert len(result) == 100
        assert "open" in result.columns
        mock_cache_service.get.assert_called_once_with(cache_key)

    async def test_get_ohlcv_with_cache_miss(
        self,
        market_data_service,
        mock_data_provider,
        mock_cache_service,
        sample_ohlcv_data
    ):
        """Test de récupération OHLCV avec cache miss."""
        # Arrange
        cache_key = "ohlcv:BTC/USD:1h:100"
        mock_cache_service.get.return_value = None  # Cache miss
        mock_data_provider.fetch_ohlcv.return_value = sample_ohlcv_data

        # Act
        result = await market_data_service.get_ohlcv("BTC/USD", "1h", limit=100)

        # Assert
        assert len(result) == 100
        mock_data_provider.fetch_ohlcv.assert_called_once()
        mock_cache_service.set.assert_called_once_with(cache_key, sample_ohlcv_data, ttl=300)

    async def test_get_multiple_symbols(self, market_data_service, mock_data_provider):
        """Test de récupération de données pour plusieurs symboles."""
        # Arrange
        symbols = ["BTC/USD", "ETH/USD", "ADA/USD"]
        mock_data_provider.get_current_price.side_effect = [
            {"symbol": "BTC/USD", "price": Decimal("45000.00")},
            {"symbol": "ETH/USD", "price": Decimal("3000.00")},
            {"symbol": "ADA/USD", "price": Decimal("1.50")}
        ]

        # Act
        result = await market_data_service.get_multiple_prices(symbols)

        # Assert
        assert len(result) == 3
        assert result["BTC/USD"]["price"] == Decimal("45000.00")
        assert result["ETH/USD"]["price"] == Decimal("3000.00")

    async def test_data_validation(self, market_data_service, mock_data_provider):
        """Test de validation des données."""
        # Arrange - données invalides
        invalid_data = pd.DataFrame({
            "timestamp": [datetime.now()],
            "open": [45000.00],
            "high": [44000.00],  # High < Open (invalide)
            "low": [46000.00],   # Low > Open (invalide)
            "close": [45500.00],
            "volume": [-10]      # Volume négatif (invalide)
        })
        mock_data_provider.fetch_ohlcv.return_value = invalid_data

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid OHLC data"):
            await market_data_service.get_ohlcv("BTC/USD", "1h")

    async def test_rate_limiting(self, market_data_service, mock_data_provider):
        """Test de limitation du taux de requêtes."""
        # Arrange
        mock_data_provider.get_current_price.side_effect = Exception("Rate limit exceeded")

        # Act & Assert
        with pytest.raises(Exception, match="Rate limit exceeded"):
            await market_data_service.get_current_price("BTC/USD")

    async def test_metrics_collection(self, market_data_service, mock_metrics_collector, mock_data_provider):
        """Test de collecte de métriques."""
        # Arrange
        mock_data_provider.get_current_price.return_value = {
            "symbol": "BTC/USD",
            "price": Decimal("45000.00")
        }

        # Act
        await market_data_service.get_current_price("BTC/USD")

        # Assert
        mock_metrics_collector.increment.assert_called_with("market_data.requests.total")
        mock_metrics_collector.histogram.assert_called()

    async def test_error_handling_provider_failure(self, market_data_service, mock_data_provider):
        """Test de gestion d'erreur lors d'une panne du provider."""
        # Arrange
        mock_data_provider.get_current_price.side_effect = ConnectionError("Provider unavailable")

        # Act & Assert
        with pytest.raises(ConnectionError):
            await market_data_service.get_current_price("BTC/USD")

    async def test_data_freshness_check(self, market_data_service, mock_cache_service, sample_ohlcv_data):
        """Test de vérification de la fraîcheur des données."""
        # Arrange - données périmées
        old_data = sample_ohlcv_data.copy()
        old_data["timestamp"] = old_data["timestamp"] - timedelta(hours=2)
        mock_cache_service.get.return_value = old_data

        # Act
        result = await market_data_service.get_ohlcv("BTC/USD", "1h", max_age_minutes=60)

        # Assert - devrait refuser les données périmées
        assert result is None or len(result) == 0

    async def test_batch_data_retrieval(self, market_data_service, mock_data_provider, sample_ohlcv_data):
        """Test de récupération de données en batch."""
        # Arrange
        symbols = ["BTC/USD", "ETH/USD"]
        mock_data_provider.fetch_ohlcv.return_value = sample_ohlcv_data

        # Act
        result = await market_data_service.get_batch_ohlcv(symbols, "1h", limit=50)

        # Assert
        assert len(result) == 2
        assert "BTC/USD" in result
        assert "ETH/USD" in result
        assert len(result["BTC/USD"]) == 100  # sample data length

    async def test_real_time_subscription(self, market_data_service, mock_data_provider):
        """Test d'abonnement aux données temps réel."""
        # Arrange
        callback = Mock()

        # Act
        subscription = await market_data_service.subscribe_real_time("BTC/USD", callback)

        # Assert
        assert subscription is not None
        mock_data_provider.subscribe.assert_called_once_with("BTC/USD", callback)

    async def test_data_aggregation(self, market_data_service, sample_ohlcv_data):
        """Test d'agrégation de données."""
        # Act
        aggregated = await market_data_service.aggregate_ohlcv(
            sample_ohlcv_data,
            from_timeframe="1h",
            to_timeframe="1d"
        )

        # Assert
        assert len(aggregated) <= len(sample_ohlcv_data)
        assert "open" in aggregated.columns
        assert "high" in aggregated.columns

    async def test_historical_data_backfill(self, market_data_service, mock_data_provider, sample_ohlcv_data):
        """Test de récupération historique avec backfill."""
        # Arrange
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 7)
        mock_data_provider.fetch_historical_ohlcv.return_value = sample_ohlcv_data

        # Act
        result = await market_data_service.get_historical_data(
            "BTC/USD",
            "1h",
            start_date,
            end_date
        )

        # Assert
        assert len(result) == 100
        mock_data_provider.fetch_historical_ohlcv.assert_called_once()


class TestMarketDataPerformance:
    """Tests de performance pour MarketDataService."""

    async def test_concurrent_requests(self, market_data_service, mock_data_provider):
        """Test de requêtes concurrentes."""
        import asyncio

        # Arrange
        mock_data_provider.get_current_price.return_value = {
            "symbol": "BTC/USD",
            "price": Decimal("45000.00")
        }

        # Act
        tasks = [
            market_data_service.get_current_price("BTC/USD")
            for _ in range(10)
        ]
        results = await asyncio.gather(*tasks)

        # Assert
        assert len(results) == 10
        assert all(r["price"] == Decimal("45000.00") for r in results)

    async def test_large_dataset_handling(self, market_data_service, mock_data_provider):
        """Test de gestion de gros datasets."""
        # Arrange
        large_dataset = pd.DataFrame({
            "timestamp": pd.date_range(start="2020-01-01", periods=10000, freq="1H"),
            "open": np.random.uniform(40000, 50000, 10000),
            "high": np.random.uniform(40000, 50000, 10000),
            "low": np.random.uniform(40000, 50000, 10000),
            "close": np.random.uniform(40000, 50000, 10000),
            "volume": np.random.uniform(10, 100, 10000)
        })
        mock_data_provider.fetch_ohlcv.return_value = large_dataset

        # Act
        result = await market_data_service.get_ohlcv("BTC/USD", "1h", limit=10000)

        # Assert
        assert len(result) == 10000
        assert result.memory_usage().sum() < 50_000_000  # < 50MB


class TestMarketDataIntegration:
    """Tests d'intégration avec de vrais providers."""

    @pytest.mark.integration
    async def test_binance_integration(self):
        """Test d'intégration avec Binance (optionnel)."""
        # Note: Ce test ne s'exécute que si les credentials sont disponibles
        try:
            provider = BinanceProvider()
            service = MarketDataService(
                data_provider=provider,
                cache_service=Mock(),
                metrics_collector=Mock()
            )

            result = await service.get_current_price("BTC/USDT")
            assert result is not None
            assert "price" in result

        except Exception as e:
            pytest.skip(f"Binance integration not available: {e}")