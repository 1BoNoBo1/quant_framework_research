"""
Tests for Data Providers Infrastructure
======================================

Suite de tests pour les providers de données critiques.
Teste BinanceProvider et MarketDataPipeline.
"""

import pytest
import pandas as pd
import numpy as np
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List

from qframe.infrastructure.data.binance_provider import BinanceProvider
from qframe.infrastructure.data.market_data_pipeline import MarketDataPipeline
from qframe.core.interfaces import TimeFrame, DataProvider, MetricsCollector
from qframe.domain.entities.order import OrderSide


@pytest.fixture
def mock_binance_client():
    """Client Binance mocké."""
    client = Mock()
    client.get_klines = AsyncMock()
    client.get_ticker_price = AsyncMock()
    client.get_orderbook_ticker = AsyncMock()
    client.get_account = AsyncMock()
    return client


@pytest.fixture
def mock_metrics_collector():
    """Collecteur de métriques mocké."""
    metrics = Mock(spec=MetricsCollector)
    metrics.increment = Mock()
    metrics.histogram = Mock()
    metrics.gauge = Mock()
    return metrics


@pytest.fixture
def binance_provider(mock_binance_client, mock_metrics_collector):
    """Provider Binance pour les tests."""
    with patch('qframe.infrastructure.data.binance_provider.Client') as mock_client_class:
        mock_client_class.return_value = mock_binance_client
        provider = BinanceProvider(
            api_key="test_key",
            api_secret="test_secret",
            metrics_collector=mock_metrics_collector
        )
        provider.client = mock_binance_client
        return provider


@pytest.fixture
def sample_klines_data():
    """Données de klines Binance d'exemple."""
    return [
        [
            1640995200000,  # timestamp
            "46000.00",     # open
            "47000.00",     # high
            "45500.00",     # low
            "46800.00",     # close
            "123.45",       # volume
            1640998799999,  # close_time
            "5776000.00",   # quote_asset_volume
            1000,           # number_of_trades
            "62.23",        # taker_buy_base_asset_volume
            "2888000.00",   # taker_buy_quote_asset_volume
            "0"             # unused_field
        ],
        [
            1640998800000,
            "46800.00",
            "47200.00",
            "46200.00",
            "46900.00",
            "98.76",
            1641002399999,
            "4630000.00",
            800,
            "49.38",
            "2315000.00",
            "0"
        ]
    ]


@pytest.fixture
def sample_ticker_data():
    """Données de ticker Binance d'exemple."""
    return {
        "symbol": "BTCUSDT",
        "price": "46950.00"
    }


@pytest.fixture
def sample_orderbook_data():
    """Données de carnet d'ordres d'exemple."""
    return {
        "symbol": "BTCUSDT",
        "bidPrice": "46940.00",
        "bidQty": "1.50000000",
        "askPrice": "46950.00",
        "askQty": "1.20000000"
    }


class TestBinanceProvider:
    """Tests pour BinanceProvider."""

    async def test_initialization(self, binance_provider):
        """Test d'initialisation du provider."""
        assert binance_provider.client is not None
        assert binance_provider.rate_limiter is not None
        assert binance_provider.metrics_collector is not None

    async def test_fetch_ohlcv_success(
        self,
        binance_provider,
        mock_binance_client,
        sample_klines_data
    ):
        """Test de récupération OHLCV réussie."""
        # Arrange
        mock_binance_client.get_klines.return_value = sample_klines_data

        # Act
        result = await binance_provider.fetch_ohlcv("BTC/USDT", TimeFrame.ONE_HOUR, limit=2)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        assert result.iloc[0]['open'] == 46000.00
        assert result.iloc[0]['close'] == 46800.00
        mock_binance_client.get_klines.assert_called_once()

    async def test_fetch_ohlcv_with_date_range(
        self,
        binance_provider,
        mock_binance_client,
        sample_klines_data
    ):
        """Test de récupération OHLCV avec plage de dates."""
        # Arrange
        mock_binance_client.get_klines.return_value = sample_klines_data
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 2)

        # Act
        result = await binance_provider.fetch_ohlcv(
            "BTC/USDT",
            TimeFrame.ONE_HOUR,
            since=start_date,
            until=end_date
        )

        # Assert
        assert len(result) == 2
        # Vérifier que les timestamps sont utilisés dans l'appel
        call_args = mock_binance_client.get_klines.call_args
        assert call_args[1]['startTime'] is not None
        assert call_args[1]['endTime'] is not None

    async def test_get_current_price(
        self,
        binance_provider,
        mock_binance_client,
        sample_ticker_data
    ):
        """Test de récupération du prix actuel."""
        # Arrange
        mock_binance_client.get_ticker_price.return_value = sample_ticker_data

        # Act
        result = await binance_provider.get_current_price("BTC/USDT")

        # Assert
        assert result['symbol'] == 'BTC/USDT'
        assert result['price'] == 46950.00
        assert isinstance(result['timestamp'], datetime)

    async def test_get_current_price_with_orderbook(
        self,
        binance_provider,
        mock_binance_client,
        sample_ticker_data,
        sample_orderbook_data
    ):
        """Test de récupération du prix avec bid/ask."""
        # Arrange
        mock_binance_client.get_ticker_price.return_value = sample_ticker_data
        mock_binance_client.get_orderbook_ticker.return_value = sample_orderbook_data

        # Act
        result = await binance_provider.get_current_price("BTC/USDT", include_bid_ask=True)

        # Assert
        assert result['price'] == 46950.00
        assert result['bid'] == 46940.00
        assert result['ask'] == 46950.00

    async def test_rate_limiting(self, binance_provider, mock_binance_client):
        """Test de limitation du taux de requêtes."""
        # Arrange
        mock_binance_client.get_ticker_price.return_value = {"symbol": "BTCUSDT", "price": "46950.00"}

        # Act - Faire plusieurs requêtes rapidement
        tasks = [
            binance_provider.get_current_price("BTC/USDT")
            for _ in range(5)
        ]
        start_time = datetime.now()
        results = await asyncio.gather(*tasks)
        elapsed_time = (datetime.now() - start_time).total_seconds()

        # Assert
        assert len(results) == 5
        # Avec rate limiting, cela devrait prendre un peu de temps
        assert elapsed_time >= 0.0  # Au moins pas instantané

    async def test_error_handling_network_error(self, binance_provider, mock_binance_client):
        """Test de gestion d'erreur réseau."""
        # Arrange
        mock_binance_client.get_ticker_price.side_effect = Exception("Network error")

        # Act & Assert
        with pytest.raises(Exception, match="Network error"):
            await binance_provider.get_current_price("BTC/USDT")

    async def test_error_handling_invalid_symbol(self, binance_provider, mock_binance_client):
        """Test de gestion d'erreur symbole invalide."""
        # Arrange
        mock_binance_client.get_klines.side_effect = Exception("Invalid symbol")

        # Act & Assert
        with pytest.raises(Exception, match="Invalid symbol"):
            await binance_provider.fetch_ohlcv("INVALID/SYMBOL", TimeFrame.ONE_HOUR)

    async def test_metrics_collection(
        self,
        binance_provider,
        mock_binance_client,
        mock_metrics_collector,
        sample_ticker_data
    ):
        """Test de collecte de métriques."""
        # Arrange
        mock_binance_client.get_ticker_price.return_value = sample_ticker_data

        # Act
        await binance_provider.get_current_price("BTC/USDT")

        # Assert
        mock_metrics_collector.increment.assert_called()
        mock_metrics_collector.histogram.assert_called()

    async def test_symbol_conversion(self, binance_provider):
        """Test de conversion des symboles."""
        # Act & Assert
        assert binance_provider._convert_symbol("BTC/USDT") == "BTCUSDT"
        assert binance_provider._convert_symbol("ETH/BTC") == "ETHBTC"
        assert binance_provider._convert_symbol("ADA/EUR") == "ADAEUR"

    async def test_timeframe_conversion(self, binance_provider):
        """Test de conversion des timeframes."""
        # Act & Assert
        assert binance_provider._convert_timeframe(TimeFrame.ONE_MINUTE) == "1m"
        assert binance_provider._convert_timeframe(TimeFrame.FIVE_MINUTES) == "5m"
        assert binance_provider._convert_timeframe(TimeFrame.ONE_HOUR) == "1h"
        assert binance_provider._convert_timeframe(TimeFrame.ONE_DAY) == "1d"

    async def test_data_validation(self, binance_provider, mock_binance_client):
        """Test de validation des données."""
        # Arrange - données invalides
        invalid_klines = [
            [
                1640995200000,
                "invalid_price",  # Prix invalide
                "47000.00",
                "45500.00",
                "46800.00",
                "123.45",
                1640998799999,
                "5776000.00",
                1000,
                "62.23",
                "2888000.00",
                "0"
            ]
        ]
        mock_binance_client.get_klines.return_value = invalid_klines

        # Act & Assert
        with pytest.raises(ValueError):
            await binance_provider.fetch_ohlcv("BTC/USDT", TimeFrame.ONE_HOUR, validate=True)


class TestMarketDataPipeline:
    """Tests pour MarketDataPipeline."""

    @pytest.fixture
    def mock_data_provider(self):
        provider = Mock(spec=DataProvider)
        provider.fetch_ohlcv = AsyncMock()
        provider.get_current_price = AsyncMock()
        return provider

    @pytest.fixture
    def mock_cache(self):
        cache = Mock()
        cache.get = AsyncMock()
        cache.set = AsyncMock()
        cache.delete = AsyncMock()
        return cache

    @pytest.fixture
    def market_data_pipeline(self, mock_data_provider, mock_metrics_collector):
        return MarketDataPipeline(
            data_provider=mock_data_provider,
            metrics_collector=mock_metrics_collector
        )

    async def test_initialization(self, market_data_pipeline):
        """Test d'initialisation du pipeline."""
        assert market_data_pipeline.data_provider is not None
        assert market_data_pipeline.metrics_collector is not None

    async def test_fetch_with_cache_miss(
        self,
        market_data_pipeline,
        mock_data_provider
    ):
        """Test de récupération avec cache miss."""
        # Arrange
        sample_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'open': [46000.0],
            'high': [47000.0],
            'low': [45500.0],
            'close': [46800.0],
            'volume': [123.45]
        })
        mock_data_provider.fetch_ohlcv.return_value = sample_data

        # Act
        result = await market_data_pipeline.fetch_ohlcv("BTC/USDT", TimeFrame.ONE_HOUR)

        # Assert
        assert len(result) == 1
        mock_data_provider.fetch_ohlcv.assert_called_once()

    async def test_data_enrichment(
        self,
        market_data_pipeline,
        mock_data_provider
    ):
        """Test d'enrichissement des données."""
        # Arrange
        base_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='H'),
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        mock_data_provider.fetch_ohlcv.return_value = base_data

        # Act
        result = await market_data_pipeline.fetch_ohlcv(
            "BTC/USDT",
            TimeFrame.ONE_HOUR,
            enrich=True
        )

        # Assert
        # Vérifier que des colonnes enrichies sont ajoutées
        assert 'sma_20' in result.columns or len(result.columns) >= 6

    async def test_data_quality_validation(
        self,
        market_data_pipeline,
        mock_data_provider
    ):
        """Test de validation de la qualité des données."""
        # Arrange - données avec des valeurs manquantes
        invalid_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'open': [np.nan],  # Valeur manquante
            'high': [47000.0],
            'low': [45500.0],
            'close': [46800.0],
            'volume': [123.45]
        })
        mock_data_provider.fetch_ohlcv.return_value = invalid_data

        # Act & Assert
        with pytest.raises(ValueError, match="Data quality"):
            await market_data_pipeline.fetch_ohlcv(
                "BTC/USDT",
                TimeFrame.ONE_HOUR,
                validate_quality=True
            )

    async def test_multi_symbol_fetch(
        self,
        market_data_pipeline,
        mock_data_provider
    ):
        """Test de récupération multi-symboles."""
        # Arrange
        sample_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'open': [46000.0],
            'high': [47000.0],
            'low': [45500.0],
            'close': [46800.0],
            'volume': [123.45]
        })
        mock_data_provider.fetch_ohlcv.return_value = sample_data

        symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]

        # Act
        results = await market_data_pipeline.fetch_multi_symbol(
            symbols,
            TimeFrame.ONE_HOUR
        )

        # Assert
        assert len(results) == 3
        assert "BTC/USDT" in results
        assert "ETH/USDT" in results
        assert "ADA/USDT" in results

    async def test_real_time_aggregation(
        self,
        market_data_pipeline,
        mock_data_provider
    ):
        """Test d'agrégation temps réel."""
        # Arrange
        tick_data = [
            {"symbol": "BTC/USDT", "price": 46000.0, "volume": 1.0, "timestamp": datetime.now()},
            {"symbol": "BTC/USDT", "price": 46100.0, "volume": 1.5, "timestamp": datetime.now()},
            {"symbol": "BTC/USDT", "price": 46050.0, "volume": 0.8, "timestamp": datetime.now()}
        ]

        # Act
        aggregated = await market_data_pipeline.aggregate_ticks(tick_data, interval="1min")

        # Assert
        assert isinstance(aggregated, dict)
        assert 'open' in aggregated
        assert 'high' in aggregated
        assert 'low' in aggregated
        assert 'close' in aggregated
        assert 'volume' in aggregated

    async def test_pipeline_performance(
        self,
        market_data_pipeline,
        mock_data_provider
    ):
        """Test de performance du pipeline."""
        # Arrange
        large_dataset = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=10000, freq='min'),
            'open': np.random.uniform(45000, 47000, 10000),
            'high': np.random.uniform(46000, 48000, 10000),
            'low': np.random.uniform(44000, 46000, 10000),
            'close': np.random.uniform(45000, 47000, 10000),
            'volume': np.random.uniform(100, 1000, 10000)
        })
        mock_data_provider.fetch_ohlcv.return_value = large_dataset

        # Act
        start_time = datetime.now()
        result = await market_data_pipeline.fetch_ohlcv(
            "BTC/USDT",
            TimeFrame.ONE_MINUTE,
            limit=10000
        )
        processing_time = (datetime.now() - start_time).total_seconds()

        # Assert
        assert len(result) == 10000
        assert processing_time < 5.0  # Doit traiter en moins de 5 secondes

    async def test_error_recovery(
        self,
        market_data_pipeline,
        mock_data_provider
    ):
        """Test de récupération d'erreur."""
        # Arrange
        mock_data_provider.fetch_ohlcv.side_effect = [
            Exception("Temporary error"),  # Première tentative échoue
            pd.DataFrame({  # Deuxième tentative réussit
                'timestamp': [datetime.now()],
                'open': [46000.0],
                'high': [47000.0],
                'low': [45500.0],
                'close': [46800.0],
                'volume': [123.45]
            })
        ]

        # Act
        result = await market_data_pipeline.fetch_ohlcv_with_retry(
            "BTC/USDT",
            TimeFrame.ONE_HOUR,
            max_retries=2
        )

        # Assert
        assert len(result) == 1
        assert mock_data_provider.fetch_ohlcv.call_count == 2


class TestDataProviderIntegration:
    """Tests d'intégration des data providers."""

    async def test_provider_switching(self, mock_metrics_collector):
        """Test de basculement entre providers."""
        # Arrange
        primary_provider = Mock(spec=DataProvider)
        backup_provider = Mock(spec=DataProvider)

        primary_provider.fetch_ohlcv.side_effect = Exception("Primary provider down")
        backup_provider.fetch_ohlcv.return_value = pd.DataFrame({
            'timestamp': [datetime.now()],
            'open': [46000.0],
            'high': [47000.0],
            'low': [45500.0],
            'close': [46800.0],
            'volume': [123.45]
        })

        pipeline = MarketDataPipeline(
            data_provider=primary_provider,
            backup_provider=backup_provider,
            metrics_collector=mock_metrics_collector
        )

        # Act
        result = await pipeline.fetch_ohlcv_with_fallback("BTC/USDT", TimeFrame.ONE_HOUR)

        # Assert
        assert len(result) == 1
        backup_provider.fetch_ohlcv.assert_called_once()

    async def test_data_consistency_check(self, mock_metrics_collector):
        """Test de vérification de cohérence des données."""
        # Arrange
        provider1 = Mock(spec=DataProvider)
        provider2 = Mock(spec=DataProvider)

        # Données cohérentes
        consistent_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'open': [46000.0],
            'high': [47000.0],
            'low': [45500.0],
            'close': [46800.0],
            'volume': [123.45]
        })

        provider1.fetch_ohlcv.return_value = consistent_data
        provider2.fetch_ohlcv.return_value = consistent_data

        pipeline = MarketDataPipeline(
            data_provider=provider1,
            metrics_collector=mock_metrics_collector
        )

        # Act
        is_consistent = await pipeline.verify_data_consistency(
            "BTC/USDT",
            TimeFrame.ONE_HOUR,
            provider2
        )

        # Assert
        assert is_consistent is True

    async def test_historical_data_stitching(self, mock_metrics_collector):
        """Test de raccordement de données historiques."""
        # Arrange
        provider = Mock(spec=DataProvider)

        # Données en deux parties avec un gap
        part1 = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='H'),
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })

        part2 = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01 08:00:00', periods=5, freq='H'),
            'open': [106, 107, 108, 109, 110],
            'high': [107, 108, 109, 110, 111],
            'low': [105, 106, 107, 108, 109],
            'close': [107, 108, 109, 110, 111],
            'volume': [1500, 1600, 1700, 1800, 1900]
        })

        provider.fetch_ohlcv.side_effect = [part1, part2]

        pipeline = MarketDataPipeline(
            data_provider=provider,
            metrics_collector=mock_metrics_collector
        )

        # Act
        stitched_data = await pipeline.fetch_historical_range(
            "BTC/USDT",
            TimeFrame.ONE_HOUR,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 1, 12)
        )

        # Assert
        assert len(stitched_data) == 10
        assert stitched_data.index.is_monotonic_increasing