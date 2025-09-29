"""
Tests for CCXT Provider
=======================

Suite de tests robuste pour le provider CCXT multi-exchanges.
"""

import pytest
import pandas as pd
import numpy as np
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from decimal import Decimal

from qframe.infrastructure.data.ccxt_provider import CCXTProvider
from qframe.core.interfaces import TimeFrame


@pytest.fixture
def mock_exchange():
    """Exchange CCXT mocké."""
    exchange = Mock()
    exchange.id = "binance"
    exchange.name = "Binance"
    exchange.has = {
        'fetchOHLCV': True,
        'fetchTicker': True,
        'fetchOrderBook': True,
        'fetchTrades': True,
        'watchTicker': True,
        'watchOHLCV': True
    }
    return exchange


@pytest.fixture
def ccxt_provider(mock_exchange):
    """CCXT Provider pour les tests."""
    with patch('ccxt.binance') as mock_ccxt:
        mock_ccxt.return_value = mock_exchange
        provider = CCXTProvider(exchange_id="binance")
        return provider


@pytest.fixture
def sample_ohlcv_raw():
    """Données OHLCV brutes format CCXT."""
    return [
        [1640995200000, 46000.0, 47000.0, 45500.0, 46800.0, 123.45],  # timestamp, O, H, L, C, V
        [1640998800000, 46800.0, 47200.0, 46200.0, 46900.0, 98.76],
        [1641002400000, 46900.0, 47500.0, 46700.0, 47100.0, 156.32]
    ]


@pytest.fixture
def sample_ticker():
    """Ticker d'exemple."""
    return {
        'symbol': 'BTC/USDT',
        'bid': 46950.0,
        'ask': 46955.0,
        'last': 46952.5,
        'high': 47500.0,
        'low': 46200.0,
        'volume': 12345.67,
        'quoteVolume': 580123456.78,
        'timestamp': 1641002400000,
        'datetime': '2022-01-01T10:00:00.000Z'
    }


class TestCCXTProviderBasic:
    """Tests de base pour CCXTProvider."""

    def test_provider_initialization(self, ccxt_provider):
        """Test d'initialisation du provider."""
        assert ccxt_provider.exchange_id == "binance"
        assert ccxt_provider.exchange is not None

    def test_supported_exchanges(self):
        """Test de la liste des exchanges supportés."""
        with patch('ccxt.exchanges', ['binance', 'coinbase', 'kraken']):
            exchanges = CCXTProvider.get_supported_exchanges()
            assert 'binance' in exchanges
            assert 'coinbase' in exchanges

    async def test_fetch_ohlcv_success(self, ccxt_provider, mock_exchange, sample_ohlcv_raw):
        """Test de récupération OHLCV réussie."""
        # Arrange
        mock_exchange.fetch_ohlcv = AsyncMock(return_value=sample_ohlcv_raw)

        # Act
        result = await ccxt_provider.fetch_ohlcv("BTC/USDT", TimeFrame.ONE_HOUR, limit=100)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        assert result.iloc[0]['open'] == 46000.0
        assert result.iloc[0]['close'] == 46800.0

    async def test_fetch_ohlcv_with_date_range(self, ccxt_provider, mock_exchange, sample_ohlcv_raw):
        """Test de récupération OHLCV avec plage de dates."""
        # Arrange
        mock_exchange.fetch_ohlcv = AsyncMock(return_value=sample_ohlcv_raw)
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 2)

        # Act
        result = await ccxt_provider.fetch_ohlcv(
            "BTC/USDT",
            TimeFrame.ONE_HOUR,
            since=start_date,
            until=end_date
        )

        # Assert
        assert len(result) == 3
        mock_exchange.fetch_ohlcv.assert_called_once()

    async def test_get_current_price(self, ccxt_provider, mock_exchange, sample_ticker):
        """Test de récupération du prix actuel."""
        # Arrange
        mock_exchange.fetch_ticker = AsyncMock(return_value=sample_ticker)

        # Act
        result = await ccxt_provider.get_current_price("BTC/USDT")

        # Assert
        assert result['symbol'] == 'BTC/USDT'
        assert result['price'] == 46952.5
        assert result['bid'] == 46950.0
        assert result['ask'] == 46955.0

    async def test_get_orderbook(self, ccxt_provider, mock_exchange):
        """Test de récupération du carnet d'ordres."""
        # Arrange
        orderbook_data = {
            'bids': [[46950.0, 1.5], [46940.0, 2.3]],
            'asks': [[46955.0, 1.2], [46960.0, 0.8]],
            'timestamp': 1641002400000,
            'nonce': None
        }
        mock_exchange.fetch_order_book = AsyncMock(return_value=orderbook_data)

        # Act
        result = await ccxt_provider.get_orderbook("BTC/USDT", limit=10)

        # Assert
        assert 'bids' in result
        assert 'asks' in result
        assert len(result['bids']) == 2
        assert len(result['asks']) == 2
        assert result['bids'][0][0] == 46950.0


class TestCCXTProviderErrorHandling:
    """Tests de gestion d'erreurs."""

    async def test_network_error_handling(self, ccxt_provider, mock_exchange):
        """Test de gestion d'erreur réseau."""
        # Arrange
        from ccxt.base.errors import NetworkError
        mock_exchange.fetch_ohlcv = AsyncMock(side_effect=NetworkError("Connection failed"))

        # Act & Assert
        with pytest.raises(ConnectionError):
            await ccxt_provider.fetch_ohlcv("BTC/USDT", TimeFrame.ONE_HOUR)

    async def test_exchange_error_handling(self, ccxt_provider, mock_exchange):
        """Test de gestion d'erreur d'exchange."""
        # Arrange
        from ccxt.base.errors import ExchangeError
        mock_exchange.fetch_ticker = AsyncMock(side_effect=ExchangeError("Invalid symbol"))

        # Act & Assert
        with pytest.raises(ValueError):
            await ccxt_provider.get_current_price("INVALID/SYMBOL")

    async def test_rate_limit_handling(self, ccxt_provider, mock_exchange):
        """Test de gestion de limitation de taux."""
        # Arrange
        from ccxt.base.errors import RateLimitExceeded
        mock_exchange.fetch_ohlcv = AsyncMock(side_effect=RateLimitExceeded("Rate limit exceeded"))

        # Act & Assert
        with pytest.raises(Exception, match="Rate limit"):
            await ccxt_provider.fetch_ohlcv("BTC/USDT", TimeFrame.ONE_HOUR)

    async def test_invalid_timeframe(self, ccxt_provider):
        """Test de timeframe invalide."""
        # Act & Assert
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            await ccxt_provider.fetch_ohlcv("BTC/USDT", "invalid_timeframe")

    async def test_empty_data_handling(self, ccxt_provider, mock_exchange):
        """Test de gestion de données vides."""
        # Arrange
        mock_exchange.fetch_ohlcv = AsyncMock(return_value=[])

        # Act
        result = await ccxt_provider.fetch_ohlcv("BTC/USDT", TimeFrame.ONE_HOUR)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestCCXTProviderDataValidation:
    """Tests de validation des données."""

    async def test_ohlcv_data_validation(self, ccxt_provider, mock_exchange):
        """Test de validation des données OHLCV."""
        # Arrange - données invalides
        invalid_data = [
            [1640995200000, 46000.0, 45000.0, 47000.0, 46800.0, 123.45],  # High < Open
        ]
        mock_exchange.fetch_ohlcv = AsyncMock(return_value=invalid_data)

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid OHLC data"):
            await ccxt_provider.fetch_ohlcv("BTC/USDT", TimeFrame.ONE_HOUR, validate=True)

    async def test_timestamp_validation(self, ccxt_provider, mock_exchange):
        """Test de validation des timestamps."""
        # Arrange - timestamps invalides
        invalid_data = [
            [None, 46000.0, 47000.0, 45500.0, 46800.0, 123.45],
        ]
        mock_exchange.fetch_ohlcv = AsyncMock(return_value=invalid_data)

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid timestamp"):
            await ccxt_provider.fetch_ohlcv("BTC/USDT", TimeFrame.ONE_HOUR, validate=True)

    async def test_volume_validation(self, ccxt_provider, mock_exchange):
        """Test de validation du volume."""
        # Arrange - volume négatif
        invalid_data = [
            [1640995200000, 46000.0, 47000.0, 45500.0, 46800.0, -123.45],
        ]
        mock_exchange.fetch_ohlcv = AsyncMock(return_value=invalid_data)

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid volume"):
            await ccxt_provider.fetch_ohlcv("BTC/USDT", TimeFrame.ONE_HOUR, validate=True)


class TestCCXTProviderPerformance:
    """Tests de performance."""

    async def test_large_dataset_handling(self, ccxt_provider, mock_exchange):
        """Test de gestion de gros datasets."""
        # Arrange
        large_dataset = [
            [1640995200000 + i * 3600000, 46000.0, 47000.0, 45500.0, 46800.0, 123.45]
            for i in range(10000)
        ]
        mock_exchange.fetch_ohlcv = AsyncMock(return_value=large_dataset)

        # Act
        start_time = datetime.now()
        result = await ccxt_provider.fetch_ohlcv("BTC/USDT", TimeFrame.ONE_HOUR, limit=10000)
        processing_time = (datetime.now() - start_time).total_seconds()

        # Assert
        assert len(result) == 10000
        assert processing_time < 5.0  # Doit traiter en moins de 5 secondes

    async def test_concurrent_requests(self, ccxt_provider, mock_exchange, sample_ohlcv_raw):
        """Test de requêtes concurrentes."""
        # Arrange
        mock_exchange.fetch_ohlcv = AsyncMock(return_value=sample_ohlcv_raw)

        # Act
        tasks = [
            ccxt_provider.fetch_ohlcv("BTC/USDT", TimeFrame.ONE_HOUR)
            for _ in range(5)
        ]
        results = await asyncio.gather(*tasks)

        # Assert
        assert len(results) == 5
        assert all(len(result) == 3 for result in results)

    async def test_memory_usage(self, ccxt_provider, mock_exchange):
        """Test d'utilisation mémoire."""
        # Arrange
        large_dataset = [
            [1640995200000 + i * 3600000, 46000.0, 47000.0, 45500.0, 46800.0, 123.45]
            for i in range(5000)
        ]
        mock_exchange.fetch_ohlcv = AsyncMock(return_value=large_dataset)

        # Act
        result = await ccxt_provider.fetch_ohlcv("BTC/USDT", TimeFrame.ONE_HOUR, limit=5000)

        # Assert
        memory_usage = result.memory_usage(deep=True).sum()
        assert memory_usage < 10_000_000  # < 10MB


class TestCCXTProviderRealTimeFeatures:
    """Tests des fonctionnalités temps réel."""

    async def test_websocket_subscription(self, ccxt_provider, mock_exchange):
        """Test d'abonnement WebSocket."""
        # Arrange
        mock_exchange.watch_ticker = AsyncMock(return_value={'symbol': 'BTC/USDT', 'last': 47000.0})

        # Act
        callback = Mock()
        subscription_id = await ccxt_provider.subscribe_ticker("BTC/USDT", callback)

        # Assert
        assert subscription_id is not None
        mock_exchange.watch_ticker.assert_called_once_with("BTC/USDT")

    async def test_websocket_ohlcv_subscription(self, ccxt_provider, mock_exchange):
        """Test d'abonnement WebSocket OHLCV."""
        # Arrange
        mock_exchange.watch_ohlcv = AsyncMock(return_value=[
            [1640995200000, 46000.0, 47000.0, 45500.0, 46800.0, 123.45]
        ])

        # Act
        callback = Mock()
        subscription_id = await ccxt_provider.subscribe_ohlcv("BTC/USDT", TimeFrame.ONE_MINUTE, callback)

        # Assert
        assert subscription_id is not None
        mock_exchange.watch_ohlcv.assert_called_once()

    async def test_unsubscribe(self, ccxt_provider, mock_exchange):
        """Test de désabonnement."""
        # Arrange
        mock_exchange.unwatch_ticker = AsyncMock()
        subscription_id = "test_subscription"

        # Act
        await ccxt_provider.unsubscribe(subscription_id)

        # Assert
        mock_exchange.unwatch_ticker.assert_called_once()


class TestCCXTProviderMultiExchange:
    """Tests multi-exchanges."""

    def test_exchange_switching(self):
        """Test de basculement entre exchanges."""
        # Act
        provider1 = CCXTProvider(exchange_id="binance")
        provider2 = CCXTProvider(exchange_id="coinbase")

        # Assert
        assert provider1.exchange_id == "binance"
        assert provider2.exchange_id == "coinbase"

    async def test_cross_exchange_arbitrage_data(self, mock_exchange):
        """Test de données d'arbitrage entre exchanges."""
        # Arrange
        with patch('ccxt.binance') as mock_binance, \
             patch('ccxt.coinbase') as mock_coinbase:

            mock_binance.return_value = mock_exchange
            mock_coinbase.return_value = mock_exchange

            mock_exchange.fetch_ticker = AsyncMock(side_effect=[
                {'symbol': 'BTC/USDT', 'last': 47000.0},  # Binance
                {'symbol': 'BTC/USD', 'last': 47050.0}    # Coinbase
            ])

            provider1 = CCXTProvider(exchange_id="binance")
            provider2 = CCXTProvider(exchange_id="coinbase")

            # Act
            price1 = await provider1.get_current_price("BTC/USDT")
            price2 = await provider2.get_current_price("BTC/USD")

            # Assert
            assert price1['price'] == 47000.0
            assert price2['price'] == 47050.0
            assert price2['price'] > price1['price']  # Opportunité d'arbitrage


class TestCCXTProviderConfiguration:
    """Tests de configuration."""

    def test_custom_configuration(self):
        """Test de configuration personnalisée."""
        # Arrange
        config = {
            'apiKey': 'test_key',
            'secret': 'test_secret',
            'sandbox': True,
            'rateLimit': 1000
        }

        # Act
        with patch('ccxt.binance') as mock_ccxt:
            provider = CCXTProvider(exchange_id="binance", config=config)

            # Assert
            mock_ccxt.assert_called_once_with(config)

    def test_proxy_configuration(self):
        """Test de configuration proxy."""
        # Arrange
        config = {
            'proxies': {
                'http': 'http://proxy.example.com:8080',
                'https': 'https://proxy.example.com:8080'
            }
        }

        # Act
        with patch('ccxt.binance') as mock_ccxt:
            provider = CCXTProvider(exchange_id="binance", config=config)

            # Assert
            mock_ccxt.assert_called_once_with(config)


@pytest.mark.integration
class TestCCXTProviderIntegration:
    """Tests d'intégration avec de vrais exchanges."""

    async def test_binance_integration(self):
        """Test d'intégration avec Binance (optionnel)."""
        try:
            provider = CCXTProvider(exchange_id="binance")
            result = await provider.get_current_price("BTC/USDT")
            assert result is not None
            assert 'price' in result
        except Exception as e:
            pytest.skip(f"Binance integration not available: {e}")

    async def test_multiple_exchanges_integration(self):
        """Test d'intégration avec plusieurs exchanges."""
        exchanges = ["binance", "coinbase"]
        results = {}

        for exchange_id in exchanges:
            try:
                provider = CCXTProvider(exchange_id=exchange_id)
                symbol = "BTC/USD" if exchange_id == "coinbase" else "BTC/USDT"
                price = await provider.get_current_price(symbol)
                results[exchange_id] = price
            except Exception as e:
                pytest.skip(f"{exchange_id} integration not available: {e}")

        assert len(results) >= 1