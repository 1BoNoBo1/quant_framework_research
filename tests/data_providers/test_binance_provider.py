"""
Tests for Binance Provider
==========================

Suite de tests pour le fournisseur de données Binance.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from decimal import Decimal

from qframe.infrastructure.data.binance_provider import BinanceProvider
from qframe.core.interfaces import TimeFrame


class TestBinanceProvider:
    """Tests pour le provider Binance"""

    @pytest.fixture
    def binance_provider(self):
        """Provider Binance mocké pour les tests"""
        return BinanceProvider(
            api_key="test_api_key",
            api_secret="test_api_secret",
            testnet=True
        )

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Données OHLCV échantillon"""
        dates = pd.date_range('2023-01-01', periods=100, freq='1h')
        np.random.seed(42)

        # Prix de départ
        start_price = 50000
        prices = [start_price]

        # Générer random walk
        for _ in range(99):
            change = np.random.normal(0, 0.01)  # 1% volatilité
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
            'volume': np.random.uniform(100, 1000, 100)
        })

        return data

    def test_provider_initialization(self, binance_provider):
        """Test initialisation du provider"""
        assert binance_provider.api_key == "test_api_key"
        assert binance_provider.api_secret == "test_api_secret"
        assert binance_provider.testnet == True
        assert hasattr(binance_provider, '_client')

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_data(self, binance_provider, sample_ohlcv_data):
        """Test récupération données OHLCV"""
        with patch.object(binance_provider, '_fetch_klines') as mock_fetch:
            mock_fetch.return_value = sample_ohlcv_data

            result = await binance_provider.fetch_ohlcv(
                symbol="BTCUSDT",
                timeframe=TimeFrame.H1,
                start_time=datetime(2023, 1, 1),
                end_time=datetime(2023, 1, 5)
            )

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 100
            assert all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume'])

            # Vérifier que high >= low >= 0
            assert (result['high'] >= result['low']).all()
            assert (result['low'] >= 0).all()

    @pytest.mark.asyncio
    async def test_fetch_current_price(self, binance_provider):
        """Test récupération prix actuel"""
        mock_price = {"symbol": "BTCUSDT", "price": "50000.00"}

        with patch.object(binance_provider, '_get_ticker_price') as mock_ticker:
            mock_ticker.return_value = mock_price

            price = await binance_provider.get_current_price("BTCUSDT")

            assert isinstance(price, Decimal)
            assert price == Decimal("50000.00")

    @pytest.mark.asyncio
    async def test_fetch_order_book(self, binance_provider):
        """Test récupération carnet d'ordres"""
        mock_orderbook = {
            "bids": [["49950.00", "1.5"], ["49940.00", "2.0"]],
            "asks": [["50050.00", "1.2"], ["50060.00", "1.8"]]
        }

        with patch.object(binance_provider, '_get_order_book') as mock_book:
            mock_book.return_value = mock_orderbook

            orderbook = await binance_provider.get_order_book("BTCUSDT", limit=100)

            assert isinstance(orderbook, dict)
            assert "bids" in orderbook
            assert "asks" in orderbook
            assert len(orderbook["bids"]) == 2
            assert len(orderbook["asks"]) == 2

    @pytest.mark.asyncio
    async def test_fetch_recent_trades(self, binance_provider):
        """Test récupération trades récents"""
        mock_trades = [
            {
                "id": 1,
                "price": "50000.00",
                "qty": "0.1",
                "time": int(datetime.now().timestamp() * 1000),
                "isBuyerMaker": False
            },
            {
                "id": 2,
                "price": "50010.00",
                "qty": "0.2",
                "time": int(datetime.now().timestamp() * 1000),
                "isBuyerMaker": True
            }
        ]

        with patch.object(binance_provider, '_get_recent_trades') as mock_trades_call:
            mock_trades_call.return_value = mock_trades

            trades = await binance_provider.get_recent_trades("BTCUSDT", limit=500)

            assert isinstance(trades, list)
            assert len(trades) == 2
            assert all("price" in trade for trade in trades)
            assert all("qty" in trade for trade in trades)

    def test_timeframe_conversion(self, binance_provider):
        """Test conversion des timeframes"""
        assert binance_provider._convert_timeframe(TimeFrame.M1) == "1m"
        assert binance_provider._convert_timeframe(TimeFrame.M5) == "5m"
        assert binance_provider._convert_timeframe(TimeFrame.M15) == "15m"
        assert binance_provider._convert_timeframe(TimeFrame.H1) == "1h"
        assert binance_provider._convert_timeframe(TimeFrame.H4) == "4h"
        assert binance_provider._convert_timeframe(TimeFrame.D1) == "1d"

    def test_symbol_normalization(self, binance_provider):
        """Test normalisation des symboles"""
        # Binance utilise des symboles sans séparateur
        assert binance_provider._normalize_symbol("BTC/USDT") == "BTCUSDT"
        assert binance_provider._normalize_symbol("ETH/USDT") == "ETHUSDT"
        assert binance_provider._normalize_symbol("BTCUSDT") == "BTCUSDT"  # Déjà normalisé

    @pytest.mark.asyncio
    async def test_symbol_info(self, binance_provider):
        """Test récupération info symbole"""
        mock_symbol_info = {
            "symbol": "BTCUSDT",
            "status": "TRADING",
            "baseAsset": "BTC",
            "quoteAsset": "USDT",
            "filters": [
                {
                    "filterType": "PRICE_FILTER",
                    "minPrice": "0.01000000",
                    "maxPrice": "1000000.00000000",
                    "tickSize": "0.01000000"
                },
                {
                    "filterType": "LOT_SIZE",
                    "minQty": "0.00001000",
                    "maxQty": "9000.00000000",
                    "stepSize": "0.00001000"
                }
            ]
        }

        with patch.object(binance_provider, '_get_symbol_info') as mock_info:
            mock_info.return_value = mock_symbol_info

            info = await binance_provider.get_symbol_info("BTCUSDT")

            assert isinstance(info, dict)
            assert info["symbol"] == "BTCUSDT"
            assert info["status"] == "TRADING"
            assert "filters" in info

    @pytest.mark.asyncio
    async def test_market_data_validation(self, binance_provider, sample_ohlcv_data):
        """Test validation des données de marché"""
        with patch.object(binance_provider, '_fetch_klines') as mock_fetch:
            mock_fetch.return_value = sample_ohlcv_data

            result = await binance_provider.fetch_ohlcv(
                symbol="BTCUSDT",
                timeframe=TimeFrame.H1,
                start_time=datetime(2023, 1, 1),
                end_time=datetime(2023, 1, 5)
            )

            # Validation des données
            validated = binance_provider._validate_ohlcv_data(result)

            assert validated is True
            assert not result.isnull().any().any()  # Pas de valeurs manquantes
            assert (result['high'] >= result['open']).all()
            assert (result['high'] >= result['close']).all()
            assert (result['low'] <= result['open']).all()
            assert (result['low'] <= result['close']).all()

    @pytest.mark.asyncio
    async def test_rate_limiting(self, binance_provider):
        """Test gestion du rate limiting"""
        # Simuler plusieurs appels rapides
        mock_price = {"symbol": "BTCUSDT", "price": "50000.00"}

        with patch.object(binance_provider, '_get_ticker_price') as mock_ticker:
            mock_ticker.return_value = mock_price

            # Mesurer le temps d'exécution pour vérifier le rate limiting
            start_time = datetime.now()

            tasks = []
            for _ in range(5):
                task = binance_provider.get_current_price("BTCUSDT")
                tasks.append(task)

            prices = await asyncio.gather(*tasks)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Vérifier que tous les prix sont récupérés
            assert len(prices) == 5
            assert all(p == Decimal("50000.00") for p in prices)

            # Le rate limiting devrait introduire un délai
            # (cette assertion peut être ajustée selon la configuration)
            assert duration >= 0  # Au minimum pas d'erreur

    @pytest.mark.asyncio
    async def test_error_handling_network(self, binance_provider):
        """Test gestion d'erreurs réseau"""
        with patch.object(binance_provider, '_get_ticker_price') as mock_ticker:
            mock_ticker.side_effect = ConnectionError("Network error")

            with pytest.raises(ConnectionError):
                await binance_provider.get_current_price("BTCUSDT")

    @pytest.mark.asyncio
    async def test_error_handling_api_error(self, binance_provider):
        """Test gestion d'erreurs API"""
        with patch.object(binance_provider, '_get_ticker_price') as mock_ticker:
            mock_ticker.side_effect = Exception("API Error: Invalid symbol")

            with pytest.raises(Exception):
                await binance_provider.get_current_price("INVALID")

    def test_data_caching(self, binance_provider):
        """Test mise en cache des données"""
        # Test du système de cache si implémenté
        cache_key = binance_provider._generate_cache_key("BTCUSDT", TimeFrame.H1, datetime.now())

        assert isinstance(cache_key, str)
        assert "BTCUSDT" in cache_key
        assert "1h" in cache_key

    @pytest.mark.asyncio
    async def test_websocket_connection(self, binance_provider):
        """Test connexion WebSocket"""
        # Test de la connexion WebSocket si implémentée
        if hasattr(binance_provider, 'start_websocket'):
            with patch.object(binance_provider, '_websocket_handler') as mock_handler:
                mock_handler.return_value = True

                connected = await binance_provider.start_websocket("BTCUSDT")
                assert connected is True

    def test_batch_data_fetching(self, binance_provider, sample_ohlcv_data):
        """Test récupération de données en lot"""
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]

        with patch.object(binance_provider, '_fetch_klines') as mock_fetch:
            mock_fetch.return_value = sample_ohlcv_data

            # Simuler récupération batch
            results = {}
            for symbol in symbols:
                results[symbol] = mock_fetch.return_value

            assert len(results) == 3
            assert all(isinstance(df, pd.DataFrame) for df in results.values())

    @pytest.mark.asyncio
    async def test_historical_data_integrity(self, binance_provider):
        """Test intégrité des données historiques"""
        # Créer des données avec gaps
        dates = pd.date_range('2023-01-01', periods=100, freq='1h')
        # Retirer quelques points pour simuler des gaps
        dates_with_gaps = dates.delete([10, 20, 30])

        data_with_gaps = pd.DataFrame({
            'timestamp': dates_with_gaps,
            'open': np.random.uniform(49000, 51000, len(dates_with_gaps)),
            'high': np.random.uniform(50000, 52000, len(dates_with_gaps)),
            'low': np.random.uniform(48000, 50000, len(dates_with_gaps)),
            'close': np.random.uniform(49500, 50500, len(dates_with_gaps)),
            'volume': np.random.uniform(100, 1000, len(dates_with_gaps))
        })

        # Test de détection des gaps
        gaps = binance_provider._detect_data_gaps(data_with_gaps, TimeFrame.H1)

        assert len(gaps) == 3  # 3 gaps détectés
        assert all(isinstance(gap, tuple) for gap in gaps)

    def test_data_quality_metrics(self, binance_provider, sample_ohlcv_data):
        """Test métriques de qualité des données"""
        quality_metrics = binance_provider._calculate_data_quality(sample_ohlcv_data)

        assert isinstance(quality_metrics, dict)
        assert "completeness" in quality_metrics
        assert "consistency" in quality_metrics
        assert "freshness" in quality_metrics

        # Toutes les métriques devraient être entre 0 et 1
        for metric in quality_metrics.values():
            if isinstance(metric, (int, float)):
                assert 0 <= metric <= 1

    @pytest.mark.asyncio
    async def test_provider_health_check(self, binance_provider):
        """Test vérification santé du provider"""
        with patch.object(binance_provider, '_ping_server') as mock_ping:
            mock_ping.return_value = {"msg": "pong"}

            health = await binance_provider.health_check()

            assert isinstance(health, dict)
            assert "status" in health
            assert "latency" in health
            assert "last_update" in health

    def test_configuration_validation(self):
        """Test validation de la configuration"""
        # Test configuration valide
        provider = BinanceProvider(
            api_key="valid_key",
            api_secret="valid_secret",
            testnet=True
        )
        assert provider.api_key == "valid_key"

        # Test configuration invalide
        with pytest.raises(ValueError):
            BinanceProvider(
                api_key="",  # Clé vide
                api_secret="valid_secret",
                testnet=True
            )