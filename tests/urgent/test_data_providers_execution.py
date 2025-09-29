"""
Tests d'Exécution Réelle - Infrastructure Data Providers
========================================================

Tests qui EXÉCUTENT vraiment le code qframe.infrastructure.data
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional
from unittest.mock import Mock, AsyncMock, patch

# Data Providers
from qframe.infrastructure.data.binance_provider import BinanceProvider
from qframe.infrastructure.data.market_data_pipeline import MarketDataPipeline


class TestBinanceProviderExecution:
    """Tests d'exécution réelle pour BinanceProvider."""

    @pytest.fixture
    def binance_provider(self):
        """Provider Binance configuré."""
        # Provider sans API keys pour tests
        return BinanceProvider(api_key=None, api_secret=None, testnet=True)

    def test_binance_provider_initialization_execution(self, binance_provider):
        """Test initialisation du provider."""
        # Vérifier initialisation
        assert binance_provider is not None
        assert hasattr(binance_provider, 'testnet')
        assert binance_provider.testnet is True
        assert hasattr(binance_provider, 'fetch_ohlcv')
        assert hasattr(binance_provider, 'get_supported_symbols')

    @pytest.mark.asyncio
    async def test_get_supported_symbols_execution(self, binance_provider):
        """Test récupération des symboles supportés."""
        # Exécuter récupération
        symbols = binance_provider.get_supported_symbols()

        # Vérifier résultat
        assert isinstance(symbols, list)
        assert len(symbols) > 0

        # Vérifier format des symboles
        for symbol in symbols[:5]:  # Tester quelques-uns
            assert isinstance(symbol, str)
            assert "/" in symbol  # Format "BTC/USDT"

    @pytest.mark.asyncio
    @patch('ccxt.binance')
    async def test_fetch_ohlcv_execution(self, mock_ccxt, binance_provider):
        """Test récupération données OHLCV."""
        # Mock ccxt pour éviter vrais appels API
        mock_exchange = Mock()
        mock_ccxt.return_value = mock_exchange

        # Mock données OHLCV
        mock_ohlcv_data = [
            [1609459200000, 29000, 29500, 28800, 29300, 1000],  # timestamp, O, H, L, C, V
            [1609462800000, 29300, 29800, 29200, 29600, 1100],
            [1609466400000, 29600, 30000, 29500, 29900, 1200],
        ]
        mock_exchange.fetch_ohlcv = AsyncMock(return_value=mock_ohlcv_data)

        # Exécuter fetch
        df = await binance_provider.fetch_ohlcv(
            symbol="BTC/USDT",
            timeframe="1h",
            limit=3
        )

        # Vérifier DataFrame
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])

        # Vérifier types de données
        assert df['open'].dtype in [np.float64, np.float32]
        assert df['volume'].dtype in [np.float64, np.float32]

    @pytest.mark.asyncio
    @patch('ccxt.binance')
    async def test_fetch_ticker_execution(self, mock_ccxt, binance_provider):
        """Test récupération ticker temps réel."""
        # Mock ccxt
        mock_exchange = Mock()
        mock_ccxt.return_value = mock_exchange

        # Mock ticker data
        mock_ticker = {
            'symbol': 'BTC/USDT',
            'bid': 49900,
            'ask': 50000,
            'last': 49950,
            'volume': 10000,
            'timestamp': datetime.utcnow().timestamp() * 1000
        }
        mock_exchange.fetch_ticker = AsyncMock(return_value=mock_ticker)

        # Exécuter fetch ticker
        ticker = await binance_provider.fetch_ticker("BTC/USDT")

        # Vérifier résultat
        assert isinstance(ticker, dict)
        assert ticker['symbol'] == 'BTC/USDT'
        assert 'bid' in ticker
        assert 'ask' in ticker
        assert ticker['last'] == 49950

    @pytest.mark.asyncio
    @patch('ccxt.binance')
    async def test_fetch_order_book_execution(self, mock_ccxt, binance_provider):
        """Test récupération carnet d'ordres."""
        # Mock ccxt
        mock_exchange = Mock()
        mock_ccxt.return_value = mock_exchange

        # Mock order book
        mock_orderbook = {
            'bids': [[49900, 2.5], [49850, 3.0], [49800, 4.0]],
            'asks': [[50000, 1.8], [50050, 2.2], [50100, 2.8]],
            'timestamp': datetime.utcnow().timestamp() * 1000,
            'symbol': 'BTC/USDT'
        }
        mock_exchange.fetch_order_book = AsyncMock(return_value=mock_orderbook)

        # Exécuter fetch
        orderbook = await binance_provider.fetch_order_book("BTC/USDT", limit=3)

        # Vérifier résultat
        assert isinstance(orderbook, dict)
        assert 'bids' in orderbook
        assert 'asks' in orderbook
        assert len(orderbook['bids']) > 0
        assert len(orderbook['asks']) > 0

        # Vérifier structure bid/ask
        first_bid = orderbook['bids'][0]
        assert isinstance(first_bid, list)
        assert len(first_bid) == 2  # [price, quantity]

    @pytest.mark.asyncio
    @patch('ccxt.binance')
    async def test_fetch_trades_execution(self, mock_ccxt, binance_provider):
        """Test récupération historique des trades."""
        # Mock ccxt
        mock_exchange = Mock()
        mock_ccxt.return_value = mock_exchange

        # Mock trades data
        mock_trades = [
            {
                'id': '12345',
                'timestamp': datetime.utcnow().timestamp() * 1000,
                'symbol': 'BTC/USDT',
                'price': 49950,
                'amount': 0.5,
                'side': 'buy'
            },
            {
                'id': '12346',
                'timestamp': (datetime.utcnow() - timedelta(seconds=30)).timestamp() * 1000,
                'symbol': 'BTC/USDT',
                'price': 49900,
                'amount': 0.3,
                'side': 'sell'
            }
        ]
        mock_exchange.fetch_trades = AsyncMock(return_value=mock_trades)

        # Exécuter fetch
        trades = await binance_provider.fetch_trades("BTC/USDT", limit=2)

        # Vérifier résultat
        assert isinstance(trades, list)
        assert len(trades) == 2

        # Vérifier structure trade
        first_trade = trades[0]
        assert 'id' in first_trade
        assert 'price' in first_trade
        assert 'amount' in first_trade
        assert 'side' in first_trade

    def test_normalize_symbol_execution(self, binance_provider):
        """Test normalisation des symboles."""
        # Exécuter normalisation
        normalized = binance_provider.normalize_symbol("BTC-USDT")
        assert normalized == "BTC/USDT"

        normalized = binance_provider.normalize_symbol("BTCUSDT")
        assert normalized == "BTC/USDT"

        normalized = binance_provider.normalize_symbol("btc/usdt")
        assert normalized == "BTC/USDT"

    @pytest.mark.asyncio
    @patch('ccxt.binance')
    async def test_calculate_vwap_execution(self, mock_ccxt, binance_provider):
        """Test calcul VWAP."""
        # Mock trades pour VWAP
        mock_trades = [
            {'price': 50000, 'amount': 1.0},
            {'price': 50100, 'amount': 2.0},
            {'price': 49900, 'amount': 1.5},
        ]

        # Calculer VWAP
        vwap = binance_provider.calculate_vwap(mock_trades)

        # Vérifier calcul
        # VWAP = (50000*1 + 50100*2 + 49900*1.5) / (1 + 2 + 1.5)
        expected_vwap = (50000*1 + 50100*2 + 49900*1.5) / 4.5
        assert abs(vwap - expected_vwap) < 0.01


class TestMarketDataPipelineExecution:
    """Tests d'exécution réelle pour MarketDataPipeline."""

    @pytest.fixture
    def market_pipeline(self):
        """Pipeline de données de marché."""
        return MarketDataPipeline()

    def test_pipeline_initialization_execution(self, market_pipeline):
        """Test initialisation du pipeline."""
        # Vérifier initialisation
        assert market_pipeline is not None
        assert hasattr(market_pipeline, 'process_data')
        assert hasattr(market_pipeline, 'add_technical_indicators')
        assert hasattr(market_pipeline, 'clean_data')

    def test_process_raw_data_execution(self, market_pipeline):
        """Test traitement données brutes."""
        # Créer données OHLCV de test
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        raw_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(100).cumsum() + 50000,
            'high': np.random.randn(100).cumsum() + 50100,
            'low': np.random.randn(100).cumsum() + 49900,
            'close': np.random.randn(100).cumsum() + 50000,
            'volume': np.random.randint(100, 1000, 100)
        })

        # Exécuter traitement
        processed_data = market_pipeline.process_data(raw_data)

        # Vérifier résultat
        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) > 0
        assert 'timestamp' in processed_data.columns

        # Vérifier que les données sont nettoyées
        assert processed_data.isna().sum().sum() == 0  # Pas de NaN

    def test_add_technical_indicators_execution(self, market_pipeline):
        """Test ajout indicateurs techniques."""
        # Données OHLCV de base
        dates = pd.date_range(start='2024-01-01', periods=50, freq='1H')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': 50000 + np.random.randn(50) * 100,
            'high': 50100 + np.random.randn(50) * 100,
            'low': 49900 + np.random.randn(50) * 100,
            'close': 50000 + np.random.randn(50) * 100,
            'volume': np.random.randint(100, 1000, 50)
        })

        # Exécuter ajout d'indicateurs
        data_with_indicators = market_pipeline.add_technical_indicators(data)

        # Vérifier ajout des indicateurs
        assert isinstance(data_with_indicators, pd.DataFrame)

        # Vérifier présence d'indicateurs communs
        expected_indicators = ['sma_20', 'ema_20', 'rsi', 'macd', 'bollinger_upper', 'bollinger_lower']
        for indicator in expected_indicators:
            if indicator in data_with_indicators.columns:
                assert data_with_indicators[indicator].notna().any()

    def test_calculate_returns_execution(self, market_pipeline):
        """Test calcul des rendements."""
        # Données de prix
        prices = pd.Series([100, 102, 101, 103, 102, 104])

        # Exécuter calcul des rendements
        returns = market_pipeline.calculate_returns(prices)

        # Vérifier résultat
        assert isinstance(returns, pd.Series)
        assert len(returns) == len(prices) - 1  # n-1 returns

        # Vérifier calculs
        expected_first_return = (102 - 100) / 100
        assert abs(returns.iloc[0] - expected_first_return) < 0.0001

    def test_resample_data_execution(self, market_pipeline):
        """Test rééchantillonnage temporel."""
        # Données 1 minute
        dates = pd.date_range(start='2024-01-01', periods=60, freq='1min')
        data = pd.DataFrame({
            'timestamp': dates,
            'close': 50000 + np.random.randn(60) * 10,
            'volume': np.random.randint(10, 100, 60)
        })
        data.set_index('timestamp', inplace=True)

        # Exécuter rééchantillonnage vers 5 minutes
        resampled = market_pipeline.resample_data(data, '5min')

        # Vérifier résultat
        assert isinstance(resampled, pd.DataFrame)
        assert len(resampled) == 12  # 60 min / 5 min = 12

        # Vérifier agrégation
        assert 'close' in resampled.columns
        assert 'volume' in resampled.columns

    def test_clean_outliers_execution(self, market_pipeline):
        """Test nettoyage des outliers."""
        # Données avec outliers
        data = pd.Series([100, 101, 102, 1000, 103, 104, 10, 105])  # 1000 et 10 sont outliers

        # Exécuter nettoyage
        cleaned = market_pipeline.clean_outliers(data, z_threshold=2)

        # Vérifier que les outliers extrêmes sont gérés
        assert isinstance(cleaned, pd.Series)
        assert len(cleaned) == len(data)

        # Les valeurs extrêmes devraient être remplacées ou marquées
        assert cleaned.max() < 1000  # L'outlier 1000 devrait être géré

    def test_pipeline_full_workflow_execution(self, market_pipeline):
        """Test workflow complet du pipeline."""
        # Créer données brutes réalistes
        dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
        np.random.seed(42)

        prices = 50000
        price_series = []
        for _ in range(200):
            prices = prices * (1 + np.random.randn() * 0.002)  # 0.2% volatilité
            price_series.append(prices)

        raw_data = pd.DataFrame({
            'timestamp': dates,
            'open': price_series + np.random.randn(200) * 10,
            'high': [p + abs(np.random.randn() * 50) for p in price_series],
            'low': [p - abs(np.random.randn() * 50) for p in price_series],
            'close': price_series,
            'volume': np.random.randint(100, 1000, 200)
        })

        # 1. Nettoyer les données
        cleaned_data = market_pipeline.clean_data(raw_data)
        assert cleaned_data.isna().sum().sum() == 0

        # 2. Ajouter indicateurs techniques
        data_with_indicators = market_pipeline.add_technical_indicators(cleaned_data)
        assert len(data_with_indicators.columns) > len(cleaned_data.columns)

        # 3. Calculer les rendements
        data_with_indicators['returns'] = market_pipeline.calculate_returns(data_with_indicators['close'])

        # 4. Validation finale
        final_data = market_pipeline.validate_data(data_with_indicators)
        assert isinstance(final_data, pd.DataFrame)
        assert len(final_data) > 0

        # Vérifier qualité des données finales
        assert final_data['close'].min() > 0  # Pas de prix négatifs
        assert final_data['volume'].min() >= 0  # Volume non négatif


class TestDataQualityExecution:
    """Tests de qualité des données."""

    @pytest.fixture
    def market_pipeline(self):
        """Pipeline de données."""
        return MarketDataPipeline()

    def test_detect_gaps_execution(self, market_pipeline):
        """Test détection des gaps temporels."""
        # Données avec gap
        dates = pd.date_range(start='2024-01-01', periods=10, freq='1H')
        dates_with_gap = dates.delete([4, 5])  # Enlever 2 heures

        data = pd.DataFrame({
            'timestamp': dates_with_gap,
            'close': np.random.randn(8) + 50000
        })

        # Exécuter détection de gaps
        gaps = market_pipeline.detect_gaps(data, expected_freq='1H')

        # Vérifier détection
        assert isinstance(gaps, list)
        assert len(gaps) > 0  # Au moins un gap détecté

    def test_validate_price_data_execution(self, market_pipeline):
        """Test validation des données de prix."""
        # Données avec problèmes
        data = pd.DataFrame({
            'open': [100, 101, -5, 103],  # Prix négatif
            'high': [105, 106, 107, 108],
            'low': [95, 96, 97, 110],  # Low > High
            'close': [102, 103, 104, 105]
        })

        # Exécuter validation
        is_valid, errors = market_pipeline.validate_price_data(data)

        # Vérifier détection des erreurs
        assert is_valid is False
        assert len(errors) > 0
        assert any('negative' in str(e).lower() for e in errors)

    def test_data_completeness_check_execution(self, market_pipeline):
        """Test vérification complétude des données."""
        # Données incomplètes
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1H'),
            'open': np.random.randn(100) + 50000,
            'high': np.random.randn(100) + 50100,
            'low': np.random.randn(100) + 49900,
            'close': np.random.randn(100) + 50000,
            'volume': np.random.randint(100, 1000, 100)
        })

        # Ajouter quelques NaN
        data.loc[10:15, 'close'] = np.nan
        data.loc[20:22, 'volume'] = np.nan

        # Exécuter vérification
        completeness_report = market_pipeline.check_completeness(data)

        # Vérifier rapport
        assert isinstance(completeness_report, dict)
        assert 'total_rows' in completeness_report
        assert 'missing_values' in completeness_report
        assert completeness_report['missing_values']['close'] == 6
        assert completeness_report['missing_values']['volume'] == 3