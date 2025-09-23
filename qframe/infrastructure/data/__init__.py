"""
Infrastructure Layer: Data Pipeline Module
=========================================

Module pour la gestion des données en temps réel.
Pipeline pour ingestion, normalisation, validation et streaming.
"""

from .market_data_pipeline import (
    DataType,
    DataQuality,
    MarketDataPoint,
    TickerData,
    OrderBookData,
    DataProvider,
    MockDataProvider,
    DataValidator,
    DataNormalizer,
    MarketDataCache,
    MarketDataPipeline,
    get_market_data_pipeline
)

from .real_time_streaming import (
    StreamingProtocol,
    SubscriptionLevel,
    StreamingSubscription,
    StreamingConsumer,
    StreamingStatistics,
    RealTimeStreamingService,
    get_streaming_service
)

from .binance_provider import BinanceProvider
from .coinbase_provider import CoinbaseProvider

__all__ = [
    # Core types
    'DataType',
    'DataQuality',
    'MarketDataPoint',
    'TickerData',
    'OrderBookData',

    # Provider interface and implementations
    'DataProvider',
    'MockDataProvider',
    'BinanceProvider',
    'CoinbaseProvider',

    # Pipeline components
    'DataValidator',
    'DataNormalizer',
    'MarketDataCache',
    'MarketDataPipeline',
    'get_market_data_pipeline',

    # Streaming components
    'StreamingProtocol',
    'SubscriptionLevel',
    'StreamingSubscription',
    'StreamingConsumer',
    'StreamingStatistics',
    'RealTimeStreamingService',
    'get_streaming_service'
]