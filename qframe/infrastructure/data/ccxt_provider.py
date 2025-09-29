"""
Infrastructure Layer: CCXT Universal Exchange Provider
====================================================

Provider universel utilisant CCXT pour connecter 100+ exchanges crypto.
Abstraction unifiée pour Binance, Coinbase, Kraken, Bitfinex, OKX, Bybit, etc.
"""

import asyncio
import ccxt.async_support as ccxt
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set
import logging

from ..observability.logging import LoggerFactory
from .market_data_pipeline import (
    DataProvider,
    DataType,
    DataQuality,
    MarketDataPoint,
    TickerData,
    OrderBookData
)


class CCXTProvider(DataProvider):
    """
    Provider universel pour tous les exchanges CCXT.

    Exchanges supportés: Binance, Coinbase, Kraken, Bitfinex, Huobi, OKX,
    Bybit, Gate.io, KuCoin, Bitget, MEXC, Crypto.com, BingX, etc.
    """

    # Mapping des exchanges les plus populaires
    SUPPORTED_EXCHANGES = {
        'binance': ccxt.binance,
        'coinbase': ccxt.coinbase,
        'kraken': ccxt.kraken,
        'bitfinex': ccxt.bitfinex,
        'huobi': ccxt.huobi,
        'okx': ccxt.okx,
        'bybit': ccxt.bybit,
        'gateio': ccxt.gateio,
        'kucoin': ccxt.kucoin,
        'bitget': ccxt.bitget,
        'mexc': ccxt.mexc,
        'cryptocom': ccxt.cryptocom,
        'bingx': ccxt.bingx,
        'phemex': ccxt.phemex,
        'bitmart': ccxt.bitmart
    }

    def __init__(
        self,
        exchange_name: str,
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        password: Optional[str] = None,
        sandbox: bool = True,
        rate_limit: bool = True,
        options: Optional[Dict[str, Any]] = None
    ):
        """
        Initialiser le provider CCXT.

        Args:
            exchange_name: Nom de l'exchange (binance, coinbase, kraken, etc.)
            api_key: Clé API (optionnel pour données publiques)
            secret: Secret API
            password: Passphrase (pour certains exchanges)
            sandbox: Utiliser le testnet/sandbox
            rate_limit: Activer la limitation de taux
            options: Options spécifiques à l'exchange
        """
        self.exchange_name = exchange_name.lower()
        self.logger = LoggerFactory.get_logger(f"qframe.ccxt.{self.exchange_name}")

        # Vérifier que l'exchange est supporté
        if self.exchange_name not in self.SUPPORTED_EXCHANGES:
            raise ValueError(f"Exchange {exchange_name} not supported. "
                           f"Supported: {list(self.SUPPORTED_EXCHANGES.keys())}")

        # Configuration de l'exchange
        exchange_class = self.SUPPORTED_EXCHANGES[self.exchange_name]
        self.config = {
            'apiKey': api_key,
            'secret': secret,
            'password': password,
            'sandbox': sandbox,
            'rateLimit': rate_limit,
            'enableRateLimit': rate_limit,
            'timeout': 30000,  # 30 secondes
        }

        if options:
            self.config.update(options)

        # Initialiser l'exchange CCXT
        self.exchange = exchange_class(self.config)

        # État du provider
        self._connected = False
        self._markets = {}
        self._symbols = []

        self.logger.info(f"Initialized CCXT provider for {exchange_name} (sandbox: {sandbox})")

    @property
    def name(self) -> str:
        return f"ccxt_{self.exchange_name}"

    async def connect(self) -> bool:
        """Connecter au exchange et charger les métadonnées"""
        try:
            # Charger les marchés
            self._markets = await self.exchange.load_markets()
            self._symbols = list(self._markets.keys())
            self._connected = True

            self.logger.info(f"Connected to {self.exchange_name}: {len(self._symbols)} symbols available")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to {self.exchange_name}: {e}")
            return False

    async def disconnect(self):
        """Déconnecter de l'exchange"""
        try:
            await self.exchange.close()
            self._connected = False
            self.logger.info(f"Disconnected from {self.exchange_name}")
        except Exception as e:
            self.logger.error(f"Error disconnecting from {self.exchange_name}: {e}")

    def is_connected(self) -> bool:
        return self._connected

    async def get_server_time(self) -> datetime:
        """Récupérer le temps serveur de l'exchange"""
        try:
            # Certains exchanges ont fetchTime, d'autres non
            if hasattr(self.exchange, 'fetch_time'):
                timestamp_ms = await self.exchange.fetch_time()
                return datetime.utcfromtimestamp(timestamp_ms / 1000)
            else:
                # Fallback sur temps local
                return datetime.utcnow()
        except Exception as e:
            self.logger.error(f"Error getting server time: {e}")
            return datetime.utcnow()

    async def get_symbols(self) -> List[str]:
        """Récupérer la liste des symboles disponibles"""
        if not self._connected:
            await self.connect()
        return self._symbols

    async def get_ticker_24hr(self, symbol: str) -> TickerData:
        """Récupérer les statistiques 24h d'un symbole"""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)

            return TickerData(
                symbol=symbol,
                bid=Decimal(str(ticker.get('bid', 0) or 0)),
                ask=Decimal(str(ticker.get('ask', 0) or 0)),
                last=Decimal(str(ticker.get('last', 0) or 0)),
                volume_24h=Decimal(str(ticker.get('baseVolume', 0) or 0)),
                change_24h=Decimal(str(ticker.get('percentage', 0) or 0)),
                high_24h=Decimal(str(ticker.get('high', 0) or 0)),
                low_24h=Decimal(str(ticker.get('low', 0) or 0)),
                timestamp=datetime.utcfromtimestamp(ticker['timestamp'] / 1000) if ticker.get('timestamp') else datetime.utcnow()
            )

        except Exception as e:
            self.logger.error(f"Error getting ticker for {symbol}: {e}")
            return self._empty_ticker_data(symbol)

    async def get_order_book(self, symbol: str, limit: int = 100) -> OrderBookData:
        """Récupérer le carnet d'ordres"""
        try:
            order_book = await self.exchange.fetch_order_book(symbol, limit)

            bids = [[Decimal(str(bid[0])), Decimal(str(bid[1]))] for bid in order_book['bids'][:limit]]
            asks = [[Decimal(str(ask[0])), Decimal(str(ask[1]))] for ask in order_book['asks'][:limit]]

            return OrderBookData(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=datetime.utcfromtimestamp(order_book['timestamp'] / 1000) if order_book.get('timestamp') else datetime.utcnow(),
                sequence=order_book.get('nonce')
            )

        except Exception as e:
            self.logger.error(f"Error getting order book for {symbol}: {e}")
            return self._empty_order_book_data(symbol)

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[MarketDataPoint]:
        """Récupérer les données OHLCV (klines)"""
        try:
            # Mapper les intervalles vers CCXT
            timeframe = self._map_interval_to_ccxt(interval)

            # Paramètres temporels
            since = None
            if start_time:
                since = int(start_time.timestamp() * 1000)

            # Récupérer les données OHLCV
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol,
                timeframe,
                since=since,
                limit=min(limit, 1000)
            )

            # Filtrer par end_time si spécifié
            if end_time:
                end_timestamp = int(end_time.timestamp() * 1000)
                ohlcv = [candle for candle in ohlcv if candle[0] <= end_timestamp]

            return self._convert_ohlcv_to_market_data_points(ohlcv, symbol)

        except Exception as e:
            self.logger.error(f"Error getting klines for {symbol}: {e}")
            return []

    async def get_exchange_info(self) -> Dict[str, Any]:
        """Récupérer les informations de l'exchange"""
        try:
            if not self._connected:
                await self.connect()

            return {
                "name": self.exchange_name,
                "id": self.exchange.id,
                "countries": getattr(self.exchange, 'countries', []),
                "has": self.exchange.has,
                "timeframes": getattr(self.exchange, 'timeframes', {}),
                "symbols": len(self._symbols),
                "markets": len(self._markets),
                "sandbox": self.config.get('sandbox', False)
            }

        except Exception as e:
            self.logger.error(f"Error getting exchange info: {e}")
            return {"error": str(e)}

    def get_capabilities(self) -> Dict[str, Any]:
        """Retourner les capacités du provider"""
        return {
            "name": f"CCXT {self.exchange_name.title()}",
            "data_types": ["ticker", "klines", "trades", "orderbook"],
            "intervals": list(getattr(self.exchange, 'timeframes', {}).keys()),
            "websocket": False,  # CCXT REST principalement
            "rest_api": True,
            "rate_limits": {
                "requests_per_minute": getattr(self.exchange, 'rateLimit', 1000),
                "enabled": self.config.get('rateLimit', True)
            },
            "features": getattr(self.exchange, 'has', {}),
            "markets": len(self._markets),
            "symbols": len(self._symbols)
        }

    def get_status(self) -> Dict[str, Any]:
        """Retourner le statut du provider"""
        return {
            "connected": self.is_connected(),
            "exchange": self.exchange_name,
            "sandbox": self.config.get('sandbox', False),
            "symbols_loaded": len(self._symbols),
            "last_update": datetime.utcnow().isoformat()
        }

    # ===== MÉTHODES UTILITAIRES =====

    def _map_interval_to_ccxt(self, interval: str) -> str:
        """Mapper les intervalles vers format CCXT"""
        mapping = {
            '1s': '1s',
            '1m': '1m',
            '3m': '3m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '2h': '2h',
            '4h': '4h',
            '6h': '6h',
            '8h': '8h',
            '12h': '12h',
            '1d': '1d',
            '3d': '3d',
            '1w': '1w',
            '1M': '1M'
        }
        return mapping.get(interval, '1h')  # Default 1h

    def _convert_ohlcv_to_market_data_points(self, ohlcv: List[List], symbol: str) -> List[MarketDataPoint]:
        """Convertir OHLCV CCXT en MarketDataPoint"""
        market_data_points = []

        for candle in ohlcv:
            try:
                # Format CCXT: [timestamp, open, high, low, close, volume]
                timestamp = datetime.utcfromtimestamp(candle[0] / 1000)

                data = {
                    "open": Decimal(str(candle[1])),
                    "high": Decimal(str(candle[2])),
                    "low": Decimal(str(candle[3])),
                    "close": Decimal(str(candle[4])),
                    "volume": Decimal(str(candle[5]))
                }

                market_data_point = MarketDataPoint(
                    symbol=symbol,
                    data_type=DataType.CANDLES,
                    timestamp=timestamp,
                    data=data,
                    provider=f"ccxt_{self.exchange_name}",
                    quality=DataQuality.HIGH
                )

                # Ajouter propriétés pour compatibilité tests
                market_data_point.open = data["open"]
                market_data_point.high = data["high"]
                market_data_point.low = data["low"]
                market_data_point.close = data["close"]
                market_data_point.volume = data["volume"]

                market_data_points.append(market_data_point)

            except Exception as e:
                self.logger.error(f"Error converting OHLCV candle: {e}")
                continue

        return market_data_points

    def _empty_ticker_data(self, symbol: str) -> TickerData:
        """Créer un TickerData vide"""
        return TickerData(
            symbol=symbol,
            bid=Decimal("0"),
            ask=Decimal("0"),
            last=Decimal("0"),
            volume_24h=Decimal("0"),
            change_24h=Decimal("0"),
            high_24h=Decimal("0"),
            low_24h=Decimal("0"),
            timestamp=datetime.utcnow()
        )

    def _empty_order_book_data(self, symbol: str) -> OrderBookData:
        """Créer un OrderBookData vide"""
        return OrderBookData(
            symbol=symbol,
            bids=[],
            asks=[],
            timestamp=datetime.utcnow(),
            sequence=None
        )

    # ===== MÉTHODES DATA PROVIDER ABSTRACT =====

    async def subscribe(self, symbol: str, data_types: List[DataType]) -> bool:
        """Souscrire aux données (non implémenté pour REST)"""
        self.logger.warning("WebSocket subscriptions not implemented for CCXT REST providers")
        return False

    async def unsubscribe(self, symbol: str, data_types: List[DataType]) -> bool:
        """Se désabonner des données"""
        return False

    def add_callback(self, callback):
        """Ajouter callback pour données en temps réel"""
        self.logger.warning("Real-time callbacks not supported for CCXT REST providers")

    async def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[MarketDataPoint]:
        """Récupérer données historiques"""
        return await self.get_klines(symbol, "1h", limit=1000, start_time=start_date, end_time=end_date)


# ===== FACTORY POUR CRÉATION FACILE =====

class CCXTProviderFactory:
    """Factory pour créer facilement des providers CCXT"""

    @staticmethod
    def create_binance(sandbox: bool = True, api_key: Optional[str] = None, secret: Optional[str] = None) -> CCXTProvider:
        """Créer provider Binance"""
        return CCXTProvider("binance", api_key=api_key, secret=secret, sandbox=sandbox)

    @staticmethod
    def create_coinbase(sandbox: bool = True, api_key: Optional[str] = None, secret: Optional[str] = None, password: Optional[str] = None) -> CCXTProvider:
        """Créer provider Coinbase"""
        return CCXTProvider("coinbase", api_key=api_key, secret=secret, password=password, sandbox=sandbox)

    @staticmethod
    def create_kraken(api_key: Optional[str] = None, secret: Optional[str] = None) -> CCXTProvider:
        """Créer provider Kraken"""
        return CCXTProvider("kraken", api_key=api_key, secret=secret, sandbox=False)  # Kraken n'a pas de sandbox

    @staticmethod
    def create_okx(sandbox: bool = True, api_key: Optional[str] = None, secret: Optional[str] = None, password: Optional[str] = None) -> CCXTProvider:
        """Créer provider OKX"""
        return CCXTProvider("okx", api_key=api_key, secret=secret, password=password, sandbox=sandbox)

    @staticmethod
    def create_bybit(sandbox: bool = True, api_key: Optional[str] = None, secret: Optional[str] = None) -> CCXTProvider:
        """Créer provider Bybit"""
        return CCXTProvider("bybit", api_key=api_key, secret=secret, sandbox=sandbox)

    @staticmethod
    def get_all_supported_exchanges() -> List[str]:
        """Retourner la liste de tous les exchanges supportés"""
        return list(CCXTProvider.SUPPORTED_EXCHANGES.keys())

    @staticmethod
    def create_provider(exchange_name: str, **kwargs) -> CCXTProvider:
        """Créer un provider pour n'importe quel exchange supporté"""
        return CCXTProvider(exchange_name, **kwargs)