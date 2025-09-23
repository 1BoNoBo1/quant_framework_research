"""
Infrastructure Layer: Binance Market Data Provider
=================================================

Provider pour récupérer les données de marché en temps réel depuis Binance
via WebSocket et API REST.
"""

import asyncio
import json
import time
import websockets
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable, Set
import aiohttp
from urllib.parse import urlencode

from ..observability.logging import LoggerFactory
from .market_data_pipeline import (
    DataProvider,
    DataType,
    DataQuality,
    MarketDataPoint,
    TickerData,
    OrderBookData
)


class BinanceProvider(DataProvider):
    """
    Provider pour données de marché Binance via WebSocket et REST API.
    Supporte spot et futures.
    """

    def __init__(
        self,
        base_url: str = "https://api.binance.com",
        ws_base_url: str = "wss://stream.binance.com:9443",
        testnet: bool = False,
        market_type: str = "spot"  # "spot" ou "futures"
    ):
        self._name = f"binance_{market_type}"
        self.base_url = base_url
        self.ws_base_url = ws_base_url
        self.testnet = testnet
        self.market_type = market_type

        # URLs pour testnet
        if testnet:
            self.base_url = "https://testnet.binance.vision"
            self.ws_base_url = "wss://testnet.binance.vision"

        # URLs pour futures
        if market_type == "futures":
            self.base_url = self.base_url.replace("api.binance", "fapi.binance")
            self.ws_base_url = self.ws_base_url.replace("stream.binance", "fstream.binance")

        self._connected = False
        self._subscriptions: Dict[str, Set[DataType]] = {}
        self._websocket: Optional[websockets.WebSocketServerProtocol] = None
        self._data_callbacks: List[Callable[[MarketDataPoint], None]] = []
        self._listen_task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None

        self.logger = LoggerFactory.get_logger(__name__)

    @property
    def name(self) -> str:
        return self._name

    async def connect(self) -> bool:
        """Se connecter à Binance WebSocket"""
        try:
            # Créer une session HTTP
            self._session = aiohttp.ClientSession()

            # Tester la connectivité REST
            ping_url = f"{self.base_url}/api/v3/ping"
            async with self._session.get(ping_url) as response:
                if response.status != 200:
                    self.logger.error(f"Failed to ping Binance API: {response.status}")
                    return False

            self._connected = True
            self.logger.info(f"Binance provider {self.name} connected successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error connecting to Binance: {e}")
            return False

    async def disconnect(self):
        """Se déconnecter"""
        self._connected = False

        # Arrêter la tâche d'écoute WebSocket
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass

        # Fermer WebSocket
        if self._websocket:
            await self._websocket.close()
            self._websocket = None

        # Fermer session HTTP
        if self._session:
            await self._session.close()
            self._session = None

        self.logger.info(f"Binance provider {self.name} disconnected")

    async def subscribe(self, symbol: str, data_types: List[DataType]) -> bool:
        """S'abonner aux données d'un symbole"""
        if not self._connected:
            return False

        try:
            # Normaliser le symbole pour Binance (ex: BTC-USD -> BTCUSDT)
            binance_symbol = self._normalize_symbol(symbol)

            # Ajouter aux subscriptions
            if binance_symbol not in self._subscriptions:
                self._subscriptions[binance_symbol] = set()
            self._subscriptions[binance_symbol].update(data_types)

            # Démarrer WebSocket si nécessaire
            if not self._listen_task:
                await self._start_websocket()

            self.logger.info(f"Subscribed to {symbol} ({binance_symbol}) for {data_types}")
            return True

        except Exception as e:
            self.logger.error(f"Error subscribing to {symbol}: {e}")
            return False

    async def unsubscribe(self, symbol: str, data_types: List[DataType]) -> bool:
        """Se désabonner des données"""
        binance_symbol = self._normalize_symbol(symbol)

        if binance_symbol in self._subscriptions:
            for data_type in data_types:
                self._subscriptions[binance_symbol].discard(data_type)

            # Supprimer si plus de subscriptions
            if not self._subscriptions[binance_symbol]:
                del self._subscriptions[binance_symbol]

        self.logger.info(f"Unsubscribed from {symbol}")
        return True

    def is_connected(self) -> bool:
        return self._connected

    def add_callback(self, callback: Callable[[MarketDataPoint], None]):
        """Ajouter un callback pour recevoir les données"""
        self._data_callbacks.append(callback)

    async def get_historical_data(
        self,
        symbol: str,
        data_type: DataType,
        start_time: datetime,
        end_time: datetime
    ) -> List[MarketDataPoint]:
        """Récupérer des données historiques via REST API"""
        if not self._session:
            return []

        try:
            binance_symbol = self._normalize_symbol(symbol)
            data_points = []

            if data_type == DataType.CANDLES:
                # Récupérer les klines (candlesticks)
                url = f"{self.base_url}/api/v3/klines"
                params = {
                    "symbol": binance_symbol,
                    "interval": "1m",  # 1 minute par défaut
                    "startTime": int(start_time.timestamp() * 1000),
                    "endTime": int(end_time.timestamp() * 1000),
                    "limit": 1000
                }

                async with self._session.get(url, params=params) as response:
                    if response.status == 200:
                        klines = await response.json()
                        for kline in klines:
                            data_point = MarketDataPoint(
                                symbol=symbol,
                                data_type=DataType.CANDLES,
                                timestamp=datetime.fromtimestamp(kline[0] / 1000),
                                data={
                                    "open": float(kline[1]),
                                    "high": float(kline[2]),
                                    "low": float(kline[3]),
                                    "close": float(kline[4]),
                                    "volume": float(kline[5])
                                },
                                provider=self.name,
                                quality=DataQuality.HIGH
                            )
                            data_points.append(data_point)

            elif data_type == DataType.TRADES:
                # Récupérer les trades récents
                url = f"{self.base_url}/api/v3/aggTrades"
                params = {
                    "symbol": binance_symbol,
                    "startTime": int(start_time.timestamp() * 1000),
                    "endTime": int(end_time.timestamp() * 1000),
                    "limit": 1000
                }

                async with self._session.get(url, params=params) as response:
                    if response.status == 200:
                        trades = await response.json()
                        for trade in trades:
                            data_point = MarketDataPoint(
                                symbol=symbol,
                                data_type=DataType.TRADES,
                                timestamp=datetime.fromtimestamp(trade["T"] / 1000),
                                data={
                                    "price": float(trade["p"]),
                                    "quantity": float(trade["q"]),
                                    "trade_id": trade["a"],
                                    "is_buyer_maker": trade["m"]
                                },
                                provider=self.name,
                                quality=DataQuality.HIGH
                            )
                            data_points.append(data_point)

            return data_points

        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return []

    async def _start_websocket(self):
        """Démarrer la connexion WebSocket"""
        try:
            # Construire l'URL WebSocket avec les streams
            streams = self._build_stream_names()
            if not streams:
                return

            ws_url = f"{self.ws_base_url}/ws/{'/'.join(streams)}"

            self.logger.info(f"Connecting to WebSocket: {ws_url}")

            self._websocket = await websockets.connect(ws_url)
            self._listen_task = asyncio.create_task(self._listen_websocket())

        except Exception as e:
            self.logger.error(f"Error starting WebSocket: {e}")

    async def _listen_websocket(self):
        """Écouter les messages WebSocket"""
        try:
            async for message in self._websocket:
                try:
                    data = json.loads(message)
                    await self._process_websocket_message(data)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Invalid JSON from WebSocket: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing WebSocket message: {e}")

        except websockets.exceptions.ConnectionClosed:
            self.logger.info("WebSocket connection closed")
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")

    async def _process_websocket_message(self, data: Dict[str, Any]):
        """Traiter un message WebSocket"""
        try:
            stream = data.get("stream", "")
            event_data = data.get("data", {})

            # Déterminer le symbole et type de données
            if "@ticker" in stream:
                await self._process_ticker_data(event_data)
            elif "@depth" in stream:
                await self._process_orderbook_data(event_data)
            elif "@aggTrade" in stream:
                await self._process_trade_data(event_data)

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

    async def _process_ticker_data(self, data: Dict[str, Any]):
        """Traiter les données ticker"""
        try:
            symbol = self._denormalize_symbol(data["s"])

            ticker_data = TickerData(
                symbol=symbol,
                bid=Decimal(data["b"]),
                ask=Decimal(data["a"]),
                last=Decimal(data["c"]),
                volume_24h=Decimal(data["v"]),
                change_24h=Decimal(data["P"]) / 100,  # Pourcentage en décimal
                high_24h=Decimal(data["h"]),
                low_24h=Decimal(data["l"]),
                timestamp=datetime.fromtimestamp(data["E"] / 1000)
            )

            data_point = MarketDataPoint(
                symbol=symbol,
                data_type=DataType.TICKER,
                timestamp=ticker_data.timestamp,
                data=ticker_data.to_dict(),
                provider=self.name,
                quality=DataQuality.HIGH
            )

            # Envoyer aux callbacks
            for callback in self._data_callbacks:
                callback(data_point)

        except Exception as e:
            self.logger.error(f"Error processing ticker data: {e}")

    async def _process_orderbook_data(self, data: Dict[str, Any]):
        """Traiter les données orderbook"""
        try:
            symbol = self._denormalize_symbol(data["s"])

            bids = [[Decimal(bid[0]), Decimal(bid[1])] for bid in data["b"]]
            asks = [[Decimal(ask[0]), Decimal(ask[1])] for ask in data["a"]]

            orderbook_data = OrderBookData(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=datetime.fromtimestamp(data["E"] / 1000),
                sequence=data.get("lastUpdateId")
            )

            data_point = MarketDataPoint(
                symbol=symbol,
                data_type=DataType.ORDERBOOK,
                timestamp=orderbook_data.timestamp,
                data=orderbook_data.to_dict(),
                provider=self.name,
                quality=DataQuality.HIGH
            )

            # Envoyer aux callbacks
            for callback in self._data_callbacks:
                callback(data_point)

        except Exception as e:
            self.logger.error(f"Error processing orderbook data: {e}")

    async def _process_trade_data(self, data: Dict[str, Any]):
        """Traiter les données de trades"""
        try:
            symbol = self._denormalize_symbol(data["s"])

            trade_data = {
                "price": float(data["p"]),
                "quantity": float(data["q"]),
                "trade_id": data["a"],
                "timestamp": data["T"],
                "is_buyer_maker": data["m"]
            }

            data_point = MarketDataPoint(
                symbol=symbol,
                data_type=DataType.TRADES,
                timestamp=datetime.fromtimestamp(data["T"] / 1000),
                data=trade_data,
                provider=self.name,
                quality=DataQuality.HIGH
            )

            # Envoyer aux callbacks
            for callback in self._data_callbacks:
                callback(data_point)

        except Exception as e:
            self.logger.error(f"Error processing trade data: {e}")

    def _build_stream_names(self) -> List[str]:
        """Construire les noms de streams pour WebSocket"""
        streams = []

        for symbol, data_types in self._subscriptions.items():
            symbol_lower = symbol.lower()

            for data_type in data_types:
                if data_type == DataType.TICKER:
                    streams.append(f"{symbol_lower}@ticker")
                elif data_type == DataType.ORDERBOOK:
                    streams.append(f"{symbol_lower}@depth20@100ms")
                elif data_type == DataType.TRADES:
                    streams.append(f"{symbol_lower}@aggTrade")

        return streams

    def _normalize_symbol(self, symbol: str) -> str:
        """Normaliser un symbole pour Binance (ex: BTC-USD -> BTCUSDT)"""
        # Supprimer les tirets et convertir en majuscules
        normalized = symbol.replace("-", "").replace("/", "").upper()

        # Remplacer USD par USDT si nécessaire
        if normalized.endswith("USD") and not normalized.endswith("USDT"):
            normalized = normalized[:-3] + "USDT"

        return normalized

    def _denormalize_symbol(self, binance_symbol: str) -> str:
        """Convertir un symbole Binance vers le format standard"""
        # Pour l'instant, retourner tel quel
        # On pourrait ajouter une logique de mapping plus sophistiquée
        return binance_symbol

    async def get_exchange_info(self) -> Dict[str, Any]:
        """Récupérer les informations sur l'exchange"""
        if not self._session:
            return {}

        try:
            url = f"{self.base_url}/api/v3/exchangeInfo"
            async with self._session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                return {}
        except Exception as e:
            self.logger.error(f"Error getting exchange info: {e}")
            return {}

    async def get_symbols(self) -> List[str]:
        """Récupérer la liste des symboles disponibles"""
        exchange_info = await self.get_exchange_info()
        symbols = []

        for symbol_info in exchange_info.get("symbols", []):
            if symbol_info.get("status") == "TRADING":
                symbols.append(symbol_info["symbol"])

        return symbols