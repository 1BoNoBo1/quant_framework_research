"""
Infrastructure Layer: Coinbase Market Data Provider
==================================================

Provider pour récupérer les données de marché en temps réel depuis Coinbase Pro
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

from ..observability.logging import LoggerFactory
from .market_data_pipeline import (
    DataProvider,
    DataType,
    DataQuality,
    MarketDataPoint,
    TickerData,
    OrderBookData
)


class CoinbaseProvider(DataProvider):
    """
    Provider pour données de marché Coinbase Pro via WebSocket et REST API.
    """

    def __init__(
        self,
        base_url: str = "https://api.exchange.coinbase.com",
        ws_url: str = "wss://ws-feed.exchange.coinbase.com",
        sandbox: bool = False
    ):
        self._name = "coinbase"
        self.base_url = base_url
        self.ws_url = ws_url
        self.sandbox = sandbox

        # URLs pour sandbox
        if sandbox:
            self.base_url = "https://api-public.sandbox.exchange.coinbase.com"
            self.ws_url = "wss://ws-feed-public.sandbox.exchange.coinbase.com"

        self._connected = False
        self._subscriptions: Dict[str, Set[DataType]] = {}
        self._websocket: Optional[websockets.WebSocketServerProtocol] = None
        self._data_callbacks: List[Callable[[MarketDataPoint], None]] = []
        self._listen_task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._sequence_numbers: Dict[str, int] = {}

        self.logger = LoggerFactory.get_logger(__name__)

    @property
    def name(self) -> str:
        return self._name

    async def connect(self) -> bool:
        """Se connecter à Coinbase Pro"""
        try:
            # Créer une session HTTP
            self._session = aiohttp.ClientSession()

            # Tester la connectivité REST
            ping_url = f"{self.base_url}/time"
            async with self._session.get(ping_url) as response:
                if response.status != 200:
                    self.logger.error(f"Failed to connect to Coinbase API: {response.status}")
                    return False

                server_time = await response.json()
                self.logger.debug(f"Coinbase server time: {server_time}")

            self._connected = True
            self.logger.info(f"Coinbase provider connected successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error connecting to Coinbase: {e}")
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

        self.logger.info("Coinbase provider disconnected")

    async def subscribe(self, symbol: str, data_types: List[DataType]) -> bool:
        """S'abonner aux données d'un symbole"""
        if not self._connected:
            return False

        try:
            # Normaliser le symbole pour Coinbase (ex: BTC-USD)
            coinbase_symbol = self._normalize_symbol(symbol)

            # Ajouter aux subscriptions
            if coinbase_symbol not in self._subscriptions:
                self._subscriptions[coinbase_symbol] = set()
            self._subscriptions[coinbase_symbol].update(data_types)

            # Démarrer WebSocket si nécessaire
            if not self._listen_task:
                await self._start_websocket()
            else:
                # Mettre à jour les subscriptions WebSocket
                await self._update_subscriptions()

            self.logger.info(f"Subscribed to {symbol} ({coinbase_symbol}) for {data_types}")
            return True

        except Exception as e:
            self.logger.error(f"Error subscribing to {symbol}: {e}")
            return False

    async def unsubscribe(self, symbol: str, data_types: List[DataType]) -> bool:
        """Se désabonner des données"""
        coinbase_symbol = self._normalize_symbol(symbol)

        if coinbase_symbol in self._subscriptions:
            for data_type in data_types:
                self._subscriptions[coinbase_symbol].discard(data_type)

            # Supprimer si plus de subscriptions
            if not self._subscriptions[coinbase_symbol]:
                del self._subscriptions[coinbase_symbol]

            # Mettre à jour les subscriptions WebSocket
            if self._websocket:
                await self._update_subscriptions()

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
            coinbase_symbol = self._normalize_symbol(symbol)
            data_points = []

            if data_type == DataType.CANDLES:
                # Récupérer les candles (chandelles)
                url = f"{self.base_url}/products/{coinbase_symbol}/candles"
                params = {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "granularity": 60  # 1 minute
                }

                async with self._session.get(url, params=params) as response:
                    if response.status == 200:
                        candles = await response.json()
                        for candle in candles:
                            # Format Coinbase: [timestamp, low, high, open, close, volume]
                            data_point = MarketDataPoint(
                                symbol=symbol,
                                data_type=DataType.CANDLES,
                                timestamp=datetime.fromtimestamp(candle[0]),
                                data={
                                    "open": float(candle[3]),
                                    "high": float(candle[2]),
                                    "low": float(candle[1]),
                                    "close": float(candle[4]),
                                    "volume": float(candle[5])
                                },
                                provider=self.name,
                                quality=DataQuality.HIGH
                            )
                            data_points.append(data_point)

            elif data_type == DataType.TRADES:
                # Récupérer les trades récents
                url = f"{self.base_url}/products/{coinbase_symbol}/trades"

                async with self._session.get(url) as response:
                    if response.status == 200:
                        trades = await response.json()
                        for trade in trades:
                            trade_time = datetime.fromisoformat(trade["time"].replace("Z", "+00:00"))

                            # Filtrer par période
                            if start_time <= trade_time <= end_time:
                                data_point = MarketDataPoint(
                                    symbol=symbol,
                                    data_type=DataType.TRADES,
                                    timestamp=trade_time,
                                    data={
                                        "price": float(trade["price"]),
                                        "size": float(trade["size"]),
                                        "trade_id": int(trade["trade_id"]),
                                        "side": trade["side"]
                                    },
                                    provider=self.name,
                                    quality=DataQuality.HIGH
                                )
                                data_points.append(data_point)

            # Trier par timestamp
            data_points.sort(key=lambda x: x.timestamp)
            return data_points

        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return []

    async def _start_websocket(self):
        """Démarrer la connexion WebSocket"""
        try:
            self.logger.info(f"Connecting to Coinbase WebSocket: {self.ws_url}")

            self._websocket = await websockets.connect(self.ws_url)

            # Envoyer les subscriptions initiales
            await self._update_subscriptions()

            self._listen_task = asyncio.create_task(self._listen_websocket())

        except Exception as e:
            self.logger.error(f"Error starting WebSocket: {e}")

    async def _update_subscriptions(self):
        """Mettre à jour les subscriptions WebSocket"""
        if not self._websocket:
            return

        try:
            # Construire les channels selon les subscriptions
            channels = []
            product_ids = list(self._subscriptions.keys())

            for symbol, data_types in self._subscriptions.items():
                for data_type in data_types:
                    if data_type == DataType.TICKER:
                        if {"name": "ticker", "product_ids": product_ids} not in channels:
                            channels.append({"name": "ticker", "product_ids": product_ids})
                    elif data_type == DataType.ORDERBOOK:
                        channels.append({"name": "level2", "product_ids": [symbol]})
                    elif data_type == DataType.TRADES:
                        if {"name": "matches", "product_ids": product_ids} not in channels:
                            channels.append({"name": "matches", "product_ids": product_ids})

            # Envoyer la subscription
            if channels:
                subscribe_msg = {
                    "type": "subscribe",
                    "channels": channels
                }

                await self._websocket.send(json.dumps(subscribe_msg))
                self.logger.debug(f"Sent subscription: {subscribe_msg}")

        except Exception as e:
            self.logger.error(f"Error updating subscriptions: {e}")

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
            self.logger.info("Coinbase WebSocket connection closed")
        except Exception as e:
            self.logger.error(f"Coinbase WebSocket error: {e}")

    async def _process_websocket_message(self, data: Dict[str, Any]):
        """Traiter un message WebSocket"""
        try:
            msg_type = data.get("type")

            if msg_type == "subscriptions":
                self.logger.info("WebSocket subscriptions confirmed")
            elif msg_type == "ticker":
                await self._process_ticker_data(data)
            elif msg_type == "snapshot" or msg_type == "l2update":
                await self._process_orderbook_data(data)
            elif msg_type == "match":
                await self._process_trade_data(data)
            elif msg_type == "error":
                self.logger.error(f"WebSocket error: {data.get('message')}")

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

    async def _process_ticker_data(self, data: Dict[str, Any]):
        """Traiter les données ticker"""
        try:
            symbol = self._denormalize_symbol(data["product_id"])

            ticker_data = TickerData(
                symbol=symbol,
                bid=Decimal(data["best_bid"]),
                ask=Decimal(data["best_ask"]),
                last=Decimal(data["price"]),
                volume_24h=Decimal(data["volume_24h"]),
                change_24h=Decimal("0"),  # Coinbase ne fournit pas le changement 24h dans ticker
                high_24h=Decimal(data.get("high_24h", data["price"])),
                low_24h=Decimal(data.get("low_24h", data["price"])),
                timestamp=datetime.fromisoformat(data["time"].replace("Z", "+00:00"))
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
            symbol = self._denormalize_symbol(data["product_id"])

            if data["type"] == "snapshot":
                # Snapshot complet
                bids = [[Decimal(bid[0]), Decimal(bid[1])] for bid in data["bids"]]
                asks = [[Decimal(ask[0]), Decimal(ask[1])] for ask in data["asks"]]
            else:
                # Update incrémental - pour simplifier, on skip (nécessiterait un state management)
                return

            orderbook_data = OrderBookData(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=datetime.utcnow(),  # Coinbase ne fournit pas toujours le timestamp
                sequence=None
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
            symbol = self._denormalize_symbol(data["product_id"])

            trade_data = {
                "price": float(data["price"]),
                "size": float(data["size"]),
                "trade_id": int(data["trade_id"]),
                "side": data["side"],
                "maker_order_id": data.get("maker_order_id"),
                "taker_order_id": data.get("taker_order_id")
            }

            data_point = MarketDataPoint(
                symbol=symbol,
                data_type=DataType.TRADES,
                timestamp=datetime.fromisoformat(data["time"].replace("Z", "+00:00")),
                data=trade_data,
                provider=self.name,
                quality=DataQuality.HIGH
            )

            # Envoyer aux callbacks
            for callback in self._data_callbacks:
                callback(data_point)

        except Exception as e:
            self.logger.error(f"Error processing trade data: {e}")

    def _normalize_symbol(self, symbol: str) -> str:
        """Normaliser un symbole pour Coinbase (ex: BTC/USD -> BTC-USD)"""
        return symbol.replace("/", "-").upper()

    def _denormalize_symbol(self, coinbase_symbol: str) -> str:
        """Convertir un symbole Coinbase vers le format standard"""
        return coinbase_symbol.replace("-", "/")

    async def get_products(self) -> List[Dict[str, Any]]:
        """Récupérer la liste des produits disponibles"""
        if not self._session:
            return []

        try:
            url = f"{self.base_url}/products"
            async with self._session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                return []
        except Exception as e:
            self.logger.error(f"Error getting products: {e}")
            return []

    async def get_symbols(self) -> List[str]:
        """Récupérer la liste des symboles disponibles"""
        products = await self.get_products()
        symbols = []

        for product in products:
            if product.get("status") == "online" and not product.get("cancel_only", False):
                symbols.append(product["id"])

        return symbols