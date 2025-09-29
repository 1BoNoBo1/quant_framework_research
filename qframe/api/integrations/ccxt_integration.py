"""
🔗 CCXT Integration
Intégration avec la bibliothèque CCXT pour les exchanges
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import ccxt.async_support as ccxt
from dataclasses import dataclass

from qframe.core.interfaces import DataProvider, OrderExecutor
from qframe.core.container import injectable
from qframe.api.services.base_service import BaseService

logger = logging.getLogger(__name__)


@dataclass
class ExchangeConfig:
    """Configuration pour un exchange."""
    name: str
    api_key: Optional[str] = None
    secret: Optional[str] = None
    password: Optional[str] = None  # Pour certains exchanges
    sandbox: bool = True
    rate_limit: int = 1200  # requests per minute
    timeout: int = 30000  # milliseconds
    enable_rate_limit: bool = True


@injectable
class CCXTIntegration(BaseService, DataProvider, OrderProvider):
    """Intégration CCXT pour les exchanges crypto."""

    def __init__(self):
        super().__init__()

        # Exchanges configurés
        self._exchanges: Dict[str, ccxt.Exchange] = {}
        self._exchange_configs: Dict[str, ExchangeConfig] = {}

        # Cache pour les données
        self._symbols_cache: Dict[str, List[str]] = {}
        self._markets_cache: Dict[str, Dict] = {}

        # Configuration par défaut
        self._default_exchange = "binance"

        # Statistiques
        self._stats = {
            "requests_made": 0,
            "requests_failed": 0,
            "last_request": None,
            "active_exchanges": 0
        }

    async def start(self):
        """Démarre l'intégration CCXT."""
        logger.info("Starting CCXT Integration...")

        self._start_time = datetime.now()

        # Configurer les exchanges par défaut
        await self._initialize_default_exchanges()

        self._is_running = True
        logger.info(f"CCXT Integration started with {len(self._exchanges)} exchanges")

    async def stop(self):
        """Arrête l'intégration CCXT."""
        logger.info("Stopping CCXT Integration...")

        self._is_running = False

        # Fermer toutes les connexions aux exchanges
        for exchange_name, exchange in self._exchanges.items():
            try:
                await exchange.close()
                logger.info(f"Closed connection to {exchange_name}")
            except Exception as e:
                logger.error(f"Error closing {exchange_name}: {e}")

        self._exchanges.clear()
        logger.info("CCXT Integration stopped")

    async def add_exchange(self, config: ExchangeConfig) -> bool:
        """Ajoute un exchange à l'intégration."""
        try:
            # Créer l'instance de l'exchange
            exchange_class = getattr(ccxt, config.name.lower())

            exchange_params = {
                'enableRateLimit': config.enable_rate_limit,
                'timeout': config.timeout,
                'sandbox': config.sandbox,
            }

            # Ajouter les credentials si fournis
            if config.api_key:
                exchange_params['apiKey'] = config.api_key
            if config.secret:
                exchange_params['secret'] = config.secret
            if config.password:
                exchange_params['password'] = config.password

            exchange = exchange_class(exchange_params)

            # Tester la connexion
            await exchange.load_markets()

            # Stocker l'exchange
            self._exchanges[config.name] = exchange
            self._exchange_configs[config.name] = config

            logger.info(f"Successfully added exchange: {config.name}")
            self._stats["active_exchanges"] = len(self._exchanges)

            return True

        except Exception as e:
            logger.error(f"Failed to add exchange {config.name}: {e}")
            return False

    async def remove_exchange(self, exchange_name: str) -> bool:
        """Supprime un exchange."""
        try:
            if exchange_name in self._exchanges:
                await self._exchanges[exchange_name].close()
                del self._exchanges[exchange_name]
                del self._exchange_configs[exchange_name]

                self._stats["active_exchanges"] = len(self._exchanges)
                logger.info(f"Removed exchange: {exchange_name}")
                return True
            return False

        except Exception as e:
            logger.error(f"Error removing exchange {exchange_name}: {e}")
            return False

    # Implémentation de l'interface DataProvider
    async def get_supported_symbols(self, exchange: Optional[str] = None) -> List[str]:
        """Récupère les symboles supportés."""
        exchange_name = exchange or self._default_exchange

        if exchange_name not in self._exchanges:
            logger.error(f"Exchange {exchange_name} not configured")
            return []

        # Vérifier le cache
        if exchange_name in self._symbols_cache:
            return self._symbols_cache[exchange_name]

        try:
            exchange_obj = self._exchanges[exchange_name]
            await exchange_obj.load_markets()

            symbols = list(exchange_obj.markets.keys())
            self._symbols_cache[exchange_name] = symbols

            self._update_stats()
            return symbols

        except Exception as e:
            logger.error(f"Error fetching symbols from {exchange_name}: {e}")
            self._stats["requests_failed"] += 1
            return []

    async def get_current_price(self, symbol: str, exchange: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Récupère le prix actuel d'un symbole."""
        exchange_name = exchange or self._default_exchange

        if exchange_name not in self._exchanges:
            logger.error(f"Exchange {exchange_name} not configured")
            return None

        try:
            exchange_obj = self._exchanges[exchange_name]
            ticker = await exchange_obj.fetch_ticker(symbol)

            self._update_stats()

            return {
                "symbol": symbol,
                "price": ticker['last'],
                "change_24h": ticker['percentage'],
                "volume_24h": ticker['quoteVolume'],
                "timestamp": datetime.now(),
                "high_24h": ticker['high'],
                "low_24h": ticker['low'],
                "open_24h": ticker['open'],
                "bid": ticker['bid'],
                "ask": ticker['ask'],
                "exchange": exchange_name
            }

        except Exception as e:
            logger.error(f"Error fetching price for {symbol} from {exchange_name}: {e}")
            self._stats["requests_failed"] += 1
            return None

    async def get_ohlcv_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        exchange: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Récupère les données OHLCV."""
        exchange_name = exchange or self._default_exchange

        if exchange_name not in self._exchanges:
            logger.error(f"Exchange {exchange_name} not configured")
            return []

        try:
            exchange_obj = self._exchanges[exchange_name]

            # Convertir les dates en timestamps
            since = None
            if start_date:
                since = int(start_date.timestamp() * 1000)

            # Récupérer les données OHLCV
            ohlcv = await exchange_obj.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit
            )

            self._update_stats()

            # Convertir au format standard
            result = []
            for candle in ohlcv:
                timestamp_ms, open_price, high, low, close, volume = candle

                # Filtrer par date de fin si spécifiée
                candle_time = datetime.fromtimestamp(timestamp_ms / 1000)
                if end_date and candle_time > end_date:
                    break

                result.append({
                    "timestamp": candle_time,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume
                })

            return result

        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol} from {exchange_name}: {e}")
            self._stats["requests_failed"] += 1
            return []

    async def get_ticker_data(self, symbol: str, exchange: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Récupère les données ticker complètes."""
        return await self.get_current_price(symbol, exchange)

    async def get_order_book(self, symbol: str, limit: int = 20, exchange: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Récupère le carnet d'ordres."""
        exchange_name = exchange or self._default_exchange

        if exchange_name not in self._exchanges:
            logger.error(f"Exchange {exchange_name} not configured")
            return None

        try:
            exchange_obj = self._exchanges[exchange_name]
            order_book = await exchange_obj.fetch_order_book(symbol, limit)

            self._update_stats()

            return {
                "symbol": symbol,
                "bids": order_book['bids'],
                "asks": order_book['asks'],
                "timestamp": datetime.now(),
                "exchange": exchange_name
            }

        except Exception as e:
            logger.error(f"Error fetching order book for {symbol} from {exchange_name}: {e}")
            self._stats["requests_failed"] += 1
            return None

    async def get_recent_trades(self, symbol: str, limit: int = 50, exchange: Optional[str] = None) -> List[Dict[str, Any]]:
        """Récupère les trades récents."""
        exchange_name = exchange or self._default_exchange

        if exchange_name not in self._exchanges:
            logger.error(f"Exchange {exchange_name} not configured")
            return []

        try:
            exchange_obj = self._exchanges[exchange_name]
            trades = await exchange_obj.fetch_trades(symbol, limit=limit)

            self._update_stats()

            result = []
            for trade in trades:
                result.append({
                    "id": trade['id'],
                    "price": trade['price'],
                    "quantity": trade['amount'],
                    "side": trade['side'],
                    "timestamp": datetime.fromtimestamp(trade['timestamp'] / 1000),
                    "exchange": exchange_name
                })

            return result

        except Exception as e:
            logger.error(f"Error fetching trades for {symbol} from {exchange_name}: {e}")
            self._stats["requests_failed"] += 1
            return []

    # Implémentation de l'interface OrderProvider
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        params: Optional[Dict] = None,
        exchange: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Crée un ordre sur l'exchange."""
        exchange_name = exchange or self._default_exchange

        if exchange_name not in self._exchanges:
            logger.error(f"Exchange {exchange_name} not configured")
            return None

        try:
            exchange_obj = self._exchanges[exchange_name]

            # Créer l'ordre
            order = await exchange_obj.create_order(
                symbol=symbol,
                type=order_type.lower(),
                side=side.lower(),
                amount=quantity,
                price=price,
                params=params or {}
            )

            self._update_stats()

            return {
                "id": order['id'],
                "symbol": order['symbol'],
                "side": order['side'].upper(),
                "type": order['type'].upper(),
                "quantity": order['amount'],
                "price": order['price'],
                "filled_quantity": order['filled'],
                "status": order['status'].upper(),
                "created_at": datetime.fromtimestamp(order['timestamp'] / 1000),
                "updated_at": datetime.now(),
                "exchange": exchange_name,
                "fees": order.get('fees'),
                "raw_response": order
            }

        except Exception as e:
            logger.error(f"Error creating order on {exchange_name}: {e}")
            self._stats["requests_failed"] += 1
            return None

    async def cancel_order(self, order_id: str, symbol: str, exchange: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Annule un ordre."""
        exchange_name = exchange or self._default_exchange

        if exchange_name not in self._exchanges:
            logger.error(f"Exchange {exchange_name} not configured")
            return None

        try:
            exchange_obj = self._exchanges[exchange_name]

            cancelled_order = await exchange_obj.cancel_order(order_id, symbol)

            self._update_stats()

            return {
                "id": cancelled_order['id'],
                "symbol": cancelled_order['symbol'],
                "status": "CANCELLED",
                "updated_at": datetime.now(),
                "exchange": exchange_name,
                "raw_response": cancelled_order
            }

        except Exception as e:
            logger.error(f"Error cancelling order {order_id} on {exchange_name}: {e}")
            self._stats["requests_failed"] += 1
            return None

    async def get_order(self, order_id: str, symbol: str, exchange: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Récupère les détails d'un ordre."""
        exchange_name = exchange or self._default_exchange

        if exchange_name not in self._exchanges:
            logger.error(f"Exchange {exchange_name} not configured")
            return None

        try:
            exchange_obj = self._exchanges[exchange_name]

            order = await exchange_obj.fetch_order(order_id, symbol)

            self._update_stats()

            return {
                "id": order['id'],
                "symbol": order['symbol'],
                "side": order['side'].upper(),
                "type": order['type'].upper(),
                "quantity": order['amount'],
                "price": order['price'],
                "filled_quantity": order['filled'],
                "status": order['status'].upper(),
                "created_at": datetime.fromtimestamp(order['timestamp'] / 1000),
                "updated_at": datetime.fromtimestamp(order['lastTradeTimestamp'] / 1000) if order['lastTradeTimestamp'] else datetime.now(),
                "exchange": exchange_name,
                "fees": order.get('fees'),
                "raw_response": order
            }

        except Exception as e:
            logger.error(f"Error fetching order {order_id} from {exchange_name}: {e}")
            self._stats["requests_failed"] += 1
            return None

    async def get_open_orders(self, symbol: Optional[str] = None, exchange: Optional[str] = None) -> List[Dict[str, Any]]:
        """Récupère les ordres ouverts."""
        exchange_name = exchange or self._default_exchange

        if exchange_name not in self._exchanges:
            logger.error(f"Exchange {exchange_name} not configured")
            return []

        try:
            exchange_obj = self._exchanges[exchange_name]

            orders = await exchange_obj.fetch_open_orders(symbol)

            self._update_stats()

            result = []
            for order in orders:
                result.append({
                    "id": order['id'],
                    "symbol": order['symbol'],
                    "side": order['side'].upper(),
                    "type": order['type'].upper(),
                    "quantity": order['amount'],
                    "price": order['price'],
                    "filled_quantity": order['filled'],
                    "status": order['status'].upper(),
                    "created_at": datetime.fromtimestamp(order['timestamp'] / 1000),
                    "exchange": exchange_name
                })

            return result

        except Exception as e:
            logger.error(f"Error fetching open orders from {exchange_name}: {e}")
            self._stats["requests_failed"] += 1
            return []

    async def get_balance(self, exchange: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Récupère le solde du compte."""
        exchange_name = exchange or self._default_exchange

        if exchange_name not in self._exchanges:
            logger.error(f"Exchange {exchange_name} not configured")
            return None

        try:
            exchange_obj = self._exchanges[exchange_name]

            balance = await exchange_obj.fetch_balance()

            self._update_stats()

            return {
                "exchange": exchange_name,
                "timestamp": datetime.now(),
                "free": balance['free'],
                "used": balance['used'],
                "total": balance['total'],
                "raw_response": balance
            }

        except Exception as e:
            logger.error(f"Error fetching balance from {exchange_name}: {e}")
            self._stats["requests_failed"] += 1
            return None

    # Méthodes utilitaires
    async def get_exchange_info(self, exchange_name: Optional[str] = None) -> Dict[str, Any]:
        """Récupère les informations sur un exchange."""
        if exchange_name and exchange_name in self._exchanges:
            exchanges_to_check = [exchange_name]
        else:
            exchanges_to_check = list(self._exchanges.keys())

        result = {}

        for name in exchanges_to_check:
            exchange = self._exchanges[name]
            config = self._exchange_configs[name]

            try:
                await exchange.load_markets()

                result[name] = {
                    "name": name,
                    "status": "connected",
                    "sandbox": config.sandbox,
                    "rate_limit": exchange.rateLimit,
                    "markets_count": len(exchange.markets),
                    "has_features": {
                        "fetchTicker": exchange.has['fetchTicker'],
                        "fetchOHLCV": exchange.has['fetchOHLCV'],
                        "fetchOrderBook": exchange.has['fetchOrderBook'],
                        "fetchTrades": exchange.has['fetchTrades'],
                        "createOrder": exchange.has['createOrder'],
                        "cancelOrder": exchange.has['cancelOrder'],
                        "fetchBalance": exchange.has['fetchBalance']
                    }
                }

            except Exception as e:
                result[name] = {
                    "name": name,
                    "status": "error",
                    "error": str(e)
                }

        return result

    async def get_statistics(self) -> Dict[str, Any]:
        """Récupère les statistiques de l'intégration."""
        return {
            **self._stats,
            "exchanges": list(self._exchanges.keys()),
            "uptime": self.get_uptime(),
            "success_rate": self._calculate_success_rate()
        }

    async def _initialize_default_exchanges(self):
        """Initialise les exchanges par défaut."""
        # Configuration par défaut pour Binance en mode sandbox
        binance_config = ExchangeConfig(
            name="binance",
            sandbox=True,  # Mode testnet par défaut
            rate_limit=1200
        )

        await self.add_exchange(binance_config)

    def _update_stats(self):
        """Met à jour les statistiques."""
        self._stats["requests_made"] += 1
        self._stats["last_request"] = datetime.now()

    def _calculate_success_rate(self) -> float:
        """Calcule le taux de succès des requêtes."""
        total_requests = self._stats["requests_made"]
        if total_requests == 0:
            return 100.0

        failed_requests = self._stats["requests_failed"]
        return ((total_requests - failed_requests) / total_requests) * 100