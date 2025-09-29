"""
📊 Market Data Service
Service pour les données de marché temps réel
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass

from qframe.core.interfaces import DataProvider
from qframe.core.container import injectable
from qframe.api.services.base_service import BaseService

logger = logging.getLogger(__name__)


@dataclass
class MarketDataConfig:
    """Configuration du service de données de marché."""
    update_interval: int = 1  # secondes
    max_history_points: int = 1000
    supported_timeframes: List[str] = None
    cache_ttl: int = 5  # secondes

    def __post_init__(self):
        if self.supported_timeframes is None:
            self.supported_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]


@injectable
class MarketDataService(BaseService):
    """Service de données de marché avec cache et simulation temps réel."""

    def __init__(self):
        super().__init__()
        self.data_provider = None  # À injecter si nécessaire
        self.config = MarketDataConfig()

        # Cache des données
        self._price_cache: Dict[str, Dict] = {}
        self._ohlcv_cache: Dict[str, List[Dict]] = {}
        self._ticker_cache: Dict[str, Dict] = {}

        # Dernière mise à jour
        self._last_update: Dict[str, datetime] = {}

        # Simulation de données
        self._simulation_data: Dict[str, Any] = {}
        self._is_simulation = True  # Activer la simulation par défaut

        # Task de mise à jour en arrière-plan
        self._update_task: Optional[asyncio.Task] = None

    async def start(self):
        """Démarre le service de données de marché."""
        logger.info("Starting Market Data Service...")

        # Initialiser les données de simulation
        await self._initialize_simulation_data()

        # Démarrer la tâche de mise à jour en arrière-plan
        self._update_task = asyncio.create_task(self._background_update_loop())

        self._is_running = True
        logger.info("Market Data Service started successfully")

    async def stop(self):
        """Arrête le service de données de marché."""
        logger.info("Stopping Market Data Service...")

        self._is_running = False

        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

        logger.info("Market Data Service stopped")

    async def get_supported_symbols(self, exchange: Optional[str] = None) -> List[str]:
        """Récupère la liste des symboles supportés."""
        # Symboles de simulation par défaut
        symbols = [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
            "SOL/USDT", "DOT/USDT", "MATIC/USDT", "AVAX/USDT", "LINK/USDT"
        ]

        if self.data_provider and not self._is_simulation:
            try:
                symbols = await self.data_provider.get_supported_symbols(exchange)
            except Exception as e:
                logger.warning(f"Failed to get symbols from provider, using simulation: {e}")

        return symbols

    async def get_current_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Récupère le prix actuel d'un symbole."""
        # Vérifier le cache
        if symbol in self._price_cache:
            cached_data = self._price_cache[symbol]
            cache_time = self._last_update.get(symbol, datetime.min)

            if (datetime.now() - cache_time).total_seconds() < self.config.cache_ttl:
                return cached_data

        # Récupérer les nouvelles données
        if self._is_simulation:
            price_data = await self._simulate_current_price(symbol)
        else:
            price_data = await self._fetch_real_price(symbol)

        if price_data:
            self._price_cache[symbol] = price_data
            self._last_update[symbol] = datetime.now()

        return price_data

    async def get_multiple_prices(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Récupère les prix de plusieurs symboles."""
        tasks = [self.get_current_price(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        prices = []
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching price for {symbol}: {result}")
            elif result:
                prices.append(result)

        return prices

    async def get_ohlcv_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Récupère les données OHLCV pour un symbole."""
        cache_key = f"{symbol}_{timeframe}"

        if self._is_simulation:
            return await self._simulate_ohlcv_data(symbol, timeframe, start_date, end_date, limit)
        else:
            return await self._fetch_real_ohlcv(symbol, timeframe, start_date, end_date, limit)

    async def get_ticker_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Récupère les données ticker complètes."""
        if self._is_simulation:
            return await self._simulate_ticker_data(symbol)
        else:
            return await self._fetch_real_ticker(symbol)

    async def get_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict[str, Any]]:
        """Récupère le carnet d'ordres."""
        if self._is_simulation:
            return await self._simulate_order_book(symbol, limit)
        else:
            return await self._fetch_real_order_book(symbol, limit)

    async def get_recent_trades(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Récupère les trades récents."""
        if self._is_simulation:
            return await self._simulate_recent_trades(symbol, limit)
        else:
            return await self._fetch_real_trades(symbol, limit)

    async def get_supported_exchanges(self) -> List[str]:
        """Récupère la liste des exchanges supportés."""
        if self._is_simulation:
            return ["simulation", "binance", "coinbase", "kraken"]
        else:
            # Logique pour exchanges réels
            return ["binance", "coinbase", "kraken", "bitfinex"]

    async def get_market_status(self) -> Dict[str, Any]:
        """Récupère le statut des marchés."""
        return {
            "status": "open",
            "trading_enabled": True,
            "last_update": datetime.now(),
            "server_time": datetime.now(),
            "maintenance": False,
            "rate_limits": {
                "requests_per_minute": 1200,
                "requests_remaining": 1150
            }
        }

    async def get_bulk_data(
        self,
        symbols: List[str],
        timeframe: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Récupère des données en bulk."""
        result = {}

        # Prix actuels
        prices = await self.get_multiple_prices(symbols)
        result["prices"] = {price["symbol"]: price for price in prices}

        # Données OHLCV si demandées
        if timeframe:
            ohlcv_tasks = [
                self.get_ohlcv_data(symbol, timeframe, start_date, end_date, limit or 100)
                for symbol in symbols
            ]
            ohlcv_results = await asyncio.gather(*ohlcv_tasks, return_exceptions=True)

            result["ohlcv"] = {}
            for symbol, ohlcv_data in zip(symbols, ohlcv_results):
                if not isinstance(ohlcv_data, Exception):
                    result["ohlcv"][symbol] = ohlcv_data

        return result

    async def get_market_stats(self, symbol: str, period: str) -> Optional[Dict[str, Any]]:
        """Récupère les statistiques de marché."""
        current_price_data = await self.get_current_price(symbol)
        if not current_price_data:
            return None

        # Simulation des statistiques
        current_price = current_price_data["price"]

        return {
            "symbol": symbol,
            "period": period,
            "current_price": current_price,
            "high": current_price * 1.15,
            "low": current_price * 0.92,
            "volume": np.random.uniform(1000000, 10000000),
            "volume_change": np.random.uniform(-20, 20),
            "price_change": current_price_data["change_24h"],
            "volatility": np.random.uniform(0.02, 0.08),
            "market_cap": current_price * np.random.uniform(10000000, 1000000000),
            "calculated_at": datetime.now()
        }

    # Méthodes de simulation
    async def _initialize_simulation_data(self):
        """Initialise les données de simulation."""
        symbols = await self.get_supported_symbols()

        # Prix de base pour la simulation
        base_prices = {
            "BTC/USDT": 43250.0,
            "ETH/USDT": 2650.0,
            "BNB/USDT": 235.0,
            "XRP/USDT": 0.52,
            "ADA/USDT": 0.38,
            "SOL/USDT": 95.0,
            "DOT/USDT": 5.2,
            "MATIC/USDT": 0.72,
            "AVAX/USDT": 28.5,
            "LINK/USDT": 14.8
        }

        for symbol in symbols:
            if symbol not in self._simulation_data:
                base_price = base_prices.get(symbol, 100.0)
                self._simulation_data[symbol] = {
                    "base_price": base_price,
                    "current_price": base_price,
                    "volatility": np.random.uniform(0.01, 0.03),
                    "trend": np.random.uniform(-0.1, 0.1),
                    "last_update": datetime.now()
                }

    async def _simulate_current_price(self, symbol: str) -> Dict[str, Any]:
        """Simule le prix actuel d'un symbole."""
        if symbol not in self._simulation_data:
            await self._initialize_simulation_data()

        sim_data = self._simulation_data[symbol]

        # Évolution du prix avec marche aléatoire + tendance
        volatility = sim_data["volatility"]
        trend = sim_data["trend"]

        change = np.random.normal(trend, volatility)
        new_price = sim_data["current_price"] * (1 + change)

        # Limiter les variations extrêmes
        max_change = 0.05  # 5% max par update
        price_change = max(-max_change, min(max_change, change))
        new_price = sim_data["current_price"] * (1 + price_change)

        # Calculer le changement 24h
        base_price = sim_data["base_price"]
        change_24h = (new_price - base_price) / base_price * 100

        # Mettre à jour les données de simulation
        sim_data["current_price"] = new_price
        sim_data["last_update"] = datetime.now()

        return {
            "symbol": symbol,
            "price": new_price,
            "change_24h": change_24h,
            "volume_24h": np.random.uniform(1000000, 50000000),
            "timestamp": datetime.now(),
            "high_24h": new_price * np.random.uniform(1.02, 1.08),
            "low_24h": new_price * np.random.uniform(0.92, 0.98),
            "open_24h": base_price
        }

    async def _simulate_ohlcv_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Simule des données OHLCV."""
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            # Calculer la date de début selon le timeframe
            timeframe_minutes = {
                "1m": 1, "5m": 5, "15m": 15, "30m": 30,
                "1h": 60, "4h": 240, "1d": 1440, "1w": 10080
            }
            minutes = timeframe_minutes.get(timeframe, 60)
            start_date = end_date - timedelta(minutes=minutes * limit)

        # Générer les bougies
        current_price = 43250.0  # Prix de base
        if symbol in self._simulation_data:
            current_price = self._simulation_data[symbol]["current_price"]

        candles = []
        current_time = start_date
        price = current_price * 0.95  # Commencer un peu plus bas

        timeframe_delta = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "30m": timedelta(minutes=30),
            "1h": timedelta(hours=1),
            "4h": timedelta(hours=4),
            "1d": timedelta(days=1),
            "1w": timedelta(weeks=1)
        }

        delta = timeframe_delta.get(timeframe, timedelta(hours=1))

        while current_time <= end_date and len(candles) < limit:
            # Simuler OHLCV
            open_price = price

            # Variation pour cette bougie
            volatility = 0.02  # 2% de volatilité
            change = np.random.normal(0, volatility)
            close_price = open_price * (1 + change)

            # High et Low
            high_price = max(open_price, close_price) * np.random.uniform(1.0, 1.02)
            low_price = min(open_price, close_price) * np.random.uniform(0.98, 1.0)

            # Volume
            volume = np.random.uniform(100, 1000)

            candles.append({
                "timestamp": current_time,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume
            })

            price = close_price
            current_time += delta

        return candles

    async def _simulate_ticker_data(self, symbol: str) -> Dict[str, Any]:
        """Simule les données ticker."""
        price_data = await self._simulate_current_price(symbol)

        return {
            **price_data,
            "bid": price_data["price"] * 0.9999,
            "ask": price_data["price"] * 1.0001,
            "bid_volume": np.random.uniform(10, 100),
            "ask_volume": np.random.uniform(10, 100),
            "last_price": price_data["price"],
            "last_quantity": np.random.uniform(0.1, 5.0)
        }

    async def _simulate_order_book(self, symbol: str, limit: int) -> Dict[str, Any]:
        """Simule un carnet d'ordres."""
        price_data = await self.get_current_price(symbol)
        if not price_data:
            return {}

        current_price = price_data["price"]

        # Générer les bids (achats)
        bids = []
        for i in range(limit):
            price = current_price * (1 - (i + 1) * 0.0001)
            quantity = np.random.uniform(0.1, 10.0)
            bids.append([price, quantity])

        # Générer les asks (ventes)
        asks = []
        for i in range(limit):
            price = current_price * (1 + (i + 1) * 0.0001)
            quantity = np.random.uniform(0.1, 10.0)
            asks.append([price, quantity])

        return {
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "timestamp": datetime.now()
        }

    async def _simulate_recent_trades(self, symbol: str, limit: int) -> List[Dict[str, Any]]:
        """Simule des trades récents."""
        price_data = await self.get_current_price(symbol)
        if not price_data:
            return []

        current_price = price_data["price"]
        trades = []

        base_time = datetime.now()

        for i in range(limit):
            # Variation de prix légère
            price = current_price * np.random.uniform(0.9995, 1.0005)
            quantity = np.random.uniform(0.01, 5.0)
            side = np.random.choice(["buy", "sell"])
            timestamp = base_time - timedelta(seconds=i * 10)

            trades.append({
                "id": f"trade_{i}",
                "price": price,
                "quantity": quantity,
                "side": side,
                "timestamp": timestamp
            })

        return trades

    async def _background_update_loop(self):
        """Boucle de mise à jour en arrière-plan."""
        while self._is_running:
            try:
                # Mettre à jour les prix simulés
                symbols = await self.get_supported_symbols()

                # Mise à jour des données de simulation
                for symbol in symbols[:5]:  # Limiter pour les performances
                    if symbol in self._simulation_data:
                        await self._simulate_current_price(symbol)

                await asyncio.sleep(self.config.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background update loop: {e}")
                await asyncio.sleep(5)

    # Méthodes pour données réelles (à implémenter avec CCXT)
    async def _fetch_real_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Récupère le prix réel via un provider."""
        if self.data_provider:
            try:
                return await self.data_provider.get_current_price(symbol)
            except Exception as e:
                logger.error(f"Error fetching real price for {symbol}: {e}")
        return None

    async def _fetch_real_ohlcv(self, symbol: str, timeframe: str, start_date, end_date, limit: int):
        """Récupère les données OHLCV réelles."""
        if self.data_provider:
            try:
                return await self.data_provider.get_ohlcv_data(symbol, timeframe, start_date, end_date, limit)
            except Exception as e:
                logger.error(f"Error fetching real OHLCV for {symbol}: {e}")
        return []

    async def _fetch_real_ticker(self, symbol: str):
        """Récupère les données ticker réelles."""
        if self.data_provider:
            try:
                return await self.data_provider.get_ticker_data(symbol)
            except Exception as e:
                logger.error(f"Error fetching real ticker for {symbol}: {e}")
        return None

    async def _fetch_real_order_book(self, symbol: str, limit: int):
        """Récupère le carnet d'ordres réel."""
        if self.data_provider:
            try:
                return await self.data_provider.get_order_book(symbol, limit)
            except Exception as e:
                logger.error(f"Error fetching real order book for {symbol}: {e}")
        return None

    async def _fetch_real_trades(self, symbol: str, limit: int):
        """Récupère les trades réels."""
        if self.data_provider:
            try:
                return await self.data_provider.get_recent_trades(symbol, limit)
            except Exception as e:
                logger.error(f"Error fetching real trades for {symbol}: {e}")
        return []