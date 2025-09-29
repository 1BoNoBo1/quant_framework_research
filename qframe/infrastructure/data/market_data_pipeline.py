"""
Infrastructure Layer: Market Data Pipeline
=========================================

Pipeline complet pour ingestion, normalisation et distribution
des données de marché en temps réel depuis multiple providers.
"""

import asyncio
import json
import time
import websockets
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set, Deque
from threading import Lock
try:
    import aioredis
except ImportError:
    aioredis = None
import logging

from ..observability.logging import LoggerFactory
from ..observability.metrics import get_business_metrics
from ..observability.tracing import get_tracer, trace


class DataType(str, Enum):
    """Types de données de marché"""
    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    TRADES = "trades"
    CANDLES = "candles"
    LIQUIDATIONS = "liquidations"
    FUNDING = "funding"
    INDEX = "index"
    STATS_24H = "stats_24h"


class DataQuality(str, Enum):
    """Niveaux de qualité des données"""
    HIGH = "high"        # Données validées et propres
    MEDIUM = "medium"    # Données partiellement validées
    LOW = "low"          # Données brutes non validées
    CORRUPTED = "corrupted"  # Données corrompues à rejeter


@dataclass
class MarketDataPoint:
    """Point de données de marché normalisé"""
    symbol: str
    data_type: DataType
    timestamp: datetime
    data: Dict[str, Any]
    provider: str
    quality: DataQuality = DataQuality.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convertir en dictionnaire pour sérialisation"""
        return {
            "symbol": self.symbol,
            "data_type": self.data_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "provider": self.provider,
            "quality": self.quality.value,
            "metadata": self.metadata,
            "correlation_id": self.correlation_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketDataPoint':
        """Créer depuis un dictionnaire"""
        return cls(
            symbol=data["symbol"],
            data_type=DataType(data["data_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data["data"],
            provider=data["provider"],
            quality=DataQuality(data.get("quality", "medium")),
            metadata=data.get("metadata", {}),
            correlation_id=data.get("correlation_id")
        )


@dataclass
class TickerData:
    """Données ticker normalisées"""
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume_24h: Decimal
    change_24h: Decimal
    high_24h: Decimal
    low_24h: Decimal
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "bid": float(self.bid),
            "ask": float(self.ask),
            "last": float(self.last),
            "volume_24h": float(self.volume_24h),
            "change_24h": float(self.change_24h),
            "high_24h": float(self.high_24h),
            "low_24h": float(self.low_24h),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class OrderBookData:
    """Données orderbook normalisées"""
    symbol: str
    bids: List[List[Decimal]]  # [[price, size], ...]
    asks: List[List[Decimal]]  # [[price, size], ...]
    timestamp: datetime
    sequence: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "bids": [[float(p), float(s)] for p, s in self.bids],
            "asks": [[float(p), float(s)] for p, s in self.asks],
            "timestamp": self.timestamp.isoformat(),
            "sequence": self.sequence
        }

    def get_best_bid(self) -> Optional[Decimal]:
        """Obtenir le meilleur bid"""
        return self.bids[0][0] if self.bids else None

    def get_best_ask(self) -> Optional[Decimal]:
        """Obtenir le meilleur ask"""
        return self.asks[0][0] if self.asks else None

    def get_spread(self) -> Optional[Decimal]:
        """Calculer le spread"""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid and ask:
            return ask - bid
        return None


class DataProvider(ABC):
    """Interface pour un fournisseur de données"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nom du provider"""
        pass

    @abstractmethod
    async def connect(self) -> bool:
        """Se connecter au provider"""
        pass

    @abstractmethod
    async def disconnect(self):
        """Se déconnecter du provider"""
        pass

    @abstractmethod
    async def subscribe(self, symbol: str, data_types: List[DataType]) -> bool:
        """S'abonner aux données d'un symbole"""
        pass

    @abstractmethod
    async def unsubscribe(self, symbol: str, data_types: List[DataType]) -> bool:
        """Se désabonner des données d'un symbole"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Vérifier la connexion"""
        pass

    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        data_type: DataType,
        start_time: datetime,
        end_time: datetime
    ) -> List[MarketDataPoint]:
        """Récupérer des données historiques"""
        pass


class MockDataProvider(DataProvider):
    """Provider de données fictif pour tests et développement"""

    def __init__(self, name: str = "mock", latency_ms: int = 50):
        self._name = name
        self.latency_ms = latency_ms
        self._connected = False
        self._subscriptions: Set[str] = set()
        self._data_callbacks: List[Callable[[MarketDataPoint], None]] = []
        self._generation_task: Optional[asyncio.Task] = None
        self.logger = LoggerFactory.get_logger(f"{__name__}.{name}")

    @property
    def name(self) -> str:
        return self._name

    async def connect(self) -> bool:
        """Se connecter (simulation)"""
        await asyncio.sleep(self.latency_ms / 1000)
        self._connected = True
        self.logger.info(f"Mock provider {self.name} connected")
        return True

    async def disconnect(self):
        """Se déconnecter"""
        self._connected = False
        if self._generation_task:
            self._generation_task.cancel()
        self.logger.info(f"Mock provider {self.name} disconnected")

    async def subscribe(self, symbol: str, data_types: List[DataType]) -> bool:
        """S'abonner aux données"""
        if not self._connected:
            return False

        key = f"{symbol}:{':'.join(dt.value for dt in data_types)}"
        self._subscriptions.add(key)

        # Démarrer la génération de données si pas déjà fait
        if not self._generation_task:
            self._generation_task = asyncio.create_task(self._generate_data())

        self.logger.info(f"Subscribed to {symbol} for {data_types}")
        return True

    async def unsubscribe(self, symbol: str, data_types: List[DataType]) -> bool:
        """Se désabonner"""
        key = f"{symbol}:{':'.join(dt.value for dt in data_types)}"
        self._subscriptions.discard(key)
        self.logger.info(f"Unsubscribed from {symbol}")
        return True

    def is_connected(self) -> bool:
        return self._connected

    def add_callback(self, callback: Callable[[MarketDataPoint], None]):
        """Ajouter un callback pour les données"""
        self._data_callbacks.append(callback)

    async def get_historical_data(
        self,
        symbol: str,
        data_type: DataType,
        start_time: datetime,
        end_time: datetime
    ) -> List[MarketDataPoint]:
        """Générer des données historiques fictives"""
        data_points = []
        current_time = start_time
        interval = timedelta(minutes=1)

        base_price = Decimal("50000") if "BTC" in symbol else Decimal("100")

        while current_time <= end_time:
            # Générer des données selon le type
            if data_type == DataType.TICKER:
                price_change = (Decimal(str(time.time())) % 100 - 50) / 1000
                current_price = base_price + price_change

                data = TickerData(
                    symbol=symbol,
                    bid=current_price - Decimal("0.5"),
                    ask=current_price + Decimal("0.5"),
                    last=current_price,
                    volume_24h=Decimal("1000000"),
                    change_24h=price_change,
                    high_24h=current_price + Decimal("100"),
                    low_24h=current_price - Decimal("100"),
                    timestamp=current_time
                ).to_dict()

            elif data_type == DataType.TRADES:
                data = {
                    "price": float(base_price),
                    "size": 0.1,
                    "side": "buy",
                    "trade_id": int(current_time.timestamp() * 1000)
                }

            else:
                data = {"mock": True}

            point = MarketDataPoint(
                symbol=symbol,
                data_type=data_type,
                timestamp=current_time,
                data=data,
                provider=self.name,
                quality=DataQuality.HIGH
            )

            data_points.append(point)
            current_time += interval

        return data_points

    async def _generate_data(self):
        """Générer des données en temps réel"""
        try:
            while self._connected and self._subscriptions:
                for subscription in list(self._subscriptions):
                    symbol, data_types_str = subscription.split(":", 1)
                    data_types = [DataType(dt) for dt in data_types_str.split(":")]

                    for data_type in data_types:
                        # Simuler des données
                        if data_type == DataType.TICKER:
                            base_price = Decimal("50000") if "BTC" in symbol else Decimal("100")
                            price_change = (Decimal(str(time.time())) % 100 - 50) / 100

                            data = TickerData(
                                symbol=symbol,
                                bid=base_price - Decimal("0.5"),
                                ask=base_price + Decimal("0.5"),
                                last=base_price + price_change,
                                volume_24h=Decimal("1000000"),
                                change_24h=price_change,
                                high_24h=base_price + Decimal("100"),
                                low_24h=base_price - Decimal("100"),
                                timestamp=datetime.utcnow()
                            ).to_dict()

                        elif data_type == DataType.TRADES:
                            data = {
                                "price": 50000.0,
                                "size": 0.1,
                                "side": "buy",
                                "trade_id": int(time.time() * 1000)
                            }

                        else:
                            continue

                        point = MarketDataPoint(
                            symbol=symbol,
                            data_type=data_type,
                            timestamp=datetime.utcnow(),
                            data=data,
                            provider=self.name,
                            quality=DataQuality.HIGH
                        )

                        # Envoyer aux callbacks
                        for callback in self._data_callbacks:
                            try:
                                callback(point)
                            except Exception as e:
                                self.logger.error(f"Error in data callback: {e}")

                # Attendre avant le prochain cycle
                await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error generating data: {e}")


class DataValidator:
    """Validateur de qualité des données"""

    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)
        self._validation_rules: Dict[DataType, List[Callable]] = {}
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Configurer les règles de validation par défaut"""

        # Règles pour les tickers
        self._validation_rules[DataType.TICKER] = [
            lambda data: "bid" in data and "ask" in data and "last" in data,
            lambda data: Decimal(str(data.get("bid", 0))) > 0,
            lambda data: Decimal(str(data.get("ask", 0))) > 0,
            lambda data: Decimal(str(data.get("ask", 0))) >= Decimal(str(data.get("bid", 0))),  # Ask >= Bid
            lambda data: abs(Decimal(str(data.get("change_24h", 0)))) < Decimal("0.5")  # Change < 50%
        ]

        # Règles pour les trades
        self._validation_rules[DataType.TRADES] = [
            lambda data: "price" in data and "size" in data,
            lambda data: Decimal(str(data.get("price", 0))) > 0,
            lambda data: Decimal(str(data.get("size", 0))) > 0,
            lambda data: data.get("side") in ["buy", "sell"]
        ]

        # Règles pour l'orderbook
        self._validation_rules[DataType.ORDERBOOK] = [
            lambda data: "bids" in data and "asks" in data,
            lambda data: isinstance(data.get("bids"), list),
            lambda data: isinstance(data.get("asks"), list),
            lambda data: len(data.get("bids", [])) > 0 and len(data.get("asks", [])) > 0
        ]

    def validate(self, data_point: MarketDataPoint) -> DataQuality:
        """Valider un point de données"""
        try:
            rules = self._validation_rules.get(data_point.data_type, [])

            failed_rules = 0
            for rule in rules:
                try:
                    if not rule(data_point.data):
                        failed_rules += 1
                except Exception:
                    failed_rules += 1

            # Déterminer la qualité
            if failed_rules == 0:
                return DataQuality.HIGH
            elif failed_rules <= len(rules) // 2:
                return DataQuality.MEDIUM
            elif failed_rules < len(rules):
                return DataQuality.LOW
            else:
                return DataQuality.CORRUPTED

        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return DataQuality.CORRUPTED

    def add_rule(self, data_type: DataType, rule: Callable):
        """Ajouter une règle de validation"""
        if data_type not in self._validation_rules:
            self._validation_rules[data_type] = []
        self._validation_rules[data_type].append(rule)


class DataNormalizer:
    """Normaliseur de données pour uniformiser les formats"""

    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)

    def normalize_ticker(self, raw_data: Dict[str, Any], symbol: str, provider: str) -> Optional[TickerData]:
        """Normaliser des données ticker selon le provider"""
        try:
            if provider == "binance":
                return TickerData(
                    symbol=symbol,
                    bid=Decimal(raw_data.get("bidPrice", "0")),
                    ask=Decimal(raw_data.get("askPrice", "0")),
                    last=Decimal(raw_data.get("lastPrice", "0")),
                    volume_24h=Decimal(raw_data.get("volume", "0")),
                    change_24h=Decimal(raw_data.get("priceChangePercent", "0")) / 100,
                    high_24h=Decimal(raw_data.get("highPrice", "0")),
                    low_24h=Decimal(raw_data.get("lowPrice", "0")),
                    timestamp=datetime.utcnow()
                )
            elif provider == "coinbase":
                return TickerData(
                    symbol=symbol,
                    bid=Decimal(raw_data.get("best_bid", "0")),
                    ask=Decimal(raw_data.get("best_ask", "0")),
                    last=Decimal(raw_data.get("price", "0")),
                    volume_24h=Decimal(raw_data.get("volume_24h", "0")),
                    change_24h=Decimal(raw_data.get("percentage_24h", "0")) / 100,
                    high_24h=Decimal(raw_data.get("high_24h", "0")),
                    low_24h=Decimal(raw_data.get("low_24h", "0")),
                    timestamp=datetime.utcnow()
                )
            else:
                # Format générique
                return TickerData(
                    symbol=symbol,
                    bid=Decimal(str(raw_data.get("bid", 0))),
                    ask=Decimal(str(raw_data.get("ask", 0))),
                    last=Decimal(str(raw_data.get("last", 0))),
                    volume_24h=Decimal(str(raw_data.get("volume_24h", 0))),
                    change_24h=Decimal(str(raw_data.get("change_24h", 0))),
                    high_24h=Decimal(str(raw_data.get("high_24h", 0))),
                    low_24h=Decimal(str(raw_data.get("low_24h", 0))),
                    timestamp=datetime.utcnow()
                )

        except Exception as e:
            self.logger.error(f"Error normalizing ticker data: {e}")
            return None

    def normalize_orderbook(self, raw_data: Dict[str, Any], symbol: str, provider: str) -> Optional[OrderBookData]:
        """Normaliser des données orderbook"""
        try:
            if provider == "binance":
                bids = [[Decimal(p), Decimal(s)] for p, s in raw_data.get("bids", [])]
                asks = [[Decimal(p), Decimal(s)] for p, s in raw_data.get("asks", [])]
            else:
                # Format générique
                bids = [[Decimal(str(item[0])), Decimal(str(item[1]))] for item in raw_data.get("bids", [])]
                asks = [[Decimal(str(item[0])), Decimal(str(item[1]))] for item in raw_data.get("asks", [])]

            return OrderBookData(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=datetime.utcnow(),
                sequence=raw_data.get("sequence")
            )

        except Exception as e:
            self.logger.error(f"Error normalizing orderbook data: {e}")
            return None


class MarketDataCache:
    """Cache en mémoire pour les données de marché"""

    def __init__(self, max_age_seconds: int = 300, max_items: int = 10000):
        self.max_age_seconds = max_age_seconds
        self.max_items = max_items
        self._cache: Dict[str, MarketDataPoint] = {}
        self._access_times: Dict[str, datetime] = {}
        self._lock = Lock()

    def put(self, key: str, data_point: MarketDataPoint):
        """Mettre en cache un point de données"""
        with self._lock:
            self._cache[key] = data_point
            self._access_times[key] = datetime.utcnow()

            # Nettoyer si nécessaire
            if len(self._cache) > self.max_items:
                self._cleanup()

    def get(self, key: str) -> Optional[MarketDataPoint]:
        """Récupérer depuis le cache"""
        with self._lock:
            if key not in self._cache:
                return None

            # Vérifier l'âge
            age = datetime.utcnow() - self._access_times[key]
            if age.total_seconds() > self.max_age_seconds:
                del self._cache[key]
                del self._access_times[key]
                return None

            self._access_times[key] = datetime.utcnow()
            return self._cache[key]

    def _cleanup(self):
        """Nettoyer les entrées expirées et anciennes"""
        now = datetime.utcnow()
        to_remove = []

        # Supprimer les expirées
        for key, access_time in self._access_times.items():
            if (now - access_time).total_seconds() > self.max_age_seconds:
                to_remove.append(key)

        for key in to_remove:
            del self._cache[key]
            del self._access_times[key]

        # Si toujours trop d'éléments, supprimer les plus anciens
        if len(self._cache) > self.max_items:
            sorted_items = sorted(self._access_times.items(), key=lambda x: x[1])
            items_to_remove = len(self._cache) - self.max_items

            for key, _ in sorted_items[:items_to_remove]:
                del self._cache[key]
                del self._access_times[key]

    def clear(self):
        """Vider le cache"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Obtenir les statistiques du cache"""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_items": self.max_items,
                "max_age_seconds": self.max_age_seconds
            }


class MarketDataPipeline:
    """
    Pipeline principal pour traiter les données de marché.
    Orchestre providers, validation, normalisation et distribution.
    """

    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)
        self.metrics = get_business_metrics()
        self.tracer = get_tracer()

        self._providers: Dict[str, DataProvider] = {}
        self._validator = DataValidator()
        self._normalizer = DataNormalizer()
        self._cache = MarketDataCache()
        self._subscribers: Dict[str, List[Callable[[MarketDataPoint], None]]] = defaultdict(list)
        self._running = False

        # Statistiques
        self._stats = {
            "messages_processed": 0,
            "messages_validated": 0,
            "messages_cached": 0,
            "validation_errors": 0
        }

    def register_provider(self, provider: DataProvider):
        """Enregistrer un provider de données"""
        self._providers[provider.name] = provider

        # Ajouter le callback pour traiter les données
        if hasattr(provider, 'add_callback'):
            provider.add_callback(self._process_data_point)

        self.logger.info(f"Registered data provider: {provider.name}")

    def get_registered_providers(self) -> List[str]:
        """Retourner la liste des noms des providers enregistrés"""
        return list(self._providers.keys())

    def get_provider(self, name: str) -> Optional[DataProvider]:
        """Récupérer un provider par son nom"""
        return self._providers.get(name)

    async def start(self):
        """Démarrer le pipeline"""
        self._running = True

        # Connecter tous les providers
        for name, provider in self._providers.items():
            try:
                success = await provider.connect()
                if success:
                    self.logger.info(f"Connected to provider: {name}")
                else:
                    self.logger.error(f"Failed to connect to provider: {name}")
            except Exception as e:
                self.logger.error(f"Error connecting to {name}: {e}")

        self.logger.info("Market data pipeline started")

    async def stop(self):
        """Arrêter le pipeline"""
        self._running = False

        # Déconnecter tous les providers
        for name, provider in self._providers.items():
            try:
                await provider.disconnect()
                self.logger.info(f"Disconnected from provider: {name}")
            except Exception as e:
                self.logger.error(f"Error disconnecting from {name}: {e}")

        self.logger.info("Market data pipeline stopped")

    def subscribe(self, pattern: str, callback: Callable[[MarketDataPoint], None]):
        """S'abonner aux données selon un pattern"""
        self._subscribers[pattern].append(callback)
        self.logger.info(f"Added subscriber for pattern: {pattern}")

    def unsubscribe(self, pattern: str, callback: Callable[[MarketDataPoint], None]):
        """Se désabonner d'un pattern"""
        if pattern in self._subscribers:
            try:
                self._subscribers[pattern].remove(callback)
            except ValueError:
                pass

    async def subscribe_symbol(self, symbol: str, data_types: List[DataType], providers: Optional[List[str]] = None):
        """S'abonner aux données d'un symbole"""
        providers_to_use = providers or list(self._providers.keys())

        for provider_name in providers_to_use:
            provider = self._providers.get(provider_name)
            if provider and provider.is_connected():
                try:
                    success = await provider.subscribe(symbol, data_types)
                    if success:
                        self.logger.info(f"Subscribed to {symbol} on {provider_name}")
                    else:
                        self.logger.warning(f"Failed to subscribe to {symbol} on {provider_name}")
                except Exception as e:
                    self.logger.error(f"Error subscribing to {symbol} on {provider_name}: {e}")

    @trace("market_data.process_point")
    def _process_data_point(self, data_point: MarketDataPoint):
        """Traiter un point de données entrant"""
        try:
            self._stats["messages_processed"] += 1

            # Validation
            original_quality = data_point.quality
            data_point.quality = self._validator.validate(data_point)

            if data_point.quality == DataQuality.CORRUPTED:
                self._stats["validation_errors"] += 1
                self.logger.warning(f"Corrupted data detected: {data_point.symbol}")
                return

            self._stats["messages_validated"] += 1

            # Cache
            cache_key = f"{data_point.symbol}:{data_point.data_type.value}:{data_point.provider}"
            self._cache.put(cache_key, data_point)
            self._stats["messages_cached"] += 1

            # Distribution aux subscribers
            self._distribute_data_point(data_point)

            # Métriques
            self.metrics.collector.increment_counter(
                "data_pipeline.messages_processed",
                labels={"provider": data_point.provider, "symbol": data_point.symbol, "type": data_point.data_type.value}
            )

            # Logger selon la qualité
            if data_point.quality != original_quality:
                self.logger.debug(
                    f"Data quality changed: {original_quality.value} -> {data_point.quality.value}",
                    symbol=data_point.symbol,
                    provider=data_point.provider
                )

        except Exception as e:
            self.logger.error(f"Error processing data point: {e}", error=e)

    def _distribute_data_point(self, data_point: MarketDataPoint):
        """Distribuer un point de données aux subscribers"""
        patterns_to_match = [
            data_point.symbol,  # Symbole exact
            data_point.data_type.value,  # Type de données
            f"{data_point.symbol}:{data_point.data_type.value}",  # Combinaison
            "*"  # Tous
        ]

        for pattern in patterns_to_match:
            if pattern in self._subscribers:
                for callback in self._subscribers[pattern]:
                    try:
                        callback(data_point)
                    except Exception as e:
                        self.logger.error(f"Error in subscriber callback: {e}")

    def get_latest_data(self, symbol: str, data_type: DataType, provider: Optional[str] = None) -> Optional[MarketDataPoint]:
        """Obtenir les dernières données pour un symbole"""
        if provider:
            cache_key = f"{symbol}:{data_type.value}:{provider}"
            return self._cache.get(cache_key)
        else:
            # Chercher dans tous les providers
            for provider_name in self._providers.keys():
                cache_key = f"{symbol}:{data_type.value}:{provider_name}"
                data = self._cache.get(cache_key)
                if data:
                    return data
            return None

    async def get_historical_data(
        self,
        symbol: str,
        data_type: DataType,
        start_time: datetime,
        end_time: datetime,
        provider: Optional[str] = None
    ) -> List[MarketDataPoint]:
        """Récupérer des données historiques"""
        providers_to_use = [provider] if provider else list(self._providers.keys())
        all_data = []

        for provider_name in providers_to_use:
            provider = self._providers.get(provider_name)
            if provider:
                try:
                    data = await provider.get_historical_data(symbol, data_type, start_time, end_time)
                    all_data.extend(data)
                except Exception as e:
                    self.logger.error(f"Error getting historical data from {provider_name}: {e}")

        # Trier par timestamp
        all_data.sort(key=lambda x: x.timestamp)
        return all_data

    def get_statistics(self) -> Dict[str, Any]:
        """Obtenir les statistiques du pipeline"""
        provider_stats = {}
        for name, provider in self._providers.items():
            provider_stats[name] = {
                "connected": provider.is_connected(),
                "name": provider.name
            }

        return {
            **self._stats,
            "providers": provider_stats,
            "cache_stats": self._cache.get_stats(),
            "subscribers_count": sum(len(subs) for subs in self._subscribers.values())
        }


# Instance globale pour faciliter l'accès
_global_pipeline = MarketDataPipeline()


def get_market_data_pipeline() -> MarketDataPipeline:
    """Obtenir l'instance globale du pipeline"""
    return _global_pipeline