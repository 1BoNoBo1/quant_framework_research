"""
Infrastructure Layer: Time-Series Database
==========================================

Système de base de données time-series pour stocker efficacement
les données de marché historiques avec InfluxDB.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd

try:
    from influxdb_client import InfluxDBClient, Point, WriteOptions
    from influxdb_client.client.write_api import SYNCHRONOUS
    import influxdb_client.client.query_api as query_api
except ImportError:
    InfluxDBClient = None
    Point = None
    WriteOptions = None
    SYNCHRONOUS = None
    query_api = None

from ..observability.logging import LoggerFactory
from ..observability.metrics import get_business_metrics
from ..observability.tracing import get_tracer, trace

from ..data.market_data_pipeline import MarketDataPoint, DataType, DataQuality


class Aggregation(str, Enum):
    """Types d'agrégation pour les requêtes"""
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    FIRST = "first"
    LAST = "last"
    SUM = "sum"
    COUNT = "count"
    STDDEV = "stddev"


@dataclass
class TimeSeriesConfig:
    """Configuration de la base de données time-series"""
    # InfluxDB configuration
    url: str = "http://localhost:8086"
    token: str = "your-token"
    org: str = "qframe"
    bucket: str = "market_data"
    
    # Connection settings
    timeout: int = 30000  # ms
    verify_ssl: bool = True
    
    # Write settings
    batch_size: int = 1000
    flush_interval: int = 1000  # ms
    retry_interval: int = 5000  # ms
    max_retries: int = 3
    
    # Retention policy
    retention_days: int = 365


@dataclass
class QueryOptions:
    """Options pour les requêtes time-series"""
    start_time: datetime
    end_time: datetime
    symbols: Optional[List[str]] = None
    data_types: Optional[List[DataType]] = None
    providers: Optional[List[str]] = None
    aggregation: Optional[Aggregation] = None
    window: Optional[str] = None  # "1m", "5m", "1h", etc.
    limit: Optional[int] = None


class TimeSeriesDB(ABC):
    """Interface pour les bases de données time-series"""

    @abstractmethod
    async def write_market_data(self, data_point: MarketDataPoint) -> bool:
        """Écrire un point de données de marché"""
        pass

    @abstractmethod
    async def write_market_data_batch(self, data_points: List[MarketDataPoint]) -> bool:
        """Écrire un batch de données de marché"""
        pass

    @abstractmethod
    async def query_market_data(self, options: QueryOptions) -> pd.DataFrame:
        """Requêter les données de marché"""
        pass

    @abstractmethod
    async def get_latest_data(self, symbol: str, data_type: DataType, provider: Optional[str] = None) -> Optional[MarketDataPoint]:
        """Obtenir les dernières données"""
        pass

    @abstractmethod
    async def get_symbols(self) -> List[str]:
        """Obtenir la liste des symboles disponibles"""
        pass


class InfluxDBManager(TimeSeriesDB):
    """
    Gestionnaire InfluxDB pour les données time-series.
    Optimisé pour les données de marché haute fréquence.
    """

    def __init__(self, config: TimeSeriesConfig):
        self.config = config
        self.logger = LoggerFactory.get_logger(__name__)
        self.metrics = get_business_metrics()
        self.tracer = get_tracer()

        self._client: Optional[InfluxDBClient] = None
        self._write_api = None
        self._query_api = None

        # Statistiques
        self._writes_count = 0
        self._writes_failed = 0
        self._queries_count = 0
        self._queries_failed = 0

    async def initialize(self):
        """Initialiser la connexion InfluxDB"""
        try:
            self._client = InfluxDBClient(
                url=self.config.url,
                token=self.config.token,
                org=self.config.org,
                timeout=self.config.timeout,
                verify_ssl=self.config.verify_ssl
            )

            # Configuration d'écriture optimisée pour haute fréquence
            write_options = WriteOptions(
                batch_size=self.config.batch_size,
                flush_interval=self.config.flush_interval,
                retry_interval=self.config.retry_interval,
                max_retries=self.config.max_retries
            )

            self._write_api = self._client.write_api(write_options=write_options)
            self._query_api = self._client.query_api()

            # Tester la connexion
            await self._test_connection()

            self.logger.info("InfluxDB manager initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize InfluxDB: {e}")
            raise

    async def _test_connection(self):
        """Tester la connexion InfluxDB"""
        try:
            # Requête simple pour tester
            query = f'from(bucket:"{self.config.bucket}") |> range(start: -1m) |> limit(n:1)'
            result = self._query_api.query(query)
            
            self.logger.info("InfluxDB connection test successful")

        except Exception as e:
            self.logger.error(f"InfluxDB connection test failed: {e}")
            raise

    @trace("influxdb.write_market_data")
    async def write_market_data(self, data_point: MarketDataPoint) -> bool:
        """Écrire un point de données de marché"""
        try:
            point = self._create_point(data_point)
            self._write_api.write(bucket=self.config.bucket, record=point)
            
            self._writes_count += 1
            self.metrics.collector.increment_counter(
                "timeseries.writes",
                labels={"symbol": data_point.symbol, "data_type": data_point.data_type.value}
            )

            return True

        except Exception as e:
            self.logger.error(f"Error writing market data point: {e}")
            self._writes_failed += 1
            self.metrics.collector.increment_counter("timeseries.write_errors")
            return False

    @trace("influxdb.write_batch")
    async def write_market_data_batch(self, data_points: List[MarketDataPoint]) -> bool:
        """Écrire un batch de données de marché"""
        try:
            points = [self._create_point(dp) for dp in data_points]
            self._write_api.write(bucket=self.config.bucket, record=points)
            
            self._writes_count += len(data_points)
            self.metrics.collector.increment_counter(
                "timeseries.batch_writes",
                value=len(data_points)
            )

            return True

        except Exception as e:
            self.logger.error(f"Error writing market data batch: {e}")
            self._writes_failed += len(data_points)
            self.metrics.collector.increment_counter("timeseries.batch_write_errors")
            return False

    def _create_point(self, data_point: MarketDataPoint) -> Point:
        """Créer un Point InfluxDB depuis MarketDataPoint"""
        point = Point("market_data") \
            .tag("symbol", data_point.symbol) \
            .tag("data_type", data_point.data_type.value) \
            .tag("provider", data_point.provider) \
            .tag("quality", data_point.quality.value) \
            .time(data_point.timestamp)

        # Ajouter les champs selon le type de données
        if data_point.data_type == DataType.TICKER:
            point = self._add_ticker_fields(point, data_point.data)
        elif data_point.data_type == DataType.TRADES:
            point = self._add_trade_fields(point, data_point.data)
        elif data_point.data_type == DataType.ORDERBOOK:
            point = self._add_orderbook_fields(point, data_point.data)
        elif data_point.data_type == DataType.CANDLES:
            point = self._add_candle_fields(point, data_point.data)
        else:
            # Données génériques
            for key, value in data_point.data.items():
                if isinstance(value, (int, float)):
                    point = point.field(key, value)
                elif isinstance(value, str):
                    point = point.tag(key, value)

        return point

    def _add_ticker_fields(self, point: Point, data: Dict[str, Any]) -> Point:
        """Ajouter les champs ticker"""
        if "bid" in data:
            point = point.field("bid", float(data["bid"]))
        if "ask" in data:
            point = point.field("ask", float(data["ask"]))
        if "last" in data:
            point = point.field("last", float(data["last"]))
        if "volume_24h" in data:
            point = point.field("volume_24h", float(data["volume_24h"]))
        if "change_24h" in data:
            point = point.field("change_24h", float(data["change_24h"]))
        if "high_24h" in data:
            point = point.field("high_24h", float(data["high_24h"]))
        if "low_24h" in data:
            point = point.field("low_24h", float(data["low_24h"]))

        return point

    def _add_trade_fields(self, point: Point, data: Dict[str, Any]) -> Point:
        """Ajouter les champs trade"""
        if "price" in data:
            point = point.field("price", float(data["price"]))
        if "size" in data or "quantity" in data:
            size = data.get("size", data.get("quantity"))
            point = point.field("size", float(size))
        if "side" in data:
            point = point.tag("side", str(data["side"]))
        if "trade_id" in data:
            point = point.tag("trade_id", str(data["trade_id"]))

        return point

    def _add_orderbook_fields(self, point: Point, data: Dict[str, Any]) -> Point:
        """Ajouter les champs orderbook"""
        if "bids" in data and data["bids"]:
            # Meilleur bid
            best_bid = data["bids"][0]
            point = point.field("best_bid_price", float(best_bid[0]))
            point = point.field("best_bid_size", float(best_bid[1]))

        if "asks" in data and data["asks"]:
            # Meilleur ask
            best_ask = data["asks"][0]
            point = point.field("best_ask_price", float(best_ask[0]))
            point = point.field("best_ask_size", float(best_ask[1]))

        # Calculer le spread
        if "bids" in data and "asks" in data and data["bids"] and data["asks"]:
            spread = float(data["asks"][0][0]) - float(data["bids"][0][0])
            point = point.field("spread", spread)

        return point

    def _add_candle_fields(self, point: Point, data: Dict[str, Any]) -> Point:
        """Ajouter les champs candle/OHLCV"""
        if "open" in data:
            point = point.field("open", float(data["open"]))
        if "high" in data:
            point = point.field("high", float(data["high"]))
        if "low" in data:
            point = point.field("low", float(data["low"]))
        if "close" in data:
            point = point.field("close", float(data["close"]))
        if "volume" in data:
            point = point.field("volume", float(data["volume"]))

        return point

    @trace("influxdb.query")
    async def query_market_data(self, options: QueryOptions) -> pd.DataFrame:
        """Requêter les données de marché"""
        try:
            query = self._build_query(options)
            
            self.logger.debug(f"Executing InfluxDB query: {query}")
            
            result = self._query_api.query_data_frame(query)
            
            self._queries_count += 1
            self.metrics.collector.increment_counter("timeseries.queries")

            return result

        except Exception as e:
            self.logger.error(f"Error querying market data: {e}")
            self._queries_failed += 1
            self.metrics.collector.increment_counter("timeseries.query_errors")
            return pd.DataFrame()

    def _build_query(self, options: QueryOptions) -> str:
        """Construire une requête Flux"""
        # Base query
        query = f'from(bucket:"{self.config.bucket}")'
        
        # Range
        start = options.start_time.isoformat()
        end = options.end_time.isoformat()
        query += f' |> range(start: {start}, stop: {end})'

        # Filtres
        filters = ['r._measurement == "market_data"']
        
        if options.symbols:
            symbol_filter = " or ".join([f'r.symbol == "{s}"' for s in options.symbols])
            filters.append(f'({symbol_filter})')
            
        if options.data_types:
            type_filter = " or ".join([f'r.data_type == "{t.value}"' for t in options.data_types])
            filters.append(f'({type_filter})')
            
        if options.providers:
            provider_filter = " or ".join([f'r.provider == "{p}"' for p in options.providers])
            filters.append(f'({provider_filter})')

        if filters:
            filter_expr = " and ".join(filters)
            query += f' |> filter(fn: (r) => {filter_expr})'

        # Agrégation
        if options.aggregation and options.window:
            query += f' |> aggregateWindow(every: {options.window}, fn: {options.aggregation.value})'

        # Limit
        if options.limit:
            query += f' |> limit(n: {options.limit})'

        return query

    async def get_latest_data(self, symbol: str, data_type: DataType, provider: Optional[str] = None) -> Optional[MarketDataPoint]:
        """Obtenir les dernières données"""
        try:
            query = f"""
            from(bucket:"{self.config.bucket}")
            |> range(start: -1h)
            |> filter(fn: (r) => r._measurement == "market_data")
            |> filter(fn: (r) => r.symbol == "{symbol}")
            |> filter(fn: (r) => r.data_type == "{data_type.value}")
            """
            
            if provider:
                query += f'|> filter(fn: (r) => r.provider == "{provider}")'
                
            query += "|> last()"
            
            result = self._query_api.query_data_frame(query)
            
            if not result.empty:
                # Reconstruire le MarketDataPoint
                row = result.iloc[0]
                return self._row_to_market_data_point(row)

            return None

        except Exception as e:
            self.logger.error(f"Error getting latest data for {symbol}: {e}")
            return None

    def _row_to_market_data_point(self, row) -> MarketDataPoint:
        """Convertir une ligne DataFrame en MarketDataPoint"""
        # Implémentation simplifiée
        return MarketDataPoint(
            symbol=row.get("symbol", ""),
            data_type=DataType(row.get("data_type", "ticker")),
            timestamp=row.get("_time", datetime.utcnow()),
            data={"value": row.get("_value", 0)},
            provider=row.get("provider", "unknown"),
            quality=DataQuality(row.get("quality", "medium"))
        )

    async def get_symbols(self) -> List[str]:
        """Obtenir la liste des symboles disponibles"""
        try:
            query = f"""
            from(bucket:"{self.config.bucket}")
            |> range(start: -24h)
            |> filter(fn: (r) => r._measurement == "market_data")
            |> distinct(column: "symbol")
            """
            
            result = self._query_api.query_data_frame(query)
            
            if not result.empty and "symbol" in result.columns:
                return result["symbol"].unique().tolist()
            
            return []

        except Exception as e:
            self.logger.error(f"Error getting symbols: {e}")
            return []

    async def get_statistics(self) -> Dict[str, Any]:
        """Obtenir les statistiques du gestionnaire"""
        stats = {
            "writes_count": self._writes_count,
            "writes_failed": self._writes_failed,
            "queries_count": self._queries_count,
            "queries_failed": self._queries_failed,
            "write_success_rate": self._writes_count / max(1, self._writes_count + self._writes_failed),
            "query_success_rate": self._queries_count / max(1, self._queries_count + self._queries_failed)
        }

        # Statistiques InfluxDB
        try:
            if self._client:
                health = self._client.health()
                stats["influxdb_status"] = health.status
                stats["influxdb_version"] = getattr(health, "version", "unknown")
        except Exception:
            stats["influxdb_status"] = "unknown"

        return stats

    async def close(self):
        """Fermer la connexion InfluxDB"""
        if self._client:
            self._client.close()
            self.logger.info("InfluxDB connection closed")


class MarketDataStorage:
    """
    Gestionnaire de stockage spécialisé pour les données de marché.
    Combine time-series DB et cache pour performances optimales.
    """

    def __init__(self, timeseries_db: TimeSeriesDB, cache_manager = None):
        self.timeseries_db = timeseries_db
        self.cache_manager = cache_manager
        self.logger = LoggerFactory.get_logger(__name__)

    async def store_market_data(self, data_point: MarketDataPoint):
        """Stocker des données de marché"""
        # Stocker en time-series DB
        success = await self.timeseries_db.write_market_data(data_point)
        
        if success and self.cache_manager:
            # Mettre en cache les dernières données
            cache_key = f"latest:{data_point.symbol}:{data_point.data_type.value}:{data_point.provider}"
            await self.cache_manager.set(cache_key, data_point, ttl=300, namespace="market_data")

        return success

    async def store_market_data_batch(self, data_points: List[MarketDataPoint]):
        """Stocker un batch de données de marché"""
        return await self.timeseries_db.write_market_data_batch(data_points)

    async def get_latest_cached(self, symbol: str, data_type: DataType, provider: Optional[str] = None) -> Optional[MarketDataPoint]:
        """Obtenir les dernières données depuis le cache"""
        if not self.cache_manager:
            return None

        cache_key = f"latest:{symbol}:{data_type.value}:{provider or '*'}"
        return await self.cache_manager.get(cache_key, namespace="market_data")

    async def get_historical_data(self, options: QueryOptions) -> pd.DataFrame:
        """Obtenir des données historiques"""
        return await self.timeseries_db.query_market_data(options)

    async def get_ohlcv(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1m",
        provider: Optional[str] = None
    ) -> pd.DataFrame:
        """Obtenir des données OHLCV"""
        options = QueryOptions(
            start_time=start_time,
            end_time=end_time,
            symbols=[symbol],
            data_types=[DataType.CANDLES],
            providers=[provider] if provider else None,
            aggregation=Aggregation.LAST,
            window=interval
        )

        return await self.get_historical_data(options)


# Instance globale
_global_timeseries_db: Optional[TimeSeriesDB] = None


def get_timeseries_db() -> Optional[TimeSeriesDB]:
    """Obtenir l'instance globale de la time-series DB"""
    return _global_timeseries_db


def create_timeseries_db(config: TimeSeriesConfig) -> TimeSeriesDB:
    """Créer l'instance globale de la time-series DB"""
    global _global_timeseries_db
    _global_timeseries_db = InfluxDBManager(config)
    return _global_timeseries_db
