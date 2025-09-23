"""
Infrastructure Layer: Cache Management
=====================================

Système de cache avancé avec Redis et cache en mémoire
pour optimiser les performances.
"""

import asyncio
import json
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
import hashlib

import redis.asyncio as redis
from redis.asyncio import Redis

from ..observability.logging import LoggerFactory
from ..observability.metrics import get_business_metrics
from ..observability.tracing import get_tracer, trace


class CacheStrategy(str, Enum):
    """Stratégies de cache"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    FIFO = "fifo"  # First In First Out


@dataclass
class CacheConfig:
    """Configuration du cache"""
    # Redis configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    
    # Connection pool
    max_connections: int = 20
    connection_timeout: int = 30
    
    # Cache behavior
    default_ttl: int = 3600  # 1 hour
    max_memory_mb: int = 100
    serialization: str = "json"  # "json", "pickle"
    compression: bool = True
    
    # Fallback
    enable_memory_fallback: bool = True


@dataclass
class CacheStats:
    """Statistiques du cache"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Taux de succès du cache"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CacheInterface(ABC):
    """Interface pour les implémentations de cache"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Récupérer une valeur"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Définir une valeur"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Supprimer une valeur"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Vérifier si une clé existe"""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """Vider le cache"""
        pass

    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Obtenir les statistiques"""
        pass


class InMemoryCache(CacheInterface):
    """
    Cache en mémoire simple avec support TTL et stratégies d'éviction.
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600, strategy: CacheStrategy = CacheStrategy.LRU):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.strategy = strategy
        self.logger = LoggerFactory.get_logger(__name__)

        # Storage
        self._cache: Dict[str, Any] = {}
        self._expiry: Dict[str, datetime] = {}
        self._access_times: Dict[str, datetime] = {}
        self._access_counts: Dict[str, int] = {}
        self._creation_order: List[str] = []

        # Stats
        self._stats = CacheStats()

    async def get(self, key: str) -> Optional[Any]:
        """Récupérer une valeur"""
        try:
            # Vérifier expiration
            if key in self._expiry and datetime.utcnow() > self._expiry[key]:
                await self.delete(key)
                self._stats.misses += 1
                return None

            if key in self._cache:
                # Mettre à jour les statistiques d'accès
                self._access_times[key] = datetime.utcnow()
                self._access_counts[key] = self._access_counts.get(key, 0) + 1
                
                self._stats.hits += 1
                return self._cache[key]

            self._stats.misses += 1
            return None

        except Exception as e:
            self.logger.error(f"Error getting cache key {key}: {e}")
            self._stats.errors += 1
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Définir une valeur"""
        try:
            # Gérer la taille maximale
            if len(self._cache) >= self.max_size and key not in self._cache:
                await self._evict_one()

            # Stocker la valeur
            self._cache[key] = value
            
            # Gérer TTL
            if ttl is not None:
                self._expiry[key] = datetime.utcnow() + timedelta(seconds=ttl)
            elif self.default_ttl > 0:
                self._expiry[key] = datetime.utcnow() + timedelta(seconds=self.default_ttl)

            # Mettre à jour les métadonnées
            self._access_times[key] = datetime.utcnow()
            self._access_counts[key] = 1
            
            if key not in self._creation_order:
                self._creation_order.append(key)

            self._stats.sets += 1
            return True

        except Exception as e:
            self.logger.error(f"Error setting cache key {key}: {e}")
            self._stats.errors += 1
            return False

    async def delete(self, key: str) -> bool:
        """Supprimer une valeur"""
        try:
            if key in self._cache:
                del self._cache[key]
                self._expiry.pop(key, None)
                self._access_times.pop(key, None)
                self._access_counts.pop(key, None)
                
                if key in self._creation_order:
                    self._creation_order.remove(key)
                
                self._stats.deletes += 1
                return True
            
            return False

        except Exception as e:
            self.logger.error(f"Error deleting cache key {key}: {e}")
            self._stats.errors += 1
            return False

    async def exists(self, key: str) -> bool:
        """Vérifier si une clé existe"""
        return key in self._cache and (key not in self._expiry or datetime.utcnow() <= self._expiry[key])

    async def clear(self) -> bool:
        """Vider le cache"""
        try:
            self._cache.clear()
            self._expiry.clear()
            self._access_times.clear()
            self._access_counts.clear()
            self._creation_order.clear()
            return True

        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return False

    def get_stats(self) -> CacheStats:
        """Obtenir les statistiques"""
        return self._stats

    async def _evict_one(self):
        """Éviction d'un élément selon la stratégie"""
        if not self._cache:
            return

        key_to_evict = None

        if self.strategy == CacheStrategy.LRU:
            # Least Recently Used
            oldest_time = min(self._access_times.values())
            key_to_evict = next(k for k, v in self._access_times.items() if v == oldest_time)

        elif self.strategy == CacheStrategy.LFU:
            # Least Frequently Used
            min_count = min(self._access_counts.values())
            key_to_evict = next(k for k, v in self._access_counts.items() if v == min_count)

        elif self.strategy == CacheStrategy.FIFO:
            # First In First Out
            key_to_evict = self._creation_order[0]

        elif self.strategy == CacheStrategy.TTL:
            # Supprimer d'abord les expirés, sinon LRU
            now = datetime.utcnow()
            expired_keys = [k for k, exp in self._expiry.items() if exp <= now]
            
            if expired_keys:
                key_to_evict = expired_keys[0]
            else:
                # Fallback to LRU
                oldest_time = min(self._access_times.values())
                key_to_evict = next(k for k, v in self._access_times.items() if v == oldest_time)

        if key_to_evict:
            await self.delete(key_to_evict)


class RedisCache(CacheInterface):
    """
    Cache Redis avec sérialisation et compression.
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = LoggerFactory.get_logger(__name__)
        self.metrics = get_business_metrics()

        self._redis: Optional[Redis] = None
        self._stats = CacheStats()

    async def initialize(self):
        """Initialiser la connexion Redis"""
        try:
            self._redis = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                ssl=self.config.redis_ssl,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.connection_timeout,
                decode_responses=False  # Gérer nous-mêmes la sérialisation
            )

            # Tester la connexion
            await self._redis.ping()
            self.logger.info("Redis cache initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Redis cache: {e}")
            raise

    @trace("cache.redis.get")
    async def get(self, key: str) -> Optional[Any]:
        """Récupérer une valeur"""
        if not self._redis:
            return None

        try:
            data = await self._redis.get(key)
            
            if data is None:
                self._stats.misses += 1
                return None

            # Désérialiser
            value = self._deserialize(data)
            self._stats.hits += 1

            # Métriques
            self.metrics.collector.increment_counter("cache.hits", labels={"backend": "redis"})

            return value

        except Exception as e:
            self.logger.error(f"Error getting Redis key {key}: {e}")
            self._stats.errors += 1
            self.metrics.collector.increment_counter("cache.errors", labels={"backend": "redis", "operation": "get"})
            return None

    @trace("cache.redis.set")
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Définir une valeur"""
        if not self._redis:
            return False

        try:
            # Sérialiser
            data = self._serialize(value)
            
            # Définir TTL
            expire_time = ttl or self.config.default_ttl

            # Stocker dans Redis
            await self._redis.setex(key, expire_time, data)
            
            self._stats.sets += 1
            self.metrics.collector.increment_counter("cache.sets", labels={"backend": "redis"})

            return True

        except Exception as e:
            self.logger.error(f"Error setting Redis key {key}: {e}")
            self._stats.errors += 1
            self.metrics.collector.increment_counter("cache.errors", labels={"backend": "redis", "operation": "set"})
            return False

    async def delete(self, key: str) -> bool:
        """Supprimer une valeur"""
        if not self._redis:
            return False

        try:
            result = await self._redis.delete(key)
            deleted = result > 0
            
            if deleted:
                self._stats.deletes += 1
                self.metrics.collector.increment_counter("cache.deletes", labels={"backend": "redis"})
            
            return deleted

        except Exception as e:
            self.logger.error(f"Error deleting Redis key {key}: {e}")
            self._stats.errors += 1
            return False

    async def exists(self, key: str) -> bool:
        """Vérifier si une clé existe"""
        if not self._redis:
            return False

        try:
            result = await self._redis.exists(key)
            return result > 0

        except Exception as e:
            self.logger.error(f"Error checking Redis key {key}: {e}")
            return False

    async def clear(self) -> bool:
        """Vider le cache"""
        if not self._redis:
            return False

        try:
            await self._redis.flushdb()
            return True

        except Exception as e:
            self.logger.error(f"Error clearing Redis cache: {e}")
            return False

    def get_stats(self) -> CacheStats:
        """Obtenir les statistiques"""
        return self._stats

    async def get_redis_info(self) -> Dict[str, Any]:
        """Obtenir les informations Redis"""
        if not self._redis:
            return {}

        try:
            info = await self._redis.info()
            return {
                "version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
                "hit_rate": info.get("keyspace_hits", 0) / max(1, info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0))
            }

        except Exception as e:
            self.logger.error(f"Error getting Redis info: {e}")
            return {}

    def _serialize(self, value: Any) -> bytes:
        """Sérialiser une valeur"""
        if self.config.serialization == "json":
            data = json.dumps(value, default=str).encode()
        else:  # pickle
            data = pickle.dumps(value)

        # Compression optionnelle
        if self.config.compression and len(data) > 100:  # Seulement si > 100 bytes
            import gzip
            data = gzip.compress(data)

        return data

    def _deserialize(self, data: bytes) -> Any:
        """Désérialiser une valeur"""
        # Décompression si nécessaire
        if self.config.compression:
            try:
                import gzip
                data = gzip.decompress(data)
            except:
                pass  # Pas compressé

        if self.config.serialization == "json":
            return json.loads(data.decode())
        else:  # pickle
            return pickle.loads(data)

    async def close(self):
        """Fermer la connexion Redis"""
        if self._redis:
            await self._redis.close()
            self.logger.info("Redis connection closed")


class CacheManager:
    """
    Gestionnaire de cache unifié avec fallback et stratégies avancées.
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = LoggerFactory.get_logger(__name__)
        self.metrics = get_business_metrics()

        # Caches
        self.redis_cache = RedisCache(config)
        self.memory_cache = InMemoryCache(
            max_size=1000,
            default_ttl=config.default_ttl,
            strategy=CacheStrategy.LRU
        ) if config.enable_memory_fallback else None

        # Statistiques globales
        self._global_stats = CacheStats()

    async def initialize(self):
        """Initialiser le gestionnaire"""
        try:
            await self.redis_cache.initialize()
            self.logger.info("Cache manager initialized")

        except Exception as e:
            if self.memory_cache:
                self.logger.warning(f"Redis failed, using memory cache fallback: {e}")
            else:
                raise

    @trace("cache.get")
    async def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Récupérer une valeur avec namespace"""
        cache_key = self._build_key(key, namespace)

        # Essayer Redis d'abord
        try:
            value = await self.redis_cache.get(cache_key)
            if value is not None:
                return value

        except Exception as e:
            self.logger.warning(f"Redis get failed for {cache_key}: {e}")

        # Fallback vers cache mémoire
        if self.memory_cache:
            return await self.memory_cache.get(cache_key)

        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None, namespace: str = "default") -> bool:
        """Définir une valeur avec namespace"""
        cache_key = self._build_key(key, namespace)
        success = False

        # Stocker dans Redis
        try:
            success = await self.redis_cache.set(cache_key, value, ttl)
        except Exception as e:
            self.logger.warning(f"Redis set failed for {cache_key}: {e}")

        # Stocker aussi dans le cache mémoire (pour le fallback)
        if self.memory_cache:
            memory_success = await self.memory_cache.set(cache_key, value, ttl)
            success = success or memory_success

        return success

    async def delete(self, key: str, namespace: str = "default") -> bool:
        """Supprimer une valeur"""
        cache_key = self._build_key(key, namespace)
        
        redis_success = False
        memory_success = False

        # Supprimer de Redis
        try:
            redis_success = await self.redis_cache.delete(cache_key)
        except Exception as e:
            self.logger.warning(f"Redis delete failed for {cache_key}: {e}")

        # Supprimer du cache mémoire
        if self.memory_cache:
            memory_success = await self.memory_cache.delete(cache_key)

        return redis_success or memory_success

    async def clear_namespace(self, namespace: str = "default") -> bool:
        """Vider un namespace"""
        # Pour Redis, on utiliserait SCAN + DEL avec pattern
        # Pour l'instant, implémentation simplifiée
        self.logger.warning(f"Clear namespace {namespace} not fully implemented")
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Obtenir toutes les statistiques"""
        stats = {
            "redis": self.redis_cache.get_stats().__dict__,
            "global": self._global_stats.__dict__
        }

        if self.memory_cache:
            stats["memory"] = self.memory_cache.get_stats().__dict__

        return stats

    def _build_key(self, key: str, namespace: str) -> str:
        """Construire une clé avec namespace"""
        return f"{namespace}:{key}"

    # Méthodes de cache avancées

    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[int] = None,
        namespace: str = "default"
    ) -> Any:
        """Récupérer ou définir via une factory function"""
        value = await self.get(key, namespace)
        
        if value is None:
            value = await factory() if asyncio.iscoroutinefunction(factory) else factory()
            await self.set(key, value, ttl, namespace)
        
        return value

    async def invalidate_pattern(self, pattern: str, namespace: str = "default"):
        """Invalider toutes les clés matchant un pattern"""
        # Implémentation Redis SCAN + DELETE
        full_pattern = self._build_key(pattern, namespace)
        
        if self.redis_cache._redis:
            try:
                async for key in self.redis_cache._redis.scan_iter(match=full_pattern):
                    await self.redis_cache._redis.delete(key)
            except Exception as e:
                self.logger.error(f"Error invalidating pattern {pattern}: {e}")

    async def close(self):
        """Fermer le gestionnaire"""
        await self.redis_cache.close()
        if self.memory_cache:
            await self.memory_cache.clear()


# Instance globale
_global_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> Optional[CacheManager]:
    """Obtenir l'instance globale du gestionnaire de cache"""
    return _global_cache_manager


def create_cache_manager(config: CacheConfig) -> CacheManager:
    """Créer l'instance globale du gestionnaire de cache"""
    global _global_cache_manager
    _global_cache_manager = CacheManager(config)
    return _global_cache_manager
