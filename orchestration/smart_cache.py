#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Cache System
Système de cache intelligent pour optimiser les performances du pipeline
"""

import os
import json
import hashlib
import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import time

import pandas as pd
import numpy as np

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartCache:
    """
    Système de cache intelligent avec TTL, validation et compression
    """
    
    def __init__(self, cache_dir: str = "cache", default_ttl_hours: float = 1):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.default_ttl_hours = default_ttl_hours
        
        # Sous-dossiers par type de cache
        self.data_cache_dir = self.cache_dir / "data"
        self.features_cache_dir = self.cache_dir / "features"
        self.models_cache_dir = self.cache_dir / "models"
        
        for dir_path in [self.data_cache_dir, self.features_cache_dir, self.models_cache_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Métadonnées cache
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
        
        logger.info(f"✅ Smart Cache initialisé: {self.cache_dir}")
    
    def _load_metadata(self) -> Dict:
        """Charge les métadonnées du cache"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Erreur chargement métadonnées: {e}")
        
        return {
            "created": datetime.now().isoformat(),
            "entries": {},
            "stats": {
                "hits": 0,
                "misses": 0,
                "invalidations": 0
            }
        }
    
    def _save_metadata(self):
        """Sauvegarde les métadonnées du cache"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde métadonnées: {e}")
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Génère une clé de cache unique"""
        # Créer un hash des arguments
        args_str = str(sorted(args)) if args else ""
        kwargs_str = str(sorted(kwargs.items())) if kwargs else ""
        
        cache_input = f"{func_name}_{args_str}_{kwargs_str}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str, ttl_hours: float) -> bool:
        """Vérifie si le cache est encore valide"""
        if cache_key not in self.metadata["entries"]:
            return False
        
        entry = self.metadata["entries"][cache_key]
        created_time = datetime.fromisoformat(entry["created"])
        expiry_time = created_time + timedelta(hours=ttl_hours)
        
        return datetime.now() < expiry_time
    
    def _get_cache_file_path(self, cache_key: str, cache_type: str = "data") -> Path:
        """Génère le chemin du fichier cache"""
        if cache_type == "data":
            return self.data_cache_dir / f"{cache_key}.pkl"
        elif cache_type == "features":
            return self.features_cache_dir / f"{cache_key}.parquet"
        elif cache_type == "models":
            return self.models_cache_dir / f"{cache_key}.pkl"
        else:
            return self.cache_dir / f"{cache_key}.pkl"
    
    def get(self, cache_key: str, cache_type: str = "data", ttl_hours: Optional[float] = None) -> Optional[Any]:
        """Récupère une valeur du cache"""
        ttl = ttl_hours or self.default_ttl_hours
        
        if not self._is_cache_valid(cache_key, ttl):
            self.metadata["stats"]["misses"] += 1
            return None
        
        cache_file = self._get_cache_file_path(cache_key, cache_type)
        if not cache_file.exists():
            self.metadata["stats"]["misses"] += 1
            return None
        
        try:
            if cache_type == "features" and cache_file.suffix == ".parquet":
                data = pd.read_parquet(cache_file)
            else:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
            
            self.metadata["stats"]["hits"] += 1
            self._save_metadata()
            
            logger.info(f"🎯 Cache HIT: {cache_key[:12]}...")
            return data
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur lecture cache {cache_key}: {e}")
            self.metadata["stats"]["misses"] += 1
            return None
    
    def set(self, cache_key: str, data: Any, cache_type: str = "data", 
            metadata: Optional[Dict] = None) -> bool:
        """Stocke une valeur dans le cache"""
        cache_file = self._get_cache_file_path(cache_key, cache_type)
        
        try:
            if cache_type == "features" and isinstance(data, pd.DataFrame):
                data.to_parquet(cache_file, compression='snappy')
            else:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Mise à jour métadonnées
            self.metadata["entries"][cache_key] = {
                "created": datetime.now().isoformat(),
                "cache_type": cache_type,
                "file_size": cache_file.stat().st_size,
                "metadata": metadata or {}
            }
            
            self._save_metadata()
            logger.info(f"💾 Cache SET: {cache_key[:12]}... ({cache_file.stat().st_size} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur écriture cache {cache_key}: {e}")
            return False
    
    def invalidate(self, cache_key: str) -> bool:
        """Invalide une entrée de cache"""
        try:
            # Supprimer le fichier
            for cache_type in ["data", "features", "models"]:
                cache_file = self._get_cache_file_path(cache_key, cache_type)
                if cache_file.exists():
                    cache_file.unlink()
                    
            # Supprimer des métadonnées
            if cache_key in self.metadata["entries"]:
                del self.metadata["entries"][cache_key]
                
            self.metadata["stats"]["invalidations"] += 1
            self._save_metadata()
            
            logger.info(f"🗑️ Cache invalidé: {cache_key[:12]}...")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur invalidation cache {cache_key}: {e}")
            return False
    
    def cleanup_expired(self):
        """Nettoie les entrées expirées"""
        logger.info("🧹 Nettoyage cache expiré...")
        
        expired_keys = []
        for cache_key, entry in self.metadata["entries"].items():
            created_time = datetime.fromisoformat(entry["created"])
            # Utiliser TTL par défaut pour le nettoyage
            expiry_time = created_time + timedelta(hours=self.default_ttl_hours * 2)  # Grace period
            
            if datetime.now() > expiry_time:
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            self.invalidate(key)
        
        logger.info(f"🗑️ {len(expired_keys)} entrées expirées supprimées")
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques du cache"""
        stats = self.metadata["stats"].copy()
        total_requests = stats["hits"] + stats["misses"]
        stats["hit_rate"] = (stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        stats["total_entries"] = len(self.metadata["entries"])
        
        # Taille totale du cache
        total_size = 0
        for entry in self.metadata["entries"].values():
            total_size += entry.get("file_size", 0)
        stats["total_size_mb"] = total_size / (1024 * 1024)
        
        return stats

class CacheDecorator:
    """
    Décorateur pour mise en cache automatique des fonctions
    """
    
    def __init__(self, cache: SmartCache, cache_type: str = "data", 
                 ttl_hours: float = 1, key_args: Optional[List[str]] = None):
        self.cache = cache
        self.cache_type = cache_type
        self.ttl_hours = ttl_hours
        self.key_args = key_args or []
    
    def __call__(self, func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Générer clé de cache
            if self.key_args:
                # Utiliser seulement les arguments spécifiés
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in self.key_args}
                cache_key = self.cache._generate_cache_key(func.__name__, (), filtered_kwargs)
            else:
                cache_key = self.cache._generate_cache_key(func.__name__, args, kwargs)
            
            # Tentative de récupération du cache
            cached_result = self.cache.get(cache_key, self.cache_type, self.ttl_hours)
            if cached_result is not None:
                return cached_result
            
            # Exécution de la fonction
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Mise en cache du résultat
            metadata = {
                "function": func.__name__,
                "execution_time": execution_time,
                "args_count": len(args),
                "kwargs_count": len(kwargs)
            }
            
            self.cache.set(cache_key, result, self.cache_type, metadata)
            
            return result
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

# Instance globale du cache
_global_cache = None

def get_cache() -> SmartCache:
    """Retourne l'instance globale du cache"""
    global _global_cache
    if _global_cache is None:
        _global_cache = SmartCache()
    return _global_cache

def cached_data(ttl_hours: float = 1, key_args: Optional[List[str]] = None):
    """Décorateur pour cache de données"""
    return CacheDecorator(get_cache(), "data", ttl_hours, key_args)

def cached_features(ttl_hours: float = 6, key_args: Optional[List[str]] = None):
    """Décorateur pour cache de features"""
    return CacheDecorator(get_cache(), "features", ttl_hours, key_args)

def cached_models(ttl_hours: float = 24, key_args: Optional[List[str]] = None):
    """Décorateur pour cache de modèles"""
    return CacheDecorator(get_cache(), "models", ttl_hours, key_args)

def main():
    """
    Test et démonstration du système de cache
    """
    cache = SmartCache()
    
    print("📊 SMART CACHE SYSTEM")
    print("=" * 50)
    
    # Test basique
    test_data = {"test": "data", "timestamp": datetime.now().isoformat()}
    cache_key = "test_key_123"
    
    print(f"💾 Test stockage: {cache_key}")
    cache.set(cache_key, test_data)
    
    print(f"🎯 Test récupération: {cache_key}")
    retrieved = cache.get(cache_key)
    print(f"✅ Données récupérées: {retrieved is not None}")
    
    # Test DataFrame
    print(f"📊 Test DataFrame cache")
    df_test = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1h'),
        'price': np.random.random(100) * 1000,
        'volume': np.random.random(100) * 1000
    })
    
    cache.set("test_df", df_test, "features")
    retrieved_df = cache.get("test_df", "features")
    print(f"✅ DataFrame récupéré: {retrieved_df is not None}")
    if retrieved_df is not None:
        print(f"   Shape: {retrieved_df.shape}")
    
    # Statistiques
    stats = cache.get_stats()
    print(f"\n📈 STATISTIQUES CACHE:")
    print(f"   Hits: {stats['hits']}")
    print(f"   Misses: {stats['misses']}")
    print(f"   Hit Rate: {stats['hit_rate']:.1f}%")
    print(f"   Total Entries: {stats['total_entries']}")
    print(f"   Taille: {stats['total_size_mb']:.2f} MB")
    
    # Test décorateur
    @cached_data(ttl_hours=0.1)
    def expensive_computation(n: int) -> Dict:
        """Simulation calcul coûteux"""
        time.sleep(0.1)  # Simulation
        return {"result": n * 2, "computed_at": datetime.now().isoformat()}
    
    print(f"\n🧮 Test décorateur:")
    start = time.time()
    result1 = expensive_computation(42)
    time1 = time.time() - start
    
    start = time.time()
    result2 = expensive_computation(42)  # Doit venir du cache
    time2 = time.time() - start
    
    print(f"   Premier appel: {time1:.3f}s")
    print(f"   Deuxième appel: {time2:.3f}s (cache)")
    print(f"   Accélération: {time1/time2:.1f}x")
    
    # Nettoyage
    print(f"\n🧹 Test nettoyage...")
    cache.cleanup_expired()
    
    final_stats = cache.get_stats()
    print(f"📊 Statistiques finales:")
    print(f"   Hit Rate: {final_stats['hit_rate']:.1f}%")

if __name__ == "__main__":
    main()