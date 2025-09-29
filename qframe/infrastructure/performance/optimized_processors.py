"""
Optimized Performance Processors for QFrame
==========================================

Processeurs haute performance avec optimisations critiques pour
trading quantitatif en temps réel.
"""

import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache, wraps
from typing import Dict, List, Any, Optional, Callable, Union, TypeVar, Generic
from dataclasses import dataclass, field
from collections import deque
import weakref
import gc
import numpy as np
import pandas as pd

try:
    import numba
    from numba import jit, njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False

from qframe.infrastructure.observability.structured_logging import (
    StructuredLogger, PerformanceLogger, LoggerFactory, performance_logged
)

T = TypeVar('T')
ResultType = TypeVar('ResultType')


@dataclass
class PerformanceConfig:
    """Configuration des optimisations performance."""

    # Threading
    max_workers: int = 4
    use_processes: bool = False
    chunk_size: int = 1000

    # Memory management
    memory_pool_size: int = 100_000_000  # 100MB
    enable_gc_optimization: bool = True
    cache_size: int = 128

    # Computations
    use_numba: bool = HAS_NUMBA
    use_vectorization: bool = True
    batch_processing: bool = True

    # I/O optimizations
    use_arrow: bool = HAS_ARROW
    compression: str = "snappy"
    lazy_loading: bool = True


class MemoryPool:
    """Pool de mémoire optimisé pour réutilisation d'objets."""

    def __init__(self, pool_size: int = 100_000_000):
        self.pool_size = pool_size
        self.arrays: Dict[tuple, deque] = {}
        self.dataframes: deque = deque(maxlen=50)
        self.lock = threading.RLock()
        self.allocated_bytes = 0

        # Weak references pour cleanup automatique
        # Utiliser un dictionnaire avec id() comme clé pour arrays non-hashables
        self.active_objects = {}

    def get_array(self, shape: tuple, dtype: np.dtype = np.float64) -> np.ndarray:
        """Récupère un array du pool ou en crée un nouveau."""
        # Normaliser le dtype pour assurer la compatibilité des clés
        key = (shape, np.dtype(dtype))

        with self.lock:
            if key in self.arrays and self.arrays[key]:
                array = self.arrays[key].popleft()
                array.fill(0)  # Reset
                # Tracker l'array récupéré du pool
                self.active_objects[id(array)] = weakref.ref(array, lambda ref: self.active_objects.pop(id(array), None))
                return array

            # Créer nouveau si pas disponible
            array = np.zeros(shape, dtype=dtype)
            # Stocker avec id() comme clé et weak reference comme valeur
            self.active_objects[id(array)] = weakref.ref(array, lambda ref: self.active_objects.pop(id(array), None))
            self.allocated_bytes += array.nbytes

            return array

    def return_array(self, array: np.ndarray) -> None:
        """Retourne un array au pool."""
        key = (array.shape, array.dtype)

        with self.lock:
            if self.allocated_bytes < self.pool_size:
                if key not in self.arrays:
                    self.arrays[key] = deque(maxlen=20)
                self.arrays[key].append(array)

    def get_dataframe_buffer(self, size_hint: int = 1000) -> pd.DataFrame:
        """Récupère un DataFrame buffer optimisé."""
        with self.lock:
            if self.dataframes:
                df = self.dataframes.popleft()
                if len(df) >= size_hint:
                    return df.iloc[:0].copy()  # Empty but with columns

            # Créer nouveau DataFrame optimisé
            df = pd.DataFrame({
                'timestamp': pd.NaT,
                'value': 0.0,
                'volume': 0.0
            }, index=range(size_hint))

            return df

    def cleanup(self) -> None:
        """Nettoyage forcé du pool."""
        with self.lock:
            self.arrays.clear()
            self.dataframes.clear()
            self.allocated_bytes = 0
            gc.collect()


# Pool global
_memory_pool = MemoryPool()


class OptimizedCache:
    """Cache haute performance avec TTL et metrics."""

    def __init__(self, maxsize: int = 128, ttl: float = 300.0):
        self.cache = {}
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self.maxsize = maxsize
        self.ttl = ttl
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur du cache."""
        current_time = time.time()

        with self.lock:
            if key in self.cache:
                # Vérifier TTL
                if current_time - self.access_times[key] < self.ttl:
                    self.hit_count += 1
                    self.access_times[key] = current_time
                    return self.cache[key]
                else:
                    # Expiré
                    del self.cache[key]
                    del self.access_times[key]

            self.miss_count += 1
            return None

    def set(self, key: str, value: Any) -> None:
        """Stocke une valeur dans le cache."""
        current_time = time.time()

        with self.lock:
            # Éviction si nécessaire
            if len(self.cache) >= self.maxsize and key not in self.cache:
                # LRU éviction
                oldest_key = min(self.access_times.keys(),
                               key=lambda k: self.access_times[k])
                del self.cache[oldest_key]
                del self.access_times[oldest_key]

            self.cache[key] = value
            self.access_times[key] = current_time

    @property
    def hit_ratio(self) -> float:
        """Ratio de cache hits."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

    def clear(self) -> None:
        """Vide le cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()


# Cache global
_global_cache = OptimizedCache()


def optimized_cache(ttl: float = 300.0, key_func: Optional[Callable] = None):
    """Décorateur de cache optimisé."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Générer clé de cache
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"

            # Essayer le cache
            result = _global_cache.get(cache_key)
            if result is not None:
                return result

            # Calculer et cacher
            result = func(*args, **kwargs)
            _global_cache.set(cache_key, result)

            return result

        return wrapper
    return decorator


if HAS_NUMBA:
    @njit(cache=True, parallel=True)
    def fast_moving_average(prices: np.ndarray, window: int) -> np.ndarray:
        """Moving average optimisé avec Numba."""
        n = len(prices)
        result = np.empty(n)

        for i in range(n):
            if i < window - 1:
                result[i] = np.nan
            else:
                result[i] = np.mean(prices[i - window + 1:i + 1])

        return result

    @njit(cache=True)
    def fast_rsi(prices: np.ndarray, window: int = 14) -> np.ndarray:
        """RSI optimisé avec Numba."""
        n = len(prices)
        deltas = np.diff(prices)
        result = np.empty(n)
        result[0] = np.nan

        # Gains et pertes
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        # Première moyenne
        avg_gain = np.mean(gains[:window])
        avg_loss = np.mean(losses[:window])

        for i in range(window, n):
            if i == window:
                result[i] = 100 - (100 / (1 + avg_gain / max(avg_loss, 1e-10)))
            else:
                # EMA
                gain = gains[i - 1]
                loss = losses[i - 1]

                avg_gain = ((avg_gain * (window - 1)) + gain) / window
                avg_loss = ((avg_loss * (window - 1)) + loss) / window

                result[i] = 100 - (100 / (1 + avg_gain / max(avg_loss, 1e-10)))

        # Fill initial NaN values
        for i in range(min(window, n)):
            result[i] = np.nan

        return result

    @njit(cache=True, parallel=True)
    def fast_correlation(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
        """Corrélation roulante optimisée."""
        n = len(x)
        result = np.empty(n)

        for i in range(n):
            if i < window - 1:
                result[i] = np.nan
            else:
                x_window = x[i - window + 1:i + 1]
                y_window = y[i - window + 1:i + 1]

                # Correlation coefficient
                mean_x = np.mean(x_window)
                mean_y = np.mean(y_window)

                num = np.sum((x_window - mean_x) * (y_window - mean_y))
                den_x = np.sum((x_window - mean_x) ** 2)
                den_y = np.sum((y_window - mean_y) ** 2)

                if den_x * den_y > 0:
                    result[i] = num / np.sqrt(den_x * den_y)
                else:
                    result[i] = np.nan

        return result

else:
    # Fallback versions sans Numba
    def fast_moving_average(prices: np.ndarray, window: int) -> np.ndarray:
        """Fallback moving average."""
        return pd.Series(prices).rolling(window).mean().values

    def fast_rsi(prices: np.ndarray, window: int = 14) -> np.ndarray:
        """Fallback RSI."""
        deltas = np.diff(prices)
        gains = pd.Series(np.where(deltas > 0, deltas, 0))
        losses = pd.Series(np.where(deltas < 0, -deltas, 0))

        avg_gains = gains.ewm(span=window).mean()
        avg_losses = losses.ewm(span=window).mean()

        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        return np.concatenate([[np.nan], rsi.values])

    def fast_correlation(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
        """Fallback correlation."""
        return pd.Series(x).rolling(window).corr(pd.Series(y)).values


class OptimizedFeatureProcessor:
    """Processeur de features haute performance."""

    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        self.logger = LoggerFactory.get_logger("feature_processor")
        self.perf_logger = PerformanceLogger(self.logger)

        # Cache des fonctions compilées
        self.compiled_functions = {}

        # Statistics
        self.processing_stats = {
            'features_computed': 0,
            'cache_hits': 0,
            'total_time': 0.0
        }

    @optimized_cache(ttl=600.0)
    def compute_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcule des features techniques optimisées."""

        with self.perf_logger.measure_operation("technical_features"):

            # Utiliser arrays numpy pour performance
            prices = data['close'].values
            volumes = data['volume'].values if 'volume' in data.columns else None

            # Pool memory pour résultats
            result_data = {}

            # Features basiques
            if self.config.use_numba and HAS_NUMBA:
                result_data['sma_20'] = fast_moving_average(prices, 20)
                result_data['sma_50'] = fast_moving_average(prices, 50)
                result_data['rsi_14'] = fast_rsi(prices, 14)

                if volumes is not None:
                    result_data['volume_sma_20'] = fast_moving_average(volumes, 20)
                    result_data['price_volume_corr'] = fast_correlation(prices, volumes, 20)
            else:
                # Fallback pandas vectorized
                result_data['sma_20'] = data['close'].rolling(20).mean()
                result_data['sma_50'] = data['close'].rolling(50).mean()

                # RSI manuel optimisé
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                result_data['rsi_14'] = 100 - (100 / (1 + rs))

                if volumes is not None:
                    result_data['volume_sma_20'] = data['volume'].rolling(20).mean()
                    result_data['price_volume_corr'] = data['close'].rolling(20).corr(data['volume'])

            # Features avancées
            result_data['bollinger_upper'] = result_data['sma_20'] + (data['close'].rolling(20).std() * 2)
            result_data['bollinger_lower'] = result_data['sma_20'] - (data['close'].rolling(20).std() * 2)

            # MACD
            ema_12 = data['close'].ewm(span=12).mean()
            ema_26 = data['close'].ewm(span=26).mean()
            result_data['macd'] = ema_12 - ema_26
            result_data['macd_signal'] = result_data['macd'].ewm(span=9).mean()

            # Assembler résultat
            result_df = pd.DataFrame(result_data, index=data.index)

            self.processing_stats['features_computed'] += len(result_data)

            return result_df

    def batch_process_features(
        self,
        data_batches: List[pd.DataFrame],
        feature_functions: List[Callable] = None
    ) -> List[pd.DataFrame]:
        """Traitement par batch optimisé."""

        feature_functions = feature_functions or [self.compute_technical_features]

        with self.perf_logger.measure_operation("batch_processing"):

            if self.config.use_processes:
                executor = ProcessPoolExecutor(max_workers=self.config.max_workers)
            else:
                executor = ThreadPoolExecutor(max_workers=self.config.max_workers)

            try:
                # Soumettre tous les jobs
                futures = []
                for data_batch in data_batches:
                    for func in feature_functions:
                        future = executor.submit(func, data_batch)
                        futures.append(future)

                # Collecter résultats
                results = []
                for future in futures:
                    try:
                        result = future.result(timeout=30)
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Batch processing error: {e}")
                        results.append(pd.DataFrame())

                return results

            finally:
                executor.shutdown(wait=True)

    @performance_logged("feature_pipeline")
    def process_pipeline(
        self,
        data: pd.DataFrame,
        pipeline_steps: List[str] = None
    ) -> pd.DataFrame:
        """Pipeline de features optimisé."""

        pipeline_steps = pipeline_steps or ['technical', 'statistical', 'momentum']

        result = data.copy()

        for step in pipeline_steps:
            if step == 'technical':
                technical_features = self.compute_technical_features(data)
                result = pd.concat([result, technical_features], axis=1)

            elif step == 'statistical':
                # Features statistiques
                result['returns'] = data['close'].pct_change()
                result['log_returns'] = np.log(data['close'] / data['close'].shift(1))
                result['volatility_20'] = result['returns'].rolling(20).std()
                result['skewness_20'] = result['returns'].rolling(20).skew()
                result['kurtosis_20'] = result['returns'].rolling(20).kurt()

            elif step == 'momentum':
                # Features de momentum
                result['momentum_10'] = data['close'] / data['close'].shift(10) - 1
                result['momentum_20'] = data['close'] / data['close'].shift(20) - 1
                result['roc_10'] = ((data['close'] - data['close'].shift(10)) / data['close'].shift(10)) * 100

        return result

    def get_performance_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de performance."""
        stats = self.processing_stats.copy()
        stats['cache_hit_ratio'] = _global_cache.hit_ratio
        stats['memory_pool_usage'] = _memory_pool.allocated_bytes

        return stats


class StreamProcessor:
    """Processeur de stream temps réel optimisé."""

    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        self.logger = LoggerFactory.get_logger("stream_processor")

        # Buffer circulaire pour données temps réel
        self.buffer_size = 10000
        self.price_buffer = np.zeros(self.buffer_size)
        self.volume_buffer = np.zeros(self.buffer_size)
        self.timestamp_buffer = np.zeros(self.buffer_size, dtype='datetime64[ns]')

        self.current_index = 0
        self.buffer_full = False

        # Cache des derniers calculs
        self.feature_cache = {}
        self.last_update = 0

        # Lock pour thread safety
        self.lock = threading.RLock()

    def add_tick(self, timestamp: pd.Timestamp, price: float, volume: float = 0.0) -> None:
        """Ajoute un tick au buffer circulaire."""

        with self.lock:
            self.timestamp_buffer[self.current_index] = timestamp
            self.price_buffer[self.current_index] = price
            self.volume_buffer[self.current_index] = volume

            self.current_index = (self.current_index + 1) % self.buffer_size

            if self.current_index == 0:
                self.buffer_full = True

            self.last_update = time.time()

    def get_recent_data(self, lookback: int = 1000) -> pd.DataFrame:
        """Récupère les données récentes du buffer."""

        with self.lock:
            if not self.buffer_full and self.current_index < lookback:
                lookback = self.current_index

            if self.buffer_full:
                # Buffer complet, prendre les derniers lookback points
                start_idx = (self.current_index - lookback) % self.buffer_size

                if start_idx + lookback <= self.buffer_size:
                    # Pas de wrap around
                    prices = self.price_buffer[start_idx:start_idx + lookback]
                    volumes = self.volume_buffer[start_idx:start_idx + lookback]
                    timestamps = self.timestamp_buffer[start_idx:start_idx + lookback]
                else:
                    # Wrap around
                    split_point = self.buffer_size - start_idx
                    prices = np.concatenate([
                        self.price_buffer[start_idx:],
                        self.price_buffer[:lookback - split_point]
                    ])
                    volumes = np.concatenate([
                        self.volume_buffer[start_idx:],
                        self.volume_buffer[:lookback - split_point]
                    ])
                    timestamps = np.concatenate([
                        self.timestamp_buffer[start_idx:],
                        self.timestamp_buffer[:lookback - split_point]
                    ])
            else:
                # Buffer pas encore plein
                prices = self.price_buffer[:self.current_index]
                volumes = self.volume_buffer[:self.current_index]
                timestamps = self.timestamp_buffer[:self.current_index]

        return pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps),
            'close': prices,
            'volume': volumes
        }).set_index('timestamp')

    def compute_realtime_features(self, lookback: int = 100) -> Dict[str, float]:
        """Calcule des features en temps réel."""

        cache_key = f"features_{lookback}_{self.last_update}"

        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        data = self.get_recent_data(lookback)

        if len(data) < 20:  # Pas assez de données
            return {}

        features = {}

        # Features basiques temps réel
        prices = data['close'].values

        if HAS_NUMBA:
            features['sma_20'] = fast_moving_average(prices, min(20, len(prices)))[-1]
            features['rsi_14'] = fast_rsi(prices, min(14, len(prices)))[-1]
        else:
            features['sma_20'] = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)
            features['rsi_14'] = float(data['close'].pct_change().rolling(14).apply(
                lambda x: 100 - 100 / (1 + x[x > 0].mean() / abs(x[x < 0].mean()))
            ).iloc[-1]) if len(data) >= 14 else np.nan

        # Features de volatilité
        returns = np.diff(prices) / prices[:-1]
        features['volatility'] = np.std(returns) if len(returns) > 1 else 0.0

        # Momentum
        if len(prices) >= 10:
            features['momentum_10'] = (prices[-1] / prices[-10] - 1) * 100

        # Cache le résultat
        self.feature_cache[cache_key] = features

        # Nettoyer vieux cache
        if len(self.feature_cache) > 100:
            oldest_key = min(self.feature_cache.keys())
            del self.feature_cache[oldest_key]

        return features


class ParallelBacktester:
    """Backtester parallèle optimisé."""

    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        self.logger = LoggerFactory.get_logger("parallel_backtester")
        self.perf_logger = PerformanceLogger(self.logger)

    @performance_logged("parallel_backtest")
    def run_parallel_backtest(
        self,
        strategies: List[Callable],
        data_splits: List[pd.DataFrame],
        parameters_grid: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Exécute backtest parallèle sur grille de paramètres."""

        with self.perf_logger.measure_operation("backtest_setup"):
            # Préparer toutes les combinaisons
            test_combinations = []

            for strategy in strategies:
                for data_split in data_splits:
                    for params in parameters_grid:
                        test_combinations.append((strategy, data_split, params))

        self.logger.info(f"🚀 Lancement backtest parallèle: {len(test_combinations)} combinaisons")

        # Traitement parallèle
        if self.config.use_processes:
            executor = ProcessPoolExecutor(max_workers=self.config.max_workers)
        else:
            executor = ThreadPoolExecutor(max_workers=self.config.max_workers)

        try:
            futures = []
            for combination in test_combinations:
                future = executor.submit(self._run_single_backtest, *combination)
                futures.append(future)

            # Collecter résultats
            results = []
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=60)
                    result['combination_id'] = i
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Backtest {i} failed: {e}")
                    results.append({
                        'combination_id': i,
                        'error': str(e),
                        'success': False
                    })

            return results

        finally:
            executor.shutdown(wait=True)

    def _run_single_backtest(
        self,
        strategy: Callable,
        data: pd.DataFrame,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Exécute un backtest unique."""

        start_time = time.time()

        try:
            # Simuler backtest (remplacer par vraie logique)
            returns = data['close'].pct_change().dropna()

            # Métriques basiques
            total_return = (1 + returns).prod() - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = (returns.mean() * 252) / (volatility + 1e-8)

            max_dd = 0.0
            peak = 1.0
            for ret in returns:
                peak = max(peak, peak * (1 + ret))
                drawdown = (peak - peak * (1 + ret)) / peak
                max_dd = max(max_dd, drawdown)

            processing_time = time.time() - start_time

            return {
                'strategy': strategy.__name__ if hasattr(strategy, '__name__') else str(strategy),
                'parameters': parameters,
                'total_return': float(total_return),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_dd),
                'processing_time': processing_time,
                'success': True
            }

        except Exception as e:
            return {
                'strategy': strategy.__name__ if hasattr(strategy, '__name__') else str(strategy),
                'parameters': parameters,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'success': False
            }


# Factory functions
def create_optimized_processor(
    processor_type: str = "feature",
    config: PerformanceConfig = None
) -> Union[OptimizedFeatureProcessor, StreamProcessor, ParallelBacktester]:
    """Factory pour créer des processeurs optimisés."""

    config = config or PerformanceConfig()

    if processor_type == "feature":
        return OptimizedFeatureProcessor(config)
    elif processor_type == "stream":
        return StreamProcessor(config)
    elif processor_type == "backtest":
        return ParallelBacktester(config)
    else:
        raise ValueError(f"Unknown processor type: {processor_type}")


def cleanup_performance_resources():
    """Nettoie les ressources de performance."""
    global _memory_pool, _global_cache

    _memory_pool.cleanup()
    _global_cache.clear()
    gc.collect()


if __name__ == "__main__":
    # Démonstration des optimisations
    import time

    print("🏁 Test des optimisations performance QFrame")

    # Configuration haute performance
    config = PerformanceConfig(
        max_workers=4,
        use_numba=True,
        cache_size=256,
        batch_processing=True
    )

    # Données de test
    np.random.seed(42)
    n_points = 10000

    test_data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(n_points) * 0.02),
        'volume': np.random.randint(1000, 10000, n_points),
        'timestamp': pd.date_range('2023-01-01', periods=n_points, freq='1T')
    }).set_index('timestamp')

    # Test feature processor
    print(f"\n📊 Test Feature Processor sur {n_points} points")
    processor = create_optimized_processor("feature", config)

    start_time = time.time()
    features = processor.compute_technical_features(test_data)
    feature_time = time.time() - start_time

    print(f"✅ Features calculées en {feature_time:.3f}s")
    print(f"📈 {len(features.columns)} features générées")
    print(f"🎯 Stats: {processor.get_performance_stats()}")

    # Test stream processor
    print(f"\n⚡ Test Stream Processor")
    stream_proc = create_optimized_processor("stream", config)

    # Simuler stream de données
    for i in range(1000):
        timestamp = pd.Timestamp.now() + pd.Timedelta(seconds=i)
        price = 100 + np.random.randn() * 0.5
        stream_proc.add_tick(timestamp, price, np.random.randint(100, 1000))

    realtime_features = stream_proc.compute_realtime_features(100)
    print(f"✅ Features temps réel: {realtime_features}")

    # Nettoyage
    cleanup_performance_resources()
    print(f"\n🧹 Ressources nettoyées")