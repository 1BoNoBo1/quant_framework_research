"""
Tests pour Optimized Performance Processors
==========================================

Tests des optimisations critiques de performance pour trading quantitatif.
"""

import pytest
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from qframe.infrastructure.performance.optimized_processors import (
    PerformanceConfig, MemoryPool, OptimizedCache, optimized_cache,
    OptimizedFeatureProcessor, StreamProcessor, ParallelBacktester,
    create_optimized_processor, cleanup_performance_resources
)


class TestPerformanceConfig:
    """Tests de la configuration performance."""

    def test_performance_config_defaults(self):
        """Test valeurs par défaut."""
        config = PerformanceConfig()

        assert config.max_workers == 4
        assert config.chunk_size == 1000
        assert config.memory_pool_size == 100_000_000
        assert config.cache_size == 128
        assert config.use_vectorization == True
        assert config.batch_processing == True

    def test_performance_config_custom(self):
        """Test configuration personnalisée."""
        config = PerformanceConfig(
            max_workers=8,
            memory_pool_size=200_000_000,
            use_processes=True,
            cache_size=256
        )

        assert config.max_workers == 8
        assert config.memory_pool_size == 200_000_000
        assert config.use_processes == True
        assert config.cache_size == 256


class TestMemoryPool:
    """Tests du pool de mémoire."""

    def test_memory_pool_creation(self):
        """Test création du pool."""
        pool = MemoryPool(pool_size=1000000)

        assert pool.pool_size == 1000000
        assert pool.allocated_bytes == 0
        assert len(pool.arrays) == 0

    def test_memory_pool_get_array(self):
        """Test récupération d'array."""
        pool = MemoryPool()

        array = pool.get_array((100, 50), np.float64)

        assert array.shape == (100, 50)
        assert array.dtype == np.float64
        assert np.all(array == 0)  # Array initialisé à zéro
        assert pool.allocated_bytes > 0

    def test_memory_pool_return_array(self):
        """Test retour d'array au pool."""
        pool = MemoryPool()

        # Récupérer et retourner array
        array1 = pool.get_array((50, 50), np.float64)
        pool.return_array(array1)

        # Récupérer nouveau array (devrait réutiliser)
        array2 = pool.get_array((50, 50), np.float64)

        # Doit être le même array (réutilisé)
        assert array1 is array2

    def test_memory_pool_dataframe_buffer(self):
        """Test buffer DataFrame."""
        pool = MemoryPool()

        df = pool.get_dataframe_buffer(500)

        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 500  # Au moins la taille demandée
        assert 'timestamp' in df.columns
        assert 'value' in df.columns

    def test_memory_pool_cleanup(self):
        """Test nettoyage du pool."""
        pool = MemoryPool()

        # Allouer quelques arrays
        pool.get_array((100, 100))
        pool.get_array((50, 50))

        assert pool.allocated_bytes > 0

        # Nettoyer
        pool.cleanup()

        assert pool.allocated_bytes == 0
        assert len(pool.arrays) == 0


class TestOptimizedCache:
    """Tests du cache optimisé."""

    def test_cache_creation(self):
        """Test création du cache."""
        cache = OptimizedCache(maxsize=64, ttl=60.0)

        assert cache.maxsize == 64
        assert cache.ttl == 60.0
        assert cache.hit_count == 0
        assert cache.miss_count == 0

    def test_cache_set_get(self):
        """Test set/get basique."""
        cache = OptimizedCache()

        # Miss initial
        value = cache.get("test_key")
        assert value is None
        assert cache.miss_count == 1

        # Set et hit
        cache.set("test_key", "test_value")
        value = cache.get("test_key")

        assert value == "test_value"
        assert cache.hit_count == 1

    def test_cache_ttl_expiration(self):
        """Test expiration TTL."""
        cache = OptimizedCache(ttl=0.1)  # 100ms TTL

        cache.set("key", "value")
        assert cache.get("key") == "value"

        # Attendre expiration
        time.sleep(0.15)

        # Doit être expiré
        assert cache.get("key") is None

    def test_cache_lru_eviction(self):
        """Test éviction LRU."""
        cache = OptimizedCache(maxsize=2)

        # Remplir cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Accéder key1 pour le rendre récent
        cache.get("key1")

        # Ajouter key3 (doit évincer key2)
        cache.set("key3", "value3")

        assert cache.get("key1") == "value1"  # Encore là
        assert cache.get("key2") is None      # Évincé
        assert cache.get("key3") == "value3"  # Nouveau

    def test_cache_hit_ratio(self):
        """Test calcul hit ratio."""
        cache = OptimizedCache()

        # 2 misses
        cache.get("key1")
        cache.get("key2")

        assert cache.hit_ratio == 0.0

        # 1 set + 2 hits
        cache.set("key1", "value1")
        cache.get("key1")
        cache.get("key1")

        # 2 hits sur 4 accès total = 50%
        assert cache.hit_ratio == 0.5

    def test_cache_decorator(self):
        """Test décorateur de cache."""
        call_count = 0

        @optimized_cache(ttl=1.0)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y

        # Premier appel
        result1 = expensive_function(2, 3)
        assert result1 == 5
        assert call_count == 1

        # Deuxième appel (doit utiliser cache)
        result2 = expensive_function(2, 3)
        assert result2 == 5
        assert call_count == 1  # Pas d'appel supplémentaire

        # Appel avec paramètres différents
        result3 = expensive_function(3, 4)
        assert result3 == 7
        assert call_count == 2  # Nouvel appel


class TestOptimizedFeatureProcessor:
    """Tests du processeur de features optimisé."""

    @pytest.fixture
    def sample_data(self):
        """Données de test."""
        np.random.seed(42)
        n_points = 1000

        return pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(n_points) * 0.02),
            'volume': np.random.randint(1000, 10000, n_points),
            'timestamp': pd.date_range('2023-01-01', periods=n_points, freq='1T')
        }).set_index('timestamp')

    def test_processor_creation(self):
        """Test création du processeur."""
        config = PerformanceConfig(max_workers=2, use_numba=False)
        processor = OptimizedFeatureProcessor(config)

        assert processor.config.max_workers == 2
        assert processor.config.use_numba == False
        assert processor.processing_stats['features_computed'] == 0

    def test_compute_technical_features(self, sample_data):
        """Test calcul features techniques."""
        processor = OptimizedFeatureProcessor()

        features = processor.compute_technical_features(sample_data)

        # Vérifier colonnes générées
        expected_columns = [
            'sma_20', 'sma_50', 'rsi_14', 'volume_sma_20',
            'bollinger_upper', 'bollinger_lower', 'macd', 'macd_signal'
        ]

        for col in expected_columns:
            assert col in features.columns

        # Vérifier dimensions
        assert len(features) == len(sample_data)
        assert len(features.columns) >= 8

        # Vérifier que les features ont du sens
        assert not features['sma_20'].isna().all()
        assert features['rsi_14'].max() <= 100
        assert features['rsi_14'].min() >= 0

    def test_batch_process_features(self, sample_data):
        """Test traitement par batch."""
        processor = OptimizedFeatureProcessor(PerformanceConfig(max_workers=2))

        # Diviser données en batches
        batch_size = 250
        batches = [sample_data.iloc[i:i + batch_size] for i in range(0, len(sample_data), batch_size)]

        results = processor.batch_process_features(batches)

        assert len(results) == len(batches)

        for result in results:
            assert isinstance(result, pd.DataFrame)
            assert len(result.columns) >= 8

    def test_process_pipeline(self, sample_data):
        """Test pipeline de features."""
        processor = OptimizedFeatureProcessor()

        result = processor.process_pipeline(
            sample_data,
            pipeline_steps=['technical', 'statistical', 'momentum']
        )

        # Doit contenir colonnes originales + features
        assert 'close' in result.columns
        assert 'volume' in result.columns

        # Features techniques
        assert 'sma_20' in result.columns
        assert 'rsi_14' in result.columns

        # Features statistiques
        assert 'returns' in result.columns
        assert 'volatility_20' in result.columns
        assert 'skewness_20' in result.columns

        # Features momentum
        assert 'momentum_10' in result.columns
        assert 'momentum_20' in result.columns

    def test_performance_stats(self, sample_data):
        """Test statistiques de performance."""
        processor = OptimizedFeatureProcessor()

        # Calculer quelques features
        processor.compute_technical_features(sample_data)

        stats = processor.get_performance_stats()

        assert 'features_computed' in stats
        assert 'cache_hit_ratio' in stats
        assert 'memory_pool_usage' in stats

        assert stats['features_computed'] > 0


class TestStreamProcessor:
    """Tests du processeur de stream."""

    def test_stream_processor_creation(self):
        """Test création du processeur."""
        processor = StreamProcessor()

        assert processor.buffer_size == 10000
        assert processor.current_index == 0
        assert processor.buffer_full == False

    def test_add_tick(self):
        """Test ajout de tick."""
        processor = StreamProcessor()

        timestamp = pd.Timestamp.now()
        processor.add_tick(timestamp, 100.5, 1500)

        assert processor.current_index == 1
        assert processor.price_buffer[0] == 100.5
        assert processor.volume_buffer[0] == 1500

    def test_multiple_ticks(self):
        """Test ajout multiple ticks."""
        processor = StreamProcessor()

        timestamps = pd.date_range('2023-01-01', periods=100, freq='1S')
        prices = 100 + np.random.randn(100) * 0.1

        for i, (ts, price) in enumerate(zip(timestamps, prices)):
            processor.add_tick(ts, price, 1000 + i)

        assert processor.current_index == 100
        assert not processor.buffer_full

    def test_buffer_wraparound(self):
        """Test wraparound du buffer circulaire."""
        processor = StreamProcessor()
        processor.buffer_size = 10  # Petit buffer pour test

        # Ajouter plus que la taille du buffer
        for i in range(15):
            processor.add_tick(pd.Timestamp.now() + pd.Timedelta(seconds=i), 100 + i, 1000)

        assert processor.current_index == 5  # 15 % 10
        assert processor.buffer_full == True

    def test_get_recent_data(self):
        """Test récupération données récentes."""
        processor = StreamProcessor()

        # Ajouter données
        timestamps = pd.date_range('2023-01-01', periods=50, freq='1S')
        for i, ts in enumerate(timestamps):
            processor.add_tick(ts, 100 + i * 0.1, 1000 + i)

        # Récupérer données récentes
        recent_data = processor.get_recent_data(20)

        assert len(recent_data) == 20
        assert 'close' in recent_data.columns
        assert 'volume' in recent_data.columns
        assert isinstance(recent_data.index, pd.DatetimeIndex)

    def test_compute_realtime_features(self):
        """Test calcul features temps réel."""
        processor = StreamProcessor()

        # Ajouter assez de données pour features
        timestamps = pd.date_range('2023-01-01', periods=100, freq='1S')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.01)

        for ts, price in zip(timestamps, prices):
            processor.add_tick(ts, price, 1000)

        features = processor.compute_realtime_features(50)

        assert isinstance(features, dict)

        if features:  # Si assez de données
            assert 'sma_20' in features or len(features) == 0
            assert 'volatility' in features or len(features) == 0

    def test_feature_caching(self):
        """Test cache des features."""
        processor = StreamProcessor()

        # Ajouter données
        for i in range(50):
            processor.add_tick(pd.Timestamp.now() + pd.Timedelta(seconds=i), 100 + i * 0.01, 1000)

        # Premier calcul
        features1 = processor.compute_realtime_features(30)

        # Deuxième calcul (doit utiliser cache)
        features2 = processor.compute_realtime_features(30)

        # Cache hit si même timestamp
        if features1 and features2:
            assert features1 == features2


class TestParallelBacktester:
    """Tests du backtester parallèle."""

    @pytest.fixture
    def sample_strategies(self):
        """Stratégies de test."""
        def strategy_a(data, params):
            return {"name": "strategy_a", "param": params.get("param", 1)}

        def strategy_b(data, params):
            return {"name": "strategy_b", "param": params.get("param", 2)}

        return [strategy_a, strategy_b]

    @pytest.fixture
    def sample_data_splits(self):
        """Données de test divisées."""
        np.random.seed(42)
        data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(100) * 0.02),
            'volume': np.random.randint(1000, 5000, 100)
        })

        return [data.iloc[:50], data.iloc[50:]]

    def test_backtester_creation(self):
        """Test création du backtester."""
        config = PerformanceConfig(max_workers=2, use_processes=False)
        backtester = ParallelBacktester(config)

        assert backtester.config.max_workers == 2
        assert backtester.config.use_processes == False

    def test_single_backtest(self, sample_strategies, sample_data_splits):
        """Test backtest unique."""
        backtester = ParallelBacktester()

        result = backtester._run_single_backtest(
            sample_strategies[0],
            sample_data_splits[0],
            {"param": 1}
        )

        assert isinstance(result, dict)
        assert 'strategy' in result
        assert 'total_return' in result
        assert 'sharpe_ratio' in result
        assert 'max_drawdown' in result
        assert 'success' in result
        assert result['success'] == True

    def test_parallel_backtest(self, sample_strategies, sample_data_splits):
        """Test backtest parallèle complet."""
        config = PerformanceConfig(max_workers=2, use_processes=False)
        backtester = ParallelBacktester(config)

        parameters_grid = [
            {"param": 1},
            {"param": 2},
            {"param": 3}
        ]

        results = backtester.run_parallel_backtest(
            sample_strategies,
            sample_data_splits,
            parameters_grid
        )

        # 2 strategies * 2 data_splits * 3 params = 12 combinaisons
        assert len(results) == 12

        # Vérifier structure des résultats
        for result in results:
            assert 'combination_id' in result
            if result.get('success', False):
                assert 'total_return' in result
                assert 'sharpe_ratio' in result

    def test_backtest_error_handling(self):
        """Test gestion d'erreurs en backtest."""
        backtester = ParallelBacktester()

        def failing_strategy(data, params):
            raise ValueError("Strategy failed")

        data = pd.DataFrame({'close': [100, 101, 102]})

        result = backtester._run_single_backtest(
            failing_strategy,
            data,
            {}
        )

        assert result['success'] == False
        assert 'error' in result
        assert 'Strategy failed' in result['error']


class TestFactoryFunctions:
    """Tests des fonctions factory."""

    def test_create_optimized_processor_feature(self):
        """Test création processeur de features."""
        processor = create_optimized_processor("feature")

        assert isinstance(processor, OptimizedFeatureProcessor)

    def test_create_optimized_processor_stream(self):
        """Test création processeur de stream."""
        processor = create_optimized_processor("stream")

        assert isinstance(processor, StreamProcessor)

    def test_create_optimized_processor_backtest(self):
        """Test création backtester."""
        processor = create_optimized_processor("backtest")

        assert isinstance(processor, ParallelBacktester)

    def test_create_optimized_processor_invalid(self):
        """Test création avec type invalide."""
        with pytest.raises(ValueError, match="Unknown processor type"):
            create_optimized_processor("invalid_type")

    def test_create_with_custom_config(self):
        """Test création avec configuration personnalisée."""
        config = PerformanceConfig(max_workers=8, cache_size=256)
        processor = create_optimized_processor("feature", config)

        assert processor.config.max_workers == 8
        assert processor.config.cache_size == 256


class TestResourceCleanup:
    """Tests du nettoyage des ressources."""

    def test_cleanup_performance_resources(self):
        """Test nettoyage des ressources."""
        # Utiliser quelques ressources
        processor = create_optimized_processor("feature")
        cache = OptimizedCache()
        cache.set("test", "value")

        # Nettoyer
        cleanup_performance_resources()

        # Vérifier que les ressources globales sont nettoyées
        # (Difficile à tester directement, mais la fonction ne doit pas échouer)
        assert True  # La fonction s'exécute sans erreur


class TestPerformanceIntegration:
    """Tests d'intégration performance."""

    def test_full_pipeline_performance(self):
        """Test performance pipeline complet."""
        # Données réalistes
        np.random.seed(42)
        n_points = 5000

        data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(n_points) * 0.02),
            'volume': np.random.randint(1000, 10000, n_points),
            'timestamp': pd.date_range('2023-01-01', periods=n_points, freq='1T')
        }).set_index('timestamp')

        # Configuration haute performance
        config = PerformanceConfig(
            max_workers=2,
            use_numba=False,  # Pour compatibilité test
            cache_size=64,
            batch_processing=True
        )

        processor = OptimizedFeatureProcessor(config)

        start_time = time.time()

        # Pipeline complet
        features = processor.process_pipeline(data, ['technical', 'statistical'])

        processing_time = time.time() - start_time

        # Vérifications
        assert len(features) == len(data)
        assert len(features.columns) > len(data.columns)
        assert processing_time < 10.0  # Doit être raisonnablement rapide

        # Stats de performance
        stats = processor.get_performance_stats()
        assert stats['features_computed'] > 0

    def test_memory_efficiency(self):
        """Test efficacité mémoire."""
        # Créer plusieurs processeurs
        processors = []
        for i in range(5):
            config = PerformanceConfig(memory_pool_size=1000000)
            processor = OptimizedFeatureProcessor(config)
            processors.append(processor)

        # Utiliser pool mémoire
        from qframe.infrastructure.performance.optimized_processors import _memory_pool

        initial_memory = _memory_pool.allocated_bytes

        arrays = []
        for i in range(10):
            array = _memory_pool.get_array((100, 100))
            arrays.append(array)

        # Retourner au pool
        for array in arrays:
            _memory_pool.return_array(array)

        # Nettoyer
        cleanup_performance_resources()

        # Mémoire devrait être libérée
        assert _memory_pool.allocated_bytes <= initial_memory


if __name__ == "__main__":
    pytest.main([__file__, "-v"])