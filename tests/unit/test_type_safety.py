"""
Tests de validation de la type safety améliorée
==============================================

Tests pour valider que les corrections MyPy fonctionnent correctement.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

from qframe.core.interfaces import Signal, SignalAction, Position
from qframe.core.parallel_processor_fixed import (
    ParallelProcessor, ProcessingTask, ProcessingResult,
    create_feature_function, validate_processing_function
)


class TestTypeSafetyImprovements:
    """Tests des améliorations de type safety."""

    def test_signal_with_default_factory(self):
        """Test que Signal utilise field(default_factory=dict) correctement."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol="BTC/USD",
            action=SignalAction.BUY,
            strength=0.8
        )

        assert signal.metadata == {}
        assert isinstance(signal.metadata, dict)

        # Test modification du metadata
        signal.metadata['test'] = 'value'
        assert signal.metadata['test'] == 'value'

    def test_position_with_default_factory(self):
        """Test que Position utilise field(default_factory=dict) correctement."""
        position = Position(
            symbol="ETH/USD",
            size=1.5,
            entry_price=2000.0,
            current_price=2100.0,
            timestamp=datetime.now()
        )

        assert position.metadata == {}
        assert isinstance(position.metadata, dict)

    def test_parallel_processor_type_safety(self):
        """Test du parallel processor type-safe."""
        processor = ParallelProcessor(max_workers=2, chunk_size=100)

        # Test création de tâche
        def dummy_processing_func(data: pd.DataFrame) -> pd.DataFrame:
            return data.copy()

        task = ProcessingTask(
            name="test_task",
            function=dummy_processing_func,
            data=pd.DataFrame({'col1': [1, 2, 3]})
        )

        assert task.name == "test_task"
        assert task.dependencies is None  # Default None value
        assert callable(task.function)

    @pytest.mark.asyncio
    async def test_parallel_feature_processing(self):
        """Test traitement parallèle des features type-safe."""
        processor = ParallelProcessor(max_workers=2)

        # Données test
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })

        # Fonctions features type-safe
        def sma_feature(df: pd.DataFrame) -> pd.Series:
            return df['close'].rolling(2).mean()

        def volume_feature(df: pd.DataFrame) -> pd.Series:
            return df['volume'].rolling(2).mean()

        feature_functions = [sma_feature, volume_feature]

        result = await processor.process_parallel_features(
            data, feature_functions, chunk_data=False
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)
        assert 'sma_feature' in result.columns
        assert 'volume_feature' in result.columns

    def test_create_feature_function_factory(self):
        """Test de la factory pour créer des fonctions features type-safe."""
        feature_func = create_feature_function("test_ma")

        assert callable(feature_func)
        assert feature_func.__name__ == "test_ma"

        # Test exécution
        data = pd.DataFrame({'close': [100, 101, 102, 103, 104]})
        result = feature_func(data)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_validate_processing_function(self):
        """Test validation des fonctions de traitement."""

        # Fonction valide
        def valid_func(data: pd.DataFrame) -> pd.DataFrame:
            return data.copy()

        # Fonction invalide (pas de paramètres)
        def invalid_func() -> pd.DataFrame:
            return pd.DataFrame()

        assert validate_processing_function(valid_func) == True
        assert validate_processing_function(invalid_func) == False

    @pytest.mark.asyncio
    async def test_processing_result_type_safety(self):
        """Test des résultats de traitement type-safe."""
        processor = ParallelProcessor()

        def success_func(data: pd.DataFrame) -> pd.DataFrame:
            return data.copy()

        def error_func(data: pd.DataFrame) -> pd.DataFrame:
            raise ValueError("Test error")

        # Test tâche réussie
        success_task = ProcessingTask(
            name="success_task",
            function=success_func,
            data=pd.DataFrame({'test': [1, 2, 3]})
        )

        result = await processor._execute_single_task(success_task)

        assert isinstance(result, ProcessingResult)
        assert result.success == True
        assert result.error is None
        assert isinstance(result.result, pd.DataFrame)
        assert result.processing_time > 0

        # Test tâche avec erreur
        error_task = ProcessingTask(
            name="error_task",
            function=error_func,
            data=pd.DataFrame({'test': [1, 2, 3]})
        )

        error_result = await processor._execute_single_task(error_task)

        assert isinstance(error_result, ProcessingResult)
        assert error_result.success == False
        assert error_result.error is not None
        assert error_result.result is None

    def test_optional_types_explicit(self):
        """Test que les types Optional sont explicites."""
        task = ProcessingTask(
            name="test",
            function=lambda x: x,
            data=pd.DataFrame(),
            dependencies=None  # Devrait accepter None explicitement
        )

        assert task.dependencies is None  # None explicite comme défini

    @pytest.mark.asyncio
    async def test_chunked_processing_type_safety(self):
        """Test du traitement par chunks type-safe."""
        processor = ParallelProcessor(max_workers=2, chunk_size=3)

        # Grande dataset pour forcer le chunking
        data = pd.DataFrame({
            'close': range(10),
            'volume': range(100, 110)
        })

        def simple_feature(df: pd.DataFrame) -> pd.Series:
            return df['close'] * 2

        result = await processor.process_parallel_features(
            data, [simple_feature], chunk_data=True
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)
        assert 'simple_feature' in result.columns
        assert (result['simple_feature'] == data['close'] * 2).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])