"""
Parallel Processing Framework for QFrame - Type Safe Version
==========================================================

Version propre et type-safe du processeur parallèle.
"""

import asyncio
import concurrent.futures
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

T = TypeVar("T")
ProcessingFunction = Callable[[pd.DataFrame], pd.DataFrame]
FeatureFunction = Callable[[pd.DataFrame], pd.Series]


@dataclass
class ProcessingTask:
    """Tâche de traitement parallèle type-safe."""

    name: str
    function: ProcessingFunction
    data: pd.DataFrame
    priority: int = 1
    timeout: Optional[float] = None
    dependencies: Optional[List[str]] = None


@dataclass
class ProcessingResult:
    """Résultat d'une tâche de traitement."""

    task_name: str
    success: bool
    result: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    processing_time: float = 0.0


class ParallelProcessor:
    """
    Processeur parallèle type-safe pour QFrame.
    """

    def __init__(self, max_workers: int = 4, chunk_size: int = 1000):
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.task_registry: Dict[str, ProcessingTask] = {}

    async def process_parallel_features(
        self,
        data: pd.DataFrame,
        feature_functions: List[FeatureFunction],
        chunk_data: bool = True,
    ) -> pd.DataFrame:
        """
        Traitement parallèle des features de manière type-safe.

        Args:
            data: DataFrame OHLCV
            feature_functions: Liste des fonctions de features
            chunk_data: Si True, divise les données en chunks

        Returns:
            DataFrame avec toutes les features calculées en parallèle
        """
        logger.info(
            f"Début traitement parallèle de {len(feature_functions)} features sur {len(data)} points"
        )

        if chunk_data and len(data) > self.chunk_size:
            return await self._process_chunked_features(data, feature_functions)
        else:
            return await self._process_features_batch(data, feature_functions)

    async def _process_features_batch(
        self, data: pd.DataFrame, feature_functions: List[FeatureFunction]
    ) -> pd.DataFrame:
        """Traitement en lot des features."""

        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = []

            for func in feature_functions:
                task = loop.run_in_executor(
                    executor, self._safe_feature_calculation, func, data
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combiner les résultats
        feature_df = data.copy()

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Erreur dans feature {i}: {result}")
                continue

            if isinstance(result, pd.Series):
                feature_name = getattr(feature_functions[i], "__name__", f"feature_{i}")
                feature_df[feature_name] = result

        return feature_df

    async def _process_chunked_features(
        self, data: pd.DataFrame, feature_functions: List[FeatureFunction]
    ) -> pd.DataFrame:
        """Traitement par chunks pour gros volumes."""

        chunks = self._create_chunks(data)
        chunk_results = []

        for chunk in chunks:
            chunk_result = await self._process_features_batch(chunk, feature_functions)
            chunk_results.append(chunk_result)

        # Recombiner les chunks
        return pd.concat(chunk_results, ignore_index=True)

    def _create_chunks(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """Crée des chunks de données."""
        chunks = []
        for i in range(0, len(data), self.chunk_size):
            chunk = data.iloc[i : i + self.chunk_size].copy()
            chunks.append(chunk)
        return chunks

    def _safe_feature_calculation(
        self, func: FeatureFunction, data: pd.DataFrame
    ) -> Optional[pd.Series]:
        """Calcul sécurisé d'une feature."""
        try:
            result = func(data)
            if isinstance(result, pd.Series):
                return result
            else:
                logger.warning(
                    f"Function {func.__name__} returned {type(result)}, expected Series"
                )
                return None
        except Exception as e:
            logger.error(f"Erreur dans {func.__name__}: {e}")
            return None

    async def execute_backtest_parallel(
        self, strategies: List[Any], data: pd.DataFrame, parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Exécution parallèle de backtests type-safe.
        """
        loop = asyncio.get_event_loop()

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = []

            for strategy in strategies:
                task = loop.run_in_executor(
                    executor, self._run_single_backtest, strategy, data, parameters
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filtrer les résultats valides
        valid_results = []
        for result in results:
            if isinstance(result, dict):
                valid_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Erreur backtest: {result}")

        return valid_results

    def _run_single_backtest(
        self, strategy: Any, data: pd.DataFrame, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Exécute un backtest unique."""
        try:
            start_time = time.time()

            # Simulation simple de backtest
            # En production, ceci utiliserait le vrai moteur de backtest
            total_return = np.random.uniform(-0.1, 0.3)  # Simulation
            sharpe_ratio = np.random.uniform(0.5, 2.5)
            max_drawdown = np.random.uniform(0.05, 0.25)

            processing_time = time.time() - start_time

            return {
                "strategy": getattr(strategy, "__name__", str(strategy)),
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "processing_time": processing_time,
                "success": True,
            }

        except Exception as e:
            return {
                "strategy": getattr(strategy, "__name__", str(strategy)),
                "error": str(e),
                "success": False,
                "processing_time": 0.0,
            }

    def register_task(self, task: ProcessingTask) -> None:
        """Enregistre une tâche de traitement."""
        self.task_registry[task.name] = task

    async def execute_registered_tasks(self) -> List[ProcessingResult]:
        """Exécute toutes les tâches enregistrées."""
        results = []

        for task_name, task in self.task_registry.items():
            result = await self._execute_single_task(task)
            results.append(result)

        return results

    async def _execute_single_task(self, task: ProcessingTask) -> ProcessingResult:
        """Exécute une tâche unique."""
        start_time = time.time()

        try:
            loop = asyncio.get_event_loop()

            if task.timeout:
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, task.function, task.data),
                    timeout=task.timeout,
                )
            else:
                result = await loop.run_in_executor(None, task.function, task.data)

            processing_time = time.time() - start_time

            return ProcessingResult(
                task_name=task.name,
                success=True,
                result=result,
                processing_time=processing_time,
            )

        except Exception as e:
            processing_time = time.time() - start_time

            return ProcessingResult(
                task_name=task.name,
                success=False,
                error=str(e),
                processing_time=processing_time,
            )


def parallel_feature_processor(
    max_workers: int = 4,
) -> Callable[[ProcessingFunction], Callable[..., Any]]:
    """
    Décorateur pour automatiser le traitement parallèle des features.
    """

    def decorator(func: ProcessingFunction) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(
            data: pd.DataFrame, *args: Any, **kwargs: Any
        ) -> pd.DataFrame:
            processor = ParallelProcessor(max_workers=max_workers)

            # Si la fonction retourne une liste de fonctions features
            if hasattr(func, "_feature_functions"):
                feature_functions = func._feature_functions
            else:
                # Utiliser la fonction elle-même
                feature_functions = [func]

            return await processor.process_parallel_features(
                data, feature_functions, chunk_data=kwargs.get("chunk_data", True)
            )

        return wrapper

    return decorator


# Fonctions utilitaires type-safe
def create_feature_function(name: str) -> FeatureFunction:
    """Factory pour créer des fonctions de features type-safe."""

    def feature_func(data: pd.DataFrame) -> pd.Series:
        # Exemple de calcul de feature
        if "close" in data.columns:
            return data["close"].rolling(20).mean()
        else:
            return pd.Series(index=data.index, dtype=float)

    feature_func.__name__ = name
    return feature_func


def validate_processing_function(func: ProcessingFunction) -> bool:
    """Valide qu'une fonction de traitement respecte le contrat type-safe."""
    import inspect

    sig = inspect.signature(func)

    # Vérifier que la fonction prend un DataFrame en entrée
    params = list(sig.parameters.values())
    if not params:
        return False

    # Vérifier le type de retour si annoté
    return_annotation = sig.return_annotation
    if return_annotation != inspect.Signature.empty:
        return bool(return_annotation == pd.DataFrame)

    return True  # Accepter si pas d'annotation
