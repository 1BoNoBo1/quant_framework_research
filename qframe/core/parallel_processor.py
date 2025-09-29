"""
Parallel Processing Framework for QFrame
=======================================

Implémente le traitement parallèle obligatoire pour toutes les opérations de données,
basé sur les meilleures pratiques Data Science de Claude Flow.
"""

import asyncio
import concurrent.futures
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ProcessingTask:
    """Tâche de traitement parallèle."""

    name: str
    function: Callable[..., Any]
    data: Any
    priority: int = 1
    timeout: Optional[float] = None
    dependencies: Optional[List[str]] = None


class ParallelProcessor:
    """
    Processeur parallèle pour opérations financières.

    Implémente les patterns de Claude Flow pour maximiser les performances
    sur les calculs quantitatifs intensifs.
    """

    def __init__(
        self, max_workers: int = 4, use_processes: bool = True, chunk_size: int = 1000
    ):
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.chunk_size = chunk_size
        self.executor = (
            ProcessPoolExecutor(max_workers)
            if use_processes
            else ThreadPoolExecutor(max_workers)
        )
        self.task_registry: Dict[str, ProcessingTask] = {}

    async def process_parallel_features(
        self,
        data: pd.DataFrame,
        feature_functions: List[Callable[..., Any]],
        chunk_data: bool = True,
    ) -> pd.DataFrame:
        """
        Traitement parallèle des features pour trading quantitatif.

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
        self, data: pd.DataFrame, feature_functions: List[Callable[..., Any]]
    ) -> pd.DataFrame:
        """Traitement batch des features."""

        loop = asyncio.get_event_loop()

        # Soumettre toutes les tâches en parallèle
        tasks = []
        for func in feature_functions:
            task = loop.run_in_executor(
                self.executor, self._safe_feature_calculation, func, data.copy()
            )
            tasks.append(task)

        # Attendre tous les résultats
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combiner les résultats
        combined_features = data.copy()
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(
                    f"Erreur dans feature {feature_functions[i].__name__}: {result}"
                )
            elif isinstance(result, pd.DataFrame):
                # Merge des nouvelles colonnes
                for col in result.columns:
                    if col not in combined_features.columns:
                        combined_features[col] = result[col]

        logger.info(
            f"Traitement parallèle terminé: {len(combined_features.columns)} colonnes"
        )
        return combined_features

    async def _process_chunked_features(
        self, data: pd.DataFrame, feature_functions: List[Callable[..., Any]]
    ) -> pd.DataFrame:
        """Traitement par chunks pour gros datasets."""

        chunks = [
            data[i : i + self.chunk_size] for i in range(0, len(data), self.chunk_size)
        ]
        logger.info(f"Traitement en {len(chunks)} chunks de {self.chunk_size} lignes")

        # Traiter chaque chunk en parallèle
        chunk_tasks = []
        for chunk in chunks:
            task = self._process_features_batch(chunk, feature_functions)
            chunk_tasks.append(task)

        processed_chunks = await asyncio.gather(*chunk_tasks)

        # Recombiner les chunks
        return pd.concat(processed_chunks, ignore_index=True)

    def _safe_feature_calculation(
        self, feature_func: Callable, data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calcul sécurisé de feature avec gestion d'erreurs."""
        try:
            start_time = time.time()

            # Créer une copie propre des données pour éviter les problèmes de sérialisation
            clean_data = data.copy()

            result = feature_func(clean_data)
            duration = time.time() - start_time

            logger.debug(f"Feature {feature_func.__name__} calculée en {duration:.3f}s")

            # S'assurer que le résultat est un DataFrame propre
            if isinstance(result, pd.DataFrame):
                return result.copy()
            elif isinstance(result, pd.Series):
                return pd.DataFrame(result).copy()
            else:
                return pd.DataFrame(index=data.index)

        except Exception as e:
            logger.error(f"Erreur dans {feature_func.__name__}: {str(e)}")
            # Retourner DataFrame vide avec même index
            return pd.DataFrame(index=data.index)

    async def parallel_risk_metrics(
        self, returns: pd.Series, confidence_levels: List[float] = [0.95, 0.99]
    ) -> Dict[str, float]:
        """
        Calcul parallèle des métriques de risque.

        Implémente les recommendations Claude Flow pour analyse de risque.
        """
        logger.info(
            f"Calcul parallèle des métriques de risque sur {len(returns)} returns"
        )

        loop = asyncio.get_event_loop()

        # Définir toutes les métriques à calculer
        risk_tasks = {
            "var_95": loop.run_in_executor(
                self.executor, self._calculate_var, returns, 0.95
            ),
            "var_99": loop.run_in_executor(
                self.executor, self._calculate_var, returns, 0.99
            ),
            "cvar_95": loop.run_in_executor(
                self.executor, self._calculate_cvar, returns, 0.95
            ),
            "cvar_99": loop.run_in_executor(
                self.executor, self._calculate_cvar, returns, 0.99
            ),
            "max_drawdown": loop.run_in_executor(
                self.executor, self._calculate_max_drawdown, returns
            ),
            "sharpe_ratio": loop.run_in_executor(
                self.executor, self._calculate_sharpe, returns
            ),
            "sortino_ratio": loop.run_in_executor(
                self.executor, self._calculate_sortino, returns
            ),
            "calmar_ratio": loop.run_in_executor(
                self.executor, self._calculate_calmar, returns
            ),
            "skewness": loop.run_in_executor(
                self.executor, self._calculate_skewness, returns
            ),
            "kurtosis": loop.run_in_executor(
                self.executor, self._calculate_kurtosis, returns
            ),
        }

        # Exécuter tous les calculs en parallèle
        results = await asyncio.gather(*risk_tasks.values(), return_exceptions=True)

        # Construire le dictionnaire des résultats
        risk_metrics = {}
        for i, (metric_name, result) in enumerate(zip(risk_tasks.keys(), results)):
            if isinstance(result, Exception):
                logger.warning(f"Erreur calcul {metric_name}: {result}")
                risk_metrics[metric_name] = np.nan
            else:
                risk_metrics[metric_name] = result

        logger.info(f"Métriques de risque calculées: {list(risk_metrics.keys())}")
        return risk_metrics

    async def parallel_backtest(
        self,
        strategy_func: Callable,
        data_chunks: List[pd.DataFrame],
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Backtesting parallèle sur multiple périodes/paramètres.
        """
        logger.info(f"Début backtesting parallèle sur {len(data_chunks)} périodes")

        loop = asyncio.get_event_loop()

        # Soumettre tous les backtests en parallèle
        backtest_tasks = []
        for i, chunk in enumerate(data_chunks):
            task = loop.run_in_executor(
                self.executor,
                self._run_single_backtest,
                strategy_func,
                chunk,
                parameters,
                f"period_{i}",
            )
            backtest_tasks.append(task)

        # Attendre tous les résultats
        results = await asyncio.gather(*backtest_tasks, return_exceptions=True)

        # Agréger les résultats
        aggregated_results = self._aggregate_backtest_results(results)

        logger.info("Backtesting parallèle terminé")
        return aggregated_results

    def _run_single_backtest(
        self,
        strategy_func: Callable,
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        period_name: str,
    ) -> Dict[str, Any]:
        """Exécution d'un backtest unique."""
        try:
            start_time = time.time()
            result = strategy_func(data, parameters)
            duration = time.time() - start_time

            result["period_name"] = period_name
            result["execution_time"] = duration

            return result

        except Exception as e:
            logger.error(f"Erreur backtest {period_name}: {str(e)}")
            return {"period_name": period_name, "error": str(e)}

    def _aggregate_backtest_results(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Agrégation des résultats de backtest."""
        successful_results = [
            r for r in results if not isinstance(r, Exception) and "error" not in r
        ]

        if not successful_results:
            return {"error": "Tous les backtests ont échoué"}

        # Calculer les statistiques agrégées
        aggregated = {
            "total_periods": len(results),
            "successful_periods": len(successful_results),
            "failed_periods": len(results) - len(successful_results),
            "avg_total_return": np.mean(
                [r.get("total_return", 0) for r in successful_results]
            ),
            "avg_sharpe_ratio": np.mean(
                [r.get("sharpe_ratio", 0) for r in successful_results]
            ),
            "avg_max_drawdown": np.mean(
                [r.get("max_drawdown", 0) for r in successful_results]
            ),
            "std_total_return": np.std(
                [r.get("total_return", 0) for r in successful_results]
            ),
            "individual_results": successful_results,
        }

        return aggregated

    # Méthodes de calcul des métriques de risque
    def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """Value at Risk."""
        return returns.quantile(1 - confidence)

    def _calculate_cvar(self, returns: pd.Series, confidence: float) -> float:
        """Conditional Value at Risk."""
        var = self._calculate_var(returns, confidence)
        return returns[returns <= var].mean()

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Maximum Drawdown."""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()

    def _calculate_sharpe(
        self, returns: pd.Series, risk_free_rate: float = 0.0
    ) -> float:
        """Sharpe Ratio."""
        excess_returns = returns - risk_free_rate
        if excess_returns.std() == 0:
            return 0.0
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    def _calculate_sortino(
        self, returns: pd.Series, risk_free_rate: float = 0.0
    ) -> float:
        """Sortino Ratio."""
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float("inf") if excess_returns.mean() > 0 else 0.0
        return excess_returns.mean() / downside_returns.std() * np.sqrt(252)

    def _calculate_calmar(self, returns: pd.Series) -> float:
        """Calmar Ratio."""
        annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        max_dd = abs(self._calculate_max_drawdown(returns))
        return annual_return / max_dd if max_dd != 0 else 0.0

    def _calculate_skewness(self, returns: pd.Series) -> float:
        """Skewness."""
        return returns.skew()

    def _calculate_kurtosis(self, returns: pd.Series) -> float:
        """Kurtosis."""
        return returns.kurtosis()

    def cleanup(self):
        """Nettoyage des ressources."""
        self.executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# Décorateurs pour parallélisation automatique
def parallel_task(task_name: str, max_workers: int = 4):
    """Décorateur pour paralléliser automatiquement une fonction."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            with ParallelProcessor(max_workers=max_workers) as processor:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    processor.executor, func, *args, **kwargs
                )

        return wrapper

    return decorator


# Factory functions pour usage simplifié
def create_parallel_processor(
    strategy_type: str = "trading", max_workers: Optional[int] = None
) -> ParallelProcessor:
    """Factory pour créer un processeur adapté au type de stratégie."""

    if max_workers is None:
        import multiprocessing

        max_workers = min(multiprocessing.cpu_count(), 8)

    # Utiliser les threads par défaut pour éviter les problèmes de sérialisation
    # avec pandas DataFrames qui contiennent des verrous
    if strategy_type == "high_frequency":
        return ParallelProcessor(
            max_workers=max_workers, use_processes=False, chunk_size=500
        )
    elif strategy_type == "research":
        return ParallelProcessor(
            max_workers=max_workers, use_processes=False, chunk_size=2000
        )
    else:  # trading
        return ParallelProcessor(
            max_workers=max_workers, use_processes=False, chunk_size=1000
        )


def create_safe_parallel_processor(max_workers: int = 4) -> ParallelProcessor:
    """
    Créer un processeur parallèle sûr utilisant uniquement les threads.

    Évite tous les problèmes de sérialisation en utilisant ThreadPoolExecutor
    au lieu de ProcessPoolExecutor.
    """
    return ParallelProcessor(
        max_workers=max_workers, use_processes=False, chunk_size=1000
    )


# Exemples d'usage
async def example_parallel_features():
    """Exemple d'usage du traitement parallèle de features."""

    # Données d'exemple
    data = pd.DataFrame(
        {
            "close": np.random.randn(1000).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, 1000),
        }
    )

    # Fonctions de features
    def rsi_feature(df):
        """Feature RSI."""
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        return df[["rsi"]]

    def ma_feature(df):
        """Feature Moving Average."""
        df["ma_20"] = df["close"].rolling(20).mean()
        df["ma_50"] = df["close"].rolling(50).mean()
        return df[["ma_20", "ma_50"]]

    def vol_feature(df):
        """Feature Volatility."""
        df["volatility"] = df["close"].pct_change().rolling(20).std()
        return df[["volatility"]]

    # Traitement parallèle
    with create_parallel_processor("trading") as processor:
        result = await processor.process_parallel_features(
            data, [rsi_feature, ma_feature, vol_feature]
        )

        print(f"Features calculées: {list(result.columns)}")
        return result


if __name__ == "__main__":
    # Test du framework
    import asyncio

    asyncio.run(example_parallel_features())
