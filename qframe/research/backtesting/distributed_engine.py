"""
🚀 Distributed Backtesting Engine

Extends QFrame BacktestingService with distributed computing capabilities
using Dask and Ray for parallel strategy evaluation.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import uuid

# Optional distributed computing imports
try:
    import dask
    import dask.distributed
    from dask.distributed import Client as DaskClient
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

import numpy as np
import pandas as pd

from qframe.core.container import get_container
from qframe.core.interfaces import Strategy
from qframe.domain.services.backtesting_service import BacktestingService
from qframe.domain.entities.backtest import BacktestResult
from qframe.domain.entities.portfolio import Portfolio


class DistributedBacktestEngine:
    """
    🔧 Distributed backtesting engine that extends QFrame BacktestingService

    Uses the existing BacktestingService but distributes work across:
    - Multiple strategies
    - Multiple time periods
    - Multiple parameter sets
    - Multiple assets
    """

    def __init__(
        self,
        compute_backend: str = "dask",  # "dask" or "ray"
        dask_scheduler: Optional[str] = None,
        ray_address: Optional[str] = None,
        max_workers: int = 4
    ):
        self.compute_backend = compute_backend
        self.max_workers = max_workers

        # Get QFrame container and services
        self.container = get_container()
        self.backtesting_service = self.container.resolve(BacktestingService)

        # Initialize distributed computing
        if compute_backend == "dask":
            if not DASK_AVAILABLE:
                print("⚠️ Dask not available, falling back to sequential processing")
                self.compute_backend = "sequential"
            else:
                self._init_dask(dask_scheduler)
        elif compute_backend == "ray":
            if not RAY_AVAILABLE:
                print("⚠️ Ray not available, falling back to sequential processing")
                self.compute_backend = "sequential"
            else:
                self._init_ray(ray_address)
        elif compute_backend == "sequential":
            # Sequential processing fallback
            pass
        else:
            raise ValueError(f"Unknown compute backend: {compute_backend}")

    def _init_dask(self, scheduler_address: Optional[str]):
        """Initialize Dask distributed client"""
        try:
            if scheduler_address:
                self.dask_client = DaskClient(scheduler_address)
            else:
                # Local cluster
                self.dask_client = DaskClient(processes=True, n_workers=self.max_workers)

            print(f"✅ Dask client initialized: {self.dask_client.dashboard_link}")
        except Exception as e:
            print(f"⚠️ Dask initialization failed: {e}")
            self.dask_client = None

    def _init_ray(self, ray_address: Optional[str]):
        """Initialize Ray cluster"""
        try:
            if ray_address:
                ray.init(address=ray_address)
            else:
                ray.init(num_cpus=self.max_workers)

            print(f"✅ Ray initialized: {ray.cluster_resources()}")
        except Exception as e:
            print(f"⚠️ Ray initialization failed: {e}")

    async def run_distributed_backtest(
        self,
        strategies: List[str],
        datasets: List[pd.DataFrame],
        parameter_grids: Optional[Dict[str, List[Any]]] = None,
        split_strategy: str = "time_series",
        n_splits: int = 5,
        initial_capital: float = 100000.0
    ) -> Dict[str, List[BacktestResult]]:
        """
        Run distributed backtests across multiple strategies and datasets

        Args:
            strategies: List of strategy names to test
            datasets: List of market data DataFrames
            parameter_grids: Parameter combinations to test
            split_strategy: How to split data ("time_series", "walk_forward")
            n_splits: Number of splits for validation
            initial_capital: Starting capital for each test
        """
        print(f"🚀 Starting distributed backtest with {self.compute_backend}")
        print(f"📊 Strategies: {len(strategies)}, Datasets: {len(datasets)}")

        if self.compute_backend == "dask":
            return await self._run_dask_backtest(
                strategies, datasets, parameter_grids, split_strategy, n_splits, initial_capital
            )
        elif self.compute_backend == "ray":
            return await self._run_ray_backtest(
                strategies, datasets, parameter_grids, split_strategy, n_splits, initial_capital
            )
        elif self.compute_backend == "sequential":
            return await self._run_sequential_backtest(
                strategies, datasets, parameter_grids, split_strategy, n_splits, initial_capital
            )
        else:
            raise ValueError(f"Unknown compute backend: {self.compute_backend}")

    async def _run_dask_backtest(
        self,
        strategies: List[str],
        datasets: List[pd.DataFrame],
        parameter_grids: Optional[Dict[str, List[Any]]],
        split_strategy: str,
        n_splits: int,
        initial_capital: float
    ) -> Dict[str, List[BacktestResult]]:
        """Run backtests using Dask"""
        if not self.dask_client:
            raise RuntimeError("Dask client not initialized")

        # Create parameter combinations
        tasks = self._create_backtest_tasks(
            strategies, datasets, parameter_grids, split_strategy, n_splits, initial_capital
        )

        print(f"📋 Created {len(tasks)} backtest tasks")

        # Submit tasks to Dask
        futures = []
        for task in tasks:
            future = self.dask_client.submit(self._run_single_backtest, task)
            futures.append(future)

        # Collect results
        print("⏳ Collecting results from Dask workers...")
        results = await asyncio.gather(*[
            asyncio.create_task(self._wait_for_dask_future(future))
            for future in futures
        ])

        # Group results by strategy
        grouped_results = self._group_results_by_strategy(results)

        print(f"✅ Distributed backtest complete: {len(results)} results")
        return grouped_results

    async def _run_ray_backtest(
        self,
        strategies: List[str],
        datasets: List[pd.DataFrame],
        parameter_grids: Optional[Dict[str, List[Any]]],
        split_strategy: str,
        n_splits: int,
        initial_capital: float
    ) -> Dict[str, List[BacktestResult]]:
        """Run backtests using Ray"""
        if not ray.is_initialized():
            raise RuntimeError("Ray not initialized")

        # Create parameter combinations
        tasks = self._create_backtest_tasks(
            strategies, datasets, parameter_grids, split_strategy, n_splits, initial_capital
        )

        print(f"📋 Created {len(tasks)} backtest tasks")

        # Submit tasks to Ray
        futures = [
            self._run_single_backtest_ray.remote(task)
            for task in tasks
        ]

        # Collect results
        print("⏳ Collecting results from Ray workers...")
        results = ray.get(futures)

        # Group results by strategy
        grouped_results = self._group_results_by_strategy(results)

        print(f"✅ Distributed backtest complete: {len(results)} results")
        return grouped_results

    async def _run_sequential_backtest(
        self,
        strategies: List[str],
        datasets: List[pd.DataFrame],
        parameter_grids: Optional[Dict[str, List[Any]]],
        split_strategy: str,
        n_splits: int,
        initial_capital: float
    ) -> Dict[str, List[BacktestResult]]:
        """Run backtests sequentially (fallback when distributed computing unavailable)"""
        # Create parameter combinations
        tasks = self._create_backtest_tasks(
            strategies, datasets, parameter_grids, split_strategy, n_splits, initial_capital
        )

        print(f"📋 Running {len(tasks)} backtest tasks sequentially...")

        # Run tasks sequentially
        results = []
        for i, task in enumerate(tasks):
            print(f"⏳ Processing task {i+1}/{len(tasks)}: {task['id']}")
            result = self._run_single_backtest(task)
            results.append(result)

        # Group results by strategy
        grouped_results = self._group_results_by_strategy(results)

        print(f"✅ Sequential backtest complete: {len(results)} results")
        return grouped_results

    def _create_backtest_tasks(
        self,
        strategies: List[str],
        datasets: List[pd.DataFrame],
        parameter_grids: Optional[Dict[str, List[Any]]],
        split_strategy: str,
        n_splits: int,
        initial_capital: float
    ) -> List[Dict[str, Any]]:
        """Create individual backtest tasks"""
        tasks = []

        for strategy_name in strategies:
            for dataset_idx, dataset in enumerate(datasets):
                # Split dataset based on strategy
                if split_strategy == "time_series":
                    data_splits = self._create_time_series_splits(dataset, n_splits)
                elif split_strategy == "walk_forward":
                    data_splits = self._create_walk_forward_splits(dataset, n_splits)
                else:
                    data_splits = [dataset]  # No splitting

                for split_idx, data_split in enumerate(data_splits):
                    # Parameter grid combinations
                    if parameter_grids and strategy_name in parameter_grids:
                        param_combinations = self._create_parameter_combinations(
                            parameter_grids[strategy_name]
                        )

                        for param_idx, params in enumerate(param_combinations):
                            task = {
                                "id": f"{strategy_name}_{dataset_idx}_{split_idx}_{param_idx}",
                                "strategy_name": strategy_name,
                                "data": data_split,
                                "parameters": params,
                                "initial_capital": initial_capital,
                                "metadata": {
                                    "dataset_idx": dataset_idx,
                                    "split_idx": split_idx,
                                    "param_idx": param_idx
                                }
                            }
                            tasks.append(task)
                    else:
                        # No parameter grid
                        task = {
                            "id": f"{strategy_name}_{dataset_idx}_{split_idx}",
                            "strategy_name": strategy_name,
                            "data": data_split,
                            "parameters": {},
                            "initial_capital": initial_capital,
                            "metadata": {
                                "dataset_idx": dataset_idx,
                                "split_idx": split_idx
                            }
                        }
                        tasks.append(task)

        return tasks

    def _run_single_backtest(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single backtest task

        This function wraps the existing QFrame BacktestingService
        """
        try:
            # Get strategy from container
            strategy = self.container.resolve(Strategy, name=task["strategy_name"])

            # Apply parameters if provided
            if task["parameters"]:
                # Update strategy configuration
                for param, value in task["parameters"].items():
                    if hasattr(strategy, param):
                        setattr(strategy, param, value)

            # Run backtest using QFrame service
            result = self.backtesting_service.run_backtest(
                strategy=strategy,
                data=task["data"],
                initial_capital=task["initial_capital"]
            )

            return {
                "task_id": task["id"],
                "strategy_name": task["strategy_name"],
                "parameters": task["parameters"],
                "metadata": task["metadata"],
                "result": result,
                "status": "success",
                "error": None
            }

        except Exception as e:
            return {
                "task_id": task["id"],
                "strategy_name": task["strategy_name"],
                "parameters": task["parameters"],
                "metadata": task["metadata"],
                "result": None,
                "status": "error",
                "error": str(e)
            }

    def _run_single_backtest_ray(task: Dict[str, Any]) -> Dict[str, Any]:
        """Ray remote version of single backtest"""
        if not RAY_AVAILABLE:
            raise RuntimeError("Ray not available")

        # Note: This needs to be a static method for Ray
        engine = DistributedBacktestEngine(compute_backend="ray")
        return engine._run_single_backtest(task)

    # Apply ray.remote decorator if Ray is available
    if RAY_AVAILABLE:
        _run_single_backtest_ray = ray.remote(_run_single_backtest_ray)

    async def _wait_for_dask_future(self, future):
        """Wait for Dask future completion"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, future.result)
        return result

    def _create_time_series_splits(self, data: pd.DataFrame, n_splits: int) -> List[pd.DataFrame]:
        """Create time series cross-validation splits"""
        splits = []
        total_length = len(data)
        split_size = total_length // n_splits

        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = min((i + 1) * split_size, total_length)

            if i == n_splits - 1:  # Last split takes remainder
                end_idx = total_length

            split_data = data.iloc[start_idx:end_idx].copy()
            splits.append(split_data)

        return splits

    def _create_walk_forward_splits(self, data: pd.DataFrame, n_splits: int) -> List[pd.DataFrame]:
        """Create walk-forward analysis splits"""
        splits = []
        total_length = len(data)
        min_train_size = total_length // (n_splits + 1)

        for i in range(n_splits):
            train_end = min_train_size + (i * min_train_size)
            test_start = train_end
            test_end = min(test_start + min_train_size, total_length)

            # Training data from start to train_end
            train_data = data.iloc[:train_end].copy()
            train_data['split_type'] = 'train'

            # Test data from test_start to test_end
            test_data = data.iloc[test_start:test_end].copy()
            test_data['split_type'] = 'test'

            # Combine for walk-forward
            combined_data = pd.concat([train_data, test_data])
            splits.append(combined_data)

        return splits

    def _create_parameter_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Create all combinations of parameters"""
        if not param_grid:
            return [{}]

        import itertools

        keys = list(param_grid.keys())
        values = list(param_grid.values())

        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)

        return combinations

    def _group_results_by_strategy(self, results: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """Group results by strategy name"""
        grouped = {}

        for result in results:
            strategy_name = result["strategy_name"]
            if strategy_name not in grouped:
                grouped[strategy_name] = []
            grouped[strategy_name].append(result)

        return grouped

    def get_performance_summary(self, results: Dict[str, List[Any]]) -> pd.DataFrame:
        """
        Create performance summary from distributed backtest results
        """
        summary_data = []

        for strategy_name, strategy_results in results.items():
            for result in strategy_results:
                if result["status"] == "success" and result["result"]:
                    backtest_result = result["result"]

                    summary_data.append({
                        "strategy": strategy_name,
                        "task_id": result["task_id"],
                        "total_return": backtest_result.total_return,
                        "sharpe_ratio": backtest_result.sharpe_ratio,
                        "max_drawdown": backtest_result.max_drawdown,
                        "volatility": backtest_result.volatility,
                        "win_rate": getattr(backtest_result, 'win_rate', None),
                        "num_trades": len(backtest_result.trades) if hasattr(backtest_result, 'trades') else 0,
                        "parameters": str(result["parameters"]),
                        "status": result["status"]
                    })
                else:
                    summary_data.append({
                        "strategy": strategy_name,
                        "task_id": result["task_id"],
                        "total_return": None,
                        "sharpe_ratio": None,
                        "max_drawdown": None,
                        "volatility": None,
                        "win_rate": None,
                        "num_trades": 0,
                        "parameters": str(result["parameters"]),
                        "status": result["status"]
                    })

        return pd.DataFrame(summary_data)

    def shutdown(self):
        """Cleanup distributed computing resources"""
        if self.compute_backend == "dask" and self.dask_client:
            self.dask_client.close()
        elif self.compute_backend == "ray" and ray.is_initialized():
            ray.shutdown()

        print(f"✅ {self.compute_backend} resources cleaned up")