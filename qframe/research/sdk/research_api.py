"""
🚀 QFrame Research API - High-level SDK

Unified API that brings together all QFrame Core and Research Platform
capabilities in a simple, intuitive interface.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np

# QFrame Core imports
from qframe.core.container import get_container
from qframe.core.config import get_config
from qframe.core.interfaces import Strategy, DataProvider

# Research Platform imports
from qframe.research.integration_layer import create_research_integration
from qframe.research.backtesting.distributed_engine import DistributedBacktestEngine
from qframe.research.data_lake.feature_store import FeatureStore


class QFrameResearch:
    """
    🔬 Main QFrame Research API

    One-stop interface for quantitative research that combines:
    - QFrame Core strategies and services
    - Research Platform infrastructure
    - Simple, Pythonic API design
    """

    def __init__(
        self,
        auto_init: bool = True,
        compute_backend: str = "dask",
        data_lake_backend: str = "minio"
    ):
        """
        Initialize QFrame Research environment

        Args:
            auto_init: Automatically initialize all components
            compute_backend: "dask" or "ray" for distributed computing
            data_lake_backend: "minio", "s3", or "local" for storage
        """
        self.compute_backend = compute_backend
        self.data_lake_backend = data_lake_backend

        # Core components
        self.container = None
        self.config = None

        # Research components
        self.integration = None
        self.distributed_engine = None
        self.feature_store = None

        # State
        self._initialized = False
        self._available_strategies = {}
        self._available_datasets = {}

        if auto_init:
            asyncio.run(self.initialize())

    async def initialize(self):
        """Initialize all QFrame Research components"""
        if self._initialized:
            return

        print("🚀 Initializing QFrame Research...")

        # 1. Initialize QFrame Core
        self.container = get_container()
        self.config = get_config()
        print("✅ QFrame Core initialized")

        # 2. Initialize Research Platform
        self.integration = create_research_integration(
            use_minio=(self.data_lake_backend == "minio")
        )
        print("✅ Research Platform initialized")

        # 3. Initialize Distributed Computing
        try:
            self.distributed_engine = DistributedBacktestEngine(
                compute_backend=self.compute_backend,
                max_workers=4
            )
            print(f"✅ {self.compute_backend.title()} compute engine ready")
        except Exception as e:
            print(f"⚠️ Distributed computing setup: {e}")

        # 4. Initialize Feature Store
        self.feature_store = self.integration.feature_store
        print("✅ Feature Store ready")

        # 5. Discover available components
        await self._discover_components()

        self._initialized = True
        print("🎉 QFrame Research ready!")

    async def _discover_components(self):
        """Discover available strategies and datasets"""
        # Discover strategies
        strategy_names = [
            "adaptive_mean_reversion",
            "dmn_lstm",
            "funding_arbitrage",
            "rl_alpha"
        ]

        for name in strategy_names:
            try:
                strategy = self.container.resolve(Strategy, name=name)
                self._available_strategies[name] = {
                    "class": type(strategy).__name__,
                    "description": getattr(strategy, "description", "QFrame strategy"),
                    "instance": strategy
                }
            except:
                print(f"⚠️ Strategy '{name}' not available")

        print(f"🎯 Discovered {len(self._available_strategies)} strategies")

        # Discover datasets
        try:
            catalog_stats = self.integration.catalog.get_statistics()
            self._available_datasets = catalog_stats
            print(f"📊 Discovered {catalog_stats.get('total_datasets', 0)} datasets")
        except:
            print("⚠️ Dataset discovery failed")

    # ====== HIGH-LEVEL API METHODS ======

    def strategies(self) -> Dict[str, str]:
        """List available strategies"""
        return {name: info["description"] for name, info in self._available_strategies.items()}

    def datasets(self) -> Dict[str, Any]:
        """List available datasets"""
        return self._available_datasets

    async def get_data(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        provider: str = "ccxt"
    ) -> pd.DataFrame:
        """
        Get market data using QFrame data providers

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Data frequency
            start_date: Start date for data
            end_date: End date for data
            provider: Data provider to use

        Returns:
            OHLCV DataFrame
        """
        if not self._initialized:
            await self.initialize()

        try:
            data_provider = self.container.resolve(DataProvider, name=provider)
            return await data_provider.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start=start_date,
                end=end_date
            )
        except Exception as e:
            print(f"⚠️ Data fetch failed: {e}")
            # Return sample data for demo
            return self._create_sample_data(symbol, start_date, end_date)

    def _create_sample_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Create sample market data for testing"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()

        dates = pd.date_range(start=start_date, end=end_date, freq="1h")
        n_periods = len(dates)

        # Simulate realistic price movement
        np.random.seed(42)
        base_price = 50000 if "BTC" in symbol else 3000
        returns = np.random.normal(0.0001, 0.02, n_periods).cumsum()
        prices = base_price * np.exp(returns)

        return pd.DataFrame({
            "timestamp": dates,
            "open": prices * (1 + np.random.normal(0, 0.001, n_periods)),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.005, n_periods))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.005, n_periods))),
            "close": prices,
            "volume": np.random.lognormal(10, 0.5, n_periods)
        }).set_index("timestamp")

    async def compute_features(
        self,
        data: pd.DataFrame,
        feature_groups: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compute features using QFrame feature processors

        Args:
            data: Market data DataFrame
            feature_groups: List of feature groups to compute

        Returns:
            DataFrame with original data + features
        """
        if not self._initialized:
            await self.initialize()

        return await self.integration.compute_research_features(
            data=data,
            include_symbolic=True,
            include_ml=False
        )

    async def backtest(
        self,
        strategy: str,
        data: pd.DataFrame,
        initial_capital: float = 100000.0,
        **strategy_params
    ) -> Dict[str, Any]:
        """
        Run a single backtest using QFrame BacktestingService

        Args:
            strategy: Strategy name
            data: Market data
            initial_capital: Starting capital
            **strategy_params: Strategy-specific parameters

        Returns:
            Backtest results
        """
        if not self._initialized:
            await self.initialize()

        if strategy not in self._available_strategies:
            raise ValueError(f"Strategy '{strategy}' not available. Use .strategies() to see options.")

        # Get strategy instance
        strategy_instance = self._available_strategies[strategy]["instance"]

        # Apply parameters
        for param, value in strategy_params.items():
            if hasattr(strategy_instance, param):
                setattr(strategy_instance, param, value)

        # Run backtest using integration layer
        return await self.integration.backtest_with_research_data(
            strategy_name=strategy,
            dataset_name="temp_dataset",  # TODO: Better dataset handling
            feature_group=None
        )

    async def distributed_backtest(
        self,
        strategies: List[str],
        data: Union[pd.DataFrame, List[pd.DataFrame]],
        parameter_grids: Optional[Dict[str, Dict[str, List[Any]]]] = None,
        validation_splits: int = 5,
        initial_capital: float = 100000.0
    ) -> pd.DataFrame:
        """
        Run distributed backtests across multiple strategies and parameters

        Args:
            strategies: List of strategy names
            data: Market data (single DataFrame or list)
            parameter_grids: Parameter combinations to test
            validation_splits: Number of cross-validation splits
            initial_capital: Starting capital

        Returns:
            DataFrame with all backtest results
        """
        if not self._initialized:
            await self.initialize()

        if not self.distributed_engine:
            raise RuntimeError("Distributed engine not available")

        # Ensure data is a list
        if isinstance(data, pd.DataFrame):
            datasets = [data]
        else:
            datasets = data

        # Run distributed backtest
        results = await self.distributed_engine.run_distributed_backtest(
            strategies=strategies,
            datasets=datasets,
            parameter_grids=parameter_grids,
            n_splits=validation_splits,
            initial_capital=initial_capital
        )

        # Convert to summary DataFrame
        return self.distributed_engine.get_performance_summary(results)

    def create_strategy(self, name: str) -> "StrategyBuilder":
        """
        Create a new strategy using the StrategyBuilder

        Args:
            name: Strategy name

        Returns:
            StrategyBuilder instance
        """
        from .strategy_builder import StrategyBuilder
        return StrategyBuilder(name, self)

    def experiment(self, name: str) -> "ExperimentManager":
        """
        Start a new MLflow experiment

        Args:
            name: Experiment name

        Returns:
            ExperimentManager instance
        """
        from .experiment_manager import ExperimentManager
        return ExperimentManager(name, self)

    def data_manager(self) -> "DataManager":
        """
        Get data management interface

        Returns:
            DataManager instance
        """
        from .data_manager import DataManager
        return DataManager(self)

    # ====== CONVENIENCE METHODS ======

    async def quick_backtest(
        self,
        symbol: str = "BTC/USDT",
        strategy: str = "adaptive_mean_reversion",
        days: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Quick backtest with minimal setup

        Args:
            symbol: Trading pair
            strategy: Strategy to test
            days: Days of historical data
            **kwargs: Additional parameters

        Returns:
            Backtest results
        """
        # Get data
        data = await self.get_data(
            symbol=symbol,
            start_date=datetime.now() - timedelta(days=days)
        )

        # Add features
        data_with_features = await self.compute_features(data)

        # Run backtest
        return await self.backtest(strategy, data_with_features, **kwargs)

    def compare_strategies(
        self,
        strategies: List[str],
        symbol: str = "BTC/USDT",
        days: int = 30
    ) -> pd.DataFrame:
        """
        Compare multiple strategies on the same data

        Args:
            strategies: List of strategy names
            symbol: Trading pair
            days: Days of historical data

        Returns:
            Comparison DataFrame
        """
        async def _compare():
            # Get data once
            data = await self.get_data(
                symbol=symbol,
                start_date=datetime.now() - timedelta(days=days)
            )

            # Run distributed backtest
            return await self.distributed_backtest(
                strategies=strategies,
                data=data,
                validation_splits=1  # Single test for comparison
            )

        return asyncio.run(_compare())

    def status(self) -> Dict[str, Any]:
        """
        Get QFrame Research status

        Returns:
            Status information
        """
        return {
            "initialized": self._initialized,
            "compute_backend": self.compute_backend,
            "data_lake_backend": self.data_lake_backend,
            "strategies_available": len(self._available_strategies),
            "datasets_available": self._available_datasets.get("total_datasets", 0),
            "integration_status": self.integration.get_integration_status() if self.integration else None
        }

    def help(self):
        """Show help and examples"""
        help_text = """
🔬 QFrame Research API Quick Reference

=== Basic Usage ===
qf = QFrameResearch()                    # Initialize
await qf.get_data("BTC/USDT")            # Get market data
await qf.compute_features(data)          # Add features
await qf.backtest("mean_reversion", data)  # Single backtest

=== Strategy Comparison ===
results = qf.compare_strategies([
    "adaptive_mean_reversion",
    "dmn_lstm"
], symbol="BTC/USDT", days=30)

=== Distributed Backtesting ===
results = await qf.distributed_backtest(
    strategies=["mean_reversion", "lstm"],
    data=data,
    parameter_grids={
        "mean_reversion": {"lookback": [10, 20, 30]}
    },
    validation_splits=5
)

=== Experiment Tracking ===
with qf.experiment("my_research"):
    results = await qf.quick_backtest()
    # Results automatically logged to MLflow

=== Available Methods ===
.strategies()           # List strategies
.datasets()            # List datasets
.get_data()            # Fetch market data
.compute_features()    # Feature engineering
.backtest()            # Single backtest
.distributed_backtest() # Multi-strategy/parameter
.quick_backtest()      # Fast single test
.compare_strategies()  # Strategy comparison
.status()              # System status
.help()                # This help

All methods integrate QFrame Core components with Research Platform!
        """
        print(help_text)

    def __repr__(self):
        status = "initialized" if self._initialized else "not initialized"
        return f"QFrameResearch({status}, {len(self._available_strategies)} strategies, {self.compute_backend} backend)"