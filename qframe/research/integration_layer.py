"""
🔗 Integration Layer - Connecte Research Platform avec QFrame Core

Ce module assure l'intégration complète entre la nouvelle infrastructure
de recherche et les composants existants de QFrame.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
from datetime import datetime

from qframe.core.container import get_container
from qframe.core.interfaces import (
    DataProvider,
    Strategy,
    FeatureProcessor,
    RiskManager,
    OrderExecutor,
    MetricsCollector
)
from qframe.infrastructure.data.binance_provider import BinanceProvider
from qframe.infrastructure.data.ccxt_provider import CCXTProvider
from qframe.strategies.research.adaptive_mean_reversion_strategy import AdaptiveMeanReversionStrategy
from qframe.strategies.research.dmn_lstm_strategy import DMNLSTMStrategy
from qframe.strategies.research.funding_arbitrage_strategy import FundingArbitrageStrategy
from qframe.strategies.research.rl_alpha_strategy import RLAlphaStrategy
from qframe.features.symbolic_operators import SymbolicFeatureProcessor
from qframe.domain.services.backtesting_service import BacktestingService
from qframe.domain.services.portfolio_service import PortfolioService

from .data_lake.storage import DataLakeStorage, LocalFileStorage
try:
    from .data_lake.storage import MinIOStorage
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    MinIOStorage = None
from .data_lake.catalog import DataCatalog, DatasetMetadata, DatasetType
from .data_lake.feature_store import FeatureStore, Feature, FeatureGroup


class ResearchIntegration:
    """
    🔧 Classe principale d'intégration entre Research Platform et QFrame Core
    """

    def __init__(
        self,
        storage_backend: Optional[DataLakeStorage] = None,
        catalog_db_url: str = "sqlite:///research_catalog.db"
    ):
        # Initialize container for DI
        self.container = get_container()

        # Initialize research components with fallback
        if storage_backend:
            self.storage = storage_backend
        elif MINIO_AVAILABLE and MinIOStorage:
            try:
                self.storage = MinIOStorage(
                    endpoint="localhost:9000",
                    access_key="minio",
                    secret_key="minio123",
                    bucket_name="qframe-research",
                    secure=False
                )
            except Exception as e:
                print(f"⚠️ MinIO initialization failed: {e}. Using local storage.")
                self.storage = LocalFileStorage("/tmp/qframe_research")
        else:
            print("⚠️ MinIO not available. Using local file storage.")
            self.storage = LocalFileStorage("/tmp/qframe_research")

        self.catalog = DataCatalog(catalog_db_url, self.storage)
        self.feature_store = FeatureStore(self.storage, catalog_db_url)

        # Register QFrame strategies in research platform
        self._register_qframe_strategies()

        # Register QFrame feature processors
        self._register_feature_processors()

        # Setup data providers integration
        self._setup_data_providers()

    def _register_qframe_strategies(self):
        """Enregistre les stratégies QFrame existantes dans la plateforme"""
        strategies = {
            "adaptive_mean_reversion": AdaptiveMeanReversionStrategy,
            "dmn_lstm": DMNLSTMStrategy,
            "funding_arbitrage": FundingArbitrageStrategy,
            "rl_alpha": RLAlphaStrategy,
        }

        for name, strategy_class in strategies.items():
            # Register in container if not already
            if not self.container.is_registered(strategy_class):
                self.container.register_transient(Strategy, strategy_class)

    def _register_feature_processors(self):
        """Intègre les feature processors QFrame dans le Feature Store"""
        # Get symbolic feature processor
        symbolic_processor = SymbolicFeatureProcessor()

        # Create feature group from symbolic operators
        features = []
        for feature_name in symbolic_processor.get_feature_names():
            feature = Feature(
                name=feature_name,
                description=f"Symbolic feature: {feature_name}",
                feature_type="numerical",
                computation_function=feature_name,
                author="qframe",
                status="production"
            )
            features.append(feature)

        # Register feature group
        symbolic_group = FeatureGroup(
            name="symbolic_operators",
            description="QFrame symbolic operators from research paper",
            features=features,
            entity_column="symbol",
            online_enabled=True,
            offline_enabled=True
        )

        self.feature_store.register_feature_group(symbolic_group)

    def _setup_data_providers(self):
        """Configure les data providers QFrame pour le data lake"""
        # Register Binance provider
        if not self.container.is_registered(BinanceProvider):
            self.container.register_singleton(DataProvider, BinanceProvider)

        # Register CCXT provider for multiple exchanges
        if not self.container.is_registered(CCXTProvider):
            self.container.register_singleton(DataProvider, CCXTProvider)

    async def sync_market_data_to_lake(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ):
        """
        Synchronise les données de marché depuis les providers QFrame vers le Data Lake
        """
        # Get data provider from container
        provider = self.container.resolve(DataProvider, name="ccxt")

        for symbol in symbols:
            # Fetch data using QFrame provider
            df = await provider.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start=start_date,
                end=end_date
            )

            # Store in data lake
            path = f"market_data/{symbol}/{timeframe}/{datetime.now().strftime('%Y%m%d')}.parquet"
            metadata = await self.storage.put_dataframe(
                df,
                path,
                format="parquet",
                partition_cols=["year", "month", "day"]
            )

            # Register in catalog
            dataset_metadata = DatasetMetadata(
                name=f"market_data_{symbol}_{timeframe}",
                description=f"Market data for {symbol} at {timeframe}",
                dataset_type=DatasetType.RAW_MARKET_DATA,
                quality_level="validated",
                storage_path=path,
                format="parquet",
                compression="snappy",
                size_bytes=metadata.size_bytes,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                data_start_date=df.index.min(),
                data_end_date=df.index.max(),
                row_count=len(df),
                column_count=len(df.columns),
                columns={col: str(df[col].dtype) for col in df.columns},
                owner="data_sync",
                tags=["market_data", symbol, timeframe]
            )

            self.catalog.register_dataset(dataset_metadata)

    async def backtest_with_research_data(
        self,
        strategy_name: str,
        dataset_name: str,
        feature_group: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute un backtest en utilisant les données du Data Lake et les stratégies QFrame
        """
        # Load data from catalog
        dataset_meta = self.catalog.get_dataset(dataset_name)
        if not dataset_meta:
            raise ValueError(f"Dataset {dataset_name} not found")

        # Load data from storage
        data = await self.storage.get_dataframe(dataset_meta.storage_path)

        # Apply features if requested
        if feature_group:
            features = self.feature_store.compute_features(feature_group, data)
            data = pd.concat([data, features], axis=1)

        # Get strategy from container
        strategy = self.container.resolve(Strategy, name=strategy_name)

        # Get backtesting service
        backtest_service = self.container.resolve(BacktestingService)

        # Run backtest
        results = await backtest_service.run_backtest(
            strategy=strategy,
            data=data,
            initial_capital=10000.0
        )

        # Store results in data lake
        results_path = f"backtest_results/{strategy_name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        results_df = pd.DataFrame([results])
        await self.storage.put_dataframe(results_df, results_path)

        # Register results in catalog
        results_metadata = DatasetMetadata(
            name=f"backtest_{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description=f"Backtest results for {strategy_name}",
            dataset_type=DatasetType.BACKTEST_RESULTS,
            quality_level="production",
            storage_path=results_path,
            format="parquet",
            size_bytes=len(results_df.to_parquet()),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_datasets=[dataset_name],
            owner="backtest_service",
            tags=["backtest", strategy_name]
        )

        self.catalog.register_dataset(results_metadata)

        return results

    async def compute_research_features(
        self,
        data: pd.DataFrame,
        include_symbolic: bool = True,
        include_ml: bool = False
    ) -> pd.DataFrame:
        """
        Compute toutes les features de recherche en utilisant les processeurs QFrame
        """
        result_df = data.copy()

        if include_symbolic:
            # Use QFrame symbolic processor
            symbolic_processor = self.container.resolve(FeatureProcessor, name="symbolic")
            symbolic_features = symbolic_processor.process(data)
            result_df = pd.concat([result_df, symbolic_features], axis=1)

        if include_ml:
            # Compute ML features from feature store
            ml_features = self.feature_store.compute_features("ml_features", data)
            result_df = pd.concat([result_df, ml_features], axis=1)

        return result_df

    def create_research_portfolio(
        self,
        portfolio_id: str,
        initial_capital: float,
        strategies: List[str]
    ):
        """
        Crée un portfolio de recherche avec multiple stratégies QFrame
        """
        # Get portfolio service
        portfolio_service = self.container.resolve(PortfolioService)

        # Create portfolio
        portfolio = portfolio_service.create_portfolio(
            portfolio_id=portfolio_id,
            initial_capital=initial_capital,
            base_currency="USD"
        )

        # Attach strategies
        for strategy_name in strategies:
            strategy = self.container.resolve(Strategy, name=strategy_name)
            portfolio_service.add_strategy(portfolio.id, strategy)

        return portfolio

    async def export_mlflow_experiments(self, experiment_name: str):
        """
        Export MLflow experiments vers le Data Lake pour archivage
        """
        import mlflow

        # Set MLflow tracking URI
        mlflow.set_tracking_uri("http://localhost:5000")

        # Get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            return

        # Get all runs
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        # Store in data lake
        path = f"mlflow_experiments/{experiment_name}/{datetime.now().strftime('%Y%m%d')}.parquet"
        await self.storage.put_dataframe(runs, path)

        # Register in catalog
        metadata = DatasetMetadata(
            name=f"mlflow_{experiment_name}",
            description=f"MLflow experiment results for {experiment_name}",
            dataset_type=DatasetType.MODEL_ARTIFACTS,
            quality_level="production",
            storage_path=path,
            format="parquet",
            size_bytes=len(runs.to_parquet()),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            row_count=len(runs),
            owner="mlflow",
            tags=["mlflow", "experiments", experiment_name]
        )

        self.catalog.register_dataset(metadata)

    def get_integration_status(self) -> Dict[str, Any]:
        """
        Retourne le status complet de l'intégration
        """
        return {
            "container_services": {
                "strategies": len([s for s in self.container._services if "Strategy" in str(s)]),
                "data_providers": len([s for s in self.container._services if "DataProvider" in str(s)]),
                "feature_processors": len([s for s in self.container._services if "FeatureProcessor" in str(s)]),
            },
            "catalog_stats": self.catalog.get_statistics(),
            "feature_store": {
                "feature_groups": len(self.feature_store.feature_groups),
                "cached_features": len(self.feature_store.feature_cache),
            },
            "storage": {
                "backend": type(self.storage).__name__,
                "connected": True
            }
        }


# 🚀 Factory function pour créer l'intégration
def create_research_integration(
    use_minio: bool = False,  # Default to False for stability
    minio_config: Optional[Dict[str, str]] = None
) -> ResearchIntegration:
    """
    Factory pour créer une instance de ResearchIntegration configurée

    Args:
        use_minio: Whether to use MinIO storage (requires minio package)
        minio_config: MinIO configuration (optional)

    Returns:
        ResearchIntegration instance with appropriate storage backend
    """
    storage = None

    if use_minio and MINIO_AVAILABLE and MinIOStorage:
        try:
            config = minio_config or {
                "endpoint": "localhost:9000",
                "access_key": "minio",
                "secret_key": "minio123",
                "bucket_name": "qframe-research",
                "secure": False
            }
            storage = MinIOStorage(**config)
            print("✅ Using MinIO storage backend")
        except Exception as e:
            print(f"⚠️ MinIO setup failed: {e}. Falling back to local storage.")
            storage = LocalFileStorage("/tmp/qframe_research")
    else:
        if use_minio:
            print("⚠️ MinIO requested but not available. Using local storage.")
        storage = LocalFileStorage("/tmp/qframe_research")

    return ResearchIntegration(storage_backend=storage)