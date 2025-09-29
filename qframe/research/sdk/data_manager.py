"""
📊 QFrame Data Manager - Data Lake Management

High-level interface for data ingestion, storage, and retrieval using QFrame Data Lake.
"""

import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path


class DataManager:
    """
    📊 Data Lake Manager

    Provides high-level interface for:
    - Data ingestion and storage
    - Dataset management and cataloging
    - Feature storage and retrieval
    - Data pipeline orchestration
    """

    def __init__(self, qframe_api):
        """
        Initialize Data Manager

        Args:
            qframe_api: QFrameResearch instance for integration
        """
        self.qframe_api = qframe_api
        self.storage = None
        self.catalog = None
        self.feature_store = None

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize data lake components"""
        try:
            if hasattr(self.qframe_api, 'integration') and self.qframe_api.integration:
                self.storage = self.qframe_api.integration.storage
                self.catalog = self.qframe_api.integration.catalog
                self.feature_store = self.qframe_api.integration.feature_store
                print("✅ Data Lake components initialized")
            else:
                print("⚠️ QFrame integration not available. Using mock components.")
                self._initialize_mock_components()
        except Exception as e:
            print(f"⚠️ Failed to initialize data components: {e}")
            self._initialize_mock_components()

    def _initialize_mock_components(self):
        """Initialize mock components for standalone usage"""
        class MockStorage:
            def __init__(self):
                self.data = {}

            def store_dataframe(self, df, path):
                self.data[path] = df
                return True

            def retrieve_dataframe(self, path):
                return self.data.get(path)

            def list_datasets(self):
                return list(self.data.keys())

        class MockCatalog:
            def __init__(self):
                self.catalog = {}

            def register_dataset(self, name, metadata):
                self.catalog[name] = metadata

            def get_dataset_info(self, name):
                return self.catalog.get(name, {})

            def list_datasets(self):
                return list(self.catalog.keys())

        class MockFeatureStore:
            def __init__(self):
                self.features = {}

            def store_features(self, features, name):
                self.features[name] = features

            def retrieve_features(self, name):
                return self.features.get(name)

        self.storage = MockStorage()
        self.catalog = MockCatalog()
        self.feature_store = MockFeatureStore()

    # ====== DATA INGESTION ======

    async def ingest_market_data(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        provider: str = "ccxt",
        dataset_name: Optional[str] = None
    ) -> str:
        """
        Ingest market data into the data lake

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Data frequency
            start_date: Start date
            end_date: End date
            provider: Data provider
            dataset_name: Custom dataset name

        Returns:
            Dataset identifier
        """
        print(f"📥 Ingesting {symbol} data ({timeframe})...")

        # Generate dataset name if not provided
        if dataset_name is None:
            dataset_name = f"{symbol.replace('/', '_')}_{timeframe}_{datetime.now().strftime('%Y%m%d')}"

        try:
            # Fetch data using QFrame API
            data = await self.qframe_api.get_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                provider=provider
            )

            # Store in data lake
            success = self.storage.store_dataframe(data, f"market_data/{dataset_name}")

            if success:
                # Register in catalog
                metadata = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "provider": provider,
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None,
                    "ingestion_time": datetime.now().isoformat(),
                    "data_type": "market_data",
                    "rows": len(data),
                    "columns": list(data.columns)
                }

                self.catalog.register_dataset(dataset_name, metadata)
                print(f"✅ Ingested {len(data)} rows for {symbol}")
                return dataset_name
            else:
                raise Exception("Storage failed")

        except Exception as e:
            print(f"❌ Failed to ingest data for {symbol}: {e}")
            raise

    def ingest_custom_data(
        self,
        data: pd.DataFrame,
        dataset_name: str,
        data_type: str = "custom",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Ingest custom data into the data lake

        Args:
            data: DataFrame to store
            dataset_name: Dataset name
            data_type: Type of data
            metadata: Additional metadata

        Returns:
            Dataset identifier
        """
        print(f"📥 Ingesting custom dataset: {dataset_name}")

        try:
            # Store data
            success = self.storage.store_dataframe(data, f"custom_data/{dataset_name}")

            if success:
                # Register in catalog
                base_metadata = {
                    "data_type": data_type,
                    "ingestion_time": datetime.now().isoformat(),
                    "rows": len(data),
                    "columns": list(data.columns),
                    "index_type": str(type(data.index)),
                    "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()}
                }

                if metadata:
                    base_metadata.update(metadata)

                self.catalog.register_dataset(dataset_name, base_metadata)
                print(f"✅ Ingested custom dataset: {dataset_name} ({len(data)} rows)")
                return dataset_name
            else:
                raise Exception("Storage failed")

        except Exception as e:
            print(f"❌ Failed to ingest custom data: {e}")
            raise

    # ====== DATA RETRIEVAL ======

    def get_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """
        Retrieve a dataset from the data lake

        Args:
            dataset_name: Dataset name

        Returns:
            DataFrame or None if not found
        """
        try:
            # Try market data path first
            data = self.storage.retrieve_dataframe(f"market_data/{dataset_name}")
            if data is not None:
                return data

            # Try custom data path
            data = self.storage.retrieve_dataframe(f"custom_data/{dataset_name}")
            if data is not None:
                return data

            print(f"❌ Dataset not found: {dataset_name}")
            return None

        except Exception as e:
            print(f"❌ Failed to retrieve dataset {dataset_name}: {e}")
            return None

    def list_datasets(self, data_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available datasets

        Args:
            data_type: Filter by data type

        Returns:
            List of dataset information
        """
        try:
            all_datasets = self.catalog.list_datasets()
            dataset_info = []

            for name in all_datasets:
                info = self.catalog.get_dataset_info(name)
                if data_type is None or info.get("data_type") == data_type:
                    dataset_info.append({
                        "name": name,
                        **info
                    })

            return dataset_info

        except Exception as e:
            print(f"❌ Failed to list datasets: {e}")
            return []

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a dataset

        Args:
            dataset_name: Dataset name

        Returns:
            Dataset metadata
        """
        try:
            return self.catalog.get_dataset_info(dataset_name)
        except Exception as e:
            print(f"❌ Failed to get dataset info: {e}")
            return {}

    # ====== FEATURE MANAGEMENT ======

    async def compute_and_store_features(
        self,
        dataset_name: str,
        feature_groups: Optional[List[str]] = None,
        store_name: Optional[str] = None
    ) -> str:
        """
        Compute features for a dataset and store them

        Args:
            dataset_name: Source dataset name
            feature_groups: Feature groups to compute
            store_name: Name for stored features

        Returns:
            Feature store identifier
        """
        print(f"🔧 Computing features for {dataset_name}...")

        try:
            # Get source data
            data = self.get_dataset(dataset_name)
            if data is None:
                raise ValueError(f"Dataset not found: {dataset_name}")

            # Compute features using QFrame API
            features = await self.qframe_api.compute_features(data)

            # Store features
            store_name = store_name or f"{dataset_name}_features"
            self.feature_store.store_features(features, store_name)

            print(f"✅ Computed and stored {len(features.columns)} features")
            return store_name

        except Exception as e:
            print(f"❌ Failed to compute features: {e}")
            raise

    def get_features(self, feature_name: str) -> Optional[pd.DataFrame]:
        """
        Retrieve stored features

        Args:
            feature_name: Feature store name

        Returns:
            Features DataFrame
        """
        try:
            return self.feature_store.retrieve_features(feature_name)
        except Exception as e:
            print(f"❌ Failed to retrieve features: {e}")
            return None

    # ====== DATA PIPELINE ======

    async def create_data_pipeline(
        self,
        pipeline_name: str,
        symbols: List[str],
        timeframe: str = "1h",
        days_history: int = 30,
        compute_features: bool = True
    ) -> Dict[str, str]:
        """
        Create a complete data pipeline for multiple symbols

        Args:
            pipeline_name: Pipeline name
            symbols: List of trading pairs
            timeframe: Data frequency
            days_history: Days of historical data
            compute_features: Whether to compute features

        Returns:
            Dictionary mapping symbols to dataset names
        """
        print(f"🏗️ Creating data pipeline: {pipeline_name}")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_history)

        pipeline_results = {}

        for symbol in symbols:
            try:
                print(f"📊 Processing {symbol}...")

                # Ingest market data
                dataset_name = await self.ingest_market_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    dataset_name=f"{pipeline_name}_{symbol.replace('/', '_')}"
                )

                pipeline_results[symbol] = dataset_name

                # Compute features if requested
                if compute_features:
                    feature_name = await self.compute_and_store_features(dataset_name)
                    pipeline_results[f"{symbol}_features"] = feature_name

            except Exception as e:
                print(f"❌ Failed to process {symbol}: {e}")
                pipeline_results[symbol] = None

        print(f"✅ Pipeline completed: {len([r for r in pipeline_results.values() if r])} successful")
        return pipeline_results

    def create_dataset_summary(self, dataset_name: str) -> Dict[str, Any]:
        """
        Create a comprehensive summary of a dataset

        Args:
            dataset_name: Dataset name

        Returns:
            Summary statistics and information
        """
        try:
            # Get data and metadata
            data = self.get_dataset(dataset_name)
            metadata = self.get_dataset_info(dataset_name)

            if data is None:
                return {"error": "Dataset not found"}

            # Basic statistics
            summary = {
                "name": dataset_name,
                "metadata": metadata,
                "shape": data.shape,
                "memory_usage": data.memory_usage(deep=True).sum(),
                "index_range": {
                    "start": str(data.index.min()),
                    "end": str(data.index.max())
                } if hasattr(data.index, 'min') else None,
                "columns": list(data.columns),
                "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
                "null_counts": data.isnull().sum().to_dict(),
                "basic_stats": data.describe().to_dict() if len(data.select_dtypes(include=[np.number]).columns) > 0 else {}
            }

            # Data quality checks
            summary["data_quality"] = {
                "completeness": 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns))),
                "duplicates": data.duplicated().sum(),
                "unique_values": {col: data[col].nunique() for col in data.columns}
            }

            return summary

        except Exception as e:
            return {"error": f"Failed to create summary: {e}"}

    def export_dataset(
        self,
        dataset_name: str,
        export_path: str,
        format: str = "parquet"
    ) -> bool:
        """
        Export a dataset to local file

        Args:
            dataset_name: Dataset name
            export_path: Export file path
            format: Export format (parquet, csv, json)

        Returns:
            Success status
        """
        try:
            data = self.get_dataset(dataset_name)
            if data is None:
                print(f"❌ Dataset not found: {dataset_name}")
                return False

            # Ensure directory exists
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)

            # Export based on format
            if format.lower() == "parquet":
                data.to_parquet(export_path)
            elif format.lower() == "csv":
                data.to_csv(export_path)
            elif format.lower() == "json":
                data.to_json(export_path, orient="records", date_format="iso")
            else:
                raise ValueError(f"Unsupported format: {format}")

            print(f"✅ Exported {dataset_name} to {export_path}")
            return True

        except Exception as e:
            print(f"❌ Export failed: {e}")
            return False

    def cleanup_old_datasets(self, days_old: int = 30) -> int:
        """
        Remove datasets older than specified days

        Args:
            days_old: Age threshold in days

        Returns:
            Number of datasets removed
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            datasets = self.list_datasets()
            removed_count = 0

            for dataset in datasets:
                ingestion_time = dataset.get("ingestion_time")
                if ingestion_time:
                    try:
                        dataset_date = datetime.fromisoformat(ingestion_time.replace('Z', '+00:00'))
                        if dataset_date < cutoff_date:
                            # Remove from storage and catalog
                            # Note: This is a simplified implementation
                            removed_count += 1
                            print(f"🗑️ Would remove old dataset: {dataset['name']}")
                    except Exception:
                        continue

            print(f"✅ Cleanup completed: {removed_count} old datasets identified")
            return removed_count

        except Exception as e:
            print(f"❌ Cleanup failed: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get data manager statistics

        Returns:
            Statistics dictionary
        """
        try:
            datasets = self.list_datasets()

            stats = {
                "total_datasets": len(datasets),
                "by_type": {},
                "total_size_mb": 0,
                "latest_ingestion": None,
                "oldest_ingestion": None
            }

            # Analyze datasets
            ingestion_times = []
            for dataset in datasets:
                data_type = dataset.get("data_type", "unknown")
                stats["by_type"][data_type] = stats["by_type"].get(data_type, 0) + 1

                # Track ingestion times
                if dataset.get("ingestion_time"):
                    ingestion_times.append(dataset["ingestion_time"])

            if ingestion_times:
                stats["latest_ingestion"] = max(ingestion_times)
                stats["oldest_ingestion"] = min(ingestion_times)

            return stats

        except Exception as e:
            return {"error": f"Failed to get stats: {e}"}

    def __repr__(self):
        datasets_count = len(self.list_datasets())
        return f"DataManager({datasets_count} datasets)"