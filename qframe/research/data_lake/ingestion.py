"""
📥 Data Ingestion Pipeline for QFrame Data Lake

Handles batch and streaming data ingestion with validation and transformation.
"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, validator

from qframe.core.interfaces import DataProvider
from .catalog import DataCatalog, DataQuality, DatasetMetadata, DatasetType
from .storage import DataLakeStorage, StorageMetadata


class IngestionMode(str, Enum):
    """Data ingestion modes"""

    BATCH = "batch"
    STREAMING = "streaming"
    INCREMENTAL = "incremental"
    FULL_REFRESH = "full_refresh"


class IngestionStatus(str, Enum):
    """Status of ingestion job"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


class DataValidationRule(BaseModel):
    """Validation rule for ingested data"""

    name: str
    check_function: str  # Name of validation function
    parameters: Dict[str, Any] = Field(default_factory=dict)
    severity: str = "warning"  # warning, error
    action: str = "log"  # log, drop, fix


class IngestionConfig(BaseModel):
    """Configuration for data ingestion"""

    source_name: str
    target_path: str
    ingestion_mode: IngestionMode
    schedule: Optional[str] = None  # Cron expression

    # Data processing
    transformations: List[str] = Field(default_factory=list)
    validation_rules: List[DataValidationRule] = Field(default_factory=list)
    deduplication_columns: List[str] = Field(default_factory=list)

    # Storage options
    format: str = "parquet"
    compression: str = "snappy"
    partition_columns: List[str] = Field(default_factory=list)

    # Performance
    batch_size: int = 10000
    parallel_workers: int = 4

    # Monitoring
    enable_metrics: bool = True
    alert_on_failure: bool = True


class IngestionJob(BaseModel):
    """Represents a data ingestion job"""

    job_id: str
    config: IngestionConfig
    status: IngestionStatus = IngestionStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    rows_processed: int = 0
    rows_failed: int = 0
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class DataValidator:
    """Data validation engine"""

    def __init__(self):
        self.validation_functions: Dict[str, Callable] = {}
        self._register_default_validations()

    def _register_default_validations(self):
        """Register built-in validation functions"""
        self.register("not_null", self._check_not_null)
        self.register("unique", self._check_unique)
        self.register("range", self._check_range)
        self.register("format", self._check_format)
        self.register("completeness", self._check_completeness)
        self.register("consistency", self._check_consistency)

    def register(self, name: str, func: Callable):
        """Register a validation function"""
        self.validation_functions[name] = func

    def validate(
        self, df: pd.DataFrame, rules: List[DataValidationRule]
    ) -> tuple[pd.DataFrame, Dict[str, List[str]]]:
        """Validate dataframe against rules"""
        validation_results = {"errors": [], "warnings": []}
        clean_df = df.copy()

        for rule in rules:
            if rule.check_function not in self.validation_functions:
                validation_results["warnings"].append(
                    f"Unknown validation function: {rule.check_function}"
                )
                continue

            func = self.validation_functions[rule.check_function]
            is_valid, message, fixed_df = func(clean_df, **rule.parameters)

            if not is_valid:
                if rule.severity == "error":
                    validation_results["errors"].append(f"{rule.name}: {message}")
                    if rule.action == "drop":
                        # Drop invalid rows
                        clean_df = fixed_df
                    elif rule.action == "fix" and fixed_df is not None:
                        clean_df = fixed_df
                else:
                    validation_results["warnings"].append(f"{rule.name}: {message}")

        return clean_df, validation_results

    @staticmethod
    def _check_not_null(
        df: pd.DataFrame, columns: List[str]
    ) -> tuple[bool, str, Optional[pd.DataFrame]]:
        """Check for null values in specified columns"""
        null_counts = df[columns].isnull().sum()
        if null_counts.any():
            null_cols = null_counts[null_counts > 0].to_dict()
            clean_df = df.dropna(subset=columns)
            return False, f"Null values found: {null_cols}", clean_df
        return True, "No null values", None

    @staticmethod
    def _check_unique(
        df: pd.DataFrame, columns: List[str]
    ) -> tuple[bool, str, Optional[pd.DataFrame]]:
        """Check for duplicate values"""
        duplicates = df.duplicated(subset=columns, keep=False)
        if duplicates.any():
            dup_count = duplicates.sum()
            clean_df = df.drop_duplicates(subset=columns, keep="first")
            return False, f"{dup_count} duplicate rows found", clean_df
        return True, "No duplicates", None

    @staticmethod
    def _check_range(
        df: pd.DataFrame, column: str, min_val: float = None, max_val: float = None
    ) -> tuple[bool, str, Optional[pd.DataFrame]]:
        """Check if values are within range"""
        if min_val is not None:
            below_min = df[column] < min_val
            if below_min.any():
                return False, f"{below_min.sum()} values below minimum {min_val}", None

        if max_val is not None:
            above_max = df[column] > max_val
            if above_max.any():
                return False, f"{above_max.sum()} values above maximum {max_val}", None

        return True, "Values within range", None

    @staticmethod
    def _check_format(
        df: pd.DataFrame, column: str, pattern: str
    ) -> tuple[bool, str, Optional[pd.DataFrame]]:
        """Check if values match a format pattern"""
        import re

        invalid = ~df[column].astype(str).str.match(pattern)
        if invalid.any():
            return False, f"{invalid.sum()} values don't match pattern {pattern}", None
        return True, "All values match pattern", None

    @staticmethod
    def _check_completeness(
        df: pd.DataFrame, threshold: float = 0.95
    ) -> tuple[bool, str, Optional[pd.DataFrame]]:
        """Check data completeness"""
        completeness = 1 - (df.isnull().sum() / len(df))
        incomplete_cols = completeness[completeness < threshold]

        if len(incomplete_cols) > 0:
            return (
                False,
                f"Columns below {threshold*100}% completeness: {incomplete_cols.to_dict()}",
                None,
            )
        return True, f"All columns above {threshold*100}% completeness", None

    @staticmethod
    def _check_consistency(
        df: pd.DataFrame, rules: Dict[str, str]
    ) -> tuple[bool, str, Optional[pd.DataFrame]]:
        """Check data consistency rules"""
        for rule_name, rule_expr in rules.items():
            try:
                # Evaluate consistency rule
                is_consistent = df.eval(rule_expr)
                if not is_consistent.all():
                    inconsistent_count = (~is_consistent).sum()
                    return (
                        False,
                        f"Consistency rule '{rule_name}' failed for {inconsistent_count} rows",
                        None,
                    )
            except Exception as e:
                return False, f"Error evaluating rule '{rule_name}': {e}", None

        return True, "All consistency rules passed", None


class DataTransformer:
    """Data transformation engine"""

    def __init__(self):
        self.transformation_functions: Dict[str, Callable] = {}
        self._register_default_transformations()

    def _register_default_transformations(self):
        """Register built-in transformation functions"""
        self.register("normalize", self._normalize)
        self.register("standardize", self._standardize)
        self.register("resample", self._resample)
        self.register("aggregate", self._aggregate)
        self.register("pivot", self._pivot)
        self.register("melt", self._melt)

    def register(self, name: str, func: Callable):
        """Register a transformation function"""
        self.transformation_functions[name] = func

    def transform(
        self, df: pd.DataFrame, transformations: List[str], **params
    ) -> pd.DataFrame:
        """Apply transformations to dataframe"""
        result_df = df.copy()

        for transformation in transformations:
            if transformation not in self.transformation_functions:
                print(f"Warning: Unknown transformation {transformation}")
                continue

            func = self.transformation_functions[transformation]
            result_df = func(result_df, **params.get(transformation, {}))

        return result_df

    @staticmethod
    def _normalize(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """Normalize numeric columns to [0, 1]"""
        if columns is None:
            columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

        result_df = df.copy()
        for col in columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                result_df[col] = (df[col] - min_val) / (max_val - min_val)

        return result_df

    @staticmethod
    def _standardize(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """Standardize numeric columns to mean=0, std=1"""
        if columns is None:
            columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

        result_df = df.copy()
        for col in columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                result_df[col] = (df[col] - mean) / std

        return result_df

    @staticmethod
    def _resample(
        df: pd.DataFrame, freq: str, date_column: str = "timestamp", agg_func: str = "mean"
    ) -> pd.DataFrame:
        """Resample time series data"""
        df_copy = df.copy()
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
        df_copy = df_copy.set_index(date_column)

        # Resample based on aggregation function
        if agg_func == "mean":
            resampled = df_copy.resample(freq).mean()
        elif agg_func == "sum":
            resampled = df_copy.resample(freq).sum()
        elif agg_func == "last":
            resampled = df_copy.resample(freq).last()
        elif agg_func == "first":
            resampled = df_copy.resample(freq).first()
        else:
            resampled = df_copy.resample(freq).mean()

        return resampled.reset_index()

    @staticmethod
    def _aggregate(
        df: pd.DataFrame, group_by: List[str], agg_config: Dict[str, str]
    ) -> pd.DataFrame:
        """Aggregate data by groups"""
        return df.groupby(group_by).agg(agg_config).reset_index()

    @staticmethod
    def _pivot(
        df: pd.DataFrame, index: str, columns: str, values: str
    ) -> pd.DataFrame:
        """Pivot dataframe"""
        return df.pivot(index=index, columns=columns, values=values).reset_index()

    @staticmethod
    def _melt(
        df: pd.DataFrame, id_vars: List[str], value_vars: List[str] = None
    ) -> pd.DataFrame:
        """Melt dataframe from wide to long format"""
        return df.melt(id_vars=id_vars, value_vars=value_vars)


class DataIngestionPipeline:
    """Main data ingestion pipeline"""

    def __init__(
        self,
        storage: DataLakeStorage,
        catalog: DataCatalog,
        data_provider: Optional[DataProvider] = None,
    ):
        self.storage = storage
        self.catalog = catalog
        self.data_provider = data_provider
        self.validator = DataValidator()
        self.transformer = DataTransformer()
        self.active_jobs: Dict[str, IngestionJob] = {}

    async def ingest(
        self, config: IngestionConfig, data: Optional[pd.DataFrame] = None
    ) -> IngestionJob:
        """Execute data ingestion job"""
        # Create job
        job_id = self._generate_job_id(config)
        job = IngestionJob(job_id=job_id, config=config, started_at=datetime.now())

        self.active_jobs[job_id] = job
        job.status = IngestionStatus.RUNNING

        try:
            # Load data if not provided
            if data is None:
                data = await self._load_source_data(config)

            job.metrics["raw_rows"] = len(data)

            # Deduplication
            if config.deduplication_columns:
                data = data.drop_duplicates(subset=config.deduplication_columns)
                job.metrics["after_dedup_rows"] = len(data)

            # Validation
            data, validation_results = self.validator.validate(data, config.validation_rules)
            job.metrics["validation_errors"] = len(validation_results["errors"])
            job.metrics["validation_warnings"] = len(validation_results["warnings"])

            if validation_results["errors"] and config.alert_on_failure:
                job.error_message = f"Validation errors: {validation_results['errors']}"
                job.status = IngestionStatus.PARTIAL

            # Transformation
            if config.transformations:
                data = self.transformer.transform(data, config.transformations)
                job.metrics["after_transform_rows"] = len(data)

            # Store data
            metadata = await self._store_data(data, config)
            job.metrics["stored_rows"] = len(data)
            job.rows_processed = len(data)

            # Register in catalog
            await self._register_dataset(metadata, config)

            # Update job status
            job.status = IngestionStatus.SUCCESS if not validation_results["errors"] else IngestionStatus.PARTIAL
            job.completed_at = datetime.now()

        except Exception as e:
            job.status = IngestionStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            raise

        finally:
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]

        return job

    async def _load_source_data(self, config: IngestionConfig) -> pd.DataFrame:
        """Load data from source"""
        if self.data_provider:
            # Use data provider to fetch data
            # This would be customized based on source type
            pass

        # For now, return empty dataframe
        return pd.DataFrame()

    async def _store_data(
        self, data: pd.DataFrame, config: IngestionConfig
    ) -> StorageMetadata:
        """Store data in data lake"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{config.target_path}/{timestamp}.{config.format}"

        metadata = await self.storage.put_dataframe(
            data,
            path,
            format=config.format,
            compression=config.compression,
            partition_cols=config.partition_columns,
        )

        return metadata

    async def _register_dataset(self, storage_metadata, config: IngestionConfig):
        """Register dataset in catalog"""
        dataset_metadata = DatasetMetadata(
            name=f"{config.source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description=f"Ingested from {config.source_name}",
            dataset_type=DatasetType.RAW_MARKET_DATA,
            quality_level=DataQuality.VALIDATED,
            storage_path=storage_metadata.path,
            format=config.format,
            compression=config.compression,
            size_bytes=storage_metadata.size_bytes,
            created_at=storage_metadata.created_at,
            updated_at=storage_metadata.modified_at,
            owner="ingestion_pipeline",
        )

        self.catalog.register_dataset(dataset_metadata)

    def _generate_job_id(self, config: IngestionConfig) -> str:
        """Generate unique job ID"""
        content = f"{config.source_name}_{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class StreamIngester:
    """Real-time streaming data ingestion"""

    def __init__(self, pipeline: DataIngestionPipeline):
        self.pipeline = pipeline
        self.buffer: Dict[str, pd.DataFrame] = {}
        self.buffer_size = 1000
        self.flush_interval = 60  # seconds

    async def ingest_stream(
        self, stream_config: IngestionConfig, data_stream: asyncio.Queue
    ):
        """Ingest data from stream"""
        buffer_key = stream_config.source_name
        self.buffer[buffer_key] = pd.DataFrame()

        while True:
            try:
                # Get data from stream
                data = await asyncio.wait_for(data_stream.get(), timeout=self.flush_interval)

                # Add to buffer
                self.buffer[buffer_key] = pd.concat(
                    [self.buffer[buffer_key], data], ignore_index=True
                )

                # Flush if buffer is full
                if len(self.buffer[buffer_key]) >= self.buffer_size:
                    await self._flush_buffer(stream_config, buffer_key)

            except asyncio.TimeoutError:
                # Flush on timeout
                if not self.buffer[buffer_key].empty:
                    await self._flush_buffer(stream_config, buffer_key)

    async def _flush_buffer(self, config: IngestionConfig, buffer_key: str):
        """Flush buffer to storage"""
        if buffer_key in self.buffer and not self.buffer[buffer_key].empty:
            await self.pipeline.ingest(config, self.buffer[buffer_key])
            self.buffer[buffer_key] = pd.DataFrame()


class BatchIngester:
    """Batch data ingestion with scheduling"""

    def __init__(self, pipeline: DataIngestionPipeline):
        self.pipeline = pipeline
        self.scheduled_jobs: Dict[str, IngestionConfig] = {}

    async def schedule_batch(self, config: IngestionConfig):
        """Schedule batch ingestion job"""
        if config.schedule:
            # Parse cron expression and schedule
            # This would integrate with a scheduler like APScheduler
            self.scheduled_jobs[config.source_name] = config

    async def run_batch(self, config: IngestionConfig):
        """Run batch ingestion job"""
        return await self.pipeline.ingest(config)