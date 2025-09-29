"""
🔧 Feature Store for QFrame Research

Centralized feature management for machine learning and strategy development.
Provides versioning, lineage tracking, and efficient serving.
"""

import hashlib
import json
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from qframe.core.interfaces import DataProvider


Base = declarative_base()


class FeatureType(str, Enum):
    """Feature data types"""

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    TIMESTAMP = "timestamp"
    TEXT = "text"
    EMBEDDING = "embedding"


class FeatureStatus(str, Enum):
    """Feature lifecycle status"""

    DRAFT = "draft"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"


class Feature(BaseModel):
    """Individual feature definition"""

    name: str
    description: str
    feature_type: FeatureType
    computation_function: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    version: int = 1
    status: FeatureStatus = FeatureStatus.DRAFT
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    author: str = "system"
    tags: List[str] = Field(default_factory=list)

    @validator("name")
    def validate_name(cls, v):
        """Ensure feature names are valid identifiers"""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Feature name must be alphanumeric with underscores/hyphens")
        return v

    def get_hash(self) -> str:
        """Generate unique hash for feature definition"""
        content = f"{self.name}_{self.computation_function}_{self.parameters}_{self.version}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class FeatureGroup(BaseModel):
    """Collection of related features"""

    name: str
    description: str
    features: List[Feature]
    entity_column: str  # Column that identifies the entity (e.g., 'symbol', 'portfolio_id')
    timestamp_column: str = "timestamp"
    online_enabled: bool = False
    offline_enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list)

    def get_feature_names(self) -> List[str]:
        """Get list of feature names in group"""
        return [f.name for f in self.features]


class FeatureComputation:
    """Feature computation engine"""

    def __init__(self):
        self.computation_registry: Dict[str, Callable] = {}
        self._register_default_computations()

    def _register_default_computations(self):
        """Register built-in feature computations"""
        # Price features
        self.register("returns", self._compute_returns)
        self.register("log_returns", self._compute_log_returns)
        self.register("volatility", self._compute_volatility)
        self.register("rsi", self._compute_rsi)
        self.register("macd", self._compute_macd)
        self.register("bollinger_bands", self._compute_bollinger_bands)

        # Volume features
        self.register("volume_profile", self._compute_volume_profile)
        self.register("vwap", self._compute_vwap)
        self.register("volume_ratio", self._compute_volume_ratio)

        # Market microstructure
        self.register("bid_ask_spread", self._compute_bid_ask_spread)
        self.register("order_flow_imbalance", self._compute_order_flow_imbalance)

    def register(self, name: str, func: Callable):
        """Register a feature computation function"""
        self.computation_registry[name] = func

    def compute(self, feature: Feature, data: pd.DataFrame) -> pd.Series:
        """Compute feature values from data"""
        if feature.computation_function in self.computation_registry:
            func = self.computation_registry[feature.computation_function]
            return func(data, **feature.parameters)
        else:
            raise ValueError(f"Unknown computation function: {feature.computation_function}")

    @staticmethod
    def _compute_returns(data: pd.DataFrame, column: str = "close", period: int = 1) -> pd.Series:
        """Compute simple returns"""
        return data[column].pct_change(period)

    @staticmethod
    def _compute_log_returns(data: pd.DataFrame, column: str = "close", period: int = 1) -> pd.Series:
        """Compute log returns"""
        return np.log(data[column] / data[column].shift(period))

    @staticmethod
    def _compute_volatility(data: pd.DataFrame, column: str = "close", window: int = 20) -> pd.Series:
        """Compute rolling volatility"""
        returns = data[column].pct_change()
        return returns.rolling(window).std()

    @staticmethod
    def _compute_rsi(data: pd.DataFrame, column: str = "close", period: int = 14) -> pd.Series:
        """Compute RSI indicator"""
        delta = data[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _compute_macd(
        data: pd.DataFrame, column: str = "close", fast: int = 12, slow: int = 26, signal: int = 9
    ) -> pd.Series:
        """Compute MACD indicator"""
        exp1 = data[column].ewm(span=fast, adjust=False).mean()
        exp2 = data[column].ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line - signal_line

    @staticmethod
    def _compute_bollinger_bands(
        data: pd.DataFrame, column: str = "close", window: int = 20, num_std: float = 2
    ) -> pd.DataFrame:
        """Compute Bollinger Bands"""
        rolling_mean = data[column].rolling(window).mean()
        rolling_std = data[column].rolling(window).std()

        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)

        return pd.DataFrame(
            {
                "bb_middle": rolling_mean,
                "bb_upper": upper_band,
                "bb_lower": lower_band,
                "bb_width": upper_band - lower_band,
                "bb_position": (data[column] - lower_band) / (upper_band - lower_band),
            }
        )

    @staticmethod
    def _compute_volume_profile(
        data: pd.DataFrame, price_col: str = "close", volume_col: str = "volume", bins: int = 20
    ) -> pd.Series:
        """Compute volume profile"""
        price_bins = pd.qcut(data[price_col], bins, duplicates="drop")
        return data.groupby(price_bins)[volume_col].sum()

    @staticmethod
    def _compute_vwap(data: pd.DataFrame, window: int = None) -> pd.Series:
        """Compute Volume-Weighted Average Price"""
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        if window:
            return (typical_price * data["volume"]).rolling(window).sum() / data[
                "volume"
            ].rolling(window).sum()
        else:
            return (typical_price * data["volume"]).cumsum() / data["volume"].cumsum()

    @staticmethod
    def _compute_volume_ratio(
        data: pd.DataFrame, short_window: int = 5, long_window: int = 20
    ) -> pd.Series:
        """Compute volume ratio"""
        short_vol = data["volume"].rolling(short_window).mean()
        long_vol = data["volume"].rolling(long_window).mean()
        return short_vol / long_vol

    @staticmethod
    def _compute_bid_ask_spread(data: pd.DataFrame) -> pd.Series:
        """Compute bid-ask spread"""
        if "bid" in data.columns and "ask" in data.columns:
            return data["ask"] - data["bid"]
        else:
            # Estimate from high-low if bid/ask not available
            return data["high"] - data["low"]

    @staticmethod
    def _compute_order_flow_imbalance(data: pd.DataFrame) -> pd.Series:
        """Compute order flow imbalance"""
        if "buy_volume" in data.columns and "sell_volume" in data.columns:
            total_volume = data["buy_volume"] + data["sell_volume"]
            return (data["buy_volume"] - data["sell_volume"]) / total_volume
        else:
            # Estimate from price changes if order flow not available
            price_change = data["close"].diff()
            return price_change.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)


class FeatureStore:
    """Main feature store interface"""

    def __init__(self, storage_backend, database_url: Optional[str] = None):
        self.storage = storage_backend
        self.computation_engine = FeatureComputation()
        self.feature_groups: Dict[str, FeatureGroup] = {}
        self.feature_cache: Dict[str, pd.DataFrame] = {}

        # Setup database for feature metadata
        if database_url:
            self.engine = create_engine(database_url)
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)

    def register_feature_group(self, feature_group: FeatureGroup):
        """Register a new feature group"""
        self.feature_groups[feature_group.name] = feature_group

    def compute_features(
        self, feature_group_name: str, data: pd.DataFrame, features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Compute features for given data"""
        if feature_group_name not in self.feature_groups:
            raise ValueError(f"Feature group {feature_group_name} not found")

        feature_group = self.feature_groups[feature_group_name]
        result_df = data.copy()

        # Filter features if specified
        features_to_compute = feature_group.features
        if features:
            features_to_compute = [f for f in feature_group.features if f.name in features]

        # Compute each feature
        for feature in features_to_compute:
            try:
                feature_values = self.computation_engine.compute(feature, data)
                if isinstance(feature_values, pd.DataFrame):
                    # Multi-column features
                    for col in feature_values.columns:
                        result_df[col] = feature_values[col]
                else:
                    result_df[feature.name] = feature_values
            except Exception as e:
                print(f"Error computing feature {feature.name}: {e}")
                result_df[feature.name] = np.nan

        return result_df

    async def save_features(
        self,
        feature_group_name: str,
        features_df: pd.DataFrame,
        timestamp: Optional[datetime] = None,
    ):
        """Save computed features to storage"""
        if timestamp is None:
            timestamp = datetime.now()

        # Create path for features
        path = f"features/{feature_group_name}/{timestamp.strftime('%Y%m%d_%H%M%S')}.parquet"

        # Save to storage
        await self.storage.put_dataframe(features_df, path, format="parquet", compression="snappy")

        # Update cache
        self.feature_cache[feature_group_name] = features_df

    async def load_features(
        self,
        feature_group_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Load features from storage"""
        # Check cache first
        if feature_group_name in self.feature_cache:
            df = self.feature_cache[feature_group_name]
            if features:
                df = df[features]
            return df

        # Load from storage
        prefix = f"features/{feature_group_name}/"
        objects = await self.storage.list_objects(prefix)

        # Filter by time range if specified
        if start_time or end_time:
            filtered_objects = []
            for obj in objects:
                # Extract timestamp from filename
                filename = obj.path.split("/")[-1].replace(".parquet", "")
                file_time = datetime.strptime(filename, "%Y%m%d_%H%M%S")

                if start_time and file_time < start_time:
                    continue
                if end_time and file_time > end_time:
                    continue

                filtered_objects.append(obj)
            objects = filtered_objects

        if not objects:
            raise ValueError(f"No features found for {feature_group_name}")

        # Load and concatenate all matching files
        dfs = []
        for obj in objects:
            df = await self.storage.get_dataframe(obj.path, columns=features)
            dfs.append(df)

        result_df = pd.concat(dfs, ignore_index=True)

        # Update cache
        self.feature_cache[feature_group_name] = result_df

        return result_df

    def get_feature_statistics(self, feature_group_name: str) -> pd.DataFrame:
        """Get statistics for features in a group"""
        if feature_group_name not in self.feature_cache:
            raise ValueError(f"Feature group {feature_group_name} not in cache")

        df = self.feature_cache[feature_group_name]
        stats = df.describe(include="all").T

        # Add additional statistics
        stats["missing_pct"] = (df.isnull().sum() / len(df)) * 100
        stats["unique_count"] = df.nunique()
        stats["dtype"] = df.dtypes

        return stats

    def validate_features(self, feature_group_name: str) -> Dict[str, List[str]]:
        """Validate feature quality"""
        if feature_group_name not in self.feature_cache:
            raise ValueError(f"Feature group {feature_group_name} not in cache")

        df = self.feature_cache[feature_group_name]
        validation_results = {"errors": [], "warnings": []}

        # Check for missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            validation_results["warnings"].append(f"Missing values in columns: {missing_cols}")

        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_cols = [col for col in numeric_cols if np.isinf(df[col]).any()]
        if inf_cols:
            validation_results["errors"].append(f"Infinite values in columns: {inf_cols}")

        # Check for constant features
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            validation_results["warnings"].append(f"Constant features: {constant_cols}")

        # Check for highly correlated features
        corr_matrix = df[numeric_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = [(col, row) for col in upper_tri.columns for row in upper_tri.index if upper_tri.loc[row, col] > 0.95]
        if high_corr:
            validation_results["warnings"].append(f"Highly correlated features (>0.95): {high_corr}")

        return validation_results