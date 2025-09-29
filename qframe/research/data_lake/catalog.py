"""
📚 Data Catalog for QFrame Data Lake

Metadata management and discovery for datasets, features, and models.
"""

import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from pydantic import BaseModel, Field
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()


class DatasetType(str, Enum):
    """Types of datasets in the data lake"""

    RAW_MARKET_DATA = "raw_market_data"
    PROCESSED_DATA = "processed_data"
    FEATURES = "features"
    PREDICTIONS = "predictions"
    BACKTEST_RESULTS = "backtest_results"
    MODEL_ARTIFACTS = "model_artifacts"
    RESEARCH_OUTPUTS = "research_outputs"


class DataQuality(str, Enum):
    """Data quality levels"""

    RAW = "raw"
    VALIDATED = "validated"
    CLEANED = "cleaned"
    PRODUCTION = "production"


class DatasetMetadata(BaseModel):
    """Metadata for a dataset"""

    name: str
    description: str
    dataset_type: DatasetType
    quality_level: DataQuality
    schema_version: str = "1.0.0"

    # Storage information
    storage_path: str
    format: str  # parquet, csv, json
    compression: Optional[str] = None
    size_bytes: int
    row_count: Optional[int] = None
    column_count: Optional[int] = None

    # Temporal information
    created_at: datetime
    updated_at: datetime
    data_start_date: Optional[datetime] = None
    data_end_date: Optional[datetime] = None

    # Lineage and versioning
    source_datasets: List[str] = Field(default_factory=list)
    derived_datasets: List[str] = Field(default_factory=list)
    version: int = 1
    previous_version_path: Optional[str] = None

    # Discovery and search
    tags: List[str] = Field(default_factory=list)
    owner: str
    team: Optional[str] = None

    # Schema information
    columns: Dict[str, str] = Field(default_factory=dict)  # column_name: data_type
    primary_keys: List[str] = Field(default_factory=list)
    partition_columns: List[str] = Field(default_factory=list)

    # Quality metrics
    completeness_score: Optional[float] = None  # 0-1
    accuracy_score: Optional[float] = None  # 0-1
    consistency_score: Optional[float] = None  # 0-1

    # Access information
    access_frequency: int = 0
    last_accessed: Optional[datetime] = None
    is_public: bool = True


class DatasetEntity(Base):
    """SQLAlchemy entity for dataset metadata"""

    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    description = Column(Text)
    dataset_type = Column(String(50))
    quality_level = Column(String(50))
    schema_version = Column(String(50))

    storage_path = Column(String(500))
    format = Column(String(50))
    compression = Column(String(50))
    size_bytes = Column(Integer)
    row_count = Column(Integer)
    column_count = Column(Integer)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    data_start_date = Column(DateTime)
    data_end_date = Column(DateTime)

    version = Column(Integer, default=1)
    previous_version_path = Column(String(500))

    tags = Column(JSON)
    owner = Column(String(255))
    team = Column(String(255))

    columns = Column(JSON)
    primary_keys = Column(JSON)
    partition_columns = Column(JSON)

    completeness_score = Column(Float)
    accuracy_score = Column(Float)
    consistency_score = Column(Float)

    access_frequency = Column(Integer, default=0)
    last_accessed = Column(DateTime)
    is_public = Column(Integer, default=1)

    # Relationships
    lineage_records = relationship("DataLineageEntity", back_populates="dataset")


class DataLineageEntity(Base):
    """Track data lineage relationships"""

    __tablename__ = "data_lineage"

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    parent_dataset = Column(String(255))
    child_dataset = Column(String(255))
    transformation = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("DatasetEntity", back_populates="lineage_records")


class DataCatalog:
    """Central data catalog for the data lake"""

    def __init__(self, database_url: str, storage_backend):
        self.storage = storage_backend
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        # In-memory cache for faster lookups
        self._cache: Dict[str, DatasetMetadata] = {}

    def register_dataset(self, metadata: DatasetMetadata) -> str:
        """Register a new dataset in the catalog"""
        session = self.Session()
        try:
            # Convert to entity
            entity = DatasetEntity(
                name=metadata.name,
                description=metadata.description,
                dataset_type=metadata.dataset_type.value,
                quality_level=metadata.quality_level.value,
                schema_version=metadata.schema_version,
                storage_path=metadata.storage_path,
                format=metadata.format,
                compression=metadata.compression,
                size_bytes=metadata.size_bytes,
                row_count=metadata.row_count,
                column_count=metadata.column_count,
                created_at=metadata.created_at,
                updated_at=metadata.updated_at,
                data_start_date=metadata.data_start_date,
                data_end_date=metadata.data_end_date,
                version=metadata.version,
                previous_version_path=metadata.previous_version_path,
                tags=metadata.tags,
                owner=metadata.owner,
                team=metadata.team,
                columns=metadata.columns,
                primary_keys=metadata.primary_keys,
                partition_columns=metadata.partition_columns,
                completeness_score=metadata.completeness_score,
                accuracy_score=metadata.accuracy_score,
                consistency_score=metadata.consistency_score,
                access_frequency=metadata.access_frequency,
                last_accessed=metadata.last_accessed,
                is_public=int(metadata.is_public),
            )

            session.add(entity)
            session.commit()

            # Update cache
            self._cache[metadata.name] = metadata

            # Track lineage
            for source in metadata.source_datasets:
                lineage = DataLineageEntity(
                    dataset_id=entity.id,
                    parent_dataset=source,
                    child_dataset=metadata.name,
                )
                session.add(lineage)

            session.commit()
            return metadata.name

        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_dataset(self, name: str) -> Optional[DatasetMetadata]:
        """Get dataset metadata by name"""
        # Check cache first
        if name in self._cache:
            return self._cache[name]

        session = self.Session()
        try:
            entity = session.query(DatasetEntity).filter_by(name=name).first()
            if entity:
                metadata = self._entity_to_metadata(entity)
                self._cache[name] = metadata

                # Update access statistics
                entity.access_frequency += 1
                entity.last_accessed = datetime.utcnow()
                session.commit()

                return metadata
            return None
        finally:
            session.close()

    def search_datasets(
        self,
        dataset_type: Optional[DatasetType] = None,
        tags: Optional[List[str]] = None,
        owner: Optional[str] = None,
        quality_level: Optional[DataQuality] = None,
        date_range: Optional[tuple[datetime, datetime]] = None,
    ) -> List[DatasetMetadata]:
        """Search for datasets matching criteria"""
        session = self.Session()
        try:
            query = session.query(DatasetEntity)

            if dataset_type:
                query = query.filter(DatasetEntity.dataset_type == dataset_type.value)

            if owner:
                query = query.filter(DatasetEntity.owner == owner)

            if quality_level:
                query = query.filter(DatasetEntity.quality_level == quality_level.value)

            if date_range:
                start_date, end_date = date_range
                query = query.filter(
                    DatasetEntity.data_start_date >= start_date,
                    DatasetEntity.data_end_date <= end_date,
                )

            entities = query.all()

            # Filter by tags if specified
            if tags:
                entities = [e for e in entities if any(tag in e.tags for tag in tags)]

            return [self._entity_to_metadata(e) for e in entities]

        finally:
            session.close()

    def get_lineage(self, dataset_name: str, depth: int = 3) -> Dict[str, Any]:
        """Get data lineage for a dataset"""
        session = self.Session()
        try:
            lineage_graph = {"nodes": [], "edges": []}
            visited = set()

            def traverse_lineage(name: str, current_depth: int):
                if current_depth > depth or name in visited:
                    return

                visited.add(name)
                lineage_graph["nodes"].append({"id": name, "label": name})

                # Get parent datasets
                entity = session.query(DatasetEntity).filter_by(name=name).first()
                if entity:
                    lineage_records = (
                        session.query(DataLineageEntity)
                        .filter_by(child_dataset=name)
                        .all()
                    )

                    for record in lineage_records:
                        parent = record.parent_dataset
                        lineage_graph["edges"].append(
                            {"source": parent, "target": name, "label": record.transformation or ""}
                        )
                        traverse_lineage(parent, current_depth + 1)

                    # Get child datasets
                    child_records = (
                        session.query(DataLineageEntity)
                        .filter_by(parent_dataset=name)
                        .all()
                    )

                    for record in child_records:
                        child = record.child_dataset
                        lineage_graph["edges"].append(
                            {"source": name, "target": child, "label": record.transformation or ""}
                        )
                        traverse_lineage(child, current_depth + 1)

            traverse_lineage(dataset_name, 0)
            return lineage_graph

        finally:
            session.close()

    def update_dataset(self, name: str, updates: Dict[str, Any]):
        """Update dataset metadata"""
        session = self.Session()
        try:
            entity = session.query(DatasetEntity).filter_by(name=name).first()
            if entity:
                for key, value in updates.items():
                    if hasattr(entity, key):
                        setattr(entity, key, value)

                entity.updated_at = datetime.utcnow()
                entity.version += 1
                session.commit()

                # Invalidate cache
                if name in self._cache:
                    del self._cache[name]

                return True
            return False

        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def delete_dataset(self, name: str, delete_storage: bool = False):
        """Delete dataset from catalog"""
        session = self.Session()
        try:
            entity = session.query(DatasetEntity).filter_by(name=name).first()
            if entity:
                # Optionally delete from storage
                if delete_storage and entity.storage_path:
                    self.storage.delete_object(entity.storage_path)

                # Delete from database
                session.delete(entity)
                session.commit()

                # Remove from cache
                if name in self._cache:
                    del self._cache[name]

                return True
            return False

        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_statistics(self) -> Dict[str, Any]:
        """Get catalog statistics"""
        session = self.Session()
        try:
            total_datasets = session.query(DatasetEntity).count()

            stats = {
                "total_datasets": total_datasets,
                "datasets_by_type": {},
                "datasets_by_quality": {},
                "total_size_gb": 0,
                "most_accessed": [],
                "recently_updated": [],
            }

            # Group by type
            for dtype in DatasetType:
                count = (
                    session.query(DatasetEntity)
                    .filter_by(dataset_type=dtype.value)
                    .count()
                )
                stats["datasets_by_type"][dtype.value] = count

            # Group by quality
            for quality in DataQuality:
                count = (
                    session.query(DatasetEntity)
                    .filter_by(quality_level=quality.value)
                    .count()
                )
                stats["datasets_by_quality"][quality.value] = count

            # Total size
            total_size = session.query(DatasetEntity.size_bytes).all()
            stats["total_size_gb"] = sum(s[0] for s in total_size if s[0]) / (1024**3)

            # Most accessed
            most_accessed = (
                session.query(DatasetEntity)
                .order_by(DatasetEntity.access_frequency.desc())
                .limit(10)
                .all()
            )
            stats["most_accessed"] = [e.name for e in most_accessed]

            # Recently updated
            recently_updated = (
                session.query(DatasetEntity)
                .order_by(DatasetEntity.updated_at.desc())
                .limit(10)
                .all()
            )
            stats["recently_updated"] = [e.name for e in recently_updated]

            return stats

        finally:
            session.close()

    def _entity_to_metadata(self, entity: DatasetEntity) -> DatasetMetadata:
        """Convert database entity to metadata model"""
        return DatasetMetadata(
            name=entity.name,
            description=entity.description or "",
            dataset_type=DatasetType(entity.dataset_type),
            quality_level=DataQuality(entity.quality_level),
            schema_version=entity.schema_version or "1.0.0",
            storage_path=entity.storage_path or "",
            format=entity.format or "parquet",
            compression=entity.compression,
            size_bytes=entity.size_bytes or 0,
            row_count=entity.row_count,
            column_count=entity.column_count,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            data_start_date=entity.data_start_date,
            data_end_date=entity.data_end_date,
            version=entity.version or 1,
            previous_version_path=entity.previous_version_path,
            tags=entity.tags or [],
            owner=entity.owner or "system",
            team=entity.team,
            columns=entity.columns or {},
            primary_keys=entity.primary_keys or [],
            partition_columns=entity.partition_columns or [],
            completeness_score=entity.completeness_score,
            accuracy_score=entity.accuracy_score,
            consistency_score=entity.consistency_score,
            access_frequency=entity.access_frequency or 0,
            last_accessed=entity.last_accessed,
            is_public=bool(entity.is_public),
        )