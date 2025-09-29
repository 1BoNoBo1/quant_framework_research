"""
💾 Data Lake Storage Layer

Unified storage interface for research data with support for
multiple backends (S3, MinIO, local filesystem).
"""

import io
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# Optional storage backend imports
try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

try:
    from minio import Minio
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
from pydantic import BaseModel, Field

from qframe.core.interfaces import DataStorage


class StorageMetadata(BaseModel):
    """Metadata for stored objects"""

    path: str
    size_bytes: int
    created_at: datetime
    modified_at: datetime
    content_type: str
    compression: Optional[str] = None
    checksum: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)


class DataLakeStorage(DataStorage):
    """Abstract base class for data lake storage"""

    @abstractmethod
    async def put_dataframe(
        self,
        df: pd.DataFrame,
        path: str,
        format: str = "parquet",
        partition_cols: Optional[List[str]] = None,
        **kwargs
    ) -> StorageMetadata:
        """Store DataFrame in data lake"""
        pass

    @abstractmethod
    async def get_dataframe(
        self,
        path: str,
        columns: Optional[List[str]] = None,
        filters: Optional[List[tuple]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Retrieve DataFrame from data lake"""
        pass

    @abstractmethod
    async def list_objects(
        self, prefix: str, recursive: bool = True
    ) -> List[StorageMetadata]:
        """List objects in data lake"""
        pass

    @abstractmethod
    async def delete_object(self, path: str) -> bool:
        """Delete object from data lake"""
        pass

    @abstractmethod
    async def get_metadata(self, path: str) -> StorageMetadata:
        """Get object metadata"""
        pass


class S3Storage(DataLakeStorage):
    """AWS S3 data lake storage implementation"""

    def __init__(
        self,
        bucket_name: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: str = "us-east-1",
    ):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )

    async def put_dataframe(
        self,
        df: pd.DataFrame,
        path: str,
        format: str = "parquet",
        partition_cols: Optional[List[str]] = None,
        compression: str = "snappy",
        **kwargs
    ) -> StorageMetadata:
        """Store DataFrame in S3"""
        buffer = io.BytesIO()

        if format == "parquet":
            if partition_cols:
                # Write partitioned dataset
                table = pa.Table.from_pandas(df)
                pq.write_to_dataset(
                    table,
                    root_path=f"s3://{self.bucket_name}/{path}",
                    partition_cols=partition_cols,
                    compression=compression,
                )
            else:
                df.to_parquet(buffer, compression=compression, **kwargs)
        elif format == "csv":
            df.to_csv(buffer, **kwargs)
        elif format == "json":
            df.to_json(buffer, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

        if not partition_cols:
            buffer.seek(0)
            self.s3_client.put_object(
                Bucket=self.bucket_name, Key=path, Body=buffer.getvalue()
            )

        # Get metadata
        response = self.s3_client.head_object(Bucket=self.bucket_name, Key=path)

        return StorageMetadata(
            path=path,
            size_bytes=response["ContentLength"],
            created_at=datetime.now(),
            modified_at=response["LastModified"],
            content_type=response.get("ContentType", f"application/{format}"),
            compression=compression,
            checksum=response.get("ETag", "").strip('"'),
        )

    async def get_dataframe(
        self,
        path: str,
        columns: Optional[List[str]] = None,
        filters: Optional[List[tuple]] = None,
        format: str = "parquet",
        **kwargs
    ) -> pd.DataFrame:
        """Retrieve DataFrame from S3"""
        if format == "parquet":
            # Use S3 filesystem for efficient parquet reading
            df = pd.read_parquet(
                f"s3://{self.bucket_name}/{path}", columns=columns, filters=filters, **kwargs
            )
        else:
            # Download to buffer for other formats
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=path)
            buffer = io.BytesIO(response["Body"].read())
            buffer.seek(0)

            if format == "csv":
                df = pd.read_csv(buffer, **kwargs)
            elif format == "json":
                df = pd.read_json(buffer, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {format}")

        return df

    async def list_objects(
        self, prefix: str, recursive: bool = True
    ) -> List[StorageMetadata]:
        """List objects in S3 bucket"""
        paginator = self.s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)

        objects = []
        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    objects.append(
                        StorageMetadata(
                            path=obj["Key"],
                            size_bytes=obj["Size"],
                            created_at=obj["LastModified"],
                            modified_at=obj["LastModified"],
                            content_type="application/octet-stream",
                            checksum=obj.get("ETag", "").strip('"'),
                        )
                    )

        return objects

    async def delete_object(self, path: str) -> bool:
        """Delete object from S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=path)
            return True
        except Exception:
            return False

    async def get_metadata(self, path: str) -> StorageMetadata:
        """Get S3 object metadata"""
        response = self.s3_client.head_object(Bucket=self.bucket_name, Key=path)

        return StorageMetadata(
            path=path,
            size_bytes=response["ContentLength"],
            created_at=response["LastModified"],
            modified_at=response["LastModified"],
            content_type=response.get("ContentType", "application/octet-stream"),
            checksum=response.get("ETag", "").strip('"'),
        )


class MinIOStorage(DataLakeStorage):
    """MinIO object storage implementation (S3-compatible)"""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket_name: str,
        secure: bool = False,
    ):
        if not MINIO_AVAILABLE:
            raise ImportError("MinIO library not available. Install with: pip install minio")

        self.bucket_name = bucket_name
        self.client = Minio(
            endpoint, access_key=access_key, secret_key=secret_key, secure=secure
        )

        # Create bucket if it doesn't exist
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
        except Exception as e:
            print(f"⚠️ MinIO connection failed: {e}. Using fallback mode.")

    async def put_dataframe(
        self,
        df: pd.DataFrame,
        path: str,
        format: str = "parquet",
        partition_cols: Optional[List[str]] = None,
        compression: str = "snappy",
        **kwargs
    ) -> StorageMetadata:
        """Store DataFrame in MinIO"""
        buffer = io.BytesIO()

        if format == "parquet":
            df.to_parquet(buffer, compression=compression, **kwargs)
        elif format == "csv":
            df.to_csv(buffer, **kwargs)
        elif format == "json":
            df.to_json(buffer, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

        buffer.seek(0)
        size = buffer.getbuffer().nbytes

        self.client.put_object(
            self.bucket_name,
            path,
            buffer,
            length=size,
            content_type=f"application/{format}",
        )

        return StorageMetadata(
            path=path,
            size_bytes=size,
            created_at=datetime.now(),
            modified_at=datetime.now(),
            content_type=f"application/{format}",
            compression=compression if format == "parquet" else None,
        )

    async def get_dataframe(
        self,
        path: str,
        columns: Optional[List[str]] = None,
        filters: Optional[List[tuple]] = None,
        format: str = "parquet",
        **kwargs
    ) -> pd.DataFrame:
        """Retrieve DataFrame from MinIO"""
        response = self.client.get_object(self.bucket_name, path)
        buffer = io.BytesIO(response.read())
        buffer.seek(0)

        if format == "parquet":
            df = pd.read_parquet(buffer, columns=columns, filters=filters, **kwargs)
        elif format == "csv":
            df = pd.read_csv(buffer, **kwargs)
        elif format == "json":
            df = pd.read_json(buffer, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return df

    async def list_objects(
        self, prefix: str, recursive: bool = True
    ) -> List[StorageMetadata]:
        """List objects in MinIO bucket"""
        objects = []
        for obj in self.client.list_objects(
            self.bucket_name, prefix=prefix, recursive=recursive
        ):
            objects.append(
                StorageMetadata(
                    path=obj.object_name,
                    size_bytes=obj.size,
                    created_at=obj.last_modified,
                    modified_at=obj.last_modified,
                    content_type=obj.content_type or "application/octet-stream",
                    checksum=obj.etag,
                )
            )
        return objects

    async def delete_object(self, path: str) -> bool:
        """Delete object from MinIO"""
        try:
            self.client.remove_object(self.bucket_name, path)
            return True
        except Exception:
            return False

    async def get_metadata(self, path: str) -> StorageMetadata:
        """Get MinIO object metadata"""
        stat = self.client.stat_object(self.bucket_name, path)

        return StorageMetadata(
            path=path,
            size_bytes=stat.size,
            created_at=stat.last_modified,
            modified_at=stat.last_modified,
            content_type=stat.content_type or "application/octet-stream",
            checksum=stat.etag,
        )


class LocalFileStorage(DataLakeStorage):
    """Local filesystem storage for development"""

    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def put_dataframe(
        self,
        df: pd.DataFrame,
        path: str,
        format: str = "parquet",
        partition_cols: Optional[List[str]] = None,
        compression: str = "snappy",
        **kwargs
    ) -> StorageMetadata:
        """Store DataFrame locally"""
        full_path = self.base_path / path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "parquet":
            df.to_parquet(full_path, compression=compression, **kwargs)
        elif format == "csv":
            df.to_csv(full_path, **kwargs)
        elif format == "json":
            df.to_json(full_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

        stat = full_path.stat()
        return StorageMetadata(
            path=path,
            size_bytes=stat.st_size,
            created_at=datetime.fromtimestamp(stat.st_ctime),
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            content_type=f"application/{format}",
            compression=compression if format == "parquet" else None,
        )

    async def get_dataframe(
        self,
        path: str,
        columns: Optional[List[str]] = None,
        filters: Optional[List[tuple]] = None,
        format: str = "parquet",
        **kwargs
    ) -> pd.DataFrame:
        """Retrieve DataFrame from local storage"""
        full_path = self.base_path / path

        if format == "parquet":
            df = pd.read_parquet(full_path, columns=columns, filters=filters, **kwargs)
        elif format == "csv":
            df = pd.read_csv(full_path, **kwargs)
        elif format == "json":
            df = pd.read_json(full_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return df

    async def list_objects(
        self, prefix: str, recursive: bool = True
    ) -> List[StorageMetadata]:
        """List objects in local storage"""
        objects = []
        search_path = self.base_path / prefix

        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"

        for file_path in search_path.glob(pattern):
            if file_path.is_file():
                stat = file_path.stat()
                rel_path = file_path.relative_to(self.base_path)
                objects.append(
                    StorageMetadata(
                        path=str(rel_path),
                        size_bytes=stat.st_size,
                        created_at=datetime.fromtimestamp(stat.st_ctime),
                        modified_at=datetime.fromtimestamp(stat.st_mtime),
                        content_type="application/octet-stream",
                    )
                )

        return objects

    async def delete_object(self, path: str) -> bool:
        """Delete object from local storage"""
        try:
            full_path = self.base_path / path
            full_path.unlink()
            return True
        except Exception:
            return False

    async def get_metadata(self, path: str) -> StorageMetadata:
        """Get local file metadata"""
        full_path = self.base_path / path
        stat = full_path.stat()

        return StorageMetadata(
            path=path,
            size_bytes=stat.st_size,
            created_at=datetime.fromtimestamp(stat.st_ctime),
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            content_type="application/octet-stream",
        )