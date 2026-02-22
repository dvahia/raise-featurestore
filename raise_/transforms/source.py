"""
Raise Transform Sources

Data source connectors for transformations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Iterator


class SourceType(Enum):
    """Source type enumeration."""
    OBJECT_STORAGE = "object_storage"
    FILE_SYSTEM = "file_system"
    COLUMNAR = "columnar"
    FEATURE_GROUP = "feature_group"
    DATABASE = "database"


class FileFormat(Enum):
    """Supported file formats."""
    PARQUET = "parquet"
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    AVRO = "avro"
    ORC = "orc"
    DELTA = "delta"
    ICEBERG = "iceberg"


@dataclass
class Source(ABC):
    """
    Base class for data sources.

    Sources define where transformation input data comes from.
    """

    name: str | None = None

    @property
    @abstractmethod
    def source_type(self) -> SourceType:
        """Return the source type."""
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize source to dictionary."""
        pass

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Source:
        """Deserialize source from dictionary."""
        source_type = data.get("source_type")
        if source_type == "object_storage":
            return ObjectStorage.from_dict(data)
        elif source_type == "file_system":
            return FileSystem.from_dict(data)
        elif source_type == "columnar":
            return ColumnarSource.from_dict(data)
        elif source_type == "feature_group":
            return FeatureGroupSource.from_dict(data)
        elif source_type == "database":
            return DatabaseSource.from_dict(data)
        else:
            raise ValueError(f"Unknown source type: {source_type}")


@dataclass
class ObjectStorage(Source):
    """
    Object storage source (S3, GCS, Azure Blob).

    Attributes:
        path: Object storage path (e.g., "s3://bucket/prefix/")
        format: File format (parquet, csv, json, etc.)
        partition_columns: Columns used for partitioning
        options: Additional reader options
    """

    path: str = ""
    format: FileFormat | str = FileFormat.PARQUET
    partition_columns: list[str] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.format, str):
            self.format = FileFormat(self.format.lower())

    @property
    def source_type(self) -> SourceType:
        return SourceType.OBJECT_STORAGE

    @property
    def protocol(self) -> str:
        """Extract protocol from path (s3, gs, az, etc.)."""
        if "://" in self.path:
            return self.path.split("://")[0]
        return "file"

    @property
    def bucket(self) -> str | None:
        """Extract bucket name from path."""
        if "://" in self.path:
            parts = self.path.split("://")[1].split("/")
            return parts[0] if parts else None
        return None

    @property
    def prefix(self) -> str:
        """Extract prefix from path."""
        if "://" in self.path:
            parts = self.path.split("://")[1].split("/", 1)
            return parts[1] if len(parts) > 1 else ""
        return self.path

    def with_partition(self, **partitions: Any) -> ObjectStorage:
        """Create a new source with partition filters applied."""
        partition_path = "/".join(f"{k}={v}" for k, v in partitions.items())
        new_path = f"{self.path.rstrip('/')}/{partition_path}/"
        return ObjectStorage(
            name=self.name,
            path=new_path,
            format=self.format,
            partition_columns=self.partition_columns,
            options=self.options,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": "object_storage",
            "name": self.name,
            "path": self.path,
            "format": self.format.value if isinstance(self.format, FileFormat) else self.format,
            "partition_columns": self.partition_columns,
            "options": self.options,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjectStorage:
        return cls(
            name=data.get("name"),
            path=data.get("path", ""),
            format=data.get("format", "parquet"),
            partition_columns=data.get("partition_columns", []),
            options=data.get("options", {}),
        )


@dataclass
class FileSystem(Source):
    """
    Local or network file system source.

    Attributes:
        path: File system path
        format: File format
        glob_pattern: Optional glob pattern for file matching
        options: Additional reader options
    """

    path: str = ""
    format: FileFormat | str = FileFormat.PARQUET
    glob_pattern: str | None = None
    recursive: bool = False
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.format, str):
            self.format = FileFormat(self.format.lower())

    @property
    def source_type(self) -> SourceType:
        return SourceType.FILE_SYSTEM

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": "file_system",
            "name": self.name,
            "path": self.path,
            "format": self.format.value if isinstance(self.format, FileFormat) else self.format,
            "glob_pattern": self.glob_pattern,
            "recursive": self.recursive,
            "options": self.options,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileSystem:
        return cls(
            name=data.get("name"),
            path=data.get("path", ""),
            format=data.get("format", "parquet"),
            glob_pattern=data.get("glob_pattern"),
            recursive=data.get("recursive", False),
            options=data.get("options", {}),
        )


@dataclass
class ColumnarSource(Source):
    """
    Columnar storage source (data warehouse tables).

    Attributes:
        table: Fully qualified table name
        database: Database/schema name
        catalog: Catalog name (for systems like Iceberg/Delta)
        columns: Specific columns to read (None = all)
        filter: SQL WHERE clause filter
        options: Additional reader options
    """

    table: str = ""
    database: str | None = None
    catalog: str | None = None
    columns: list[str] | None = None
    filter: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def source_type(self) -> SourceType:
        return SourceType.COLUMNAR

    @property
    def qualified_name(self) -> str:
        """Return fully qualified table name."""
        parts = []
        if self.catalog:
            parts.append(self.catalog)
        if self.database:
            parts.append(self.database)
        parts.append(self.table)
        return ".".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": "columnar",
            "name": self.name,
            "table": self.table,
            "database": self.database,
            "catalog": self.catalog,
            "columns": self.columns,
            "filter": self.filter,
            "options": self.options,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ColumnarSource:
        return cls(
            name=data.get("name"),
            table=data.get("table", ""),
            database=data.get("database"),
            catalog=data.get("catalog"),
            columns=data.get("columns"),
            filter=data.get("filter"),
            options=data.get("options", {}),
        )


@dataclass
class FeatureGroupSource(Source):
    """
    Feature group source (read from existing features).

    Attributes:
        feature_group: Feature group path (e.g., "org/domain/project/group")
        features: List of feature names to read (None = all)
        version: Feature version to read (None = latest)
        filter: SQL WHERE clause filter
    """

    feature_group: str = ""
    features: list[str] | None = None
    version: str | None = None
    filter: str | None = None

    @property
    def source_type(self) -> SourceType:
        return SourceType.FEATURE_GROUP

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": "feature_group",
            "name": self.name,
            "feature_group": self.feature_group,
            "features": self.features,
            "version": self.version,
            "filter": self.filter,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureGroupSource:
        return cls(
            name=data.get("name"),
            feature_group=data.get("feature_group", ""),
            features=data.get("features"),
            version=data.get("version"),
            filter=data.get("filter"),
        )


@dataclass
class DatabaseSource(Source):
    """
    Database source (JDBC/ODBC connection).

    Attributes:
        connection: Connection string or connection name
        query: SQL query to execute
        table: Table name (alternative to query)
        options: Additional connection options
    """

    connection: str = ""
    query: str | None = None
    table: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def source_type(self) -> SourceType:
        return SourceType.DATABASE

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": "database",
            "name": self.name,
            "connection": self.connection,
            "query": self.query,
            "table": self.table,
            "options": self.options,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatabaseSource:
        return cls(
            name=data.get("name"),
            connection=data.get("connection", ""),
            query=data.get("query"),
            table=data.get("table"),
            options=data.get("options", {}),
        )


@dataclass
class SourceRegistry:
    """Registry for managing named sources."""

    _sources: dict[str, Source] = field(default_factory=dict)

    def register(self, source: Source) -> None:
        """Register a source by name."""
        if not source.name:
            raise ValueError("Source must have a name to be registered")
        self._sources[source.name] = source

    def get(self, name: str) -> Source:
        """Get a source by name."""
        if name not in self._sources:
            raise ValueError(f"Source not found: {name}")
        return self._sources[name]

    def list(self) -> list[Source]:
        """List all registered sources."""
        return list(self._sources.values())

    def delete(self, name: str) -> bool:
        """Delete a source by name."""
        if name in self._sources:
            del self._sources[name]
            return True
        return False
