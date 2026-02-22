"""
Raise Transformations

Core transformation definitions for SQL and Python transforms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, TYPE_CHECKING

from raise_.transforms.source import Source
from raise_.transforms.checkpoint import IncrementalConfig, ProcessingMode


class TransformType(Enum):
    """Transform type enumeration."""
    SQL = "sql"
    PYTHON = "python"
    HYBRID = "hybrid"  # SQL + Python post-processing


@dataclass
class TransformContext:
    """
    Runtime context for transform execution.

    Provides access to execution metadata and utilities.
    """

    job_id: str
    run_id: str
    execution_date: datetime
    is_incremental: bool = True
    checkpoint_value: Any = None
    params: dict[str, Any] = field(default_factory=dict)

    # Runtime data (populated during execution)
    source_data: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def log_metric(self, name: str, value: Any) -> None:
        """Log a metric for observability."""
        self.metrics[name] = value

    def get_param(self, name: str, default: Any = None) -> Any:
        """Get a parameter value."""
        return self.params.get(name, default)


@dataclass
class Transform(ABC):
    """
    Base class for transformations.

    Transforms define how to convert source data into feature data.
    """

    name: str
    description: str | None = None
    owner: str | None = None
    tags: list[str] = field(default_factory=list)

    @property
    @abstractmethod
    def transform_type(self) -> TransformType:
        """Return the transform type."""
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize transform to dictionary."""
        pass

    @abstractmethod
    def get_sql(self, context: TransformContext) -> str | None:
        """Get SQL query for execution (if applicable)."""
        pass

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transform:
        """Deserialize transform from dictionary."""
        transform_type = data.get("transform_type")
        if transform_type == "sql":
            return SQLTransform.from_dict(data)
        elif transform_type == "python":
            return PythonTransform.from_dict(data)
        elif transform_type == "hybrid":
            return HybridTransform.from_dict(data)
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")


@dataclass
class SQLTransform(Transform):
    """
    SQL-based transformation.

    Attributes:
        sql: SQL query template
        source_aliases: Mapping of source names to SQL aliases
        parameters: Named parameters for SQL template
    """

    sql: str = ""
    source_aliases: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def transform_type(self) -> TransformType:
        return TransformType.SQL

    def get_sql(self, context: TransformContext) -> str:
        """
        Get SQL query with parameters substituted.

        Supports:
        - {{param_name}} for parameter substitution
        - {{checkpoint}} for checkpoint value
        - {{execution_date}} for execution date
        """
        sql = self.sql

        # Substitute built-in context values
        sql = sql.replace("{{checkpoint}}", self._format_value(context.checkpoint_value))
        sql = sql.replace("{{execution_date}}", f"'{context.execution_date.isoformat()}'")
        sql = sql.replace("{{run_id}}", f"'{context.run_id}'")

        # Substitute parameters
        all_params = {**self.parameters, **context.params}
        for key, value in all_params.items():
            sql = sql.replace(f"{{{{{key}}}}}", self._format_value(value))

        return sql

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format a value for SQL."""
        if value is None:
            return "NULL"
        elif isinstance(value, str):
            return f"'{value}'"
        elif isinstance(value, datetime):
            return f"'{value.isoformat()}'"
        elif isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        else:
            return str(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform_type": "sql",
            "name": self.name,
            "description": self.description,
            "owner": self.owner,
            "tags": self.tags,
            "sql": self.sql,
            "source_aliases": self.source_aliases,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SQLTransform:
        return cls(
            name=data.get("name", ""),
            description=data.get("description"),
            owner=data.get("owner"),
            tags=data.get("tags", []),
            sql=data.get("sql", ""),
            source_aliases=data.get("source_aliases", {}),
            parameters=data.get("parameters", {}),
        )


# Type alias for Python transform functions
TransformFunction = Callable[[TransformContext, Any], Any]


@dataclass
class PythonTransform(Transform):
    """
    Python-based transformation.

    Attributes:
        function: Python function that performs the transformation
        function_name: Name of the function (for serialization)
        module_path: Module path containing the function
        dependencies: Python package dependencies
    """

    function: TransformFunction | None = None
    function_name: str | None = None
    module_path: str | None = None
    dependencies: list[str] = field(default_factory=list)

    @property
    def transform_type(self) -> TransformType:
        return TransformType.PYTHON

    def get_sql(self, context: TransformContext) -> str | None:
        return None  # Python transforms don't use SQL

    def execute(self, context: TransformContext, data: Any) -> Any:
        """Execute the Python transform function."""
        if self.function is None:
            raise ValueError("Transform function not set")
        return self.function(context, data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform_type": "python",
            "name": self.name,
            "description": self.description,
            "owner": self.owner,
            "tags": self.tags,
            "function_name": self.function_name,
            "module_path": self.module_path,
            "dependencies": self.dependencies,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PythonTransform:
        return cls(
            name=data.get("name", ""),
            description=data.get("description"),
            owner=data.get("owner"),
            tags=data.get("tags", []),
            function_name=data.get("function_name"),
            module_path=data.get("module_path"),
            dependencies=data.get("dependencies", []),
        )

    @staticmethod
    def from_function(
        func: TransformFunction,
        name: str | None = None,
        description: str | None = None,
        dependencies: list[str] | None = None,
    ) -> PythonTransform:
        """Create a PythonTransform from a function."""
        return PythonTransform(
            name=name or func.__name__,
            description=description or func.__doc__,
            function=func,
            function_name=func.__name__,
            module_path=func.__module__,
            dependencies=dependencies or [],
        )


@dataclass
class HybridTransform(Transform):
    """
    Hybrid SQL + Python transformation.

    Executes SQL first, then applies Python post-processing.

    Attributes:
        sql_transform: SQL transform to execute first
        python_transform: Python transform for post-processing
    """

    sql_transform: SQLTransform | None = None
    python_transform: PythonTransform | None = None

    @property
    def transform_type(self) -> TransformType:
        return TransformType.HYBRID

    def get_sql(self, context: TransformContext) -> str | None:
        if self.sql_transform:
            return self.sql_transform.get_sql(context)
        return None

    def execute_python(self, context: TransformContext, data: Any) -> Any:
        """Execute the Python post-processing."""
        if self.python_transform:
            return self.python_transform.execute(context, data)
        return data

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform_type": "hybrid",
            "name": self.name,
            "description": self.description,
            "owner": self.owner,
            "tags": self.tags,
            "sql_transform": self.sql_transform.to_dict() if self.sql_transform else None,
            "python_transform": self.python_transform.to_dict() if self.python_transform else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HybridTransform:
        return cls(
            name=data.get("name", ""),
            description=data.get("description"),
            owner=data.get("owner"),
            tags=data.get("tags", []),
            sql_transform=SQLTransform.from_dict(data["sql_transform"]) if data.get("sql_transform") else None,
            python_transform=PythonTransform.from_dict(data["python_transform"]) if data.get("python_transform") else None,
        )


def sql_transform(
    name: str,
    sql: str,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
    **kwargs,
) -> SQLTransform:
    """Convenience function to create a SQL transform."""
    return SQLTransform(
        name=name,
        sql=sql,
        description=description,
        parameters=parameters or {},
        **kwargs,
    )


def python_transform(
    func: TransformFunction | None = None,
    name: str | None = None,
    description: str | None = None,
    dependencies: list[str] | None = None,
):
    """
    Decorator to create a Python transform from a function.

    Usage:
        @python_transform(name="my_transform")
        def my_transform(context, data):
            return data.transform()
    """
    def decorator(f: TransformFunction) -> PythonTransform:
        return PythonTransform.from_function(
            f,
            name=name,
            description=description,
            dependencies=dependencies,
        )

    if func is not None:
        return decorator(func)
    return decorator
