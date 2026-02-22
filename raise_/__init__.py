"""
Raise - Feature Store API for ML Infrastructure

A Python API for managing ML features in columnar storage backends.
Designed for AI researchers working in notebook environments.
"""

from raise_.client import FeatureStore
from raise_.models.acl import ACL
from raise_.models.feature import Feature
from raise_.models.feature_group import FeatureGroup
from raise_.models.project import Project
from raise_.models.domain import Domain
from raise_.models.organization import Organization
from raise_.models.lineage import Lineage, LineageGraph
from raise_.models.audit import AuditEntry, AuditQuery, AuditAlert
from raise_.models.types import (
    FeatureType,
    Int64,
    Float32,
    Float64,
    Bool,
    String,
    Bytes,
    Timestamp,
    Embedding,
    Array,
    Struct,
)
from raise_.validation import ValidationResult, ValidationError, ValidationWarning
from raise_.analytics import (
    Analysis,
    Aggregation,
    Distribution,
    Correlation,
    VersionDiff,
    StatTest,
    RecordLookup,
    Freshness,
    AnalysisResult,
    AnalysisJob,
    LiveTable,
    RefreshPolicy,
    Dashboard,
    Chart,
    ChartType,
    DashboardParameter,
    ParameterType,
    AnalyticsAlert,
    Condition,
)
from raise_.transforms import (
    # Sources
    Source,
    ObjectStorage,
    FileSystem,
    ColumnarSource,
    FeatureGroupSource,
    # Schedules
    Schedule,
    # Transforms
    Transform,
    SQLTransform,
    PythonTransform,
    sql_transform,
    python_transform,
    # Jobs
    Job,
    Target,
    IncrementalConfig,
    # Orchestrators
    AirflowConfig,
    generate_airflow_dag,
    # Quality
    QualityCheck,
    NullCheck,
    UniqueCheck,
    RangeCheck,
    RowCountCheck,
)
from raise_.exceptions import (
    RaiseError,
    FeatureExistsError,
    FeatureNotFoundError,
    ValidationError as FeatureValidationError,
    AccessDeniedError,
    CrossOrgAccessError,
)

__version__ = "0.1.0"

__all__ = [
    # Client
    "FeatureStore",
    # Models
    "ACL",
    "Feature",
    "FeatureGroup",
    "Project",
    "Domain",
    "Organization",
    "Lineage",
    "LineageGraph",
    # Audit
    "AuditEntry",
    "AuditQuery",
    "AuditAlert",
    # Types
    "FeatureType",
    "Int64",
    "Float32",
    "Float64",
    "Bool",
    "String",
    "Bytes",
    "Timestamp",
    "Embedding",
    "Array",
    "Struct",
    # Validation
    "ValidationResult",
    "ValidationError",
    "ValidationWarning",
    # Analytics
    "Analysis",
    "Aggregation",
    "Distribution",
    "Correlation",
    "VersionDiff",
    "StatTest",
    "RecordLookup",
    "Freshness",
    "AnalysisResult",
    "AnalysisJob",
    "LiveTable",
    "RefreshPolicy",
    "Dashboard",
    "Chart",
    "ChartType",
    "DashboardParameter",
    "ParameterType",
    "AnalyticsAlert",
    "Condition",
    # Transforms
    "Source",
    "ObjectStorage",
    "FileSystem",
    "ColumnarSource",
    "FeatureGroupSource",
    "Schedule",
    "Transform",
    "SQLTransform",
    "PythonTransform",
    "sql_transform",
    "python_transform",
    "Job",
    "Target",
    "IncrementalConfig",
    "AirflowConfig",
    "generate_airflow_dag",
    "QualityCheck",
    "NullCheck",
    "UniqueCheck",
    "RangeCheck",
    "RowCountCheck",
    # Exceptions
    "RaiseError",
    "FeatureExistsError",
    "FeatureNotFoundError",
    "FeatureValidationError",
    "AccessDeniedError",
    "CrossOrgAccessError",
]
