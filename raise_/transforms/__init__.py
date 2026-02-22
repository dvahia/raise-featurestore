"""
Raise Transforms Module

ETL and data transformation capabilities for the Feature Store.
"""

from raise_.transforms.source import (
    Source,
    SourceType,
    FileFormat,
    ObjectStorage,
    FileSystem,
    ColumnarSource,
    FeatureGroupSource,
    DatabaseSource,
    SourceRegistry,
)

from raise_.transforms.schedule import (
    Schedule,
    ScheduleType,
    CronSchedule,
    IntervalSchedule,
    OnChangeSchedule,
    ManualSchedule,
    OnceSchedule,
)

from raise_.transforms.checkpoint import (
    Checkpoint,
    CheckpointType,
    CheckpointStore,
    IncrementalConfig,
    ProcessingMode,
)

from raise_.transforms.transform import (
    Transform,
    TransformType,
    TransformContext,
    SQLTransform,
    PythonTransform,
    HybridTransform,
    sql_transform,
    python_transform,
)

from raise_.transforms.job import (
    Job,
    JobStatus,
    JobRun,
    RunStatus,
    Target,
)

from raise_.transforms.observability import (
    LogLevel,
    LogEntry,
    Metric,
    MetricType,
    QualityCheck,
    QualityCheckResult,
    CheckResult,
    CheckSeverity,
    NullCheck,
    UniqueCheck,
    RangeCheck,
    RowCountCheck,
    CustomCheck,
    FreshnessCheck,
    BlobIntegrityCheck,
    QualityReport,
    JobMetrics,
    StandardMetrics,
)

from raise_.transforms.orchestrator import (
    Orchestrator,
    OrchestratorType,
    DeploymentResult,
    JobOrchestratorStatus,
    InternalOrchestrator,
)

from raise_.transforms.airflow import (
    AirflowOrchestrator,
    AirflowConfig,
    generate_airflow_dag,
)

from raise_.transforms.client import TransformsClient

from raise_.transforms.multimodal import (
    # Core types
    BlobReference,
    BlobRegistry,
    BlobStatus,
    ContentType,
    HashAlgorithm,
    # Integrity
    IntegrityMode,
    IntegrityPolicy,
    IntegrityError,
    ValidationResult,
    ReferenceNotFoundError,
    # Sources
    MultimodalSource,
    MultimodalContext,
    BlobReferenceType,
    # Utilities
    create_reference,
    infer_content_type,
)

from raise_.transforms.inference import (
    # Enums
    ModelFramework,
    AcceleratorType,
    GPUType,
    TPUType,
    InferenceMode,
    ModelPrecision,
    # Core classes
    ModelSpec,
    AcceleratorConfig,
    BatchConfig,
    InferenceRuntime,
    InferenceTransform,
    InferenceResult,
    # Decorator
    inference_transform,
    # Convenience constructors
    embedding_inference,
    classification_inference,
    image_inference,
    llm_inference,
)


__all__ = [
    # Sources
    "Source",
    "SourceType",
    "FileFormat",
    "ObjectStorage",
    "FileSystem",
    "ColumnarSource",
    "FeatureGroupSource",
    "DatabaseSource",
    "SourceRegistry",
    # Schedules
    "Schedule",
    "ScheduleType",
    "CronSchedule",
    "IntervalSchedule",
    "OnChangeSchedule",
    "ManualSchedule",
    "OnceSchedule",
    # Checkpoints
    "Checkpoint",
    "CheckpointType",
    "CheckpointStore",
    "IncrementalConfig",
    "ProcessingMode",
    # Transforms
    "Transform",
    "TransformType",
    "TransformContext",
    "SQLTransform",
    "PythonTransform",
    "HybridTransform",
    "sql_transform",
    "python_transform",
    # Jobs
    "Job",
    "JobStatus",
    "JobRun",
    "RunStatus",
    "Target",
    # Observability
    "LogLevel",
    "LogEntry",
    "Metric",
    "MetricType",
    "QualityCheck",
    "QualityCheckResult",
    "CheckResult",
    "CheckSeverity",
    "NullCheck",
    "UniqueCheck",
    "RangeCheck",
    "RowCountCheck",
    "CustomCheck",
    "FreshnessCheck",
    "BlobIntegrityCheck",
    "QualityReport",
    "JobMetrics",
    "StandardMetrics",
    # Orchestrators
    "Orchestrator",
    "OrchestratorType",
    "DeploymentResult",
    "JobOrchestratorStatus",
    "InternalOrchestrator",
    "AirflowOrchestrator",
    "AirflowConfig",
    "generate_airflow_dag",
    # Client
    "TransformsClient",
    # Multimodal
    "BlobReference",
    "BlobRegistry",
    "BlobStatus",
    "ContentType",
    "HashAlgorithm",
    "IntegrityMode",
    "IntegrityPolicy",
    "IntegrityError",
    "ValidationResult",
    "ReferenceNotFoundError",
    "MultimodalSource",
    "MultimodalContext",
    "BlobReferenceType",
    "create_reference",
    "infer_content_type",
    # Inference
    "ModelFramework",
    "AcceleratorType",
    "GPUType",
    "TPUType",
    "InferenceMode",
    "ModelPrecision",
    "ModelSpec",
    "AcceleratorConfig",
    "BatchConfig",
    "InferenceRuntime",
    "InferenceTransform",
    "InferenceResult",
    "inference_transform",
    "embedding_inference",
    "classification_inference",
    "image_inference",
    "llm_inference",
]
