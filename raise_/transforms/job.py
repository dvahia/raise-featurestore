"""
Raise Transform Jobs

Job definitions that combine sources, transforms, targets, and schedules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import uuid

from raise_.transforms.source import Source
from raise_.transforms.transform import Transform, TransformContext
from raise_.transforms.schedule import Schedule, ManualSchedule
from raise_.transforms.checkpoint import IncrementalConfig, Checkpoint, ProcessingMode


class JobStatus(Enum):
    """Job status enumeration."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    FAILED = "failed"
    DEPRECATED = "deprecated"


class RunStatus(Enum):
    """Run status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


@dataclass
class Target:
    """
    Target definition for where transformed data is written.

    Attributes:
        feature_group: Target feature group path
        features: Mapping of output columns to feature names
        write_mode: How to write data (append, overwrite, upsert)
    """

    feature_group: str
    features: dict[str, str] | None = None  # {output_col: feature_name}
    write_mode: str = "append"  # append, overwrite, upsert
    key_columns: list[str] = field(default_factory=list)  # For upsert

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_group": self.feature_group,
            "features": self.features,
            "write_mode": self.write_mode,
            "key_columns": self.key_columns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Target:
        return cls(
            feature_group=data.get("feature_group", ""),
            features=data.get("features"),
            write_mode=data.get("write_mode", "append"),
            key_columns=data.get("key_columns", []),
        )


@dataclass
class JobRun:
    """
    Record of a single job execution.

    Attributes:
        id: Unique run ID
        job_id: Parent job ID
        status: Current run status
        execution_date: Logical execution date
        started_at: When run started
        completed_at: When run completed
        rows_read: Number of rows read from sources
        rows_written: Number of rows written to target
        error: Error message if failed
        metrics: Additional run metrics
    """

    id: str
    job_id: str
    status: RunStatus = RunStatus.PENDING
    execution_date: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    rows_read: int = 0
    rows_written: int = 0
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    checkpoint_before: Any = None
    checkpoint_after: Any = None

    @property
    def duration_seconds(self) -> float | None:
        """Calculate run duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "status": self.status.value,
            "execution_date": self.execution_date.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "rows_read": self.rows_read,
            "rows_written": self.rows_written,
            "error": self.error,
            "metrics": self.metrics,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class Job:
    """
    A transformation job definition.

    Jobs combine sources, transforms, targets, and schedules into
    a complete data pipeline that produces features.

    Attributes:
        name: Unique job name
        sources: Input data sources
        transform: Transformation logic
        target: Output target (feature group)
        schedule: When to run
        incremental: Incremental processing configuration
        description: Job description
        owner: Job owner
        tags: Job tags
        retries: Number of retries on failure
        retry_delay: Delay between retries
        timeout: Maximum execution time
        alerts: Alert recipients on failure
    """

    name: str
    sources: list[Source] = field(default_factory=list)
    transform: Transform | None = None
    target: Target | None = None
    schedule: Schedule = field(default_factory=ManualSchedule)
    incremental: IncrementalConfig = field(default_factory=IncrementalConfig)

    # Metadata
    description: str | None = None
    owner: str | None = None
    tags: list[str] = field(default_factory=list)

    # Execution config
    retries: int = 3
    retry_delay: str = "5m"
    timeout: str = "1h"
    alerts: list[str] = field(default_factory=list)

    # Internal state
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = JobStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_run: JobRun | None = None
    next_run: datetime | None = None

    # Runtime (not serialized)
    _runs: list[JobRun] = field(default_factory=list, repr=False)
    _checkpoint: Checkpoint | None = field(default=None, repr=False)
    _client: Any = field(default=None, repr=False)

    def __post_init__(self):
        if not self.sources:
            self.sources = []

    # =========================================================================
    # Job Configuration
    # =========================================================================

    def add_source(self, source: Source) -> Job:
        """Add a source to the job."""
        self.sources.append(source)
        self.updated_at = datetime.now()
        return self

    def set_transform(self, transform: Transform) -> Job:
        """Set the transformation logic."""
        self.transform = transform
        self.updated_at = datetime.now()
        return self

    def set_target(self, target: Target | str) -> Job:
        """Set the target feature group."""
        if isinstance(target, str):
            target = Target(feature_group=target)
        self.target = target
        self.updated_at = datetime.now()
        return self

    def set_schedule(self, schedule: Schedule) -> Job:
        """Set the schedule."""
        self.schedule = schedule
        self.updated_at = datetime.now()
        return self

    def set_incremental(self, config: IncrementalConfig) -> Job:
        """Set incremental processing configuration."""
        self.incremental = config
        self.updated_at = datetime.now()
        return self

    # =========================================================================
    # Job Lifecycle
    # =========================================================================

    def activate(self) -> Job:
        """Activate the job for scheduled runs."""
        self._validate()
        self.status = JobStatus.ACTIVE
        self.updated_at = datetime.now()
        return self

    def pause(self) -> Job:
        """Pause the job."""
        self.status = JobStatus.PAUSED
        self.updated_at = datetime.now()
        return self

    def resume(self) -> Job:
        """Resume a paused job."""
        if self.status == JobStatus.PAUSED:
            self.status = JobStatus.ACTIVE
            self.updated_at = datetime.now()
        return self

    def deprecate(self) -> Job:
        """Deprecate the job."""
        self.status = JobStatus.DEPRECATED
        self.updated_at = datetime.now()
        return self

    def _validate(self) -> None:
        """Validate job configuration."""
        errors = []

        if not self.sources:
            errors.append("Job must have at least one source")
        if not self.transform:
            errors.append("Job must have a transform")
        if not self.target:
            errors.append("Job must have a target")

        if self.incremental.mode != ProcessingMode.FULL:
            if not self.incremental.checkpoint_column:
                errors.append("Incremental jobs must specify checkpoint_column")

        if errors:
            raise ValueError(f"Invalid job configuration: {'; '.join(errors)}")

    # =========================================================================
    # Job Execution
    # =========================================================================

    def run(self, execution_date: datetime | None = None) -> JobRun:
        """
        Execute the job manually.

        Args:
            execution_date: Logical execution date (defaults to now)

        Returns:
            JobRun with execution results
        """
        execution_date = execution_date or datetime.now()

        run = JobRun(
            id=str(uuid.uuid4()),
            job_id=self.id,
            status=RunStatus.PENDING,
            execution_date=execution_date,
        )

        self._runs.append(run)
        run.started_at = datetime.now()
        run.status = RunStatus.RUNNING

        try:
            # Create execution context
            context = TransformContext(
                job_id=self.id,
                run_id=run.id,
                execution_date=execution_date,
                is_incremental=self.incremental.mode != ProcessingMode.FULL,
                checkpoint_value=self._checkpoint.value if self._checkpoint else None,
            )

            run.checkpoint_before = context.checkpoint_value

            # Execute transform (mock implementation)
            # In production, this would:
            # 1. Read from sources
            # 2. Apply transformation
            # 3. Write to target
            # 4. Update checkpoint

            run.rows_read = 1000  # Mock
            run.rows_written = 1000  # Mock
            run.metrics = context.metrics

            # Update checkpoint
            if self._checkpoint:
                run.checkpoint_after = execution_date
                self._checkpoint.advance(execution_date)

            run.status = RunStatus.SUCCESS
            run.completed_at = datetime.now()

        except Exception as e:
            run.status = RunStatus.FAILED
            run.error = str(e)
            run.completed_at = datetime.now()

        self.last_run = run
        self.updated_at = datetime.now()
        return run

    def reset_checkpoint(self) -> Job:
        """Reset checkpoint to trigger full refresh on next run."""
        if self._checkpoint:
            self._checkpoint.value = None
            self._checkpoint.updated_at = datetime.now()
        return self

    # =========================================================================
    # Run History
    # =========================================================================

    def get_runs(
        self,
        status: RunStatus | None = None,
        limit: int = 100,
    ) -> list[JobRun]:
        """Get job run history."""
        runs = self._runs

        if status:
            runs = [r for r in runs if r.status == status]

        return sorted(runs, key=lambda r: r.execution_date, reverse=True)[:limit]

    def get_run(self, run_id: str) -> JobRun | None:
        """Get a specific run by ID."""
        for run in self._runs:
            if run.id == run_id:
                return run
        return None

    # =========================================================================
    # Lineage
    # =========================================================================

    def get_source_lineage(self) -> list[dict[str, Any]]:
        """Get lineage information for sources."""
        lineage = []
        for source in self.sources:
            lineage.append({
                "type": "source",
                "source_type": source.source_type.value,
                "source": source.to_dict(),
            })
        return lineage

    def get_target_lineage(self) -> dict[str, Any] | None:
        """Get lineage information for target."""
        if self.target:
            return {
                "type": "target",
                "feature_group": self.target.feature_group,
                "features": self.target.features,
            }
        return None

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Serialize job to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "owner": self.owner,
            "tags": self.tags,
            "sources": [s.to_dict() for s in self.sources],
            "transform": self.transform.to_dict() if self.transform else None,
            "target": self.target.to_dict() if self.target else None,
            "schedule": self.schedule.to_dict(),
            "incremental": self.incremental.to_dict(),
            "retries": self.retries,
            "retry_delay": self.retry_delay,
            "timeout": self.timeout,
            "alerts": self.alerts,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_run": self.last_run.to_dict() if self.last_run else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Job:
        """Deserialize job from dictionary."""
        from raise_.transforms.source import Source
        from raise_.transforms.transform import Transform
        from raise_.transforms.schedule import Schedule

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            description=data.get("description"),
            owner=data.get("owner"),
            tags=data.get("tags", []),
            sources=[Source.from_dict(s) for s in data.get("sources", [])],
            transform=Transform.from_dict(data["transform"]) if data.get("transform") else None,
            target=Target.from_dict(data["target"]) if data.get("target") else None,
            schedule=Schedule.from_dict(data["schedule"]) if data.get("schedule") else ManualSchedule(),
            incremental=IncrementalConfig.from_dict(data["incremental"]) if data.get("incremental") else IncrementalConfig(),
            retries=data.get("retries", 3),
            retry_delay=data.get("retry_delay", "5m"),
            timeout=data.get("timeout", "1h"),
            alerts=data.get("alerts", []),
            status=JobStatus(data.get("status", "draft")),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(),
        )

    def __repr__(self) -> str:
        return (
            f"Job(name={self.name!r}, status={self.status.value}, "
            f"sources={len(self.sources)}, schedule={self.schedule})"
        )
