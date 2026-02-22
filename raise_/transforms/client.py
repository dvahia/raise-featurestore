"""
Raise Transforms Client

Main client for managing transform jobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TYPE_CHECKING

from raise_.transforms.source import Source, ObjectStorage, FileSystem, ColumnarSource, FeatureGroupSource
from raise_.transforms.transform import Transform, SQLTransform, PythonTransform, HybridTransform, TransformContext
from raise_.transforms.schedule import Schedule
from raise_.transforms.checkpoint import IncrementalConfig, Checkpoint, CheckpointStore
from raise_.transforms.job import Job, JobStatus, Target
from raise_.transforms.orchestrator import Orchestrator, InternalOrchestrator, DeploymentResult
from raise_.transforms.airflow import AirflowOrchestrator, AirflowConfig
from raise_.transforms.observability import QualityCheck, QualityReport


@dataclass
class TransformsClient:
    """
    Client for managing transform jobs.

    Provides methods for creating, deploying, and monitoring
    transformation jobs that populate features.

    Attributes:
        feature_store: Parent FeatureStore instance
        orchestrator: Job orchestrator (Airflow, internal, etc.)
    """

    _feature_store: Any = field(repr=False)
    orchestrator: Orchestrator = field(default_factory=InternalOrchestrator)

    # Internal state
    _jobs: dict[str, Job] = field(default_factory=dict, repr=False)
    _checkpoints: CheckpointStore = field(default_factory=CheckpointStore, repr=False)

    # =========================================================================
    # Job Management
    # =========================================================================

    def create_job(
        self,
        name: str,
        sources: list[Source] | Source | None = None,
        transform: Transform | None = None,
        target: Target | str | None = None,
        schedule: Schedule | None = None,
        incremental: IncrementalConfig | None = None,
        description: str | None = None,
        owner: str | None = None,
        tags: list[str] | None = None,
        **kwargs,
    ) -> Job:
        """
        Create a new transformation job.

        Args:
            name: Unique job name
            sources: Input data source(s)
            transform: Transformation logic
            target: Output feature group
            schedule: Execution schedule
            incremental: Incremental processing config
            description: Job description
            owner: Job owner
            tags: Job tags

        Returns:
            The created Job
        """
        # Normalize sources to list
        if sources is None:
            sources = []
        elif isinstance(sources, Source):
            sources = [sources]

        # Normalize target
        if isinstance(target, str):
            target = Target(feature_group=target)

        job = Job(
            name=name,
            sources=sources,
            transform=transform,
            target=target,
            schedule=schedule or Schedule.manual(),
            incremental=incremental or IncrementalConfig.full(),
            description=description,
            owner=owner,
            tags=tags or [],
            _client=self,
            **kwargs,
        )

        # Initialize checkpoint for incremental jobs
        if incremental and incremental.mode.value != "full":
            checkpoint = Checkpoint(
                job_id=job.id,
                checkpoint_type=incremental.checkpoint_type,
                column=incremental.checkpoint_column,
            )
            self._checkpoints.save(checkpoint)
            job._checkpoint = checkpoint

        self._jobs[job.name] = job
        return job

    def get_job(self, name: str) -> Job:
        """Get a job by name."""
        if name not in self._jobs:
            raise ValueError(f"Job not found: {name}")
        return self._jobs[name]

    def list_jobs(
        self,
        status: JobStatus | str | None = None,
        tags: list[str] | None = None,
        owner: str | None = None,
    ) -> list[Job]:
        """List jobs with optional filters."""
        jobs = list(self._jobs.values())

        if status:
            if isinstance(status, str):
                status = JobStatus(status)
            jobs = [j for j in jobs if j.status == status]

        if tags:
            jobs = [j for j in jobs if any(t in j.tags for t in tags)]

        if owner:
            jobs = [j for j in jobs if j.owner == owner]

        return sorted(jobs, key=lambda j: j.name)

    def delete_job(self, name: str) -> bool:
        """Delete a job."""
        if name in self._jobs:
            job = self._jobs[name]
            # Undeploy from orchestrator
            if job.status == JobStatus.ACTIVE:
                self.orchestrator.undeploy(job)
            # Remove checkpoint
            self._checkpoints.delete(job.id)
            del self._jobs[name]
            return True
        return False

    # =========================================================================
    # Job Deployment
    # =========================================================================

    def deploy(self, job: Job | str) -> DeploymentResult:
        """
        Deploy a job to the orchestrator.

        Args:
            job: Job instance or job name

        Returns:
            DeploymentResult with status
        """
        if isinstance(job, str):
            job = self.get_job(job)

        # Validate and activate
        job.activate()

        # Deploy to orchestrator
        result = self.orchestrator.deploy(job)

        return result

    def undeploy(self, job: Job | str) -> bool:
        """Remove a job from the orchestrator."""
        if isinstance(job, str):
            job = self.get_job(job)

        job.pause()
        return self.orchestrator.undeploy(job)

    def trigger(
        self,
        job: Job | str,
        execution_date: datetime | None = None,
    ) -> str:
        """
        Manually trigger a job run.

        Args:
            job: Job instance or job name
            execution_date: Logical execution date

        Returns:
            Run ID
        """
        if isinstance(job, str):
            job = self.get_job(job)

        return self.orchestrator.trigger(job, execution_date)

    # =========================================================================
    # Orchestrator Management
    # =========================================================================

    def use_airflow(
        self,
        config: AirflowConfig | None = None,
        airflow_url: str = "http://localhost:8080",
    ) -> TransformsClient:
        """
        Configure Airflow as the orchestrator.

        Args:
            config: Airflow configuration
            airflow_url: Airflow web UI URL

        Returns:
            Self for chaining
        """
        self.orchestrator = AirflowOrchestrator(
            config=config or AirflowConfig(),
            airflow_url=airflow_url,
        )
        return self

    def use_internal(self) -> TransformsClient:
        """Use the internal orchestrator (for development/testing)."""
        self.orchestrator = InternalOrchestrator()
        return self

    def generate_dag(self, job: Job | str) -> str:
        """
        Generate the orchestrator definition for a job.

        For Airflow, this returns the DAG Python code.

        Args:
            job: Job instance or job name

        Returns:
            Orchestrator-specific definition string
        """
        if isinstance(job, str):
            job = self.get_job(job)

        return self.orchestrator.generate_definition(job)

    # =========================================================================
    # Convenience Builders
    # =========================================================================

    def sql_job(
        self,
        name: str,
        sql: str,
        source: Source,
        target: str,
        schedule: Schedule | None = None,
        incremental: IncrementalConfig | None = None,
        **kwargs,
    ) -> Job:
        """
        Create a SQL-based transformation job.

        Convenience method for common SQL transformations.

        Args:
            name: Job name
            sql: SQL transformation query
            source: Input source
            target: Target feature group path
            schedule: Execution schedule
            incremental: Incremental config

        Returns:
            The created Job
        """
        transform = SQLTransform(
            name=f"{name}_transform",
            sql=sql,
        )

        return self.create_job(
            name=name,
            sources=[source],
            transform=transform,
            target=target,
            schedule=schedule,
            incremental=incremental,
            **kwargs,
        )

    def python_job(
        self,
        name: str,
        function: Any,
        source: Source,
        target: str,
        schedule: Schedule | None = None,
        incremental: IncrementalConfig | None = None,
        **kwargs,
    ) -> Job:
        """
        Create a Python-based transformation job.

        Args:
            name: Job name
            function: Python transform function
            source: Input source
            target: Target feature group path
            schedule: Execution schedule
            incremental: Incremental config

        Returns:
            The created Job
        """
        transform = PythonTransform.from_function(
            function,
            name=f"{name}_transform",
        )

        return self.create_job(
            name=name,
            sources=[source],
            transform=transform,
            target=target,
            schedule=schedule,
            incremental=incremental,
            **kwargs,
        )

    # =========================================================================
    # Lineage
    # =========================================================================

    def get_job_lineage(self, job: Job | str) -> dict[str, Any]:
        """
        Get lineage information for a job.

        Returns sources and target feature relationships.
        """
        if isinstance(job, str):
            job = self.get_job(job)

        return {
            "job_id": job.id,
            "job_name": job.name,
            "sources": job.get_source_lineage(),
            "target": job.get_target_lineage(),
        }

    def get_feature_producers(self, feature_group: str) -> list[Job]:
        """
        Get all jobs that produce features for a feature group.

        Args:
            feature_group: Feature group path

        Returns:
            List of jobs that write to this feature group
        """
        return [
            job for job in self._jobs.values()
            if job.target and job.target.feature_group == feature_group
        ]

    def get_feature_consumers(self, feature_group: str) -> list[Job]:
        """
        Get all jobs that consume features from a feature group.

        Args:
            feature_group: Feature group path

        Returns:
            List of jobs that read from this feature group
        """
        consumers = []
        for job in self._jobs.values():
            for source in job.sources:
                if isinstance(source, FeatureGroupSource):
                    if source.feature_group == feature_group:
                        consumers.append(job)
                        break
        return consumers
