"""
Raise Transform Orchestrators

Base orchestrator interface and implementations for scheduling jobs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from raise_.transforms.job import Job


class OrchestratorType:
    """Orchestrator type constants."""
    AIRFLOW = "airflow"
    DAGSTER = "dagster"
    PREFECT = "prefect"
    INTERNAL = "internal"


@dataclass
class Orchestrator(ABC):
    """
    Base class for job orchestrators.

    Orchestrators are responsible for scheduling and executing jobs.
    """

    name: str = "orchestrator"

    @property
    @abstractmethod
    def orchestrator_type(self) -> str:
        """Return the orchestrator type."""
        pass

    @abstractmethod
    def deploy(self, job: Job) -> DeploymentResult:
        """Deploy a job to the orchestrator."""
        pass

    @abstractmethod
    def undeploy(self, job: Job) -> bool:
        """Remove a job from the orchestrator."""
        pass

    @abstractmethod
    def trigger(self, job: Job, execution_date: datetime | None = None) -> str:
        """Manually trigger a job run. Returns run ID."""
        pass

    @abstractmethod
    def get_status(self, job: Job) -> JobOrchestratorStatus:
        """Get the orchestrator status for a job."""
        pass

    @abstractmethod
    def generate_definition(self, job: Job) -> str:
        """Generate the orchestrator-specific definition (DAG, flow, etc.)."""
        pass


@dataclass
class DeploymentResult:
    """Result of deploying a job to an orchestrator."""

    success: bool
    job_id: str
    orchestrator_id: str | None = None  # ID in the orchestrator system
    message: str | None = None
    url: str | None = None  # URL to view in orchestrator UI

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "job_id": self.job_id,
            "orchestrator_id": self.orchestrator_id,
            "message": self.message,
            "url": self.url,
        }


@dataclass
class JobOrchestratorStatus:
    """Status of a job in an orchestrator."""

    deployed: bool
    enabled: bool = False
    last_run: datetime | None = None
    last_run_status: str | None = None
    next_run: datetime | None = None
    orchestrator_id: str | None = None
    orchestrator_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "deployed": self.deployed,
            "enabled": self.enabled,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "last_run_status": self.last_run_status,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "orchestrator_id": self.orchestrator_id,
            "orchestrator_url": self.orchestrator_url,
        }


@dataclass
class InternalOrchestrator(Orchestrator):
    """
    Internal orchestrator for simple scheduling.

    Uses in-process scheduling - suitable for development and testing.
    In production, this could use a background job queue.
    """

    name: str = "internal"
    _jobs: dict[str, Job] = field(default_factory=dict, repr=False)
    _runs: dict[str, list[str]] = field(default_factory=dict, repr=False)

    @property
    def orchestrator_type(self) -> str:
        return OrchestratorType.INTERNAL

    def deploy(self, job: Job) -> DeploymentResult:
        self._jobs[job.id] = job
        return DeploymentResult(
            success=True,
            job_id=job.id,
            orchestrator_id=job.id,
            message="Deployed to internal orchestrator",
        )

    def undeploy(self, job: Job) -> bool:
        if job.id in self._jobs:
            del self._jobs[job.id]
            return True
        return False

    def trigger(self, job: Job, execution_date: datetime | None = None) -> str:
        run = job.run(execution_date)
        if job.id not in self._runs:
            self._runs[job.id] = []
        self._runs[job.id].append(run.id)
        return run.id

    def get_status(self, job: Job) -> JobOrchestratorStatus:
        deployed = job.id in self._jobs
        return JobOrchestratorStatus(
            deployed=deployed,
            enabled=deployed and job.status.value == "active",
            last_run=job.last_run.execution_date if job.last_run else None,
            last_run_status=job.last_run.status.value if job.last_run else None,
        )

    def generate_definition(self, job: Job) -> str:
        """Generate a simple JSON definition."""
        import json
        return json.dumps(job.to_dict(), indent=2, default=str)
