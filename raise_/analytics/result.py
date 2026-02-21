"""
Raise Analysis Results

Persisted results from analysis executions.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from raise_.analytics.analysis import Analysis


@dataclass
class AnalysisResult:
    """
    Result of an analysis execution.

    Results are automatically persisted and can be exported
    to various formats.

    Attributes:
        id: Unique result identifier.
        analysis: The analysis definition that produced this result.
        data: The result data.
        created_at: When the result was computed.
        freshness_at: Timestamp of the underlying data.
        metadata: Additional metadata about the execution.
    """

    id: str
    analysis: Analysis
    data: dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    freshness_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Analysis-specific result accessors
    @property
    def metrics(self) -> dict[str, Any]:
        """Get computed metrics (for Aggregation results)."""
        return self.data.get("metrics", {})

    @property
    def histogram(self) -> dict[str, Any]:
        """Get histogram data (for Distribution results)."""
        return self.data.get("histogram", {})

    @property
    def percentiles(self) -> dict[str, float]:
        """Get percentile values (for Distribution results)."""
        return self.data.get("percentiles", {})

    @property
    def correlation_matrix(self) -> list[list[float]]:
        """Get correlation matrix (for Correlation results)."""
        return self.data.get("correlation_matrix", [])

    @property
    def schema_changes(self) -> dict[str, Any]:
        """Get schema changes (for VersionDiff results)."""
        return self.data.get("schema_changes", {})

    @property
    def distribution_drift(self) -> dict[str, float]:
        """Get drift metrics (for VersionDiff results)."""
        return self.data.get("distribution_drift", {})

    @property
    def p_value(self) -> float | None:
        """Get p-value (for StatTest results)."""
        return self.data.get("p_value")

    @property
    def confidence_interval(self) -> tuple[float, float] | None:
        """Get confidence interval (for StatTest results)."""
        ci = self.data.get("confidence_interval")
        if ci:
            return tuple(ci)
        return None

    @property
    def effect_size(self) -> float | None:
        """Get effect size (for StatTest results)."""
        return self.data.get("effect_size")

    @property
    def records(self) -> list[dict[str, Any]]:
        """Get records (for RecordLookup results)."""
        return self.data.get("records", [])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over records (for RecordLookup results)."""
        return iter(self.records)

    def to_dataframe(self) -> Any:
        """
        Convert result to pandas DataFrame.

        Returns:
            pandas.DataFrame with the result data.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install it with: pip install pandas"
            )

        analysis_type = self.analysis.analysis_type

        if analysis_type == "aggregation":
            return pd.DataFrame([self.metrics])
        elif analysis_type == "distribution":
            if "histogram" in self.data:
                return pd.DataFrame({
                    "bin_edges": self.data["histogram"].get("bin_edges", []),
                    "counts": self.data["histogram"].get("counts", []),
                })
            return pd.DataFrame([self.data])
        elif analysis_type == "correlation":
            features = self.data.get("features", [])
            matrix = self.correlation_matrix
            return pd.DataFrame(matrix, index=features, columns=features)
        elif analysis_type == "record_lookup":
            return pd.DataFrame(self.records)
        elif analysis_type == "stat_test":
            return pd.DataFrame([{
                "p_value": self.p_value,
                "effect_size": self.effect_size,
                "ci_lower": self.confidence_interval[0] if self.confidence_interval else None,
                "ci_upper": self.confidence_interval[1] if self.confidence_interval else None,
            }])
        else:
            return pd.DataFrame([self.data])

    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    def to_csv(self) -> str:
        """
        Convert result to CSV string.

        Returns:
            CSV formatted string.

        Raises:
            ImportError: If pandas is not installed.
        """
        df = self.to_dataframe()
        return df.to_csv(index=False)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "analysis": self.analysis.to_dict(),
            "data": self.data,
            "created_at": self.created_at.isoformat(),
            "freshness_at": self.freshness_at.isoformat() if self.freshness_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict, analysis: Analysis) -> AnalysisResult:
        """Create an AnalysisResult from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.now()

        freshness_at = data.get("freshness_at")
        if isinstance(freshness_at, str):
            freshness_at = datetime.fromisoformat(freshness_at)

        return cls(
            id=data["id"],
            analysis=analysis,
            data=data["data"],
            created_at=created_at,
            freshness_at=freshness_at,
            metadata=data.get("metadata", {}),
        )


JobStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


@dataclass
class AnalysisJob:
    """
    An asynchronous analysis job.

    Used for long-running analyses that are executed in the background.

    Attributes:
        id: Unique job identifier.
        analysis: The analysis being executed.
        status: Current job status.
        created_at: When the job was created.
        started_at: When execution started.
        completed_at: When execution completed.
        error: Error message if failed.
        result_id: ID of the result (when completed).
    """

    id: str
    analysis: Analysis
    status: JobStatus = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    result_id: str | None = None
    progress: float = 0.0  # 0.0 to 1.0

    # Internal reference to client for fetching result
    _client: Any = field(default=None, repr=False)

    def refresh(self) -> AnalysisJob:
        """
        Refresh job status from the server.

        Returns:
            Self with updated status.
        """
        if self._client:
            updated = self._client.get_job(self.id)
            self.status = updated.status
            self.started_at = updated.started_at
            self.completed_at = updated.completed_at
            self.error = updated.error
            self.result_id = updated.result_id
            self.progress = updated.progress
        return self

    def wait(self, timeout: float = 300, poll_interval: float = 1.0) -> AnalysisResult:
        """
        Wait for the job to complete.

        Args:
            timeout: Maximum time to wait in seconds.
            poll_interval: Time between status checks.

        Returns:
            The analysis result.

        Raises:
            TimeoutError: If job doesn't complete within timeout.
            RuntimeError: If job fails.
        """
        start_time = time.time()

        while self.status in ("pending", "running"):
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Job {self.id} did not complete within {timeout}s")

            time.sleep(poll_interval)
            self.refresh()

        if self.status == "failed":
            raise RuntimeError(f"Job {self.id} failed: {self.error}")

        if self.status == "cancelled":
            raise RuntimeError(f"Job {self.id} was cancelled")

        return self.result()

    def result(self) -> AnalysisResult:
        """
        Get the job result.

        Returns:
            The analysis result.

        Raises:
            RuntimeError: If job is not completed.
        """
        if self.status != "completed":
            raise RuntimeError(f"Job {self.id} is not completed (status: {self.status})")

        if self._client and self.result_id:
            return self._client.get_result(self.result_id)

        raise RuntimeError(f"No result available for job {self.id}")

    def cancel(self) -> bool:
        """
        Cancel the job.

        Returns:
            True if cancellation was successful.
        """
        if self._client:
            return self._client.cancel_job(self.id)
        return False

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "analysis": self.analysis.to_dict(),
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "result_id": self.result_id,
            "progress": self.progress,
        }


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    uid = uuid.uuid4().hex[:12]
    if prefix:
        return f"{prefix}_{uid}"
    return uid
