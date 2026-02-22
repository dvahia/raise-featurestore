"""
Raise Transform Observability

Logging, metrics, and data quality checks for transformations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Metric type enumeration."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class CheckSeverity(Enum):
    """Quality check severity."""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class CheckResult(Enum):
    """Quality check result."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class LogEntry:
    """
    A log entry.

    Attributes:
        timestamp: When the log was created
        level: Log level
        message: Log message
        job_id: Associated job ID
        run_id: Associated run ID
        extra: Additional context
    """

    timestamp: datetime
    level: LogLevel
    message: str
    job_id: str | None = None
    run_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "job_id": self.job_id,
            "run_id": self.run_id,
            "extra": self.extra,
        }


@dataclass
class Metric:
    """
    A metric measurement.

    Attributes:
        name: Metric name
        value: Metric value
        metric_type: Type of metric
        labels: Metric labels
        timestamp: When measured
    """

    name: str
    value: float
    metric_type: MetricType = MetricType.GAUGE
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "labels": self.labels,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class QualityCheck(ABC):
    """
    Base class for data quality checks.

    Quality checks validate data during or after transformation.
    """

    name: str
    description: str | None = None
    severity: CheckSeverity = CheckSeverity.ERROR

    @abstractmethod
    def check(self, data: Any) -> QualityCheckResult:
        """Execute the quality check."""
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize check to dictionary."""
        pass


@dataclass
class QualityCheckResult:
    """
    Result of a quality check execution.

    Attributes:
        check_name: Name of the check
        result: Pass/fail/skip status
        message: Human-readable message
        actual_value: Actual measured value
        expected_value: Expected value or threshold
        details: Additional details
        timestamp: When check was executed
    """

    check_name: str
    result: CheckResult
    message: str
    actual_value: Any = None
    expected_value: Any = None
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def passed(self) -> bool:
        return self.result == CheckResult.PASSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_name": self.check_name,
            "result": self.result.value,
            "message": self.message,
            "actual_value": self.actual_value,
            "expected_value": self.expected_value,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class NullCheck(QualityCheck):
    """Check for null/missing values."""

    column: str = ""
    max_null_rate: float = 0.0  # 0-1

    def check(self, data: Any) -> QualityCheckResult:
        # Mock implementation - in production would check actual data
        null_rate = 0.01  # Mock value

        passed = null_rate <= self.max_null_rate

        return QualityCheckResult(
            check_name=self.name,
            result=CheckResult.PASSED if passed else CheckResult.FAILED,
            message=f"Null rate for {self.column}: {null_rate:.2%} (max: {self.max_null_rate:.2%})",
            actual_value=null_rate,
            expected_value=self.max_null_rate,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "null_check",
            "name": self.name,
            "description": self.description,
            "severity": self.severity.value,
            "column": self.column,
            "max_null_rate": self.max_null_rate,
        }


@dataclass
class UniqueCheck(QualityCheck):
    """Check for unique values."""

    columns: list[str] = field(default_factory=list)

    def check(self, data: Any) -> QualityCheckResult:
        # Mock implementation
        duplicate_count = 0

        passed = duplicate_count == 0

        return QualityCheckResult(
            check_name=self.name,
            result=CheckResult.PASSED if passed else CheckResult.FAILED,
            message=f"Uniqueness check on {self.columns}: {duplicate_count} duplicates",
            actual_value=duplicate_count,
            expected_value=0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "unique_check",
            "name": self.name,
            "description": self.description,
            "severity": self.severity.value,
            "columns": self.columns,
        }


@dataclass
class RangeCheck(QualityCheck):
    """Check that values fall within a range."""

    column: str = ""
    min_value: float | None = None
    max_value: float | None = None

    def check(self, data: Any) -> QualityCheckResult:
        # Mock implementation
        out_of_range = 0

        passed = out_of_range == 0

        return QualityCheckResult(
            check_name=self.name,
            result=CheckResult.PASSED if passed else CheckResult.FAILED,
            message=f"Range check on {self.column}: {out_of_range} out of range",
            actual_value=out_of_range,
            expected_value=0,
            details={"min": self.min_value, "max": self.max_value},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "range_check",
            "name": self.name,
            "description": self.description,
            "severity": self.severity.value,
            "column": self.column,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }


@dataclass
class RowCountCheck(QualityCheck):
    """Check row count meets expectations."""

    min_rows: int | None = None
    max_rows: int | None = None
    expected_rows: int | None = None
    tolerance: float = 0.0  # For expected_rows, allow +/- tolerance

    def check(self, data: Any) -> QualityCheckResult:
        # Mock implementation
        row_count = 1000

        passed = True
        if self.min_rows is not None and row_count < self.min_rows:
            passed = False
        if self.max_rows is not None and row_count > self.max_rows:
            passed = False
        if self.expected_rows is not None:
            tolerance_range = self.expected_rows * self.tolerance
            if abs(row_count - self.expected_rows) > tolerance_range:
                passed = False

        return QualityCheckResult(
            check_name=self.name,
            result=CheckResult.PASSED if passed else CheckResult.FAILED,
            message=f"Row count: {row_count}",
            actual_value=row_count,
            expected_value=self.expected_rows or f"[{self.min_rows}, {self.max_rows}]",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "row_count_check",
            "name": self.name,
            "description": self.description,
            "severity": self.severity.value,
            "min_rows": self.min_rows,
            "max_rows": self.max_rows,
            "expected_rows": self.expected_rows,
            "tolerance": self.tolerance,
        }


@dataclass
class CustomCheck(QualityCheck):
    """Custom quality check using a function."""

    check_function: Callable[[Any], bool] | None = None
    function_name: str | None = None

    def check(self, data: Any) -> QualityCheckResult:
        if self.check_function is None:
            return QualityCheckResult(
                check_name=self.name,
                result=CheckResult.SKIPPED,
                message="Check function not defined",
            )

        try:
            passed = self.check_function(data)
            return QualityCheckResult(
                check_name=self.name,
                result=CheckResult.PASSED if passed else CheckResult.FAILED,
                message=f"Custom check: {'passed' if passed else 'failed'}",
            )
        except Exception as e:
            return QualityCheckResult(
                check_name=self.name,
                result=CheckResult.FAILED,
                message=f"Check raised exception: {e}",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "custom_check",
            "name": self.name,
            "description": self.description,
            "severity": self.severity.value,
            "function_name": self.function_name,
        }


@dataclass
class FreshnessCheck(QualityCheck):
    """Check data freshness based on a timestamp column."""

    column: str = ""
    max_age: str = "1h"  # Maximum age of newest record

    def check(self, data: Any) -> QualityCheckResult:
        # Mock implementation
        max_timestamp = datetime.now()
        age_seconds = 0

        passed = True  # Mock

        return QualityCheckResult(
            check_name=self.name,
            result=CheckResult.PASSED if passed else CheckResult.FAILED,
            message=f"Data freshness: newest record is {age_seconds}s old",
            actual_value=age_seconds,
            expected_value=self.max_age,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "freshness_check",
            "name": self.name,
            "description": self.description,
            "severity": self.severity.value,
            "column": self.column,
            "max_age": self.max_age,
        }


@dataclass
class QualityReport:
    """
    Aggregated report of quality check results.

    Attributes:
        job_id: Associated job ID
        run_id: Associated run ID
        checks: List of check results
        timestamp: When report was generated
    """

    job_id: str
    run_id: str
    checks: list[QualityCheckResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def passed(self) -> bool:
        """Whether all checks passed."""
        return all(c.passed for c in self.checks)

    @property
    def passed_count(self) -> int:
        """Number of passed checks."""
        return sum(1 for c in self.checks if c.result == CheckResult.PASSED)

    @property
    def failed_count(self) -> int:
        """Number of failed checks."""
        return sum(1 for c in self.checks if c.result == CheckResult.FAILED)

    @property
    def skipped_count(self) -> int:
        """Number of skipped checks."""
        return sum(1 for c in self.checks if c.result == CheckResult.SKIPPED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "passed": self.passed,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "skipped_count": self.skipped_count,
            "checks": [c.to_dict() for c in self.checks],
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class JobMetrics:
    """
    Collected metrics for a job run.

    Standard metrics collected during execution.
    """

    job_id: str
    run_id: str
    metrics: list[Metric] = field(default_factory=list)

    def record(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a metric."""
        self.metrics.append(Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            labels=labels or {},
        ))

    def record_counter(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Record a counter metric."""
        self.record(name, value, MetricType.COUNTER, labels)

    def record_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Record a gauge metric."""
        self.record(name, value, MetricType.GAUGE, labels)

    def record_histogram(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Record a histogram metric."""
        self.record(name, value, MetricType.HISTOGRAM, labels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "metrics": [m.to_dict() for m in self.metrics],
        }


# Standard metric names
class StandardMetrics:
    """Standard metric names for consistency."""

    ROWS_READ = "transform_rows_read"
    ROWS_WRITTEN = "transform_rows_written"
    ROWS_FILTERED = "transform_rows_filtered"
    DURATION_SECONDS = "transform_duration_seconds"
    SOURCE_READ_SECONDS = "transform_source_read_seconds"
    TRANSFORM_SECONDS = "transform_transform_seconds"
    SINK_WRITE_SECONDS = "transform_sink_write_seconds"
    BYTES_READ = "transform_bytes_read"
    BYTES_WRITTEN = "transform_bytes_written"
    CHECKPOINT_LAG_SECONDS = "transform_checkpoint_lag_seconds"
    QUALITY_CHECKS_PASSED = "transform_quality_checks_passed"
    QUALITY_CHECKS_FAILED = "transform_quality_checks_failed"
