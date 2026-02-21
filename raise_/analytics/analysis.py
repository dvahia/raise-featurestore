"""
Raise Analysis Types

Defines the various types of analyses that can be performed on features.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


class Analysis(ABC):
    """Base class for all analysis types."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        pass

    @property
    @abstractmethod
    def analysis_type(self) -> str:
        """Return the analysis type identifier."""
        pass


@dataclass
class Aggregation(Analysis):
    """
    Aggregation analysis on feature data.

    Computes metrics like sum, avg, min, max, count, null_rate
    with optional time windows and grouping.

    Args:
        feature: Feature name to aggregate.
        metrics: List of metrics to compute.
        window: Time window (e.g., "7d", "1h", "30m").
        group_by: Feature to group results by.
        rolling: If True, compute rolling aggregates.
        periods: Number of periods for rolling aggregates.
        filter: SQL-like filter expression.

    Examples:
        >>> Aggregation(
        ...     feature="click_count",
        ...     metrics=["sum", "avg", "null_rate"],
        ...     window="7d",
        ...     group_by="user_tier",
        ... )
    """

    feature: str
    metrics: list[str] = field(default_factory=lambda: ["count", "avg"])
    window: str | None = None
    group_by: str | None = None
    rolling: bool = False
    periods: int | None = None
    filter: str | None = None

    # Supported metrics
    SUPPORTED_METRICS = {
        "count", "sum", "avg", "mean", "min", "max",
        "stddev", "variance", "median",
        "null_rate", "null_count", "distinct_count",
        "p50", "p75", "p90", "p95", "p99",
    }

    def __post_init__(self):
        for metric in self.metrics:
            if metric.lower() not in self.SUPPORTED_METRICS:
                raise ValueError(f"Unsupported metric: {metric}")
        if self.rolling and not self.periods:
            raise ValueError("rolling=True requires periods to be set")

    @property
    def analysis_type(self) -> str:
        return "aggregation"

    def to_dict(self) -> dict:
        return {
            "type": self.analysis_type,
            "feature": self.feature,
            "metrics": self.metrics,
            "window": self.window,
            "group_by": self.group_by,
            "rolling": self.rolling,
            "periods": self.periods,
            "filter": self.filter,
        }


@dataclass
class Distribution(Analysis):
    """
    Distribution analysis on feature data.

    Computes histograms, percentiles, and distribution statistics.

    Args:
        feature: Feature name to analyze.
        metrics: Distribution metrics to compute.
        bins: Number of histogram bins.
        segment_by: Feature to segment distributions by.
        sample_size: Sample size for large datasets.
        filter: SQL-like filter expression.

    Examples:
        >>> Distribution(
        ...     feature="user_embedding",
        ...     metrics=["histogram", "percentiles"],
        ...     bins=50,
        ... )
    """

    feature: str
    metrics: list[str] = field(default_factory=lambda: ["histogram", "percentiles"])
    bins: int = 50
    segment_by: str | None = None
    sample_size: int | None = None
    filter: str | None = None

    SUPPORTED_METRICS = {
        "histogram", "percentiles", "skewness", "kurtosis",
        "density", "cdf", "quantiles",
    }

    def __post_init__(self):
        for metric in self.metrics:
            if metric.lower() not in self.SUPPORTED_METRICS:
                raise ValueError(f"Unsupported distribution metric: {metric}")

    @property
    def analysis_type(self) -> str:
        return "distribution"

    def to_dict(self) -> dict:
        return {
            "type": self.analysis_type,
            "feature": self.feature,
            "metrics": self.metrics,
            "bins": self.bins,
            "segment_by": self.segment_by,
            "sample_size": self.sample_size,
            "filter": self.filter,
        }


@dataclass
class Correlation(Analysis):
    """
    Correlation analysis between multiple features.

    Computes correlation matrices and pairwise correlations.

    Args:
        features: List of feature names (can include paths for cross-group).
        method: Correlation method (pearson, spearman, kendall).
        window: Time window for the analysis.
        sample_size: Sample size for large datasets.
        filter: SQL-like filter expression.

    Examples:
        >>> Correlation(
        ...     features=["click_count", "impression_count", "revenue"],
        ...     method="pearson",
        ...     window="30d",
        ... )
    """

    features: list[str]
    method: Literal["pearson", "spearman", "kendall"] = "pearson"
    window: str | None = None
    sample_size: int | None = None
    filter: str | None = None

    def __post_init__(self):
        if len(self.features) < 2:
            raise ValueError("Correlation requires at least 2 features")

    @property
    def analysis_type(self) -> str:
        return "correlation"

    def to_dict(self) -> dict:
        return {
            "type": self.analysis_type,
            "features": self.features,
            "method": self.method,
            "window": self.window,
            "sample_size": self.sample_size,
            "filter": self.filter,
        }


@dataclass
class VersionDiff(Analysis):
    """
    Compare two versions of a feature.

    Compares schema changes, data distribution, and statistics.

    Args:
        feature: Feature name to compare.
        version_a: First version (e.g., "v1").
        version_b: Second version (e.g., "v2").
        compare: What to compare (schema, distribution, statistics).
        sample_size: Sample size for distribution comparison.

    Examples:
        >>> VersionDiff(
        ...     feature="user_embedding",
        ...     version_a="v1",
        ...     version_b="v2",
        ...     compare=["schema", "distribution"],
        ... )
    """

    feature: str
    version_a: str
    version_b: str
    compare: list[str] = field(default_factory=lambda: ["schema", "distribution", "statistics"])
    sample_size: int | None = None

    SUPPORTED_COMPARISONS = {"schema", "distribution", "statistics", "samples"}

    def __post_init__(self):
        for c in self.compare:
            if c.lower() not in self.SUPPORTED_COMPARISONS:
                raise ValueError(f"Unsupported comparison type: {c}")

    @property
    def analysis_type(self) -> str:
        return "version_diff"

    def to_dict(self) -> dict:
        return {
            "type": self.analysis_type,
            "feature": self.feature,
            "version_a": self.version_a,
            "version_b": self.version_b,
            "compare": self.compare,
            "sample_size": self.sample_size,
        }


@dataclass
class StatTest(Analysis):
    """
    Statistical hypothesis testing.

    Supports A/B tests, distribution comparisons, and significance testing.

    Args:
        feature: Feature name to test.
        test: Statistical test type.
        segment_by: Feature to segment by (for A/B tests).
        control: Control group value.
        treatment: Treatment group value.
        compare_versions: Tuple of versions to compare.
        confidence_level: Confidence level (default 0.95).
        filter: SQL-like filter expression.

    Examples:
        >>> StatTest(
        ...     feature="conversion_rate",
        ...     test="ttest",
        ...     segment_by="experiment_group",
        ...     control="control",
        ...     treatment="variant_a",
        ... )
    """

    feature: str
    test: Literal["ttest", "mannwhitney", "chi2", "ks", "anova", "welch"] = "ttest"
    segment_by: str | None = None
    control: str | None = None
    treatment: str | None = None
    compare_versions: tuple[str, str] | None = None
    confidence_level: float = 0.95
    filter: str | None = None

    def __post_init__(self):
        if self.segment_by and (not self.control or not self.treatment):
            raise ValueError("segment_by requires control and treatment values")
        if not self.segment_by and not self.compare_versions:
            raise ValueError("Either segment_by or compare_versions must be specified")

    @property
    def analysis_type(self) -> str:
        return "stat_test"

    def to_dict(self) -> dict:
        return {
            "type": self.analysis_type,
            "feature": self.feature,
            "test": self.test,
            "segment_by": self.segment_by,
            "control": self.control,
            "treatment": self.treatment,
            "compare_versions": self.compare_versions,
            "confidence_level": self.confidence_level,
            "filter": self.filter,
        }


@dataclass
class RecordLookup(Analysis):
    """
    Lookup and inspect specific records.

    Used for data inspection and debugging.

    Args:
        features: Features to include in the lookup.
        filter: SQL-like filter expression for specific records.
        sample: Number of random samples (alternative to filter).
        limit: Maximum records to return.
        order_by: Feature to order results by.
        descending: Sort descending if True.

    Examples:
        >>> RecordLookup(
        ...     features=["user_embedding", "click_count"],
        ...     filter="user_id = 'user_12345'",
        ...     limit=10,
        ... )

        >>> RecordLookup(
        ...     features=["user_embedding"],
        ...     sample=100,
        ...     filter="user_tier = 'gold'",
        ... )
    """

    features: list[str]
    filter: str | None = None
    sample: int | None = None
    limit: int = 100
    order_by: str | None = None
    descending: bool = False

    def __post_init__(self):
        if not self.features:
            raise ValueError("At least one feature must be specified")

    @property
    def analysis_type(self) -> str:
        return "record_lookup"

    def to_dict(self) -> dict:
        return {
            "type": self.analysis_type,
            "features": self.features,
            "filter": self.filter,
            "sample": self.sample,
            "limit": self.limit,
            "order_by": self.order_by,
            "descending": self.descending,
        }


@dataclass
class DataQuality(Analysis):
    """
    Data quality analysis on features.

    Computes completeness, validity, and quality metrics.

    Args:
        features: Features to analyze (or None for all in group).
        checks: Quality checks to perform.
        window: Time window for the analysis.

    Examples:
        >>> DataQuality(
        ...     features=["click_count", "user_embedding"],
        ...     checks=["completeness", "validity", "freshness"],
        ... )
    """

    features: list[str] | None = None
    checks: list[str] = field(default_factory=lambda: [
        "completeness", "validity", "uniqueness", "freshness"
    ])
    window: str | None = None

    SUPPORTED_CHECKS = {
        "completeness",   # Null rate, missing values
        "validity",       # Type conformance, range checks
        "uniqueness",     # Duplicate detection
        "freshness",      # Data recency
        "consistency",    # Cross-feature consistency
        "outliers",       # Outlier detection
    }

    def __post_init__(self):
        for check in self.checks:
            if check.lower() not in self.SUPPORTED_CHECKS:
                raise ValueError(f"Unsupported quality check: {check}")

    @property
    def analysis_type(self) -> str:
        return "data_quality"

    def to_dict(self) -> dict:
        return {
            "type": self.analysis_type,
            "features": self.features,
            "checks": self.checks,
            "window": self.window,
        }


@dataclass
class Drift(Analysis):
    """
    Data drift detection analysis.

    Compares current data distribution to a baseline.

    Args:
        feature: Feature to analyze for drift.
        baseline: Baseline specification (version, date range, or snapshot).
        metrics: Drift metrics to compute.
        window: Current data window.
        threshold: Drift threshold for alerts.

    Examples:
        >>> Drift(
        ...     feature="user_embedding",
        ...     baseline={"version": "v1"},
        ...     metrics=["psi", "kl_divergence"],
        ... )
    """

    feature: str
    baseline: dict[str, Any]  # {"version": "v1"} or {"date": "2025-01-01"} or {"snapshot": "id"}
    metrics: list[str] = field(default_factory=lambda: ["psi", "kl_divergence"])
    window: str | None = None
    threshold: float | None = None

    SUPPORTED_METRICS = {
        "psi",              # Population Stability Index
        "kl_divergence",    # KL Divergence
        "js_divergence",    # Jensen-Shannon Divergence
        "wasserstein",      # Wasserstein Distance
        "ks_statistic",     # Kolmogorov-Smirnov Statistic
    }

    def __post_init__(self):
        for metric in self.metrics:
            if metric.lower() not in self.SUPPORTED_METRICS:
                raise ValueError(f"Unsupported drift metric: {metric}")

    @property
    def analysis_type(self) -> str:
        return "drift"

    def to_dict(self) -> dict:
        return {
            "type": self.analysis_type,
            "feature": self.feature,
            "baseline": self.baseline,
            "metrics": self.metrics,
            "window": self.window,
            "threshold": self.threshold,
        }
