"""
Raise Dataset Curation Transforms

Transforms for data curation: quality scoring, near-deduplication, and
compliance filtering. These compose naturally with the existing Transform
and Job APIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from raise_.models.dataset import (
    QualityDimension,
    QualityThreshold,
    DeduplicationAlgorithm,
    DeduplicationConfig,
    CompliancePolicy,
    ComplianceFlag,
    ComplianceAction,
    ComplianceRule,
)
from raise_.transforms.transform import Transform, TransformType


# =============================================================================
# Quality Scoring
# =============================================================================

@dataclass
class QualityScorer(Transform):
    """
    Transform that computes quality scores for text or multimodal data.

    Each configured dimension produces one output feature column. A composite
    score column (weighted average) is also written unless disabled.

    Attributes:
        name: Transform name.
        dimensions: Quality dimensions to score.
        thresholds: Per-dimension minimum score thresholds; records below any
                    threshold are filtered.
        model_uri: URI of the scoring model. If None, heuristic scorers are used.
        input_columns: Columns to pass to the scorer (e.g., ["text", "image_ref"]).
        composite_column: Output column for the weighted composite score.
        write_per_dimension: Whether to write individual dimension scores.
        filter_below_threshold: If True, filter records that fail any threshold.

    Example:
        >>> scorer = QualityScorer(
        ...     name="text_quality",
        ...     dimensions=[QualityDimension.FLUENCY, QualityDimension.SAFETY],
        ...     thresholds=[
        ...         QualityThreshold(QualityDimension.FLUENCY, min_score=0.6),
        ...         QualityThreshold(QualityDimension.SAFETY, min_score=0.8),
        ...     ],
        ...     input_columns=["text"],
        ... )
    """
    name: str = "quality_scorer"
    dimensions: list[QualityDimension] = field(default_factory=lambda: [QualityDimension.FLUENCY])
    thresholds: list[QualityThreshold] = field(default_factory=list)
    model_uri: str | None = None
    input_columns: list[str] = field(default_factory=lambda: ["text"])
    composite_column: str = "quality_score"
    write_per_dimension: bool = True
    filter_below_threshold: bool = True
    description: str | None = None

    # Transform base fields
    transform_type: TransformType = field(default=TransformType.PYTHON, init=False)

    @property
    def output_columns(self) -> list[str]:
        """All columns this scorer writes."""
        cols = [self.composite_column]
        if self.write_per_dimension:
            for dim in self.dimensions:
                threshold = self._threshold_for(dim)
                col = threshold.output_column if threshold else f"quality_{dim.value}"
                cols.append(col)
        return cols

    def _threshold_for(self, dim: QualityDimension) -> QualityThreshold | None:
        for t in self.thresholds:
            if t.dimension == dim:
                return t
        return None

    def with_threshold(self, dimension: QualityDimension, min_score: float) -> "QualityScorer":
        """Return a new scorer with an added/updated threshold."""
        existing = [t for t in self.thresholds if t.dimension != dimension]
        return QualityScorer(
            name=self.name,
            dimensions=self.dimensions,
            thresholds=existing + [QualityThreshold(dimension, min_score)],
            model_uri=self.model_uri,
            input_columns=self.input_columns,
            composite_column=self.composite_column,
            write_per_dimension=self.write_per_dimension,
            filter_below_threshold=self.filter_below_threshold,
        )


# =============================================================================
# Deduplication
# =============================================================================

@dataclass
class DeduplicationTransform(Transform):
    """
    Transform that detects and handles near-duplicate records.

    Produces an `is_duplicate` boolean column and a `duplicate_of` string
    column (entity key of the canonical record). Depending on config, duplicates
    are either removed or retained with these columns populated.

    Attributes:
        name: Transform name.
        config: Deduplication algorithm and threshold settings.
        is_duplicate_column: Output column name for the duplicate flag.
        duplicate_of_column: Output column for the canonical record key.
        entity_key_column: Column containing the entity key for cross-referencing.

    Example:
        >>> dedup = DeduplicationTransform(
        ...     name="web_dedup",
        ...     config=DeduplicationConfig(
        ...         algorithm=DeduplicationAlgorithm.MINHASH,
        ...         threshold=0.8,
        ...         key_columns=["text"],
        ...     ),
        ... )
    """
    name: str = "deduplication"
    config: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    is_duplicate_column: str = "is_duplicate"
    duplicate_of_column: str = "duplicate_of"
    entity_key_column: str = "id"
    description: str | None = None

    transform_type: TransformType = field(default=TransformType.PYTHON, init=False)

    @property
    def output_columns(self) -> list[str]:
        return [self.is_duplicate_column, self.duplicate_of_column]


# =============================================================================
# Compliance Filtering
# =============================================================================

@dataclass
class ComplianceFilterTransform(Transform):
    """
    Transform that applies a CompliancePolicy to flag or filter records.

    For each rule in the policy, a score column is written. Depending on
    the rule action (filter, flag, redact), records may be removed or
    annotated. Redaction requires a redactor model or function.

    Attributes:
        name: Transform name.
        policy: The compliance policy to enforce.
        input_columns: Columns to pass to compliance classifiers.
        passed_column: Output boolean column indicating the record passed all rules.
        redactor: Optional callable for in-place redaction (for REDACT actions).

    Example:
        >>> compliance = ComplianceFilterTransform(
        ...     name="pretraining_compliance",
        ...     policy=CompliancePolicy(
        ...         name="standard",
        ...         rules=[
        ...             ComplianceRule(ComplianceFlag.NSFW),
        ...             ComplianceRule(ComplianceFlag.PII, action=ComplianceAction.REDACT),
        ...         ],
        ...     ),
        ...     input_columns=["text"],
        ... )
    """
    name: str = "compliance_filter"
    policy: CompliancePolicy = field(default_factory=lambda: CompliancePolicy(name="default"))
    input_columns: list[str] = field(default_factory=lambda: ["text"])
    passed_column: str = "compliance_passed"
    redactor: Callable[[str, ComplianceFlag], str] | None = field(default=None, repr=False)
    description: str | None = None

    transform_type: TransformType = field(default=TransformType.PYTHON, init=False)

    @property
    def output_columns(self) -> list[str]:
        cols = [self.passed_column]
        for rule in self.policy.rules:
            cols.append(rule.output_column or f"compliance_{rule.flag.value}")
        return cols


# =============================================================================
# Curation Pipeline
# =============================================================================

@dataclass
class CurationPipeline:
    """
    An ordered sequence of curation transforms applied to a dataset.

    Pipelines compose QualityScorer, DeduplicationTransform, and
    ComplianceFilterTransform steps into a single named workflow.
    The pipeline is declarative: define the steps, then execute via a Job.

    Attributes:
        name: Pipeline name.
        steps: Ordered list of curation transforms.
        stop_on_first_filter: If True, a record is dropped as soon as any
                              step marks it for removal (faster but less annotative).
        description: Human-readable description.

    Example:
        >>> pipeline = CurationPipeline(
        ...     name="pretraining_curation",
        ...     steps=[
        ...         QualityScorer(dimensions=[QualityDimension.FLUENCY]),
        ...         DeduplicationTransform(config=dedup_config),
        ...         ComplianceFilterTransform(policy=compliance_policy),
        ...     ],
        ... )
    """
    name: str
    steps: list[Transform] = field(default_factory=list)
    stop_on_first_filter: bool = False
    description: str | None = None
    tags: list[str] = field(default_factory=list)

    def add_step(self, transform: Transform) -> "CurationPipeline":
        """Append a step and return self for chaining."""
        self.steps.append(transform)
        return self

    def quality_scorer(self) -> QualityScorer | None:
        """Return the first QualityScorer step, if any."""
        return next((s for s in self.steps if isinstance(s, QualityScorer)), None)

    def deduplication(self) -> DeduplicationTransform | None:
        """Return the first DeduplicationTransform step, if any."""
        return next((s for s in self.steps if isinstance(s, DeduplicationTransform)), None)

    def compliance(self) -> ComplianceFilterTransform | None:
        """Return the first ComplianceFilterTransform step, if any."""
        return next((s for s in self.steps if isinstance(s, ComplianceFilterTransform)), None)

    @property
    def all_output_columns(self) -> list[str]:
        """Collect all output columns written by this pipeline."""
        cols = []
        for step in self.steps:
            if hasattr(step, "output_columns"):
                cols.extend(step.output_columns)
        return cols

    def __repr__(self) -> str:
        step_names = [s.name for s in self.steps]
        return f"CurationPipeline({self.name!r}, steps={step_names})"
