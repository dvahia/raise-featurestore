"""
Raise Annotation and Evaluation Models

Data models for human annotation workflows, evaluation suites, and result
collection. Supports RLHF preference collection, model evaluation, and
human-in-the-loop quality assurance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal


# =============================================================================
# Enums
# =============================================================================

class AnnotationTaskType(Enum):
    """The type of judgment humans are asked to make."""
    BINARY_PREFERENCE = "binary_preference"  # A vs B, pick one
    RATING = "rating"                        # Score on a numeric scale
    LIKERT = "likert"                        # Ordered categorical (e.g., 1-5)
    RANKING = "ranking"                      # Order a list of candidates
    OPEN_ENDED = "open_ended"                # Free-text annotation
    BINARY_LABEL = "binary_label"            # Yes/No for a single item
    MULTI_LABEL = "multi_label"              # Select all that apply


class AgreementMetric(Enum):
    """Inter-annotator agreement metrics."""
    COHENS_KAPPA = "cohens_kappa"
    KRIPPENDORFFS_ALPHA = "krippendorffs_alpha"
    PERCENT_AGREEMENT = "percent_agreement"
    FLEISS_KAPPA = "fleiss_kappa"
    SPEARMANS_RHO = "spearmans_rho"


class EvalMetric(Enum):
    """Metrics for automated and human evaluation."""
    # Automated
    ACCURACY = "accuracy"
    F1 = "f1"
    BLEU = "bleu"
    ROUGE_L = "rouge_l"
    BERTSCORE = "bertscore"
    PERPLEXITY = "perplexity"
    # Preference-based
    WIN_RATE = "win_rate"
    ELO_SCORE = "elo_score"
    BRADLEY_TERRY = "bradley_terry"
    # Human
    MEAN_OPINION_SCORE = "mean_opinion_score"
    PREFERENCE_RATE = "preference_rate"


class AnnotatorPoolType(Enum):
    """Where annotators come from."""
    INTERNAL = "internal"          # Internal ML team
    CROWD = "crowd"                # Crowdsourcing (MTurk, etc.)
    EXPERT = "expert"              # Domain experts
    MODEL = "model"                # LLM-as-judge annotation
    HYBRID = "hybrid"              # Mix of model + human verification


# =============================================================================
# Annotation Task
# =============================================================================

@dataclass
class AnnotatorConfig:
    """
    Configuration for the annotator pool for a task.

    Attributes:
        pool_type: Where annotators come from.
        min_annotations_per_item: How many independent annotations per item.
        annotator_qualifications: Required skills or certifications.
        languages: Language proficiency requirements (ISO 639-1 codes).
        region_restrictions: Geographical restrictions if any.
        model_judge_uri: Model URI for LLM-as-judge tasks.
        model_judge_prompt_template: Prompt template for model judges.
    """
    pool_type: AnnotatorPoolType = AnnotatorPoolType.INTERNAL
    min_annotations_per_item: int = 3
    annotator_qualifications: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=lambda: ["en"])
    region_restrictions: list[str] = field(default_factory=list)
    model_judge_uri: str | None = None
    model_judge_prompt_template: str | None = None


@dataclass
class HumanEvalTask:
    """
    Definition of a human annotation or evaluation task.

    A HumanEvalTask specifies what human annotators (or model judges) will
    evaluate, what judgment they will make, and how that judgment is stored
    back in the feature store.

    Attributes:
        name: Unique task name.
        task_type: The kind of judgment being collected.
        instructions: Instructions shown to annotators.
        source_feature_group: Feature group containing items to annotate.
        source_columns: Which feature columns annotators will see.
        label_column: Feature column where collected labels are stored.
        options: Valid label options for categorical tasks.
        rating_scale: (min, max) for RATING tasks.
        min_annotations: Minimum annotations required per item before aggregation.
        annotator_config: Configuration for the annotator pool.
        agreement_metric: Metric used to measure annotator agreement.
        consensus_strategy: How to aggregate multiple annotations.
        example_items: Gold-standard examples for annotator calibration.

    Example:
        >>> task = HumanEvalTask(
        ...     name="response-quality-preference",
        ...     task_type=AnnotationTaskType.BINARY_PREFERENCE,
        ...     instructions="Select the response that is more helpful and harmless.",
        ...     source_feature_group="acme/posttraining/preference-candidates",
        ...     source_columns=["prompt", "response_a", "response_b"],
        ...     label_column="preferred_response",
        ...     options=["response_a", "response_b", "tie"],
        ...     min_annotations=5,
        ... )
    """
    name: str
    task_type: AnnotationTaskType
    instructions: str
    source_feature_group: str
    source_columns: list[str]
    label_column: str
    options: list[str] | None = None
    rating_scale: tuple[float, float] | None = None
    min_annotations: int = 3
    annotator_config: AnnotatorConfig = field(default_factory=AnnotatorConfig)
    agreement_metric: AgreementMetric = AgreementMetric.COHENS_KAPPA
    consensus_strategy: Literal["majority", "mean", "weighted_mean", "unanimous"] = "majority"
    example_items: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.task_type == AnnotationTaskType.RATING and self.rating_scale is None:
            self.rating_scale = (1.0, 5.0)
        if self.task_type in (
            AnnotationTaskType.BINARY_PREFERENCE,
            AnnotationTaskType.BINARY_LABEL,
            AnnotationTaskType.MULTI_LABEL,
            AnnotationTaskType.LIKERT,
        ) and self.options is None:
            raise ValueError(f"options must be provided for task_type={self.task_type.value}")


# =============================================================================
# Annotation Results
# =============================================================================

@dataclass
class AnnotationResult:
    """
    A single annotation from one annotator for one item.

    Attributes:
        task_name: Name of the HumanEvalTask this result belongs to.
        item_id: The entity key identifying the item being annotated.
        annotator_id: Identifier for the annotator (may be anonymized).
        label: The annotation value (preference choice, rating, label, etc.).
        confidence: Annotator-reported confidence (0.0–1.0), if collected.
        duration_sec: Time spent on this annotation.
        timestamp: When the annotation was submitted.
        notes: Optional free-text from the annotator.
        is_model_judge: True if this annotation came from an LLM judge.
        model_judge_reasoning: Model judge chain-of-thought (if applicable).
    """
    task_name: str
    item_id: str
    annotator_id: str
    label: Any
    confidence: float | None = None
    duration_sec: float | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    notes: str | None = None
    is_model_judge: bool = False
    model_judge_reasoning: str | None = None


@dataclass
class AnnotationBatch:
    """
    A batch of items sent to annotators in one round.

    Batches are the unit of work dispatched to annotation platforms or
    internal annotators. Results are collected back into the feature store
    via a job.

    Attributes:
        batch_id: Unique batch identifier.
        task_name: Name of the HumanEvalTask this batch belongs to.
        item_ids: Entity keys of items in this batch.
        status: Current batch status.
        dispatched_at: When the batch was sent to annotators.
        due_at: Deadline for annotation completion.
        num_completed: How many items have collected sufficient annotations.
    """
    batch_id: str
    task_name: str
    item_ids: list[str]
    status: Literal["pending", "in_progress", "complete", "expired"] = "pending"
    dispatched_at: datetime | None = None
    due_at: datetime | None = None
    num_completed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_items(self) -> int:
        return len(self.item_ids)

    @property
    def completion_rate(self) -> float:
        if not self.item_ids:
            return 0.0
        return self.num_completed / len(self.item_ids)


@dataclass
class AgreementScore:
    """Inter-annotator agreement measurement for a task."""
    task_name: str
    metric: AgreementMetric
    score: float
    num_annotators: int
    num_items: int
    computed_at: datetime = field(default_factory=datetime.utcnow)
    by_category: dict[str, float] = field(default_factory=dict)

    @property
    def is_acceptable(self) -> bool:
        """Rough heuristic: kappa >= 0.6 is considered substantial agreement."""
        if self.metric in (AgreementMetric.COHENS_KAPPA, AgreementMetric.FLEISS_KAPPA,
                           AgreementMetric.KRIPPENDORFFS_ALPHA):
            return self.score >= 0.6
        if self.metric == AgreementMetric.PERCENT_AGREEMENT:
            return self.score >= 0.8
        return True


# =============================================================================
# Evaluation Suites
# =============================================================================

@dataclass
class EvalExample:
    """
    A single example in an evaluation suite.

    Attributes:
        example_id: Unique identifier for this example.
        input: Input features (prompt, image refs, audio refs, etc.).
        ground_truth: Expected output for auto-eval (None for human-only evals).
        difficulty: Difficulty tag (e.g., "easy", "hard", "adversarial").
        domain: Domain or capability being tested.
        metadata: Additional example-level metadata.
    """
    example_id: str
    input: dict[str, Any]
    ground_truth: dict[str, Any] | None = None
    difficulty: str | None = None
    domain: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSuite:
    """
    A versioned collection of evaluation examples for benchmarking.

    EvalSuites serve as a stable benchmark across model iterations. Each
    suite is stored as a feature group so that results can be tracked over
    time.

    Attributes:
        name: Suite name.
        description: What capability or domain this suite evaluates.
        version: Version string (bump when examples change).
        examples: The evaluation examples.
        metrics: Metrics to compute on results.
        human_eval_task: Associated HumanEvalTask for human evaluation.
        feature_group: Path to the feature group storing this suite.
        tags: Discoverability tags.

    Example:
        >>> suite = EvalSuite(
        ...     name="multimodal-reasoning",
        ...     description="Tests visual reasoning and grounding",
        ...     version="v1.0",
        ...     metrics=[EvalMetric.ACCURACY, EvalMetric.WIN_RATE],
        ... )
        >>> suite.add_example(EvalExample(
        ...     example_id="vr_001",
        ...     input={"image_ref": ref, "question": "How many objects?"},
        ...     ground_truth={"answer": "3"},
        ... ))
    """
    name: str
    description: str | None = None
    version: str = "v1.0"
    examples: list[EvalExample] = field(default_factory=list)
    metrics: list[EvalMetric] = field(default_factory=list)
    human_eval_task: HumanEvalTask | None = None
    feature_group: str | None = None
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_example(self, example: EvalExample) -> None:
        """Add an example to the suite."""
        self.examples.append(example)

    def sample(self, n: int, seed: int = 42) -> "EvalSuite":
        """Return a new EvalSuite with a random subset of examples."""
        import random
        rng = random.Random(seed)
        sampled = rng.sample(self.examples, min(n, len(self.examples)))
        return EvalSuite(
            name=self.name,
            description=self.description,
            version=self.version,
            examples=sampled,
            metrics=self.metrics,
            human_eval_task=self.human_eval_task,
            feature_group=self.feature_group,
            tags=self.tags,
            metadata={**self.metadata, "sampled_from": len(self.examples), "sample_seed": seed},
        )

    def filter_by_domain(self, domain: str) -> "EvalSuite":
        """Return a new EvalSuite with only examples from the given domain."""
        return EvalSuite(
            name=self.name,
            description=self.description,
            version=self.version,
            examples=[e for e in self.examples if e.domain == domain],
            metrics=self.metrics,
            feature_group=self.feature_group,
            tags=self.tags,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __repr__(self) -> str:
        return f"EvalSuite({self.name!r}@{self.version}, {len(self.examples)} examples)"


@dataclass
class EvalResult:
    """
    Aggregated result of running a model against an EvalSuite.

    Attributes:
        suite_name: Name of the EvalSuite evaluated.
        suite_version: Version of the suite.
        model_id: Identifier of the model being evaluated.
        scores: Metric name → score value.
        num_evaluated: Number of examples evaluated.
        num_failed: Examples where the model failed to produce output.
        human_win_rate: Fraction of human preferences for this model (if applicable).
        agreement_score: Inter-annotator agreement if human eval was run.
        evaluated_at: When the evaluation was run.
    """
    suite_name: str
    suite_version: str
    model_id: str
    scores: dict[str, float] = field(default_factory=dict)
    num_evaluated: int = 0
    num_failed: int = 0
    human_win_rate: float | None = None
    agreement_score: AgreementScore | None = None
    evaluated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if self.num_evaluated == 0:
            return 0.0
        return (self.num_evaluated - self.num_failed) / self.num_evaluated

    def __repr__(self) -> str:
        score_str = ", ".join(f"{k}={v:.3f}" for k, v in self.scores.items())
        return f"EvalResult({self.model_id} on {self.suite_name}@{self.suite_version}: {score_str})"
