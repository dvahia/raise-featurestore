"""
Raise Dataset Models

Data models for dataset versioning, mixing, curation configuration, and provenance.
Designed for managing large-scale training datasets for pre- and post-training workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal


# =============================================================================
# Enums
# =============================================================================

class MixingStrategy(Enum):
    """How to combine samples from multiple dataset sources."""
    UNIFORM = "uniform"               # Equal probability across all sources
    WEIGHTED = "weighted"             # Proportional to declared weights
    TEMPERATURE = "temperature"       # Weights raised to 1/T; smooths imbalance
    PROPORTIONAL = "proportional"     # Proportional to source dataset size
    ROUND_ROBIN = "round_robin"       # Strict interleaving, one sample per source


class DeduplicationAlgorithm(Enum):
    """Algorithm used to detect near-duplicate records."""
    EXACT_HASH = "exact_hash"                    # Exact byte-level match
    MINHASH = "minhash"                          # Locality-sensitive hashing (text)
    SIMHASH = "simhash"                          # Bit-level similarity
    EMBEDDING_SIMILARITY = "embedding_similarity" # Vector cosine distance
    BLOOM_FILTER = "bloom_filter"                # Probabilistic membership test


class ComplianceFlag(Enum):
    """Categories of compliance concern."""
    PII = "pii"                   # Personally identifiable information
    NSFW = "nsfw"                 # Not safe for work content
    COPYRIGHT = "copyright"       # Potentially copyrighted material
    HATE_SPEECH = "hate_speech"   # Hate speech or harassment
    TOXIC = "toxic"               # General toxicity
    MEDICAL = "medical"           # Sensitive medical information
    LEGAL = "legal"               # Legal privilege or sensitive legal content
    CHILD_SAFETY = "child_safety" # CSAM or related


class ComplianceAction(Enum):
    """What to do when a compliance flag is triggered."""
    FILTER = "filter"   # Remove the record from the dataset
    FLAG = "flag"       # Keep but mark with a compliance flag feature
    REDACT = "redact"   # Attempt in-place redaction (e.g., PII masking)


class QualityDimension(Enum):
    """Dimensions along which data quality is assessed."""
    FLUENCY = "fluency"           # Grammatical and linguistic quality
    FACTUALITY = "factuality"     # Factual accuracy and grounding
    SAFETY = "safety"             # Absence of harmful content
    DIVERSITY = "diversity"       # Lexical and semantic diversity
    RELEVANCE = "relevance"       # Relevance to a target domain or task
    COMPLETENESS = "completeness" # Completeness of information
    COHERENCE = "coherence"       # Logical coherence and consistency
    FORMAT = "format"             # Structural correctness (JSON, code, etc.)


# =============================================================================
# Dataset Provenance
# =============================================================================

@dataclass
class DatasetProvenance:
    """
    Tracks the origin and lineage of a dataset.

    Attributes:
        source_name: Human-readable name of the data source.
        source_uri: Location of the raw data (s3://, huggingface://, etc.).
        collection_date: When the data was collected or licensed.
        license: SPDX identifier or description of the license.
        citation: BibTeX or plain-text citation for the source.
        language_codes: ISO 639-1 language codes present in the data.
        domain_tags: High-level domain labels (e.g., "web", "books", "code").
        pii_reviewed: Whether a PII review has been completed.
        consent_documented: Whether user consent is documented.
    """
    source_name: str
    source_uri: str | None = None
    collection_date: datetime | None = None
    license: str | None = None
    citation: str | None = None
    language_codes: list[str] = field(default_factory=list)
    domain_tags: list[str] = field(default_factory=list)
    pii_reviewed: bool = False
    consent_documented: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Dataset Version
# =============================================================================

@dataclass
class DatasetVersion:
    """
    A versioned snapshot of a feature group used as a training dataset.

    Captures the full state needed to reproduce a dataset: which feature
    group, which filters were applied, how many records survived, and what
    transformations were run before this snapshot was taken.

    Attributes:
        name: Human-readable name for this version.
        version: Semantic or hash version string (e.g., "v1.2.0").
        feature_group: Path to the source feature group.
        created_at: When this snapshot was created.
        created_by: User or system that created the snapshot.
        num_records: Number of records in this version.
        size_bytes: Total size of data in bytes.
        applied_filters: Human-readable description of filters applied.
        parent_version: Version string of the version this was derived from.
        provenance: Source and lineage information.
        tags: Arbitrary tags for discoverability.
        metadata: Additional version metadata.

    Example:
        >>> version = DatasetVersion(
        ...     name="web-crawl-deduped",
        ...     version="v2.1.0",
        ...     feature_group="acme/pretraining/web-crawl",
        ...     num_records=50_000_000,
        ...     size_bytes=400 * 1024**3,
        ...     applied_filters=["quality_score >= 0.7", "is_duplicate == False"],
        ... )
    """
    name: str
    version: str
    feature_group: str
    num_records: int = 0
    size_bytes: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    applied_filters: list[str] = field(default_factory=list)
    parent_version: str | None = None
    provenance: DatasetProvenance | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_initial(self) -> bool:
        """True if this is the first version (no parent)."""
        return self.parent_version is None

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)

    def derive(
        self,
        version: str,
        applied_filters: list[str],
        num_records: int = 0,
        size_bytes: int = 0,
        **kwargs,
    ) -> "DatasetVersion":
        """Create a derived version with this version as parent."""
        return DatasetVersion(
            name=self.name,
            version=version,
            feature_group=self.feature_group,
            num_records=num_records,
            size_bytes=size_bytes,
            applied_filters=applied_filters,
            parent_version=self.version,
            provenance=self.provenance,
            tags=self.tags,
            **kwargs,
        )

    def __repr__(self) -> str:
        return f"DatasetVersion({self.name}@{self.version}, {self.num_records:,} records)"


# =============================================================================
# Dataset Mixing
# =============================================================================

@dataclass
class MixSource:
    """
    A single source in a dataset mix with its weight and optional filters.

    Attributes:
        feature_group: Path to the feature group to draw samples from.
        weight: Relative weight for this source (unnormalized).
        filters: SQL-style filter expressions applied before sampling.
        max_samples: Hard cap on samples drawn from this source.
        version: Feature group version to use (None = latest).
    """
    feature_group: str
    weight: float = 1.0
    filters: list[str] = field(default_factory=list)
    max_samples: int | None = None
    version: str | None = None
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.weight <= 0:
            raise ValueError(f"weight must be positive, got {self.weight}")


@dataclass
class DatasetMix:
    """
    Defines how multiple dataset sources are combined into one training dataset.

    Mixing is a first-class concept in pre-training data preparation. This
    class captures the full configuration so that mixes are reproducible and
    auditable.

    Attributes:
        name: Human-readable name for this mix configuration.
        sources: List of source datasets with weights and optional filters.
        strategy: How to sample across sources.
        total_samples: Total samples in the output (None = sum of sources).
        temperature: Temperature for TEMPERATURE strategy (default 1.0 = weighted).
        seed: Random seed for reproducibility.

    Examples:
        >>> mix = DatasetMix(
        ...     name="pretraining-v3",
        ...     sources=[
        ...         MixSource("acme/pretraining/web-crawl", weight=0.7),
        ...         MixSource("acme/pretraining/books", weight=0.15),
        ...         MixSource("acme/pretraining/code", weight=0.15),
        ...     ],
        ...     strategy=MixingStrategy.WEIGHTED,
        ...     total_samples=2_000_000_000,
        ... )
    """
    name: str
    sources: list[MixSource]
    strategy: MixingStrategy = MixingStrategy.WEIGHTED
    total_samples: int | None = None
    temperature: float = 1.0
    seed: int = 42
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.sources:
            raise ValueError("DatasetMix must have at least one source")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")

    def normalized_weights(self) -> list[float]:
        """Return weights normalized to sum to 1.0."""
        raw = [s.weight for s in self.sources]
        total = sum(raw)
        return [w / total for w in raw]

    def effective_weights(self) -> list[float]:
        """
        Return effective sampling probabilities for the configured strategy.

        For TEMPERATURE strategy, applies temperature scaling before normalization.
        """
        if self.strategy == MixingStrategy.TEMPERATURE:
            raw = [s.weight ** (1.0 / self.temperature) for s in self.sources]
        elif self.strategy == MixingStrategy.UNIFORM:
            raw = [1.0] * len(self.sources)
        else:
            raw = [s.weight for s in self.sources]
        total = sum(raw)
        return [w / total for w in raw]

    def summary(self) -> dict[str, Any]:
        weights = self.effective_weights()
        return {
            "name": self.name,
            "strategy": self.strategy.value,
            "total_samples": self.total_samples,
            "sources": [
                {
                    "feature_group": s.feature_group,
                    "weight": s.weight,
                    "effective_prob": f"{w:.1%}",
                }
                for s, w in zip(self.sources, weights)
            ],
        }

    def __repr__(self) -> str:
        return f"DatasetMix({self.name!r}, {len(self.sources)} sources, {self.strategy.value})"


# =============================================================================
# Deduplication
# =============================================================================

@dataclass
class DeduplicationConfig:
    """
    Configuration for near-duplicate detection.

    Attributes:
        algorithm: The deduplication algorithm to use.
        threshold: Similarity threshold above which two records are considered
                   duplicates (0.0 = any match, 1.0 = exact match only).
        key_columns: Columns used to compute the similarity hash.
        embedding_column: Column storing pre-computed embeddings (for
                          EMBEDDING_SIMILARITY algorithm).
        num_perm: Number of permutations for MinHash (higher = more accurate).
        ngram_size: N-gram size for text hashing algorithms.
        action: What to do with detected duplicates.
        keep: Which duplicate to keep ("first", "last", or "highest_quality").
        quality_column: Column used to select the kept record when keep="highest_quality".

    Example:
        >>> dedup_config = DeduplicationConfig(
        ...     algorithm=DeduplicationAlgorithm.MINHASH,
        ...     threshold=0.85,
        ...     key_columns=["text"],
        ...     action="remove",
        ... )
    """
    algorithm: DeduplicationAlgorithm = DeduplicationAlgorithm.MINHASH
    threshold: float = 0.85
    key_columns: list[str] = field(default_factory=lambda: ["text"])
    embedding_column: str | None = None
    num_perm: int = 128
    ngram_size: int = 5
    action: Literal["remove", "flag"] = "remove"
    keep: Literal["first", "last", "highest_quality"] = "first"
    quality_column: str | None = None

    def __post_init__(self):
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        if self.keep == "highest_quality" and self.quality_column is None:
            raise ValueError("quality_column is required when keep='highest_quality'")


# =============================================================================
# Compliance Filtering
# =============================================================================

@dataclass
class ComplianceRule:
    """
    A single compliance rule targeting one category of concern.

    Attributes:
        flag: The compliance category to check.
        action: What to do when this flag is triggered.
        threshold: Confidence threshold for triggering (0.0–1.0).
        model_uri: Optional URI to the classifier model for this flag.
        output_column: Column to write the flag probability or label into.
    """
    flag: ComplianceFlag
    action: ComplianceAction = ComplianceAction.FILTER
    threshold: float = 0.5
    model_uri: str | None = None
    output_column: str | None = None

    def __post_init__(self):
        if self.output_column is None:
            self.output_column = f"compliance_{self.flag.value}"


@dataclass
class CompliancePolicy:
    """
    A collection of compliance rules applied as a curation step.

    Attributes:
        name: Policy name.
        rules: Individual compliance rules to enforce.
        require_all_pass: If True, a record is retained only if all rules pass.
                          If False, the strictest triggered action wins.
        annotate_only: If True, never filter/redact; only write annotation columns.

    Example:
        >>> policy = CompliancePolicy(
        ...     name="pretraining-compliance",
        ...     rules=[
        ...         ComplianceRule(ComplianceFlag.PII, action=ComplianceAction.REDACT),
        ...         ComplianceRule(ComplianceFlag.NSFW, threshold=0.7),
        ...         ComplianceRule(ComplianceFlag.COPYRIGHT, threshold=0.9),
        ...     ],
        ... )
    """
    name: str
    rules: list[ComplianceRule] = field(default_factory=list)
    require_all_pass: bool = True
    annotate_only: bool = False
    description: str | None = None

    def add_rule(
        self,
        flag: ComplianceFlag,
        action: ComplianceAction = ComplianceAction.FILTER,
        threshold: float = 0.5,
        model_uri: str | None = None,
    ) -> "CompliancePolicy":
        """Add a rule and return self for chaining."""
        self.rules.append(ComplianceRule(
            flag=flag,
            action=action,
            threshold=threshold,
            model_uri=model_uri,
        ))
        return self

    @property
    def flagged_categories(self) -> list[ComplianceFlag]:
        return [r.flag for r in self.rules]


# =============================================================================
# Quality Filtering
# =============================================================================

@dataclass
class QualityThreshold:
    """
    Specifies minimum quality requirements for a dimension.

    Attributes:
        dimension: The quality dimension to threshold.
        min_score: Records below this score are filtered.
        output_column: Column storing the score for this dimension.
        weight: Relative weight when computing a composite score.
    """
    dimension: QualityDimension
    min_score: float
    output_column: str | None = None
    weight: float = 1.0

    def __post_init__(self):
        if self.output_column is None:
            self.output_column = f"quality_{self.dimension.value}"
