"""
Example 12: Dataset Curation — Quality Scoring, Deduplication, Compliance

Demonstrates the three core curation operations in the Raise Feature Store:
quality scoring, near-deduplication, and compliance filtering. Each step is
a Transform that runs as part of a Job and writes annotation columns back
into the feature group alongside the original data.

Key concepts:
- QualityScorer: multi-dimensional quality scoring with filter thresholds
- DeduplicationTransform: MinHash near-dedup that writes is_duplicate flag
- ComplianceFilterTransform: PII/NSFW/copyright detection with per-rule actions
- CurationPipeline: composing multiple curation steps into one named workflow
- Feature schemas for curation outputs (quality scores, flags)
"""

from raise_ import (
    FeatureStore,
    BlobRef,
    # Curation transforms
    QualityScorer,
    DeduplicationTransform,
    ComplianceFilterTransform,
    CurationPipeline,
    # Dataset models
    QualityDimension,
    QualityThreshold,
    DeduplicationAlgorithm,
    DeduplicationConfig,
    ComplianceFlag,
    ComplianceAction,
    ComplianceRule,
    CompliancePolicy,
    # Job infra
    Schedule,
    Target,
    IncrementalConfig,
    FeatureGroupSource,
)

# =============================================================================
# Setup: Dataset Feature Group
# =============================================================================

fs = FeatureStore("acme/pretraining/web-crawl")

# Raw data feature group — as ingested
raw_data = fs.create_feature_group(
    "raw-documents",
    description="Raw web-crawl documents before curation",
    entity_key="doc_id",
    if_exists="skip",
)

raw_data.create_features_from_schema({
    "doc_id": "string",
    "url": "string",
    "text": "string",
    "html": "string",
    "crawl_timestamp": "timestamp",
    "content_language": "string",
    "word_count": "int64",
    # Image references if the page has associated images
    "primary_image_ref": BlobRef(content_types=["image/png", "image/jpeg"]),
}, if_exists="skip")

# Curated data: same entity key, adds curation annotation columns
curated_data = fs.create_feature_group(
    "curated-documents",
    description="Documents with quality, dedup, and compliance annotations",
    entity_key="doc_id",
    if_exists="skip",
)

curated_data.create_features_from_schema({
    "doc_id": "string",
    # Quality scores
    "quality_score": "float64",           # composite
    "quality_fluency": "float64",
    "quality_coherence": "float64",
    "quality_format": "float64",
    # Deduplication
    "is_duplicate": "bool",
    "duplicate_of": "string",             # doc_id of canonical record
    # Compliance
    "compliance_passed": "bool",
    "compliance_pii": "float64",          # PII classifier confidence
    "compliance_nsfw": "float64",
    "compliance_copyright": "float64",
    # Final disposition
    "include_in_training": "bool",
}, if_exists="skip")

print("Feature groups created")

# =============================================================================
# Quality Scoring
# =============================================================================

print("\n" + "=" * 60)
print("QUALITY SCORING")
print("=" * 60)

# Score across three dimensions; filter anything below thresholds
quality_scorer = QualityScorer(
    name="web_quality_scorer",
    dimensions=[
        QualityDimension.FLUENCY,
        QualityDimension.COHERENCE,
        QualityDimension.FORMAT,
    ],
    thresholds=[
        QualityThreshold(QualityDimension.FLUENCY,   min_score=0.5),
        QualityThreshold(QualityDimension.COHERENCE, min_score=0.4),
        QualityThreshold(QualityDimension.FORMAT,    min_score=0.6),
    ],
    model_uri="hf://acme/web-quality-scorer-v2",
    input_columns=["text"],
    composite_column="quality_score",
    write_per_dimension=True,
    filter_below_threshold=True,
)

print(f"Quality scorer: {quality_scorer.name}")
print(f"  Dimensions: {[d.value for d in quality_scorer.dimensions]}")
print(f"  Output columns: {quality_scorer.output_columns}")
print(f"  Thresholds:")
for t in quality_scorer.thresholds:
    print(f"    {t.dimension.value}: min={t.min_score}")
print(f"  Model: {quality_scorer.model_uri}")

# Quality scorer with updated threshold (fluent API)
strict_scorer = quality_scorer.with_threshold(QualityDimension.FLUENCY, min_score=0.7)
print(f"\nStrict scorer fluency threshold: {strict_scorer.thresholds[0].min_score}")

# =============================================================================
# Deduplication
# =============================================================================

print("\n" + "=" * 60)
print("DEDUPLICATION")
print("=" * 60)

# MinHash near-dedup on the text column
dedup_transform = DeduplicationTransform(
    name="web_minhash_dedup",
    config=DeduplicationConfig(
        algorithm=DeduplicationAlgorithm.MINHASH,
        threshold=0.85,           # 85% similarity → duplicate
        key_columns=["text"],
        num_perm=256,             # more permutations = higher accuracy
        ngram_size=5,
        action="flag",            # write is_duplicate=True rather than delete
        keep="first",
    ),
    is_duplicate_column="is_duplicate",
    duplicate_of_column="duplicate_of",
    entity_key_column="doc_id",
)

print(f"Deduplication transform: {dedup_transform.name}")
print(f"  Algorithm: {dedup_transform.config.algorithm.value}")
print(f"  Threshold: {dedup_transform.config.threshold}")
print(f"  Permutations: {dedup_transform.config.num_perm}")
print(f"  Action: {dedup_transform.config.action}")
print(f"  Output columns: {dedup_transform.output_columns}")

# Embedding-based dedup — for multimodal data (image captions)
embedding_dedup = DeduplicationTransform(
    name="caption_embedding_dedup",
    config=DeduplicationConfig(
        algorithm=DeduplicationAlgorithm.EMBEDDING_SIMILARITY,
        threshold=0.95,
        embedding_column="text_embedding",
        action="remove",
        keep="highest_quality",
        quality_column="quality_score",
    ),
)

print(f"\nEmbedding dedup: {embedding_dedup.name}")
print(f"  Algorithm: {embedding_dedup.config.algorithm.value}")
print(f"  Embedding column: {embedding_dedup.config.embedding_column}")
print(f"  Keep: {embedding_dedup.config.keep}")

# =============================================================================
# Compliance Filtering
# =============================================================================

print("\n" + "=" * 60)
print("COMPLIANCE FILTERING")
print("=" * 60)

# Build policy using the fluent API
compliance_policy = CompliancePolicy(name="web-crawl-standard")
compliance_policy.add_rule(
    ComplianceFlag.NSFW,
    action=ComplianceAction.FILTER,
    threshold=0.7,
    model_uri="hf://acme/nsfw-classifier-v1",
)
compliance_policy.add_rule(
    ComplianceFlag.PII,
    action=ComplianceAction.REDACT,   # Don't filter — redact in-place
    threshold=0.6,
    model_uri="hf://acme/pii-detector-v2",
)
compliance_policy.add_rule(
    ComplianceFlag.COPYRIGHT,
    action=ComplianceAction.FLAG,     # Keep but mark for legal review
    threshold=0.85,
    model_uri="hf://acme/copyright-detector-v1",
)

compliance_transform = ComplianceFilterTransform(
    name="web_compliance_filter",
    policy=compliance_policy,
    input_columns=["text", "url"],
    passed_column="compliance_passed",
)

print(f"Compliance policy: {compliance_policy.name}")
print(f"  Flagged categories: {[f.value for f in compliance_policy.flagged_categories]}")
print(f"\nCompliance transform: {compliance_transform.name}")
print(f"  Output columns: {compliance_transform.output_columns}")
print(f"  Rules:")
for rule in compliance_policy.rules:
    print(f"    {rule.flag.value}: threshold={rule.threshold}, action={rule.action.value}")

# Annotate-only mode (for audit / no data removal)
audit_compliance = ComplianceFilterTransform(
    name="compliance_audit",
    policy=CompliancePolicy(
        name="audit-only",
        rules=[
            ComplianceRule(ComplianceFlag.HATE_SPEECH, threshold=0.5),
            ComplianceRule(ComplianceFlag.TOXIC, threshold=0.5),
        ],
        annotate_only=True,    # Never filter; only add columns
    ),
    input_columns=["text"],
)
print(f"\nAudit-only compliance: annotate_only={audit_compliance.policy.annotate_only}")

# =============================================================================
# Curation Pipeline
# =============================================================================

print("\n" + "=" * 60)
print("CURATION PIPELINE")
print("=" * 60)

# Compose steps into a named pipeline
curation_pipeline = CurationPipeline(
    name="web-crawl-curation",
    description="Full curation pipeline: quality → dedup → compliance",
    steps=[
        quality_scorer,
        dedup_transform,
        compliance_transform,
    ],
    stop_on_first_filter=False,   # Run all steps; collect all annotations
)

print(f"Pipeline: {curation_pipeline.name}")
print(f"  Steps: {[s.name for s in curation_pipeline.steps]}")
print(f"  All output columns: {curation_pipeline.all_output_columns}")
print(f"  Quality scorer: {curation_pipeline.quality_scorer().name}")
print(f"  Deduplication: {curation_pipeline.deduplication().name}")
print(f"  Compliance: {curation_pipeline.compliance().name}")

# =============================================================================
# Running as Jobs
# =============================================================================

print("\n" + "=" * 60)
print("CURATION JOBS")
print("=" * 60)

# Quality scoring job
quality_job = fs.create_job(
    name="web_quality_scoring",
    description="Score quality for all raw documents",
    sources=[FeatureGroupSource(
        feature_group="acme/pretraining/web-crawl/raw-documents",
        features=["doc_id", "text", "word_count"],
    )],
    transform=quality_scorer,
    target=Target(
        feature_group="curated-documents",
        features={
            "quality_score":    "quality_score",
            "quality_fluency":  "quality_fluency",
            "quality_coherence": "quality_coherence",
            "quality_format":   "quality_format",
        },
        write_mode="upsert",
        key_columns=["doc_id"],
    ),
    schedule=Schedule.daily(hour=1),
    incremental=IncrementalConfig.incremental("crawl_timestamp"),
    tags=["curation", "quality"],
)

# Deduplication job (runs after quality; only considers high-quality docs)
dedup_job = fs.create_job(
    name="web_near_dedup",
    description="Near-deduplication with MinHash on high-quality docs",
    sources=[FeatureGroupSource(
        feature_group="acme/pretraining/web-crawl/curated-documents",
        features=["doc_id", "text"],
        filters=["quality_score >= 0.5"],     # Only dedup quality docs
    )],
    transform=dedup_transform,
    target=Target(
        feature_group="curated-documents",
        features={
            "is_duplicate": "is_duplicate",
            "duplicate_of": "duplicate_of",
        },
        write_mode="upsert",
        key_columns=["doc_id"],
    ),
    schedule=Schedule.daily(hour=3),
    tags=["curation", "deduplication"],
)

# Compliance job
compliance_job = fs.create_job(
    name="web_compliance",
    description="Flag/filter non-compliant content",
    sources=[FeatureGroupSource(
        feature_group="acme/pretraining/web-crawl/curated-documents",
        features=["doc_id", "text", "url"],
        filters=["is_duplicate == False", "quality_score >= 0.5"],
    )],
    transform=compliance_transform,
    target=Target(
        feature_group="curated-documents",
        features={
            "compliance_passed":    "compliance_passed",
            "compliance_pii":       "compliance_pii",
            "compliance_nsfw":      "compliance_nsfw",
            "compliance_copyright": "compliance_copyright",
        },
        write_mode="upsert",
        key_columns=["doc_id"],
    ),
    schedule=Schedule.daily(hour=5),
    tags=["curation", "compliance"],
)

# Final disposition job: compose all signals into include_in_training
disposition_job = fs.create_job(
    name="web_training_disposition",
    description="Compute final include_in_training flag",
    sources=[FeatureGroupSource(
        feature_group="acme/pretraining/web-crawl/curated-documents",
        features=["doc_id", "quality_score", "is_duplicate", "compliance_passed"],
    )],
    transform=None,   # Pure SQL
    target=Target(
        feature_group="curated-documents",
        features={"include_in_training": "include_in_training"},
        write_mode="upsert",
        key_columns=["doc_id"],
    ),
    schedule=Schedule.daily(hour=6),
    tags=["curation", "disposition"],
)

print(f"Curation jobs created:")
print(f"  1. {quality_job.name} — daily at 01:00")
print(f"  2. {dedup_job.name} — daily at 03:00")
print(f"  3. {compliance_job.name} — daily at 05:00")
print(f"  4. {disposition_job.name} — daily at 06:00")

# =============================================================================
# Curation Statistics (what you'd query after curation runs)
# =============================================================================

print("\n" + "=" * 60)
print("CURATION STATISTICS SCHEMA")
print("=" * 60)

# These are the analytics you'd run on the curated-documents feature group
# to understand the effect of each curation step.
curation_stats = {
    "total_raw_documents": "COUNT(*)",
    "quality_pass_rate": "AVG(quality_score >= 0.5)",
    "dedup_rate": "AVG(is_duplicate)",
    "compliance_pass_rate": "AVG(compliance_passed)",
    "final_training_rate": "AVG(include_in_training)",
    "avg_quality_score": "AVG(quality_score)",
    "pii_rate": "AVG(compliance_pii >= 0.6)",
    "nsfw_rate": "AVG(compliance_nsfw >= 0.7)",
}

print("\nCuration funnel statistics (queried from curated-documents):")
for stat, query in curation_stats.items():
    print(f"  {stat}: {query}")

print("\n" + "=" * 60)
print("ALL DATASET CURATION EXAMPLES COMPLETE!")
print("=" * 60)
