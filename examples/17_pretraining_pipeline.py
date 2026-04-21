"""
Example 17: Pre-Training Data Pipeline (System-Level)

End-to-end pipeline for building a foundation model pre-training dataset:
raw ingestion → quality scoring → near-deduplication → compliance filtering →
dataset versioning → multi-source mixing → training-ready export.

This example shows how the feature store acts as the backbone of a production
pre-training data infrastructure, with each step tracked, versioned, and
auditable.

Coverage:
- Multi-modal ingestion: web text, images, video, audio blobs
- Curation pipeline: quality + dedup + compliance
- Multi-source mixing with temperature strategy
- Dataset versioning and provenance tracking
- Lineage from raw crawl to training checkpoint
- Scheduled orchestration of the full pipeline
"""

from datetime import datetime

from raise_ import (
    FeatureStore,
    BlobRef,
    Array,
    # Multimodal
    BlobRegistry,
    ContentType,
    IntegrityPolicy,
    MultimodalSource,
    TimeRange,
    # Curation
    QualityScorer,
    DeduplicationTransform,
    ComplianceFilterTransform,
    CurationPipeline,
    QualityDimension,
    QualityThreshold,
    DeduplicationAlgorithm,
    DeduplicationConfig,
    CompliancePolicy,
    ComplianceRule,
    ComplianceFlag,
    ComplianceAction,
    # Dataset
    MixingStrategy,
    MixSource,
    DatasetMix,
    DatasetVersion,
    DatasetProvenance,
    # Job infra
    Schedule,
    Target,
    IncrementalConfig,
    FeatureGroupSource,
    SQLTransform,
    ObjectStorage,
    Job,
)
from raise_.transforms import HashAlgorithm

print("=" * 70)
print("PRE-TRAINING DATA PIPELINE")
print("=" * 70)

# =============================================================================
# 1. Feature Store Context
# =============================================================================

fs = FeatureStore("acme/pretraining/foundation-v4")

# =============================================================================
# 2. Raw Ingestion Feature Groups
# =============================================================================

print("\n[STEP 1] RAW INGESTION FEATURE GROUPS")
print("-" * 50)

# Text documents from web crawl
raw_text = fs.create_feature_group(
    "raw-text",
    description="Raw web-crawl text documents",
    entity_key="doc_id",
    if_exists="skip",
)
raw_text.create_features_from_schema({
    "doc_id": "string",
    "url": "string",
    "text": "string",
    "language": "string",
    "crawl_timestamp": "timestamp",
    "word_count": "int64",
    "source_domain": "string",
}, if_exists="skip")

# Image-text pairs (alt-text from web)
raw_image_text = fs.create_feature_group(
    "raw-image-text",
    description="Web-sourced image + alt-text pairs",
    entity_key="pair_id",
    if_exists="skip",
)
raw_image_text.create_features_from_schema({
    "pair_id": "string",
    "image_ref": BlobRef(content_types=["image/jpeg", "image/png", "image/webp"]),
    "alt_text": "string",
    "caption": "string",
    "url": "string",
    "language": "string",
}, if_exists="skip")

# Video-text pairs (from subtitled video)
raw_video_text = fs.create_feature_group(
    "raw-video-text",
    description="Video clips with subtitles for interleaved training",
    entity_key="clip_id",
    if_exists="skip",
)
raw_video_text.create_features_from_schema({
    "clip_id": "string",
    "video_ref": BlobRef(content_types=["video/mp4", "video/webm"]),
    "start_sec": "float64",
    "end_sec": "float64",
    "subtitle_text": "string",
    "language": "string",
    "source": "string",
}, if_exists="skip")

# Audio with transcription
raw_audio_text = fs.create_feature_group(
    "raw-audio-text",
    description="Audio segments with transcript for speech training",
    entity_key="segment_id",
    if_exists="skip",
)
raw_audio_text.create_features_from_schema({
    "segment_id": "string",
    "audio_ref": BlobRef(content_types=["audio/wav", "audio/flac"]),
    "transcript": "string",
    "language": "string",
    "duration_sec": "float64",
    "source": "string",
}, if_exists="skip")

print(f"  ✓ raw-text")
print(f"  ✓ raw-image-text")
print(f"  ✓ raw-video-text")
print(f"  ✓ raw-audio-text")

# =============================================================================
# 3. Curated Feature Groups (curation annotations)
# =============================================================================

print("\n[STEP 2] CURATED FEATURE GROUPS")
print("-" * 50)

curated_text = fs.create_feature_group(
    "curated-text",
    description="Text documents with curation annotations",
    entity_key="doc_id",
    if_exists="skip",
)
curated_text.create_features_from_schema({
    "doc_id": "string",
    # Quality
    "quality_score": "float64",
    "quality_fluency": "float64",
    "quality_coherence": "float64",
    # Dedup
    "is_duplicate": "bool",
    "duplicate_of": "string",
    # Compliance
    "compliance_passed": "bool",
    "compliance_nsfw": "float64",
    "compliance_pii": "float64",
    "compliance_copyright": "float64",
    # Final
    "include_in_training": "bool",
    "token_count": "int64",
}, if_exists="skip")

curated_image_text = fs.create_feature_group(
    "curated-image-text",
    description="Image-text pairs with curation annotations",
    entity_key="pair_id",
    if_exists="skip",
)
curated_image_text.create_features_from_schema({
    "pair_id": "string",
    "quality_score": "float64",
    "quality_relevance": "float64",       # how relevant is alt_text to image
    "quality_completeness": "float64",
    "is_duplicate": "bool",
    "compliance_passed": "bool",
    "include_in_training": "bool",
}, if_exists="skip")

print(f"  ✓ curated-text")
print(f"  ✓ curated-image-text")

# =============================================================================
# 4. Curation Pipeline Definition
# =============================================================================

print("\n[STEP 3] CURATION PIPELINE")
print("-" * 50)

# Compliance policy (applied across all modalities)
compliance_policy = CompliancePolicy(name="pretraining-v4-compliance")
compliance_policy.add_rule(ComplianceFlag.NSFW, threshold=0.6)
compliance_policy.add_rule(ComplianceFlag.HATE_SPEECH, threshold=0.5)
compliance_policy.add_rule(ComplianceFlag.COPYRIGHT, action=ComplianceAction.FLAG, threshold=0.85)
compliance_policy.add_rule(ComplianceFlag.PII, action=ComplianceAction.REDACT, threshold=0.6)
compliance_policy.add_rule(ComplianceFlag.CHILD_SAFETY, threshold=0.3)   # very strict

# Text curation pipeline
text_curation = CurationPipeline(
    name="text-curation-v4",
    description="Quality → dedup → compliance for web text",
    steps=[
        QualityScorer(
            name="text_quality",
            dimensions=[
                QualityDimension.FLUENCY,
                QualityDimension.COHERENCE,
                QualityDimension.FORMAT,
            ],
            thresholds=[
                QualityThreshold(QualityDimension.FLUENCY,   min_score=0.5),
                QualityThreshold(QualityDimension.COHERENCE, min_score=0.4),
            ],
            model_uri="hf://acme/text-quality-scorer-v3",
            input_columns=["text"],
        ),
        DeduplicationTransform(
            name="text_minhash_dedup",
            config=DeduplicationConfig(
                algorithm=DeduplicationAlgorithm.MINHASH,
                threshold=0.80,
                key_columns=["text"],
                num_perm=256,
                action="flag",
            ),
        ),
        ComplianceFilterTransform(
            name="text_compliance",
            policy=compliance_policy,
            input_columns=["text", "url"],
        ),
    ],
)

# Image-text curation pipeline
image_text_curation = CurationPipeline(
    name="image-text-curation-v4",
    steps=[
        QualityScorer(
            name="image_text_quality",
            dimensions=[
                QualityDimension.RELEVANCE,     # image ↔ alt-text relevance
                QualityDimension.COMPLETENESS,
            ],
            thresholds=[
                QualityThreshold(QualityDimension.RELEVANCE, min_score=0.4),
            ],
            model_uri="hf://acme/image-text-relevance-scorer-v1",
            input_columns=["alt_text", "image_ref"],
        ),
        DeduplicationTransform(
            name="image_perceptual_dedup",
            config=DeduplicationConfig(
                algorithm=DeduplicationAlgorithm.EMBEDDING_SIMILARITY,
                threshold=0.95,
                embedding_column="image_embedding",
                action="flag",
            ),
        ),
        ComplianceFilterTransform(
            name="image_compliance",
            policy=compliance_policy,
            input_columns=["alt_text", "image_ref"],
        ),
    ],
)

print(f"  Text curation pipeline: {text_curation}")
print(f"    Output columns: {text_curation.all_output_columns}")
print(f"  Image-text curation: {image_text_curation}")

# =============================================================================
# 5. Curation Jobs
# =============================================================================

print("\n[STEP 4] CURATION JOBS")
print("-" * 50)

def make_curation_job(
    name: str,
    source_fg: str,
    source_features: list[str],
    pipeline: CurationPipeline,
    target_fg: str,
    target_features: dict,
    timestamp_col: str,
) -> Job:
    return fs.create_job(
        name=name,
        sources=[FeatureGroupSource(
            feature_group=source_fg,
            features=source_features,
        )],
        transform=pipeline.steps[0],  # first step; orchestrator chains the rest
        target=Target(
            feature_group=target_fg,
            features=target_features,
            write_mode="upsert",
            key_columns=[source_features[0]],
        ),
        schedule=Schedule.daily(hour=2),
        incremental=IncrementalConfig.incremental(timestamp_col),
        tags=["curation", "pretraining"],
    )

text_curation_job = fs.create_job(
    name="text_curation_v4",
    sources=[FeatureGroupSource(
        feature_group="acme/pretraining/foundation-v4/raw-text",
        features=["doc_id", "text", "url", "crawl_timestamp", "language"],
    )],
    transform=text_curation.steps[0],
    target=Target(
        feature_group="curated-text",
        features={c: c for c in text_curation.all_output_columns + ["doc_id"]},
        write_mode="upsert",
        key_columns=["doc_id"],
    ),
    schedule=Schedule.daily(hour=2),
    incremental=IncrementalConfig.incremental("crawl_timestamp"),
    tags=["curation", "text"],
)

image_curation_job = fs.create_job(
    name="image_text_curation_v4",
    sources=[FeatureGroupSource(
        feature_group="acme/pretraining/foundation-v4/raw-image-text",
        features=["pair_id", "image_ref", "alt_text", "url"],
    )],
    transform=image_text_curation.steps[0],
    target=Target(
        feature_group="curated-image-text",
        features={c: c for c in image_text_curation.all_output_columns + ["pair_id"]},
        write_mode="upsert",
        key_columns=["pair_id"],
    ),
    schedule=Schedule.daily(hour=3),
    tags=["curation", "image-text"],
)

print(f"  ✓ {text_curation_job.name}")
print(f"  ✓ {image_curation_job.name}")

# =============================================================================
# 6. Dataset Mixing
# =============================================================================

print("\n[STEP 5] DATASET MIXING")
print("-" * 50)

# Curated sources with training-ready filters
text_mix_source = MixSource(
    feature_group="acme/pretraining/foundation-v4/curated-text",
    weight=0.60,
    filters=["include_in_training == True", "language == 'en'"],
)
multilingual_text_source = MixSource(
    feature_group="acme/pretraining/foundation-v4/curated-text",
    weight=0.10,
    filters=["include_in_training == True", "language != 'en'"],
)
image_text_mix_source = MixSource(
    feature_group="acme/pretraining/foundation-v4/curated-image-text",
    weight=0.20,
    filters=["include_in_training == True"],
)
video_mix_source = MixSource(
    feature_group="acme/pretraining/foundation-v4/raw-video-text",
    weight=0.05,
    filters=["language == 'en'"],   # use raw; video dedup not yet run
)
audio_mix_source = MixSource(
    feature_group="acme/pretraining/foundation-v4/raw-audio-text",
    weight=0.05,
    filters=["language == 'en'"],
)

pretraining_mix = DatasetMix(
    name="foundation-v4-mix",
    sources=[
        text_mix_source, multilingual_text_source,
        image_text_mix_source, video_mix_source, audio_mix_source,
    ],
    strategy=MixingStrategy.TEMPERATURE,
    temperature=0.7,
    total_samples=5_000_000_000,
    description="Foundation v4 multi-modal training mix",
)

print(f"  Mix: {pretraining_mix}")
source_labels = ["EN Text", "Multilingual Text", "Image+Text", "Video+Text", "Audio+Text"]
for label, weight in zip(source_labels, pretraining_mix.effective_weights()):
    print(f"    {label}: {weight:.1%}")

# =============================================================================
# 7. Dataset Versioning
# =============================================================================

print("\n[STEP 6] DATASET VERSIONING")
print("-" * 50)

web_provenance = DatasetProvenance(
    source_name="CommonCrawl-2024-Q1 + Q2",
    source_uri="s3://commoncrawl/crawl-data/",
    collection_date=datetime(2024, 6, 1),
    license="CC0",
    language_codes=["en", "zh", "es", "fr", "de", "ja", "ko", "pt", "ru", "ar", "ar", "hi"],
    domain_tags=["web", "news", "forum", "blog", "academic"],
    pii_reviewed=True,
    metadata={"pages": 5_000_000_000},
)

foundation_v4 = DatasetVersion(
    name="acme-foundation-v4",
    version="v4.0.0",
    feature_group="acme/pretraining/foundation-v4/training-mix",
    num_records=5_000_000_000,
    size_bytes=20_000 * 1024**3,  # 20 TB
    applied_filters=[
        "include_in_training == True",
        "quality_score >= 0.5",
        "is_duplicate == False",
        "compliance_passed == True",
    ],
    provenance=web_provenance,
    tags=["pretraining", "multimodal", "multilingual", "v4"],
    metadata={
        "mix_strategy": "temperature",
        "temperature": 0.7,
        "modalities": ["text", "image", "video", "audio"],
    },
)

print(f"  Dataset version: {foundation_v4}")
print(f"  Size: {foundation_v4.size_gb:.0f} GB")
print(f"  Records: {foundation_v4.num_records:,}")
print(f"  Modalities: {foundation_v4.metadata['modalities']}")

# =============================================================================
# 8. Pipeline Summary
# =============================================================================

print("\n" + "=" * 70)
print("PIPELINE SUMMARY")
print("=" * 70)

pipeline_stages = [
    ("Stage 1: Ingestion",   "Scan object storage → register BlobReferences → write raw feature groups"),
    ("Stage 2: Quality",     "QualityScorer → write quality_score, per-dimension scores → daily at 02:00"),
    ("Stage 3: Dedup",       "MinHash near-dedup → write is_duplicate, duplicate_of → daily at 04:00"),
    ("Stage 4: Compliance",  "PII/NSFW/copyright classifiers → write compliance_* → daily at 06:00"),
    ("Stage 5: Disposition", "SQL: include_in_training = quality_score≥0.5 AND NOT is_duplicate AND compliance_passed"),
    ("Stage 6: Mixing",      f"Temperature mix (T=0.7) → {pretraining_mix.total_samples:,} total samples"),
    ("Stage 7: Versioning",  f"Tag snapshot as {foundation_v4.version} → write to DatasetVersion registry"),
]

for stage, description in pipeline_stages:
    print(f"  {stage}")
    print(f"    {description}")

print(f"\nPipeline schedule:")
print(f"  Stages 1-5: daily cadence (runs overnight)")
print(f"  Stage 6-7: weekly cadence or triggered manually before training run")
print(f"\nLineage: raw-text → curated-text → training-mix@{foundation_v4.version}")

print("\n" + "=" * 70)
print("PRE-TRAINING PIPELINE COMPLETE!")
print("=" * 70)
