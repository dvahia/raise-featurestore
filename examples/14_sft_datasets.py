"""
Example 14: Supervised Fine-Tuning (SFT) Dataset Management

Demonstrates building and managing SFT datasets in the feature store:
instruction-response pairs, multi-turn conversations, and multimodal SFT
(image + instruction + response). Quality filtering specific to SFT data
and dataset versioning for experiment tracking are also shown.

Key concepts:
- Feature schemas for instruction-response pairs and conversations
- Multimodal SFT: image_ref or video_ref alongside text turns
- Array(BlobRef) for multi-image inputs
- SFT-specific quality signals: response length, refusal rate, format compliance
- Dataset versioning keyed to model training runs
- Mixing SFT sources with different instruction styles
"""

from raise_ import (
    FeatureStore,
    BlobRef,
    Array,
    Struct,
    # Mixing / versioning
    MixingStrategy,
    MixSource,
    DatasetMix,
    DatasetVersion,
    DatasetProvenance,
    # Curation
    QualityScorer,
    QualityDimension,
    QualityThreshold,
    ComplianceFilterTransform,
    CompliancePolicy,
    ComplianceRule,
    ComplianceFlag,
    ComplianceAction,
    # Job infra
    Schedule,
    Target,
    IncrementalConfig,
    FeatureGroupSource,
)
from datetime import datetime

# =============================================================================
# Setup
# =============================================================================

fs = FeatureStore("acme/posttraining/sft")

# =============================================================================
# Instruction-Response Pairs (simplest SFT format)
# =============================================================================

print("=" * 60)
print("INSTRUCTION-RESPONSE PAIRS")
print("=" * 60)

ir_pairs = fs.create_feature_group(
    "instruction-response-pairs",
    description="Single-turn instruction-response SFT examples",
    entity_key="example_id",
    if_exists="skip",
)

ir_pairs.create_features_from_schema({
    "example_id": "string",
    "system_prompt": "string",
    "instruction": "string",
    "response": "string",
    # Provenance
    "source": "string",              # "sharegpt", "alpaca", "internal", etc.
    "original_model": "string",      # model that generated the response (if synthetic)
    # Quality signals
    "instruction_length_tokens": "int64",
    "response_length_tokens": "int64",
    "language": "string",
    "quality_score": "float64",
    "is_refusal": "bool",            # did the model refuse the instruction?
    "format_valid": "bool",          # does response follow expected format?
    # Curation flags
    "is_duplicate": "bool",
    "compliance_passed": "bool",
    "include_in_training": "bool",
}, if_exists="skip")

print(f"Created: instruction-response-pairs")

# =============================================================================
# Multi-Turn Conversations
# =============================================================================

print("\n" + "=" * 60)
print("MULTI-TURN CONVERSATIONS")
print("=" * 60)

# Each row is one complete conversation (serialized as JSON string).
# Turn-level metadata (token counts, role) are tracked separately
# for efficient querying without deserializing the full conversation.
conversations = fs.create_feature_group(
    "conversations",
    description="Multi-turn chat conversations for SFT",
    entity_key="conversation_id",
    if_exists="skip",
)

conversations.create_features_from_schema({
    "conversation_id": "string",
    "system_prompt": "string",
    # The full conversation stored as JSON (list of {"role": ..., "content": ...})
    # Using string as the storage type; the feature store doesn't parse JSON.
    "turns_json": "string",
    # Aggregate stats (derived from turns_json, computed at write time)
    "num_turns": "int64",
    "num_user_turns": "int64",
    "num_assistant_turns": "int64",
    "total_tokens": "int64",
    "max_turn_tokens": "int64",
    # Metadata
    "source": "string",
    "language": "string",
    "topic_tags": "string",          # comma-separated topic labels
    "quality_score": "float64",
    "is_duplicate": "bool",
    "compliance_passed": "bool",
    "include_in_training": "bool",
}, if_exists="skip")

print(f"Created: conversations")
print("  Multi-turn conversations store turns as JSON string.")
print("  Aggregate stats (num_turns, total_tokens) are scalar features.")

# =============================================================================
# Multimodal SFT: Single Image + Instruction
# =============================================================================

print("\n" + "=" * 60)
print("MULTIMODAL SFT — IMAGE + INSTRUCTION")
print("=" * 60)

image_sft = fs.create_feature_group(
    "image-instruction-pairs",
    description="Single-image SFT examples (VQA, captioning, grounding)",
    entity_key="example_id",
    if_exists="skip",
)

image_sft.create_features_from_schema({
    "example_id": "string",
    # Image input — single image per example
    "image_ref": BlobRef(content_types=["image/png", "image/jpeg", "image/webp"]),
    "image_width": "int64",
    "image_height": "int64",
    # Text turns
    "system_prompt": "string",
    "instruction": "string",
    "response": "string",
    # Task type (guides training loss weighting)
    "task_type": "string",           # "vqa", "captioning", "ocr", "grounding", "reasoning"
    # Quality
    "quality_score": "float64",
    "language": "string",
    "source": "string",
    "is_duplicate": "bool",
    "compliance_passed": "bool",
    "include_in_training": "bool",
}, if_exists="skip")

print(f"Created: image-instruction-pairs")
print("  task_type field guides loss weighting per task during training.")

# =============================================================================
# Multimodal SFT: Multiple Images (Interleaved)
# =============================================================================

print("\n" + "=" * 60)
print("MULTIMODAL SFT — MULTI-IMAGE (INTERLEAVED)")
print("=" * 60)

# For interleaved image-text, a conversation can reference multiple images.
# We store image refs as an array alongside the conversation JSON
# (which embeds placeholder tokens like <image_1>, <image_2>).
multi_image_sft = fs.create_feature_group(
    "multi-image-conversations",
    description="Interleaved multi-image conversations for SFT",
    entity_key="conversation_id",
    if_exists="skip",
)

multi_image_sft.create_features_from_schema({
    "conversation_id": "string",
    # Ordered list of image references matching <image_N> placeholders in turns_json
    "image_refs": Array(BlobRef(content_types=["image/png", "image/jpeg", "image/webp"])),
    "num_images": "int64",
    # Conversation text with <image_N> placeholder tokens
    "turns_json": "string",
    "num_turns": "int64",
    "total_tokens": "int64",
    # Metadata
    "source": "string",
    "task_type": "string",           # "multi-image-reasoning", "document-qa", "comic-understanding"
    "quality_score": "float64",
    "include_in_training": "bool",
}, if_exists="skip")

print(f"Created: multi-image-conversations")
print("  image_refs is Array(BlobRef) — ordered list matching <image_N> tokens in turns_json.")

# =============================================================================
# Video + Instruction SFT
# =============================================================================

print("\n" + "=" * 60)
print("MULTIMODAL SFT — VIDEO + INSTRUCTION")
print("=" * 60)

video_sft = fs.create_feature_group(
    "video-instruction-pairs",
    description="Video QA and description SFT examples",
    entity_key="example_id",
    if_exists="skip",
)

video_sft.create_features_from_schema({
    "example_id": "string",
    # Full video or clip reference
    "video_ref": BlobRef(content_types=["video/mp4", "video/webm"]),
    "video_duration_sec": "float64",
    "video_fps": "float64",
    # For clip-based examples: populate start_sec/end_sec
    "clip_start_sec": "float64",
    "clip_end_sec": "float64",
    # Optional sampled frames for frame-level tasks
    "frame_refs": Array(BlobRef(content_types=["image/jpeg"])),
    "num_frames": "int64",
    # Text turns
    "system_prompt": "string",
    "instruction": "string",
    "response": "string",
    "task_type": "string",           # "video-qa", "temporal-grounding", "description"
    # Quality
    "quality_score": "float64",
    "include_in_training": "bool",
}, if_exists="skip")

print(f"Created: video-instruction-pairs")
print("  clip_start_sec/end_sec let the model loader extract the relevant segment.")
print("  frame_refs provides pre-sampled frames for frame-level task variants.")

# =============================================================================
# SFT-Specific Quality Scoring
# =============================================================================

print("\n" + "=" * 60)
print("SFT QUALITY SCORING")
print("=" * 60)

# SFT quality cares about response completeness, relevance, and safety
sft_quality_scorer = QualityScorer(
    name="sft_quality_scorer",
    dimensions=[
        QualityDimension.RELEVANCE,     # Does response address the instruction?
        QualityDimension.COMPLETENESS,  # Is the response fully formed?
        QualityDimension.SAFETY,        # Is the response safe?
        QualityDimension.FORMAT,        # Does it follow format requirements?
    ],
    thresholds=[
        QualityThreshold(QualityDimension.RELEVANCE,    min_score=0.7),
        QualityThreshold(QualityDimension.COMPLETENESS, min_score=0.6),
        QualityThreshold(QualityDimension.SAFETY,       min_score=0.9),  # safety is strict
        QualityThreshold(QualityDimension.FORMAT,       min_score=0.5),
    ],
    model_uri="hf://acme/sft-quality-scorer-v1",
    input_columns=["instruction", "response"],
    composite_column="quality_score",
    filter_below_threshold=True,
)

print(f"SFT quality scorer: {sft_quality_scorer.name}")
for dim, thr in zip(sft_quality_scorer.dimensions, sft_quality_scorer.thresholds):
    print(f"  {dim.value}: min={thr.min_score}")

# SFT compliance (stricter than pre-training)
sft_compliance = ComplianceFilterTransform(
    name="sft_compliance",
    policy=CompliancePolicy(
        name="sft-strict",
        rules=[
            ComplianceRule(ComplianceFlag.NSFW,       threshold=0.5),  # strict NSFW
            ComplianceRule(ComplianceFlag.TOXIC,      threshold=0.4),
            ComplianceRule(ComplianceFlag.HATE_SPEECH,threshold=0.3),
            ComplianceRule(ComplianceFlag.PII,        action=ComplianceAction.REDACT, threshold=0.5),
        ],
    ),
    input_columns=["instruction", "response"],
)

print(f"\nSFT compliance: {sft_compliance.name}")
print(f"  Rules: {[r.flag.value for r in sft_compliance.policy.rules]}")

# =============================================================================
# SFT Dataset Versioning
# =============================================================================

print("\n" + "=" * 60)
print("SFT DATASET VERSIONING")
print("=" * 60)

# SFT datasets version with training runs
sft_v1 = DatasetVersion(
    name="acme-sft-mix",
    version="sft-v1.0",
    feature_group="acme/posttraining/sft/conversations",
    num_records=500_000,
    size_bytes=2 * 1024**3,
    applied_filters=["include_in_training == True"],
    created_by="posttraining-team@acme.com",
    tags=["sft", "text-only", "english"],
)

# Add multimodal examples for v2
sft_v2 = sft_v1.derive(
    version="sft-v2.0",
    applied_filters=["include_in_training == True"],
    num_records=1_200_000,
    size_bytes=15 * 1024**3,   # larger due to image data references
    metadata={
        "modalities": ["text", "image"],
        "image_examples": 700_000,
        "text_examples": 500_000,
    },
)

print(f"SFT v1: {sft_v1}")
print(f"  Records: {sft_v1.num_records:,}")
print(f"SFT v2: {sft_v2}")
print(f"  Records: {sft_v2.num_records:,} ({sft_v2.metadata['image_examples']:,} image)")
print(f"  Parent: {sft_v2.parent_version}")

# =============================================================================
# SFT Mixing: Text + Multimodal Sources
# =============================================================================

print("\n" + "=" * 60)
print("SFT MIXING: TEXT + MULTIMODAL")
print("=" * 60)

# Balance text-only and multimodal SFT examples
text_sft_source = MixSource(
    feature_group="acme/posttraining/sft/conversations",
    weight=0.50,
    filters=["include_in_training == True"],
    tags=["text-only"],
)
image_sft_source = MixSource(
    feature_group="acme/posttraining/sft/image-instruction-pairs",
    weight=0.30,
    filters=["include_in_training == True"],
    tags=["image"],
)
video_sft_source = MixSource(
    feature_group="acme/posttraining/sft/video-instruction-pairs",
    weight=0.10,
    filters=["include_in_training == True"],
    tags=["video"],
)
multi_image_source = MixSource(
    feature_group="acme/posttraining/sft/multi-image-conversations",
    weight=0.10,
    filters=["include_in_training == True"],
    tags=["multi-image"],
)

sft_mix = DatasetMix(
    name="sft-v2-mix",
    sources=[text_sft_source, image_sft_source, video_sft_source, multi_image_source],
    strategy=MixingStrategy.WEIGHTED,
    total_samples=1_200_000,
    description="SFT v2 mix: 50% text, 30% image, 10% video, 10% multi-image",
)

print(f"SFT mix: {sft_mix}")
labels = ["Text-only", "Image", "Video", "Multi-image"]
for label, prob in zip(labels, sft_mix.effective_weights()):
    print(f"  {label}: {prob:.0%}")

print("\n" + "=" * 60)
print("ALL SFT DATASET EXAMPLES COMPLETE!")
print("=" * 60)
