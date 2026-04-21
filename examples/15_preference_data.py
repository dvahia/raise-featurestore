"""
Example 15: Preference Data — RLHF, DPO, and Reward Modeling

Demonstrates managing human preference data and reward model training data.
Covers: collecting preference pairs, DPO triplets, reward model training sets,
annotator metadata, and per-response reward scores.

Key concepts:
- Preference pair schema: prompt + chosen + rejected
- DPO triplets: same pair but includes reference model log probabilities
- Reward model training: prompt + response + scalar score
- Annotator metadata stored alongside labels for quality tracking
- Computing gold vs. silver preference labels
- Multimodal preference data (image/video + instruction + responses)
"""

from raise_ import (
    FeatureStore,
    BlobRef,
    Array,
    # Annotation models
    AnnotationTaskType,
    AnnotatorPoolType,
    AnnotatorConfig,
    HumanEvalTask,
    AnnotationResult,
    AnnotationBatch,
    AgreementMetric,
    # Mixing / versioning
    DatasetVersion,
    MixSource,
    DatasetMix,
    MixingStrategy,
    # Job infra
    Schedule,
    Target,
    IncrementalConfig,
    FeatureGroupSource,
    SQLTransform,
)

# =============================================================================
# Setup
# =============================================================================

fs = FeatureStore("acme/posttraining/rlhf")

# =============================================================================
# Preference Pairs (RLHF)
# =============================================================================

print("=" * 60)
print("PREFERENCE PAIRS (RLHF)")
print("=" * 60)

preference_pairs = fs.create_feature_group(
    "preference-pairs",
    description="Human-labeled preference pairs for RLHF training",
    entity_key="pair_id",
    if_exists="skip",
)

preference_pairs.create_features_from_schema({
    "pair_id": "string",
    "prompt": "string",
    "system_prompt": "string",
    "response_a": "string",
    "response_b": "string",
    # Human preference label
    "preferred": "string",           # "response_a" | "response_b" | "tie"
    "preference_confidence": "float64",   # annotator-reported confidence
    # Annotator metadata
    "annotator_id": "string",
    "annotation_duration_sec": "float64",
    "num_annotations": "int64",          # how many annotators saw this pair
    "annotator_agreement": "float64",    # pairwise agreement rate
    # Quality signals
    "is_gold": "bool",                   # high-agreement, high-confidence pair
    "quality_score": "float64",
    # Provenance
    "source": "string",                  # "internal", "crowd", "synthetic"
    "model_a_id": "string",              # which model generated response_a
    "model_b_id": "string",
    # Curation
    "include_in_training": "bool",
}, if_exists="skip")

print("Created: preference-pairs")
print("  preferred: 'response_a' | 'response_b' | 'tie'")
print("  is_gold: high-agreement pairs for reward model training signal")

# =============================================================================
# DPO Triplets
# =============================================================================

print("\n" + "=" * 60)
print("DPO TRIPLETS")
print("=" * 60)

dpo_triplets = fs.create_feature_group(
    "dpo-triplets",
    description="DPO training triplets: prompt + chosen + rejected",
    entity_key="triplet_id",
    if_exists="skip",
)

dpo_triplets.create_features_from_schema({
    "triplet_id": "string",
    "prompt": "string",
    "system_prompt": "string",
    "chosen": "string",               # preferred response
    "rejected": "string",             # dispreferred response
    # DPO-specific: log probabilities from the reference model
    # These are computed offline before training to avoid the reference model
    # overhead during the DPO training loop
    "chosen_ref_logprob": "float64",
    "rejected_ref_logprob": "float64",
    # Margin: how different the two responses are (guides loss weighting)
    "preference_margin": "float64",   # abs(chosen_score - rejected_score)
    # Provenance
    "source_pair_id": "string",       # FK to preference-pairs
    "source": "string",
    "chosen_model_id": "string",
    "rejected_model_id": "string",
    # Quality
    "include_in_training": "bool",
    "split": "string",                # "train" | "validation"
}, if_exists="skip")

print("Created: dpo-triplets")
print("  chosen_ref_logprob, rejected_ref_logprob: pre-computed from reference model")
print("  preference_margin: |chosen_score - rejected_score|; used for filtering easy pairs")

# =============================================================================
# Reward Model Training Data
# =============================================================================

print("\n" + "=" * 60)
print("REWARD MODEL TRAINING DATA")
print("=" * 60)

reward_training = fs.create_feature_group(
    "reward-training-data",
    description="Scored (prompt, response) pairs for reward model training",
    entity_key="example_id",
    if_exists="skip",
)

reward_training.create_features_from_schema({
    "example_id": "string",
    "prompt": "string",
    "system_prompt": "string",
    "response": "string",
    # Reward signal
    "reward_score": "float64",        # normalized scalar score in [0, 1]
    "reward_components": "string",    # JSON: {"helpfulness": 0.8, "safety": 0.9, ...}
    # Annotator-aggregated signals
    "mean_rating": "float64",         # mean of Likert ratings across annotators
    "rating_std": "float64",          # std dev (low = high agreement)
    "num_ratings": "int64",
    # Bradley-Terry scores (if using pairwise comparisons)
    "bt_score": "float64",            # Bradley-Terry win probability
    "elo_score": "float64",           # Elo rating
    # Provenance
    "model_id": "string",             # which model generated the response
    "source": "string",
    # Curation
    "include_in_training": "bool",
    "split": "string",
}, if_exists="skip")

print("Created: reward-training-data")
print("  reward_score: final scalar for RM loss")
print("  bt_score, elo_score: rankings from pairwise comparison tournaments")

# =============================================================================
# Multimodal Preference Data
# =============================================================================

print("\n" + "=" * 60)
print("MULTIMODAL PREFERENCE PAIRS")
print("=" * 60)

multimodal_preferences = fs.create_feature_group(
    "multimodal-preference-pairs",
    description="Preference pairs for multimodal (image/video) responses",
    entity_key="pair_id",
    if_exists="skip",
)

multimodal_preferences.create_features_from_schema({
    "pair_id": "string",
    # Multimodal input
    "image_ref": BlobRef(content_types=["image/png", "image/jpeg", "image/webp"]),
    "video_ref": BlobRef(content_types=["video/mp4"]),  # nullable; one or the other
    "prompt": "string",
    "system_prompt": "string",
    # Response pair (text only; the image/video is shared)
    "response_a": "string",
    "response_b": "string",
    # For grounding tasks, responses may also include bounding box annotations
    "response_a_grounding": "string",  # JSON: list of {label, bbox}
    "response_b_grounding": "string",
    # Preference
    "preferred": "string",
    "preference_confidence": "float64",
    "annotation_criteria": "string",   # which criteria annotators used
    # Provenance & quality
    "annotator_id": "string",
    "task_type": "string",             # "vqa", "captioning", "grounding", "safety"
    "include_in_training": "bool",
}, if_exists="skip")

print("Created: multimodal-preference-pairs")
print("  Supports both image and video inputs; one or the other per example.")

# =============================================================================
# Annotation Task for Preference Collection
# =============================================================================

print("\n" + "=" * 60)
print("ANNOTATION TASK: PREFERENCE COLLECTION")
print("=" * 60)

preference_task = HumanEvalTask(
    name="response-quality-preference",
    task_type=AnnotationTaskType.BINARY_PREFERENCE,
    instructions="""
    You will see a user prompt and two AI responses (A and B).
    Select the response that is:
    1. More helpful: answers the user's question fully and accurately
    2. More harmless: avoids harmful, misleading, or offensive content
    3. More honest: is factually accurate and doesn't hallucinate

    If both responses are equally good or bad, select 'Tie'.
    """.strip(),
    source_feature_group="acme/posttraining/rlhf/preference-pairs",
    source_columns=["prompt", "response_a", "response_b"],
    label_column="preferred",
    options=["response_a", "response_b", "tie"],
    min_annotations=5,                # 5 annotators per pair
    annotator_config=AnnotatorConfig(
        pool_type=AnnotatorPoolType.INTERNAL,
        min_annotations_per_item=5,
        languages=["en"],
        annotator_qualifications=["passed-safety-training", "domain-expert"],
    ),
    agreement_metric=AgreementMetric.FLEISS_KAPPA,
    consensus_strategy="majority",
    tags=["rlhf", "preference"],
)

print(f"Annotation task: {preference_task.name}")
print(f"  Task type: {preference_task.task_type.value}")
print(f"  Min annotations: {preference_task.min_annotations}")
print(f"  Consensus: {preference_task.consensus_strategy}")
print(f"  Pool: {preference_task.annotator_config.pool_type.value}")

# LLM-as-judge for high-throughput preference labeling
llm_judge_task = HumanEvalTask(
    name="llm-judge-preference",
    task_type=AnnotationTaskType.BINARY_PREFERENCE,
    instructions="(LLM judge — see model_judge_prompt_template)",
    source_feature_group="acme/posttraining/rlhf/preference-pairs",
    source_columns=["prompt", "response_a", "response_b"],
    label_column="preferred",
    options=["response_a", "response_b", "tie"],
    min_annotations=1,
    annotator_config=AnnotatorConfig(
        pool_type=AnnotatorPoolType.MODEL,
        model_judge_uri="hf://acme/preference-judge-v2",
        model_judge_prompt_template="""
        System: You are an impartial judge evaluating AI responses.
        User prompt: {prompt}
        Response A: {response_a}
        Response B: {response_b}
        Which response is better? Answer with "A", "B", or "TIE".
        """.strip(),
    ),
    tags=["llm-judge", "high-throughput"],
)

print(f"\nLLM judge task: {llm_judge_task.name}")
print(f"  Pool type: {llm_judge_task.annotator_config.pool_type.value}")
print(f"  Model: {llm_judge_task.annotator_config.model_judge_uri}")

# =============================================================================
# Annotation Results
# =============================================================================

print("\n" + "=" * 60)
print("ANNOTATION RESULTS")
print("=" * 60)

# Simulated annotation results for one pair
annotations_for_pair_001 = [
    AnnotationResult(
        task_name="response-quality-preference",
        item_id="pair_001",
        annotator_id="annotator_A",
        label="response_b",
        confidence=0.9,
        duration_sec=45.0,
    ),
    AnnotationResult(
        task_name="response-quality-preference",
        item_id="pair_001",
        annotator_id="annotator_B",
        label="response_b",
        confidence=0.75,
        duration_sec=62.0,
    ),
    AnnotationResult(
        task_name="response-quality-preference",
        item_id="pair_001",
        annotator_id="annotator_C",
        label="response_a",
        confidence=0.55,
        duration_sec=38.0,
        notes="Both are reasonable but A is more concise",
    ),
    AnnotationResult(
        task_name="response-quality-preference",
        item_id="pair_001",
        annotator_id="annotator_D",
        label="response_b",
        confidence=0.85,
        duration_sec=51.0,
    ),
    AnnotationResult(
        task_name="response-quality-preference",
        item_id="pair_001",
        annotator_id="annotator_E",
        label="response_b",
        confidence=0.70,
        duration_sec=44.0,
    ),
]

# Majority vote: response_b wins 4-1
votes = [r.label for r in annotations_for_pair_001]
from collections import Counter
majority = Counter(votes).most_common(1)[0]
agreement_rate = majority[1] / len(votes)

print(f"Pair 001 annotations: {len(annotations_for_pair_001)}")
print(f"  Votes: {dict(Counter(votes))}")
print(f"  Majority: {majority[0]} ({majority[1]}/{len(votes)} annotators)")
print(f"  Agreement: {agreement_rate:.0%}")
print(f"  Is gold: {agreement_rate >= 0.8}")
print(f"  Avg confidence: {sum(r.confidence for r in annotations_for_pair_001)/len(annotations_for_pair_001):.2f}")

# =============================================================================
# Annotation Batches
# =============================================================================

print("\n" + "=" * 60)
print("ANNOTATION BATCHES")
print("=" * 60)

batch_1 = AnnotationBatch(
    batch_id="batch_2024_001",
    task_name="response-quality-preference",
    item_ids=[f"pair_{i:06d}" for i in range(500)],
    status="in_progress",
)

print(f"Batch: {batch_1.batch_id}")
print(f"  Items: {batch_1.total_items}")
print(f"  Status: {batch_1.status}")
print(f"  Completion: {batch_1.completion_rate:.0%}")

# =============================================================================
# Converting Preferences to DPO Triplets
# =============================================================================

print("\n" + "=" * 60)
print("PREFERENCE → DPO TRIPLET JOB")
print("=" * 60)

# A job that reads preference pairs and prepares DPO triplets,
# including filtering low-confidence and tied pairs.
dpo_prep_job = fs.create_job(
    name="prepare_dpo_triplets",
    description="Convert high-quality preference pairs to DPO format",
    sources=[FeatureGroupSource(
        feature_group="acme/posttraining/rlhf/preference-pairs",
        features=["pair_id", "prompt", "system_prompt", "response_a", "response_b",
                  "preferred", "preference_confidence", "annotator_agreement"],
        filters=[
            "include_in_training == True",
            "preferred != 'tie'",
            "annotator_agreement >= 0.7",
            "preference_confidence >= 0.65",
        ],
    )],
    transform=SQLTransform(
        sql="""
        SELECT
            pair_id AS triplet_id,
            prompt,
            system_prompt,
            CASE WHEN preferred = 'response_a' THEN response_a ELSE response_b END AS chosen,
            CASE WHEN preferred = 'response_a' THEN response_b ELSE response_a END AS rejected,
            ABS(preference_confidence - 0.5) * 2 AS preference_margin,
            pair_id AS source_pair_id,
            'internal' AS source,
            CASE WHEN RAND() < 0.95 THEN 'train' ELSE 'validation' END AS split,
            True AS include_in_training
        FROM source
        """,
    ),
    target=Target(
        feature_group="dpo-triplets",
        write_mode="overwrite",
    ),
    schedule=Schedule.manual(),
    tags=["dpo", "post-training"],
)

print(f"DPO prep job: {dpo_prep_job.name}")
print(f"  Filters out: ties, low-agreement, low-confidence pairs")
print(f"  Chosen = response preferred by majority; Rejected = other response")
print(f"  preference_margin: normalized confidence gap")

# =============================================================================
# Dataset Versioning for RLHF Datasets
# =============================================================================

print("\n" + "=" * 60)
print("RLHF DATASET VERSIONING")
print("=" * 60)

rlhf_v1 = DatasetVersion(
    name="acme-rlhf",
    version="rlhf-v1.0",
    feature_group="acme/posttraining/rlhf/preference-pairs",
    num_records=50_000,
    size_bytes=500 * 1024**2,
    applied_filters=["include_in_training == True", "annotator_agreement >= 0.6"],
    tags=["rlhf", "text-only", "internal"],
)

rlhf_v2 = rlhf_v1.derive(
    version="rlhf-v2.0",
    applied_filters=["include_in_training == True", "annotator_agreement >= 0.7"],
    num_records=150_000,
    size_bytes=3 * 1024**3,
    metadata={
        "includes_multimodal": True,
        "image_pairs": 100_000,
        "text_pairs": 50_000,
    },
)

print(f"RLHF v1: {rlhf_v1}")
print(f"RLHF v2: {rlhf_v2}")
print(f"  Includes multimodal: {rlhf_v2.metadata['includes_multimodal']}")

print("\n" + "=" * 60)
print("ALL PREFERENCE DATA EXAMPLES COMPLETE!")
print("=" * 60)
