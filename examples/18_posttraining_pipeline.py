"""
Example 18: Post-Training Pipeline (System-Level)

End-to-end pipeline for post-training a foundation model: collecting SFT data,
gathering human preferences, training a reward model, preparing DPO triplets,
running evaluations, and managing the human annotation loop.

This example shows the feature store as the coordinating layer for all
post-training data — tracking every artifact from raw human annotation through
model-ready training datasets, with full lineage.

Coverage:
- SFT dataset collection: text + multimodal conversations
- Preference collection: human annotation batches + LLM judge bootstrap
- Reward model training data: converting preferences to scored examples
- DPO preparation: filtering + reference logprob enrichment
- Evaluation: automated benchmarks + human win rate
- Human annotation loop: dispatch → collect → aggregate → gate training
"""

from raise_ import (
    FeatureStore,
    BlobRef,
    Array,
    # Multimodal
    ContentType,
    BlobRegistry,
    IntegrityPolicy,
    TimeRange,
    # Curation
    QualityScorer,
    ComplianceFilterTransform,
    CompliancePolicy,
    ComplianceRule,
    ComplianceFlag,
    ComplianceAction,
    QualityDimension,
    QualityThreshold,
    # Dataset
    MixingStrategy,
    MixSource,
    DatasetMix,
    DatasetVersion,
    # Annotation / eval
    AnnotationTaskType,
    AnnotatorPoolType,
    AnnotatorConfig,
    HumanEvalTask,
    AnnotationResult,
    AnnotationBatch,
    AgreementScore,
    AgreementMetric,
    EvalExample,
    EvalSuite,
    EvalResult,
    EvalMetric,
    # Job infra
    Schedule,
    Target,
    IncrementalConfig,
    FeatureGroupSource,
    SQLTransform,
)

print("=" * 70)
print("POST-TRAINING DATA PIPELINE")
print("=" * 70)

# =============================================================================
# 1. Feature Store Context
# =============================================================================

fs = FeatureStore("acme/posttraining/v2")

# =============================================================================
# 2. Feature Groups
# =============================================================================

print("\n[STEP 1] FEATURE GROUPS")
print("-" * 50)

# --- SFT ---
sft_conversations = fs.create_feature_group(
    "sft-conversations",
    entity_key="example_id",
    if_exists="skip",
)
sft_conversations.create_features_from_schema({
    "example_id": "string",
    "system_prompt": "string",
    "turns_json": "string",
    "num_turns": "int64",
    "total_tokens": "int64",
    "source": "string",
    "language": "string",
    "quality_score": "float64",
    "compliance_passed": "bool",
    "include_in_training": "bool",
    "split": "string",
}, if_exists="skip")

sft_image_pairs = fs.create_feature_group(
    "sft-image-pairs",
    entity_key="example_id",
    if_exists="skip",
)
sft_image_pairs.create_features_from_schema({
    "example_id": "string",
    "image_ref": BlobRef(content_types=["image/jpeg", "image/png", "image/webp"]),
    "instruction": "string",
    "response": "string",
    "task_type": "string",
    "quality_score": "float64",
    "include_in_training": "bool",
    "split": "string",
}, if_exists="skip")

# --- Preference data ---
preference_pairs = fs.create_feature_group(
    "preference-pairs",
    entity_key="pair_id",
    if_exists="skip",
)
preference_pairs.create_features_from_schema({
    "pair_id": "string",
    "prompt": "string",
    "system_prompt": "string",
    "response_a": "string",
    "response_b": "string",
    "preferred": "string",
    "preference_confidence": "float64",
    "annotator_agreement": "float64",
    "annotator_id": "string",
    "num_annotations": "int64",
    "is_gold": "bool",
    "source": "string",
    "model_a_id": "string",
    "model_b_id": "string",
    "include_in_training": "bool",
}, if_exists="skip")

multimodal_prefs = fs.create_feature_group(
    "multimodal-preference-pairs",
    entity_key="pair_id",
    if_exists="skip",
)
multimodal_prefs.create_features_from_schema({
    "pair_id": "string",
    "image_ref": BlobRef(content_types=["image/jpeg", "image/png"]),
    "video_ref": BlobRef(content_types=["video/mp4"]),
    "prompt": "string",
    "response_a": "string",
    "response_b": "string",
    "preferred": "string",
    "task_type": "string",
    "include_in_training": "bool",
}, if_exists="skip")

# --- DPO ---
dpo_triplets = fs.create_feature_group(
    "dpo-triplets",
    entity_key="triplet_id",
    if_exists="skip",
)
dpo_triplets.create_features_from_schema({
    "triplet_id": "string",
    "prompt": "string",
    "system_prompt": "string",
    "chosen": "string",
    "rejected": "string",
    "chosen_ref_logprob": "float64",
    "rejected_ref_logprob": "float64",
    "preference_margin": "float64",
    "source_pair_id": "string",
    "include_in_training": "bool",
    "split": "string",
}, if_exists="skip")

# --- Reward model training ---
reward_training = fs.create_feature_group(
    "reward-training-data",
    entity_key="example_id",
    if_exists="skip",
)
reward_training.create_features_from_schema({
    "example_id": "string",
    "prompt": "string",
    "response": "string",
    "reward_score": "float64",
    "mean_rating": "float64",
    "rating_std": "float64",
    "num_ratings": "int64",
    "bt_score": "float64",
    "model_id": "string",
    "include_in_training": "bool",
    "split": "string",
}, if_exists="skip")

# --- Eval ---
eval_outputs = fs.create_feature_group(
    "eval-outputs",
    entity_key="output_id",
    if_exists="skip",
)
eval_outputs.create_features_from_schema({
    "output_id": "string",
    "suite_name": "string",
    "example_id": "string",
    "model_a_id": "string",
    "model_b_id": "string",
    "response_a": "string",
    "response_b": "string",
    "preferred": "string",
    "preference_confidence": "float64",
    "annotator_id": "string",
}, if_exists="skip")

print("  ✓ sft-conversations, sft-image-pairs")
print("  ✓ preference-pairs, multimodal-preference-pairs")
print("  ✓ dpo-triplets, reward-training-data")
print("  ✓ eval-outputs")

# =============================================================================
# 3. SFT Data Curation
# =============================================================================

print("\n[STEP 2] SFT DATA CURATION")
print("-" * 50)

# SFT quality and compliance (stricter than pre-training)
sft_quality = QualityScorer(
    name="sft_quality",
    dimensions=[
        QualityDimension.RELEVANCE,
        QualityDimension.COMPLETENESS,
        QualityDimension.SAFETY,
        QualityDimension.FORMAT,
    ],
    thresholds=[
        QualityThreshold(QualityDimension.RELEVANCE,    min_score=0.7),
        QualityThreshold(QualityDimension.COMPLETENESS, min_score=0.6),
        QualityThreshold(QualityDimension.SAFETY,       min_score=0.9),
    ],
    model_uri="hf://acme/sft-quality-scorer-v2",
    input_columns=["turns_json"],
)

sft_compliance = ComplianceFilterTransform(
    name="sft_compliance",
    policy=CompliancePolicy(
        name="sft-strict",
        rules=[
            ComplianceRule(ComplianceFlag.NSFW,        threshold=0.5),
            ComplianceRule(ComplianceFlag.TOXIC,       threshold=0.4),
            ComplianceRule(ComplianceFlag.HATE_SPEECH, threshold=0.3),
            ComplianceRule(ComplianceFlag.PII, action=ComplianceAction.REDACT, threshold=0.5),
        ],
    ),
    input_columns=["turns_json"],
)

sft_curation_job = fs.create_job(
    name="sft_curation",
    sources=[FeatureGroupSource(
        feature_group="acme/posttraining/v2/sft-conversations",
        features=["example_id", "turns_json", "system_prompt", "language"],
    )],
    transform=sft_quality,
    target=Target(
        feature_group="sft-conversations",
        features={
            "quality_score": "quality_score",
            "compliance_passed": "compliance_passed",
            "include_in_training": "include_in_training",
        },
        write_mode="upsert",
        key_columns=["example_id"],
    ),
    schedule=Schedule.daily(hour=1),
    tags=["sft", "curation"],
)

print(f"  SFT curation job: {sft_curation_job.name}")
print(f"  Quality dimensions: {[d.value for d in sft_quality.dimensions]}")
print(f"  Safety threshold: {sft_quality.thresholds[2].min_score} (strict)")

# =============================================================================
# 4. Preference Collection: Human + LLM Judge Bootstrap
# =============================================================================

print("\n[STEP 3] PREFERENCE COLLECTION")
print("-" * 50)

# Human annotation task for high-stakes preference pairs
human_pref_task = HumanEvalTask(
    name="human-preference-v2",
    task_type=AnnotationTaskType.BINARY_PREFERENCE,
    instructions="""
    Compare two AI responses to the same prompt. Select the better one based on:
    1. Helpfulness: fully and accurately addresses the request
    2. Safety: no harmful, biased, or misleading content
    3. Honesty: factually correct, appropriate uncertainty

    Select 'Tie' only if both are genuinely equivalent.
    """.strip(),
    source_feature_group="acme/posttraining/v2/preference-pairs",
    source_columns=["prompt", "system_prompt", "response_a", "response_b"],
    label_column="preferred",
    options=["response_a", "response_b", "tie"],
    min_annotations=5,
    annotator_config=AnnotatorConfig(
        pool_type=AnnotatorPoolType.INTERNAL,
        min_annotations_per_item=5,
        annotator_qualifications=["passed-safety-training", "rlhf-certified"],
    ),
    agreement_metric=AgreementMetric.FLEISS_KAPPA,
    consensus_strategy="majority",
    tags=["rlhf", "gold-label"],
)

# LLM judge for high-volume bootstrap labeling
llm_pref_task = HumanEvalTask(
    name="llm-judge-preference-v2",
    task_type=AnnotationTaskType.BINARY_PREFERENCE,
    instructions="(LLM judge)",
    source_feature_group="acme/posttraining/v2/preference-pairs",
    source_columns=["prompt", "response_a", "response_b"],
    label_column="preferred",
    options=["response_a", "response_b", "tie"],
    min_annotations=1,
    annotator_config=AnnotatorConfig(
        pool_type=AnnotatorPoolType.MODEL,
        model_judge_uri="hf://acme/preference-judge-v2",
        model_judge_prompt_template=(
            "Prompt: {prompt}\n\nA: {response_a}\n\nB: {response_b}\n\n"
            "Which is better? Reply A, B, or TIE."
        ),
    ),
    tags=["llm-judge", "silver-label"],
)

# Annotation batches
batch_1 = AnnotationBatch(
    batch_id="pref_batch_2024_001",
    task_name="human-preference-v2",
    item_ids=[f"pair_{i:06d}" for i in range(1000)],
    status="in_progress",
)
batch_2 = AnnotationBatch(
    batch_id="pref_batch_2024_002",
    task_name="human-preference-v2",
    item_ids=[f"pair_{i:06d}" for i in range(1000, 2000)],
    status="pending",
)

print(f"  Human pref task: {human_pref_task.name}")
print(f"    Min annotations: {human_pref_task.min_annotations}")
print(f"    Pool: {human_pref_task.annotator_config.pool_type.value}")
print(f"  LLM judge: {llm_pref_task.name}")
print(f"    Model: {llm_pref_task.annotator_config.model_judge_uri}")
print(f"  Batch 1: {batch_1.total_items} items ({batch_1.status})")
print(f"  Batch 2: {batch_2.total_items} items ({batch_2.status})")

# =============================================================================
# 5. DPO Preparation
# =============================================================================

print("\n[STEP 4] DPO PREPARATION")
print("-" * 50)

# Filter preference pairs → DPO triplets
dpo_prep_job = fs.create_job(
    name="dpo_triplet_preparation",
    description="Convert gold preference pairs to DPO triplets",
    sources=[FeatureGroupSource(
        feature_group="acme/posttraining/v2/preference-pairs",
        features=["pair_id", "prompt", "system_prompt", "response_a", "response_b",
                  "preferred", "preference_confidence", "annotator_agreement", "is_gold"],
        filters=[
            "include_in_training == True",
            "preferred != 'tie'",
            "is_gold == True",             # gold labels only for DPO
            "annotator_agreement >= 0.75",
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
            CASE WHEN RAND() < 0.95 THEN 'train' ELSE 'validation' END AS split,
            True AS include_in_training
        FROM source
        WHERE preference_margin > 0.3   -- discard near-ties
        """,
    ),
    target=Target(
        feature_group="dpo-triplets",
        write_mode="overwrite",
    ),
    schedule=Schedule.manual(),
    tags=["dpo", "posttraining"],
)

# Reference logprob enrichment job (run against the current reference model)
logprob_job = fs.create_job(
    name="dpo_ref_logprob_enrichment",
    description="Compute reference model log probabilities for DPO training",
    sources=[FeatureGroupSource(
        feature_group="acme/posttraining/v2/dpo-triplets",
        features=["triplet_id", "prompt", "system_prompt", "chosen", "rejected"],
        filters=["chosen_ref_logprob IS NULL"],   # Only un-enriched triplets
    )],
    transform=None,   # Python transform calling reference model API
    target=Target(
        feature_group="dpo-triplets",
        features={
            "chosen_ref_logprob": "chosen_ref_logprob",
            "rejected_ref_logprob": "rejected_ref_logprob",
        },
        write_mode="upsert",
        key_columns=["triplet_id"],
    ),
    schedule=Schedule.manual(),
    tags=["dpo", "logprob"],
)

print(f"  DPO prep job: {dpo_prep_job.name}")
print(f"    Filters: gold-only, agreement≥0.75, margin>0.3")
print(f"  Logprob enrichment: {logprob_job.name}")
print(f"    Runs reference model over all unenriched triplets")

# =============================================================================
# 6. Dataset Versioning
# =============================================================================

print("\n[STEP 5] DATASET VERSIONING")
print("-" * 50)

sft_v2 = DatasetVersion(
    name="acme-sft-v2",
    version="sft-v2.0",
    feature_group="acme/posttraining/v2/sft-conversations",
    num_records=1_200_000,
    size_bytes=15 * 1024**3,
    applied_filters=["include_in_training == True"],
    tags=["sft", "multimodal"],
    metadata={"modalities": ["text", "image"]},
)

rlhf_v2 = DatasetVersion(
    name="acme-rlhf-v2",
    version="rlhf-v2.0",
    feature_group="acme/posttraining/v2/preference-pairs",
    num_records=200_000,
    size_bytes=3 * 1024**3,
    applied_filters=["include_in_training == True", "annotator_agreement >= 0.7"],
    tags=["rlhf", "gold-label"],
)

dpo_v1 = DatasetVersion(
    name="acme-dpo-v1",
    version="dpo-v1.0",
    feature_group="acme/posttraining/v2/dpo-triplets",
    num_records=120_000,
    size_bytes=1 * 1024**3,
    applied_filters=["include_in_training == True", "preference_margin > 0.3"],
    parent_version="rlhf-v2.0",   # DPO data derived from RLHF pairs
    tags=["dpo", "gold-label"],
)

print(f"  {sft_v2}")
print(f"  {rlhf_v2}")
print(f"  {dpo_v1}  (parent: {dpo_v1.parent_version})")

# =============================================================================
# 7. Evaluation Suite and Human Win Rate
# =============================================================================

print("\n[STEP 6] EVALUATION")
print("-" * 50)

# Build eval suite for post-training checkpoint gating
checkpoint_eval = EvalSuite(
    name="posttraining-checkpoint-eval",
    description="Gating eval run before each major model release",
    version="v2.0",
    metrics=[EvalMetric.WIN_RATE, EvalMetric.MEAN_OPINION_SCORE, EvalMetric.ACCURACY],
    tags=["checkpoint-gate", "mandatory"],
)

# Add a sample of held-out examples from each capability domain
for domain, count in [("instruction-following", 50), ("coding", 30),
                       ("math", 30), ("safety", 40), ("multimodal", 30)]:
    for i in range(count):
        checkpoint_eval.add_example(EvalExample(
            example_id=f"{domain}_{i:04d}",
            input={"prompt": f"[{domain} example {i}]", "domain": domain},
            ground_truth={"answer": f"[expected_{i}]"} if domain in ("math", "coding") else None,
            difficulty="medium",
            domain=domain,
        ))

print(f"  Eval suite: {checkpoint_eval}")
print(f"  Domains: {list({e.domain for e in checkpoint_eval.examples})}")

# Human win rate task on eval outputs
win_rate_task = HumanEvalTask(
    name="checkpoint-win-rate",
    task_type=AnnotationTaskType.BINARY_PREFERENCE,
    instructions="Compare the two responses and select the better one overall.",
    source_feature_group="acme/posttraining/v2/eval-outputs",
    source_columns=["prompt", "response_a", "response_b"],
    label_column="preferred",
    options=["response_a", "response_b", "tie"],
    min_annotations=3,
    annotator_config=AnnotatorConfig(
        pool_type=AnnotatorPoolType.HYBRID,
        min_annotations_per_item=3,
        model_judge_uri="hf://acme/preference-judge-v2",   # LLM pre-screen
    ),
    tags=["checkpoint", "win-rate", "release-gate"],
)

print(f"  Win rate task: {win_rate_task.name}")
print(f"  Pool: {win_rate_task.annotator_config.pool_type.value} (LLM pre-screen + human review)")

# Simulated eval results for two model checkpoints
checkpoint_a_result = EvalResult(
    suite_name="posttraining-checkpoint-eval",
    suite_version="v2.0",
    model_id="acme-v2-checkpoint-a",
    scores={
        "win_rate": 0.61,
        "mean_opinion_score": 4.12,
        "accuracy": 0.847,
    },
    num_evaluated=180,
    num_failed=2,
    human_win_rate=0.61,
)

checkpoint_b_result = EvalResult(
    suite_name="posttraining-checkpoint-eval",
    suite_version="v2.0",
    model_id="acme-v2-checkpoint-b",
    scores={
        "win_rate": 0.54,
        "mean_opinion_score": 3.98,
        "accuracy": 0.831,
    },
    num_evaluated=180,
    num_failed=1,
    human_win_rate=0.54,
)

print(f"\n  Eval results:")
print(f"    {checkpoint_a_result}")
print(f"    {checkpoint_b_result}")
print(f"  Selected checkpoint: acme-v2-checkpoint-a "
      f"(win_rate={checkpoint_a_result.scores['win_rate']:.0%} "
      f"> {checkpoint_b_result.scores['win_rate']:.0%})")

# =============================================================================
# 8. Pipeline Summary
# =============================================================================

print("\n" + "=" * 70)
print("PIPELINE SUMMARY")
print("=" * 70)

stages = [
    ("Stage 1: SFT Data Collection",
     "Gather conversations (text + image) from internal and partner sources"),
    ("Stage 2: SFT Curation",
     "QualityScorer (relevance, completeness, safety) + compliance redaction"),
    ("Stage 3: SFT Training",
     f"Train on {sft_v2.num_records:,} examples → SFT checkpoint"),
    ("Stage 4: Preference Collection",
     f"Human annotators ({human_pref_task.min_annotations} per pair) + LLM judge bootstrap"),
    ("Stage 5: DPO Preparation",
     "Filter gold pairs → DPO triplets → enrich with reference model log probs"),
    ("Stage 6: DPO/RLHF Training",
     "DPO on 120k triplets OR PPO with reward model trained on 200k pairs"),
    ("Stage 7: Evaluation",
     f"{len(checkpoint_eval)} eval examples, {win_rate_task.min_annotations} annotations/pair"),
    ("Stage 8: Checkpoint Gating",
     f"Release if win_rate > 0.55 AND safety score ≥ 0.9"),
]

for stage, description in stages:
    print(f"  {stage}")
    print(f"    {description}")

print(f"\nData artifacts produced:")
print(f"  {sft_v2}")
print(f"  {rlhf_v2}")
print(f"  {dpo_v1}")
print(f"  Eval suite: {checkpoint_eval.name}@{checkpoint_eval.version} ({len(checkpoint_eval)} examples)")

print("\n" + "=" * 70)
print("POST-TRAINING PIPELINE COMPLETE!")
print("=" * 70)
