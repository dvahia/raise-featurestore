# Raise — Design Principles

This document captures the design principles that govern the Raise API. Every principle here is derived from patterns already present in the codebase and examples. New APIs, types, and transforms must be consistent with these principles or explicitly justify a departure.

---

## Overarching Theme: Optimized for Researchers and Agentic Systems

Every design decision in Raise can be traced back to one of two primary users: **researchers** working interactively in notebooks, and **agentic systems** constructing and executing ML pipelines programmatically.

These users have different needs from the DevOps or ML-engineer users that most feature stores target, and those differences are load-bearing:

### For Researchers

Researchers iterate fast, re-run cells constantly, think in terms of expressions rather than infrastructure, and want analytics in the same environment as their model code. Every friction point that distinguishes Raise from Feast, SageMaker, or Databricks maps to reducing researcher overhead:

- `if_exists="skip"` everywhere — notebooks get re-run; creation must be safe to call again
- String types (`"float32[512]"`) — no imports, no ceremony, just describe what you want
- Inline `derived_from` SQL — researchers think in expressions, not ETL jobs
- Built-in analytics — researchers do not want to leave the notebook to check distributions
- Path syntax over config files — a researcher can describe a resource in one string

The COMPARISON.md makes this concrete: the same end-to-end task takes 15 lines in Raise and 40–60 in Feast or SageMaker. That gap is not an accident; it is the entire thesis.

### For Agentic Systems

An agent constructing API calls has strict requirements that differ from a human's but are equally well served by Raise's design:

- **Predictable, compact addressing**: an agent can construct `"acme/mlplatform/rec/user-signals"` from context. It cannot reliably construct an IAM role ARN, an S3 bucket URI, or a Databricks Unity Catalog path.
- **Idempotency**: an agent running in a retry loop must be able to call `create_feature_group(..., if_exists="skip")` without producing duplicate state or erroring out.
- **Declarative intent**: an agent describes *what* it wants, not *how* to execute it. `SQLTransform`, `QualityScorer`, and `DatasetMix` are all high-level intent objects — the system figures out the execution.
- **String-based types**: an agent can emit `"float32[512]"` without knowing which module to import.
- **Write-once versioning**: an agent does not need to track version state. Raise auto-increments and the agent can always say `if_exists="update"`.
- **LLM-as-judge built in**: `AnnotatorPoolType.MODEL` makes the agent itself an annotator without any external plumbing.
- **Low surface area**: fewer concepts means fewer ways to be wrong. Raise needs ~4 core concepts; SageMaker needs 8+ before writing the first feature.

This is not a coincidence. OpenAI's internal data tooling uses a Codex-based agent to enrich table metadata — that is an agent querying and writing to a feature/signal store. The gap identified in COMPARISON.md ("no unified registry, versioning, or declarative definitions for pre-training signals at labs") is exactly the gap an agentic pipeline hits first. SageMaker requires session management, role ARNs, and S3 configuration before a single feature can be declared. Raise requires one string.

### Implications for New API Design

When evaluating a proposed API addition, ask both questions:

1. **Researcher test**: Can a researcher write this in a notebook cell with minimal imports and no prior infrastructure setup?
2. **Agent test**: Can an LLM construct a syntactically and semantically valid call to this API using only the context of surrounding code — without needing to look up ARNs, session objects, or cloud-specific configuration?

If the answer to either is "no," the API should be redesigned before it is added.

---

## 1. Dual API Style: Declarative and Procedural

Raise supports two styles, and both are first-class:

**Declarative** — declare *what* you want; the system handles creation, validation, and idempotency.

```python
group.create_features_from_schema({
    "user_id": "string",
    "click_count": "int64",
    "embedding": "float32[512]",
}, if_exists="skip")

job = fs.create_job(
    name="daily_clicks",
    sources=[ObjectStorage(path="s3://events/")],
    transform=SQLTransform(sql="SELECT user_id, COUNT(*) AS clicks FROM source GROUP BY 1"),
    target=Target(feature_group="user-signals"),
    schedule=Schedule.daily(hour=2),
)
```

**Procedural** — control each step explicitly. Useful for debugging, dynamic logic, or incremental workflows.

```python
org = fs.create_organization("acme")
domain = org.create_domain("mlplatform")
project = domain.create_project("recommendation")
group = project.create_feature_group("user-signals", entity_key="user_id")
feature = group.create_feature("click_count", dtype="int64")
```

Both styles produce the same underlying objects. Neither is preferred over the other — choose based on the use case. Most production pipelines use declarative; most interactive notebook work mixes both.

---

## 2. Hierarchical Path Syntax

Resources live in a strict five-level hierarchy:

```
{org} / {domain} / {project} / {feature_group} / {feature} @ {version}
```

The path is the canonical identity of every resource. It can be passed as a string wherever an object is expected:

```python
# These are equivalent:
feature = group.feature("click_count")
feature = fs.feature("user-signals/click_count")
feature = fs.feature("acme/mlplatform/recommendation/user-signals/click_count")

# Cross-org reference (absolute path)
external = fs.feature("@partner/analytics/shared/pageviews/page_view_count")
```

**Rules:**
- Relative paths are resolved against the current `FeatureStore` context.
- Absolute paths start with `@org/` and are resolved globally.
- Version is appended with `@`: `click_count@v2`. Omitting `@version` always means latest active.
- Path parsing fails fast with a clear error; there is no implicit fallback.

---

## 3. Idempotency via `if_exists`

Every creation method accepts `if_exists` to control re-run behavior. This is essential for notebooks and pipeline restarts.

| Value | Behavior |
|-------|----------|
| `"error"` | Raise if the resource already exists *(default)* |
| `"skip"` | Return the existing resource unchanged |
| `"update"` | Create a new version of the existing resource |

```python
group.create_feature("click_count", dtype="int64", if_exists="skip")
fs.create_feature_group("user-signals", entity_key="user_id", if_exists="skip")
```

**`"error"` is the default** to catch bugs early — silent no-ops would mask mistakes. Opt into idempotency explicitly. For convenience, `get_or_create_*` wrappers are available as shorthand for `if_exists="skip"`.

---

## 4. Immutable Versioning

Features are versioned immutably. You cannot change `dtype`, `derived_from`, or any structural field of an existing feature. Schema evolution always produces a new version:

```python
# Immutable: must create a new version to change dtype
feature = group.create_feature("score", dtype="float32", if_exists="update")  # → v2

# Address by version
v1 = group.feature("score@v1")
v2 = group.feature("score@v2")
current = group.feature("score")  # → latest active version
```

Versions are auto-incremented (`v1`, `v2`, ...). There is no rollback — old versions are read-only history. Deprecation and archival are supported as status transitions that do not delete data.

This principle extends to `DatasetVersion`: dataset snapshots are write-once. `derive()` creates child versions with explicit parent lineage, not mutations of the parent.

---

## 5. Types Are Parseable Strings and Frozen Objects

Feature types can be specified as strings or typed objects; both parse into the same representation:

```python
# Equivalent
group.create_feature("clicks",     dtype="int64")
group.create_feature("clicks",     dtype=Int64())

group.create_feature("embedding",  dtype="float32[512]")   # Fixed-length embedding
group.create_feature("embedding",  dtype=Embedding(dim=512, dtype="float32"))

group.create_feature("tags",       dtype="string[]")        # Variable-length array
group.create_feature("images",     dtype=Array(BlobRef(content_types=["image/png"])))

group.create_feature("image_ref",  dtype=BlobRef(content_types=["image/jpeg", "image/webp"]))
```

Type objects are **frozen dataclasses** — immutable and hashable. This enables safe caching, set membership, and use as dict keys. No type can be mutated in-place; update() returns a new object.

The string syntax covers all common cases. Use typed objects only when you need to express constraints (content types, max array length) that the string syntax cannot encode.

---

## 6. BlobRef: References, Not Bytes

Multimodal data (images, audio, video) is stored as **references** to assets in object storage, never as inline bytes in the feature store. A `BlobReference` is an immutable record of where content lives and how to verify it:

```python
ref = registry.register(
    uri="s3://media/images/001.png",
    content_type=ContentType.IMAGE_PNG,
    checksum="sha256:abc123...",
    size_bytes=2_048_000,
)
```

**Why references:**
- Feature tables stay small; blobs stay in object storage.
- The same asset can be referenced from multiple feature groups without copying.
- Integrity is validated by the registry (on write, on read, or lazily), not by callers.
- Transforms receive references; they dereference only when they need the bytes.

**Time ranges for clips:** Video and audio assets support sub-clip references via `TimeRange`:

```python
clip_ref = full_video_ref.clip(start_sec=30.0, end_sec=90.0)
# clip_ref.uri == full_video_ref.uri (same file)
# clip_ref.time_range == TimeRange(start_sec=30.0, end_sec=90.0)
```

`TimeRange` is also a frozen dataclass. Clip references do not copy bytes; they annotate the parent URI with a time window.

---

## 7. Entity Key for Serving

A feature group can declare one `entity_key` — the column used for serving-time point lookups:

```python
group = fs.create_feature_group(
    "user-signals",
    entity_key="user_id",
    entity_dtype="string",
)

# Online serving
rows = group.get(["user_001", "user_002"], features=["click_count", "embedding"])
```

**Rules:**
- `entity_key` is a single column name (no composite keys).
- `.get()` requires `entity_key` to be set; raises `ValueError` otherwise.
- Entity keys are for online serving, not for ingestion or transformation (Jobs use Sources and SQL).
- One entity key per group. Multiple lookup strategies require multiple groups.

---

## 8. The Job Is the Unit of Pipeline Composition

A `Job` combines four orthogonal declarations into one deployable unit:

```
Job = Sources + Transform + Target + Schedule
```

```python
job = fs.create_job(
    name="daily_clicks",
    sources=[FeatureGroupSource(feature_group="raw-events", features=["user_id", "event_time"])],
    transform=SQLTransform(sql="SELECT user_id, COUNT(*) AS clicks FROM source GROUP BY 1"),
    target=Target(
        feature_group="user-signals",
        features={"clicks": "click_count"},
        write_mode="upsert",
        key_columns=["user_id"],
    ),
    schedule=Schedule.daily(hour=2),
    incremental=IncrementalConfig.incremental("event_time"),
)
```

**Constraints that follow from this design:**
- One target per job. Fan-out (writing to multiple feature groups) requires multiple jobs.
- Chaining is explicit: the output feature group of one job is declared as the input source of the next.
- SQL and Python transforms are separate types. `SQLTransform` takes a SQL string; `PythonTransform` (via `@python_transform`) takes a Python function. Combine them with a `HybridTransform` only when necessary.

**Write modes:**

| Mode | Behavior |
|------|----------|
| `"append"` | Add rows; may produce duplicates across runs |
| `"overwrite"` | Replace partition or table |
| `"upsert"` | Merge on `key_columns`; requires `key_columns` |

---

## 9. Incremental Processing via Checkpoints

Jobs support incremental execution where a checkpoint column (typically a timestamp) tracks the watermark of processed data:

```python
IncrementalConfig.incremental(
    checkpoint_column="event_time",
    lookback="1h",   # re-process last 1h to catch late arrivals
)
```

The checkpoint value is injected into SQL as `{{checkpoint}}` and into Python transforms via `context.checkpoint_value`. After a successful run, the checkpoint advances to the max value seen in that run.

**Reset:** `job.reset_checkpoint()` forces a full refresh on the next run. This is the mechanism for backfill, not a separate backfill API.

**One checkpoint per job.** If you need to track multiple watermarks (e.g., multiple source tables), split into multiple jobs.

---

## 10. Curation Transforms as First-Class Pipeline Stages

Data curation — quality scoring, near-deduplication, compliance filtering — is expressed as composable `Transform` subclasses, not as ad-hoc scripts. Each transform is a deployable, versioned, lineage-tracked unit:

```python
pipeline = CurationPipeline(
    name="web-text-curation-v3",
    steps=[
        QualityScorer(
            name="text_quality",
            dimensions=[QualityDimension.FLUENCY, QualityDimension.COHERENCE],
            thresholds=[QualityThreshold(QualityDimension.FLUENCY, min_score=0.5)],
            model_uri="hf://acme/text-quality-scorer-v3",
            input_columns=["text"],
        ),
        DeduplicationTransform(
            name="minhash_dedup",
            config=DeduplicationConfig(
                algorithm=DeduplicationAlgorithm.MINHASH,
                threshold=0.80,
                key_columns=["text"],
                action="flag",
            ),
        ),
        ComplianceFilterTransform(
            name="compliance",
            policy=compliance_policy,
            input_columns=["text", "url"],
        ),
    ],
)
```

Each step writes **annotation columns** alongside the source data. It does not delete records (unless `action="remove"`); it flags them with `is_duplicate`, `compliance_passed`, `quality_score`, etc. The final disposition (`include_in_training`) is a downstream SQL step that combines all flags.

`CurationPipeline` is **not** a `Transform` subclass — it is a container that the orchestrator unwraps into chained Jobs. This keeps the pipeline inspectable (`pipeline.all_output_columns`) without inheriting Job scheduling complexity.

---

## 11. Dataset Mixing Is Declared, Not Scripted

Multi-source mixing is a first-class declarative object, not a shell script or Spark job written by hand:

```python
mix = DatasetMix(
    name="foundation-v4-mix",
    sources=[
        MixSource(feature_group="curated-text",       weight=0.60, filters=["language == 'en'"]),
        MixSource(feature_group="curated-image-text", weight=0.20),
        MixSource(feature_group="raw-video-text",     weight=0.10),
    ],
    strategy=MixingStrategy.TEMPERATURE,
    temperature=0.7,
    total_samples=5_000_000_000,
)
```

**Mixing strategies:**
- `WEIGHTED` — sample proportional to declared weights.
- `TEMPERATURE` — apply `p_i ∝ w_i^(1/T)`: T < 1 amplifies the dominant source; T > 1 smooths toward uniform.
- `UNIFORM` — ignore weights; equal probability.
- `PROPORTIONAL` — weight by actual dataset size (no upsampling).

`mix.effective_weights()` always returns the true sampling probabilities after strategy is applied — what gets sampled is explicit, not inferred.

---

## 12. Dataset Versioning Is Write-Once with Explicit Lineage

A `DatasetVersion` is an immutable snapshot. Once created, it cannot be updated:

```python
v1 = DatasetVersion(
    name="acme-foundation",
    version="v1.0.0",
    feature_group="acme/pretraining/curated-text",
    num_records=2_000_000_000,
    size_bytes=8_000 * 1024**3,
    applied_filters=["include_in_training == True"],
    provenance=DatasetProvenance(source_name="CommonCrawl-2024", license="CC0", ...),
    tags=["pretraining", "text"],
)

# Derive a child version (parent_version set automatically)
v2 = v1.derive(
    version="v2.0.0",
    num_records=5_000_000_000,
    applied_filters=["include_in_training == True", "quality_score >= 0.5"],
)
```

`derive()` records explicit parent–child lineage. Lineage is queryable and appears in the lineage graph alongside feature-level lineage. This means a training checkpoint can trace back to the exact dataset version, which traces back to the raw crawl and curation pipeline that produced it.

---

## 13. Annotation and Evaluation Are Pipeline Citizens

Human annotation tasks and model evaluations are not one-off scripts — they are declared, versioned, and queryable entities:

```python
task = HumanEvalTask(
    name="response-preference",
    task_type=AnnotationTaskType.BINARY_PREFERENCE,
    source_feature_group="acme/posttraining/rlhf/preference-pairs",
    source_columns=["prompt", "response_a", "response_b"],
    label_column="preferred",
    options=["response_a", "response_b", "tie"],
    min_annotations=5,
    annotator_config=AnnotatorConfig(pool_type=AnnotatorPoolType.INTERNAL),
    agreement_metric=AgreementMetric.FLEISS_KAPPA,
)

suite = EvalSuite(name="safety-v2", version="2.0", eval_type=EvalType.SAFETY)
result = EvalResult(suite_name="safety-v2", suite_version="2.0", model_id="claude-3-opus")
```

The LLM-as-judge pattern is a first-class `AnnotatorPoolType.MODEL` option — not an external script that writes results back by hand.

`EvalResult` objects stored in a feature group enable historical tracking of model quality across checkpoints, making eval gating (block a new checkpoint if it regresses on a key eval) a standard query rather than custom reporting.

---

## 14. ACL Inheritance Down the Hierarchy

Access control cascades from organization → domain → project → feature group → feature:

```python
# Set at org level
org.set_acl(ACL(readers=["team-data"], writers=["admin"]))

# Override at project level (breaks inheritance from above)
project.set_acl(ACL(readers=["team-rec"], writers=["admin"], inherit=False))

# Merge at group level (layered on top of inherited ACL)
group.set_acl(ACL(readers=["team-rec-read-only"], inherit=True))

# Effective ACL merges all levels
effective = group.get_effective_acl()
chain = group.get_acl_chain()  # [org_acl, domain_acl, project_acl, group_acl]
```

`inherit=True` (default) merges with the parent's effective ACL. `inherit=False` breaks the chain and makes the resource's ACL standalone. Cross-organization access is granted via `ExternalGrant` on a feature group, which is a separate, audited operation.

---

## 15. Audit Logging Is Orthogonal and Append-Only

Every resource carries an `audit_log()` method. Logging is transparent — callers never write audit events explicitly; the system does.

```python
logs = feature.audit_log(actions=["READ", "WRITE"], since=datetime(2025, 1, 1))
logs = group.audit_log(category="schema")
logs = job.audit_log(actions=["RUN", "FAILURE"])
```

Audit logs are **append-only and immutable**. There is no purge API. Export is supported for compliance reporting. This is intentional: the audit trail must be trustworthy.

---

## 16. Quality Checks Are Declarative, Not Procedural

Validation logic belongs in the job definition, not in transform code:

```python
job = fs.create_job(
    ...,
    quality_checks=[
        NullCheck(column="user_id",      allow_nulls=False,  severity=CheckSeverity.ERROR),
        RangeCheck(column="score",       min_value=0.0, max_value=1.0, severity=CheckSeverity.WARNING),
        RowCountCheck(min_rows=1_000,    max_rows=10_000_000, severity=CheckSeverity.WARNING),
        BlobIntegrityCheck(columns=["image_ref"], verify_existence=True, severity=CheckSeverity.ERROR),
    ],
)
```

Checks run after the transform and before the write. `ERROR` severity fails the job run; `WARNING` logs and continues; `INFO` records as a metric only. Checks are synchronous and run on all data (no sampling).

---

## 17. Frozen Dataclasses for Core Value Types

Types that are meant to be values — not mutable objects with identity — are `@dataclass(frozen=True)`:

- `FeatureType` and all subtypes (`Int64`, `Float32`, `Embedding`, `BlobRef`, ...)
- `BlobReference` (the runtime reference instance)
- `TimeRange`
- `DatasetProvenance`
- `QualityThreshold`, `DeduplicationConfig`, `ComplianceRule`

This makes them hashable, safe for use as dict keys, and eliminates a class of mutation bugs. When a "mutation" is needed, return a new instance. `BlobReference.clip()` and `BlobReference.with_metadata()` both return new `BlobReference` instances rather than modifying in-place.

Entities with identity (Feature, FeatureGroup, Job) are mutable for metadata (`description`, `tags`, `owner`) but not for structural fields (`dtype`, `derived_from`). Structural changes produce a new version.

---

## 18. Batch-Oriented by Default; Serving Is Explicit

The core execution model is batch:

- **Jobs** are the unit of data movement (scheduled or manual).
- **Analyses** are point-in-time computations.
- **Live tables** are batch refreshes triggered by CDC events or schedules.

There is no streaming ingestion API. Incremental processing (section 9) simulates near-real-time with short batch intervals; it is still batch.

**Serving** (online, low-latency) is an explicit, separate concern accessed via `.get()` on a feature group with an entity key. The separation between the write path (Jobs → columnar storage) and the read path (`.get()` → serving store) is intentional and load-bearing — do not conflate them.

---

## Summary

**R** = primarily serves researchers · **A** = primarily serves agentic systems · **R+A** = both

| Principle | Key Signal in Code | Who Benefits |
|-----------|-------------------|----|
| Dual API style | `create_features_from_schema()` vs. `create_feature()` | R+A |
| Hierarchical path syntax | `"acme/mlplatform/rec/user-signals/clicks@v2"` | A |
| Idempotency via `if_exists` | Every `create_*` method | R+A |
| Immutable versioning | `feature.update()` returns new version; `DatasetVersion.derive()` | A |
| Types as parseable strings | `dtype="float32[512]"` → `Embedding(dim=512)` | R+A |
| BlobRef: references, not bytes | `registry.register()`, `BlobReference.clip()` | R+A |
| Entity key for serving | `entity_key=` on `create_feature_group`; `.get()` | R |
| Job as composition unit | `Sources + Transform + Target + Schedule` | R+A |
| Checkpointed incremental processing | `IncrementalConfig.incremental("event_time")` | A |
| Curation as transform pipeline | `CurationPipeline([QualityScorer, DeduplicationTransform, ...])` | R+A |
| Mixing is declared | `DatasetMix(sources=[...], strategy=MixingStrategy.TEMPERATURE)` | R+A |
| Dataset versions are write-once | `DatasetVersion`; `derive()` for lineage | A |
| Annotation & eval as pipeline objects | `HumanEvalTask`, `EvalSuite`, `EvalResult` | R+A |
| ACL inheritance | `inherit=True/False` on `ACL` | R |
| Audit logging is orthogonal | `feature.audit_log(...)` | R+A |
| Quality checks are declarative | `quality_checks=[NullCheck(...)]` on Job | R+A |
| Frozen value types | `@dataclass(frozen=True)` on all type objects | A |
| Batch-first; serving is explicit | `.get()` requires `entity_key`; no streaming | R |

Notice that most principles serve both audiences. The few that serve only researchers (entity key serving, ACL inheritance, serving separation) are about interactivity and governance — concerns that arise when humans are in the loop. The few that serve primarily agents (path syntax, write-once versions, frozen types) are about predictability and statelessness — concerns that arise when code is in the loop. The fact that the same API serves both without contradiction is the point.
