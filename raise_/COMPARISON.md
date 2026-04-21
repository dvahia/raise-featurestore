# Raise — Comparative Analysis

A comparative analysis of Raise against tools researchers and agentic ML systems actually use: data management platforms, distributed compute frameworks, serverless execution environments, LLM pipeline orchestrators, and vector databases.

---

## How to Read This Comparison

The tools compared here are not all competitors in the same category. They occupy different layers of the ML stack:

| Tool | Layer | Primary Job |
|------|-------|-------------|
| **Raise** | Data Management | Declare, version, curate, and serve ML features and training datasets |
| **Databricks** | Data Platform | Unified lakehouse: compute + storage + governance at scale |
| **Ray** | Distributed Compute | Scale Python workloads across clusters; training orchestration |
| **Modal** | Serverless Execution | Run GPU/CPU Python functions in the cloud with zero infra setup |
| **Haystack** | LLM Pipeline Orchestration | Compose and run LLM inference pipelines (RAG, agents, QA) |
| **Milvus** | Vector Database | Store, index, and query dense embeddings at scale |

Raise's relationship with each tool is different: Databricks is the closest direct overlap (both do data management); Ray and Modal are **what runs Raise's compute jobs** in production; Haystack and Milvus are **downstream consumers** of features Raise produces.

---

## Executive Summary

### Data Management (direct comparison)

| Dimension | Raise | Databricks |
|-----------|-------|------------|
| **Ease of Use** | High | Medium |
| **Developer Velocity** | High | Medium |
| **Declarative API** | Yes | Partial |
| **Notebook-First** | Yes | Yes |
| **Derived Features** | Inline SQL (`derived_from=`) | Spark SQL (separate table) |
| **Pre-training Signal Management** | Yes (native) | No |
| **Curation Pipelines** | Yes (native) | DIY |
| **Dataset Versioning** | Yes (native) | Delta time-travel |
| **Human Annotation Workflows** | Yes (native) | No |
| **Lines of Code** | Low | Medium |
| **Learning Curve** | Low | Medium |
| **Production Scale** | Design prototype | Battle-tested |

### Compute & Execution

| Dimension | Raise | Ray | Modal |
|-----------|-------|-----|-------|
| **Role vs. Raise** | Data management | Runs Raise's compute jobs | Runs Raise's compute jobs |
| **API Style** | Declarative | Imperative (distributed Python) | Decorator-based |
| **Infra Setup** | None (API only) | Cluster required | None (fully managed) |
| **GPU Support** | Declared in Job spec | Native (Actor, remote tasks) | Native (`@modal.function(gpu=...)`) |
| **Scheduling** | Built-in (`Schedule.daily()`) | External (cron, Airflow) | Built-in triggers |
| **Data Lineage** | Yes | No | No |
| **Feature Versioning** | Yes | No | No |

### Downstream Consumers

| Dimension | Raise | Haystack | Milvus |
|-----------|-------|----------|--------|
| **Role vs. Raise** | Produces features/embeddings | Consumes features for LLM pipelines | Stores and retrieves embeddings |
| **Pipeline Type** | Training data pipelines | Inference/retrieval pipelines | Similarity search |
| **Declarative API** | Yes | Yes (YAML + Python) | No (SDK imperative) |
| **Versioning** | Yes (DatasetVersion) | No | Collections only |
| **Multimodal** | Yes (text, image, video, audio) | Text + image (partial) | Vectors + metadata |
| **Curation / Quality** | Yes (native) | No | No |

---

## 1. Databricks Feature Engineering

### What It Is

Databricks Feature Engineering (formerly Feature Store) is the feature management layer of the Databricks Lakehouse. It is built on top of Delta Lake and Unity Catalog and is best suited for teams already inside the Databricks ecosystem running large-scale Spark workloads.

### Feature Definition

#### Raise
```python
fs = FeatureStore("acme/mlplatform/recommendation")
group = fs.create_feature_group("user-signals", entity_key="user_id")

group.create_features_from_schema({
    "clicks": "int64",
    "impressions": "int64",
    "user_embedding": "float32[512]",
})
group.create_feature("ctr", dtype="float64", derived_from="clicks / NULLIF(impressions, 0)")
```
**Concepts**: 3 (FeatureStore, FeatureGroup, Feature) · **Lines**: 8

#### Databricks
```python
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# Requires a Spark DataFrame to already exist
df = spark.read.table("acme.mlplatform.raw_events")

# Derived features must be computed before table creation
df = df.withColumn("ctr", df.clicks / df.impressions)

fe.create_table(
    name="acme.mlplatform.user_signals",
    primary_keys=["user_id"],
    df=df,
    description="User engagement signals",
)
```
**Concepts**: 5 (FeatureEngineeringClient, Spark session, DataFrame, Unity Catalog path, primary_keys) · **Lines**: 12

**Key difference**: Databricks requires live data to create a table; Raise lets you declare schema without data. Derived features in Databricks require pre-computing in Spark and writing a new table. Raise's `derived_from` is a first-class inline declaration.

### Pre-Training Signal Management

Databricks has no native concept of curation pipelines, compliance filtering, or dataset mixing for pre-training. The equivalent requires:

```python
# Databricks: quality scoring requires a custom UDF + Spark job
@udf("float")
def quality_score(text):
    # custom scorer
    ...

df = spark.read.table("acme.pretraining.raw_text") \
    .withColumn("quality_score", quality_score(col("text"))) \
    .withColumn("is_duplicate", ...)  # separate dedup pass
    .filter(col("quality_score") >= 0.5)

df.write.format("delta").saveAsTable("acme.pretraining.curated_text")
```

The equivalent in Raise is a `CurationPipeline` with `QualityScorer`, `DeduplicationTransform`, and `ComplianceFilterTransform` — each versioned, lineage-tracked, and orchestrated automatically.

### Where Databricks Wins

- **Production scale**: Delta Lake and Unity Catalog are battle-tested at petabyte scale
- **Ecosystem**: Native Spark, MLflow, dbt, and Airflow integration
- **Governance**: Unity Catalog provides row-level security, column masking, data lineage
- **Time-travel**: Delta's snapshot history is a form of dataset versioning

### Where Raise Wins

- **Researcher ergonomics**: Schema declaration without live data, inline derived features, fewer concepts
- **Pre-training native**: Curation pipelines, compliance filtering, dataset mixing, and dataset versioning are first-class
- **Agent-legible**: Path syntax and declarative intent are constructible by an LLM; Spark DataFrame APIs are not

---

## 2. Ray

### What It Is

Ray is a distributed Python execution framework, not a data management tool. It is what actually runs the compute behind feature computation, model training, and batch inference at scale. OpenAI uses Ray extensively for training orchestration.

Ray and Raise are **complementary, not competing**: Raise declares *what* should happen; Ray is one implementation of *how* to execute it.

### Computing Features in Ray vs. Raise

#### Ray — imperative distributed compute
```python
import ray
from ray.data import read_parquet

ray.init()

# Load raw data
ds = read_parquet("s3://acme/raw-text/")

# Distributed quality scoring (custom)
def score_quality(batch):
    import torch
    model = load_quality_model()  # loaded per worker
    batch["quality_score"] = model.score(batch["text"])
    return batch

# Distributed dedup (custom)
def compute_minhash(batch):
    from datasketch import MinHash
    # ... custom implementation
    return batch

ds = ds.map_batches(score_quality,  batch_size=512, num_gpus=1)
ds = ds.map_batches(compute_minhash, batch_size=1024)
ds = ds.filter(lambda row: row["quality_score"] >= 0.5)
ds.write_parquet("s3://acme/curated-text/")
```

No versioning. No lineage. No schema registration. No scheduling. Reruns overwrite by default.

#### Raise — declarative, tracked
```python
pipeline = CurationPipeline(
    name="text-curation-v3",
    steps=[
        QualityScorer(
            name="text_quality",
            dimensions=[QualityDimension.FLUENCY, QualityDimension.COHERENCE],
            thresholds=[QualityThreshold(QualityDimension.FLUENCY, min_score=0.5)],
            model_uri="hf://acme/quality-scorer-v3",
            input_columns=["text"],
        ),
        DeduplicationTransform(
            name="minhash_dedup",
            config=DeduplicationConfig(algorithm=DeduplicationAlgorithm.MINHASH, threshold=0.80),
        ),
    ],
)

job = fs.create_job(
    name="curate_text_v3",
    sources=[FeatureGroupSource(feature_group="raw-text")],
    transform=pipeline.steps[0],
    target=Target(feature_group="curated-text", write_mode="upsert"),
    schedule=Schedule.daily(hour=2),
    incremental=IncrementalConfig.incremental("crawl_timestamp"),
)
```

The backend could execute this job on Ray. Raise is the declaration layer; Ray is the execution layer.

### Scheduling and Orchestration

| Capability | Ray | Raise |
|------------|-----|-------|
| Distributed execution | Yes (native) | Declared; backend-dependent |
| GPU resource management | Yes (Actor, remote) | Declared in Job spec |
| Scheduling | No (needs Airflow/cron) | Built-in (`Schedule.daily()`, etc.) |
| Incremental checkpointing | No | Built-in (`IncrementalConfig`) |
| Lineage tracking | No | Yes |
| Feature versioning | No | Yes |
| Data quality checks | No | Built-in (`quality_checks=`) |

### Where Ray Wins

- **Execution performance**: purpose-built for distributed Python with minimal overhead
- **Flexibility**: any arbitrary Python code, any data format
- **Training integration**: Ray Train, Ray Tune, Ray Serve form a complete ML platform
- **Scale**: proven at OpenAI-scale training workloads

### Where Raise Wins

- **Data management**: versioning, lineage, schema registration, serving — Ray has none of this
- **Declarative curation**: a researcher can compose a curation pipeline in 10 lines; Ray requires implementing every stage from scratch
- **Agent-constructible**: an LLM can write a Raise job definition; Ray distributed code requires deep framework knowledge

### Integration Opportunity

Raise job execution backends should be pluggable. A `RayBackend` would execute `Job` definitions as Ray Data pipelines, combining Raise's declarative management with Ray's execution performance:

```python
fs = FeatureStore("acme/pretraining", execution_backend=RayBackend(cluster="ray://head:10001"))
```

---

## 3. Modal

### What It Is

Modal is a serverless cloud platform for running Python workloads — particularly GPU inference, batch processing, and scheduled jobs — with zero infrastructure configuration. Like Ray, it is an execution layer, not a data management layer.

### Defining a GPU Inference Job

#### Modal — great for one-off execution
```python
import modal

app = modal.App("quality-scorer")

@app.function(
    gpu="A10G",
    image=modal.Image.debian_slim().pip_install("transformers", "torch"),
    schedule=modal.Cron("0 2 * * *"),  # daily at 2am
)
def score_quality():
    import boto3
    import pandas as pd
    from transformers import pipeline

    scorer = pipeline("text-classification", model="acme/quality-scorer-v3")

    df = pd.read_parquet("s3://acme/raw-text/latest.parquet")
    df["quality_score"] = [r["score"] for r in scorer(df["text"].tolist(), batch_size=256)]
    df.to_parquet("s3://acme/curated-text/latest.parquet")

if __name__ == "__main__":
    score_quality.remote()
```

No versioning. No lineage. No schema registration. Output path is hardcoded. Reruns overwrite.

#### Raise — the same intent, with management
```python
job = fs.create_job(
    name="quality_scoring",
    sources=[FeatureGroupSource(feature_group="raw-text", features=["doc_id", "text"])],
    transform=QualityScorer(
        name="text_quality",
        model_uri="hf://acme/quality-scorer-v3",
        dimensions=[QualityDimension.FLUENCY],
        input_columns=["text"],
    ),
    target=Target(feature_group="curated-text", write_mode="upsert", key_columns=["doc_id"]),
    schedule=Schedule.daily(hour=2),
    incremental=IncrementalConfig.incremental("crawl_timestamp"),
)
```

The backend could dispatch this to Modal. Raise manages what gets computed, versioned, and tracked; Modal runs the GPU container.

### API Ergonomics for Researchers

Modal's `@app.function` decorator is genuinely researcher-friendly: one decorator, no YAML, no Kubernetes config. It is arguably the best execution experience available today for ad-hoc GPU work.

| Dimension | Modal | Raise |
|-----------|-------|-------|
| Time to first GPU execution | ~2 minutes | N/A (no execution layer) |
| Infrastructure config | Zero | N/A |
| Feature versioning | No | Yes |
| Lineage tracking | No | Yes |
| Incremental processing | Manual | Built-in |
| Data quality checks | No | Built-in |
| Scheduling | Cron string | Rich schedule objects |
| Schema registration | No | Yes |

### Where Modal Wins

- **Zero-infra GPU execution**: by far the lowest friction path to running a GPU workload
- **Execution isolation**: each function runs in its own container; dependencies are per-function
- **Secrets management**: built-in, per-function secrets
- **Cost efficiency**: pay-per-second GPU billing

### Where Raise Wins

- **Everything after execution**: what to do with the results — register them, version them, track lineage, serve them — is Raise's entire job
- **Reproducibility**: Modal runs produce outputs at hardcoded paths; Raise produces versioned, queryable artifacts

### Integration Opportunity

A `ModalBackend` for Raise would let job definitions execute as Modal functions, inheriting Modal's zero-infra GPU access while Raise handles all the data management:

```python
fs = FeatureStore("acme/pretraining", execution_backend=ModalBackend(app="raise-jobs"))
```

---

## 4. Haystack

### What It Is

Haystack (by deepset) is an LLM pipeline orchestration framework for building **inference-time** AI applications: RAG systems, document QA, agentic pipelines, and search. It is fundamentally about routing inputs through a sequence of components at serving time.

Haystack and Raise look superficially similar — both have a "pipeline" concept and both involve multimodal data — but they operate at opposite ends of the ML lifecycle:

| Aspect | Raise | Haystack |
|--------|-------|----------|
| **When it runs** | Training data preparation (offline batch) | Inference / application serving (online) |
| **What flows through** | Feature rows, training samples, curation annotations | User queries, documents, LLM responses |
| **Output** | Versioned dataset for training | Answer, generated text, retrieved documents |
| **Persistence** | Feature groups, dataset versions | Typically stateless per request |

### Pipeline Definition

#### Haystack — inference pipeline
```python
from haystack import Pipeline
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.document_stores.in_memory import InMemoryDocumentStore

document_store = InMemoryDocumentStore()

# Build a RAG pipeline
pipe = Pipeline()
pipe.add_component("retriever", InMemoryBM25Retriever(document_store=document_store))
pipe.add_component("prompt_builder", PromptBuilder(template="""
    Context: {% for doc in documents %}{{ doc.content }}{% endfor %}
    Question: {{ question }}
    Answer:
"""))
pipe.add_component("llm", OpenAIGenerator(model="gpt-4o-mini"))

pipe.connect("retriever", "prompt_builder.documents")
pipe.connect("prompt_builder", "llm")

# Run at query time
result = pipe.run({"retriever": {"query": "What is a feature store?"}})
```

#### Raise — training data pipeline (different problem entirely)
```python
pipeline = CurationPipeline(
    name="pretraining-curation",
    steps=[
        QualityScorer(name="quality", dimensions=[QualityDimension.FLUENCY], ...),
        DeduplicationTransform(name="dedup", config=DeduplicationConfig(...)),
        ComplianceFilterTransform(name="compliance", policy=policy),
    ],
)

job = fs.create_job(
    name="curate_documents",
    sources=[FeatureGroupSource(feature_group="raw-text")],
    transform=pipeline.steps[0],
    target=Target(feature_group="curated-text", write_mode="upsert"),
    schedule=Schedule.daily(hour=2),
)
```

These pipelines solve different problems. Haystack answers a user query; Raise prepares the data that trains the model answering that query.

### Where They Connect

The natural integration point: **Raise produces the training data; Haystack consumes the trained model**. More concretely, a RAG system using Haystack retrieves from a document store that was populated by a Raise pipeline:

```
Raise pipeline:
  raw-text → [QualityScorer + Dedup + Compliance] → curated-text → embeddings → Milvus

Haystack pipeline (at serving time):
  user query → Milvus retriever → prompt builder → LLM → answer
```

### Where Haystack Wins

- **Inference pipeline composition**: rich component library (40+ retrievers, generators, routers)
- **Agent support**: native tool-calling, multi-agent coordination
- **Rapid prototyping of LLM applications**: minutes to a working RAG demo

### Where Raise Wins

- **Training data management**: Haystack has no concept of feature versioning, lineage, curation, or dataset mixing
- **Data quality**: Haystack doesn't score, deduplicate, or filter training data — it consumes the output of that process
- **Reproducibility**: Haystack pipelines are stateless at request time; Raise produces versioned artifacts

---

## 5. Milvus

### What It Is

Milvus is a purpose-built vector database for storing, indexing, and querying dense embeddings at scale. It is the primary storage and retrieval layer for RAG systems and semantic search applications.

Milvus can be thought of as a highly specialized feature store for one feature type: floating-point embedding vectors. Where Raise is general-purpose (any feature type, full data management lifecycle), Milvus is purpose-built for fast ANN (approximate nearest neighbor) search over embeddings.

### Storing and Retrieving Embeddings

#### Milvus — insert and search
```python
from pymilvus import MilvusClient, DataType

client = MilvusClient("http://localhost:19530")

# Create collection (schema must be defined upfront)
schema = client.create_schema()
schema.add_field("doc_id",    DataType.VARCHAR, max_length=64, is_primary=True)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1536)
schema.add_field("text",      DataType.VARCHAR, max_length=65535)

index_params = client.prepare_index_params()
index_params.add_index("embedding", metric_type="COSINE", index_type="HNSW",
                        params={"M": 16, "efConstruction": 200})

client.create_collection("documents", schema=schema, index_params=index_params)

# Insert vectors
client.insert("documents", [
    {"doc_id": "doc_001", "embedding": [0.1, 0.2, ...], "text": "..."},
    ...
])

# ANN search
results = client.search(
    "documents",
    data=[[0.15, 0.22, ...]],   # query vector
    anns_field="embedding",
    limit=10,
    output_fields=["doc_id", "text"],
)
```

No curation. No lineage. No dataset versioning. No mixing. No quality scoring. No compliance filtering. The schema must be fully defined before any data is inserted.

#### Raise — managing the embeddings as features (different concern)
```python
# Embeddings are a feature type in Raise; generation is a Job
embed_group = fs.create_feature_group(
    "document-embeddings",
    entity_key="doc_id",
)
embed_group.create_features_from_schema({
    "doc_id":     "string",
    "embedding":  "float32[1536]",
    "embed_model": "string",   # which model generated this
    "embed_version": "string", # version of the embedding pipeline
})

# Embedding generation is an InferenceTransform Job
embed_job = fs.create_job(
    name="generate_embeddings",
    sources=[FeatureGroupSource(feature_group="curated-text", features=["doc_id", "text"])],
    transform=embedding_inference(
        model_uri="hf://openai/text-embedding-3-large",
        input_column="text",
        output_column="embedding",
        batch_size=256,
        gpu_type=GPUType.NVIDIA_A100,
    ),
    target=Target(
        feature_group="document-embeddings",
        write_mode="upsert",
        key_columns=["doc_id"],
    ),
    schedule=Schedule.daily(hour=4),
    incremental=IncrementalConfig.incremental("crawl_timestamp"),
)
```

Raise registers, versions, and tracks lineage for the embedding pipeline. Milvus handles the ANN index for retrieval. They are complementary.

### Feature Comparison

| Dimension | Raise | Milvus |
|-----------|-------|--------|
| **Data types** | Any (scalar, embedding, blob, struct) | Vectors + scalar metadata |
| **ANN search** | No | Yes (HNSW, IVF, etc.) |
| **Query semantics** | SQL (point lookup by entity key) | ANN similarity search |
| **Lineage tracking** | Yes | No |
| **Dataset versioning** | Yes | Collections only (no version lineage) |
| **Curation / quality** | Yes | No |
| **Embedding generation** | Declared (InferenceTransform Job) | External |
| **Schema evolution** | Versioned (new version) | Collections are fixed-schema |
| **Throughput** | Batch (Jobs) | Millions of QPS (online) |
| **Latency** | Batch (minutes to hours) | Milliseconds (online retrieval) |

### Where Milvus Wins

- **ANN search performance**: HNSW and IVF indexes deliver sub-10ms retrieval over billions of vectors
- **Scale**: horizontal sharding, distributed index across multiple nodes
- **Dynamic schema**: as of Milvus 2.4, fields can be added post-creation
- **Filtering**: scalar filter + vector search in one query (e.g., filter by language before ANN)

### Where Raise Wins

- **Embedding lifecycle management**: Raise tracks which model generated the embedding, which pipeline, and which dataset version — Milvus stores the resulting vector with no provenance
- **Upstream curation**: the quality of embeddings depends on the quality of the text they were generated from; Raise manages that entire upstream process
- **Reusability**: embeddings stored in Raise as a feature group can be served to multiple downstream consumers (Milvus, training jobs, analytics) without re-generation

### Integration Pattern

The canonical integration:

```
Raise (manages the lifecycle):
  raw-text FG → curation job → curated-text FG → embedding job → embeddings FG

Milvus (handles retrieval):
  embeddings FG → export job → Milvus collection → ANN search at serving time
```

Raise owns the pipeline and the data management. Milvus owns the index and the query path.

---

## 6. Quantitative Comparison

### Lines of Code for Common Tasks

| Task | Raise | Databricks | Ray | Modal |
|------|-------|------------|-----|-------|
| Declare 5 features | 5 | 12+ (needs DataFrame) | N/A | N/A |
| Add derived feature | 1 | 8 (new table) | N/A | N/A |
| Bulk create from schema | 3 | N/A | N/A | N/A |
| Schedule daily curation job | 15 | 20 (Airflow DAG) | 25+ | 10 (Modal cron) |
| GPU inference pipeline | 10 | 20 | 20 | 8 (Modal GPU fn) |
| Dataset versioning | 8 | Manual (Delta snapshot) | N/A | N/A |
| Pre-training signal pipeline | 20 | 50+ (DIY Spark) | 40+ (DIY Ray) | 25+ (DIY Modal) | 

### Concepts to Learn

| Platform | Core Concepts | Notes |
|----------|--------------|-------|
| **Raise** | 4 | FeatureStore, FeatureGroup, Feature, Job |
| **Databricks** | 6 | + Unity Catalog, Spark session, Delta format |
| **Ray** | 5 | Actor, remote(), Dataset, runtime_env, cluster |
| **Modal** | 3 | App, @function, Stub — but no data management |
| **Haystack** | 5 | Pipeline, Component, connect(), Document, DocumentStore |
| **Milvus** | 6 | Collection, Schema, Field, Index, search(), partition |

### Stack Coverage

| Capability | Raise | Databricks | Ray | Modal | Haystack | Milvus |
|------------|-------|------------|-----|-------|----------|--------|
| Feature declaration | ✓ | ✓ | — | — | — | partial |
| Derived features | ✓ | partial | — | — | — | — |
| Lineage tracking | ✓ | ✓ | — | — | — | — |
| Curation pipelines | ✓ | DIY | DIY | DIY | — | — |
| Dataset versioning | ✓ | partial | — | — | — | — |
| Dataset mixing | ✓ | DIY | DIY | DIY | — | — |
| Human annotation | ✓ | — | — | — | — | — |
| Eval suites | ✓ | — | — | — | — | — |
| Distributed compute | backend | ✓ | ✓ | ✓ | — | — |
| ANN retrieval | — | — | — | — | partial | ✓ |
| LLM inference pipelines | — | — | — | — | ✓ | — |
| GPU serverless | — | — | partial | ✓ | — | — |

---

## 7. AI Lab Data Infrastructure

For a deeper analysis of how OpenAI, Anthropic, Google DeepMind, Meta, and Microsoft have built their data infrastructure — including pre-training signal pipelines, post-training annotation workflows, and the tooling gap this creates — see [AI_LAB_DATA_INFRA.md](./AI_LAB_DATA_INFRA.md).

Summary of findings relevant to this comparison:

- Foundation model labs compute the same types of derived signals as traditional feature stores ("perplexity score", "toxicity classifier", "detected language") but call them "quality signals" rather than "features" and store them as Parquet columns with no unified registry or lineage.
- No lab has a unified signal store that spans pre-training curation, post-training annotation, and eval tracking. All have custom pipelines that reinvent the same patterns per dataset.
- OpenAI built a Codex-powered agent to enrich their internal data metadata — an agentic system writing to what is functionally a feature store. They built it because no off-the-shelf tool fit.
- The signal management gap identified in that document is exactly the layer Raise occupies in the stack diagram below.

---

## 8. Where Raise Fits in the Modern ML Stack

```
┌────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  Haystack (RAG, agents)    Milvus (similarity search)      │
└─────────────────────────┬──────────────────────────────────┘
                          │  consumes embeddings, features
┌─────────────────────────▼──────────────────────────────────┐
│               Data Management Layer  ← RAISE               │
│  Feature declaration, versioning, lineage, curation,        │
│  dataset mixing, annotation, eval tracking                  │
└─────────────────────────┬──────────────────────────────────┘
                          │  dispatches jobs to
┌─────────────────────────▼──────────────────────────────────┐
│                   Execution Layer                           │
│  Ray (distributed cluster)    Modal (serverless GPU)        │
└─────────────────────────┬──────────────────────────────────┘
                          │  reads from / writes to
┌─────────────────────────▼──────────────────────────────────┐
│                    Storage Layer                            │
│  Databricks / Delta Lake    S3 / GCS    Postgres (metadata) │
└────────────────────────────────────────────────────────────┘
```

Raise occupies the **data management layer** — the layer that currently has the biggest gap in foundation model workflows. Researchers and agentic systems need a place to declare what signals exist, track which version of which dataset was used to train which checkpoint, curate data through a composable pipeline, and annotate responses for RLHF — without having to own the distributed compute or the storage backend.

---

## 9. Conclusions

### Raise Is Better For:
1. **Signal/feature declaration** without live data or infrastructure setup
2. **Pre-training and post-training data lifecycle** management (curation, mixing, versioning, annotation, evals)
3. **Researcher workflows** where iteration speed and low ceremony matter
4. **Agentic systems** that need to construct and execute ML pipelines programmatically — path syntax, idempotency, and declarative intent are LLM-constructible; IAM roles and Spark sessions are not

### Raise Is Worse For:
1. **Distributed execution**: use Ray or Modal; Raise should dispatch to them
2. **ANN retrieval**: use Milvus; Raise should export to it
3. **Inference pipeline orchestration**: use Haystack; Raise produces what Haystack consumes
4. **Production lakehouse at scale**: use Databricks; Raise could layer on top of it

### The Right Mental Model

Raise is not a replacement for any of these tools. It is the **missing data management layer** between raw data and the tools that consume it — specifically designed for the researcher and agentic-system users who currently cobble this layer together with ad-hoc Parquet files, custom scripts, and spreadsheets tracking which dataset trained which model.

---

## Sources

### Feature Store Platforms
- [Databricks Feature Engineering](https://docs.databricks.com/aws/en/machine-learning/feature-store/)
- [Databricks Unity Catalog Feature Tables](https://docs.databricks.com/aws/en/machine-learning/feature-store/uc/feature-tables-uc)
- [Vertex AI Feature Store](https://docs.cloud.google.com/vertex-ai/docs/featurestore/latest/overview)
- [Azure ML Feature Store](https://learn.microsoft.com/en-us/azure/machine-learning/concept-what-is-managed-feature-store)

### Compute & Execution
- [Ray - Distributed AI Framework](https://www.ray.io/)
- [Ray Data](https://docs.ray.io/en/latest/data/data.html)
- [Modal - Serverless Cloud](https://modal.com/)
- [How Ray Powers ChatGPT](https://thenewstack.io/how-ray-a-distributed-ai-framework-helps-power-chatgpt/)
- [OpenAI at Ray Summit](https://thenewstack.io/openai-chats-about-scaling-llms-at-anyscales-ray-summit/)

### LLM Pipeline & Vector DB
- [Haystack by deepset](https://haystack.deepset.ai/)
- [Milvus Vector Database](https://milvus.io/)
- [Milvus vs. pgvector](https://milvus.io/docs/comparison.md)

### AI Lab Data Infrastructure
See [AI_LAB_DATA_INFRA.md](./AI_LAB_DATA_INFRA.md) for full sources on lab infrastructure, pre-training pipelines, and post-training data practices.
