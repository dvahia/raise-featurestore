# OpenAI Infrastructure Deep Dive

A workflow-by-workflow account of OpenAI's data and training infrastructure — from the first web crawl byte to a deployed model checkpoint. Organized by lifecycle phase, with the underlying infrastructure for each phase called out explicitly.

OpenAI is deliberately opaque about their stack starting with GPT-4 ("the report contains no further details about the architecture, hardware, training compute, or dataset construction"). What follows is sourced from primary documents (papers, blog posts, open-source repositories, conference talks, job postings) and noted where it is reverse-engineered inference.

---

## The Stack at a Glance

| Phase | Primary Infrastructure |
|---|---|
| Data acquisition | CommonCrawl WARC → Azure Blob Storage (raw), Parquet (processed) |
| Storage (raw) | Azure Data Lake / Blob: cold tier for WARCs, warm for Parquet |
| Storage (enriched) | Apache Iceberg or Delta Lake on Azure ADLS Gen2 for ACID signal writes |
| Training-time storage | MDS (Mosaic Data Shard) or packed Parquet shards on NVMe |
| Signal computation | Ray/Spark workers, FastText, KenLM, custom PyTorch classifiers |
| Signal versioning | Column-at-a-time writes via Iceberg schema evolution |
| Compliance filtering | Multi-stage batch: regex → FastText → neural classifiers, audit log per doc |
| Dedup | MinHash LSH + exact 50-token span matching |
| Pipeline orchestration | Apache Airflow DAGs + custom Ray DAG pipelines |
| Pipeline monitoring | Datadog / Prometheus + Great Expectations + row-count anomaly detection |
| Dataset exploration | DuckDB over Parquet, Jupyter notebooks, custom Streamlit/Gradio viewers |
| Lineage tracking | Document-level metadata sidecar + Parquet file metadata headers |
| Experiment (small scale) | Scaling law extrapolation, Ray Tune, W&B |
| Hero run (training) | Azure NDm A100 v4 → GB300 NVL72, NCCL + InfiniBand, PyTorch + Triton |
| Parallelism | 3D parallel: tensor + pipeline + data (PTD-P) |
| Checkpointing | Azure Blob Storage, every ~30–60 min |
| Training monitoring | W&B + custom dashboards |
| SFT annotation | Scale AI + direct contractor pool |
| RLHF orchestration | Ray (policy, critic, reward, reference model placement) |
| RLHF inference | vLLM + Ray for policy rollouts during PPO |
| RLHF memory | DeepSpeed ZeRO2 + gradient checkpointing |
| Evals | openai/evals framework, Snowflake for results, continuous eval |
| Internal data agent | Kepler (GPT-5 + RAG over 70K datasets + Codex table knowledge) |
| Kernel authoring | Triton (compiled through PyTorch Inductor / torch.compile) |

---

## Phase 1: Pre-Training Data Preparation

### Acquisition

OpenAI's pre-training corpus is anchored on **CommonCrawl** — monthly snapshots of the public web available as WARC (Web ARChive) files. For GPT-3, OpenAI downloaded 41 monthly shards (2016–2019), totaling ~45 TB of compressed plaintext before any filtering.

CommonCrawl is supplemented by:
- **WebText** — web pages linked from Reddit posts with ≥3 upvotes (the Reddit quality proxy, established with GPT-2)
- **Books1 / Books2** — licensed book corpora
- **Wikipedia** — English Wikipedia

Starting with GPT-4, internally curated data of unknown composition is also included.

### Filtering: Classifier-First, Not Heuristic-First

The crucial design choice is **classifier-first filtering**. Rather than applying heuristic rules (line length, symbol density, etc.) as the primary gate, OpenAI trains a **logistic regression classifier** to distinguish documents similar to their high-quality reference corpora (WebText, Wikipedia, Books) from raw web pages. Only pages scoring above a threshold are retained.

The result: ~570 GB retained from ~45 TB of input — a **98.7% discard rate** from raw CommonCrawl. The classifier approach selects for documents that resemble known-good content, rather than filtering against a checklist of badness.

For GPT-4, additional safety classifiers were applied to remove: erotic content, hate speech, self-harm material, and other policy-violating content. These classifiers run on top of the quality classifier output.

**Infrastructure**: Signal computation at this scale is distributed across hundreds of workers. OpenAI's job postings reference distributed data systems engineers and "large-scale multimodal training data pipelines." Industry standard at this scale is Ray Data or Apache Spark for embarrassingly parallel per-document processing. OpenAI has not confirmed the specific framework.

### Deduplication

Deduplication runs at two levels:

1. **Document-level near-dedup**: MinHash + LSH (Locality Sensitive Hashing), identifying documents with Jaccard similarity above a threshold (~80%). The Spark MLlib `MinHashLSH` implementation or a Ray-based equivalent is the standard approach at this scale.

2. **Span-level exact dedup**: 50-token windows. Any document containing a span that appears verbatim in the dev or test sets of known benchmarks is removed — preventing benchmark contamination.

### Weighted Oversampling

OpenAI's documented mixing strategy oversamples high-quality sources relative to their corpus size:

```
CommonCrawl:   ~410B tokens   → 60% of training samples
WebText2:       ~19B tokens   → 22% of training samples  ← 11.5x oversampled
Books1:         ~12B tokens   →  8% of training samples  ← 6.5x oversampled
Books2:         ~55B tokens   →  8% of training samples  ← 1.4x oversampled
Wikipedia:       ~3B tokens   →  3% of training samples  ← 10x oversampled
```

This means the model sees WebText and Books content approximately once per epoch, while CommonCrawl content is under-sampled relative to its size. The design philosophy: token count is not quality, and high-quality data deserves more training exposure.

**Key infrastructure**: This weighted mixing is implemented at the data loader level — not by physically creating a pre-mixed dataset. The data loader draws samples from each source according to the configured probabilities. In practice this requires a data pipeline that can efficiently sample from five separate large datasets simultaneously, which requires either a custom data loader or a framework like `datasets` from HuggingFace with interleaved streaming.

### What Changes for GPT-4 and Beyond

OpenAI explicitly states the GPT-4 Technical Report "contains no further details about the architecture, hardware, training compute, or dataset construction." What is inferable from other public information:

- **LLM-based quality scoring**: FineWeb (HuggingFace, 2024) explicitly models GPT-4 data practices using GPT-4 as a quality scorer on a sample, then distilling into a FastText classifier. The implication: OpenAI uses or used LLM judges to assign quality scores at corpus scale, then trains a cheaper classifier to apply the judgment at scale.
- **Multimodal data**: GPT-4V implies a pre-training corpus that includes image-text pairs, image captions, OCR'd documents, and interleaved image-text data.
- **Stricter safety filtering**: The GPT-4 system card describes more aggressive content filtering than GPT-3.

---

## Phase 1b: Data Engineering Infrastructure

This section covers the infrastructure layer that sits beneath the ML pipeline described in Phase 1 — the storage systems, processing engines, monitoring, and tooling that make it possible for hundreds of researchers to work against the same massive dataset without stepping on each other.

### Storage Architecture: Three Tiers

Pre-training data at AI lab scale lives in three distinct storage tiers, each optimized for a different access pattern:

```
Tier 1 — Raw Archive (cold)
  Format:  WARC (Web ARChive) files from CommonCrawl
  Storage: Azure Blob Storage / ADLS Gen2, cold access tier
  Size:    45–100+ TB per monthly CommonCrawl shard (OpenAI downloads 41 shards)
  Access:  Read once during extraction; rarely touched again
  Cost:    Cheapest per GB; high latency acceptable

Tier 2 — Processed & Signal-Enriched (warm)
  Format:  Apache Parquet, partitioned by domain/language/crawl-date
  Storage: Azure Data Lake Storage (ADLS) Gen2 or S3-compatible
  Size:    Fraction of Tier 1 after filtering (GPT-3: 45 TB → 570 GB retained)
  Access:  Read/write during signal computation; read during mixing/training prep
  Cost:    Mid-range; SSD-backed for interactive query, HDD for bulk

Tier 3 — Training-Ready (hot)
  Format:  MDS (Mosaic Data Shard) or packed Parquet shards
  Storage: NVMe SSDs local to training nodes OR high-throughput distributed store
  Size:    Final training set; aggressively compressed
  Access:  Random-access streaming during training; must sustain GB/s throughput
  Cost:    Most expensive; sized to avoid GPU starvation
```

**WARC → text extraction**: The first pipeline stage reads raw WARC files and extracts clean text. The dominant open-source tools are Trafilatura (linguistic quality-focused), Resiliparse (speed-focused), and CCNet (Facebook, used for CCNet corpus and mC4). The output is JSONL — one JSON object per document with extracted text, URL, crawl date, and language.

**JSONL → Parquet**: After text extraction, documents are written to **Apache Parquet** partitioned by crawl shard and language. Parquet's columnar format is critical here: when a researcher wants to read only the `text` and `url_domain` columns across 100B documents, Parquet reads only those column files rather than full rows — a 10–50x I/O reduction vs. row-oriented formats.

**Partitioning strategy**: Typical partition scheme:
```
/data/processed/
  year=2024/month=03/language=en/shard=00001.parquet
  year=2024/month=03/language=zh/shard=00001.parquet
  ...
```

This allows queries like "give me all English documents from Q1 2024" to skip irrelevant partitions entirely (partition pruning), which at petabyte scale is the difference between a 10-minute query and a 10-hour one.

### The Signal Enrichment Problem: Concurrent Writes at Scale

The hardest data engineering problem in pre-training pipelines is supporting **concurrent signal enrichment** — multiple teams computing different signals against the same base corpus simultaneously, without conflicts or full rewrites.

**The naive approach** — recompute the entire Parquet file every time a new signal is added — doesn't scale. A 570 GB corpus takes hours to rewrite. If 10 teams are computing signals in parallel, you'd need to coordinate them into a single batch, which serializes the research process.

**The Iceberg/Delta Lake approach** (what labs almost certainly use at scale):

**Apache Iceberg** and **Delta Lake** are open table formats that add ACID transactions, schema evolution, and time-travel to Parquet files on object storage. The key capabilities for signal enrichment:

1. **Schema evolution without rewrite**: Adding a new column (`educational_quality_score`) to a 100B-row Iceberg table doesn't rewrite existing data. Iceberg records the column addition in its metadata layer; existing files simply return `null` for the new column until their rows are backfilled.

2. **Column-at-a-time writes**: The perplexity team and the language ID team can write their signals as separate Iceberg merge operations targeting different columns. Iceberg's optimistic concurrency control (OCC) lets these run in parallel and resolves any conflicts via snapshot isolation — each operation sees a consistent snapshot of the table.

3. **Time-travel**: `SELECT * FROM documents VERSION AS OF '2024-03-15'` returns the table as it existed on that date — before the dedup run was applied. This lets a researcher reproduce any earlier experiment exactly, even after signals have been updated or records deleted.

4. **Hidden partitioning**: Iceberg can automatically maintain hidden partition metadata (e.g., partition by language) even as the schema evolves, without requiring users to know the physical layout.

**In practice at Meta** (the most documented case): Meta uses a similar pattern for their production feature platform. Palette (their feature store) uses columnar storage with append/update operations rather than full rewrites. The design principle — signals are columns, and columns can be added independently — is the same.

**The alternative pattern** (simpler, used at smaller scale): Keep signals in **separate Parquet stores**, one per signal type or team. Merge at read time via a join on `doc_id`. This eliminates write conflicts entirely (each team owns their store) at the cost of join overhead at query time. For exploration and experimentation this is often sufficient; for the final training pipeline the stores are merged into a single enriched Parquet.

### Storage Format for Training: MDS

Once the signal-enriched dataset is filtered and mixed, it's converted into a format optimized for **streaming during training**. The dominant open-source format is **MDS (Mosaic Data Shard)** from MosaicML (now Databricks):

```
training_data/
  shard_00001.mds   ← binary, ~256 MB per shard
  shard_00002.mds
  ...
  index.json        ← shard manifest with record counts, byte offsets
```

Each MDS shard is a binary file containing fixed-size records. The training data loader:
1. Reads `index.json` to learn the total record count and shard layout
2. Shuffles at the shard level (not record level — too expensive at this scale)
3. Streams shards from storage (Azure Blob / S3) with prefetching
4. Decodes records in parallel on CPU workers while GPU trains

MDS achieves **near-disk-speed streaming throughput** even from object storage because it reads sequential large chunks rather than random small reads. For a 15T-token dataset, the data loader must sustain ~10–50 GB/s to avoid GPU starvation on a large cluster — impossible with row-at-a-time access, achievable with sequential shard streaming.

### Compliance Filtering Pipeline: Multi-Stage Architecture

Safety and compliance filtering at 100B+ document scale uses a **multi-stage funnel** design — fast cheap filters first, expensive accurate filters only on survivors. The design principle: at 100B documents, even a 1ms per-document operation takes 28 CPU-hours. A 100ms LLM inference call on every document is simply not feasible.

```
Stage 1 — URL blocklists (nanoseconds/doc)
  Adult domain blocklists, known spam domains, DMCA takedown lists
  Applied at WARC level before text extraction
  Eliminates ~5–15% of documents

Stage 2 — Heuristic rules (microseconds/doc)
  Regex patterns: SSN, credit card numbers, email addresses (PII detection)
  Symbol density thresholds (>50% non-alphanumeric → likely code dump/spam)
  Short document filters (<50 tokens → insufficient signal)
  Applied in vectorized Python/Numpy on Spark executors
  Eliminates an additional 10–30%

Stage 3 — FastText classifiers (milliseconds/doc)
  Language identification (FastText lid.176.bin — 176 languages, ~0.5ms/doc)
  Coarse NSFW classifier (FastText trained on labeled adult/non-adult URLs)
  Hate speech coarse classifier (FastText trained on toxic/non-toxic samples)
  Applied as Spark UDFs; trivially parallelized across executors
  Eliminates an additional 5–20%

Stage 4 — Neural classifiers (10–100ms/doc, on GPU workers)
  Fine-grained NSFW classifier (CNN or small transformer, trained on labeled images/text)
  PII NER model (BERT-based, identifies named personal information in context)
  Copyright signal classifier (identifies likely copyrighted verbatim passages)
  Run only on documents that pass Stages 1–3; ~50% of original corpus
  Applied on GPU workers via Ray or Spark + GPU support
  Eliminates an additional 5–15%

Stage 5 — LLM-based judgment (seconds/doc, on GPU; run on sample only)
  GPT-4 / Claude quality scoring on representative sample (~500K documents)
  Used to train Stage 3/4 classifiers, not applied to full corpus
  Output: labeled training data for cheaper classifiers
```

**Audit trail**: Each filtering stage writes a decision record — `{"doc_id": "...", "stage": 3, "classifier": "fasttext_nsfw_v2.1", "score": 0.94, "threshold": 0.8, "action": "filter"}` — to a separate audit log Parquet. This audit log is critical for:
- Reproducing filtering decisions when a pipeline version changes
- Debugging unexpected data quality drops
- Compliance documentation (demonstrating due diligence for specific content categories)
- Comparing across classifier versions (did v2.2 filter more or fewer docs than v2.1?)

**Soft vs. hard filtering**: "Hard" filtering removes documents entirely; "soft" filtering adds a score column but keeps the document, allowing downstream consumers to apply their own threshold. Most labs use soft filtering — the `is_nsfw`, `nsfw_score`, `pii_score` columns are written to the enriched Parquet, and `include_in_training = (nsfw_score < 0.3 AND pii_score < 0.5 AND ...)` is computed as a derived column. This preserves the document and all its signals, enabling researchers to explore the effect of different thresholds without re-running classifiers.

### Pipeline Orchestration

Data preparation pipelines at AI lab scale are **long-running batch workflows** with complex dependencies between stages. A full CommonCrawl processing pipeline might look like:

```
WARC download → text extraction → language ID → quality scoring → dedup → compliance → mixing
```

Each stage depends on the previous one, takes hours to days to complete, and can fail partway through. The orchestration layer must handle:
- **DAG scheduling**: run dedup only after quality scoring is done
- **Distributed execution**: each stage runs as hundreds or thousands of parallel Spark tasks
- **Failure recovery**: if quality scoring fails on shard 4,523 of 10,000, resume from that shard
- **Progress tracking**: humans watching a multi-day job need status updates
- **Data version gating**: only proceed to next stage if the output passed data quality checks

**Apache Airflow** is the industry standard for this use case, and the most likely choice at OpenAI based on job postings referencing "data orchestration" and the prevalence of Airflow in Meta's documented stack. Airflow represents the pipeline as a DAG of tasks; each task is a Spark submit, a Ray job, or a Python function. Tasks have retry policies, SLAs, and dependency declarations.

A typical Airflow DAG for data curation:
```python
with DAG("web_curation_2024_03", schedule="@monthly") as dag:
    extract    = SparkSubmitOperator(task_id="extract_text",    ...)
    lang_id    = SparkSubmitOperator(task_id="language_id",     ...)
    quality    = RayJobOperator(task_id="quality_scoring",      ...)
    dedup      = SparkSubmitOperator(task_id="minhash_dedup",   ...)
    compliance = RayJobOperator(task_id="compliance_filter",    ...)
    validate   = PythonOperator(task_id="data_quality_check",   ...)
    publish    = IcebergWriteOperator(task_id="publish_shard",  ...)

    extract >> lang_id >> quality >> dedup >> compliance >> validate >> publish
```

**Failure recovery for long Spark jobs**: The standard pattern is **checkpoint-based recovery** — Spark jobs write intermediate results to durable storage (Azure Blob) at regular intervals (every N shards). If the job fails, it restarts from the last checkpoint rather than from scratch. For a 3-day job that fails on day 2, this limits rework to a few hours rather than two full days.

**Ray DAGs** are used for Python-native pipelines that are more dynamic than Spark can express — for example, an LLM-based quality scoring pipeline where the batch size per worker depends on document length.

### Pipeline Monitoring and Failure Management

Data pipelines at this scale fail constantly — not catastrophically, but with a steady background rate of infrastructure failures, quota exhaustion, data format surprises, and network timeouts. The monitoring stack must distinguish "this task failed and needs a page" from "this shard had 3% corrupt documents which is within normal range."

**Infrastructure monitoring** (Datadog or Prometheus + Grafana):
- Spark executor health: memory usage, GC time, task failure rate per executor
- Ray worker health: GPU utilization, memory, task queue depth
- Azure Blob / ADLS read/write throughput and error rates
- Pipeline progress: shards processed / total shards, estimated completion time

**Data quality monitoring** (the harder problem):
- **Row count anomaly detection**: after quality filtering, the expected pass rate is ~2%. If a new run passes 0.5% or 8%, something changed — either the data or the classifier. Automated anomaly detection (Z-score or ARIMA-based) on pass rates alerts the pipeline team.
- **Distribution drift**: the distribution of quality scores, language codes, and domain sources should be stable across monthly CommonCrawl shards. A sudden shift (e.g., language ID returns 80% English vs. the usual 45%) indicates a classifier bug or a crawl artifact.
- **Great Expectations**: the open-source data validation framework, widely used for defining "expectations" (assertions about data) that run after each pipeline stage. Example expectations: `expect_column_values_to_be_between("quality_score", 0.0, 1.0)`, `expect_column_proportion_of_unique_values_to_be_between("doc_id", 0.99, 1.0)`. Failed expectations block pipeline progression and trigger alerts.
- **Schema drift alerts**: if the output Parquet has an unexpected column or a column changes type, the pipeline fails with a descriptive error rather than silently writing corrupt data.

**On-call rotation**: Meta, Google, and likely OpenAI maintain a data infrastructure on-call rotation — an engineer paged for any pipeline failure that blocks a downstream training run. The SLA for pre-training pipelines is typically "unblock within 4 hours for any failure affecting an in-progress hero run."

**Common failure modes and mitigations**:

| Failure | Detection | Mitigation |
|---|---|---|
| Spark executor OOM | Executor logs, task retry rate spike | Increase executor memory, reduce partition size |
| Azure Blob quota throttling | HTTP 429 error rate in Datadog | Retry with exponential backoff, increase quota |
| FastText classifier segfault | Task failure with non-zero exit code | Isolate in subprocess with timeout; skip + log bad doc |
| Corrupt WARC shard | Parse errors > 1% on a shard | Mark shard as bad, skip and alert |
| Quality score distribution shift | Great Expectations check fails | Block pipeline, diff classifier version, human review |
| Dedup graph too large for memory | Spark stage failure | Repartition into smaller components, use disk spill |

### Dataset Exploration and Visualization

A 15-trillion-token dataset is not something you can `head -n 10` and understand. Researchers need purpose-built tooling to answer questions like: "What fraction of our code data is Python vs. JavaScript?", "Why did the quality score drop for the March 2024 crawl?", "Show me 20 representative examples of documents that passed safety filtering but look borderline."

**The query layer — DuckDB**: DuckDB has become the standard tool for interactive exploration of Parquet files at scales that don't fit in Pandas (up to a few TB on a laptop). It reads Parquet columns directly, pushes down filters to avoid unnecessary I/O, and returns results in seconds for queries over files that would take hours in Spark.

```python
import duckdb
conn = duckdb.connect()

# Distribution of quality scores — runs in seconds on a 50 GB Parquet
conn.execute("""
    SELECT
        FLOOR(quality_score * 10) / 10 AS score_bucket,
        COUNT(*) AS doc_count,
        AVG(token_count) AS avg_tokens
    FROM read_parquet('adls://datasets/processed/year=2024/month=03/language=en/*.parquet')
    WHERE quality_score IS NOT NULL
    GROUP BY 1
    ORDER BY 1
""").df()

# Sample 100 borderline-NSFW documents for human review
conn.execute("""
    SELECT doc_id, url, text[:500], nsfw_score
    FROM read_parquet('...')
    WHERE nsfw_score BETWEEN 0.4 AND 0.6
    ORDER BY RANDOM()
    LIMIT 100
""").df()
```

**Jupyter notebooks** are the primary research interface for dataset analysis. Researchers write notebooks that:
- Load signal distributions from Parquet via DuckDB
- Plot histograms and CDFs of quality scores, token counts, language distributions
- Sample and display documents for qualitative inspection
- Run A/B comparisons: "what does the data filtered at threshold 0.5 look like vs. 0.65?"
- Generate reports that become artifacts in the data version's provenance record

**Custom visualization dashboards** for the data team: Typically a **Streamlit** or **Gradio** app that exposes:
- A random document sampler with filter controls (language, domain, score range)
- Side-by-side comparison of two dataset versions
- A "problematic document" viewer (high nsfw score, high perplexity, flagged for review)
- Domain distribution treemaps (what fraction of the corpus comes from `.edu`, `.gov`, Reddit, Wikipedia, etc.)
- Quality score time series (how has the average quality score changed across monthly crawls?)

**Lilac** (open-source from HuggingFace): A data curation tool specifically designed for ML datasets. Supports semantic search over dataset examples, clustering by embedding similarity, and labeling interfaces. Used by smaller teams; at OpenAI scale the data volumes require custom tooling.

**The data room**: For sensitive datasets (containing PII, safety-relevant content, or legally sensitive material), a "data room" is an isolated compute environment where researchers can explore examples without exfiltration risk. All queries run inside the data room; only aggregated statistics are exported. This is the standard approach for handling RLHF data that contains user conversations.

### Data Lineage: The Provenance Problem at 100B Scale

Lineage answers the question: "which processing steps, with which versions, produced this document in the final training set?" At 100B+ document scale, maintaining per-document lineage is itself a significant engineering problem.

**Document-level metadata** (what goes in the Parquet): Every document carries a `pipeline_run_id` and a set of signal version tags in its row:
```json
{
  "doc_id": "cc-2024-10-00001-en-0000042",
  "pipeline_run_id": "run-2024-03-15-v4.2",
  "quality_scorer_version": "kenlm-v3.1",
  "dedup_run_id": "dedup-2024-03-17",
  "nsfw_classifier_version": "fasttext-nsfw-v2.1",
  "include_in_training": true,
  "quality_score": 0.73,
  ...
}
```

**File-level metadata** (what goes in the Parquet file header): Apache Parquet supports key-value metadata in the file footer — a place to store pipeline version, processing date, source shard ID, and a checksum of the input data. This enables coarse lineage ("this Parquet file was produced by pipeline run X from source shard Y") without per-row overhead.

**Iceberg's built-in lineage**: Apache Iceberg maintains a full snapshot history — every table write creates a new snapshot, and older snapshots are retained for a configurable period. This gives time-travel (see the table as it was at any past point) and coarse lineage (which operation created which snapshot). Iceberg doesn't provide column-level or row-level lineage natively, but snapshot metadata can record the pipeline run that created each snapshot.

**What labs actually do in practice**: The honest answer, based on public dataset documentation and job posting language, is that most labs maintain lineage via a combination of:
1. Per-document metadata columns (the version tags above)
2. Parquet file metadata headers
3. Pipeline configuration files in Git (which pipeline version ran when)
4. README files in object storage prefixes documenting each major processing step

**What labs don't have**: A unified lineage graph system (like Apache Atlas or DataHub) that traces from raw WARC shard → extracted text → quality-scored → deduped → filtered → mixed → training shard at the individual document level. This doesn't exist at production scale at any lab. Building it is part of what Raise is designed to do — `DatasetVersion.derive()` and the job system provide the managed lineage layer that labs currently reconstruct from README files.

---

## Phase 2: Small-Scale Experiments

### The Core Method: Scaling Law Extrapolation

Before committing to a multi-month, multi-million-dollar training run, OpenAI validates decisions at small scale using **scaling laws**. The foundational paper (Kaplan et al., 2020) established power-law relationships:

```
L(C) = a · C^(-b) + c
```

where L is cross-entropy loss and C is compute (FLOPs). The relationship holds across 7 orders of magnitude of compute. The practical implication: you can run a model at 1/1000th the target compute, fit the power law, and predict the loss at full scale before running the full experiment.

**GPT-4's predictive scaling** was the clearest public demonstration. Using models trained at 1,000–10,000x less compute than GPT-4, OpenAI fitted the power law and **predicted GPT-4's final loss before the training run completed**. They also predicted task-specific performance using:

```
−E_P[log(pass_rate(C))] = α · C^{−k}
```

This allows data mixing decisions, architecture choices, and hyperparameter settings to be validated at small scale (billions of tokens, days of compute) before committing to full scale (trillions of tokens, months of compute).

### Infrastructure for Small-Scale Experiments

**Cluster size**: Small-scale experiments run on clusters of 8–64 GPUs — a single Azure NDm A100 v4 node to a handful of nodes. The same Azure infrastructure, just a tiny fraction of it.

**Hyperparameter search**: **Ray Tune** for distributed hyperparameter optimization. OpenAI is a confirmed Ray user; Ray Tune's ASHA (Asynchronous Successive Halving) and PBT (Population Based Training) schedulers allow efficient parallel search where bad trials are terminated early, freeing resources for promising configurations.

**Experiment tracking**: **Weights & Biases (W&B)** is used for tracking. W&B's integration with OpenAI's fine-tuning is publicly documented, and W&B publicly lists OpenAI as a customer. W&B tracks:
- Training loss and validation loss per step
- Gradient norms (critical for detecting instabilities before they cascade)
- GPU utilization and memory usage
- Learning rate schedule
- Hyperparameter sweeps across multiple concurrent runs

**Notebook-based exploration**: Jupyter / JupyterHub is the standard interface for data scientists doing exploratory analysis on sampled datasets. A researcher might pull 0.1% of the pre-training corpus, run quality signal distributions, visualize perplexity curves, and identify problem data slices — all in a notebook before writing a formal pipeline.

### Data Ablations at Small Scale

Before committing the mixing weights for a hero run, OpenAI almost certainly runs **data ablations** — training small models (~1B parameters, ~10B tokens) on different data mixes to measure the effect on downstream benchmarks. The DCLM benchmark (Apple, 2024) formalized exactly this methodology and confirmed it's the right approach: fixing the compute budget and varying the data recipe to maximize benchmark performance.

The infrastructure for ablations is: a templated training config, a sweep of dataset version variants, Ray Tune to orchestrate parallel runs, W&B to collect results, and a comparison dashboard to select the winning mix for the hero run.

---

## Phase 3: Hero Runs

### What is a Hero Run

A hero run is the full-scale training run — months of continuous training on thousands or tens of thousands of GPUs to produce a production model. The term "hero" reflects the commitment: unlike small-scale experiments, you don't abort a hero run because of a suboptimal hyperparameter. Hero runs are planned, resourced, and staffed like critical infrastructure projects.

### The Cluster

**GPT-3 (2020)**:
- **10,000 NVIDIA V100 Tensor Core GPUs**
- **285,000 AMD EPYC CPU cores**
- InfiniBand network: 400 Gbps per GPU server
- Cloud: Microsoft Azure
- At launch: 5th largest supercomputer in the world

**GPT-4 (2022–2023, inferred)**:
- ~**20,000 NVIDIA A100 80GB GPUs** for ~90–100 days (SemiAnalysis estimate)
- ~**2.15 × 10²⁵ FLOP** total compute
- Training cost: estimated ~$63 million
- Cluster: Azure NDm A100 v4 VMs

**Next generation (2025–2026)**:
- Azure is now deploying the first large-scale cluster with **NVIDIA GB300 NVL72** chips for OpenAI workloads
- GB300 NVL72: 72 GPUs per node, NVLink 5.0 interconnect, ~4x the memory bandwidth of H100

### The Node: Azure NDm A100 v4

Each compute node in OpenAI's A100 training clusters is an Azure NDm A100 v4 VM:
- **8 × NVIDIA A100 80GB Tensor Core GPUs** (within-node NVLink 3.0)
- **96 vCPUs** (AMD EPYC 7V12 Rome)
- **1,900 GiB RAM**
- **InfiniBand per GPU**: 200 Gbps NVIDIA Mellanox HDR — dedicated, topology-agnostic
- **GPU Direct RDMA**: supported (GPU-to-GPU memory transfer without CPU involvement)
- **NCCL2**: supported (the AllReduce library that coordinates gradient synchronization across nodes)

At training scale, hundreds to thousands of these VMs are connected via Azure's InfiniBand fabric into a single logical cluster.

### Parallelism: The PTD-P Strategy

OpenAI's blog post "Techniques for Training Large Neural Networks" (Brockman & Weng, 2021) describes their three-dimensional parallelism strategy — PTD-P: **P**ipeline + **T**ensor + **D**ata Parallel:

```
3D Parallelism Layout (example: 1,024 GPUs)
─────────────────────────────────────────────
Data Parallel groups:   8 replicas of the full model
  └── Each replica uses:
       Pipeline Parallel: 16 stages (one set of layers per stage)
         └── Each stage uses:
              Tensor Parallel:  8 GPUs (each holds a shard of each weight matrix)
```

**Tensor parallelism (intra-layer)**: Individual matrix multiplications are split across GPUs. For a weight matrix W of shape [d_model, 4d_model], each of 8 GPUs holds a [d_model, 512] column shard. Each GPU computes its dot product independently; an AllReduce combines the partial sums. This is the NVIDIA Megatron-LM approach.

**Pipeline parallelism (inter-layer)**: Different transformer layers run on different GPUs. With 96 layers across 16 pipeline stages, each stage holds 6 layers. Forward and backward passes are pipelined across stages with a "1F1B interleaved" schedule to minimize the idle "pipeline bubble."

**Data parallelism (batch replication)**: Identical model replicas process different data batches simultaneously. Gradients are synchronized via AllReduce across replicas at the end of each micro-batch.

**Communication stack**: NCCL (NVIDIA Collective Communications Library) handles all AllReduce, AllGather, and ReduceScatter operations. GPU Direct RDMA allows NCCL to transfer data directly between GPU memory cards across the InfiniBand network without routing through the CPU — critical for the latency-sensitive gradient synchronization that constrains training throughput.

### Mixed Precision and Triton

Training uses **BF16 mixed precision**: weights are stored in BF16, gradient accumulation and optimizer states are in FP32. BF16 gives ~2x memory savings vs. FP32 while maintaining training stability better than FP16 (larger dynamic range).

Custom GPU kernels are written and maintained in **Triton** — OpenAI's open-source Python-like GPU programming language. Triton allows researchers to write hardware-efficient kernels (FlashAttention, fused layer norms, fused activation functions) without low-level CUDA expertise. The Triton compiler generates optimized PTX (NVIDIA's assembly language) from Python-level code, and **PyTorch 2.0's `torch.compile`** uses PyTorch Inductor to generate Triton kernels for standard operations — enabling automatic operator fusion.

### Checkpointing

Full model checkpointing is the primary fault recovery mechanism. At 20,000 GPUs, hardware failures are not exceptional events — they happen multiple times per week. Each checkpoint includes:
- Model weights (all layers, all shards)
- Optimizer states (Adam momentum and variance — 2× the size of the weights)
- Data loader state (which batches have been processed)
- Learning rate schedule state

**Checkpoint frequency**: Not officially disclosed. Meta's LLaMA training checkpointed every 30 minutes; failure recovery required rewinding ~30 minutes of work. The tradeoff is checkpoint I/O time (minutes of training pause every 30 minutes) vs. expected recovery cost (average failure rate × average time to next checkpoint).

**Storage**: Azure Blob Storage. For a GPT-4 scale model (~1.8T parameters if the leaked architecture is accurate), a single checkpoint at BF16 would be ~3.6TB before optimizer states; with optimizer states (FP32), ~11TB per checkpoint. With hundreds of checkpoints over a multi-month run, checkpoint storage is a non-trivial infrastructure problem.

**Gradient checkpointing** (activation recomputation) is a separate technique: rather than storing all intermediate activations for the backward pass, activations are recomputed from saved inputs during the backward pass. OpenAI published the original gradient checkpointing package. The tradeoff: ~33% extra compute to reduce activation memory by ~10x.

### Real-Time Monitoring During Hero Runs

**What's monitored in real time:**
- Training loss (per step, per gradient accumulation)
- Validation loss (every N steps on held-out data)
- Gradient norm (to detect spikes before they cause instability)
- GPU utilization per device (to detect stragglers)
- Memory usage (to catch leaks early)
- MFU (Model FLOP Utilization) — actual FLOPs / theoretical peak FLOPs; target is >45%
- Loss spikes (sudden jumps in training loss, often caused by problematic data batches)

**Loss spikes** are the most common training emergency. At large batch sizes, a batch containing a very long document or a pathological sequence can cause a loss spike that, if uncorrected, can destabilize training. Standard mitigations: gradient clipping, logit softcapping (Gemma), z-loss regularization (T5/PaLM), QK-norm (Phi). Teams watch loss curves in real time and can roll back to a previous checkpoint if a spike doesn't recover within 100–200 steps.

**W&B dashboards** show live training curves. For hero runs, the W&B project is typically shared across the training team and viewed by dozens of engineers simultaneously. Custom alerting (Slack notifications) triggers if loss spikes above a threshold or if GPU utilization drops (indicating a node failure or network issue).

---

## Phase 4: Post-Training — SFT

### The Transition from Pretrained Base to Instruction Model

After the hero run produces a base model checkpoint, post-training begins. The base model predicts next tokens but has no concept of instructions, helpfulness, or appropriate behavior. SFT (Supervised Fine-Tuning) is the first alignment step: teaching the model what an ideal response looks like for a given prompt.

### Data Collection

**InstructGPT / ChatGPT SFT data** was collected via two channels:
1. **~40 direct contractors** hired and managed by OpenAI, screened for performance on sensitive content identification and labeling consistency. This small pool allows high-bandwidth communication with researchers.
2. **Scale AI's annotator platform** for higher-volume tasks. Scale AI maintains annotator pools with quality control pipelines, inter-annotator agreement tracking, and task-specific screening.

**What annotators produce**:
- Written responses to sampled prompts (demonstrating ideal behavior)
- Edited model responses (improving existing outputs to be more helpful or appropriate)

**Data volume for InstructGPT**: ~13,000 (prompt, demonstration) pairs. Surprisingly small — quality over quantity. The model generalizes the behavior from these demonstrations to the broader prompt distribution.

**Infrastructure**: Scale AI provides a managed annotation interface and delivers labeled data in bulk. OpenAI ingests this into their internal data pipeline, applies quality filtering (removing demonstrations where annotators were disagreed or unusually fast), and packages it into the SFT training set.

### SFT Training

**Scale**: SFT trains on the full pretrained model (175B for InstructGPT). Unlike pre-training, SFT is fast — typically 1–3 epochs over the SFT dataset, which is orders of magnitude smaller than the pre-training corpus.

**Infrastructure**: Same Azure cluster, but a tiny fraction. SFT for a 175B model requires a multi-node setup but nothing approaching the hero run scale. The learning rate is much lower (~9.5e-6) and training is measured in hours, not months.

**Overfitting risk**: SFT on a small dataset can cause the model to overfit to the demonstration format (becoming verbose, formulaic) while losing pre-training capabilities. OpenAI addresses this by mixing a small fraction of pre-training tokens into the SFT training objective — keeping the model general while learning the instruction format.

---

## Phase 5: Post-Training — RLHF

### Overview

RLHF (Reinforcement Learning from Human Feedback) is the most infrastructure-intensive post-training phase. It requires running four simultaneous models: the **Actor** (policy being trained), the **Critic** (value function estimating future rewards), the **Reward Model** (trained to predict human preferences), and the **Reference Model** (the SFT model, held frozen to compute KL divergence).

### Step 1: Reward Model Training

The reward model is trained on **comparison data** (not demonstrations). Annotators see pairs of model responses to the same prompt and rank which is better. The reward model learns to predict these rankings.

**Annotation infrastructure**: Same as SFT — Scale AI + direct contractors. For InstructGPT, ~330,000 comparison pairs were collected. Inter-annotator agreement: ~73%. The reward model is initialized from the SFT model weights (same architecture) and fine-tuned on the preference pairs.

**Training**: Reward model training is faster than SFT. Standard cross-entropy loss over pairs (maximize the log-probability difference between preferred and non-preferred response).

### Step 2: PPO Training Loop

PPO (Proximal Policy Optimization) is the RL algorithm that uses the reward model's scores to improve the policy. The loop:

```
1. Sample prompts from a prompt distribution
2. Actor (policy) generates completions for each prompt [inference]
3. Reward model scores each (prompt, completion) pair
4. Reference model computes log-probs for each completion (KL penalty)
5. Critic estimates value function for each token
6. PPO update: gradient step on Actor and Critic using reward + KL penalty
7. Repeat from 1
```

**The infrastructure challenge**: Steps 2–4 require running three models in inference mode simultaneously (Actor, Reward, Reference), while step 6 requires gradient computation on two models (Actor, Critic). The memory and compute requirements are substantially higher than pre-training a model of the same size.

### Ray as the RLHF Orchestrator

**Ray** is OpenAI's orchestration layer for the RLHF training loop. Anyscale confirms: "Ray was instrumental in training OpenAI's ChatGPT 3.5 and 4.0." The architecture (inferred from the OpenRLHF open-source implementation, which follows OpenAI's approach):

```
Ray Placement Groups:
┌─────────────────┐   ┌─────────────────┐
│   Actor GPUs    │   │   Critic GPUs   │
│ (policy model)  │   │ (value model)   │
│ gradient-on     │   │ gradient-on     │
└─────────────────┘   └─────────────────┘
        ↕ weight sync every N steps
┌─────────────────┐   ┌─────────────────┐
│  Reference GPUs │   │  Reward GPUs    │
│ (SFT model,     │   │ (reward model,  │
│  frozen)        │   │  frozen)        │
└─────────────────┘   └─────────────────┘
```

Ray's `placement_group` API places each model on a dedicated GPU group, coordinating weight synchronization and data movement between groups. Ray handles scheduling, failure recovery, and load balancing across the four model groups.

**vLLM for rollouts**: During step 2 (sampling completions), **vLLM** is used to accelerate inference — PagedAttention eliminates KV cache fragmentation, allowing much higher throughput during the rollout phase than naive HuggingFace inference. Ray manages the vLLM inference servers as actors.

**DeepSpeed ZeRO2**: The Actor and Critic require gradient computation, so they use **DeepSpeed ZeRO2** with gradient checkpointing to reduce memory footprint. ZeRO2 shards optimizer states and gradients (not weights) across data-parallel workers — reducing optimizer state memory by 4x relative to standard data parallel.

### PPO Hyperparameters and Stability

The "Secrets of RLHF" paper (closest public documentation of production RLHF) reveals:
- **KL coefficient (η)**: 0.05 — controls how far the policy is allowed to stray from the reference. OpenAI's early InstructGPT used ~0.001; more recent work uses 0.02–0.1. Too low: the policy ignores the KL penalty and reward-hacks. Too high: no learning.
- **Reward clipping**: Clip rewards to [−5, 5] to prevent extreme reward signals from destabilizing training.
- **Adaptive KL penalty**: If KL divergence drifts above a threshold, the KL coefficient is increased; if below, decreased. This keeps the policy in a "trust region" relative to the reference.
- **PPO-ptx**: Mix pre-training tokens into the PPO loss (weight ~0.1) to prevent catastrophic forgetting of pre-training capabilities. This is the "ptx" in PPO-ptx.

### RLAIF and Constitutional AI Influence

For newer models, AI feedback at scale replaces (or supplements) human annotators. Rather than collecting hundreds of thousands of human comparison pairs, a judge model (GPT-4, Claude) evaluates response pairs and generates preference labels. OpenAI's Model Spec (public, December 2025) serves as the written guidelines for both human annotators and LLM-based judgment systems — effectively externalizing the "constitution" that drives preference labeling.

The **RLVR (RL from Verifiable Rewards)** approach used for o1/o3 goes further: for problems with ground-truth verifiable answers (math, code), the reward signal is binary (correct/incorrect), eliminating the reward model entirely. The reported training recipe:
- Hundreds to thousands of challenging problems
- Hundreds of epochs of RL
- Binary reward from programmatic verifier (code execution, math solver) or LLM-as-judge for subjective criteria
- ~10x the compute of o1 was used to train o3

---

## Phase 6: Evals

### The Evals Framework

OpenAI open-sourced their evaluation framework at [github.com/openai/evals](https://github.com/openai/evals). It is not just a benchmark runner — it is the infrastructure OpenAI uses internally for model quality assurance. The README states: "OpenAI staff actively review these evals when considering improvements to upcoming models."

**Eval types**:
- **Match**: `a.startswith(b)` — exact prefix match
- **Includes**: `b in a` — substring match
- **FuzzyMatch**: bidirectional inclusion
- **JsonMatch**: JSON object equality
- **Model-graded**: LLM-as-judge, where GPT-4 evaluates a response against a YAML rubric

**Storage**: Eval results are logged to **Snowflake** — the framework has native `SNOWFLAKE_ACCOUNT` and `SNOWFLAKE_DATABASE` environment variable support. This enables querying eval results across model versions, running time-series analysis of benchmark performance, and triggering alerts when scores regress.

**Infrastructure**: Evals run as async API calls with configurable parallelism. A full MMLU eval (14K questions × multiple model calls per question) takes minutes at maximum parallelism. Evals are designed to be cheap enough to run on every model checkpoint during training, not just at release time.

### Eval Hierarchy

```
Level 1: Automated benchmarks (continuous, per-checkpoint)
  MMLU, HumanEval, GSM8K, MATH, HellaSwag, WinoGrande, TruthfulQA
  → Minutes to run; noisy signal on any individual checkpoint
  → Used to detect regressions during training

Level 2: LLM-judge evals (daily, on recent checkpoints)
  AlpacaEval, MT-Bench, Arena-Hard
  → Hours to run; reliable signal on capabilities
  → Used to track post-training effectiveness

Level 3: Human evals (weekly/release-gated)
  Internal human preference studies, Chatbot Arena (external)
  → Days to run; ground truth signal
  → Used for release decisions

Level 4: Safety evals (continuous, release-gated)
  Safety Evaluations Hub: harmful content, hate speech, self-harm, jailbreaks
  → Specialized eval team runs this; must pass before any deployment
```

### Process Reward Model Evals

For reasoning models (o1, o3), a specialized **Process Reward Model (PRM)** scores intermediate reasoning steps, not just final answers. OpenAI released ~800,000 step-level human annotations ("Let's Verify Step by Step," 2023) to train and evaluate PRMs. The PRM eval infrastructure scores:
- Correctness of each reasoning step (not just the final answer)
- Identification of the first incorrect step in a chain
- Pass@k accuracy across multiple reasoning paths

---

## Phase 7: The Agentic Layer — Kepler

### What Kepler Is

**Kepler** is OpenAI's internal data agent — a GPT-5-powered system deployed to all ~3,500 OpenAI employees for querying internal data. Published January 2026. It is simultaneously an example of the most advanced internal tooling at any AI lab and a direct illustration of the data management gap Raise is designed to fill.

**Scale**:
- ~600 petabytes of new data generated daily
- 70,000+ internal datasets
- Accessible via Slack, Cursor IDE, mobile clients
- Reduces data request iteration time by ~75%

### Architecture

**RAG over metadata, not data**: Kepler does not scan 600PB of raw logs. It maintains an index over the descriptions, schemas, and query history of each of the 70,000 datasets. Natural language questions are routed to this metadata index to identify the relevant datasets, then targeted queries are run.

**Codex-powered table knowledge**: Beyond metadata, Kepler crawls the codebase using Codex to understand how each dataset is constructed — pipeline logic, freshness guarantees, business intent that never surface in SQL schemas. A table named `user_daily_metrics` tells you nothing about whether it includes bot traffic; Kepler knows because it read the pipeline code that generates the table.

**SQL generation + validation**: Kepler generates multiple SQL/Python query candidates per question. "Codex tests" automatically validate syntax and logic before execution — a form of execution-grounded hallucination prevention.

**MCP connectivity**: Kepler uses the **Model Context Protocol (MCP)** to connect to internal tools — Slack, IDEs, database engines, Snowflake — in a standardized way. MCP is Anthropic's open protocol, adopted internally at OpenAI, which illustrates how quickly agent connectivity protocols become infrastructure-layer decisions.

**Memory**: Kepler maintains adaptive memory across sessions, learning from past interaction patterns. A researcher who always queries by cohort automatically gets cohort-stratified results.

### Why Kepler Exists

OpenAI built Kepler because no commercial tool handles the combination of:
1. Cross-dataset queries at 70,000-dataset scale
2. Pipeline-aware context (understanding table semantics from code, not just schema)
3. Conversational iteration with memory
4. Organizational context (knowing which team owns which data)

This is the same gap Raise targets from the other direction — Raise provides the managed data layer, and a Kepler-like agent sits on top of it. The difference: Kepler was built after the data was already scattered across 70,000 namespaced datasets with no unified schema. Raise is designed to prevent that situation by providing the registry and lineage infrastructure from day one.

---

## The Workflow End-to-End

```
Web Crawl (CommonCrawl WARC files, 45+ TB)
    │
    ▼ Quality classifier (logistic regression, trained on WebText/Books/Wiki)
    │ 98.7% discard rate → ~570 GB retained
    ▼ Dedup (MinHash LSH + 50-token exact span matching)
    │
    ▼ Mixing (weighted oversampling: WebText/Books/Wiki at 6–12x their corpus fraction)
    │
    ▼ Small-scale experiments
    │   Scaling law extrapolation at 1/1000th compute
    │   Ray Tune hyperparameter search
    │   W&B experiment tracking
    │   Data ablations: train 1B model on each candidate mix, compare benchmarks
    │
    ▼ Hero run
    │   Azure NDm A100 v4 (8× A100 per node × thousands of nodes)
    │   3D parallel: tensor (8-way) + pipeline (16-way) + data (N-way)
    │   NCCL + InfiniBand (200 Gbps per GPU) for gradient sync
    │   Triton / PyTorch Inductor for custom GPU kernels
    │   Checkpoint to Azure Blob Storage every ~30–60 min
    │   W&B real-time monitoring: loss, grad norm, GPU utilization
    │
    ▼ SFT
    │   Scale AI + direct contractors collect ~13K demonstration pairs
    │   Fine-tune base model: hours on same Azure cluster (small fraction)
    │   Mix pre-training tokens to prevent capability forgetting
    │
    ▼ Reward model training
    │   ~330K human comparison pairs (Scale AI)
    │   Initialize from SFT weights, fine-tune on preference ranking objective
    │
    ▼ RLHF / PPO
    │   Ray orchestrates 4 model groups: Actor, Critic, Reward, Reference
    │   vLLM for fast policy rollouts
    │   DeepSpeed ZeRO2 + gradient checkpointing for Actor/Critic memory
    │   PPO-ptx: mix pre-training tokens to prevent forgetting
    │   KL penalty (η=0.05) against reference model
    │   Continuous W&B monitoring: reward distribution, KL divergence
    │
    ▼ Evals
    │   Continuous: MMLU, HumanEval, GSM8K per checkpoint (openai/evals → Snowflake)
    │   Daily: AlpacaEval, MT-Bench
    │   Release-gated: Human preference evals, Safety Evaluations Hub
    │
    ▼ Kepler (ongoing)
        70,000 internal datasets, 600 PB/day
        GPT-5 + RAG over metadata + Codex table knowledge
        MCP connectivity to Slack, IDEs, databases
        75% reduction in data request iteration time
```

---

## What This Implies for Raise

The OpenAI stack illustrates both the problem and the opportunity:

1. **The data preparation layer has no unified system.** CommonCrawl → filtered corpus is custom Python + distributed compute. Signals are columns in Parquet files, not versioned features in a registry.

2. **Mixing is configured per-run, not managed.** The 60%/22%/8%/8%/3% split for GPT-3 was a research decision encoded in a config file, not a versioned `DatasetMix` object with lineage.

3. **The RLHF data pipeline has no formal schema.** Preference pairs, reward model scores, annotator metadata — all stored in JSON files in S3 with naming conventions.

4. **Evals are tracked in Snowflake but not linked to training data.** There is no formal connection between `EvalResult("mmlu", checkpoint="gpt-4-step-50000")` and `DatasetVersion("gpt-4-pretrain", "v1.0")`.

5. **Kepler is a workaround.** OpenAI built a GPT-5-powered data agent to navigate 70,000 internal datasets because there is no unified registry. Raise provides the registry before the data becomes ungovernable.

Raise's role in this stack is not to replace Ray (execution), Azure (compute), Scale AI (annotation), or W&B (experiment tracking). It is to be the **data management layer** connecting them — the system that knows which version of which signal ran on which corpus snapshot, which DatasetVersion was used for which hero run, and which preference pair batch produced the reward model checkpoint that produced the RLHF policy that scored 87.3 on AlpacaEval.

---

## Sources

### Data Engineering Infrastructure
- [Apache Iceberg: A Table Format for Huge Analytic Datasets](https://iceberg.apache.org/docs/latest/)
- [Delta Lake: High-Performance ACID Table Storage](https://docs.delta.io/latest/index.html)
- [MosaicML Streaming (MDS format)](https://docs.mosaicml.com/projects/streaming/en/stable/)
- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Great Expectations: Data Quality](https://docs.greatexpectations.io/)
- [DuckDB: In-Process Analytical Database](https://duckdb.org/docs/)
- [Lilac: Dataset Exploration Tool](https://lilacml.com/)
- [Dolma Tagger Framework — AllenAI](https://github.com/allenai/dolma)
- [CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data](https://arxiv.org/abs/1911.00359)
- [Meta: Composable Data Management](https://engineering.fb.com/2024/05/22/data-infrastructure/composable-data-management-at-meta/)
- [Meta: FBLearner Flow](https://engineering.fb.com/2016/05/09/core-infra/introducing-fblearner-flow-facebook-s-ai-backbone/)
- [DataHub: Data Discovery and Lineage](https://datahubproject.io/)
- [OpenLineage: Open Standard for Data Lineage](https://openlineage.io/)

### Pre-Training Data
- [Language Models are Few-Shot Learners (GPT-3) — arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- [GPT-4 Technical Report — arXiv:2303.08774](https://arxiv.org/abs/2303.08774)
- [GPT-4 System Card](https://cdn.openai.com/papers/gpt-4-system-card.pdf)
- [OpenAI: Approach to Data and AI](https://openai.com/index/approach-to-data-and-ai/)

### Scaling Laws and Small-Scale Experiments
- [Scaling Laws for Neural Language Models — arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- [OpenAI Scaling Laws blog](https://openai.com/index/scaling-laws-for-neural-language-models/)

### Cluster and Training Infrastructure
- [Azure NDm A100 v4 Series](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/ndma100v4-series)
- [Azure GB300 NVL72 announcement for OpenAI](https://azure.microsoft.com/en-us/blog/microsoft-azure-delivers-the-first-large-scale-cluster-with-nvidia-gb300-nvl72-for-openai-workloads/)
- [SemiAnalysis: 100,000 H100 Clusters](https://newsletter.semianalysis.com/p/100000-h100-clusters-power-network)
- [Techniques for Training Large Neural Networks — OpenAI](https://openai.com/index/techniques-for-training-large-neural-networks/)
- [NVIDIA GPT-3 blog](https://developer.nvidia.com/blog/openai-presents-gpt-3-a-175-billion-parameters-language-model/)

### Triton
- [Introducing Triton — OpenAI](https://openai.com/index/triton/)
- [Triton GitHub](https://github.com/triton-lang/triton)
- [NVIDIA: Triton on Blackwell](https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/)

### Experiment Tracking
- [W&B OpenAI integration](https://docs.wandb.ai/guides/integrations/openai/)
- [OpenAI Fine-tuning with W&B (Cookbook)](https://cookbook.openai.com/examples/third_party/gpt_finetuning_with_wandb)

### Ray and RLHF Infrastructure
- [Anyscale: Ray and ChatGPT](https://www.anyscale.com/blog/open-source-rl-libraries-for-llms)
- [OpenRLHF: Ray-based RLHF framework](https://github.com/OpenRLHF/OpenRLHF)
- [Secrets of RLHF Part I — arXiv:2307.04964](https://arxiv.org/html/2307.04964v1)

### Post-Training
- [InstructGPT — arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- [Introducing ChatGPT — OpenAI](https://openai.com/index/chatgpt/)
- [OpenAI Reinforcement Fine-Tuning](https://platform.openai.com/docs/guides/reinforcement-fine-tuning)
- [Improving Mathematical Reasoning with Process Supervision — OpenAI](https://openai.com/index/improving-mathematical-reasoning-with-process-supervision/)
- [interconnects.ai: Reverse-Engineering o1](https://www.interconnects.ai/p/reverse-engineering-openai-o1)
- [Scale AI on InstructGPT](https://exchange.scale.com/public/blogs/openais-instructgpt-2022-11-18)

### Evals
- [openai/evals GitHub](https://github.com/openai/evals)
- [OpenAI Safety Evaluations Hub](https://openai.com/safety/evaluations-hub/)

### Kepler Data Agent
- [OpenAI Blog: Inside our in-house data agent](https://openai.com/index/inside-our-in-house-data-agent/)
- [The New Stack: Kepler](https://thenewstack.io/kepler-openais-internal-agent-platform-for-synthesizing-data/)
- [OpenAI Developers on X: Kepler announcement](https://x.com/OpenAIDevs/status/2016943147239329872)

### OpenAI Model Spec
- [OpenAI Model Spec — December 2025](https://model-spec.openai.com/2025-12-18.html)
