# Data Infrastructure at AI Labs

How leading AI research labs manage training data — pre-training signal pipelines, post-training data workflows, annotation systems, and the tooling gap Raise is designed to fill.

---

## The Vocabulary Problem

Before comparing labs, one naming clarification is necessary. The term "feature engineering" is conspicuously absent from foundation model lab discourse. Labs don't talk about "feature stores" or "derived features" for pre-training. They talk about **quality signals**, **data enrichment**, **classifiers**, and **curation pipelines**.

The underlying operation is identical:

| Traditional ML | Foundation Model Equivalent |
|---------------|---------------------------|
| Derive `ctr = clicks / impressions` | Compute `perplexity_score` from a language model |
| Assign `user_segment` via classifier | Assign `educational_quality_score` via LLM judge |
| Compute `is_fraud` via rule engine | Compute `is_nsfw` via safety classifier |
| Flag `duplicate_user_id` | Flag `near_duplicate_document` via MinHash |
| Store in feature group with lineage | Store as Parquet column with... no lineage |

The last row is the gap. Traditional ML feature stores provide versioning, lineage, and a managed lifecycle for derived signals. Pre-training pipelines compute equivalent signals but store them as columns in Parquet files with no unified registry, no lineage tracking, and no versioning. Every dataset has a custom pipeline. Nothing is reused across datasets.

---

## Universal Pre-Training Pipeline Architecture

Despite differences in tooling, all major labs converge on the same logical stages for pre-training data preparation:

```
Stage 1: Acquisition
  Web crawl (WARC) / licensed datasets / synthetic generation
  → URL deduplication, robots.txt compliance, domain filtering

Stage 2: Extraction
  Raw HTML → clean text (Trafilatura, Resiliparse, custom extractors)
  Multimodal: image extraction, alt-text alignment, OCR

Stage 3: Language Identification
  FastText or CLD3 per document
  → language code, confidence score, script detection

Stage 4: Quality Scoring
  Perplexity-based (KenLM against reference corpus)
  Heuristic rules (line-length ratio, symbol density, etc.)
  LLM-based (Llama-70B scoring educational value 0–5)
  → composite quality_score + per-dimension subscores

Stage 5: Deduplication
  Exact dedup: MD5/SHA256 hash of normalized text
  Near-dedup: MinHash LSH (Jaccard) or SimHash (Hamming)
  Semantic dedup: embedding cosine similarity (rarer; expensive)
  → is_duplicate flag, cluster_id, dedup_method

Stage 6: Safety & Compliance
  NSFW classifier (image + text)
  Hate speech classifier
  PII detection (regex + NER models)
  Copyright signals (URL heuristics, DMCA lists)
  → per-signal confidence scores, action flags

Stage 7: Disposition
  Combine quality + dedup + safety signals
  → include_in_training boolean
  Apply mixing weights by domain/language/source

Stage 8: Tokenization & Packing
  BPE or SentencePiece tokenization
  Sequence packing into fixed-length chunks
  → token count, packing metadata

Stage 9: Dataset Versioning
  Snapshot with filters, signal versions, mixing weights
  → reproducible training run linkage
```

No lab has a single unified system that manages all nine stages with shared lineage. Most have custom tooling per stage, with hand-offs between Parquet files and custom metadata sidecars.

---

## Lab Deep Dives

### OpenAI

**Pre-training data infrastructure** is not publicly documented, but the architecture can be inferred from job postings, published research, and one unusually transparent post about their internal data tooling.

**Compute layer**: Ray is the primary distributed execution framework. OpenAI has been the most vocal Ray user at scale; their training orchestration, hyperparameter search, and inference all run on Ray. At the 2023 Ray Summit, OpenAI described using Ray to coordinate training across thousands of GPUs with custom fault tolerance.

**Data pipeline**: The specifics are private, but the FineWeb paper (from HuggingFace, modeling GPT-4 data practices) implies an LLM-based quality scoring stage. OpenAI's own work on data filtering (e.g., the WebText dataset used for GPT-2) established the pattern of using model perplexity as a quality proxy — a pattern now universal.

**The internal data agent**: The most revealing public insight comes from OpenAI's engineering blog post ["Inside our in-house data agent"](https://openai.com/index/inside-our-in-house-data-agent/):

> "OpenAI runs a daily offline pipeline that aggregates table usage, human annotations, and Codex-derived enrichment into a single, normalized representation."

This describes a Codex-powered agent that reads internal data tables, understands their semantics, and writes enriched metadata back. In other words: an LLM agent querying and writing to what is functionally a feature/signal store. OpenAI built an agentic data management system because no off-the-shelf tool existed for this use case.

**Post-training data**: OpenAI uses Scale AI as a primary annotation vendor for RLHF data collection. The preference pairs and safety demonstrations that trained InstructGPT and ChatGPT were collected at scale through Scale's platform, then used in supervised fine-tuning and PPO training. The data pipeline connecting raw annotations to training batches is internal and not documented publicly.

**The gap**: OpenAI has purpose-built tooling at each stage (Ray for compute, Scale for annotation, custom pipelines for curation) but no unified data management layer. The Codex data agent is itself evidence that they felt the gap — they built an LLM to understand their own data rather than building a proper registry.

---

### Anthropic

**Philosophy**: Anthropic's most distinctive contribution to training data philosophy is **Constitutional AI (CAI)** and the model spec as a data-generating document. Rather than relying solely on human annotators to judge response quality, Anthropic trains models to critique and revise their own outputs against a written constitution. This means:

1. A supervised fine-tuning dataset is generated by having an early model revise harmful responses
2. An RLAIF (RL from AI Feedback) preference dataset is generated by having the model judge its own output against constitutional principles
3. Human annotators focus on edge cases and calibration rather than bulk labeling

This is architecturally significant: a significant portion of Anthropic's training data is **generated by the model being trained**, which creates a very different data pipeline from labs that rely primarily on human annotation.

**Infrastructure**: Anthropic runs on multi-cloud: AWS (Trainium2 chips via Project Rainier — 500K+ chips), Google TPUs (Project Dario — 1M+ TPUs), and NVIDIA GPUs. The data pipelines that feed these clusters are not publicly documented but the scale of compute implies proportionate scale of data preparation.

**Model Context Protocol (MCP)**: Anthropic's MCP is explicitly designed to let agentic systems connect to data sources in a standardized way. While MCP is positioned as an external integration protocol (connecting Claude to databases, APIs, file systems), the design philosophy is directly relevant to how agentic data management works: structured, declarative access to data with clear capabilities and permissions. A Raise-backed MCP server would give agents standardized access to feature groups and training datasets.

**Post-training data**: Constitutional AI means Anthropic's RLHF loop involves AI feedback at scale, with human annotators primarily in a calibration role. The data formats (preference pairs, Constitutional AI revisions, model spec critiques) are non-standard relative to industry norms — Anthropic has had to build much of this data infrastructure themselves.

**The gap**: Anthropic faces the same signal management gap as other labs. Constitutional AI generates structured preference data at scale, but the pipeline connecting constitutional principles → generated revisions → training batches is custom and not generalizable without a proper data management layer.

---

### Google DeepMind

**Research vs. Production split**: DeepMind operates with a clearer research/production split than most labs.

- **Research**: JAX is dominant. The functional, XLA-compiled, TPU-native execution model is ideal for research that needs to iterate quickly on novel architectures. JAX's `jit`, `vmap`, and `pmap` primitives map directly to TPU parallelism.
- **Production/Platform**: TFX (TensorFlow Extended) handles production ML pipelines. TFX provides a full pipeline SDK with components for data validation (TFDV), transformation (TFT), training, evaluation, and serving — the closest existing tool to what Raise is trying to do, but built around TensorFlow and requiring significant infra expertise.

**Vertex AI Feature Store**: Google offers a managed feature store through Vertex AI, but it targets enterprise ML teams doing traditional ML, not foundation model researchers. The pre-training pipelines for Gemini don't use Vertex AI Feature Store — they use internal Google-scale infrastructure that isn't publicly documented.

**Data scale**: Gemini training data is sourced from Google's massive internal corpus (Search index, YouTube transcripts, Books, Scholar, etc.) plus web crawl. The signals computed on this data are not public, but the Gemini Technical Report describes multimodal training data that spans text, images, video, and audio — implying a signal pipeline at least as complex as the text-only public pipelines.

**SynthID and data provenance**: DeepMind's SynthID watermarking system (for both text and images) reflects a strong interest in data provenance — knowing which content is synthetic vs. human-generated. This is a signal management problem: every synthetic document needs a `is_synthetic`, `generation_model`, and `generation_prompt` annotation stored alongside it.

**The gap**: TFX exists but is heavyweight (Beam, Kubeflow, Airflow required) and not researcher-friendly. JAX researchers typically bypass TFX entirely and write ad-hoc data prep scripts. There is no unified system that works for both research iteration speed and production data governance.

---

### Meta AI

Meta is the most instructive case study because they have the most publicly documented ML infrastructure and have solved different versions of the feature store problem at different layers of the organization.

**FBLearner and Palette (Applied ML)**: Meta's production ML platform — FBLearner Flow — includes Palette, widely credited as the industry's first feature store. Palette is a crowd-sourced feature marketplace: teams publish features (engagement signals, social graph features, ad signals) and other teams consume them. Features in Palette are used across News Feed ranking, Ads prediction, Friend recommendation, and dozens of other product surfaces.

Palette is architecturally mature: features are computed in Spark/Hive, stored in HDFS with Cassandra for low-latency serving, and managed with a central registry. It is exactly what Raise aspires to be — but built for applied ML on structured behavioral data, not for foundation model pre-training.

**FAIR (Research)**: Meta's fundamental AI research arm operates differently from the production ML teams. FAIR researchers primarily use PyTorch (which Meta developed) and have much lighter data management infrastructure than the Palette-equipped production teams. Pre-training for LLaMA and other open research models uses custom pipelines rather than Palette.

**LLaMA data pipeline** (public): The LLaMA 2 and 3 papers describe a data pipeline with the same structure as the universal architecture above: web crawl → quality filtering (perplexity, heuristics) → deduplication (MinHash) → safety filtering → tokenization. The LLaMA 3 paper notes:

> "We performed extensive data cleaning... using a series of quality, diversity, and safety filters."

But the pipeline is custom Python/Spark, not a reusable system. The LLaMA 3.1 dataset (15T+ tokens) was prepared by a dedicated data team using tooling built specifically for that dataset.

**Meta Workflow Service (MWFS)**: Meta's more recent orchestration system (post-FBLearner) is event-driven and horizontally scalable. The design philosophy is similar to Raise's Job model: declare what computation should happen, the system handles scheduling and execution. But MWFS is a compute orchestrator, not a data management system — it doesn't handle feature versioning, lineage, or signal registration.

**The gap**: Meta has the most sophisticated applied ML data infrastructure in the industry, but it was built for structured behavioral features, not for document-level quality signals. FAIR researchers building foundation models don't use Palette — they write custom pipelines that reinvent the same patterns at smaller scale.

---

### Microsoft

**DeepSpeed and ZeRO**: Microsoft's primary contribution to the training infrastructure ecosystem is DeepSpeed, specifically the ZeRO (Zero Redundancy Optimizer) memory optimization technique. ZeRO enables training of very large models on relatively modest GPU clusters by sharding optimizer states, gradients, and parameters across devices. This is a compute infrastructure contribution, not a data infrastructure one.

**Azure ML and the declarative DSL**: Microsoft's Azure ML Managed Feature Store is the most direct commercial competition to Raise's design philosophy. Notably, Microsoft added a declarative DSL for feature definitions (in preview as of 2025) — the same pattern Raise uses. This validates the direction: a major cloud provider independently converged on declarative feature declaration as the right ergonomic model.

```python
# Azure ML's declarative feature definition (preview)
# Conceptually similar to Raise's approach
feature_set_spec = FeatureSetSpec(
    source=FeatureSource(type="parquet", path="abfss://..."),
    feature_transformation=SparkTransformation(code="./transforms/"),
    features=[
        Feature(name="clicks", type=FeatureDataType.INTEGER),
        Feature(name="ctr",    type=FeatureDataType.FLOAT),
    ],
)
```

**OpenAI partnership**: Microsoft's infrastructure partnership with OpenAI (Azure as OpenAI's primary cloud) has exposed Microsoft to OpenAI's data pipeline practices at close range. The Azure ML roadmap appears to be incorporating lessons from this partnership — the move toward declarative, researcher-friendly APIs is likely informed by observing what researchers actually want.

**The gap**: Azure ML Feature Store is designed for Azure-native enterprise ML teams, not for foundation model researchers. It has no native curation pipeline, no pre-training signal management, and no LLM-as-judge annotation support.

---

## Public Datasets as Windows into Lab Practice

Public datasets produced by research labs and the ML community provide the most detailed view into how pre-training data pipelines actually work. Each represents a team's best practice at the time of publication.

### RedPajama-V2 (Together AI, 2023)

RedPajama-V2 is the most transparent example of a production-quality signal pipeline. It computes approximately 40 signals per document across five categories, all stored as metadata alongside the document:

**Quality signals:**
- `perplexity_score` — KenLM perplexity against a Wikipedia reference model
- `avg_word_length`, `avg_line_length` — structural quality heuristics
- `fraction_unique_words`, `fraction_unique_bigrams` — lexical diversity
- `fraction_all_caps`, `fraction_lines_end_with_ellipsis` — formatting quality
- `unigram_entropy` — Shannon entropy of token distribution

**Content signals:**
- `url_domain`, `url_path_depth`, `url_num_queries` — source metadata
- `detected_language`, `language_confidence` — language identification
- `has_math`, `has_code` — content type signals

**Deduplication signals:**
- `minhash_signature` — 128-bit MinHash for Jaccard near-dedup
- `exact_line_dedup_count` — exact duplicate line fraction

This is a feature group. The 40 signals are derived features computed on raw documents. RedPajama-V2 stores them as Parquet metadata columns — but with no versioning, no lineage tracking, no centralized registry. Each consumer re-reads and re-filters from scratch.

**Scale**: 30 trillion tokens, ~100 billion documents. Signal computation at this scale requires tens of thousands of GPU-hours for the ML-based signals alone.

### FineWeb (HuggingFace, 2024)

FineWeb's key innovation is replacing heuristic quality filters with a learned quality classifier, trained using GPT-4 annotations:

1. Sample 500K documents from CommonCrawl
2. Score each with GPT-4 on educational quality (0–5 scale)
3. Train a FastText classifier to reproduce GPT-4 judgments at scale
4. Apply FastText to 15 trillion tokens

> "Applying the classifier to 15 trillion tokens required 6,000 H100 GPU hours."

This is the LLM-as-judge pattern at training data scale. The pipeline is:
```
raw document → GPT-4 annotation (sample) → FastText training → FastText scoring (full corpus)
```

In Raise terms: a `HumanEvalTask` with `pool_type=AnnotatorPoolType.MODEL` (GPT-4) generates labels on a sample; a `QualityScorer` transform applies a trained classifier across the full corpus.

FineWeb-Edu (the educational subset) applies this at 1.3T tokens and produces substantially better models on knowledge benchmarks than equivalently-sized data from raw web text.

### Dolma (AllenAI, 2024)

Dolma takes a different architectural approach: rather than a monolithic pipeline, it provides an **extensible toolkit** where each processing stage is a pluggable tagger:

```python
# Dolma tagger interface (conceptual)
class LanguageTagger(BaseTagger):
    def predict(self, doc: Document) -> DocResult:
        lang = detect_language(doc.text)
        return DocResult(doc=doc, spans=[
            SpanResult(start=0, end=len(doc.text),
                       type="language", score=lang.confidence,
                       label=lang.code)
        ])
```

Each tagger writes structured spans or document-level scores. The pipeline is composable: run any subset of taggers, in any order, and outputs are merged. This is conceptually similar to Raise's `CurationPipeline` — ordered, composable stages with structured output columns.

Dolma taggers cover: language identification, perplexity scoring, URL quality, toxic content detection, copyright signals, PII detection, and deduplication. The design explicitly supports adding new taggers without modifying the pipeline.

**What Dolma gets right**: Composable, extensible stages with well-defined interfaces. **What it lacks**: No versioning of tagger outputs, no lineage between raw corpus and curated version, no centralized registry.

### DCLM (Apple / collaborators, 2024)

DCLM (DataComp for Language Models) frames data curation as a **benchmark problem**: given a fixed compute budget, which filtering decisions produce the best downstream model quality? It is the most rigorous scientific study of what data quality signals actually matter.

Key findings:
- FastText classifiers trained on GPT-3/4 outputs substantially outperform heuristic filters
- Deduplication improves quality up to a point, then has diminishing returns
- Language filtering to English-only hurts multilingual capability but improves English benchmark scores
- The specific quality threshold matters more than the choice of classifier architecture

DCLM computed signals on 240 trillion tokens — the largest publicly documented data pipeline. At this scale, even microseconds-per-document overhead multiplies to days of wall-clock time; efficiency of the signal computation is as important as signal quality.

---

## Post-Training Data Infrastructure

Post-training data management — SFT datasets, preference data, reward model training, evaluation suites — is a distinct problem from pre-training curation, and labs handle it differently.

### The RLHF Data Pipeline

The canonical post-training pipeline:

```
1. Prompt collection
   Human-written prompts + templated prompts + adversarial prompts
   → prompt_id, prompt_text, domain, difficulty

2. Response generation
   Generate N responses per prompt from current model + reference models
   → response_id, prompt_id, response_text, model_id, generation_params

3. Human preference annotation
   Present pairs (A vs. B) to annotators; collect preference labels
   → pair_id, annotator_id, preferred, confidence, duration_sec

4. Quality filtering
   Filter ties, low-agreement pairs, fast annotations (potential cheating)
   → annotator_agreement, is_gold, include_in_training

5. DPO / PPO preparation
   Convert to training format: (prompt, chosen, rejected) triplets
   For DPO: compute reference model log-probs offline
   → chosen_ref_logprob, rejected_ref_logprob, preference_margin

6. Reward model training
   Train scalar reward model on (prompt, response, score) pairs
   → reward_score per response for PPO training signal
```

Every stage produces a structured artifact that needs to be versioned and linked to the next stage. In practice, labs store these as JSON files in S3 with naming conventions — no versioning, no lineage, no formal schema.

### Annotation at Scale: The Vendor Landscape

The annotation step (step 3 above) is where labs diverge most sharply:

**Scale AI / Labelbox / Surge**: External vendors used for high-volume annotation by OpenAI, Anthropic, and others. Scale AI in particular runs large annotator pools with quality control pipelines. The data leaves the lab, is labeled externally, and is returned in bulk. Latency: days to weeks per batch.

**Internal red teams**: For safety-critical data (jailbreaks, harmful content, edge cases), labs maintain internal red teams who understand the model's failure modes. This data is too sensitive for external annotation.

**AI feedback (RLAIF)**: Anthropic's Constitutional AI uses the model itself as an annotator at scale. The model reviews its own outputs against constitutional principles. Quality and speed are high; cost is compute, not headcount. Meta, Google, and others have since adopted versions of this.

**LLM-as-judge**: For evaluation tasks (is response A better than B?), labs increasingly use strong frontier models (GPT-4, Claude) as judges rather than humans. Correlation with human judgment is high for well-defined criteria; cost is orders of magnitude lower.

The trend is clear: human annotation for bulk preference data is being replaced by AI feedback, with humans retained for calibration, edge cases, and safety-critical decisions. This changes the data infrastructure requirements — AI-generated annotations need different quality controls than human annotations (no inter-annotator agreement; instead, ensemble voting across judge models and prompt variations).

### Evaluation Infrastructure

Evals are the feedback loop of post-training. Labs run hundreds of evaluations continuously across model checkpoints:

- **Automated benchmarks**: MMLU, HumanEval, GSM8K, MATH — scalar metrics, runnable in minutes
- **LLM judge evals**: AlpacaEval, MT-Bench, Arena — model-judged win rates against reference responses
- **Human evals**: expensive, slow, but required for final model release decisions
- **Red-teaming**: targeted adversarial prompts to find failure modes

The infrastructure challenge: tracking which model checkpoint scored how on which eval suite, correlating eval results with training dataset versions, and identifying which training data changes caused capability regressions. Most labs do this with spreadsheets and custom dashboards rather than a unified eval registry.

---

## The Unified Gap

Synthesizing across labs, the gap has a consistent shape:

**What exists (good):**
- Distributed compute (Ray, Spark, TPU clusters)
- Object storage (S3, GCS, HDFS)
- Annotation vendors (Scale AI, Labelbox)
- Individual pipeline tools (Dolma tagger framework, FineWeb classifier)

**What doesn't exist (the gap):**
- A **signal registry** with versioned definitions, lineage, and schema for pre-training signals
- A **managed curation pipeline** that tracks which version of which classifier ran on which corpus snapshot
- A **dataset version store** that links `DatasetVersion("llama-3-pretrain", "v1.0")` to the exact filters, signal thresholds, and mixing weights used
- A **post-training data store** that tracks preference pairs, reward scores, and annotation quality across RLHF iterations with lineage back to the annotators and models involved
- A **unified eval registry** that tracks model checkpoint × eval suite × score with queryable history

The result: when a model regresses between checkpoints, the team has to manually compare spreadsheets, Git history, and Parquet metadata to trace what changed. When a new dataset version is created, the lineage back to the raw crawl is in a README file. When annotation quality drops, there is no automated alert — someone notices in a meeting.

This is the problem Raise is designed to solve. Not by replacing Ray, Spark, or Scale AI — those are best-in-class for what they do — but by providing the data management layer on top of them that foundation model teams currently assemble ad-hoc from Parquet files and institutional memory.

---

## Sources

### Pre-Training Data Pipelines
- [RedPajama-V2 Dataset](https://www.emergentmind.com/topics/redpajama-dataset)
- [FineWeb: Decanting the Web for the Finest Text Data at Scale](https://arxiv.org/html/2406.17557v1)
- [Dolma: An Open Corpus of Three Trillion Tokens](https://allenai.org/dolma)
- [DCLM: Data-Centric Language Model Development](https://arxiv.org/html/2505.05427v1)
- [LLaMA 3 Technical Report](https://arxiv.org/abs/2407.21783)

### Lab Infrastructure
- [OpenAI: Inside our in-house data agent](https://openai.com/index/inside-our-in-house-data-agent/)
- [OpenAI at Ray Summit: Scaling LLMs](https://thenewstack.io/openai-chats-about-scaling-llms-at-anyscales-ray-summit/)
- [How Ray Powers ChatGPT](https://thenewstack.io/how-ray-a-distributed-ai-framework-helps-power-chatgpt/)
- [Anthropic Constitutional AI Paper](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback)
- [Anthropic Model Card and Evaluations](https://www.anthropic.com/research)
- [Anthropic: Model Context Protocol](https://www.anthropic.com/engineering)
- [Google DeepMind: Gemini Technical Report](https://deepmind.google/technologies/gemini/)
- [Google DeepMind: SynthID](https://deepmind.google/technologies/synthid/)
- [Meta: FBLearner Flow](https://engineering.fb.com/2016/05/09/core-infra/introducing-fblearner-flow-facebook-s-ai-backbone/)
- [Meta: Composable Data Management](https://engineering.fb.com/2024/05/22/data-infrastructure/composable-data-management-at-meta/)
- [Meta: LLaMA 2 Paper](https://arxiv.org/abs/2307.09288)
- [Meta GenAI Infrastructure](https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/)
- [Microsoft DeepSpeed](https://www.microsoft.com/en-us/research/project/deepspeed/)
- [Azure ML Managed Feature Store](https://learn.microsoft.com/en-us/azure/machine-learning/concept-what-is-managed-feature-store)

### Post-Training and Annotation
- [InstructGPT: Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [Scale AI Research](https://scale.com/research)
- [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/)
- [RLHF: Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)
