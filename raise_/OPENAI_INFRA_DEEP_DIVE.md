# OpenAI Infrastructure Deep Dive

A workflow-by-workflow account of OpenAI's data and training infrastructure — from the first web crawl byte to a deployed model checkpoint. Organized by lifecycle phase, with the underlying infrastructure for each phase called out explicitly.

OpenAI is deliberately opaque about their stack starting with GPT-4 ("the report contains no further details about the architecture, hardware, training compute, or dataset construction"). What follows is sourced from primary documents (papers, blog posts, open-source repositories, conference talks, job postings) and noted where it is reverse-engineered inference.

---

## The Stack at a Glance

| Phase | Primary Infrastructure |
|---|---|
| Data acquisition | CommonCrawl WARC files, Azure Blob Storage |
| Signal computation | Custom Python + distributed compute (Ray/Spark), FastText, KenLM |
| Dedup | MinHash LSH + exact 50-token span matching |
| Experiment (small scale) | Scaling law extrapolation, Ray Tune, W&B |
| Hero run (training) | Azure NDm A100 v4 → GB300 NVL72, NCCL + InfiniBand, PyTorch + Triton |
| Parallelism | 3D parallel: tensor + pipeline + data (PTD-P) |
| Checkpointing | Azure Blob Storage, every ~30–60 min |
| Monitoring | W&B + custom dashboards |
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
