"""
Example 9: Bulk Inference Transformations

Demonstrates running batch inference jobs on AI models as transformations.
These jobs support GPU/TPU acceleration and integrate with model registries.

Key concepts:
- InferenceTransform: Transform that runs model inference
- ModelSpec: Model location, framework, and configuration
- AcceleratorConfig: GPU/TPU requirements
- BatchConfig: Batching and parallelism settings
"""

from raise_ import (
    FeatureStore,
    BlobRef,
    # Inference
    ModelSpec,
    AcceleratorConfig,
    InferenceTransform,
    GPUType,
    ModelFramework,
    embedding_inference,
    image_inference,
    llm_inference,
    # Transforms
    Schedule,
    Target,
    IncrementalConfig,
    FeatureGroupSource,
)
from raise_.transforms import (
    # Additional inference types
    TPUType,
    InferenceMode,
    ModelPrecision,
    BatchConfig,
    InferenceRuntime,
    InferenceResult,
    inference_transform,
    classification_inference,
)

# =============================================================================
# Setup
# =============================================================================

fs = FeatureStore("acme/mlplatform/embeddings")

# Create feature groups
text_features = fs.create_feature_group(
    "text-features",
    description="Text features with embeddings",
    if_exists="skip",
)

text_features.create_features_from_schema({
    "doc_id": "string",
    "text": "string",
    "text_embedding": "float32[384]",
    "sentiment": "string",
    "sentiment_score": "float64",
}, if_exists="skip")

image_features = fs.create_feature_group(
    "image-features",
    description="Image features with CLIP embeddings",
    if_exists="skip",
)

image_features.create_features_from_schema({
    "image_id": "string",
    "image_ref": BlobRef(content_types=["image/png", "image/jpeg"]),
    "image_embedding": "float32[512]",
    "caption": "string",
}, if_exists="skip")

print("Feature groups created")

# =============================================================================
# Model Specifications
# =============================================================================

print("\n" + "=" * 60)
print("MODEL SPECIFICATIONS")
print("=" * 60)

# HuggingFace model
hf_model = ModelSpec(
    uri="hf://sentence-transformers/all-MiniLM-L6-v2",
    framework=ModelFramework.HUGGINGFACE,
    version="main",
    name="MiniLM Embeddings",
    input_schema={"text": "string"},
    output_schema={"embedding": "float32[384]"},
)
print(f"\nHuggingFace model: {hf_model.uri}")
print(f"  Framework: {hf_model.framework.value}")
print(f"  Is HuggingFace: {hf_model.is_huggingface}")

# From HuggingFace Hub shortcut
clip_model = ModelSpec.from_huggingface(
    model_id="openai/clip-vit-base-patch32",
    revision="main",
    task="image-feature-extraction",
)
print(f"\nCLIP model: {clip_model.uri}")

# MLflow model
mlflow_model = ModelSpec.from_mlflow(
    model_uri="models:/sentiment-classifier/Production",
    version="3",
)
print(f"MLflow model: {mlflow_model.uri}")

# PyTorch model from S3
pytorch_model = ModelSpec(
    uri="s3://models/custom/text-encoder-v2.pt",
    framework=ModelFramework.PYTORCH,
    version="v2.1",
    precision=ModelPrecision.FP16,
    entry_point="TextEncoder",
)
print(f"PyTorch model: {pytorch_model.uri}")
print(f"  Precision: {pytorch_model.precision.value}")

# ONNX model for optimized inference
onnx_model = ModelSpec(
    uri="s3://models/optimized/embedding-model.onnx",
    framework=ModelFramework.ONNX,
    compile_options={"optimization_level": 99},
)
print(f"ONNX model: {onnx_model.uri}")

# =============================================================================
# Accelerator Configurations
# =============================================================================

print("\n" + "=" * 60)
print("ACCELERATOR CONFIGURATIONS")
print("=" * 60)

# CPU only
cpu_config = AcceleratorConfig.cpu(cores=8)
print(f"\nCPU config: {cpu_config.count} cores")

# Single GPU
single_gpu = AcceleratorConfig.gpu(
    gpu_type=GPUType.NVIDIA_T4,
    count=1,
    memory_gb=16,
)
print(f"Single GPU: {single_gpu.gpu_type.value}")

# Multi-GPU with data parallelism
multi_gpu = AcceleratorConfig.multi_gpu(
    gpu_type=GPUType.NVIDIA_A100,
    count=4,
    strategy="data_parallel",
)
print(f"Multi-GPU: {multi_gpu.count}x {multi_gpu.gpu_type.value}")
print(f"  Strategy: {multi_gpu.multi_gpu_strategy}")

# A100 80GB for large models
large_gpu = AcceleratorConfig.gpu(
    gpu_type=GPUType.NVIDIA_A100_80GB,
    count=1,
)
print(f"Large GPU: {large_gpu.gpu_type.value}")

# TPU for JAX/TensorFlow
tpu_config = AcceleratorConfig.tpu(
    tpu_type=TPUType.TPU_V4,
    count=8,
)
print(f"TPU config: {tpu_config.count}x {tpu_config.tpu_type.value}")

# =============================================================================
# Batch Configurations
# =============================================================================

print("\n" + "=" * 60)
print("BATCH CONFIGURATIONS")
print("=" * 60)

# Standard batching
standard_batch = BatchConfig(
    batch_size=64,
    max_concurrent_batches=2,
    prefetch_batches=1,
    timeout_seconds=300,
)
print(f"\nStandard batch: size={standard_batch.batch_size}")

# High-throughput batching
high_throughput = BatchConfig(
    batch_size=256,
    max_concurrent_batches=8,
    prefetch_batches=4,
    dynamic_batching=True,
    min_batch_size=32,
    max_batch_size=512,
)
print(f"High-throughput batch: size={high_throughput.batch_size}")
print(f"  Dynamic batching: {high_throughput.dynamic_batching}")

# Small batch for large models
small_batch = BatchConfig(
    batch_size=8,
    max_concurrent_batches=1,
    timeout_seconds=600,
)
print(f"Small batch (LLM): size={small_batch.batch_size}")

# =============================================================================
# Inference Runtime
# =============================================================================

print("\n" + "=" * 60)
print("INFERENCE RUNTIME")
print("=" * 60)

runtime = InferenceRuntime(
    image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
    python_version="3.10",
    packages=["transformers>=4.35", "accelerate", "sentence-transformers"],
    env_vars={
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_VISIBLE_DEVICES": "0",
    },
    memory_limit_gb=32,
    shm_size_gb=4,
)
print(f"Runtime image: {runtime.image}")
print(f"  Python: {runtime.python_version}")
print(f"  Packages: {runtime.packages}")

# =============================================================================
# Text Embedding Inference
# =============================================================================

print("\n" + "=" * 60)
print("TEXT EMBEDDING INFERENCE")
print("=" * 60)

# Using convenience constructor
text_embed_transform = embedding_inference(
    model_uri="hf://sentence-transformers/all-MiniLM-L6-v2",
    input_column="text",
    output_column="text_embedding",
    batch_size=128,
    gpu_type=GPUType.NVIDIA_T4,
)

print(f"\nText embedding transform: {text_embed_transform.name}")
print(f"  Model: {text_embed_transform.model.uri}")
print(f"  GPU: {text_embed_transform.accelerator.gpu_type.value}")
print(f"  Batch size: {text_embed_transform.batch_config.batch_size}")

# Create job
text_embedding_job = fs.create_job(
    name="generate_text_embeddings",
    description="Generate embeddings for text documents",
    sources=[
        FeatureGroupSource(
            feature_group="acme/mlplatform/embeddings/text-features",
            features=["doc_id", "text"],
        )
    ],
    transform=text_embed_transform,
    target=Target(
        feature_group="text-features",
        features={"text_embedding": "text_embedding"},
        write_mode="upsert",
        key_columns=["doc_id"],
    ),
    schedule=Schedule.daily(hour=2),
    incremental=IncrementalConfig.incremental("updated_at"),
    owner="ml-team@acme.com",
    tags=["embeddings", "inference", "gpu"],
)

print(f"\nCreated job: {text_embedding_job.name}")
print(f"  Status: {text_embedding_job.status.value}")
print(f"  Requires GPU: {text_embed_transform.requires_gpu}")

# =============================================================================
# Image Embedding Inference
# =============================================================================

print("\n" + "=" * 60)
print("IMAGE EMBEDDING INFERENCE")
print("=" * 60)

# Using convenience constructor for images
image_embed_transform = image_inference(
    model_uri="hf://openai/clip-vit-base-patch32",
    image_column="image_ref",
    output_column="image_embedding",
    batch_size=32,
    gpu_type=GPUType.NVIDIA_A10G,
)

print(f"\nImage embedding transform: {image_embed_transform.name}")
print(f"  Model: {image_embed_transform.model.uri}")
print(f"  Input schema: {image_embed_transform.model.input_schema}")

# Full InferenceTransform for more control
clip_transform = InferenceTransform(
    name="clip_embed_images",
    model=ModelSpec(
        uri="hf://openai/clip-vit-large-patch14",
        framework=ModelFramework.HUGGINGFACE,
        input_schema={"image_ref": "blob_ref<image/*>"},
        output_schema={"image_embedding": "float32[768]"},
        precision=ModelPrecision.FP16,
    ),
    accelerator=AcceleratorConfig.gpu(GPUType.NVIDIA_A100, memory_gb=40),
    batch_config=BatchConfig(
        batch_size=64,
        max_concurrent_batches=4,
        prefetch_batches=2,
    ),
    runtime=InferenceRuntime(
        packages=["transformers", "pillow", "torchvision"],
    ),
    input_mapping={"image_ref": "pixel_values"},
    output_mapping={"image_embeds": "image_embedding"},
    warm_up_batches=2,
    checkpoint_interval=50,
)

print(f"\nCLIP transform: {clip_transform.name}")
print(f"  Precision: {clip_transform.model.precision.value}")
print(f"  Warm-up batches: {clip_transform.warm_up_batches}")

# =============================================================================
# Classification Inference
# =============================================================================

print("\n" + "=" * 60)
print("CLASSIFICATION INFERENCE")
print("=" * 60)

sentiment_transform = classification_inference(
    model_uri="hf://distilbert-base-uncased-finetuned-sst-2-english",
    input_column="text",
    output_column="sentiment",
    probabilities_column="sentiment_score",
    batch_size=64,
)

print(f"\nSentiment transform: {sentiment_transform.name}")
print(f"  Output mapping: {sentiment_transform.output_mapping}")

# Create job
sentiment_job = fs.create_job(
    name="classify_sentiment",
    description="Classify text sentiment",
    sources=[
        FeatureGroupSource(
            feature_group="acme/mlplatform/embeddings/text-features",
            features=["doc_id", "text"],
        )
    ],
    transform=sentiment_transform,
    target=Target(
        feature_group="text-features",
        features={
            "sentiment": "sentiment",
            "sentiment_score": "sentiment_score",
        },
        write_mode="upsert",
        key_columns=["doc_id"],
    ),
    schedule=Schedule.hourly(minute=30),
    incremental=IncrementalConfig.incremental("updated_at"),
)

print(f"Created job: {sentiment_job.name}")

# =============================================================================
# LLM Inference
# =============================================================================

print("\n" + "=" * 60)
print("LLM INFERENCE")
print("=" * 60)

# LLM for caption generation
caption_transform = llm_inference(
    model_uri="hf://meta-llama/Llama-2-7b-hf",
    prompt_column="image_description",
    output_column="caption",
    max_tokens=128,
    batch_size=4,
    gpu_type=GPUType.NVIDIA_A100,
    gpu_count=1,
)

print(f"\nLLM transform: {caption_transform.name}")
print(f"  GPU count: {caption_transform.accelerator.count}")
print(f"  Batch size: {caption_transform.batch_config.batch_size}")

# Large model with multiple GPUs
large_llm_transform = InferenceTransform(
    name="llm_summarize",
    model=ModelSpec(
        uri="hf://meta-llama/Llama-2-70b-hf",
        framework=ModelFramework.HUGGINGFACE,
        precision=ModelPrecision.INT8,
        device_map="auto",
        quantization_config={
            "load_in_8bit": True,
            "llm_int8_threshold": 6.0,
        },
    ),
    accelerator=AcceleratorConfig.multi_gpu(
        gpu_type=GPUType.NVIDIA_A100_80GB,
        count=4,
        strategy="tensor_parallel",
    ),
    batch_config=BatchConfig(
        batch_size=2,
        timeout_seconds=600,
    ),
    runtime=InferenceRuntime(
        packages=["transformers", "accelerate", "bitsandbytes"],
        memory_limit_gb=320,
    ),
)

print(f"\nLarge LLM transform: {large_llm_transform.name}")
print(f"  Precision: {large_llm_transform.model.precision.value}")
print(f"  GPUs: {large_llm_transform.accelerator.count}x {large_llm_transform.accelerator.gpu_type.value}")
print(f"  Strategy: {large_llm_transform.accelerator.multi_gpu_strategy}")

# =============================================================================
# Custom Inference with Decorator
# =============================================================================

print("\n" + "=" * 60)
print("CUSTOM INFERENCE")
print("=" * 60)

@inference_transform(
    name="custom_scorer",
    model=ModelSpec(
        uri="s3://models/custom/scorer-v1.pt",
        framework=ModelFramework.PYTORCH,
    ),
    accelerator=AcceleratorConfig.gpu(GPUType.NVIDIA_T4),
    batch_size=128,
    description="Custom scoring model",
)
def custom_score(model, batch):
    """Score documents using custom model."""
    scores = model(batch["features"])
    return {"score": scores}

print(f"Custom transform: {custom_score.name}")
print(f"  Description: {custom_score.description}")

# =============================================================================
# Job with Chained Inference
# =============================================================================

print("\n" + "=" * 60)
print("CHAINED INFERENCE JOBS")
print("=" * 60)

# First job: Generate embeddings
embed_job = fs.create_job(
    name="step1_embed",
    sources=[FeatureGroupSource(feature_group="text-features", features=["doc_id", "text"])],
    transform=text_embed_transform,
    target=Target(feature_group="text-features", features={"text_embedding": "text_embedding"}),
    schedule=Schedule.manual(),
)

# Second job: Classify using embeddings
classify_job = fs.create_job(
    name="step2_classify",
    sources=[FeatureGroupSource(feature_group="text-features", features=["doc_id", "text_embedding"])],
    transform=InferenceTransform(
        name="classify_from_embeddings",
        model=ModelSpec(
            uri="s3://models/classifier/embedding-classifier.onnx",
            framework=ModelFramework.ONNX,
        ),
        accelerator=AcceleratorConfig.cpu(cores=4),  # ONNX on CPU is fast
        batch_config=BatchConfig(batch_size=256),
        input_mapping={"text_embedding": "input"},
        output_mapping={"logits": "category"},
    ),
    target=Target(feature_group="text-features", features={"category": "category"}),
    schedule=Schedule.manual(),
)

print(f"Created chained jobs:")
print(f"  1. {embed_job.name} (GPU)")
print(f"  2. {classify_job.name} (CPU)")

# =============================================================================
# Inference Result
# =============================================================================

print("\n" + "=" * 60)
print("INFERENCE RESULT")
print("=" * 60)

# Mock inference result
result = InferenceResult(
    job_id="job-123",
    run_id="run-456",
    status="success",
    total_samples=10000,
    successful_samples=9995,
    failed_samples=5,
    total_batches=157,
    duration_seconds=120.5,
    throughput=82.9,
    model_load_time=15.2,
    inference_time=105.3,
    gpu_utilization=0.85,
    memory_peak_gb=12.4,
)

print(f"\nInference result:")
print(f"  Status: {result.status}")
print(f"  Total samples: {result.total_samples:,}")
print(f"  Success rate: {result.success_rate:.2%}")
print(f"  Throughput: {result.throughput:.1f} samples/sec")
print(f"  Duration: {result.duration_seconds:.1f}s")
print(f"  GPU utilization: {result.gpu_utilization:.0%}")
print(f"  Peak memory: {result.memory_peak_gb:.1f} GB")

# =============================================================================
# Transform Helpers
# =============================================================================

print("\n" + "=" * 60)
print("TRANSFORM HELPERS")
print("=" * 60)

# Modify transform configuration
fast_transform = text_embed_transform.with_gpu(GPUType.NVIDIA_A100, count=2)
print(f"Fast transform GPU: {fast_transform.accelerator.gpu_type.value} x{fast_transform.accelerator.count}")

large_batch_transform = text_embed_transform.with_batch_size(512)
print(f"Large batch size: {large_batch_transform.batch_config.batch_size}")

# =============================================================================
# GPU Types Reference
# =============================================================================

print("\n" + "=" * 60)
print("GPU TYPES REFERENCE")
print("=" * 60)

gpu_specs = [
    (GPUType.NVIDIA_T4, "16GB", "Entry-level inference"),
    (GPUType.NVIDIA_A10G, "24GB", "Balanced cost/performance"),
    (GPUType.NVIDIA_A100, "40GB", "High-performance training/inference"),
    (GPUType.NVIDIA_A100_80GB, "80GB", "Large models"),
    (GPUType.NVIDIA_H100, "80GB", "Latest generation, highest performance"),
    (GPUType.NVIDIA_L4, "24GB", "Efficient inference"),
]

print("\nAvailable GPU types:")
for gpu, memory, use_case in gpu_specs:
    print(f"  {gpu.value}: {memory} - {use_case}")

# =============================================================================
# Framework Support
# =============================================================================

print("\n" + "=" * 60)
print("FRAMEWORK SUPPORT")
print("=" * 60)

print("\nSupported frameworks:")
for framework in ModelFramework:
    print(f"  - {framework.value}")

print("\nPrecision options:")
for precision in ModelPrecision:
    print(f"  - {precision.value}")

print("\n" + "=" * 60)
print("ALL BULK INFERENCE EXAMPLES COMPLETE!")
print("=" * 60)
