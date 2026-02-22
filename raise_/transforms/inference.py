"""
Bulk Inference Transformations for Raise Feature Store

Supports batch inference jobs that run AI models on feature data.
These jobs often require special accelerators (GPUs, TPUs) and have
different resource requirements than traditional ETL transforms.

Key concepts:
- InferenceTransform: Transform that runs model inference
- ModelSpec: Model location, framework, and version
- AcceleratorConfig: GPU/TPU requirements
- BatchConfig: Batching and parallelism settings
- InferenceRuntime: Execution environment configuration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable
import uuid


# =============================================================================
# Enums
# =============================================================================

class ModelFramework(Enum):
    """Supported ML frameworks."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    TRITON = "triton"
    HUGGINGFACE = "huggingface"
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    JAX = "jax"
    CUSTOM = "custom"


class AcceleratorType(Enum):
    """Accelerator hardware types."""
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    NEURON = "neuron"      # AWS Inferentia
    HABANA = "habana"      # Intel Habana Gaudi


class GPUType(Enum):
    """Specific GPU types for resource allocation."""
    NVIDIA_T4 = "nvidia-t4"
    NVIDIA_V100 = "nvidia-v100"
    NVIDIA_A10G = "nvidia-a10g"
    NVIDIA_A100 = "nvidia-a100"
    NVIDIA_A100_80GB = "nvidia-a100-80gb"
    NVIDIA_H100 = "nvidia-h100"
    NVIDIA_L4 = "nvidia-l4"
    ANY = "any"


class TPUType(Enum):
    """TPU types for resource allocation."""
    TPU_V2 = "tpu-v2"
    TPU_V3 = "tpu-v3"
    TPU_V4 = "tpu-v4"
    TPU_V5E = "tpu-v5e"


class InferenceMode(Enum):
    """Inference execution mode."""
    BATCH = "batch"              # Process all data in batches
    STREAMING = "streaming"      # Process as stream (future)
    REALTIME = "realtime"        # Low-latency inference (future)


class ModelPrecision(Enum):
    """Model precision for inference."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


# =============================================================================
# Model Specification
# =============================================================================

@dataclass
class ModelSpec:
    """
    Specification for a model to use in inference.

    Attributes:
        uri: Location of model artifacts (s3://, gs://, hf://, mlflow://, etc.)
        framework: ML framework (pytorch, tensorflow, onnx, etc.)
        version: Model version identifier
        name: Human-readable model name
        entry_point: Entry point function/class for custom models
        input_schema: Expected input schema (feature names -> types)
        output_schema: Output schema (output names -> types)
        precision: Model precision (fp32, fp16, int8, etc.)
        compile_options: Framework-specific compilation options
        environment: Environment variables for model loading

    Example:
        >>> model = ModelSpec(
        ...     uri="s3://models/embeddings/clip-vit-base",
        ...     framework=ModelFramework.PYTORCH,
        ...     version="v2.1",
        ...     input_schema={"image_ref": "blob_ref"},
        ...     output_schema={"embedding": "float32[512]"},
        ... )
    """
    uri: str
    framework: ModelFramework = ModelFramework.PYTORCH
    version: str = "latest"
    name: str | None = None
    entry_point: str | None = None
    input_schema: dict[str, str] = field(default_factory=dict)
    output_schema: dict[str, str] = field(default_factory=dict)
    precision: ModelPrecision = ModelPrecision.FP32
    compile_options: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)

    # Model loading options
    trust_remote_code: bool = False
    device_map: str | dict | None = None
    torch_dtype: str | None = None
    quantization_config: dict[str, Any] | None = None

    def __post_init__(self):
        if isinstance(self.framework, str):
            self.framework = ModelFramework(self.framework)
        if isinstance(self.precision, str):
            self.precision = ModelPrecision(self.precision)

    @property
    def scheme(self) -> str:
        """Extract URI scheme (s3, gs, hf, mlflow, etc.)."""
        if "://" in self.uri:
            return self.uri.split("://")[0]
        return "file"

    @property
    def is_huggingface(self) -> bool:
        """Check if model is from HuggingFace Hub."""
        return self.scheme == "hf" or self.framework == ModelFramework.HUGGINGFACE

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "framework": self.framework.value,
            "version": self.version,
            "name": self.name,
            "entry_point": self.entry_point,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "precision": self.precision.value,
            "compile_options": self.compile_options,
            "environment": self.environment,
        }

    @classmethod
    def from_huggingface(
        cls,
        model_id: str,
        revision: str = "main",
        task: str | None = None,
        **kwargs,
    ) -> "ModelSpec":
        """Create ModelSpec from HuggingFace Hub model."""
        return cls(
            uri=f"hf://{model_id}",
            framework=ModelFramework.HUGGINGFACE,
            version=revision,
            name=model_id.split("/")[-1],
            compile_options={"task": task} if task else {},
            **kwargs,
        )

    @classmethod
    def from_mlflow(
        cls,
        model_uri: str,
        version: str = "latest",
        **kwargs,
    ) -> "ModelSpec":
        """Create ModelSpec from MLflow model registry."""
        return cls(
            uri=model_uri if model_uri.startswith("mlflow://") else f"mlflow://{model_uri}",
            version=version,
            **kwargs,
        )


# =============================================================================
# Accelerator Configuration
# =============================================================================

@dataclass
class AcceleratorConfig:
    """
    Configuration for hardware accelerators.

    Attributes:
        accelerator_type: Type of accelerator (gpu, tpu, cpu)
        gpu_type: Specific GPU model (for GPU accelerator)
        tpu_type: Specific TPU version (for TPU accelerator)
        count: Number of accelerators
        memory_gb: Required memory per accelerator (for scheduling)
        multi_gpu_strategy: Strategy for multi-GPU inference

    Example:
        >>> # Single A100 GPU
        >>> config = AcceleratorConfig(
        ...     accelerator_type=AcceleratorType.GPU,
        ...     gpu_type=GPUType.NVIDIA_A100,
        ...     count=1,
        ... )

        >>> # Multi-GPU setup
        >>> config = AcceleratorConfig.multi_gpu(
        ...     gpu_type=GPUType.NVIDIA_A100,
        ...     count=4,
        ...     strategy="data_parallel",
        ... )
    """
    accelerator_type: AcceleratorType = AcceleratorType.GPU
    gpu_type: GPUType | None = None
    tpu_type: TPUType | None = None
    count: int = 1
    memory_gb: int | None = None
    multi_gpu_strategy: str | None = None  # data_parallel, tensor_parallel, pipeline_parallel

    def __post_init__(self):
        if isinstance(self.accelerator_type, str):
            self.accelerator_type = AcceleratorType(self.accelerator_type)
        if isinstance(self.gpu_type, str):
            self.gpu_type = GPUType(self.gpu_type)
        if isinstance(self.tpu_type, str):
            self.tpu_type = TPUType(self.tpu_type)

    @classmethod
    def cpu(cls, cores: int = 4) -> "AcceleratorConfig":
        """CPU-only configuration."""
        return cls(accelerator_type=AcceleratorType.CPU, count=cores)

    @classmethod
    def gpu(
        cls,
        gpu_type: GPUType = GPUType.ANY,
        count: int = 1,
        memory_gb: int | None = None,
    ) -> "AcceleratorConfig":
        """Single or multi-GPU configuration."""
        return cls(
            accelerator_type=AcceleratorType.GPU,
            gpu_type=gpu_type,
            count=count,
            memory_gb=memory_gb,
        )

    @classmethod
    def multi_gpu(
        cls,
        gpu_type: GPUType,
        count: int,
        strategy: str = "data_parallel",
    ) -> "AcceleratorConfig":
        """Multi-GPU configuration with parallelism strategy."""
        return cls(
            accelerator_type=AcceleratorType.GPU,
            gpu_type=gpu_type,
            count=count,
            multi_gpu_strategy=strategy,
        )

    @classmethod
    def tpu(cls, tpu_type: TPUType, count: int = 1) -> "AcceleratorConfig":
        """TPU configuration."""
        return cls(
            accelerator_type=AcceleratorType.TPU,
            tpu_type=tpu_type,
            count=count,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "accelerator_type": self.accelerator_type.value,
            "gpu_type": self.gpu_type.value if self.gpu_type else None,
            "tpu_type": self.tpu_type.value if self.tpu_type else None,
            "count": self.count,
            "memory_gb": self.memory_gb,
            "multi_gpu_strategy": self.multi_gpu_strategy,
        }


# =============================================================================
# Batch Configuration
# =============================================================================

@dataclass
class BatchConfig:
    """
    Configuration for batch inference.

    Attributes:
        batch_size: Number of samples per batch
        max_concurrent_batches: Maximum concurrent batches in flight
        prefetch_batches: Number of batches to prefetch
        timeout_seconds: Timeout per batch
        retry_failed_batches: Whether to retry failed batches
        max_retries: Maximum retries for failed batches

    Example:
        >>> config = BatchConfig(
        ...     batch_size=32,
        ...     max_concurrent_batches=4,
        ...     prefetch_batches=2,
        ... )
    """
    batch_size: int = 32
    max_concurrent_batches: int = 1
    prefetch_batches: int = 1
    timeout_seconds: int = 300
    retry_failed_batches: bool = True
    max_retries: int = 3

    # Dynamic batching options
    dynamic_batching: bool = False
    min_batch_size: int = 1
    max_batch_size: int = 64
    batch_timeout_ms: int = 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "max_concurrent_batches": self.max_concurrent_batches,
            "prefetch_batches": self.prefetch_batches,
            "timeout_seconds": self.timeout_seconds,
            "retry_failed_batches": self.retry_failed_batches,
            "max_retries": self.max_retries,
            "dynamic_batching": self.dynamic_batching,
        }


# =============================================================================
# Inference Runtime
# =============================================================================

@dataclass
class InferenceRuntime:
    """
    Runtime environment configuration for inference.

    Attributes:
        image: Container image for inference
        python_version: Python version
        packages: Additional Python packages to install
        system_packages: System packages to install
        env_vars: Environment variables
        working_dir: Working directory
        startup_script: Script to run before inference

    Example:
        >>> runtime = InferenceRuntime(
        ...     image="pytorch/pytorch:2.0-cuda11.8-cudnn8-runtime",
        ...     packages=["transformers", "accelerate"],
        ... )
    """
    image: str | None = None
    python_version: str = "3.10"
    packages: list[str] = field(default_factory=list)
    system_packages: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    working_dir: str | None = None
    startup_script: str | None = None

    # Resource limits
    memory_limit_gb: int | None = None
    cpu_limit: float | None = None
    shm_size_gb: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "python_version": self.python_version,
            "packages": self.packages,
            "system_packages": self.system_packages,
            "env_vars": self.env_vars,
            "memory_limit_gb": self.memory_limit_gb,
            "cpu_limit": self.cpu_limit,
            "shm_size_gb": self.shm_size_gb,
        }


# =============================================================================
# Inference Transform
# =============================================================================

@dataclass
class InferenceTransform:
    """
    Transform that runs model inference on input data.

    This is a specialized transform for bulk inference jobs that:
    - Load and run ML models on batches of data
    - Support GPU/TPU acceleration
    - Handle model loading, batching, and output collection
    - Integrate with model registries (MLflow, HuggingFace, etc.)

    Attributes:
        name: Transform name
        model: Model specification
        accelerator: Accelerator configuration
        batch_config: Batch processing configuration
        runtime: Runtime environment configuration
        input_mapping: Map source columns to model inputs
        output_mapping: Map model outputs to target columns
        preprocessing: Optional preprocessing function
        postprocessing: Optional postprocessing function
        mode: Inference mode (batch, streaming, realtime)

    Example:
        >>> transform = InferenceTransform(
        ...     name="embed_images",
        ...     model=ModelSpec(
        ...         uri="hf://openai/clip-vit-base-patch32",
        ...         framework=ModelFramework.HUGGINGFACE,
        ...     ),
        ...     accelerator=AcceleratorConfig.gpu(GPUType.NVIDIA_A100),
        ...     batch_config=BatchConfig(batch_size=64),
        ...     input_mapping={"image_ref": "pixel_values"},
        ...     output_mapping={"image_features": "embedding"},
        ... )
    """
    name: str
    model: ModelSpec
    accelerator: AcceleratorConfig = field(default_factory=AcceleratorConfig)
    batch_config: BatchConfig = field(default_factory=BatchConfig)
    runtime: InferenceRuntime = field(default_factory=InferenceRuntime)
    input_mapping: dict[str, str] = field(default_factory=dict)
    output_mapping: dict[str, str] = field(default_factory=dict)
    preprocessing: Callable | str | None = None
    postprocessing: Callable | str | None = None
    mode: InferenceMode = InferenceMode.BATCH
    description: str | None = None

    # Execution options
    warm_up_batches: int = 1
    checkpoint_interval: int = 100  # Save progress every N batches
    fail_on_error: bool = False  # Continue on individual sample errors

    # Internal
    _id: str = field(default_factory=lambda: str(uuid.uuid4()))
    _transform_type: str = field(default="inference", init=False)

    def __post_init__(self):
        if isinstance(self.mode, str):
            self.mode = InferenceMode(self.mode)

    @property
    def transform_type(self) -> str:
        return self._transform_type

    @property
    def requires_gpu(self) -> bool:
        return self.accelerator.accelerator_type == AcceleratorType.GPU

    @property
    def requires_tpu(self) -> bool:
        return self.accelerator.accelerator_type == AcceleratorType.TPU

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self._id,
            "name": self.name,
            "transform_type": self._transform_type,
            "model": self.model.to_dict(),
            "accelerator": self.accelerator.to_dict(),
            "batch_config": self.batch_config.to_dict(),
            "runtime": self.runtime.to_dict(),
            "input_mapping": self.input_mapping,
            "output_mapping": self.output_mapping,
            "mode": self.mode.value,
            "description": self.description,
            "warm_up_batches": self.warm_up_batches,
            "checkpoint_interval": self.checkpoint_interval,
            "fail_on_error": self.fail_on_error,
        }

    def with_gpu(self, gpu_type: GPUType, count: int = 1) -> "InferenceTransform":
        """Return a copy with GPU configuration."""
        return InferenceTransform(
            name=self.name,
            model=self.model,
            accelerator=AcceleratorConfig.gpu(gpu_type, count),
            batch_config=self.batch_config,
            runtime=self.runtime,
            input_mapping=self.input_mapping,
            output_mapping=self.output_mapping,
            preprocessing=self.preprocessing,
            postprocessing=self.postprocessing,
            mode=self.mode,
            description=self.description,
        )

    def with_batch_size(self, batch_size: int) -> "InferenceTransform":
        """Return a copy with different batch size."""
        new_batch_config = BatchConfig(
            batch_size=batch_size,
            max_concurrent_batches=self.batch_config.max_concurrent_batches,
            prefetch_batches=self.batch_config.prefetch_batches,
        )
        return InferenceTransform(
            name=self.name,
            model=self.model,
            accelerator=self.accelerator,
            batch_config=new_batch_config,
            runtime=self.runtime,
            input_mapping=self.input_mapping,
            output_mapping=self.output_mapping,
            preprocessing=self.preprocessing,
            postprocessing=self.postprocessing,
            mode=self.mode,
            description=self.description,
        )


# =============================================================================
# Inference Job Result
# =============================================================================

@dataclass
class InferenceResult:
    """
    Result of an inference job run.

    Attributes:
        job_id: Job identifier
        run_id: Run identifier
        status: Run status
        total_samples: Total samples processed
        successful_samples: Successfully processed samples
        failed_samples: Failed samples
        total_batches: Total batches
        duration_seconds: Total execution time
        throughput: Samples per second
        model_load_time: Time to load model
        inference_time: Time spent on inference
        gpu_utilization: Average GPU utilization
        memory_peak_gb: Peak memory usage
    """
    job_id: str
    run_id: str
    status: str
    total_samples: int = 0
    successful_samples: int = 0
    failed_samples: int = 0
    total_batches: int = 0
    duration_seconds: float = 0.0
    throughput: float = 0.0
    model_load_time: float = 0.0
    inference_time: float = 0.0
    gpu_utilization: float | None = None
    memory_peak_gb: float | None = None
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.successful_samples / self.total_samples

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "status": self.status,
            "total_samples": self.total_samples,
            "successful_samples": self.successful_samples,
            "failed_samples": self.failed_samples,
            "total_batches": self.total_batches,
            "duration_seconds": self.duration_seconds,
            "throughput": self.throughput,
            "model_load_time": self.model_load_time,
            "inference_time": self.inference_time,
            "gpu_utilization": self.gpu_utilization,
            "memory_peak_gb": self.memory_peak_gb,
            "success_rate": self.success_rate,
        }


# =============================================================================
# Decorator for Custom Inference
# =============================================================================

def inference_transform(
    func: Callable | None = None,
    *,
    name: str | None = None,
    model: ModelSpec | None = None,
    accelerator: AcceleratorConfig | None = None,
    batch_size: int = 32,
    description: str | None = None,
):
    """
    Decorator to create an inference transform from a function.

    The function should accept (model, batch) and return predictions.

    Example:
        >>> @inference_transform(
        ...     name="custom_embedding",
        ...     model=ModelSpec(uri="s3://models/custom"),
        ...     accelerator=AcceleratorConfig.gpu(GPUType.NVIDIA_T4),
        ...     batch_size=64,
        ... )
        ... def embed_texts(model, batch):
        ...     embeddings = model.encode(batch["text"])
        ...     return {"embedding": embeddings}
    """
    def decorator(fn: Callable) -> InferenceTransform:
        transform = InferenceTransform(
            name=name or fn.__name__,
            model=model or ModelSpec(uri="custom://", framework=ModelFramework.CUSTOM),
            accelerator=accelerator or AcceleratorConfig(),
            batch_config=BatchConfig(batch_size=batch_size),
            description=description or fn.__doc__,
            preprocessing=fn,
        )
        return transform

    if func is not None:
        return decorator(func)
    return decorator


# =============================================================================
# Convenience Constructors
# =============================================================================

def embedding_inference(
    model_uri: str,
    input_column: str,
    output_column: str = "embedding",
    batch_size: int = 64,
    gpu_type: GPUType = GPUType.NVIDIA_T4,
    **kwargs,
) -> InferenceTransform:
    """
    Create an inference transform for embedding generation.

    Example:
        >>> transform = embedding_inference(
        ...     model_uri="hf://sentence-transformers/all-MiniLM-L6-v2",
        ...     input_column="text",
        ...     output_column="text_embedding",
        ... )
    """
    return InferenceTransform(
        name=f"embed_{input_column}",
        model=ModelSpec(
            uri=model_uri,
            framework=ModelFramework.HUGGINGFACE,
            output_schema={output_column: "float32[*]"},
        ),
        accelerator=AcceleratorConfig.gpu(gpu_type),
        batch_config=BatchConfig(batch_size=batch_size),
        input_mapping={input_column: "input"},
        output_mapping={"embeddings": output_column},
        **kwargs,
    )


def classification_inference(
    model_uri: str,
    input_column: str,
    output_column: str = "prediction",
    probabilities_column: str | None = "probabilities",
    batch_size: int = 64,
    gpu_type: GPUType = GPUType.NVIDIA_T4,
    **kwargs,
) -> InferenceTransform:
    """
    Create an inference transform for classification.

    Example:
        >>> transform = classification_inference(
        ...     model_uri="hf://distilbert-base-uncased-finetuned-sst-2-english",
        ...     input_column="text",
        ...     output_column="sentiment",
        ... )
    """
    output_mapping = {"label": output_column}
    if probabilities_column:
        output_mapping["scores"] = probabilities_column

    return InferenceTransform(
        name=f"classify_{input_column}",
        model=ModelSpec(
            uri=model_uri,
            framework=ModelFramework.HUGGINGFACE,
        ),
        accelerator=AcceleratorConfig.gpu(gpu_type),
        batch_config=BatchConfig(batch_size=batch_size),
        input_mapping={input_column: "input"},
        output_mapping=output_mapping,
        **kwargs,
    )


def image_inference(
    model_uri: str,
    image_column: str,
    output_column: str = "embedding",
    batch_size: int = 32,
    gpu_type: GPUType = GPUType.NVIDIA_A10G,
    **kwargs,
) -> InferenceTransform:
    """
    Create an inference transform for image processing.

    Example:
        >>> transform = image_inference(
        ...     model_uri="hf://openai/clip-vit-base-patch32",
        ...     image_column="image_ref",
        ...     output_column="image_embedding",
        ... )
    """
    return InferenceTransform(
        name=f"process_{image_column}",
        model=ModelSpec(
            uri=model_uri,
            framework=ModelFramework.HUGGINGFACE,
            input_schema={image_column: "blob_ref<image/*>"},
            output_schema={output_column: "float32[*]"},
        ),
        accelerator=AcceleratorConfig.gpu(gpu_type),
        batch_config=BatchConfig(batch_size=batch_size),
        input_mapping={image_column: "pixel_values"},
        output_mapping={"image_embeds": output_column},
        **kwargs,
    )


def llm_inference(
    model_uri: str,
    prompt_column: str,
    output_column: str = "response",
    max_tokens: int = 256,
    batch_size: int = 8,
    gpu_type: GPUType = GPUType.NVIDIA_A100,
    gpu_count: int = 1,
    **kwargs,
) -> InferenceTransform:
    """
    Create an inference transform for LLM text generation.

    Example:
        >>> transform = llm_inference(
        ...     model_uri="hf://meta-llama/Llama-2-7b-hf",
        ...     prompt_column="prompt",
        ...     output_column="completion",
        ...     max_tokens=512,
        ... )
    """
    return InferenceTransform(
        name=f"llm_{prompt_column}",
        model=ModelSpec(
            uri=model_uri,
            framework=ModelFramework.HUGGINGFACE,
            compile_options={"max_new_tokens": max_tokens},
        ),
        accelerator=AcceleratorConfig.gpu(gpu_type, count=gpu_count),
        batch_config=BatchConfig(batch_size=batch_size),
        input_mapping={prompt_column: "input_ids"},
        output_mapping={"generated_text": output_column},
        **kwargs,
    )
