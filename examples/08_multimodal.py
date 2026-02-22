"""
Example 8: Multimodal Data and Blob References

Demonstrates working with multimodal data (images, audio, video) using
blob references. References allow transformations to work with metadata
without moving the actual data bytes.

Key concepts:
- BlobReference: Immutable reference with integrity metadata
- BlobRegistry: Centralized tracking and validation
- MultimodalSource: Scan storage for multimodal assets
- Integrity checks: Ensure references remain valid
"""

from raise_ import (
    FeatureStore,
    BlobRef,
    # Multimodal
    BlobReference,
    BlobRegistry,
    ContentType,
    IntegrityPolicy,
    MultimodalSource,
    create_reference,
    # Quality checks
    BlobIntegrityCheck,
)
from raise_.transforms import (
    CheckSeverity,
    MultimodalContext,
    HashAlgorithm,
    IntegrityMode,
    infer_content_type,
)

# =============================================================================
# Setup
# =============================================================================

fs = FeatureStore("acme/mlplatform/vision")

# Create a feature group for image features
image_features = fs.create_feature_group(
    "image-features",
    description="Features extracted from product images",
    if_exists="skip",
)

# Create features using the blob_ref type
image_features.create_features_from_schema({
    "product_id": "string",
    "image_ref": BlobRef(content_types=["image/png", "image/jpeg"]),
    "thumbnail_ref": BlobRef(content_types=["image/jpeg"]),
    "embedding": "float32[512]",
}, if_exists="skip")

print("Feature group created with blob reference columns")

# =============================================================================
# Creating Blob References
# =============================================================================

print("\n" + "=" * 60)
print("BLOB REFERENCES")
print("=" * 60)

# Create a blob registry for tracking references
registry = BlobRegistry(
    name="product-images",
    policy=IntegrityPolicy.on_write(),
)

# Register a blob reference
image_ref = registry.register(
    uri="s3://products/images/product-001.png",
    content_type=ContentType.IMAGE_PNG,
    checksum="abc123def456789...",
    hash_algorithm=HashAlgorithm.SHA256,
    size_bytes=1024000,
    etag="abc123",
    metadata={
        "width": 1920,
        "height": 1080,
        "format": "PNG",
        "color_space": "sRGB",
    },
)

print(f"\nRegistered blob reference:")
print(f"  URI: {image_ref.uri}")
print(f"  Content type: {image_ref.content_type.value}")
print(f"  Size: {image_ref.size_bytes:,} bytes")
print(f"  Checksum: {image_ref.checksum[:16]}...")
print(f"  Scheme: {image_ref.scheme}")
print(f"  Bucket: {image_ref.bucket}")
print(f"  Key: {image_ref.key}")
print(f"  Is image: {image_ref.is_image}")

# Use convenience function
thumbnail_ref = create_reference(
    uri="s3://products/thumbnails/product-001-thumb.jpg",
    content_type="image/jpeg",
    checksum="def789abc123456...",
    size_bytes=50000,
    metadata={"width": 200, "height": 200},
)

print(f"\nCreated thumbnail reference:")
print(f"  URI: {thumbnail_ref.uri}")
print(f"  Content type: {thumbnail_ref.content_type}")

# Infer content type from URI
inferred_type = infer_content_type("s3://data/audio/sample.wav")
print(f"\nInferred content type for .wav: {inferred_type.value}")

# =============================================================================
# BlobRef Feature Type
# =============================================================================

print("\n" + "=" * 60)
print("BLOB REFERENCE FEATURE TYPE")
print("=" * 60)

# Create typed blob reference columns
from raise_.models.types import BlobRef, parse_dtype

# Parse from string
blob_type = parse_dtype("blob_ref")
print(f"Parsed 'blob_ref': {blob_type}")

# Parse with content type constraints
image_type = parse_dtype("blob_ref<image/png|image/jpeg>")
print(f"Parsed 'blob_ref<image/png|image/jpeg>': {image_type}")
print(f"  Content types: {image_type.content_types}")
print(f"  Accepts PNG: {image_type.accepts('image/png')}")
print(f"  Accepts MP4: {image_type.accepts('video/mp4')}")

# Create directly
audio_type = BlobRef(
    content_types=["audio/wav", "audio/mp3", "audio/flac"],
    registry="audio-store",
    validate_on_write=True,
)
print(f"\nAudio blob type: {audio_type}")
print(f"  Registry: {audio_type.registry}")

# =============================================================================
# Multimodal Source
# =============================================================================

print("\n" + "=" * 60)
print("MULTIMODAL SOURCE")
print("=" * 60)

# Create a source for scanning multimodal data
image_source = MultimodalSource(
    name="product_images",
    uri_prefix="s3://products/images/",
    content_types=[ContentType.IMAGE_JPEG, ContentType.IMAGE_PNG],
    registry=registry,
    compute_checksums=True,
    hash_algorithm=HashAlgorithm.SHA256,
    recursive=True,
)

print(f"Created multimodal source:")
print(f"  Name: {image_source.name}")
print(f"  URI prefix: {image_source.uri_prefix}")
print(f"  Content types: {[ct.value for ct in image_source.content_types]}")

# Scan for references
refs = image_source.scan()
print(f"\nScanned {len(refs)} blob references:")
for ref in refs:
    print(f"  - {ref.uri} ({ref.content_type.value if hasattr(ref.content_type, 'value') else ref.content_type})")

# =============================================================================
# Integrity Validation
# =============================================================================

print("\n" + "=" * 60)
print("INTEGRITY VALIDATION")
print("=" * 60)

# Configure integrity policy
strict_policy = IntegrityPolicy.strict()
print(f"Strict policy: mode={strict_policy.mode.value}")

lazy_policy = IntegrityPolicy.lazy()
print(f"Lazy policy: mode={lazy_policy.mode.value}, fail_on_missing={lazy_policy.fail_on_missing}")

on_write_policy = IntegrityPolicy(
    mode=IntegrityMode.ON_WRITE,
    fail_on_missing=True,
    fail_on_mismatch=True,
    cache_validation_seconds=3600,
)
print(f"On-write policy: cache={on_write_policy.cache_validation_seconds}s")

# Validate a reference
result = registry.validate(image_ref)
print(f"\nValidation result for {image_ref.uri}:")
print(f"  Status: {result.status.value}")
print(f"  Valid: {result.valid}")
print(f"  Message: {result.message}")
print(f"  Checksum matches: {result.checksum_matches}")
print(f"  Size matches: {result.size_matches}")

# Batch validation
batch_results = registry.validate_batch(refs)
print(f"\nBatch validation: {len(batch_results)} references checked")

# =============================================================================
# Blob Integrity Check
# =============================================================================

print("\n" + "=" * 60)
print("BLOB INTEGRITY CHECK")
print("=" * 60)

# Create integrity check for quality validation
integrity_check = BlobIntegrityCheck(
    name="check_image_integrity",
    description="Verify all image references are valid",
    column="image_ref",
    verify_checksum=True,
    verify_existence=True,
    max_missing_rate=0.01,  # Allow 1% missing
    max_invalid_rate=0.0,   # No invalid checksums allowed
    sample_rate=1.0,        # Check all references
    severity=CheckSeverity.ERROR,
)

# Run the check
check_result = integrity_check.check(None)  # Mock data
print(f"\nIntegrity check result:")
print(f"  Check: {check_result.check_name}")
print(f"  Result: {check_result.result.value}")
print(f"  Message: {check_result.message}")
print(f"  Details: {check_result.details}")

# =============================================================================
# Reference Metadata
# =============================================================================

print("\n" + "=" * 60)
print("REFERENCE METADATA")
print("=" * 60)

# Add metadata to a reference (creates new immutable reference)
enriched_ref = image_ref.with_metadata(
    ml_processed=True,
    model_version="resnet50-v2",
    processing_timestamp="2024-02-21T10:00:00Z",
)

print(f"Original metadata: {image_ref.metadata}")
print(f"Enriched metadata: {enriched_ref.metadata}")

# Serialize/deserialize
ref_dict = image_ref.to_dict()
print(f"\nSerialized reference keys: {list(ref_dict.keys())}")

restored_ref = BlobReference.from_dict(ref_dict)
print(f"Restored URI: {restored_ref.uri}")
print(f"Restored checksum matches: {restored_ref.checksum == image_ref.checksum}")

# =============================================================================
# Multimodal Context for Transforms
# =============================================================================

print("\n" + "=" * 60)
print("MULTIMODAL TRANSFORM CONTEXT")
print("=" * 60)

# Create context for use in transforms
ctx = MultimodalContext(registry=registry)

# Batch validate
validation_results = ctx.batch_validate(refs)
print(f"Validated {len(validation_results)} references")

# Example: Create derived blob (e.g., thumbnail from image)
def create_thumbnail(data: bytes) -> bytes:
    """Mock thumbnail creation."""
    return b"thumbnail_" + data[:100]

# In a real transform, you would use this pattern:
# derived_ref = ctx.create_derived(
#     source_ref=image_ref,
#     new_uri="s3://products/thumbnails/product-001-auto.jpg",
#     content_type=ContentType.IMAGE_JPEG,
#     processor=create_thumbnail,
# )

print("Multimodal context ready for transforms")

# =============================================================================
# Writing Features with Blob References
# =============================================================================

print("\n" + "=" * 60)
print("WRITING FEATURES WITH BLOB REFERENCES")
print("=" * 60)

# Prepare feature data with blob references
feature_data = {
    "product_id": "PROD-001",
    "image_ref": image_ref.to_dict(),  # Serialize for storage
    "thumbnail_ref": thumbnail_ref.to_dict(),
    "embedding": [0.1] * 512,  # Mock embedding
}

print(f"Feature data prepared:")
print(f"  Product ID: {feature_data['product_id']}")
print(f"  Image ref URI: {feature_data['image_ref']['uri']}")
print(f"  Thumbnail ref URI: {feature_data['thumbnail_ref']['uri']}")
print(f"  Embedding dims: {len(feature_data['embedding'])}")

# =============================================================================
# Registry Operations
# =============================================================================

print("\n" + "=" * 60)
print("REGISTRY OPERATIONS")
print("=" * 60)

# List references with filtering
all_refs = registry.list_references()
print(f"All registered references: {len(all_refs)}")

image_refs = registry.list_references(content_type=ContentType.IMAGE_PNG)
print(f"PNG references: {len(image_refs)}")

prefixed_refs = registry.list_references(prefix="s3://products/images/")
print(f"References with prefix 's3://products/images/': {len(prefixed_refs)}")

# Find by URI
found_ref = registry.get_by_uri("s3://products/images/product-001.png")
print(f"\nFound by URI: {found_ref is not None}")

# Find orphaned references (blobs that no longer exist)
orphans = registry.find_orphans()
print(f"Orphaned references: {len(orphans)}")

# =============================================================================
# Content Type Utilities
# =============================================================================

print("\n" + "=" * 60)
print("CONTENT TYPES")
print("=" * 60)

# Available content types
print("Image types:")
for ct in [ContentType.IMAGE_PNG, ContentType.IMAGE_JPEG, ContentType.IMAGE_WEBP]:
    print(f"  - {ct.value}")

print("\nAudio types:")
for ct in [ContentType.AUDIO_WAV, ContentType.AUDIO_MP3, ContentType.AUDIO_FLAC]:
    print(f"  - {ct.value}")

print("\nVideo types:")
for ct in [ContentType.VIDEO_MP4, ContentType.VIDEO_WEBM, ContentType.VIDEO_AVI]:
    print(f"  - {ct.value}")

print("\nML types:")
for ct in [ContentType.ARRAY_NPY, ContentType.TENSOR_PT, ContentType.TENSOR_SAFETENSORS]:
    print(f"  - {ct.value}")

# =============================================================================
# Example: Image Processing Pipeline
# =============================================================================

print("\n" + "=" * 60)
print("EXAMPLE: IMAGE PROCESSING PIPELINE")
print("=" * 60)

# This demonstrates how a typical multimodal pipeline would work:
# 1. Scan for images
# 2. Create references (without loading data)
# 3. Run transforms that work with references
# 4. Store enriched features

pipeline_steps = [
    "1. Scan object storage for new images",
    "2. Register blob references with checksums",
    "3. Validate references exist and are accessible",
    "4. Pass references to embedding model (model loads data)",
    "5. Store embeddings + references in feature store",
    "6. Run integrity checks periodically",
]

print("Pipeline steps (data stays in place):")
for step in pipeline_steps:
    print(f"  {step}")

print("\nKey benefit: Only references move through the pipeline,")
print("not the actual image bytes (which may be gigabytes).")

# =============================================================================
# Hash Algorithms
# =============================================================================

print("\n" + "=" * 60)
print("HASH ALGORITHMS")
print("=" * 60)

# Compute checksums
sample_data = b"sample image data for checksum demo"

for algo in [HashAlgorithm.SHA256, HashAlgorithm.SHA512, HashAlgorithm.MD5]:
    checksum = registry.compute_checksum(sample_data, algo)
    print(f"{algo.value}: {checksum[:32]}...")

print("\n" + "=" * 60)
print("ALL MULTIMODAL EXAMPLES COMPLETE!")
print("=" * 60)
