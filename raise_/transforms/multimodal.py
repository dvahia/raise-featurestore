"""
Multimodal Data Support for Raise Feature Store

Provides blob references for multimodal data (images, audio, video, etc.)
that allow transformations to work with references rather than moving
the actual data bytes.

Key concepts:
- BlobReference: Immutable reference to a multimodal blob with integrity metadata
- BlobRegistry: Centralized registry for tracking and validating blob references
- MultimodalSource: Source connector for multimodal datasets
- IntegrityPolicy: Controls when and how referential integrity is enforced
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Union
import hashlib
import uuid


# =============================================================================
# Enums
# =============================================================================

class ContentType(Enum):
    """Supported multimodal content types."""
    # Images
    IMAGE_PNG = "image/png"
    IMAGE_JPEG = "image/jpeg"
    IMAGE_WEBP = "image/webp"
    IMAGE_TIFF = "image/tiff"
    IMAGE_BMP = "image/bmp"
    IMAGE_GIF = "image/gif"

    # Audio
    AUDIO_WAV = "audio/wav"
    AUDIO_MP3 = "audio/mpeg"
    AUDIO_FLAC = "audio/flac"
    AUDIO_OGG = "audio/ogg"
    AUDIO_AAC = "audio/aac"

    # Video
    VIDEO_MP4 = "video/mp4"
    VIDEO_WEBM = "video/webm"
    VIDEO_AVI = "video/avi"
    VIDEO_MOV = "video/quicktime"

    # Documents
    DOCUMENT_PDF = "application/pdf"

    # 3D/Point Clouds
    MODEL_PLY = "model/ply"
    MODEL_OBJ = "model/obj"
    MODEL_GLTF = "model/gltf+json"

    # Scientific
    ARRAY_NPY = "application/x-numpy"
    ARRAY_NPZ = "application/x-numpy-compressed"
    TENSOR_PT = "application/x-pytorch"
    TENSOR_SAFETENSORS = "application/x-safetensors"

    # Generic binary
    BINARY = "application/octet-stream"


class HashAlgorithm(Enum):
    """Supported hash algorithms for integrity verification."""
    SHA256 = "sha256"
    SHA512 = "sha512"
    MD5 = "md5"  # Not recommended for security, but fast for integrity
    BLAKE3 = "blake3"
    XXH3 = "xxh3"  # Very fast, good for large files


class IntegrityMode(Enum):
    """When to enforce referential integrity."""
    STRICT = "strict"      # Validate on every access
    ON_WRITE = "on_write"  # Validate only when creating/updating references
    ON_READ = "on_read"    # Validate when reading/resolving references
    LAZY = "lazy"          # Validate only on explicit check
    PERIODIC = "periodic"  # Background validation jobs


class BlobStatus(Enum):
    """Status of a blob reference."""
    VALID = "valid"              # Blob exists and integrity verified
    PENDING = "pending"          # Blob being uploaded/created
    INVALID = "invalid"          # Integrity check failed
    MISSING = "missing"          # Blob not found at URI
    EXPIRED = "expired"          # Blob TTL exceeded
    UNKNOWN = "unknown"          # Not yet validated


# =============================================================================
# Blob Reference
# =============================================================================

@dataclass(frozen=True)
class BlobReference:
    """
    Immutable reference to a multimodal blob.

    References contain all metadata needed to locate and verify blob integrity
    without loading the actual data. References are hashable and can be stored
    in feature columns.

    Attributes:
        uri: Location of the blob (s3://, gs://, az://, file://, etc.)
        content_type: MIME type of the blob content
        checksum: Hash of the blob content for integrity verification
        hash_algorithm: Algorithm used to compute the checksum
        size_bytes: Size of the blob in bytes
        etag: Object storage ETag for versioning (optional)
        version_id: Explicit version identifier (optional)
        created_at: When the blob was created
        metadata: Additional metadata (dimensions, duration, etc.)

    Example:
        >>> ref = BlobReference(
        ...     uri="s3://data/images/img001.png",
        ...     content_type=ContentType.IMAGE_PNG,
        ...     checksum="abc123...",
        ...     hash_algorithm=HashAlgorithm.SHA256,
        ...     size_bytes=1024000,
        ... )
        >>> # Reference can be stored in a feature column
        >>> feature_group.write({"image_ref": ref, "label": "cat"})
    """
    uri: str
    content_type: ContentType | str
    checksum: str
    hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256
    size_bytes: int = 0
    etag: str | None = None
    version_id: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict, hash=False)

    # Internal tracking
    _registry_id: str | None = field(default=None, repr=False, hash=False)

    def __post_init__(self):
        # Convert string content_type to enum if needed
        if isinstance(self.content_type, str):
            try:
                object.__setattr__(self, 'content_type', ContentType(self.content_type))
            except ValueError:
                pass  # Keep as string for unknown types

    @property
    def scheme(self) -> str:
        """Extract URI scheme (s3, gs, az, file, etc.)."""
        if "://" in self.uri:
            return self.uri.split("://")[0]
        return "file"

    @property
    def path(self) -> str:
        """Extract path from URI."""
        if "://" in self.uri:
            return self.uri.split("://", 1)[1]
        return self.uri

    @property
    def bucket(self) -> str | None:
        """Extract bucket name for object storage URIs."""
        if self.scheme in ("s3", "gs", "az"):
            parts = self.path.split("/", 1)
            return parts[0] if parts else None
        return None

    @property
    def key(self) -> str | None:
        """Extract object key for object storage URIs."""
        if self.scheme in ("s3", "gs", "az"):
            parts = self.path.split("/", 1)
            return parts[1] if len(parts) > 1 else None
        return None

    @property
    def is_image(self) -> bool:
        """Check if blob is an image type."""
        ct = self.content_type
        if isinstance(ct, ContentType):
            return ct.value.startswith("image/")
        return str(ct).startswith("image/")

    @property
    def is_audio(self) -> bool:
        """Check if blob is an audio type."""
        ct = self.content_type
        if isinstance(ct, ContentType):
            return ct.value.startswith("audio/")
        return str(ct).startswith("audio/")

    @property
    def is_video(self) -> bool:
        """Check if blob is a video type."""
        ct = self.content_type
        if isinstance(ct, ContentType):
            return ct.value.startswith("video/")
        return str(ct).startswith("video/")

    def to_dict(self) -> dict[str, Any]:
        """Serialize reference to dictionary."""
        ct = self.content_type
        if isinstance(ct, ContentType):
            ct = ct.value

        return {
            "uri": self.uri,
            "content_type": ct,
            "checksum": self.checksum,
            "hash_algorithm": self.hash_algorithm.value,
            "size_bytes": self.size_bytes,
            "etag": self.etag,
            "version_id": self.version_id,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BlobReference":
        """Deserialize reference from dictionary."""
        return cls(
            uri=data["uri"],
            content_type=data["content_type"],
            checksum=data["checksum"],
            hash_algorithm=HashAlgorithm(data.get("hash_algorithm", "sha256")),
            size_bytes=data.get("size_bytes", 0),
            etag=data.get("etag"),
            version_id=data.get("version_id"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            metadata=data.get("metadata", {}),
        )

    def with_metadata(self, **kwargs) -> "BlobReference":
        """Create new reference with additional metadata."""
        new_metadata = {**self.metadata, **kwargs}
        return BlobReference(
            uri=self.uri,
            content_type=self.content_type,
            checksum=self.checksum,
            hash_algorithm=self.hash_algorithm,
            size_bytes=self.size_bytes,
            etag=self.etag,
            version_id=self.version_id,
            created_at=self.created_at,
            metadata=new_metadata,
        )


# =============================================================================
# Integrity Validation
# =============================================================================

@dataclass
class IntegrityPolicy:
    """
    Policy for enforcing referential integrity.

    Attributes:
        mode: When to perform integrity checks
        fail_on_missing: Raise error if blob is missing
        fail_on_mismatch: Raise error if checksum doesn't match
        cache_validation: Cache validation results for duration
        periodic_interval: Interval for periodic validation (if mode=PERIODIC)
    """
    mode: IntegrityMode = IntegrityMode.ON_WRITE
    fail_on_missing: bool = True
    fail_on_mismatch: bool = True
    cache_validation_seconds: int = 3600  # 1 hour
    periodic_interval: str = "1h"

    @classmethod
    def strict(cls) -> "IntegrityPolicy":
        """Create strict policy that validates on every access."""
        return cls(mode=IntegrityMode.STRICT)

    @classmethod
    def on_write(cls) -> "IntegrityPolicy":
        """Create policy that validates only on write."""
        return cls(mode=IntegrityMode.ON_WRITE)

    @classmethod
    def lazy(cls) -> "IntegrityPolicy":
        """Create lazy policy with manual validation."""
        return cls(mode=IntegrityMode.LAZY, fail_on_missing=False, fail_on_mismatch=False)


@dataclass
class ValidationResult:
    """Result of integrity validation."""
    reference: BlobReference
    status: BlobStatus
    valid: bool
    message: str = ""
    checked_at: datetime = field(default_factory=datetime.utcnow)
    actual_checksum: str | None = None
    actual_size: int | None = None

    @property
    def checksum_matches(self) -> bool:
        """Check if actual checksum matches reference."""
        if self.actual_checksum is None:
            return True  # Not checked
        return self.actual_checksum == self.reference.checksum

    @property
    def size_matches(self) -> bool:
        """Check if actual size matches reference."""
        if self.actual_size is None or self.reference.size_bytes == 0:
            return True  # Not checked or not specified
        return self.actual_size == self.reference.size_bytes


# =============================================================================
# Blob Registry
# =============================================================================

@dataclass
class BlobRegistry:
    """
    Centralized registry for tracking and validating blob references.

    The registry maintains a catalog of all blob references and provides:
    - Reference registration and tracking
    - Integrity validation
    - Reference resolution
    - Garbage collection for orphaned blobs

    Example:
        >>> registry = BlobRegistry(policy=IntegrityPolicy.on_write())
        >>> ref = registry.register(
        ...     uri="s3://bucket/image.png",
        ...     content_type=ContentType.IMAGE_PNG,
        ...     checksum="abc123...",
        ... )
        >>> # Later, validate the reference
        >>> result = registry.validate(ref)
        >>> assert result.valid
    """
    name: str = "default"
    policy: IntegrityPolicy = field(default_factory=IntegrityPolicy)

    # Storage backend for registry data
    _references: dict[str, BlobReference] = field(default_factory=dict, repr=False)
    _validation_cache: dict[str, ValidationResult] = field(default_factory=dict, repr=False)
    _uri_index: dict[str, str] = field(default_factory=dict, repr=False)  # uri -> registry_id

    # Pluggable storage resolver
    _storage_resolver: Callable[[str], Any] | None = field(default=None, repr=False)

    def register(
        self,
        uri: str,
        content_type: ContentType | str,
        checksum: str,
        hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256,
        size_bytes: int = 0,
        etag: str | None = None,
        version_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        validate: bool | None = None,
    ) -> BlobReference:
        """
        Register a new blob reference.

        Args:
            uri: Location of the blob
            content_type: MIME type of content
            checksum: Content hash for integrity
            hash_algorithm: Algorithm used for checksum
            size_bytes: Size of blob in bytes
            etag: Object storage ETag
            version_id: Version identifier
            metadata: Additional metadata
            validate: Override policy to validate immediately

        Returns:
            BlobReference: Registered reference with registry ID

        Raises:
            IntegrityError: If validation fails and policy requires it
        """
        registry_id = str(uuid.uuid4())

        ref = BlobReference(
            uri=uri,
            content_type=content_type,
            checksum=checksum,
            hash_algorithm=hash_algorithm,
            size_bytes=size_bytes,
            etag=etag,
            version_id=version_id,
            metadata=metadata or {},
            _registry_id=registry_id,
        )

        # Validate if policy requires or explicitly requested
        should_validate = validate if validate is not None else (
            self.policy.mode in (IntegrityMode.STRICT, IntegrityMode.ON_WRITE)
        )

        if should_validate:
            result = self._validate_reference(ref)
            if not result.valid:
                if self.policy.fail_on_missing and result.status == BlobStatus.MISSING:
                    raise IntegrityError(f"Blob not found: {uri}")
                if self.policy.fail_on_mismatch and result.status == BlobStatus.INVALID:
                    raise IntegrityError(f"Checksum mismatch for: {uri}")

        # Store in registry
        self._references[registry_id] = ref
        self._uri_index[uri] = registry_id

        return ref

    def register_batch(
        self,
        references: list[dict[str, Any]],
        validate: bool | None = None,
    ) -> list[BlobReference]:
        """
        Register multiple blob references.

        Args:
            references: List of reference dictionaries
            validate: Override policy to validate

        Returns:
            List of registered BlobReferences
        """
        return [
            self.register(validate=validate, **ref_data)
            for ref_data in references
        ]

    def get(self, registry_id: str) -> BlobReference | None:
        """Get reference by registry ID."""
        return self._references.get(registry_id)

    def get_by_uri(self, uri: str) -> BlobReference | None:
        """Get reference by URI."""
        registry_id = self._uri_index.get(uri)
        if registry_id:
            return self._references.get(registry_id)
        return None

    def validate(self, ref: BlobReference) -> ValidationResult:
        """
        Validate a blob reference.

        Checks that the blob exists and integrity is maintained.

        Args:
            ref: Reference to validate

        Returns:
            ValidationResult with status and details
        """
        # Check cache
        cache_key = f"{ref.uri}:{ref.checksum}"
        if cache_key in self._validation_cache:
            cached = self._validation_cache[cache_key]
            age = (datetime.utcnow() - cached.checked_at).total_seconds()
            if age < self.policy.cache_validation_seconds:
                return cached

        result = self._validate_reference(ref)
        self._validation_cache[cache_key] = result
        return result

    def validate_batch(self, refs: list[BlobReference]) -> list[ValidationResult]:
        """Validate multiple references."""
        return [self.validate(ref) for ref in refs]

    def _validate_reference(self, ref: BlobReference) -> ValidationResult:
        """Internal validation logic."""
        # Mock implementation - in production would check actual storage
        # For now, simulate validation
        return ValidationResult(
            reference=ref,
            status=BlobStatus.VALID,
            valid=True,
            message="Validation successful (mock)",
            actual_checksum=ref.checksum,
            actual_size=ref.size_bytes,
        )

    def list_references(
        self,
        content_type: ContentType | str | None = None,
        prefix: str | None = None,
        status: BlobStatus | None = None,
    ) -> list[BlobReference]:
        """
        List references with optional filtering.

        Args:
            content_type: Filter by content type
            prefix: Filter by URI prefix
            status: Filter by validation status

        Returns:
            List of matching references
        """
        results = []
        for ref in self._references.values():
            if content_type and ref.content_type != content_type:
                continue
            if prefix and not ref.uri.startswith(prefix):
                continue
            # Status filtering would require cached validation results
            results.append(ref)
        return results

    def delete(self, ref: BlobReference) -> bool:
        """
        Remove reference from registry.

        Note: This does NOT delete the actual blob, only the reference.
        """
        registry_id = ref._registry_id
        if registry_id and registry_id in self._references:
            del self._references[registry_id]
            if ref.uri in self._uri_index:
                del self._uri_index[ref.uri]
            return True
        return False

    def find_orphans(self) -> list[BlobReference]:
        """
        Find references to blobs that no longer exist.

        Returns:
            List of orphaned references
        """
        orphans = []
        for ref in self._references.values():
            result = self.validate(ref)
            if result.status == BlobStatus.MISSING:
                orphans.append(ref)
        return orphans

    def compute_checksum(self, data: bytes, algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> str:
        """
        Compute checksum for data.

        Args:
            data: Raw bytes to hash
            algorithm: Hash algorithm to use

        Returns:
            Hex-encoded checksum string
        """
        if algorithm == HashAlgorithm.SHA256:
            return hashlib.sha256(data).hexdigest()
        elif algorithm == HashAlgorithm.SHA512:
            return hashlib.sha512(data).hexdigest()
        elif algorithm == HashAlgorithm.MD5:
            return hashlib.md5(data).hexdigest()
        else:
            # For BLAKE3 and XXH3, would need additional libraries
            return hashlib.sha256(data).hexdigest()


# =============================================================================
# Multimodal Source
# =============================================================================

@dataclass
class MultimodalSource:
    """
    Source connector for multimodal datasets.

    Scans object storage for multimodal files and produces BlobReferences
    rather than loading the actual data.

    Attributes:
        name: Source identifier
        uri_prefix: Base path to scan for blobs
        content_types: Filter by content types (None = all)
        registry: Registry to use for references
        compute_checksums: Whether to compute checksums on scan
        recursive: Scan subdirectories
        glob_pattern: Pattern to match files

    Example:
        >>> source = MultimodalSource(
        ...     name="training_images",
        ...     uri_prefix="s3://datasets/imagenet/train/",
        ...     content_types=[ContentType.IMAGE_JPEG, ContentType.IMAGE_PNG],
        ... )
        >>> refs = source.scan()  # Returns BlobReferences, not image data
    """
    name: str
    uri_prefix: str
    content_types: list[ContentType] | None = None
    registry: BlobRegistry | None = None
    compute_checksums: bool = True
    hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256
    recursive: bool = True
    glob_pattern: str = "*"

    # Metadata extraction
    extract_metadata: bool = True
    metadata_fields: list[str] | None = None  # None = extract all available

    def scan(self) -> list[BlobReference]:
        """
        Scan source location and return blob references.

        Returns:
            List of BlobReferences for matching files
        """
        # Mock implementation - would use boto3, gcsfs, etc. in production
        # Returns sample references for demonstration
        refs = []

        # Simulate finding some files
        sample_files = [
            ("image_001.png", ContentType.IMAGE_PNG, 1024000),
            ("image_002.jpg", ContentType.IMAGE_JPEG, 2048000),
            ("audio_001.wav", ContentType.AUDIO_WAV, 4096000),
        ]

        for filename, content_type, size in sample_files:
            if self.content_types and content_type not in self.content_types:
                continue

            uri = f"{self.uri_prefix.rstrip('/')}/{filename}"
            checksum = hashlib.sha256(uri.encode()).hexdigest()[:16]  # Mock checksum

            ref = BlobReference(
                uri=uri,
                content_type=content_type,
                checksum=checksum,
                hash_algorithm=self.hash_algorithm,
                size_bytes=size,
                metadata={"source": self.name},
            )

            if self.registry:
                ref = self.registry.register(
                    uri=uri,
                    content_type=content_type,
                    checksum=checksum,
                    hash_algorithm=self.hash_algorithm,
                    size_bytes=size,
                    metadata={"source": self.name},
                    validate=False,  # Don't validate during scan
                )

            refs.append(ref)

        return refs

    def get_reference(self, relative_path: str) -> BlobReference | None:
        """
        Get reference for a specific file.

        Args:
            relative_path: Path relative to uri_prefix

        Returns:
            BlobReference if found, None otherwise
        """
        uri = f"{self.uri_prefix.rstrip('/')}/{relative_path}"
        if self.registry:
            return self.registry.get_by_uri(uri)
        return None


# =============================================================================
# Reference-Aware Transform Context
# =============================================================================

@dataclass
class MultimodalContext:
    """
    Context for multimodal transforms.

    Provides utilities for working with blob references in transforms
    without loading the actual data.
    """
    registry: BlobRegistry

    # Caches for resolved data (when actually needed)
    _data_cache: dict[str, bytes] = field(default_factory=dict, repr=False)
    _max_cache_bytes: int = 1024 * 1024 * 100  # 100MB cache

    def resolve(self, ref: BlobReference) -> bytes:
        """
        Resolve reference to actual data.

        Use sparingly - defeats purpose of reference passing.

        Args:
            ref: Reference to resolve

        Returns:
            Raw bytes of the blob
        """
        if ref.uri in self._data_cache:
            return self._data_cache[ref.uri]

        # Mock - would fetch from actual storage
        data = b"mock data for " + ref.uri.encode()

        # Cache if under limit
        if len(self._data_cache) < 1000:  # Simple cache limit
            self._data_cache[ref.uri] = data

        return data

    def create_derived(
        self,
        source_ref: BlobReference,
        new_uri: str,
        content_type: ContentType | str,
        processor: Callable[[bytes], bytes],
    ) -> BlobReference:
        """
        Create a derived blob from a source reference.

        The processor function transforms the source data, and a new
        reference is created for the output.

        Args:
            source_ref: Source blob reference
            new_uri: URI for the derived blob
            content_type: Content type of derived blob
            processor: Function to transform source -> derived

        Returns:
            BlobReference for the derived blob
        """
        # Resolve source, process, and create new reference
        source_data = self.resolve(source_ref)
        derived_data = processor(source_data)

        checksum = self.registry.compute_checksum(derived_data)

        return self.registry.register(
            uri=new_uri,
            content_type=content_type,
            checksum=checksum,
            size_bytes=len(derived_data),
            metadata={
                "derived_from": source_ref.uri,
                "source_checksum": source_ref.checksum,
            },
        )

    def batch_validate(self, refs: list[BlobReference]) -> dict[str, ValidationResult]:
        """
        Validate multiple references in batch.

        Args:
            refs: List of references to validate

        Returns:
            Dict mapping URI to validation result
        """
        return {
            ref.uri: self.registry.validate(ref)
            for ref in refs
        }


# =============================================================================
# Reference Column Type
# =============================================================================

@dataclass
class BlobReferenceType:
    """
    Feature type for storing blob references.

    This type can be used when defining feature schemas to indicate
    that a column stores blob references.

    Example:
        >>> fg.create_features_from_schema({
        ...     "image": BlobReferenceType(content_types=[ContentType.IMAGE_PNG]),
        ...     "embedding": "embedding[512]",
        ...     "label": "string",
        ... })
    """
    content_types: list[ContentType] | None = None
    registry: str = "default"  # Name of registry to use
    validate_on_write: bool = True

    @property
    def type_name(self) -> str:
        return "blob_reference"

    def validate(self, value: Any) -> bool:
        """Check if value is a valid blob reference."""
        if not isinstance(value, (BlobReference, dict)):
            return False

        if isinstance(value, dict):
            required_fields = {"uri", "checksum", "content_type"}
            if not required_fields.issubset(value.keys()):
                return False

        return True


# =============================================================================
# Exceptions
# =============================================================================

class IntegrityError(Exception):
    """Raised when blob integrity check fails."""
    pass


class ReferenceNotFoundError(Exception):
    """Raised when a blob reference cannot be resolved."""
    pass


# =============================================================================
# Utility Functions
# =============================================================================

def create_reference(
    uri: str,
    content_type: ContentType | str,
    checksum: str | None = None,
    data: bytes | None = None,
    **kwargs,
) -> BlobReference:
    """
    Convenience function to create a blob reference.

    Either checksum or data must be provided. If data is provided,
    the checksum will be computed automatically.

    Args:
        uri: Blob location
        content_type: MIME type
        checksum: Pre-computed checksum (optional if data provided)
        data: Raw data to compute checksum from (optional if checksum provided)
        **kwargs: Additional BlobReference parameters

    Returns:
        BlobReference
    """
    if checksum is None and data is None:
        raise ValueError("Either checksum or data must be provided")

    if checksum is None:
        algorithm = kwargs.get("hash_algorithm", HashAlgorithm.SHA256)
        if algorithm == HashAlgorithm.SHA256:
            checksum = hashlib.sha256(data).hexdigest()
        elif algorithm == HashAlgorithm.SHA512:
            checksum = hashlib.sha512(data).hexdigest()
        else:
            checksum = hashlib.sha256(data).hexdigest()
        kwargs["size_bytes"] = len(data)

    return BlobReference(
        uri=uri,
        content_type=content_type,
        checksum=checksum,
        **kwargs,
    )


def infer_content_type(uri: str) -> ContentType | None:
    """
    Infer content type from file extension.

    Args:
        uri: File URI or path

    Returns:
        ContentType if recognized, None otherwise
    """
    extension_map = {
        ".png": ContentType.IMAGE_PNG,
        ".jpg": ContentType.IMAGE_JPEG,
        ".jpeg": ContentType.IMAGE_JPEG,
        ".webp": ContentType.IMAGE_WEBP,
        ".gif": ContentType.IMAGE_GIF,
        ".tiff": ContentType.IMAGE_TIFF,
        ".bmp": ContentType.IMAGE_BMP,
        ".wav": ContentType.AUDIO_WAV,
        ".mp3": ContentType.AUDIO_MP3,
        ".flac": ContentType.AUDIO_FLAC,
        ".ogg": ContentType.AUDIO_OGG,
        ".mp4": ContentType.VIDEO_MP4,
        ".webm": ContentType.VIDEO_WEBM,
        ".avi": ContentType.VIDEO_AVI,
        ".mov": ContentType.VIDEO_MOV,
        ".pdf": ContentType.DOCUMENT_PDF,
        ".npy": ContentType.ARRAY_NPY,
        ".npz": ContentType.ARRAY_NPZ,
        ".pt": ContentType.TENSOR_PT,
        ".safetensors": ContentType.TENSOR_SAFETENSORS,
    }

    uri_lower = uri.lower()
    for ext, content_type in extension_map.items():
        if uri_lower.endswith(ext):
            return content_type

    return None
