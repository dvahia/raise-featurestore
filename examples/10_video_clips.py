"""
Example 10: Video Blobs and Sub-Clip References

Demonstrates managing video data in the feature store: full-video blob
references, sub-clip extraction via TimeRange, and video-specific feature
schemas. The clip() API lets downstream transforms reference a segment of a
video without copying any bytes.

Key concepts:
- BlobReference with time_range for clips within a video file
- TimeRange: start/end in seconds, duration property
- video_ref.clip(start_sec, end_sec): create clip references
- Video-specific metadata (fps, resolution, codec, duration)
- Scanning a video dataset and extracting fixed-length clips
- Multi-clip feature groups for dense video understanding
"""

from raise_ import (
    FeatureStore,
    BlobRef,
    BlobReference,
    BlobRegistry,
    ContentType,
    TimeRange,
    IntegrityPolicy,
    create_reference,
)
from raise_.transforms import HashAlgorithm, infer_content_type

# =============================================================================
# Setup
# =============================================================================

fs = FeatureStore("acme/mlplatform/video-understanding")

# Feature group for full video metadata
video_catalog = fs.create_feature_group(
    "video-catalog",
    description="Full video files with metadata",
    entity_key="video_id",
    if_exists="skip",
)

video_catalog.create_features_from_schema({
    "video_id": "string",
    "video_ref": BlobRef(content_types=["video/mp4", "video/webm"]),
    # Video-level metadata stored as scalar features
    "duration_sec": "float64",
    "fps": "float64",
    "width": "int64",
    "height": "int64",
    "codec": "string",
    "bitrate_kbps": "int64",
    "has_audio": "bool",
    "language": "string",
    "source": "string",
}, if_exists="skip")

# Feature group for video clips (sub-segments)
clip_features = fs.create_feature_group(
    "video-clips",
    description="Sub-clip references with per-clip features",
    entity_key="clip_id",
    if_exists="skip",
)

clip_features.create_features_from_schema({
    "clip_id": "string",
    "video_id": "string",           # FK back to video-catalog
    "clip_ref": BlobRef(content_types=["video/mp4", "video/webm"]),
    "start_sec": "float64",
    "end_sec": "float64",
    "duration_sec": "float64",
    # Per-clip semantics
    "scene_label": "string",
    "activity_label": "string",
    "transcript": "string",
    "transcript_language": "string",
    "num_speakers": "int64",
    # Embeddings computed from the clip
    "visual_embedding": "float32[768]",
    "audio_embedding": "float32[512]",
}, if_exists="skip")

print("Feature groups created")

# =============================================================================
# Registering Full Video References
# =============================================================================

print("\n" + "=" * 60)
print("FULL VIDEO REFERENCES")
print("=" * 60)

registry = BlobRegistry(
    name="video-store",
    policy=IntegrityPolicy.on_write(),
)

# Register a full video file
video_ref = registry.register(
    uri="s3://acme-videos/raw/interview_2024_001.mp4",
    content_type=ContentType.VIDEO_MP4,
    checksum="e3b0c44298fc1c149afb...",
    hash_algorithm=HashAlgorithm.SHA256,
    size_bytes=2_500_000_000,   # 2.5 GB
    etag="abc123def456",
    metadata={
        "duration_sec": 3720.0,   # 62 minutes
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "codec": "h264",
        "bitrate_kbps": 5000,
        "has_audio": True,
        "language": "en",
    },
)

print(f"Registered video: {video_ref.uri}")
print(f"  Size: {video_ref.size_bytes / 1024**3:.2f} GB")
print(f"  Duration: {video_ref.metadata['duration_sec'] / 60:.1f} min")
print(f"  Resolution: {video_ref.metadata['width']}x{video_ref.metadata['height']}")
print(f"  Is video: {video_ref.is_video}")
print(f"  Is clip: {video_ref.is_clip}")

# =============================================================================
# Creating Sub-Clip References
# =============================================================================

print("\n" + "=" * 60)
print("SUB-CLIP REFERENCES")
print("=" * 60)

# Create a 30-second clip starting at 2 minutes
intro_clip = video_ref.clip(start_sec=120.0, end_sec=150.0)
print(f"\nIntro clip:")
print(f"  URI: {intro_clip.uri}  (same file, different range)")
print(f"  Time range: {intro_clip.time_range}")
print(f"  Duration: {intro_clip.time_range.duration_sec:.1f}s")
print(f"  Is clip: {intro_clip.is_clip}")

# TimeRange as a standalone value
interview_segment = TimeRange(start_sec=300.0, end_sec=900.0)
print(f"\nInterview segment: {interview_segment}")
print(f"  Duration: {interview_segment.duration_sec / 60:.1f} min")

# Create clip from TimeRange
key_moment = video_ref.clip(
    start_sec=interview_segment.start_sec,
    end_sec=interview_segment.end_sec,
)
print(f"\nKey moment clip:")
print(f"  Start: {key_moment.time_range.start_sec}s")
print(f"  End: {key_moment.time_range.end_sec}s")
print(f"  Duration: {key_moment.duration_sec:.1f}s")

# Clips share the source URI and checksum — no data is copied
assert intro_clip.uri == video_ref.uri
assert intro_clip.checksum == video_ref.checksum
print("\nClips share source URI and checksum — no data movement")

# =============================================================================
# Fixed-Length Clip Extraction
# =============================================================================

print("\n" + "=" * 60)
print("FIXED-LENGTH CLIP EXTRACTION")
print("=" * 60)

def extract_clips(
    video_ref: BlobReference,
    video_id: str,
    clip_duration_sec: float = 10.0,
    stride_sec: float | None = None,
    overlap: float = 0.0,
) -> list[dict]:
    """
    Slice a video into fixed-length clip references.

    No data is read or copied. Returns dicts ready to write as feature rows.
    """
    stride = stride_sec or clip_duration_sec * (1.0 - overlap)
    total_duration = video_ref.metadata.get("duration_sec", 0.0)

    clips = []
    t = 0.0
    clip_index = 0

    while t + clip_duration_sec <= total_duration:
        clip_ref = video_ref.clip(start_sec=t, end_sec=t + clip_duration_sec)
        clips.append({
            "clip_id": f"{video_id}_clip_{clip_index:05d}",
            "video_id": video_id,
            "clip_ref": clip_ref.to_dict(),
            "start_sec": clip_ref.time_range.start_sec,
            "end_sec": clip_ref.time_range.end_sec,
            "duration_sec": clip_ref.time_range.duration_sec,
        })
        t += stride
        clip_index += 1

    return clips

# Extract 10-second clips with 50% overlap
clips = extract_clips(
    video_ref=video_ref,
    video_id="interview_2024_001",
    clip_duration_sec=10.0,
    overlap=0.5,
)

print(f"\nExtracted {len(clips)} clips from {video_ref.metadata['duration_sec']/60:.1f}-min video")
print(f"  Clip duration: 10.0s")
print(f"  Stride: 5.0s (50% overlap)")
print(f"  First clip: {clips[0]['start_sec']:.1f}s – {clips[0]['end_sec']:.1f}s")
print(f"  Last clip: {clips[-1]['start_sec']:.1f}s – {clips[-1]['end_sec']:.1f}s")

# =============================================================================
# Batch Registration: Video Dataset
# =============================================================================

print("\n" + "=" * 60)
print("VIDEO DATASET REGISTRATION")
print("=" * 60)

# Simulate a dataset of multiple videos
video_manifest = [
    {
        "video_id": "lecture_001",
        "uri": "s3://acme-videos/lectures/ml_101_lecture1.mp4",
        "size_bytes": 800_000_000,
        "duration_sec": 4200.0,
        "fps": 25.0,
        "width": 1280,
        "height": 720,
        "codec": "h264",
        "language": "en",
    },
    {
        "video_id": "lecture_002",
        "uri": "s3://acme-videos/lectures/ml_101_lecture2.mp4",
        "size_bytes": 750_000_000,
        "duration_sec": 3900.0,
        "fps": 25.0,
        "width": 1280,
        "height": 720,
        "codec": "h264",
        "language": "en",
    },
    {
        "video_id": "tutorial_001",
        "uri": "s3://acme-videos/tutorials/python_basics.webm",
        "size_bytes": 300_000_000,
        "duration_sec": 1800.0,
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "codec": "vp9",
        "language": "en",
    },
]

all_clips = []
for item in video_manifest:
    ref = registry.register(
        uri=item["uri"],
        content_type=infer_content_type(item["uri"]),
        checksum=f"mock_checksum_{item['video_id']}",
        size_bytes=item["size_bytes"],
        metadata={k: v for k, v in item.items() if k not in ("uri", "video_id", "size_bytes")},
        validate=False,
    )
    clips = extract_clips(ref, item["video_id"], clip_duration_sec=30.0, overlap=0.0)
    all_clips.extend(clips)
    print(f"  {item['video_id']}: {len(clips)} clips from {item['duration_sec']/60:.0f}-min video")

print(f"\nTotal clips across dataset: {len(all_clips)}")
total_clip_hours = sum(c["duration_sec"] for c in all_clips) / 3600
print(f"Total clip duration: {total_clip_hours:.1f} hours")

# =============================================================================
# Clip Reference Serialization
# =============================================================================

print("\n" + "=" * 60)
print("CLIP REFERENCE SERIALIZATION")
print("=" * 60)

# Serialize a clip reference to dict (for storage in feature rows)
clip_ref = video_ref.clip(start_sec=600.0, end_sec=660.0)
clip_dict = clip_ref.to_dict()

print(f"Serialized clip reference keys: {list(clip_dict.keys())}")
print(f"  time_range: {clip_dict['time_range']}")

# Restore from dict
restored = BlobReference.from_dict(clip_dict)
print(f"\nRestored clip:")
print(f"  URI: {restored.uri}")
print(f"  Is clip: {restored.is_clip}")
print(f"  Start: {restored.time_range.start_sec}s")
print(f"  End: {restored.time_range.end_sec}s")
print(f"  Duration: {restored.time_range.duration_sec}s")

# =============================================================================
# Clip with Enriched Metadata
# =============================================================================

print("\n" + "=" * 60)
print("CLIP WITH ENRICHED METADATA")
print("=" * 60)

# Add per-clip metadata without mutating the original
annotated_clip = clip_ref.with_metadata(
    scene="indoor_classroom",
    activity="lecture",
    num_speakers=1,
    noise_level="low",
    contains_slides=True,
)

print(f"Original clip metadata: {clip_ref.metadata}")
print(f"Annotated clip metadata: {annotated_clip.metadata}")
print(f"Time range preserved: {annotated_clip.time_range}")

# =============================================================================
# Video Content Types Reference
# =============================================================================

print("\n" + "=" * 60)
print("VIDEO CONTENT TYPES")
print("=" * 60)

video_types = [
    (ContentType.VIDEO_MP4,  ".mp4",  "H.264/H.265 — most common"),
    (ContentType.VIDEO_WEBM, ".webm", "VP9/AV1 — open format"),
    (ContentType.VIDEO_AVI,  ".avi",  "Legacy Windows format"),
    (ContentType.VIDEO_MOV,  ".mov",  "QuickTime — Apple ecosystem"),
]

print("\nSupported video content types:")
for ct, ext, note in video_types:
    print(f"  {ct.value:<25} {ext:<8}  {note}")
    inferred = infer_content_type(f"video{ext}")
    print(f"    infer_content_type('{ext}') → {inferred.value if inferred else 'None'}")

print("\n" + "=" * 60)
print("ALL VIDEO CLIP EXAMPLES COMPLETE!")
print("=" * 60)
