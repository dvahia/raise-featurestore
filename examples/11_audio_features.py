"""
Example 11: Audio Blobs and Segment Features

Demonstrates the audio modality: registering audio blob references,
extracting time-range segments (clips), and building feature schemas for
speech, music, and general audio datasets. Audio segment references use the
same clip() API as video — no audio bytes are read or copied.

Key concepts:
- Audio blob references with time ranges (segments)
- Audio-specific metadata: sample_rate, channels, bit_depth, duration
- Sliding-window segment extraction from long recordings
- Features alongside audio refs: transcripts, speaker IDs, language codes
- Audio quality scoring based on SNR, silence ratio, clipping
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
# Setup: Feature Groups
# =============================================================================

fs = FeatureStore("acme/mlplatform/speech-data")

# Full audio recordings catalog
recording_catalog = fs.create_feature_group(
    "audio-recordings",
    description="Full audio recordings with file-level metadata",
    entity_key="recording_id",
    if_exists="skip",
)

recording_catalog.create_features_from_schema({
    "recording_id": "string",
    "audio_ref": BlobRef(content_types=["audio/wav", "audio/flac", "audio/mpeg"]),
    "duration_sec": "float64",
    "sample_rate_hz": "int64",
    "num_channels": "int64",
    "bit_depth": "int64",
    "language": "string",
    "source_domain": "string",   # "telephony", "broadcast", "studio", "podcast"
    "recording_environment": "string",
}, if_exists="skip")

# Audio segments (sub-clips with per-segment features)
segment_features = fs.create_feature_group(
    "audio-segments",
    description="Audio segments with transcript and speaker features",
    entity_key="segment_id",
    if_exists="skip",
)

segment_features.create_features_from_schema({
    "segment_id": "string",
    "recording_id": "string",         # FK to audio-recordings
    "segment_ref": BlobRef(content_types=["audio/wav", "audio/flac", "audio/mpeg"]),
    "start_sec": "float64",
    "end_sec": "float64",
    "duration_sec": "float64",
    # Transcript
    "transcript": "string",
    "transcript_confidence": "float64",
    "language": "string",
    # Speaker
    "speaker_id": "string",
    "num_speakers": "int64",
    "is_overlapping_speech": "bool",
    # Audio quality signals
    "snr_db": "float64",              # Signal-to-noise ratio
    "silence_ratio": "float64",       # Fraction of the segment that is silence
    "clipping_ratio": "float64",      # Fraction of samples that clip
    "is_music": "bool",
    # Embeddings
    "audio_embedding": "float32[512]",
    "speaker_embedding": "float32[256]",
}, if_exists="skip")

print("Audio feature groups created")

# =============================================================================
# Registering Full Audio References
# =============================================================================

print("\n" + "=" * 60)
print("FULL AUDIO REFERENCES")
print("=" * 60)

registry = BlobRegistry(
    name="audio-store",
    policy=IntegrityPolicy.on_write(),
)

# Studio recording — high quality
studio_ref = registry.register(
    uri="s3://acme-audio/studio/speaker_001_session_a.flac",
    content_type=ContentType.AUDIO_FLAC,
    checksum="9f86d081884c7d659a2fe...",
    hash_algorithm=HashAlgorithm.SHA256,
    size_bytes=450_000_000,
    metadata={
        "duration_sec": 7200.0,    # 2 hours
        "sample_rate_hz": 44100,
        "num_channels": 1,
        "bit_depth": 24,
        "language": "en",
        "source_domain": "studio",
        "recording_environment": "anechoic_chamber",
    },
)

# Telephone recording — 8kHz narrowband
phone_ref = registry.register(
    uri="s3://acme-audio/telephony/call_center_batch_001.wav",
    content_type=ContentType.AUDIO_WAV,
    checksum="a665a45920422f9d417e...",
    size_bytes=80_000_000,
    metadata={
        "duration_sec": 3600.0,
        "sample_rate_hz": 8000,
        "num_channels": 2,         # stereo: agent + customer
        "bit_depth": 16,
        "language": "en",
        "source_domain": "telephony",
    },
)

print(f"Studio recording: {studio_ref.uri}")
print(f"  Duration: {studio_ref.metadata['duration_sec']/3600:.1f}h")
print(f"  Sample rate: {studio_ref.metadata['sample_rate_hz']:,} Hz")
print(f"  Is audio: {studio_ref.is_audio}")
print(f"  Is clip: {studio_ref.is_clip}")

print(f"\nTelephone recording: {phone_ref.uri}")
print(f"  Sample rate: {phone_ref.metadata['sample_rate_hz']:,} Hz (narrowband)")

# =============================================================================
# Segment Extraction via Clips
# =============================================================================

print("\n" + "=" * 60)
print("AUDIO SEGMENT EXTRACTION")
print("=" * 60)

# Create a 30-second segment starting at 5 minutes
intro_segment = studio_ref.clip(start_sec=300.0, end_sec=330.0)
print(f"Intro segment:")
print(f"  URI: {intro_segment.uri}  (same file)")
print(f"  Range: {intro_segment.time_range}")
print(f"  Duration: {intro_segment.time_range.duration_sec}s")

# Variable-length segment (from a diarization result)
speaker_turn = TimeRange(start_sec=1234.5, end_sec=1267.2)
turn_segment = studio_ref.clip(speaker_turn.start_sec, speaker_turn.end_sec)
print(f"\nSpeaker turn segment:")
print(f"  Range: {turn_segment.time_range}")
print(f"  Duration: {turn_segment.time_range.duration_sec:.1f}s")

# =============================================================================
# Speaker Diarization Segments
# =============================================================================

print("\n" + "=" * 60)
print("DIARIZATION-BASED SEGMENTS")
print("=" * 60)

# Simulated output from a speaker diarization model
diarization_output = [
    {"speaker": "SPEAKER_00", "start": 0.0,    "end": 12.4},
    {"speaker": "SPEAKER_01", "start": 12.4,   "end": 28.9},
    {"speaker": "SPEAKER_00", "start": 28.9,   "end": 45.1},
    {"speaker": "SPEAKER_01", "start": 45.1,   "end": 62.3},
    {"speaker": "SPEAKER_00", "start": 62.3,   "end": 80.0},
]

def diarization_to_feature_rows(
    audio_ref: BlobReference,
    recording_id: str,
    diarization: list[dict],
) -> list[dict]:
    """Convert diarization output to feature rows with clip references."""
    rows = []
    for i, turn in enumerate(diarization):
        segment_ref = audio_ref.clip(start_sec=turn["start"], end_sec=turn["end"])
        rows.append({
            "segment_id": f"{recording_id}_turn_{i:05d}",
            "recording_id": recording_id,
            "segment_ref": segment_ref.to_dict(),
            "start_sec": turn["start"],
            "end_sec": turn["end"],
            "duration_sec": turn["end"] - turn["start"],
            "speaker_id": turn["speaker"],
            "num_speakers": 1,
            "is_overlapping_speech": False,
        })
    return rows

rows = diarization_to_feature_rows(studio_ref, "speaker_001_session_a", diarization_output)
print(f"Created {len(rows)} segment feature rows from diarization")
for row in rows:
    print(f"  {row['segment_id']}: {row['speaker_id']} "
          f"[{row['start_sec']:.1f}s – {row['end_sec']:.1f}s] "
          f"({row['duration_sec']:.1f}s)")

# =============================================================================
# Fixed-Length Windowing (for Pre-Training)
# =============================================================================

print("\n" + "=" * 60)
print("SLIDING-WINDOW SEGMENTS (PRE-TRAINING)")
print("=" * 60)

def sliding_window_segments(
    audio_ref: BlobReference,
    recording_id: str,
    window_sec: float = 10.0,
    hop_sec: float = 5.0,
) -> list[dict]:
    """
    Extract overlapping fixed-length windows.

    This is common for pre-training audio encoders where aligned
    transcripts are not available.
    """
    total = audio_ref.metadata.get("duration_sec", 0.0)
    segments = []
    t, idx = 0.0, 0
    while t + window_sec <= total:
        seg_ref = audio_ref.clip(start_sec=t, end_sec=t + window_sec)
        segments.append({
            "segment_id": f"{recording_id}_win_{idx:06d}",
            "recording_id": recording_id,
            "segment_ref": seg_ref.to_dict(),
            "start_sec": t,
            "end_sec": t + window_sec,
            "duration_sec": window_sec,
        })
        t += hop_sec
        idx += 1
    return segments

windows = sliding_window_segments(
    audio_ref=studio_ref,
    recording_id="speaker_001_session_a",
    window_sec=10.0,
    hop_sec=5.0,
)

print(f"Sliding-window segments: {len(windows)}")
print(f"  Window: 10s, hop: 5s (50% overlap)")
print(f"  Total segment-hours: {sum(w['duration_sec'] for w in windows) / 3600:.1f}h")
print(f"  First 3 windows:")
for w in windows[:3]:
    print(f"    {w['segment_id']}: [{w['start_sec']:.1f}s – {w['end_sec']:.1f}s]")

# =============================================================================
# Multi-Channel Audio (Telephony: Agent + Customer)
# =============================================================================

print("\n" + "=" * 60)
print("MULTI-CHANNEL AUDIO")
print("=" * 60)

# For telephony recordings, both channels are sub-references of the same file.
# The metadata distinguishes them; the actual channel splitting happens in
# the transform that reads the blob.

agent_segment = phone_ref.clip(start_sec=0.0, end_sec=30.0).with_metadata(
    channel=0,
    channel_role="agent",
)
customer_segment = phone_ref.clip(start_sec=0.0, end_sec=30.0).with_metadata(
    channel=1,
    channel_role="customer",
)

print(f"Agent segment: channel={agent_segment.metadata['channel_role']}")
print(f"  URI: {agent_segment.uri}")
print(f"  Is clip: {agent_segment.is_clip}")
print(f"  Range: {agent_segment.time_range}")

print(f"\nCustomer segment: channel={customer_segment.metadata['channel_role']}")
print(f"  Channel: {customer_segment.metadata['channel']}")

# =============================================================================
# Audio Content Types
# =============================================================================

print("\n" + "=" * 60)
print("AUDIO CONTENT TYPES")
print("=" * 60)

audio_types = [
    (ContentType.AUDIO_WAV,  ".wav",  "16/24-bit PCM, no compression — archival"),
    (ContentType.AUDIO_FLAC, ".flac", "Lossless compression — preferred for training"),
    (ContentType.AUDIO_MP3,  ".mp3",  "Lossy, widely available — web data"),
    (ContentType.AUDIO_OGG,  ".ogg",  "Vorbis — open format"),
    (ContentType.AUDIO_AAC,  ".aac",  "Lossy — mobile/streaming"),
]

print("\nSupported audio content types:")
for ct, ext, note in audio_types:
    print(f"  {ct.value:<20} {ext:<8}  {note}")

print("\n" + "=" * 60)
print("ALL AUDIO EXAMPLES COMPLETE!")
print("=" * 60)
