"""
Example 13: Dataset Mixing, Versioning, and Provenance

Demonstrates how to combine multiple data sources into a training dataset
using weighted, temperature-scaled, and proportional mixing strategies.
Also shows dataset versioning for reproducibility and provenance tracking
for compliance and attribution.

Key concepts:
- MixSource: one dataset source with a weight and optional filters
- DatasetMix: the full mixing configuration (strategy, weights, sample counts)
- MixingStrategy: WEIGHTED, TEMPERATURE, UNIFORM, PROPORTIONAL
- DatasetVersion: immutable snapshot with parent lineage
- DatasetProvenance: source attribution, license, collection date
- Executing a mix as a Job in the Raise feature store
"""

from datetime import datetime

from raise_ import (
    FeatureStore,
    # Mixing
    MixingStrategy,
    MixSource,
    DatasetMix,
    # Versioning
    DatasetVersion,
    DatasetProvenance,
    # Job infra
    Schedule,
    Target,
    SQLTransform,
    FeatureGroupSource,
)

# =============================================================================
# Setup
# =============================================================================

fs = FeatureStore("acme/pretraining/foundation")

# Target feature group that will receive the mixed dataset
mixed_dataset = fs.create_feature_group(
    "training-mix",
    description="Mixed training dataset for foundation model pre-training",
    entity_key="example_id",
    if_exists="skip",
)

mixed_dataset.create_features_from_schema({
    "example_id": "string",
    "text": "string",
    "source_domain": "string",    # which MixSource this came from
    "source_weight": "float64",   # effective sampling weight
    "language": "string",
    "token_count": "int64",
}, if_exists="skip")

print("Feature groups created")

# =============================================================================
# Defining Mix Sources
# =============================================================================

print("\n" + "=" * 60)
print("MIX SOURCES")
print("=" * 60)

# Web data: largest but noisiest — filtered before mixing
web_source = MixSource(
    feature_group="acme/pretraining/web-crawl/curated-documents",
    weight=0.70,
    filters=[
        "include_in_training == True",
        "language == 'en'",
        "token_count >= 50",
    ],
    tags=["web", "english"],
)

# Books: smaller but high quality
books_source = MixSource(
    feature_group="acme/pretraining/books/curated-documents",
    weight=0.15,
    filters=["include_in_training == True"],
    tags=["books", "long-form"],
)

# Code: specialized domain
code_source = MixSource(
    feature_group="acme/pretraining/code/curated-documents",
    weight=0.10,
    filters=["include_in_training == True"],
    tags=["code", "programming"],
)

# Academic papers: high quality, domain-specific
academic_source = MixSource(
    feature_group="acme/pretraining/arxiv/curated-documents",
    weight=0.05,
    filters=["include_in_training == True", "language == 'en'"],
    tags=["academic", "stem"],
)

print("Mix sources:")
for src, label in [
    (web_source,      "Web crawl"),
    (books_source,    "Books"),
    (code_source,     "Code"),
    (academic_source, "Academic"),
]:
    print(f"  {label}: weight={src.weight}, filters={src.filters}")

# =============================================================================
# Mixing Strategies
# =============================================================================

print("\n" + "=" * 60)
print("MIXING STRATEGIES")
print("=" * 60)

sources = [web_source, books_source, code_source, academic_source]
source_labels = ["Web", "Books", "Code", "Academic"]

# --- Weighted mixing (standard) ---
weighted_mix = DatasetMix(
    name="pretraining-v3-weighted",
    sources=sources,
    strategy=MixingStrategy.WEIGHTED,
    total_samples=2_000_000_000,
    seed=42,
    description="Standard weighted mix for pretraining v3",
)

print("\nWeighted mix — effective sampling probabilities:")
for label, prob in zip(source_labels, weighted_mix.effective_weights()):
    print(f"  {label}: {prob:.1%}")

# --- Temperature mixing (T < 1 amplifies dominant source, T > 1 smooths) ---
temperature_mix = DatasetMix(
    name="pretraining-v3-temperature",
    sources=sources,
    strategy=MixingStrategy.TEMPERATURE,
    temperature=0.7,    # cool → amplifies high-weight sources
    total_samples=2_000_000_000,
    seed=42,
)

print("\nTemperature mix (T=0.7) — effective sampling probabilities:")
for label, prob in zip(source_labels, temperature_mix.effective_weights()):
    print(f"  {label}: {prob:.1%}")

# --- Uniform mixing (ignore weights — useful for ablation studies) ---
uniform_mix = DatasetMix(
    name="pretraining-v3-uniform",
    sources=sources,
    strategy=MixingStrategy.UNIFORM,
    total_samples=500_000_000,
    seed=42,
)

print("\nUniform mix — effective sampling probabilities:")
for label, prob in zip(source_labels, uniform_mix.effective_weights()):
    print(f"  {label}: {prob:.1%}")

# --- DatasetMix summary ---
print(f"\nMix summary for {weighted_mix.name}:")
summary = weighted_mix.summary()
for src in summary["sources"]:
    print(f"  {src['feature_group'].split('/')[-2]}: "
          f"weight={src['weight']}, effective={src['effective_prob']}")

# =============================================================================
# Multi-Lingual Mix
# =============================================================================

print("\n" + "=" * 60)
print("MULTI-LINGUAL MIX")
print("=" * 60)

# Language-specific sources with adjusted weights
en_source = MixSource(
    feature_group="acme/pretraining/web-crawl/curated-documents",
    weight=0.45,
    filters=["include_in_training == True", "language == 'en'"],
    tags=["english"],
)
zh_source = MixSource(
    feature_group="acme/pretraining/web-crawl/curated-documents",
    weight=0.20,
    filters=["include_in_training == True", "language == 'zh'"],
    tags=["chinese"],
)
es_source = MixSource(
    feature_group="acme/pretraining/web-crawl/curated-documents",
    weight=0.15,
    filters=["include_in_training == True", "language == 'es'"],
    tags=["spanish"],
)
multilingual_other = MixSource(
    feature_group="acme/pretraining/web-crawl/curated-documents",
    weight=0.20,
    filters=["include_in_training == True", "language NOT IN ('en', 'zh', 'es')"],
    tags=["other-languages"],
)

multilingual_mix = DatasetMix(
    name="multilingual-v1",
    sources=[en_source, zh_source, es_source, multilingual_other],
    strategy=MixingStrategy.WEIGHTED,
    total_samples=1_000_000_000,
    description="Multi-lingual mix across top 3 + other languages",
)

print("Multi-lingual mix:")
labels = ["English", "Chinese", "Spanish", "Other"]
for label, prob in zip(labels, multilingual_mix.effective_weights()):
    print(f"  {label}: {prob:.1%}")

# =============================================================================
# Dataset Versioning
# =============================================================================

print("\n" + "=" * 60)
print("DATASET VERSIONING")
print("=" * 60)

# Initial version of the training dataset
v1 = DatasetVersion(
    name="foundation-pretraining",
    version="v1.0.0",
    feature_group="acme/pretraining/foundation/training-mix",
    num_records=500_000_000,
    size_bytes=2_000 * 1024**3,  # 2 TB
    applied_filters=["quality_score >= 0.5"],
    created_by="ml-platform@acme.com",
    tags=["pretraining", "english-only"],
)

print(f"Version: {v1}")
print(f"  Records: {v1.num_records:,}")
print(f"  Size: {v1.size_gb:.0f} GB")
print(f"  Is initial: {v1.is_initial}")

# Derive v2 with tighter quality filters and dedup
v2 = v1.derive(
    version="v2.0.0",
    applied_filters=[
        "quality_score >= 0.65",    # higher quality bar
        "is_duplicate == False",    # deduped
        "compliance_passed == True",
    ],
    num_records=380_000_000,
    size_bytes=1_520 * 1024**3,
)

print(f"\nDerived version: {v2}")
print(f"  Parent: {v2.parent_version}")
print(f"  Records: {v2.num_records:,} (↓ {(1 - v2.num_records/v1.num_records):.0%} from v1)")
print(f"  Filters: {v2.applied_filters}")

# Derive v3 with multi-lingual additions
v3 = v2.derive(
    version="v3.0.0",
    applied_filters=[
        "quality_score >= 0.65",
        "is_duplicate == False",
        "compliance_passed == True",
        # Multi-lingual: no language filter
    ],
    num_records=2_000_000_000,
    size_bytes=8_000 * 1024**3,
    metadata={"mix_strategy": "temperature", "temperature": 0.7},
)

print(f"\nMulti-lingual version: {v3}")
print(f"  Parent: {v3.parent_version}")

# =============================================================================
# Dataset Provenance
# =============================================================================

print("\n" + "=" * 60)
print("DATASET PROVENANCE")
print("=" * 60)

web_provenance = DatasetProvenance(
    source_name="CommonCrawl-2024-Q1",
    source_uri="s3://commoncrawl/crawl-data/CC-MAIN-2024-10/",
    collection_date=datetime(2024, 3, 1),
    license="CC0",
    language_codes=["en", "zh", "es", "fr", "de", "ja", "ko", "pt", "ru", "ar"],
    domain_tags=["web", "news", "forum", "blog"],
    pii_reviewed=True,
    consent_documented=False,  # Public web crawl
    metadata={"crawl_pages": 3_800_000_000, "wet_files": 56000},
)

books_provenance = DatasetProvenance(
    source_name="Books3-Licensed",
    source_uri="s3://acme-datasets/books3-licensed/",
    collection_date=datetime(2023, 6, 15),
    license="Commercial license — see legal/books3-license.pdf",
    citation="Books3: A Large Corpus of Books for Language Modeling",
    language_codes=["en"],
    domain_tags=["books", "fiction", "non-fiction", "academic"],
    pii_reviewed=True,
    consent_documented=True,
)

print(f"Web provenance:")
print(f"  Source: {web_provenance.source_name}")
print(f"  License: {web_provenance.license}")
print(f"  Languages: {web_provenance.language_codes[:5]}... ({len(web_provenance.language_codes)} total)")
print(f"  PII reviewed: {web_provenance.pii_reviewed}")

print(f"\nBooks provenance:")
print(f"  Source: {books_provenance.source_name}")
print(f"  License: {books_provenance.license}")
print(f"  Consent documented: {books_provenance.consent_documented}")

# Attach provenance to a dataset version
v3_with_provenance = DatasetVersion(
    name="foundation-pretraining",
    version="v3.0.0",
    feature_group="acme/pretraining/foundation/training-mix",
    num_records=2_000_000_000,
    size_bytes=8_000 * 1024**3,
    applied_filters=v3.applied_filters,
    parent_version="v2.0.0",
    provenance=web_provenance,   # primary provenance; others tracked via MixSource
    tags=["pretraining", "multilingual"],
)

print(f"\nVersioned dataset with provenance: {v3_with_provenance}")

# =============================================================================
# Executing a Mix as a Job
# =============================================================================

print("\n" + "=" * 60)
print("MIX EXECUTION JOB")
print("=" * 60)

# The mixing job reads from each MixSource and samples according to weights.
# In practice this runs as a distributed SQL/Spark job.
mix_job = fs.create_job(
    name="pretraining_mix_v3",
    description="Build the v3 pre-training mix",
    sources=[
        FeatureGroupSource(
            feature_group=src.feature_group,
            features=["doc_id", "text", "language", "token_count"],
            filters=src.filters,
        )
        for src in weighted_mix.sources
    ],
    transform=SQLTransform(
        sql="""
        -- Sampling and tagging by source; actual mixing weights
        -- are passed as job parameters from DatasetMix.effective_weights()
        SELECT
            CONCAT(source_domain, '_', doc_id) AS example_id,
            text,
            source_domain,
            :source_weight AS source_weight,
            language,
            token_count
        FROM source
        WHERE RAND() < :sample_rate   -- reservoir sampling per source
        """,
        params={
            "source_weight": weighted_mix.normalized_weights()[0],  # per-source in real execution
            "sample_rate": 0.001,     # placeholder; real rate set per source
        },
    ),
    target=Target(
        feature_group="training-mix",
        features={"example_id", "text", "source_domain", "source_weight", "language", "token_count"},
        write_mode="overwrite",
    ),
    schedule=Schedule.manual(),   # Triggered explicitly for each version
    tags=["mixing", "pretraining"],
)

print(f"Mix job: {mix_job.name}")
print(f"  Sources: {len(mix_job.sources)}")
print(f"  Strategy: {weighted_mix.strategy.value}")
print(f"  Target records: {weighted_mix.total_samples:,}")
print(f"  Schedule: manual (triggered per version bump)")

print("\n" + "=" * 60)
print("ALL DATASET MIXING EXAMPLES COMPLETE!")
print("=" * 60)
