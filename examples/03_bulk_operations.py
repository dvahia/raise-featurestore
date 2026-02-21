"""
Example 3: Bulk Feature Operations

Demonstrates bulk creation of features for pre-training pipelines
and production feature engineering workflows.
"""

from raise_ import FeatureStore

# =============================================================================
# Setup
# =============================================================================

fs = FeatureStore("acme/mlplatform/pretraining")

# Create feature group for bulk operations
embeddings = fs.create_feature_group(
    "embeddings",
    description="Pre-computed embeddings for various entities",
    tags=["embeddings", "pretraining"],
    if_exists="skip",
)

# =============================================================================
# Bulk Creation from Schema Dict
# =============================================================================

print("=" * 60)
print("BULK CREATION FROM SCHEMA")
print("=" * 60)

# Simple name-to-type mapping for quick bulk creation
features = embeddings.create_features_from_schema({
    "user_embedding_v1": "float32[256]",
    "user_embedding_v2": "float32[512]",
    "item_embedding": "float32[512]",
    "query_embedding": "float32[768]",
    "context_embedding": "float32[256]",
    "session_embedding": "float32[128]",
}, if_exists="skip")

print(f"Created {len(features)} features from schema:")
for f in features:
    print(f"  - {f.name}: {f.dtype}")

# =============================================================================
# Bulk Creation from List of Dicts
# =============================================================================

print("\n" + "=" * 60)
print("BULK CREATION FROM LIST")
print("=" * 60)

# Create feature group for user features
user_features = fs.create_feature_group(
    "user-features",
    description="User profile and behavioral features",
    if_exists="skip",
)

# More detailed specifications as list of dicts
feature_specs = [
    {
        "name": "age",
        "dtype": "int64",
        "description": "User age in years",
        "nullable": False,
    },
    {
        "name": "gender",
        "dtype": "string",
        "description": "User gender",
        "tags": ["demographic"],
    },
    {
        "name": "location_lat",
        "dtype": "float64",
        "description": "User location latitude",
    },
    {
        "name": "location_lon",
        "dtype": "float64",
        "description": "User location longitude",
    },
    {
        "name": "account_age_days",
        "dtype": "int64",
        "description": "Days since account creation",
    },
    {
        "name": "total_purchases",
        "dtype": "int64",
        "description": "Lifetime purchase count",
        "default": 0,
    },
    {
        "name": "total_revenue",
        "dtype": "float64",
        "description": "Lifetime revenue",
        "default": 0.0,
    },
    {
        "name": "avg_session_duration",
        "dtype": "float64",
        "description": "Average session duration in seconds",
    },
    {
        "name": "preferred_categories",
        "dtype": "string[]",
        "description": "List of preferred product categories",
    },
    # Derived features can be included in bulk operations
    {
        "name": "revenue_per_purchase",
        "dtype": "float64",
        "derived_from": "total_revenue / NULLIF(total_purchases, 0)",
        "description": "Average revenue per purchase",
        "tags": ["derived"],
    },
    {
        "name": "user_value_tier",
        "dtype": "string",
        "derived_from": """
            CASE
                WHEN total_revenue > 5000 THEN 'high'
                WHEN total_revenue > 500 THEN 'medium'
                ELSE 'low'
            END
        """,
        "description": "User value tier",
        "tags": ["derived", "segmentation"],
    },
]

features = user_features.create_features(feature_specs, if_exists="skip")

print(f"Created {len(features)} features from list:")
for f in features:
    marker = " (derived)" if f.is_derived else ""
    print(f"  - {f.name}: {f.dtype}{marker}")

# =============================================================================
# Bulk Creation from YAML File
# =============================================================================

print("\n" + "=" * 60)
print("BULK CREATION FROM YAML FILE")
print("=" * 60)

# First, let's create a sample YAML file
yaml_content = """
features:
  - name: click_count
    dtype: int64
    description: Total clicks by user
    tags:
      - engagement
      - core

  - name: impression_count
    dtype: int64
    description: Total impressions shown
    tags:
      - engagement
      - core

  - name: ctr
    dtype: float64
    derived_from: click_count / NULLIF(impression_count, 0)
    description: Click-through rate
    tags:
      - derived
      - ratio

  - name: session_count
    dtype: int64
    description: Total sessions

  - name: pages_per_session
    dtype: float64
    description: Average pages viewed per session

  - name: bounce_rate
    dtype: float64
    description: Session bounce rate

  - name: engagement_score
    dtype: float64
    derived_from: (ctr * 0.4 + (1 - bounce_rate) * 0.3 + LOG(session_count + 1) * 0.3)
    description: Composite engagement score
    tags:
      - derived
      - composite
"""

# Write the YAML file
import os
yaml_path = "/tmp/features.yaml"
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"Created sample YAML file at {yaml_path}")

# Create feature group and load from YAML
engagement = fs.create_feature_group(
    "engagement",
    description="User engagement metrics",
    if_exists="skip",
)

features = engagement.create_features_from_file(yaml_path, if_exists="skip")

print(f"\nCreated {len(features)} features from YAML:")
for f in features:
    marker = " (derived)" if f.is_derived else ""
    print(f"  - {f.name}: {f.dtype}{marker}")

# Clean up
os.remove(yaml_path)

# =============================================================================
# Listing and Filtering Features
# =============================================================================

print("\n" + "=" * 60)
print("LISTING AND FILTERING")
print("=" * 60)

# List all features in a group
all_features = user_features.list_features()
print(f"\nAll features in user-features: {len(all_features)}")

# Filter by tags
derived_features = user_features.list_features(tags=["derived"])
print(f"Derived features: {len(derived_features)}")
for f in derived_features:
    print(f"  - {f.name}")

# Search across project
print("\nSearching for 'embedding' features across project:")
results = fs.search_features(query="embedding", limit=10)
for f in results:
    print(f"  - {f.qualified_name}")

# Search by dtype pattern
print("\nSearching for float32 embeddings of any dimension:")
results = fs.search_features(dtype="float32[*]", limit=10)
for f in results:
    print(f"  - {f.name}: {f.dtype}")

# =============================================================================
# Versioning in Bulk
# =============================================================================

print("\n" + "=" * 60)
print("FEATURE VERSIONING")
print("=" * 60)

# Create a new version of an embedding
new_embedding = embeddings.create_feature(
    "user_embedding_v1",
    dtype="float32[384]",  # Changed dimension
    description="Updated user embedding with new architecture",
    version="v2",
    if_exists="skip",
)

print(f"Created new version: {new_embedding.qualified_name}")

# List all versions
original = embeddings.feature("user_embedding_v1")
versions = original.list_versions()
print(f"\nAll versions of user_embedding_v1:")
for v in versions:
    print(f"  - {v.version}: {v.dtype}")

# Access specific version
v1 = embeddings.feature("user_embedding_v1@v1")
print(f"\nAccessed specific version: {v1.qualified_name}")
