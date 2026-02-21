"""
Example 1: Basic Feature Creation

Demonstrates simple feature creation in the Raise Feature Store.
This is the most common use case for AI researchers in notebooks.
"""

from raise_ import FeatureStore

# =============================================================================
# Connect to the Feature Store
# =============================================================================

# Connect with a path string (org/domain/project)
fs = FeatureStore("acme/mlplatform/recommendation")

# Alternative: explicit parameters
# fs = FeatureStore(org="acme", domain="mlplatform", project="recommendation")

print(f"Connected: {fs}")

# =============================================================================
# Create a Feature Group
# =============================================================================

# Feature groups organize related features
user_signals = fs.create_feature_group(
    "user-signals",
    description="User behavioral signals for recommendation models",
    tags=["user", "behavioral", "ranking"],
)

print(f"Created: {user_signals}")

# =============================================================================
# Create Simple Features
# =============================================================================

# Create a single feature with minimal parameters
clicks = user_signals.create_feature(
    "click_count",
    dtype="int64",
)
print(f"Created: {clicks}")

# Create a feature with more metadata
impressions = user_signals.create_feature(
    "impression_count",
    dtype="int64",
    description="Total ad impressions shown to user",
    tags=["engagement", "ads"],
    nullable=False,
    default=0,
)
print(f"Created: {impressions}")

# Create an embedding feature
embedding = user_signals.create_feature(
    "user_embedding",
    dtype="float32[512]",  # 512-dimensional float32 embedding
    description="User profile embedding from transformer model v2",
    tags=["embedding", "transformer", "prod"],
)
print(f"Created: {embedding}")

# Create a timestamp feature
last_active = user_signals.create_feature(
    "last_active_ts",
    dtype="timestamp",
    description="Last time the user was active",
)
print(f"Created: {last_active}")

# =============================================================================
# Retrieve Features
# =============================================================================

# Get a feature by name
retrieved = user_signals.feature("click_count")
print(f"\nRetrieved feature: {retrieved.name}")
print(f"  Type: {retrieved.dtype}")
print(f"  Qualified name: {retrieved.qualified_name}")

# List all features in the group
print("\nAll features in user-signals:")
for feature in user_signals.list_features():
    print(f"  - {feature.name}: {feature.dtype}")

# =============================================================================
# Idempotent Creation
# =============================================================================

# Using if_exists="skip" allows safe re-runs (idempotent)
# This won't raise an error if the feature already exists
same_clicks = user_signals.create_feature(
    "click_count",
    dtype="int64",
    if_exists="skip",  # Returns existing feature instead of error
)
print(f"\nIdempotent creation returned: {same_clicks.qualified_name}")

# Alternative: get_or_create_feature
another = user_signals.get_or_create_feature(
    "session_count",
    dtype="int64",
    description="Number of sessions",
)
print(f"Get or create returned: {another.name}")

# =============================================================================
# Using Path Syntax
# =============================================================================

# Create feature using path syntax (group/feature)
score = fs.create_feature(
    "user-signals/relevance_score",
    dtype="float64",
    description="ML model relevance score",
)
print(f"\nCreated via path: {score.qualified_name}")

# Retrieve using path syntax
retrieved_score = fs.feature("user-signals/relevance_score")
print(f"Retrieved via path: {retrieved_score.name}")
