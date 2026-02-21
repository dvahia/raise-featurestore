"""
Example 2: Derived Features with SQL Expressions

Demonstrates creating derived features that are computed from other features.
Uses SQL-like syntax for expressions with automatic lineage tracking.
"""

from raise_ import FeatureStore

# =============================================================================
# Setup
# =============================================================================

fs = FeatureStore("acme/mlplatform/recommendation")

# Create feature group with base features
user_signals = fs.create_feature_group(
    "user-signals",
    description="User engagement signals",
    if_exists="skip",
)

# Create base features first
user_signals.create_features_from_schema({
    "click_count": "int64",
    "impression_count": "int64",
    "purchase_count": "int64",
    "revenue": "float64",
    "days_since_signup": "int64",
    "user_embedding": "float32[512]",
}, if_exists="skip")

print("Base features created")

# =============================================================================
# Simple Derived Features
# =============================================================================

# Click-through rate with division-by-zero protection
ctr = user_signals.create_feature(
    "click_through_rate",
    dtype="float64",
    derived_from="click_count / NULLIF(impression_count, 0)",
    description="Click-through rate (protected against division by zero)",
    tags=["derived", "ratio", "engagement"],
    if_exists="skip",
)
print(f"\nCreated derived feature: {ctr.name}")
print(f"  Expression: {ctr.derived_from}")

# Conversion rate
conversion_rate = user_signals.create_feature(
    "conversion_rate",
    dtype="float64",
    derived_from="purchase_count / NULLIF(click_count, 0)",
    description="Purchase conversion rate from clicks",
    if_exists="skip",
)

# Average order value
aov = user_signals.create_feature(
    "avg_order_value",
    dtype="float64",
    derived_from="revenue / NULLIF(purchase_count, 0)",
    description="Average order value",
    if_exists="skip",
)

# =============================================================================
# Using SQL Functions
# =============================================================================

# Normalized revenue using statistical functions
normalized_revenue = user_signals.create_feature(
    "normalized_revenue",
    dtype="float64",
    derived_from="(revenue - AVG(revenue)) / NULLIF(STDDEV(revenue), 0)",
    description="Z-score normalized revenue",
    if_exists="skip",
)

# Log-transformed days since signup
log_tenure = user_signals.create_feature(
    "log_tenure",
    dtype="float64",
    derived_from="LOG(days_since_signup + 1)",
    description="Log-transformed account age",
    if_exists="skip",
)

# Engagement score combining multiple signals
engagement_score = user_signals.create_feature(
    "engagement_score",
    dtype="float64",
    derived_from="(click_through_rate * 0.3 + conversion_rate * 0.5 + LOG(purchase_count + 1) * 0.2)",
    description="Composite engagement score",
    if_exists="skip",
)

print(f"Created engagement_score: {engagement_score.derived_from}")

# =============================================================================
# Conditional Expressions (CASE WHEN)
# =============================================================================

# User tier based on revenue
user_tier = user_signals.create_feature(
    "user_tier",
    dtype="string",
    derived_from="""
        CASE
            WHEN revenue > 10000 THEN 'platinum'
            WHEN revenue > 1000 THEN 'gold'
            WHEN revenue > 100 THEN 'silver'
            ELSE 'bronze'
        END
    """,
    description="User tier based on lifetime revenue",
    tags=["derived", "segmentation"],
    if_exists="skip",
)
print(f"\nCreated conditional feature: {user_tier.name}")

# Risk flag
is_high_value = user_signals.create_feature(
    "is_high_value",
    dtype="bool",
    derived_from="CASE WHEN revenue > 1000 AND purchase_count > 5 THEN TRUE ELSE FALSE END",
    description="Flag for high-value customers",
    if_exists="skip",
)

# =============================================================================
# Cross-Group References
# =============================================================================

# Create another feature group
item_signals = fs.create_feature_group(
    "item-signals",
    description="Item/product features",
    if_exists="skip",
)

item_signals.create_feature(
    "item_embedding",
    dtype="float32[512]",
    description="Item embedding from product model",
    if_exists="skip",
)

# Create a feature that references another group
user_item_affinity = user_signals.create_feature(
    "user_item_affinity",
    dtype="float32",
    derived_from="DOT(user_embedding, item-signals.item_embedding)",
    description="User-item affinity score via embedding dot product",
    if_exists="skip",
)
print(f"\nCross-group feature: {user_item_affinity.name}")
print(f"  Expression: {user_item_affinity.derived_from}")

# Cosine similarity variant
user_item_similarity = user_signals.create_feature(
    "user_item_similarity",
    dtype="float32",
    derived_from="COSINE_SIMILARITY(user_embedding, item-signals.item_embedding)",
    description="User-item similarity via cosine distance",
    if_exists="skip",
)

# =============================================================================
# Lineage Tracking
# =============================================================================

print("\n" + "=" * 60)
print("LINEAGE TRACKING")
print("=" * 60)

# Get lineage for a derived feature
lineage = ctr.get_lineage()
print(f"\nLineage for {ctr.name}:")
print(f"  Upstream (dependencies): {[f.name for f in lineage.upstream]}")
print(f"  Downstream (dependents): {[f.name for f in lineage.downstream]}")

# Get lineage for engagement_score (multi-level)
es_lineage = engagement_score.get_lineage()
print(f"\nLineage for {engagement_score.name}:")
print(f"  Direct upstream: {[f.name for f in es_lineage.upstream]}")
print(f"  All upstream (transitive): {[f.name for f in es_lineage.all_upstream()]}")

# Visualize as ASCII graph
print(f"\nLineage graph for {ctr.name}:")
print(ctr.get_lineage().as_graph().to_ascii())

# =============================================================================
# Validation
# =============================================================================

print("\n" + "=" * 60)
print("EXPRESSION VALIDATION")
print("=" * 60)

# Validate an expression before creating
result = user_signals.validate_feature(
    "test_ratio",
    dtype="float64",
    derived_from="click_count / impression_count",
)

print(f"\nValidation result:")
print(f"  Valid: {result.valid}")
print(f"  Warnings: {[w.message for w in result.warnings]}")
print(f"  Inferred type: {result.inferred_type}")
print(f"  References: {[r.name for r in result.references]}")

# Validate with an unknown reference
bad_result = user_signals.validate_feature(
    "bad_feature",
    dtype="float64",
    derived_from="nonexistent_column * 2",
)

print(f"\nInvalid expression validation:")
print(f"  Valid: {bad_result.valid}")
if bad_result.errors:
    for error in bad_result.errors:
        print(f"  Error: {error.code} - {error.message}")
        if error.suggestion:
            print(f"    Suggestion: {error.suggestion}")
