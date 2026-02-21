"""
Example 4: Cross-Organization Access

Demonstrates sharing features across organizations and
managing external access grants.
"""

from datetime import datetime, timedelta
from raise_ import FeatureStore, ACL

# =============================================================================
# Setup: Two Organizations
# =============================================================================

print("=" * 60)
print("SETTING UP TWO ORGANIZATIONS")
print("=" * 60)

# Organization 1: Data provider (has the features)
provider_fs = FeatureStore("data-provider/shared/embeddings")
provider_group = provider_fs.create_feature_group(
    "product-embeddings",
    description="Shared product embeddings for partner integrations",
    if_exists="skip",
)

# Create features to share
provider_group.create_features_from_schema({
    "product_embedding": "float32[512]",
    "category_embedding": "float32[128]",
    "brand_embedding": "float32[64]",
}, if_exists="skip")

print("Provider org features:")
for f in provider_group.list_features():
    print(f"  - {f.qualified_name}")

# Organization 2: Data consumer (wants to use the features)
consumer_fs = FeatureStore("partner-corp/mlplatform/recommendation")
consumer_group = consumer_fs.create_feature_group(
    "user-signals",
    description="User signals for recommendation",
    if_exists="skip",
)

consumer_group.create_feature(
    "user_embedding",
    dtype="float32[512]",
    description="User preference embedding",
    if_exists="skip",
)

print("\nConsumer org features:")
for f in consumer_group.list_features():
    print(f"  - {f.qualified_name}")

# =============================================================================
# Granting Cross-Org Access
# =============================================================================

print("\n" + "=" * 60)
print("GRANTING CROSS-ORG ACCESS")
print("=" * 60)

# Provider grants access to consumer (owner-only approval)
grant = provider_group.grant_external_access(
    org="partner-corp",
    features=["product_embedding", "category_embedding"],  # Specific features
    permission="read",
    expires_at=datetime.now() + timedelta(days=365),  # 1 year expiration
)

print(f"Created grant:")
print(f"  Org: {grant.org}")
print(f"  Features: {grant.features}")
print(f"  Permission: {grant.permission}")
print(f"  Expires: {grant.expires_at}")

# Grant all features with wildcard
grant_all = provider_group.grant_external_access(
    org="another-partner",
    features=["*"],  # All features in this group
    permission="read",
)

print(f"\nCreated wildcard grant for another-partner")

# =============================================================================
# Using Cross-Org Features
# =============================================================================

print("\n" + "=" * 60)
print("USING CROSS-ORG FEATURES")
print("=" * 60)

# Consumer creates derived features using provider's embeddings
# Uses @ prefix for absolute cross-org reference

# First, validate the cross-org expression
result = consumer_group.validate_feature(
    "user_product_affinity",
    dtype="float32",
    derived_from="DOT(user_embedding, @data-provider/shared/embeddings/product-embeddings.product_embedding)",
)

print(f"Validation result:")
print(f"  Valid: {result.valid}")
print(f"  Cross-org references: {[r.qualified_name for r in result.cross_org_references]}")

if result.valid:
    # Create the cross-org derived feature
    affinity = consumer_group.create_feature(
        "user_product_affinity",
        dtype="float32",
        derived_from="DOT(user_embedding, @data-provider/shared/embeddings/product-embeddings.product_embedding)",
        description="User-product affinity using partner's product embeddings",
        tags=["derived", "cross-org"],
        if_exists="skip",
    )
    print(f"\nCreated cross-org derived feature: {affinity.name}")

    # Create similarity feature
    similarity = consumer_group.create_feature(
        "user_product_similarity",
        dtype="float32",
        derived_from="COSINE_SIMILARITY(user_embedding, @data-provider/shared/embeddings/product-embeddings.product_embedding)",
        description="Cosine similarity with partner's product embeddings",
        if_exists="skip",
    )

# =============================================================================
# Viewing Lineage Across Organizations
# =============================================================================

print("\n" + "=" * 60)
print("CROSS-ORG LINEAGE")
print("=" * 60)

# Get lineage including external references
affinity = consumer_group.feature("user_product_affinity")
lineage = affinity.get_lineage()

print(f"Lineage for {affinity.name}:")
print(f"  Upstream features:")
for f in lineage.upstream:
    org_marker = " [EXTERNAL]" if f.org != "partner-corp" else ""
    print(f"    - {f.qualified_name}{org_marker}")

# Show all upstream including transitive
print(f"\n  All upstream (transitive, including external):")
for f in lineage.all_upstream(include_external=True):
    org_marker = " [EXTERNAL]" if f.org != "partner-corp" else ""
    print(f"    - {f.qualified_name}{org_marker}")

# =============================================================================
# Managing Grants
# =============================================================================

print("\n" + "=" * 60)
print("MANAGING GRANTS")
print("=" * 60)

# List all external grants
grants = provider_group.list_external_grants()
print(f"Active grants for product-embeddings:")
for g in grants:
    print(f"  - Org: {g.org}")
    print(f"    Features: {g.features}")
    print(f"    Permission: {g.permission}")
    print(f"    Expires: {g.expires_at or 'never'}")
    print()

# Revoke access
revoked = provider_group.revoke_external_access("another-partner")
print(f"Revoked access for another-partner: {revoked}")

# List grants again
grants = provider_group.list_external_grants()
print(f"\nRemaining grants: {len(grants)}")

# =============================================================================
# ACL Cascading
# =============================================================================

print("\n" + "=" * 60)
print("ACL CASCADING")
print("=" * 60)

# Set ACL at organization level
provider_org = provider_fs.organization("data-provider")
provider_org.set_acl(ACL(
    readers=["public@data-provider.com"],
    writers=["data-team@data-provider.com"],
    admins=["data-leads@data-provider.com"],
))

# Set additional ACL at group level (additive, inherits from parent)
provider_group.set_acl(ACL(
    readers=["partner-integrations@data-provider.com"],
    writers=["embedding-team@data-provider.com"],
))

# Get effective ACL (combined from all levels)
product_embedding = provider_group.feature("product_embedding")
effective_acl = product_embedding.get_effective_acl()

print(f"Effective ACL for {product_embedding.name}:")
print(f"  Readers: {effective_acl.readers}")
print(f"  Writers: {effective_acl.writers}")
print(f"  Admins: {effective_acl.admins}")

# View ACL chain
print(f"\nACL inheritance chain:")
chain = product_embedding.get_acl_chain()
for i, acl in enumerate(chain):
    level = ["Organization", "Domain", "Project", "FeatureGroup", "Feature"][i] if i < 5 else f"Level {i}"
    print(f"  {level}:")
    print(f"    Readers: {acl.readers}")
    print(f"    Inherit: {acl.inherit}")

# =============================================================================
# Feature Discovery Across Orgs
# =============================================================================

print("\n" + "=" * 60)
print("FEATURE DISCOVERY")
print("=" * 60)

# Search for features including those you have external access to
# Note: This would typically show features from orgs that have granted you access
results = consumer_fs.search_features(
    query="embedding",
    include_external=True,
    limit=20,
)

print("Embedding features (including external):")
for f in results:
    print(f"  - {f.qualified_name}")
