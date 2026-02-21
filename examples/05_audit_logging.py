"""
Example 5: Audit Logging

Demonstrates querying audit logs, setting up alerts,
and configuring audit retention for compliance.
"""

from datetime import datetime, timedelta
from raise_ import FeatureStore, ACL
from raise_.models.audit import AuditQuery

# =============================================================================
# Setup
# =============================================================================

fs = FeatureStore("acme/mlplatform/recommendation")

# Create some features to generate audit events
user_signals = fs.create_feature_group(
    "user-signals",
    description="User behavioral signals",
    if_exists="skip",
)

user_signals.create_features_from_schema({
    "click_count": "int64",
    "impression_count": "int64",
    "user_embedding": "float32[512]",
    "revenue": "float64",
}, if_exists="skip")

# Create a derived feature
user_signals.create_feature(
    "ctr",
    dtype="float64",
    derived_from="click_count / NULLIF(impression_count, 0)",
    if_exists="skip",
)

# Modify ACL to generate audit events
user_signals.set_acl(ACL(
    readers=["ml-engineers@acme.com"],
    writers=["ml-team@acme.com"],
))

print("Setup complete - features created")

# =============================================================================
# Querying Audit Logs
# =============================================================================

print("\n" + "=" * 60)
print("QUERYING AUDIT LOGS")
print("=" * 60)

# Query recent logs for this feature group
logs = fs.audit.query(
    resource="user-signals/*",
    since=datetime.now() - timedelta(days=7),
    limit=100,
)

print(f"Found {logs.total_count} audit entries for user-signals")
print("\nRecent entries:")
for entry in logs.entries[:5]:
    print(f"  {entry.timestamp} | {entry.actor} | {entry.action} | {entry.resource}")

# =============================================================================
# Filtering by Action Type
# =============================================================================

print("\n" + "=" * 60)
print("FILTERING BY ACTION TYPE")
print("=" * 60)

# Get all schema changes
schema_changes = fs.audit.query(
    resource="user-signals/*",
    actions=["CREATE", "UPDATE_SCHEMA", "DELETE"],
    since=datetime.now() - timedelta(days=30),
)

print(f"Schema changes in last 30 days: {schema_changes.total_count}")
for entry in schema_changes.entries:
    print(f"  {entry.timestamp} | {entry.action} | {entry.resource}")
    if entry.previous_state and entry.new_state:
        print(f"    Changed: {entry.previous_state} -> {entry.new_state}")

# Get all ACL changes
acl_changes = fs.audit.query(
    category="acl",
    since=datetime.now() - timedelta(days=30),
)

print(f"\nACL changes in last 30 days: {acl_changes.total_count}")

# =============================================================================
# Tracking External Access
# =============================================================================

print("\n" + "=" * 60)
print("TRACKING EXTERNAL ACCESS")
print("=" * 60)

# Find all access from external organizations
external_access = fs.audit.query(
    resource="user-signals/*",
    actions=["READ", "WRITE"],
    exclude_actor_orgs=["acme"],  # Exclude our own org
    since=datetime.now() - timedelta(days=7),
)

print(f"External access in last 7 days: {external_access.total_count}")

# Group by organization
by_org: dict = {}
for entry in external_access.entries:
    org = entry.actor_org
    if org not in by_org:
        by_org[org] = []
    by_org[org].append(entry)

for org, entries in by_org.items():
    print(f"\n  {org}: {len(entries)} accesses")
    for e in entries[:3]:
        print(f"    - {e.actor} | {e.action} | {e.resource}")

# =============================================================================
# Feature-Level Audit Logs
# =============================================================================

print("\n" + "=" * 60)
print("FEATURE-LEVEL AUDIT")
print("=" * 60)

# Get audit logs for a specific feature
embedding = user_signals.feature("user_embedding")
feature_logs = embedding.audit_log(
    since=datetime.now() - timedelta(days=30),
    limit=50,
)

print(f"Audit log for {embedding.name}:")
print(f"  Total entries: {feature_logs.total_count}")

# Get schema history
schema_history = embedding.audit_log(
    category="schema",
)

print(f"\n  Schema history:")
for entry in schema_history.entries:
    print(f"    {entry.timestamp} | {entry.action} by {entry.actor}")

# Get access history
access_history = embedding.audit_log(
    actions=["READ", "WRITE"],
)

print(f"\n  Access history:")
unique_actors = set(e.actor for e in access_history.entries)
print(f"    Unique accessors: {len(unique_actors)}")

# =============================================================================
# Setting Up Alerts
# =============================================================================

print("\n" + "=" * 60)
print("AUDIT ALERTS")
print("=" * 60)

# Alert on external access
external_alert = fs.audit.create_alert(
    name="external-feature-access",
    query=AuditQuery(
        resource="user-signals/*",
        actions=["READ", "WRITE"],
        exclude_actor_orgs=["acme"],
    ),
    notify=["security@acme.com", "ml-platform@acme.com"],
    channels=["email", "slack"],
)

print(f"Created alert: {external_alert.name}")
print(f"  Notify: {external_alert.notify}")
print(f"  Channels: {external_alert.channels}")

# Alert on sensitive feature access
sensitive_alert = fs.audit.create_alert(
    name="sensitive-data-access",
    query=AuditQuery(
        resource="*revenue*",  # Any feature with 'revenue' in the name
        actions=["READ"],
    ),
    notify=["compliance@acme.com"],
    channels=["email"],
)

print(f"\nCreated alert: {sensitive_alert.name}")

# Alert on ACL changes
acl_alert = fs.audit.create_alert(
    name="acl-modifications",
    query=AuditQuery(
        category="acl",
        actions=["UPDATE_ACL", "GRANT_EXTERNAL", "REVOKE_EXTERNAL"],
    ),
    notify=["security@acme.com"],
    channels=["slack"],
)

print(f"Created alert: {acl_alert.name}")

# List all alerts
alerts = fs.audit.list_alerts()
print(f"\nAll configured alerts: {len(alerts)}")
for alert in alerts:
    print(f"  - {alert.name} (enabled: {alert.enabled})")

# Delete an alert
fs.audit.delete_alert("sensitive-data-access")
print("\nDeleted 'sensitive-data-access' alert")

# =============================================================================
# Exporting Audit Logs
# =============================================================================

print("\n" + "=" * 60)
print("EXPORTING AUDIT LOGS")
print("=" * 60)

# Export to file for compliance
export_path = fs.audit.export(
    query=AuditQuery(
        since=datetime(2025, 1, 1),
        until=datetime(2025, 12, 31),
    ),
    format="jsonl",
    destination="/tmp/audit_export_2025.jsonl",
)

print(f"Exported audit logs to: {export_path}")

# Export to cloud storage (S3 example)
# export_path = fs.audit.export(
#     query=AuditQuery(since=datetime(2025, 1, 1)),
#     format="parquet",
#     destination="s3://acme-compliance/audit-logs/2025/",
# )

# Streaming export for large datasets
print("\nStreaming export example:")
with fs.audit.stream(AuditQuery(since=datetime.now() - timedelta(days=90))) as stream:
    batch_count = 0
    for batch in stream.batches(size=1000):
        batch_count += 1
        # Process batch...
        print(f"  Processing batch {batch_count} with {len(batch)} entries")
        if batch_count >= 3:
            print("  (stopping early for demo)")
            break

# =============================================================================
# Audit Configuration
# =============================================================================

print("\n" + "=" * 60)
print("AUDIT CONFIGURATION")
print("=" * 60)

# Configure audit retention at org level
org = fs.organization("acme")

config = org.set_audit_config(
    retention_days=365,  # Keep logs for 1 year
    immutable=True,  # Prevent log tampering
    export_destination="s3://acme-audit-archive/",  # Auto-export before deletion
)

print(f"Audit configuration for {org.name}:")
print(f"  Retention: {config.retention_days} days")
print(f"  Immutable: {config.immutable}")
print(f"  Export destination: {config.export_destination}")

# Get current config
current_config = org.get_audit_config()
print(f"\nCurrent configuration retrieved successfully")

# =============================================================================
# Pagination for Large Result Sets
# =============================================================================

print("\n" + "=" * 60)
print("PAGINATION")
print("=" * 60)

# First page
page1 = fs.audit.query(
    resource="*",
    limit=10,
)

print(f"Page 1: {len(page1.entries)} entries")
print(f"  Has more: {page1.has_more}")
print(f"  Next cursor: {page1.next_cursor}")

# Get next page
if page1.has_more and page1.next_cursor:
    page2 = fs.audit.query(
        resource="*",
        limit=10,
        cursor=page1.next_cursor,
    )
    print(f"\nPage 2: {len(page2.entries)} entries")
