# Raise - Feature Store API for ML Infrastructure

A Python API for managing ML features in columnar storage backends. Designed for AI researchers working in notebook environments.

## Installation

```bash
pip install raise-featurestore

# Optional: for YAML file support
pip install pyyaml
```

## Quick Start

```python
from raise_ import FeatureStore

# Connect to the feature store
fs = FeatureStore("acme/mlplatform/recommendation")

# Create a feature group
user_signals = fs.create_feature_group("user-signals")

# Create features
user_signals.create_feature("click_count", dtype="int64")
user_signals.create_feature("user_embedding", dtype="float32[512]")

# Create a derived feature with SQL expression
user_signals.create_feature(
    "ctr",
    dtype="float64",
    derived_from="click_count / NULLIF(impression_count, 0)",
)
```

## Table of Contents

- [Namespace Hierarchy](#namespace-hierarchy)
- [Data Types](#data-types)
- [Feature Creation](#feature-creation)
- [Derived Features](#derived-features)
- [Bulk Operations](#bulk-operations)
- [Versioning](#versioning)
- [Lineage Tracking](#lineage-tracking)
- [Access Control (ACL)](#access-control-acl)
- [Cross-Organization Access](#cross-organization-access)
- [Audit Logging](#audit-logging)
- [Validation](#validation)
- [Analytics](#analytics)
  - [Analysis Types](#analysis-types)
  - [Freshness Control](#freshness-control)
  - [Live Tables](#live-tables)
  - [Dashboards](#dashboards)
  - [Analytics Alerts](#analytics-alerts)
- [API Reference](#api-reference)

---

## Namespace Hierarchy

Raise uses a 5-level namespace hierarchy:

```
organization / domain / project / feature_group / feature @ version
```

| Level | Purpose | Example |
|-------|---------|---------|
| `organization` | Company/team boundary, billing | `acme` |
| `domain` | Business or technical domain | `mlplatform`, `search`, `ads` |
| `project` | Specific initiative or model | `recommendation`, `fraud-detection` |
| `feature_group` | Logical grouping of related features | `user-signals`, `item-attributes` |
| `feature` | Individual feature | `user_embedding` |

### Client Initialization

```python
from raise_ import FeatureStore

# Using path string
fs = FeatureStore("acme/mlplatform/recommendation")

# Using explicit parameters
fs = FeatureStore(org="acme", domain="mlplatform", project="recommendation")

# Change context
fs = fs.with_context(project="fraud-detection")
```

### Hierarchy Navigation

```python
# Organizations
org = fs.organization("acme")

# Domains
domain = fs.domain("mlplatform")
domains = fs.list_domains()

# Projects
project = fs.project("recommendation")
projects = fs.list_projects(tags=["ml"])

# Feature Groups
group = fs.feature_group("user-signals")
groups = fs.list_feature_groups()
```

---

## Data Types

### String Shortcuts

```python
# Primitive types
"int64", "float32", "float64", "bool", "string", "bytes", "timestamp"

# Embeddings (fixed-size arrays)
"float32[512]"      # 512-dimensional float32 embedding
"float16[768]"      # 768-dimensional float16 embedding

# Variable-length arrays
"int64[]"           # Variable-length int array
"float64[:100]"     # Array with max length 100

# Bounded strings
"string[128]"       # String with max length 128
```

### Typed Constructors

```python
from raise_ import Int64, Float32, Embedding, Array, Struct

# Embedding with explicit dtype
Embedding(512)                        # float32[512]
Embedding(768, dtype="float16")       # float16[768]

# Array types
Array(Int64(), max_length=100)

# Nested structures
Struct({"lat": Float64(), "lon": Float64()})
```

---

## Feature Creation

### Single Feature

```python
feature = group.create_feature(
    name="user_embedding",
    dtype="float32[512]",
    description="User profile embedding",
    tags=["embedding", "prod"],
    owner="ml-team@acme.com",
    nullable=True,
    default=None,
    if_exists="error",  # "error" | "skip" | "update"
)
```

### Path Syntax

```python
# Create using path: group/feature
feature = fs.create_feature(
    "user-signals/click_count",
    dtype="int64",
)

# Retrieve using path
feature = fs.feature("user-signals/click_count")
```

### Idempotent Creation

```python
# Won't raise error if feature exists
feature = group.create_feature("clicks", dtype="int64", if_exists="skip")

# Alternative
feature = group.get_or_create_feature("clicks", dtype="int64")
```

---

## Derived Features

Derived features are computed from other features using SQL-like expressions.

### Basic Expressions

```python
# Arithmetic with null protection
ctr = group.create_feature(
    "ctr",
    dtype="float64",
    derived_from="click_count / NULLIF(impression_count, 0)",
)

# Using SQL functions
normalized = group.create_feature(
    "normalized_score",
    dtype="float64",
    derived_from="(raw_score - AVG(raw_score)) / STDDEV(raw_score)",
)
```

### Supported Functions

| Category | Functions |
|----------|-----------|
| Aggregations | `AVG`, `SUM`, `MIN`, `MAX`, `COUNT`, `STDDEV`, `VARIANCE` |
| Math | `ABS`, `CEIL`, `FLOOR`, `ROUND`, `LOG`, `EXP`, `POWER`, `SQRT` |
| Vector | `DOT`, `COSINE_SIMILARITY`, `L2_DISTANCE`, `NORM` |
| String | `CONCAT`, `LOWER`, `UPPER`, `TRIM`, `SUBSTRING`, `LENGTH` |
| Conditional | `CASE WHEN`, `COALESCE`, `NULLIF`, `IF` |
| Window | `OVER`, `PARTITION BY`, `ORDER BY`, `ROWS/RANGE` |

### Conditional Expressions

```python
user_tier = group.create_feature(
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
)
```

### Cross-Group References

```python
# Reference features from another group in the same project
affinity = group.create_feature(
    "user_item_affinity",
    dtype="float32",
    derived_from="DOT(user_embedding, item-signals.item_embedding)",
)
```

---

## Bulk Operations

### From Schema Dictionary

```python
features = group.create_features_from_schema({
    "click_count": "int64",
    "impression_count": "int64",
    "user_embedding": "float32[512]",
})
```

### From List of Definitions

```python
features = group.create_features([
    {"name": "clicks", "dtype": "int64", "description": "Total clicks"},
    {"name": "ctr", "dtype": "float64", "derived_from": "clicks / impressions"},
])
```

### From YAML File

```yaml
# features.yaml
features:
  - name: click_count
    dtype: int64
    description: Total clicks
    tags: [engagement]

  - name: ctr
    dtype: float64
    derived_from: click_count / NULLIF(impression_count, 0)
    tags: [derived, ratio]
```

```python
features = group.create_features_from_file("features.yaml")
```

---

## Versioning

Features are automatically versioned. Schema changes require creating a new version.

```python
# Creates user_embedding@v1
feat_v1 = group.create_feature("user_embedding", dtype="float32[512]")

# Creates user_embedding@v2
feat_v2 = group.create_feature(
    "user_embedding",
    dtype="float32[768]",  # Schema change
    version="v2",
)

# Access specific versions
latest = group.feature("user_embedding")       # Latest version
v1 = group.feature("user_embedding@v1")        # Specific version

# List all versions
versions = feat_v1.list_versions()
```

---

## Lineage Tracking

Derived features automatically track their dependencies.

```python
# Get lineage
lineage = ctr.get_lineage()

# Direct dependencies
print(lineage.upstream)      # [click_count, impression_count]
print(lineage.downstream)    # Features that depend on ctr

# Transitive dependencies
all_deps = lineage.all_upstream()
all_dependents = lineage.all_downstream(include_external=True)

# Visualize as ASCII graph
print(lineage.as_graph().to_ascii())
#
# click_count ─────────┐
#                      ├──▶ ctr
# impression_count ────┘
#

# Export as JSON
lineage.to_dict()
```

---

## Access Control (ACL)

ACLs cascade from parent to child by default.

### Setting ACLs

```python
from raise_ import ACL

# Set at any level
group.set_acl(ACL(
    readers=["ml-engineers@acme.com"],
    writers=["ml-team@acme.com"],
    admins=["ml-leads@acme.com"],
))

# Override without inheritance
feature.set_acl(ACL(
    readers=["restricted@acme.com"],
    inherit=False,
))
```

### Querying ACLs

```python
# Get effective ACL (resolved inheritance)
effective = feature.get_effective_acl()

# View inheritance chain
chain = feature.get_acl_chain()
```

---

## Cross-Organization Access

Share features with external organizations.

### Granting Access

```python
from datetime import datetime, timedelta

# Grant read access to specific features
group.grant_external_access(
    org="partner-org",
    features=["user_embedding", "item_embedding"],
    permission="read",
    expires_at=datetime.now() + timedelta(days=365),
)

# Grant access to all features
group.grant_external_access(
    org="partner-org",
    features=["*"],
    permission="read",
)
```

### Using Cross-Org Features

```python
# Reference with @ prefix for absolute path
affinity = group.create_feature(
    "cross_org_affinity",
    dtype="float32",
    derived_from="DOT(user_embedding, @partner-org/shared/embeddings/items.item_embedding)",
)
```

### Managing Grants

```python
# List grants
grants = group.list_external_grants()

# Revoke access
group.revoke_external_access(org="partner-org")
```

### Path Resolution

| Scope | Syntax | Example |
|-------|--------|---------|
| Same group | `feature` | `click_count` |
| Same project | `group.feature` | `user-signals.click_count` |
| Same domain | `project/group.feature` | `rec/user-signals.click_count` |
| Same org | `domain/project/group.feature` | `ml/rec/user-signals.click_count` |
| Cross-org | `@org/domain/project/group.feature` | `@partner/ml/rec/items.embedding` |

---

## Audit Logging

Track all access and changes to features (metadata only).

### Querying Logs

```python
from datetime import datetime, timedelta

# Query recent logs
logs = fs.audit.query(
    resource="user-signals/*",
    actions=["READ", "WRITE", "CREATE"],
    since=datetime.now() - timedelta(days=7),
)

for entry in logs:
    print(f"{entry.timestamp} | {entry.actor} | {entry.action}")

# Track external access
external = fs.audit.query(
    resource="user-signals/*",
    exclude_actor_orgs=["acme"],  # Exclude own org
)

# Feature-level logs
logs = feature.audit_log(category="schema")
```

### Audit Actions

| Category | Actions |
|----------|---------|
| Access | `READ`, `WRITE`, `QUERY` |
| Schema | `CREATE`, `UPDATE_SCHEMA`, `UPDATE_METADATA`, `DELETE`, `DEPRECATE`, `CREATE_VERSION` |
| ACL | `UPDATE_ACL`, `GRANT_EXTERNAL`, `REVOKE_EXTERNAL` |
| Lineage | `UPDATE_DERIVATION`, `LINEAGE_BREAK` |

### Alerts

```python
from raise_.models.audit import AuditQuery

# Create alert for external access
fs.audit.create_alert(
    name="external-access-alert",
    query=AuditQuery(
        resource="user-signals/*",
        exclude_actor_orgs=["acme"],
    ),
    notify=["security@acme.com"],
    channels=["email", "slack"],
)

# List and manage alerts
alerts = fs.audit.list_alerts()
fs.audit.delete_alert("external-access-alert")
```

### Export

```python
# Export for compliance
fs.audit.export(
    query=AuditQuery(since=datetime(2025, 1, 1)),
    format="parquet",  # jsonl, csv, parquet
    destination="s3://bucket/audit/",
)

# Streaming export for large datasets
with fs.audit.stream(query) as stream:
    for batch in stream.batches(size=10000):
        process(batch)
```

### Retention Configuration

```python
org.set_audit_config(
    retention_days=365,
    immutable=True,
    export_destination="s3://audit-archive/",
)
```

---

## Validation

Validate expressions before creating features.

```python
result = group.validate_feature(
    "ctr",
    dtype="float64",
    derived_from="click_count / impression_count",
)

if result.valid:
    print(f"Inferred type: {result.inferred_type}")
    print(f"References: {[r.name for r in result.references]}")
else:
    for error in result.errors:
        print(f"Error: {error.code} - {error.message}")

for warning in result.warnings:
    print(f"Warning: {warning.message}")
```

### Validation Levels

```python
# Strict: warnings become errors
feature = group.create_feature(..., validation="strict")

# Standard: warnings shown but allowed (default)
feature = group.create_feature(..., validation="standard")

# Permissive: syntax checks only
feature = group.create_feature(..., validation="permissive")
```

### Validation Checks

- Syntax errors
- Unknown feature references
- Type compatibility
- Function signatures
- Circular dependencies
- Cross-org permissions

---

## Analytics

Raise provides comprehensive analytics capabilities for analyzing feature data, creating dashboards, setting up live tables with CDC-based refresh, and configuring alerts.

### Analysis Types

#### Aggregation

Compute aggregated metrics on features with optional time windows and grouping.

```python
from raise_ import Aggregation, Freshness

# Simple aggregation
result = feature_group.analyze(
    Aggregation(
        feature="click_count",
        metrics=["count", "sum", "avg", "min", "max", "null_rate"],
    )
)

print(result.metrics)  # {"count": 10000, "sum": 150000, "avg": 15.0, ...}

# Time-windowed aggregation with grouping
result = feature_group.analyze(
    Aggregation(
        feature="revenue",
        metrics=["sum", "avg"],
        window="7d",
        group_by="user_tier",
    )
)

# Rolling aggregation
result = feature_group.analyze(
    Aggregation(
        feature="click_count",
        metrics=["sum"],
        window="1d",
        rolling=True,
        periods=30,
    )
)
```

#### Distribution

Analyze data distributions with histograms and percentiles.

```python
from raise_ import Distribution

result = feature_group.analyze(
    Distribution(
        feature="revenue",
        metrics=["histogram", "percentiles"],
        bins=50,
    )
)

print(result.histogram)      # {"bin_edges": [...], "counts": [...]}
print(result.percentiles)    # {"p50": 22.5, "p75": 35.0, "p90": 42.0, ...}

# Segmented distribution
result = feature_group.analyze(
    Distribution(
        feature="revenue",
        segment_by="user_tier",
        metrics=["histogram", "percentiles"],
    )
)
```

#### Correlation

Compute correlation matrices across multiple features.

```python
from raise_ import Correlation

result = feature_group.analyze(
    Correlation(
        features=["click_count", "impression_count", "revenue"],
        method="pearson",  # or "spearman", "kendall"
    )
)

print(result.correlation_matrix)
df = result.to_dataframe()  # Convert to pandas DataFrame
```

#### Version Diff

Compare schema and data distribution between feature versions.

```python
from raise_ import VersionDiff

result = feature_group.analyze(
    VersionDiff(
        feature="user_embedding",
        version_a="v1",
        version_b="v2",
        compare=["schema", "distribution"],
    )
)

print(result.schema_changes)       # {"dtype": {"old": "float32[512]", "new": "float32[768]"}}
print(result.distribution_drift)   # {"psi": 0.15, "kl_divergence": 0.08}
```

#### Statistical Testing

Run statistical tests for A/B experiments.

```python
from raise_ import StatTest

result = feature_group.analyze(
    StatTest(
        feature="conversion_rate",
        test="ttest",            # or "mann_whitney", "chi_squared", "ks_test"
        segment_by="experiment_group",
        control="control",
        treatment="variant_a",
    )
)

print(f"P-value: {result.p_value}")
print(f"Effect size: {result.effect_size}")
print(f"Confidence interval: {result.confidence_interval}")
print(f"Significant: {result.p_value < 0.05}")
```

#### Record Lookup

Sample and filter specific records for inspection.

```python
from raise_ import RecordLookup

# Sample random records
result = feature_group.analyze(
    RecordLookup(
        features=["click_count", "revenue", "user_tier"],
        sample=10,
    )
)

for record in result.records:
    print(record)

# Filter specific records
result = feature_group.analyze(
    RecordLookup(
        features=["click_count", "revenue"],
        filter="user_tier = 'gold'",
        limit=5,
    )
)
```

### Freshness Control

Control how fresh the analysis results need to be.

```python
from raise_ import Freshness, Aggregation

# Real-time: always compute fresh
result = feature_group.analyze(
    Aggregation(feature="click_count", metrics=["sum"]),
    freshness=Freshness.REAL_TIME,
)

# Within 1 hour: use cache if fresh enough
result = feature_group.analyze(
    Aggregation(feature="click_count", metrics=["sum"]),
    freshness=Freshness.WITHIN("1h"),
)

# Cached: always use cache if available
result = feature_group.analyze(
    Aggregation(feature="click_count", metrics=["sum"]),
    freshness=Freshness.CACHED,
)
```

### Live Tables

Create materialized tables that auto-refresh using CDC (Change Data Capture).

```python
# Create a live table
daily_metrics = feature_group.create_live_table(
    name="daily_engagement",
    analysis=Aggregation(
        feature="click_count",
        metrics=["sum", "avg", "count"],
        window="1d",
        group_by="user_tier",
    ),
    refresh="on_change",  # Uses CDC to detect changes
    description="Daily engagement metrics by tier",
)

print(daily_metrics.status)          # "active"
print(daily_metrics.refresh_policy)  # RefreshPolicy(type="on_change")
print(daily_metrics.last_refresh)    # datetime

# Query the live table
df = daily_metrics.query()

# Query with filter
df = daily_metrics.query(filter="user_tier = 'gold'")

# Manual refresh
event = daily_metrics.refresh()
print(event.status)  # "completed"

# Refresh history
history = daily_metrics.refresh_history(limit=5)
```

#### Refresh Policies

```python
from raise_.analytics import RefreshPolicy

# CDC-based (default)
RefreshPolicy.on_change()

# Scheduled
RefreshPolicy.hourly()
RefreshPolicy.daily()
RefreshPolicy.cron("0 */6 * * *")  # Every 6 hours

# Manual only
RefreshPolicy.manual()
```

### Dashboards

Create interactive dashboards with parameterized charts.

```python
from raise_ import (
    Dashboard, Chart, ChartType,
    DashboardParameter, Aggregation, Distribution, Correlation
)

# Create dashboard
dashboard = fs.create_dashboard(
    name="engagement-overview",
    description="User engagement metrics dashboard",
)

# Add parameters for filtering
dashboard.add_parameter(
    DashboardParameter.date_range(
        name="date_range",
        label="Date Range",
    )
)

dashboard.add_parameter(
    DashboardParameter.dropdown(
        name="tier",
        label="User Tier",
        options=["all", "bronze", "silver", "gold", "platinum"],
        default="all",
    )
)

# Add charts
dashboard.add_chart(
    Chart(
        title="Daily Click Trend",
        analysis=Aggregation(
            feature="click_count",
            metrics=["sum"],
            window="1d",
            rolling=True,
            periods=30,
        ),
        chart_type=ChartType.LINE,
        x="date",
        y="sum",
    )
)

dashboard.add_chart(
    Chart(
        title="Revenue by Tier",
        analysis=Aggregation(
            feature="revenue",
            metrics=["sum"],
            group_by="user_tier",
        ),
        chart_type=ChartType.BAR,
        x="user_tier",
        y="sum",
    )
)

dashboard.add_chart(
    Chart(
        title="Click Distribution",
        analysis=Distribution(
            feature="click_count",
            metrics=["histogram"],
        ),
        chart_type=ChartType.HISTOGRAM,
    )
)

dashboard.add_chart(
    Chart(
        title="Feature Correlations",
        analysis=Correlation(
            features=["click_count", "impression_count", "revenue"],
        ),
        chart_type=ChartType.HEATMAP,
    )
)

# Link chart to live table
dashboard.add_chart(
    Chart(
        title="Live Engagement Metrics",
        live_table="daily_engagement",
        chart_type=ChartType.TABLE,
    )
)

# Render dashboard
spec = dashboard.render(format="json")  # or "html", "yaml"

# Publish dashboard
url = dashboard.publish()
print(f"Published at: {url}")
```

#### Chart Types

| ChartType | Description |
|-----------|-------------|
| `LINE` | Time series line chart |
| `BAR` | Bar chart for categorical data |
| `HISTOGRAM` | Distribution histogram |
| `SCATTER` | Scatter plot |
| `HEATMAP` | Correlation heatmap |
| `PIE` | Pie chart |
| `AREA` | Stacked area chart |
| `TABLE` | Data table |
| `METRIC` | Single metric display |
| `GAUGE` | Gauge visualization |
| `BOX` | Box plot |
| `VIOLIN` | Violin plot |

### Analytics Alerts

Set up alerts based on analysis conditions.

```python
from raise_ import Condition, Aggregation, VersionDiff, StatTest

# Alert on high null rate
null_alert = fs.create_alert(
    name="high-null-rate",
    analysis=Aggregation(
        feature="user_embedding",
        metrics=["null_rate"],
    ),
    condition=Condition.GREATER_THAN(0.05),
    notify=["data-quality@acme.com"],
    channels=["email", "slack"],
    check_interval="1h",
)

# Alert on distribution drift
drift_alert = fs.create_alert(
    name="embedding-drift",
    analysis=VersionDiff(
        feature="user_embedding",
        version_a="v1",
        version_b="v2",
        compare=["distribution"],
    ),
    condition=Condition.PSI_GREATER_THAN(0.2),
    notify=["ml-team@acme.com"],
    check_interval="1d",
)

# Alert on statistical significance
significance_alert = fs.create_alert(
    name="ab-test-significance",
    analysis=StatTest(
        feature="conversion_rate",
        test="ttest",
        segment_by="experiment_group",
        control="control",
        treatment="variant_a",
    ),
    condition=Condition.P_VALUE_LESS_THAN(0.05),
    notify=["experiments@acme.com"],
    check_interval="6h",
)

# List alerts
alerts = fs.list_alerts()
for alert in alerts:
    print(f"{alert.name}: {alert.status}")
```

#### Conditions

| Condition | Description |
|-----------|-------------|
| `GREATER_THAN(value)` | Metric exceeds threshold |
| `LESS_THAN(value)` | Metric below threshold |
| `BETWEEN(low, high)` | Metric in range |
| `OUTSIDE(low, high)` | Metric outside range |
| `EQUALS(value)` | Metric equals value |
| `NOT_EQUALS(value)` | Metric differs from value |
| `PSI_GREATER_THAN(value)` | Distribution drift (PSI) |
| `P_VALUE_LESS_THAN(value)` | Statistical significance |
| `KL_DIVERGENCE_GREATER_THAN(value)` | KL divergence threshold |

### Async Analysis

Submit large analyses for asynchronous execution.

```python
# Submit async job
job = fs.analyze_async(
    Correlation(
        features=["click_count", "impression_count", "revenue", "conversion_rate"],
        method="spearman",
    )
)

print(f"Job ID: {job.id}")
print(f"Status: {job.status}")  # "pending", "running", "completed", "failed"

# Wait for completion
result = job.wait(timeout=60)
print(result.to_dict())

# Or check status periodically
while job.status in ("pending", "running"):
    time.sleep(5)
    job.refresh()

if job.status == "completed":
    result = job.result()
```

### Result Formats

Analysis results can be exported to various formats.

```python
result = feature_group.analyze(
    Correlation(features=["click_count", "revenue"])
)

# Convert to pandas DataFrame
df = result.to_dataframe()

# Export to JSON
json_str = result.to_json()

# Export to CSV
csv_str = result.to_csv()

# Access raw data
print(result.data)
```

---

## API Reference

### FeatureStore

```python
class FeatureStore:
    def __init__(self, path: str = None, *, org: str = None, domain: str = None, project: str = None)
    def with_context(self, org: str = None, domain: str = None, project: str = None) -> FeatureStore

    # Organization
    def create_organization(self, name: str, **kwargs) -> Organization
    def organization(self, name: str) -> Organization
    def list_organizations(self) -> list[Organization]

    # Domain
    def create_domain(self, name: str, **kwargs) -> Domain
    def domain(self, name: str = None) -> Domain
    def list_domains(self, tags: list[str] = None) -> list[Domain]

    # Project
    def create_project(self, name: str, **kwargs) -> Project
    def project(self, name: str = None) -> Project
    def list_projects(self, tags: list[str] = None) -> list[Project]

    # Feature Group
    def create_feature_group(self, name: str, **kwargs) -> FeatureGroup
    def feature_group(self, name: str) -> FeatureGroup
    def list_feature_groups(self, tags: list[str] = None) -> list[FeatureGroup]

    # Feature (path syntax)
    def create_feature(self, path: str, dtype: str, **kwargs) -> Feature
    def feature(self, path: str) -> Feature
    def search_features(self, query: str = None, **kwargs) -> list[Feature]

    # Audit
    audit: AuditClient

    # Analytics
    def analyze(self, analysis: Analysis, freshness: Freshness = None) -> AnalysisResult
    def analyze_async(self, analysis: Analysis) -> AnalysisJob
    def create_dashboard(self, name: str, description: str = None) -> Dashboard
    def create_alert(self, name: str, analysis: Analysis, condition: Condition, ...) -> AnalyticsAlert
    def list_alerts(self) -> list[AnalyticsAlert]
```

### FeatureGroup

```python
class FeatureGroup:
    def create_feature(self, name: str, dtype: str, **kwargs) -> Feature
    def create_features(self, features: list[dict], **kwargs) -> list[Feature]
    def create_features_from_schema(self, schema: dict, **kwargs) -> list[Feature]
    def create_features_from_file(self, path: str, **kwargs) -> list[Feature]
    def get_or_create_feature(self, name: str, dtype: str, **kwargs) -> Feature
    def feature(self, name: str) -> Feature
    def list_features(self, tags: list[str] = None, **kwargs) -> list[Feature]
    def validate_feature(self, name: str, dtype: str, **kwargs) -> ValidationResult

    # ACL
    def set_acl(self, acl: ACL) -> None
    def get_effective_acl(self) -> ACL
    def get_acl_chain(self) -> list[ACL]

    # External access
    def grant_external_access(self, org: str, features: list[str], **kwargs) -> ExternalGrant
    def list_external_grants(self) -> list[ExternalGrant]
    def revoke_external_access(self, org: str) -> bool

    # Analytics
    def analyze(self, analysis: Analysis, freshness: Freshness = None) -> AnalysisResult
    def create_live_table(self, name: str, analysis: Analysis, refresh: str = "on_change", ...) -> LiveTable
```

### Feature

```python
class Feature:
    # Properties
    name: str
    qualified_name: str
    dtype: FeatureType
    version: str
    description: str | None
    tags: list[str]
    owner: str
    derived_from: str | None
    is_derived: bool
    status: str  # "active", "deprecated", "archived"

    # Methods
    def update(self, **kwargs) -> Feature
    def delete(self) -> None
    def deprecate(self, message: str = None) -> None
    def get_lineage(self) -> Lineage
    def list_versions(self) -> list[Feature]
    def audit_log(self, **kwargs) -> AuditQueryResult

    # ACL
    def set_acl(self, acl: ACL) -> None
    def get_effective_acl(self) -> ACL
    def get_acl_chain(self) -> list[ACL]
```

---

## Examples

See the `examples/` directory for complete working examples:

1. **01_basic_feature_creation.py** - Simple feature CRUD operations
2. **02_derived_features.py** - SQL expressions and lineage tracking
3. **03_bulk_operations.py** - Bulk creation from schema, list, and YAML
4. **04_cross_org_access.py** - Cross-organization sharing and ACLs
5. **05_audit_logging.py** - Audit queries, alerts, and exports
6. **06_analytics.py** - Aggregations, distributions, live tables, dashboards, and alerts

---

## License

MIT License
