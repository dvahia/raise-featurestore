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

# Create a feature group with an entity key for serving-time lookups
user_signals = fs.create_feature_group("user-signals", entity_key="user_id")

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

## API Design Philosophy

Raise supports both **declarative** and **procedural** API patterns. Choose the style that fits your workflow:

### Declarative Style

Declare *what* you want. The system handles creation, validation, and idempotency.

```python
# Schema-based: declare all features at once
group.create_features_from_schema({
    "click_count": "int64",
    "impression_count": "int64",
    "ctr": "float64",
})

# YAML-based: declare features from a file
group.create_features_from_file("features.yaml")

# Derived features: declare the computation, not how to compute
group.create_feature(
    "ctr",
    dtype="float64",
    derived_from="click_count / NULLIF(impression_count, 0)",  # SQL expression
)

# Idempotent operations: safe to run multiple times
group.create_feature("clicks", dtype="int64", if_exists="skip")

# Analytics: declare what analysis you want
group.analyze(Aggregation(feature="clicks", metrics=["sum", "avg", "p99"]))

# Jobs: declare the pipeline, not the execution steps
job = fs.create_job(
    name="daily_clicks",
    sources=[ObjectStorage(path="s3://events/")],
    transform=SQLTransform(sql="SELECT user_id, COUNT(*) as clicks FROM source GROUP BY 1"),
    target=Target(feature_group="user-signals"),
    schedule=Schedule.daily(hour=2),
)

# Alerts: declare conditions, not monitoring logic
fs.create_alert(
    name="high-null-rate",
    analysis=Aggregation(feature="embedding", metrics=["null_rate"]),
    condition=Condition.GREATER_THAN(0.05),
    notify=["team@example.com"],
)
```

### Procedural Style

Control each step explicitly. Useful for debugging, learning, or complex workflows.

```python
# Step-by-step hierarchy creation
org = fs.create_organization("acme")
domain = fs.create_domain("mlplatform")
project = fs.create_project("recommendation")
group = fs.create_feature_group("user-signals", entity_key="user_id")

# Explicit feature creation with full control
feature = group.create_feature(
    "click_count",
    dtype="int64",
    description="Total clicks per user",
    tags=["engagement", "core"],
    nullable=False,
)

# Job lifecycle management
job = fs.create_job(name="compute_clicks", ...)
job.activate()                    # Enable scheduling
result = fs.deploy_job(job)       # Deploy to orchestrator
run_id = fs.trigger_job(job)      # Manual trigger
job.pause()                       # Pause scheduling
job.resume()                      # Resume

# Validation before creation
result = group.validate_feature(
    "ctr",
    dtype="float64",
    derived_from="click_count / impressions",
)
if result.valid:
    group.create_feature("ctr", ...)
else:
    print(result.errors)
```

### When to Use Each Style

| Use Declarative When | Use Procedural When |
|---------------------|---------------------|
| Automating with scripts or agents | Debugging issues step-by-step |
| Defining infrastructure-as-code | Learning the API |
| Reproducible, version-controlled configs | Handling partial failures |
| Bulk operations | Fine-grained control over timing |
| You care about *what*, not *how* | You need to inspect intermediate state |

### Mixing Both Styles

The styles are complementary. A common pattern:

```python
# Declarative: bulk setup
group.create_features_from_schema({
    "clicks": "int64",
    "impressions": "int64",
})

# Procedural: selective updates
ctr = group.create_feature("ctr", dtype="float64", derived_from="clicks / impressions")
ctr.update(description="Click-through rate", tags=["derived", "ratio"])

# Declarative: analytics
result = group.analyze(Aggregation(feature="ctr", metrics=["avg", "p50", "p99"]))

# Procedural: conditional logic
if result.metrics["null_rate"] > 0.1:
    ctr.deprecate(message="High null rate detected")
```

## Table of Contents

- [API Design Philosophy](#api-design-philosophy)
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
- [Transformations](#transformations)
  - [Creating Jobs](#creating-jobs)
  - [Sources](#sources)
  - [Transforms (SQL/Python)](#transforms-sqlpython)
  - [Schedules](#schedules)
  - [Incremental Processing](#incremental-processing)
  - [Quality Checks](#quality-checks)
  - [Airflow Integration](#airflow-integration)
- [Multimodal Data](#multimodal-data)
  - [Blob References](#blob-references)
  - [Video and Audio Clips (TimeRange)](#video-and-audio-clips-timerange)
  - [BlobRef Feature Type](#blobref-feature-type)
  - [Blob Registry](#blob-registry)
  - [Multimodal Sources](#multimodal-sources)
  - [Integrity Validation](#integrity-validation)
  - [Blob Integrity Checks](#blob-integrity-checks)
- [Bulk Inference](#bulk-inference)
  - [Model Specification](#model-specification)
  - [Accelerator Configuration](#accelerator-configuration)
  - [Inference Transforms](#inference-transforms)
  - [Convenience Functions](#convenience-functions)
  - [Chained Inference](#chained-inference)
- [Dataset Curation](#dataset-curation)
  - [Quality Scoring](#quality-scoring)
  - [Deduplication](#deduplication)
  - [Compliance Filtering](#compliance-filtering)
  - [Curation Pipeline](#curation-pipeline)
- [Dataset Mixing and Versioning](#dataset-mixing-and-versioning)
  - [Mix Sources](#mix-sources)
  - [Mixing Strategies](#mixing-strategies)
  - [Dataset Versioning](#dataset-versioning)
  - [Provenance Tracking](#provenance-tracking)
- [Post-Training Data](#post-training-data)
  - [SFT Datasets](#sft-datasets)
  - [Preference Data and DPO](#preference-data-and-dpo)
  - [Evaluation Suites](#evaluation-suites)
  - [Human Annotation Tasks](#human-annotation-tasks)
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

## Transformations

Raise supports ETL transformations that populate features from various data sources.

### Creating Jobs

```python
from raise_ import (
    FeatureStore,
    ObjectStorage,
    SQLTransform,
    Schedule,
    Target,
    IncrementalConfig,
)

fs = FeatureStore("acme/mlplatform/recommendation")

# Create a transformation job
job = fs.create_job(
    name="compute_daily_clicks",
    sources=[
        ObjectStorage(
            path="s3://data-lake/events/clickstream/",
            format="parquet",
        )
    ],
    transform=SQLTransform(
        name="daily_clicks_transform",
        sql="""
            SELECT user_id, DATE(event_time) as date, COUNT(*) as clicks
            FROM source
            WHERE event_time >= '{{checkpoint}}'
            GROUP BY user_id, DATE(event_time)
        """,
    ),
    target=Target(
        feature_group="user-signals",
        features={"clicks": "daily_clicks"},
        write_mode="upsert",
        key_columns=["user_id", "date"],
    ),
    schedule=Schedule.daily(hour=2),
    incremental=IncrementalConfig.incremental("event_time"),
)

# Activate and deploy
job.activate()
result = fs.deploy_job(job)
print(f"Deployed: {result.url}")
```

### Sources

```python
from raise_ import ObjectStorage, FileSystem, ColumnarSource, FeatureGroupSource

# S3/GCS/Azure Blob
s3 = ObjectStorage(
    path="s3://bucket/prefix/",
    format="parquet",
    partition_columns=["year", "month", "day"],
)

# Local/network filesystem
local = FileSystem(
    path="/data/exports/",
    format="csv",
    glob_pattern="*.csv",
)

# Data warehouse table
warehouse = ColumnarSource(
    catalog="analytics",
    database="production",
    table="events",
    filter="timestamp >= CURRENT_DATE - INTERVAL '7 days'",
)

# Read from existing features
features = FeatureGroupSource(
    feature_group="acme/mlplatform/recommendation/user-signals",
    features=["daily_clicks", "daily_revenue"],
)
```

### Transforms (SQL/Python)

#### SQL Transform

```python
from raise_ import SQLTransform

transform = SQLTransform(
    name="aggregate_clicks",
    sql="""
        SELECT
            user_id,
            DATE(event_time) as date,
            COUNT(*) as clicks,
            SUM(revenue) as revenue
        FROM source
        WHERE event_time >= '{{checkpoint}}'
        GROUP BY user_id, DATE(event_time)
    """,
)
```

**Template Variables:**
- `{{checkpoint}}` - Current checkpoint value
- `{{execution_date}}` - Logical execution date
- `{{run_id}}` - Unique run identifier

#### Python Transform

```python
from raise_.transforms import python_transform

@python_transform(name="segment_users", dependencies=["pandas"])
def segment_users(context, data):
    import pandas as pd
    df = pd.DataFrame(data)

    def assign_segment(row):
        if row["revenue"] > 100:
            return "high_value"
        elif row["clicks"] > 50:
            return "engaged"
        else:
            return "casual"

    df["segment"] = df.apply(assign_segment, axis=1)
    return df.to_dict("records")

# Use in a job
job = fs.create_job(
    name="segment_users",
    sources=[FeatureGroupSource(feature_group="user-signals")],
    transform=segment_users,
    target="user-signals",
)
```

### Schedules

```python
from raise_ import Schedule

# Cron expression
Schedule.cron("0 */6 * * *")  # Every 6 hours

# Daily at specific time
Schedule.daily(hour=2, minute=30)

# Hourly
Schedule.hourly(minute=15)

# Fixed interval
Schedule.every("30m")

# CDC-triggered (when source changes)
Schedule.on_change(sources=["clickstream"])

# Manual only
Schedule.manual()
```

### Incremental Processing

```python
from raise_ import IncrementalConfig

# Full refresh (recompute everything)
IncrementalConfig.full()

# Timestamp-based incremental
IncrementalConfig.incremental(
    checkpoint_column="event_timestamp",
    lookback="1h",  # Handle late-arriving data
)

# Append-only
IncrementalConfig.append(checkpoint_column="id")

# Upsert with keys
IncrementalConfig.upsert(
    key_columns=["user_id", "date"],
    checkpoint_column="updated_at",
)
```

### Quality Checks

```python
from raise_ import NullCheck, RangeCheck, RowCountCheck

# Check for nulls
null_check = NullCheck(
    name="check_user_id",
    column="user_id",
    max_null_rate=0.0,  # No nulls allowed
)

# Check value range
range_check = RangeCheck(
    name="check_ctr",
    column="click_through_rate",
    min_value=0.0,
    max_value=1.0,
)

# Check row count
count_check = RowCountCheck(
    name="check_minimum_rows",
    min_rows=1000,
)
```

### Airflow Integration

```python
from raise_ import AirflowConfig, generate_airflow_dag

# Configure Airflow
fs.transforms.use_airflow(
    config=AirflowConfig(
        dag_folder="/opt/airflow/dags",
        catchup=False,
        max_active_runs=1,
        tags=["raise", "feature-store"],
    ),
    airflow_url="http://airflow.example.com",
)

# Deploy job as Airflow DAG
result = fs.deploy_job(job)
print(f"DAG URL: {result.url}")

# Generate DAG code without deploying
dag_code = fs.transforms.generate_dag(job)
```

---

## Multimodal Data

Raise supports multimodal data (images, audio, video, etc.) through blob references. References allow transformations to work with metadata without moving the actual data bytes, which is critical for large assets.

### Blob References

A `BlobReference` is an immutable reference to a multimodal blob with integrity metadata.

```python
from raise_ import BlobReference, BlobRegistry, ContentType, create_reference

# Create a registry for tracking references
registry = BlobRegistry(name="product-images")

# Register a blob reference
ref = registry.register(
    uri="s3://products/images/product-001.png",
    content_type=ContentType.IMAGE_PNG,
    checksum="abc123def456...",
    size_bytes=1024000,
    metadata={
        "width": 1920,
        "height": 1080,
        "color_space": "sRGB",
    },
)

print(ref.uri)           # s3://products/images/product-001.png
print(ref.scheme)        # s3
print(ref.bucket)        # products
print(ref.key)           # images/product-001.png
print(ref.is_image)      # True
print(ref.size_bytes)    # 1024000

# Convenience function
ref = create_reference(
    uri="s3://bucket/audio.wav",
    content_type="audio/wav",
    checksum="def789...",
)
```

#### Reference Properties

| Property | Description |
|----------|-------------|
| `uri` | Full URI to the blob |
| `content_type` | MIME type (ContentType enum or string) |
| `checksum` | Content hash for integrity |
| `hash_algorithm` | Algorithm used (SHA256, SHA512, MD5, etc.) |
| `size_bytes` | Size in bytes |
| `etag` | Object storage ETag |
| `version_id` | Version identifier |
| `scheme` | URI scheme (s3, gs, az, file) |
| `bucket` | Bucket name for object storage |
| `key` | Object key for object storage |
| `is_image` | True if content type is image/* |
| `is_audio` | True if content type is audio/* |
| `is_video` | True if content type is video/* |

### Video and Audio Clips (TimeRange)

Every `BlobReference` can represent either a complete file or a **sub-clip** — a contiguous time range within a video or audio file. No data is copied; only the URI plus the time window are stored.

```python
from raise_ import TimeRange, BlobReference, BlobRegistry, ContentType

registry = BlobRegistry(name="video-store")

# Register the full video file
video_ref = registry.register(
    uri="s3://datasets/videos/lecture_001.mp4",
    content_type=ContentType.VIDEO_MP4,
    checksum="e3b0c44298fc1c...",
    size_bytes=2_500_000_000,
    metadata={"duration_sec": 3600.0, "fps": 30.0, "width": 1920, "height": 1080},
)

# Create a clip by calling .clip() — same URI, different time window
intro_clip = video_ref.clip(start_sec=0.0, end_sec=60.0)
interview  = video_ref.clip(start_sec=300.0, end_sec=900.0)

print(intro_clip.is_clip)                  # True
print(intro_clip.time_range)               # TimeRange(0.000s – 60.000s, 60.000s)
print(intro_clip.time_range.duration_sec)  # 60.0

# Clips share the source URI and checksum — no data is duplicated
assert intro_clip.uri == video_ref.uri
assert intro_clip.checksum == video_ref.checksum
```

#### TimeRange

```python
from raise_ import TimeRange

tr = TimeRange(start_sec=30.0, end_sec=90.0)
print(tr.duration_sec)   # 60.0
print(tr.to_dict())      # {"start_sec": 30.0, "end_sec": 90.0}

# Restore from dict
tr2 = TimeRange.from_dict({"start_sec": 30.0, "end_sec": 90.0})
```

#### Clip Properties

| Property | Description |
|----------|-------------|
| `is_clip` | True if the reference has a `time_range` |
| `time_range` | `TimeRange` (start/end in seconds) or None |
| `duration_sec` | Duration of clip (or full-file duration from metadata) |

#### Extracting Fixed-Length Clips

```python
def extract_clips(video_ref, video_id, clip_sec=10.0, stride_sec=5.0):
    total = video_ref.metadata["duration_sec"]
    clips, t, i = [], 0.0, 0
    while t + clip_sec <= total:
        clip = video_ref.clip(start_sec=t, end_sec=t + clip_sec)
        clips.append({"clip_id": f"{video_id}_{i:05d}", "clip_ref": clip.to_dict(),
                      "start_sec": t, "end_sec": t + clip_sec})
        t += stride_sec
        i += 1
    return clips

clips = extract_clips(video_ref, "lecture_001", clip_sec=30.0, stride_sec=15.0)
print(f"{len(clips)} clips, 50% overlap")
```

#### Audio Segments

The same API applies to audio files. Diarization output maps naturally to clips:

```python
registry = BlobRegistry(name="audio-store")
audio_ref = registry.register(
    uri="s3://audio/interview.wav",
    content_type=ContentType.AUDIO_WAV,
    checksum="abc123...",
    metadata={"duration_sec": 3600.0, "sample_rate_hz": 44100},
)

# Diarization: one clip per speaker turn
for turn in diarization_output:
    segment = audio_ref.clip(start_sec=turn["start"], end_sec=turn["end"])
    segment = segment.with_metadata(speaker=turn["speaker"])
    print(f"  {segment.time_range}  speaker={segment.metadata['speaker']}")
```

#### Serialization

Clips serialize and deserialize correctly:

```python
d = clip.to_dict()          # {"uri": ..., "time_range": {"start_sec": ..., "end_sec": ...}, ...}
restored = BlobReference.from_dict(d)
assert restored.is_clip
assert restored.time_range.start_sec == clip.time_range.start_sec
```

---

### BlobRef Feature Type

Use `BlobRef` to define feature columns that store blob references.

```python
from raise_ import FeatureStore, BlobRef

fs = FeatureStore("acme/mlplatform/vision")

# Create feature group with blob reference columns
image_features = fs.create_feature_group("image-features")

image_features.create_features_from_schema({
    "product_id": "string",
    "image_ref": BlobRef(content_types=["image/png", "image/jpeg"]),
    "thumbnail_ref": BlobRef(content_types=["image/jpeg"]),
    "embedding": "float32[512]",
})
```

#### String Syntax

```python
# Any blob type
"blob_ref"

# Constrained to specific content types
"blob_ref<image/png|image/jpeg>"
"blob_ref<audio/wav|audio/mp3|audio/flac>"
"blob_ref<video/mp4>"
```

#### Typed Constructor

```python
from raise_ import BlobRef

# Any content type
BlobRef()

# Only images
BlobRef(content_types=["image/png", "image/jpeg"])

# With registry and validation settings
BlobRef(
    content_types=["video/mp4", "video/webm"],
    registry="video-store",
    validate_on_write=True,
)

# Check if type accepts a content type
image_type = BlobRef(content_types=["image/png", "image/jpeg"])
image_type.accepts("image/png")   # True
image_type.accepts("video/mp4")   # False
```

### Blob Registry

The `BlobRegistry` tracks and validates blob references.

```python
from raise_ import BlobRegistry, IntegrityPolicy

# Create registry with integrity policy
registry = BlobRegistry(
    name="product-images",
    policy=IntegrityPolicy.on_write(),
)

# Register references
ref = registry.register(
    uri="s3://bucket/image.png",
    content_type="image/png",
    checksum="abc123...",
)

# Get by registry ID or URI
ref = registry.get(registry_id)
ref = registry.get_by_uri("s3://bucket/image.png")

# List references with filtering
all_refs = registry.list_references()
png_refs = registry.list_references(content_type=ContentType.IMAGE_PNG)
s3_refs = registry.list_references(prefix="s3://bucket/")

# Find orphaned references (blobs that no longer exist)
orphans = registry.find_orphans()

# Compute checksums
checksum = registry.compute_checksum(data, HashAlgorithm.SHA256)
```

### Multimodal Sources

Scan object storage for multimodal assets and get references (not data).

```python
from raise_ import MultimodalSource, ContentType

# Create source for scanning images
source = MultimodalSource(
    name="product_images",
    uri_prefix="s3://products/images/",
    content_types=[ContentType.IMAGE_JPEG, ContentType.IMAGE_PNG],
    registry=registry,
    compute_checksums=True,
    recursive=True,
)

# Scan returns BlobReferences, not actual data
refs = source.scan()
for ref in refs:
    print(f"{ref.uri} - {ref.content_type}")

# Get reference for specific file
ref = source.get_reference("product-001.png")
```

### Integrity Validation

Configure when and how referential integrity is enforced.

```python
from raise_ import IntegrityPolicy, IntegrityMode

# Validate on every access
strict = IntegrityPolicy.strict()

# Validate only when creating/updating references (default)
on_write = IntegrityPolicy.on_write()

# Validate only on explicit check
lazy = IntegrityPolicy.lazy()

# Custom policy
policy = IntegrityPolicy(
    mode=IntegrityMode.ON_WRITE,
    fail_on_missing=True,
    fail_on_mismatch=True,
    cache_validation_seconds=3600,
)

# Validate a reference
result = registry.validate(ref)
print(result.status)           # BlobStatus.VALID
print(result.valid)            # True
print(result.checksum_matches) # True
print(result.size_matches)     # True

# Batch validation
results = registry.validate_batch(refs)
```

#### Integrity Modes

| Mode | Description |
|------|-------------|
| `STRICT` | Validate on every access |
| `ON_WRITE` | Validate when creating/updating references |
| `ON_READ` | Validate when reading/resolving references |
| `LAZY` | Validate only on explicit check |
| `PERIODIC` | Background validation jobs |

### Blob Integrity Checks

Quality check for validating blob references in transformation pipelines.

```python
from raise_ import BlobIntegrityCheck
from raise_.transforms import CheckSeverity

# Create integrity check
check = BlobIntegrityCheck(
    name="check_image_integrity",
    column="image_ref",
    verify_checksum=True,
    verify_existence=True,
    max_missing_rate=0.01,  # Allow 1% missing
    max_invalid_rate=0.0,   # No invalid checksums
    sample_rate=1.0,        # Check all references
    severity=CheckSeverity.ERROR,
)

# Run the check
result = check.check(data)
print(result.result)   # CheckResult.PASSED
print(result.details)  # {"total_references": 100, "missing_count": 0, ...}
```

### Content Types

Raise supports many multimodal content types.

| Category | Content Types |
|----------|---------------|
| **Images** | `image/png`, `image/jpeg`, `image/webp`, `image/tiff`, `image/bmp`, `image/gif` |
| **Audio** | `audio/wav`, `audio/mpeg` (MP3), `audio/flac`, `audio/ogg`, `audio/aac` |
| **Video** | `video/mp4`, `video/webm`, `video/avi`, `video/quicktime` (MOV) |
| **Documents** | `application/pdf` |
| **3D/Point Clouds** | `model/ply`, `model/obj`, `model/gltf+json` |
| **ML Tensors** | `application/x-numpy` (NPY), `application/x-numpy-compressed` (NPZ), `application/x-pytorch` (PT), `application/x-safetensors` |

```python
from raise_.transforms import ContentType, infer_content_type

# Use enum
ContentType.IMAGE_PNG
ContentType.AUDIO_WAV
ContentType.TENSOR_SAFETENSORS

# Infer from URI
content_type = infer_content_type("s3://bucket/image.png")  # ContentType.IMAGE_PNG
content_type = infer_content_type("data/model.safetensors")  # ContentType.TENSOR_SAFETENSORS
```

### Hash Algorithms

```python
from raise_.transforms import HashAlgorithm

# Supported algorithms
HashAlgorithm.SHA256   # Default, recommended
HashAlgorithm.SHA512   # More secure
HashAlgorithm.MD5      # Fast but not cryptographically secure
HashAlgorithm.BLAKE3   # Very fast, modern
HashAlgorithm.XXH3     # Extremely fast, good for large files
```

### Example: Image Processing Pipeline

```python
from raise_ import (
    FeatureStore, BlobRef, BlobRegistry, MultimodalSource,
    ContentType, IntegrityPolicy, BlobIntegrityCheck,
)

# 1. Create feature group with blob reference column
fs = FeatureStore("acme/mlplatform/vision")
fg = fs.create_feature_group("product-images")
fg.create_features_from_schema({
    "product_id": "string",
    "image_ref": BlobRef(content_types=["image/png", "image/jpeg"]),
    "embedding": "float32[512]",
})

# 2. Create registry with integrity policy
registry = BlobRegistry(
    name="product-images",
    policy=IntegrityPolicy.on_write(),
)

# 3. Scan for images (returns references, not data)
source = MultimodalSource(
    name="product_images",
    uri_prefix="s3://products/images/",
    content_types=[ContentType.IMAGE_JPEG, ContentType.IMAGE_PNG],
    registry=registry,
)
refs = source.scan()

# 4. Process with integrity check
check = BlobIntegrityCheck(
    name="verify_images",
    column="image_ref",
    verify_checksum=True,
)

# Key benefit: Only references move through the pipeline,
# not the actual image bytes (which may be gigabytes)
```

---

## Bulk Inference

Raise supports bulk inference transformations that run AI models on feature data, distinct from traditional ETL. These jobs require specialized accelerators (GPU/TPU) and model-aware configuration.

### Model Specification

Define model location, framework, and configuration.

```python
from raise_ import ModelSpec, ModelFramework, ModelPrecision

# HuggingFace model
model = ModelSpec(
    uri="hf://sentence-transformers/all-MiniLM-L6-v2",
    framework=ModelFramework.HUGGINGFACE,
    version="latest",
    input_schema={"text": "string"},
    output_schema={"embedding": "float32[384]"},
)

# MLflow model from registry
model = ModelSpec(
    uri="mlflow://models:/sentiment-classifier/Production",
    framework=ModelFramework.SKLEARN,
)

# PyTorch model from S3
model = ModelSpec(
    uri="s3://models/custom/text-encoder-v2.pt",
    framework=ModelFramework.PYTORCH,
    precision=ModelPrecision.FP16,
)

# ONNX optimized model
model = ModelSpec(
    uri="s3://models/optimized/embedding.onnx",
    framework=ModelFramework.ONNX,
)
```

#### Supported Frameworks

| Framework | URI Prefix | Description |
|-----------|------------|-------------|
| `PYTORCH` | `s3://`, `file://` | Native PyTorch models |
| `TENSORFLOW` | `s3://`, `gs://` | SavedModel or frozen graphs |
| `ONNX` | `s3://`, `file://` | ONNX Runtime optimized |
| `HUGGINGFACE` | `hf://` | HuggingFace Hub models |
| `MLFLOW` | `mlflow://` | MLflow Model Registry |
| `TRITON` | `triton://` | NVIDIA Triton Inference Server |
| `SKLEARN` | `s3://` | Scikit-learn models |
| `XGBOOST` | `s3://` | XGBoost models |
| `JAX` | `s3://` | JAX/Flax models |

#### Model Precision

| Precision | Description | Use Case |
|-----------|-------------|----------|
| `FP32` | Full precision | Default, highest accuracy |
| `FP16` | Half precision | Standard optimization |
| `BF16` | Brain floating point | Training on A100+ |
| `INT8` | 8-bit quantized | Efficient inference |
| `INT4` | 4-bit quantized | Large LLMs |

### Accelerator Configuration

Configure GPU/TPU resources for inference.

```python
from raise_ import AcceleratorConfig, GPUType, TPUType

# Single GPU
gpu_config = AcceleratorConfig.gpu(GPUType.NVIDIA_T4)

# Multi-GPU with data parallelism
multi_gpu = AcceleratorConfig.multi_gpu(
    gpu_type=GPUType.NVIDIA_A100,
    count=4,
    strategy="data_parallel",
)

# Large GPU for LLMs
large_gpu = AcceleratorConfig.gpu(
    GPUType.NVIDIA_A100_80GB,
    memory_gb=80,
)

# TPU configuration
tpu_config = AcceleratorConfig.tpu(TPUType.TPU_V4, count=8)

# CPU only
cpu_config = AcceleratorConfig.cpu(cores=8)
```

#### GPU Types

| GPU Type | Memory | Use Case |
|----------|--------|----------|
| `NVIDIA_T4` | 16 GB | Entry-level inference |
| `NVIDIA_A10G` | 24 GB | Balanced cost/performance |
| `NVIDIA_A100` | 40 GB | High-performance |
| `NVIDIA_A100_80GB` | 80 GB | Large models (LLMs) |
| `NVIDIA_H100` | 80 GB | Latest generation |
| `NVIDIA_L4` | 24 GB | Efficient inference |

#### Multi-GPU Strategies

| Strategy | Description |
|----------|-------------|
| `data_parallel` | Replicate model, split data batches |
| `tensor_parallel` | Split model across GPUs |
| `pipeline_parallel` | Stage model across GPUs |

### Inference Transforms

Create inference transforms that integrate with the Job infrastructure.

```python
from raise_ import (
    InferenceTransform, ModelSpec, AcceleratorConfig,
    BatchConfig, InferenceRuntime, GPUType,
)

# Define inference transform
transform = InferenceTransform(
    name="embed_text",
    model=ModelSpec(
        uri="hf://sentence-transformers/all-MiniLM-L6-v2",
        framework=ModelFramework.HUGGINGFACE,
    ),
    accelerator=AcceleratorConfig.gpu(GPUType.NVIDIA_T4),
    batch_config=BatchConfig(batch_size=128),
    input_mapping={"text": "user_bio"},
    output_mapping={"embedding": "bio_embedding"},
)

# Create job with inference transform
job = fs.create_job(
    name="generate_embeddings",
    sources=[FeatureGroupSource(group="user-signals")],
    transform=transform,
    target=Target(
        feature_group="user-embeddings",
        features={"bio_embedding": "embedding"},
        write_mode="upsert",
        key_columns=["user_id"],
    ),
    schedule=Schedule.daily(hour=2),
)
```

#### Batch Configuration

```python
from raise_ import BatchConfig

# Standard batch config
batch = BatchConfig(
    batch_size=64,
    prefetch_batches=2,
    num_workers=4,
)

# Dynamic batching for variable-length inputs
dynamic = BatchConfig(
    batch_size=32,
    max_batch_size=256,
    dynamic_batching=True,
)

# Small batches for LLM inference
llm_batch = BatchConfig(
    batch_size=8,
    timeout_seconds=300.0,
    retry_failed=True,
    max_retries=3,
)
```

#### Inference Runtime

```python
from raise_ import InferenceRuntime

runtime = InferenceRuntime(
    image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
    python_version="3.10",
    packages=["transformers>=4.35", "accelerate", "sentence-transformers"],
    env_vars={"HF_HOME": "/cache/huggingface"},
    warmup_batches=2,
)
```

### Convenience Functions

Use convenience functions for common inference patterns.

#### Embedding Inference

```python
from raise_ import embedding_inference, GPUType

transform = embedding_inference(
    model_uri="hf://sentence-transformers/all-MiniLM-L6-v2",
    input_column="text",
    output_column="embedding",
    batch_size=128,
    gpu_type=GPUType.NVIDIA_T4,
)
```

#### Image Inference

```python
from raise_ import image_inference, GPUType

transform = image_inference(
    model_uri="hf://openai/clip-vit-base-patch32",
    image_column="image_ref",
    output_column="image_embedding",
    batch_size=64,
    gpu_type=GPUType.NVIDIA_A10G,
)
```

#### LLM Inference

```python
from raise_ import llm_inference, GPUType, ModelPrecision

transform = llm_inference(
    model_uri="hf://meta-llama/Llama-2-70b-chat-hf",
    prompt_column="prompt",
    output_column="completion",
    max_tokens=512,
    temperature=0.7,
    gpu_type=GPUType.NVIDIA_A100_80GB,
    gpu_count=4,
    multi_gpu_strategy="tensor_parallel",
    precision=ModelPrecision.INT8,
)
```

#### Custom Inference with Decorator

```python
from raise_.transforms import inference_transform, GPUType

@inference_transform(
    name="custom_scorer",
    gpu_type=GPUType.NVIDIA_T4,
)
def custom_scorer(context, batch):
    # Custom inference logic
    scores = model.predict(batch["features"])
    return {"score": scores}
```

### Chained Inference

Chain multiple inference jobs for complex pipelines.

```python
# Step 1: Generate embeddings
job1 = fs.create_job(
    name="step1_embed",
    sources=[FeatureGroupSource(group="user-signals")],
    transform=embedding_inference(
        model_uri="hf://sentence-transformers/all-MiniLM-L6-v2",
        input_column="text",
        output_column="embedding",
    ),
    target=Target(feature_group="embeddings"),
)

# Step 2: Classify based on embeddings
job2 = fs.create_job(
    name="step2_classify",
    sources=[FeatureGroupSource(group="embeddings")],
    transform=classification_inference(
        model_uri="mlflow://models:/classifier/Production",
        input_column="embedding",
        output_column="category",
    ),
    target=Target(feature_group="classifications"),
)
```

### Inference Result

Track inference execution metrics.

```python
# Get inference run details
run = job.runs[-1]
result = run.inference_result

print(f"Total samples: {result.total_samples:,}")
print(f"Success rate: {result.success_rate:.2%}")
print(f"Throughput: {result.throughput_samples_per_sec:.1f} samples/sec")
print(f"Duration: {result.duration_seconds:.1f}s")
print(f"GPU utilization: {result.gpu_utilization_pct:.0f}%")
print(f"Peak memory: {result.peak_memory_gb:.1f} GB")
```

### Example: Full Inference Pipeline

```python
from raise_ import (
    FeatureStore, FeatureGroupSource, Target, Schedule,
    embedding_inference, image_inference, GPUType,
    IncrementalConfig, BlobIntegrityCheck,
)

fs = FeatureStore("acme/mlplatform/vision")

# Create feature groups
user_profiles = fs.create_feature_group("user-profiles")
user_embeddings = fs.create_feature_group("user-embeddings")

# Job 1: Generate text embeddings
text_job = fs.create_job(
    name="embed_user_bios",
    sources=[FeatureGroupSource(group=user_profiles)],
    transform=embedding_inference(
        model_uri="hf://sentence-transformers/all-MiniLM-L6-v2",
        input_column="bio",
        output_column="bio_embedding",
        batch_size=256,
        gpu_type=GPUType.NVIDIA_T4,
    ),
    target=Target(
        feature_group=user_embeddings,
        features={"bio_embedding": "bio_embedding"},
        write_mode="upsert",
        key_columns=["user_id"],
    ),
    schedule=Schedule.daily(hour=2),
    incremental=IncrementalConfig.incremental("updated_at"),
)

# Job 2: Generate image embeddings
image_job = fs.create_job(
    name="embed_profile_images",
    sources=[FeatureGroupSource(group=user_profiles)],
    transform=image_inference(
        model_uri="hf://openai/clip-vit-base-patch32",
        image_column="avatar_ref",
        output_column="avatar_embedding",
        batch_size=64,
        gpu_type=GPUType.NVIDIA_A10G,
    ),
    target=Target(
        feature_group=user_embeddings,
        features={"avatar_embedding": "avatar_embedding"},
    ),
    quality_checks=[
        BlobIntegrityCheck(
            name="verify_avatars",
            column="avatar_ref",
            verify_existence=True,
        ),
    ],
)

# Deploy both jobs
fs.deploy_job(text_job)
fs.deploy_job(image_job)
```

---

## Dataset Curation

Raise provides first-class curation transforms for building high-quality training datasets: quality scoring, near-deduplication, and compliance filtering. Each step is a `Transform` that runs in a `Job` and writes annotation columns back into the feature group alongside the original data.

### Quality Scoring

`QualityScorer` computes per-dimension quality scores and an optional composite score. Records that fail threshold checks can be filtered automatically.

```python
from raise_ import (
    QualityScorer, QualityDimension, QualityThreshold,
    FeatureStore, Schedule, Target, FeatureGroupSource,
)

scorer = QualityScorer(
    name="web_quality",
    dimensions=[
        QualityDimension.FLUENCY,
        QualityDimension.COHERENCE,
        QualityDimension.FORMAT,
    ],
    thresholds=[
        QualityThreshold(QualityDimension.FLUENCY,   min_score=0.5),
        QualityThreshold(QualityDimension.COHERENCE, min_score=0.4),
    ],
    model_uri="hf://acme/web-quality-scorer-v2",
    input_columns=["text"],
    composite_column="quality_score",
    filter_below_threshold=True,
)

# Output columns written per run
print(scorer.output_columns)
# ["quality_score", "quality_fluency", "quality_coherence", "quality_format"]
```

#### Quality Dimensions

| Dimension | Description |
|-----------|-------------|
| `FLUENCY` | Grammatical and linguistic quality |
| `COHERENCE` | Logical coherence and consistency |
| `FACTUALITY` | Factual accuracy and grounding |
| `SAFETY` | Absence of harmful content |
| `RELEVANCE` | Relevance to a target domain or task |
| `COMPLETENESS` | Completeness of information |
| `FORMAT` | Structural correctness (JSON, code, etc.) |
| `DIVERSITY` | Lexical and semantic diversity |

### Deduplication

`DeduplicationTransform` detects near-duplicate records and writes an `is_duplicate` flag and a `duplicate_of` FK.

```python
from raise_ import (
    DeduplicationTransform, DeduplicationConfig, DeduplicationAlgorithm,
)

dedup = DeduplicationTransform(
    name="web_minhash_dedup",
    config=DeduplicationConfig(
        algorithm=DeduplicationAlgorithm.MINHASH,
        threshold=0.85,           # 85% similarity → duplicate
        key_columns=["text"],
        num_perm=256,             # accuracy of MinHash estimation
        action="flag",            # "flag" or "remove"
        keep="first",
    ),
)

# Embedding-based dedup (for multimodal or semantic similarity)
emb_dedup = DeduplicationTransform(
    name="semantic_dedup",
    config=DeduplicationConfig(
        algorithm=DeduplicationAlgorithm.EMBEDDING_SIMILARITY,
        threshold=0.95,
        embedding_column="text_embedding",
        action="remove",
        keep="highest_quality",
        quality_column="quality_score",
    ),
)
```

#### Deduplication Algorithms

| Algorithm | Description |
|-----------|-------------|
| `EXACT_HASH` | Exact byte-level match |
| `MINHASH` | Locality-sensitive hashing for text |
| `SIMHASH` | Bit-level similarity hashing |
| `EMBEDDING_SIMILARITY` | Cosine distance on pre-computed vectors |
| `BLOOM_FILTER` | Probabilistic membership test |

### Compliance Filtering

`ComplianceFilterTransform` applies a `CompliancePolicy` — a set of classifier-backed rules that detect PII, NSFW content, copyright, toxicity, and other compliance concerns.

```python
from raise_ import (
    ComplianceFilterTransform, CompliancePolicy, ComplianceRule,
    ComplianceFlag, ComplianceAction,
)

policy = CompliancePolicy(name="pretraining-standard")
policy.add_rule(ComplianceFlag.NSFW,      threshold=0.7)
policy.add_rule(ComplianceFlag.COPYRIGHT, action=ComplianceAction.FLAG, threshold=0.85)
policy.add_rule(ComplianceFlag.PII,       action=ComplianceAction.REDACT, threshold=0.6)
policy.add_rule(ComplianceFlag.CHILD_SAFETY, threshold=0.3)   # strict

compliance = ComplianceFilterTransform(
    name="pretraining_compliance",
    policy=policy,
    input_columns=["text", "url"],
    passed_column="compliance_passed",
)

print(compliance.output_columns)
# ["compliance_passed", "compliance_nsfw", "compliance_copyright",
#  "compliance_pii", "compliance_child_safety"]
```

#### Compliance Flags and Actions

| Flag | Description |
|------|-------------|
| `PII` | Personally identifiable information |
| `NSFW` | Not safe for work content |
| `COPYRIGHT` | Potentially copyrighted material |
| `HATE_SPEECH` | Hate speech or harassment |
| `TOXIC` | General toxicity |
| `MEDICAL` | Sensitive medical information |
| `LEGAL` | Legal privilege or sensitive legal content |
| `CHILD_SAFETY` | CSAM or related |

| Action | Description |
|--------|-------------|
| `FILTER` | Remove the record (default) |
| `FLAG` | Keep but write annotation columns |
| `REDACT` | Attempt in-place redaction (e.g., PII masking) |

### Curation Pipeline

Compose curation steps into a named, ordered pipeline.

```python
from raise_ import CurationPipeline

pipeline = CurationPipeline(
    name="pretraining-curation",
    steps=[scorer, dedup, compliance],
    stop_on_first_filter=False,  # collect all annotations before filtering
)

print(pipeline.all_output_columns)
# All columns written by all steps

# Inspect individual steps
pipeline.quality_scorer()   # → QualityScorer
pipeline.deduplication()    # → DeduplicationTransform
pipeline.compliance()       # → ComplianceFilterTransform

# Run as a Job
job = fs.create_job(
    name="web_curation",
    sources=[FeatureGroupSource(feature_group="raw-documents", features=["doc_id", "text"])],
    transform=pipeline.steps[0],
    target=Target(feature_group="curated-documents", write_mode="upsert", key_columns=["doc_id"]),
    schedule=Schedule.daily(hour=2),
)
```

---

## Dataset Mixing and Versioning

Pre-training datasets are assembled from multiple curated sources using controlled mixing strategies. Raise tracks the full configuration, weights, and provenance so that mixes are reproducible and auditable.

### Mix Sources

```python
from raise_ import MixSource

web_source = MixSource(
    feature_group="acme/pretraining/web/curated-documents",
    weight=0.70,
    filters=["include_in_training == True", "language == 'en'"],
)

books_source = MixSource(
    feature_group="acme/pretraining/books/curated-documents",
    weight=0.15,
    filters=["include_in_training == True"],
)

code_source = MixSource(
    feature_group="acme/pretraining/code/curated-documents",
    weight=0.15,
)
```

### Mixing Strategies

```python
from raise_ import DatasetMix, MixingStrategy

# Standard weighted mix
weighted = DatasetMix(
    name="pretraining-v3",
    sources=[web_source, books_source, code_source],
    strategy=MixingStrategy.WEIGHTED,
    total_samples=2_000_000_000,
)
print(weighted.effective_weights())   # [0.70, 0.15, 0.15]

# Temperature mixing — cools high-weight sources, warms low-weight ones
temperature = DatasetMix(
    name="pretraining-v3-temp",
    sources=[web_source, books_source, code_source],
    strategy=MixingStrategy.TEMPERATURE,
    temperature=0.7,    # < 1 amplifies dominant source; > 1 flattens toward uniform
    total_samples=2_000_000_000,
)

# Uniform — ignores weights; useful for ablation studies
uniform = DatasetMix(
    name="ablation-uniform",
    sources=[web_source, books_source, code_source],
    strategy=MixingStrategy.UNIFORM,
    total_samples=100_000_000,
)
```

| Strategy | Description |
|----------|-------------|
| `WEIGHTED` | Sample proportional to declared weights |
| `TEMPERATURE` | Apply temperature scaling before normalization |
| `UNIFORM` | Equal probability regardless of weight |
| `PROPORTIONAL` | Proportional to source dataset size |
| `ROUND_ROBIN` | Strict interleaving |

### Dataset Versioning

```python
from raise_ import DatasetVersion

v1 = DatasetVersion(
    name="foundation-pretraining",
    version="v1.0.0",
    feature_group="acme/pretraining/foundation/training-mix",
    num_records=500_000_000,
    size_bytes=2_000 * 1024**3,
    applied_filters=["quality_score >= 0.5"],
)

# Derive v2 from v1 — parent lineage is preserved
v2 = v1.derive(
    version="v2.0.0",
    applied_filters=["quality_score >= 0.65", "is_duplicate == False", "compliance_passed == True"],
    num_records=380_000_000,
    size_bytes=1_520 * 1024**3,
)

print(v2.parent_version)   # "v1.0.0"
print(v2.is_initial)       # False
print(v2.size_gb)          # 1520.0
```

### Provenance Tracking

```python
from raise_ import DatasetProvenance
from datetime import datetime

provenance = DatasetProvenance(
    source_name="CommonCrawl-2024-Q1",
    source_uri="s3://commoncrawl/crawl-data/CC-MAIN-2024-10/",
    collection_date=datetime(2024, 3, 1),
    license="CC0",
    language_codes=["en", "zh", "es", "fr", "de"],
    domain_tags=["web", "news", "blog"],
    pii_reviewed=True,
    consent_documented=False,
)

# Attach to a dataset version
v3 = DatasetVersion(
    name="foundation-pretraining",
    version="v3.0.0",
    feature_group="acme/pretraining/foundation/training-mix",
    num_records=2_000_000_000,
    size_bytes=8_000 * 1024**3,
    provenance=provenance,
)
```

---

## Post-Training Data

Raise supports the full post-training data lifecycle: collecting SFT examples, gathering human preferences, preparing DPO triplets, training reward models, running evaluations, and managing human annotation workflows.

### SFT Datasets

SFT data lives in ordinary feature groups with schemas designed for instruction-response pairs and multi-turn conversations.

```python
from raise_ import FeatureStore, BlobRef, Array

fs = FeatureStore("acme/posttraining/sft")

# Single-turn instruction-response pairs
ir_pairs = fs.create_feature_group("instruction-response-pairs", entity_key="example_id")
ir_pairs.create_features_from_schema({
    "example_id": "string",
    "system_prompt": "string",
    "instruction": "string",
    "response": "string",
    "source": "string",
    "quality_score": "float64",
    "include_in_training": "bool",
})

# Multi-turn conversations (turns serialized as JSON)
conversations = fs.create_feature_group("conversations", entity_key="conversation_id")
conversations.create_features_from_schema({
    "conversation_id": "string",
    "system_prompt": "string",
    "turns_json": "string",     # list of {"role": ..., "content": ...}
    "num_turns": "int64",
    "total_tokens": "int64",
    "quality_score": "float64",
    "include_in_training": "bool",
})

# Multimodal SFT — single image
image_sft = fs.create_feature_group("image-instruction-pairs", entity_key="example_id")
image_sft.create_features_from_schema({
    "example_id": "string",
    "image_ref": BlobRef(content_types=["image/jpeg", "image/png", "image/webp"]),
    "instruction": "string",
    "response": "string",
    "task_type": "string",    # "vqa" | "captioning" | "ocr" | "grounding"
    "quality_score": "float64",
    "include_in_training": "bool",
})

# Multi-image interleaved SFT
multi_image = fs.create_feature_group("multi-image-conversations", entity_key="conversation_id")
multi_image.create_features_from_schema({
    "conversation_id": "string",
    # Ordered list of image refs matching <image_N> placeholders in turns_json
    "image_refs": Array(BlobRef(content_types=["image/jpeg", "image/png"])),
    "num_images": "int64",
    "turns_json": "string",
    "total_tokens": "int64",
    "include_in_training": "bool",
})
```

#### SFT Dataset Versioning and Mixing

```python
from raise_ import DatasetVersion, DatasetMix, MixSource, MixingStrategy

sft_v2 = DatasetVersion(
    name="acme-sft-v2",
    version="sft-v2.0",
    feature_group="acme/posttraining/sft/conversations",
    num_records=1_200_000,
    metadata={"modalities": ["text", "image"]},
)

sft_mix = DatasetMix(
    name="sft-v2-mix",
    sources=[
        MixSource("acme/posttraining/sft/conversations",             weight=0.50),
        MixSource("acme/posttraining/sft/image-instruction-pairs",   weight=0.30),
        MixSource("acme/posttraining/sft/video-instruction-pairs",   weight=0.10),
        MixSource("acme/posttraining/sft/multi-image-conversations", weight=0.10),
    ],
    strategy=MixingStrategy.WEIGHTED,
    total_samples=1_200_000,
)
```

### Preference Data and DPO

Preference pairs, DPO triplets, and reward model training data are all stored as feature groups.

```python
# Preference pairs
preference_pairs = fs.create_feature_group("preference-pairs", entity_key="pair_id")
preference_pairs.create_features_from_schema({
    "pair_id": "string",
    "prompt": "string",
    "response_a": "string",
    "response_b": "string",
    "preferred": "string",         # "response_a" | "response_b" | "tie"
    "preference_confidence": "float64",
    "annotator_agreement": "float64",
    "is_gold": "bool",             # high-agreement, high-confidence pairs
    "include_in_training": "bool",
})

# DPO triplets (derived from preference pairs)
dpo_triplets = fs.create_feature_group("dpo-triplets", entity_key="triplet_id")
dpo_triplets.create_features_from_schema({
    "triplet_id": "string",
    "prompt": "string",
    "chosen": "string",
    "rejected": "string",
    # Pre-computed from the reference model (avoids overhead during DPO training loop)
    "chosen_ref_logprob": "float64",
    "rejected_ref_logprob": "float64",
    "preference_margin": "float64",
    "include_in_training": "bool",
})

# Reward model training data
reward_data = fs.create_feature_group("reward-training-data", entity_key="example_id")
reward_data.create_features_from_schema({
    "example_id": "string",
    "prompt": "string",
    "response": "string",
    "reward_score": "float64",     # normalized scalar in [0, 1]
    "bt_score": "float64",         # Bradley-Terry win probability
    "elo_score": "float64",
    "num_ratings": "int64",
    "include_in_training": "bool",
})
```

### Evaluation Suites

`EvalSuite` is a versioned collection of evaluation examples that serves as a stable benchmark across model iterations.

```python
from raise_ import EvalSuite, EvalExample, EvalMetric, BlobReference

suite = EvalSuite(
    name="instruction-following-v1",
    description="Tests model ability to follow detailed instructions",
    version="v1.0",
    metrics=[EvalMetric.ACCURACY, EvalMetric.WIN_RATE, EvalMetric.MEAN_OPINION_SCORE],
    tags=["instruction-following"],
)

# Text example
suite.add_example(EvalExample(
    example_id="if_001",
    input={"prompt": "Write a haiku about machine learning."},
    ground_truth=None,     # no single correct answer — human judgment required
    difficulty="easy",
    domain="creative-writing",
))

# Multimodal example
suite.add_example(EvalExample(
    example_id="vr_001",
    input={"image_ref": image_ref.to_dict(), "prompt": "How many objects are visible?"},
    ground_truth={"answer": "7"},
    difficulty="medium",
    domain="visual-counting",
))

# Video clip example
suite.add_example(EvalExample(
    example_id="vr_002",
    input={"video_ref": clip_ref.to_dict(), "prompt": "What action occurs between 5s and 15s?"},
    ground_truth=None,
    difficulty="hard",
    domain="temporal-reasoning",
))

# Sampling and filtering
quick_suite = suite.sample(n=50, seed=42)
chart_suite = suite.filter_by_domain("visual-counting")

print(len(suite))         # total examples
print(quick_suite)        # EvalSuite('instruction-following-v1'@v1.0, 50 examples)
```

#### EvalResult

```python
from raise_ import EvalResult

result = EvalResult(
    suite_name="instruction-following-v1",
    suite_version="v1.0",
    model_id="acme-claude-v2",
    scores={
        "accuracy": 0.823,
        "win_rate": 0.614,
        "mean_opinion_score": 3.87,
    },
    num_evaluated=200,
    human_win_rate=0.614,
)

print(result.success_rate)   # 1.0 (or less if there were failures)
```

### Human Annotation Tasks

`HumanEvalTask` defines what human annotators (or LLM judges) evaluate and how results are collected back into the feature store.

```python
from raise_ import (
    HumanEvalTask, AnnotationTaskType, AnnotatorConfig,
    AnnotatorPoolType, AgreementMetric,
    AnnotationResult, AnnotationBatch,
)

# Human preference collection
pref_task = HumanEvalTask(
    name="response-quality-preference",
    task_type=AnnotationTaskType.BINARY_PREFERENCE,
    instructions="Select the more helpful and harmless response.",
    source_feature_group="acme/posttraining/rlhf/preference-pairs",
    source_columns=["prompt", "response_a", "response_b"],
    label_column="preferred",
    options=["response_a", "response_b", "tie"],
    min_annotations=5,
    annotator_config=AnnotatorConfig(
        pool_type=AnnotatorPoolType.INTERNAL,
        min_annotations_per_item=5,
        annotator_qualifications=["passed-safety-training"],
    ),
    agreement_metric=AgreementMetric.FLEISS_KAPPA,
    consensus_strategy="majority",
)

# LLM-as-judge (high-throughput bootstrap)
llm_judge = HumanEvalTask(
    name="llm-judge-preference",
    task_type=AnnotationTaskType.BINARY_PREFERENCE,
    instructions="(LLM judge)",
    source_feature_group="acme/posttraining/rlhf/preference-pairs",
    source_columns=["prompt", "response_a", "response_b"],
    label_column="preferred",
    options=["response_a", "response_b", "tie"],
    min_annotations=1,
    annotator_config=AnnotatorConfig(
        pool_type=AnnotatorPoolType.MODEL,
        model_judge_uri="hf://acme/preference-judge-v2",
        model_judge_prompt_template=(
            "Prompt: {prompt}\n\nA: {response_a}\n\nB: {response_b}\n\n"
            "Which is better? Reply A, B, or TIE."
        ),
    ),
)

# Dispatching batches
batch = AnnotationBatch(
    batch_id="pref_batch_001",
    task_name="response-quality-preference",
    item_ids=[f"pair_{i:06d}" for i in range(500)],
    status="in_progress",
)
print(batch.total_items)       # 500
print(batch.completion_rate)   # 0.0 (not yet completed)

# Collecting results
result = AnnotationResult(
    task_name="response-quality-preference",
    item_id="pair_000001",
    annotator_id="evaluator_01",
    label="response_b",
    confidence=0.85,
    duration_sec=45.0,
)
```

#### Annotation Task Types

| Task Type | Description |
|-----------|-------------|
| `BINARY_PREFERENCE` | A vs B, pick one (RLHF pairwise) |
| `RATING` | Numeric score on a scale |
| `LIKERT` | Ordered categorical (e.g., 1–5) |
| `RANKING` | Order a list of candidates |
| `BINARY_LABEL` | Yes/No for a single item |
| `MULTI_LABEL` | Select all that apply |
| `OPEN_ENDED` | Free-text annotation |

#### Annotator Pool Types

| Pool Type | Description |
|-----------|-------------|
| `INTERNAL` | Internal ML team |
| `CROWD` | Crowdsourcing (MTurk, etc.) |
| `EXPERT` | Domain experts |
| `MODEL` | LLM-as-judge |
| `HYBRID` | LLM pre-screen + human review |

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

    # Transforms
    transforms: TransformsClient
    def create_job(self, name: str, sources: list[Source], transform: Transform, target: str, ...) -> Job
    def get_job(self, name: str) -> Job
    def list_jobs(self, status: str = None, tags: list[str] = None) -> list[Job]
    def deploy_job(self, job: Job | str) -> DeploymentResult
    def trigger_job(self, job: Job | str) -> str
```

### FeatureGroup

```python
class FeatureGroup:
    # Entity key (set at creation time via entity_key / entity_dtype)
    entity_key: str | None
    entity_dtype: str  # default: "string"

    def create_feature(self, name: str, dtype: str, **kwargs) -> Feature
    def create_features(self, features: list[dict], **kwargs) -> list[Feature]
    def create_features_from_schema(self, schema: dict, **kwargs) -> list[Feature]
    def create_features_from_file(self, path: str, **kwargs) -> list[Feature]
    def get_or_create_feature(self, name: str, dtype: str, **kwargs) -> Feature
    def feature(self, name: str) -> Feature
    def list_features(self, tags: list[str] = None, **kwargs) -> list[Feature]
    def validate_feature(self, name: str, dtype: str, **kwargs) -> ValidationResult

    # Serving (requires entity_key)
    def get(self, entity_ids: list, features: list[str] = None) -> list[dict]

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
7. **07_transformations.py** - ETL jobs, SQL/Python transforms, scheduling, Airflow integration
8. **08_multimodal.py** - Blob references, registries, integrity validation, multimodal sources
9. **09_bulk_inference.py** - Inference transforms, GPU/TPU configuration, model specifications
10. **10_video_clips.py** - Video BlobReferences, TimeRange sub-clips, clip serialization
11. **11_audio_features.py** - Audio segments, diarization-driven splits, sliding-window pre-training
12. **12_dataset_curation.py** - QualityScorer, deduplication, compliance filtering, CurationPipeline
13. **13_dataset_mixing.py** - Weighted/temperature mixing, DatasetVersion lineage, DatasetProvenance
14. **14_sft_datasets.py** - SFT schemas for text, single-image, multi-image, and video instruction-following
15. **15_preference_data.py** - Preference pairs, DPO triplets, reward model training data, LLM-as-judge
16. **16_evals_and_human_feedback.py** - EvalSuites, human annotation tasks, inter-annotator agreement
17. **17_pretraining_pipeline.py** - End-to-end pre-training pipeline: ingestion → curation → mixing → versioning
18. **18_posttraining_pipeline.py** - End-to-end post-training pipeline: SFT → RLHF → DPO → eval gating

---

## License

MIT License
