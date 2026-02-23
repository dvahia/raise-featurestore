# Feature Store API Comparison

A comparative analysis of Raise against existing feature store solutions across ease-of-use, developer velocity, and infrastructure efficiency.

## Executive Summary

| Dimension | Raise | Feast | Tecton | Databricks | SageMaker | Vertex AI |
|-----------|-------|-------|--------|------------|-----------|-----------|
| **Ease of Use** | High | Medium | Medium | Medium | Low | Medium |
| **Developer Velocity** | High | Medium | Medium | Medium | Low | Low |
| **Declarative API** | Yes | Partial | No | Partial | No | No |
| **Notebook-First** | Yes | No | No | Yes | Partial | Yes |
| **Derived Features** | SQL inline | Python decorator | Python/SQL | Spark SQL | Manual | Manual |
| **Lines of Code** | Low | Medium | Medium | Medium | High | High |
| **Learning Curve** | Low | Medium | High | Medium | High | High |

**Verdict**: Raise offers meaningful improvements in API ergonomics for notebook-based research workflows. However, it currently lacks production-grade infrastructure that mature solutions provide.

---

## Detailed Comparison

### 1. Feature Definition

#### Raise
```python
from raise_ import FeatureStore

fs = FeatureStore("acme/mlplatform/recommendation")
group = fs.create_feature_group("user-signals")

# Single line per feature
group.create_feature("clicks", dtype="int64")
group.create_feature("impressions", dtype="int64")
group.create_feature("ctr", dtype="float64", derived_from="clicks / NULLIF(impressions, 0)")

# Or bulk from schema
group.create_features_from_schema({
    "clicks": "int64",
    "impressions": "int64",
    "user_embedding": "float32[512]",
})
```
**Lines of code**: 6-10
**Concepts to learn**: 3 (FeatureStore, FeatureGroup, Feature)

#### Feast
```python
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, Project
from feast.types import Float32, Int64

# Must define project
project = Project(name="my_project", description="...")

# Must define entity separately
driver = Entity(name="driver", join_keys=["driver_id"])

# Must define data source separately
driver_stats_source = FileSource(
    name="driver_hourly_stats_source",
    path="/path/to/data.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

# Then define feature view referencing all above
driver_stats_fv = FeatureView(
    name="driver_hourly_stats",
    entities=[driver],
    ttl=timedelta(days=1),
    schema=[
        Field(name="conv_rate", dtype=Float32),
        Field(name="acc_rate", dtype=Float32),
        Field(name="avg_daily_trips", dtype=Int64),
    ],
    online=True,
    source=driver_stats_source,
    tags={"team": "driver_performance"},
)

# Must run CLI or apply() to register
# feast apply
```
**Lines of code**: 25-30
**Concepts to learn**: 6 (Project, Entity, FileSource, FeatureView, Field, types)

#### Databricks Feature Store
```python
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# Must have a Spark DataFrame first
# df = spark.read.table("...")

# Create feature table from DataFrame (not declarative)
fe.create_table(
    name="ml.recommendation.user_signals",
    primary_keys=["user_id"],
    df=df,  # Must have data already
    description="User engagement signals",
)

# For derived features, use Spark SQL separately
# spark.sql("CREATE TABLE ... AS SELECT clicks/impressions as ctr FROM ...")
```
**Lines of code**: 10-15
**Concepts to learn**: 4 (FeatureEngineeringClient, DataFrame, primary_keys, Unity Catalog paths)

#### SageMaker Feature Store
```python
import boto3
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_definition import FeatureDefinition, FeatureTypeEnum

session = sagemaker.Session()
region = boto3.Session().region_name
role = sagemaker.get_execution_role()

# Define feature definitions manually
feature_definitions = [
    FeatureDefinition(feature_name="user_id", feature_type=FeatureTypeEnum.STRING),
    FeatureDefinition(feature_name="clicks", feature_type=FeatureTypeEnum.INTEGRAL),
    FeatureDefinition(feature_name="impressions", feature_type=FeatureTypeEnum.INTEGRAL),
    # No derived features - must compute externally
]

# Create feature group
feature_group = FeatureGroup(
    name="user-signals",
    sagemaker_session=session,
    feature_definitions=feature_definitions,
)

# Must create explicitly with S3 location
feature_group.create(
    s3_uri=f"s3://{bucket}/feature-store/",
    record_identifier_name="user_id",
    event_time_feature_name="event_time",
    role_arn=role,
    enable_online_store=True,
)
```
**Lines of code**: 25-30
**Concepts to learn**: 8+ (boto3, sagemaker session, roles, FeatureGroup, FeatureDefinition, S3, event_time, online store)

### 2. Derived/Computed Features

#### Raise - Inline SQL (Simplest)
```python
# Derived feature in one line
group.create_feature(
    "ctr",
    dtype="float64",
    derived_from="clicks / NULLIF(impressions, 0)",
)

# Complex derived feature
group.create_feature(
    "engagement_score",
    dtype="float64",
    derived_from="""
        (clicks * 1.0 + shares * 2.0 + comments * 3.0) /
        NULLIF(impressions, 0) * 100
    """,
)
```

#### Feast - Python Decorator
```python
from feast.on_demand_feature_view import on_demand_feature_view
import pandas as pd

@on_demand_feature_view(
    sources=[driver_stats_fv],  # Must reference existing feature view
    schema=[Field(name="ctr", dtype=Float64)],
)
def computed_ctr(inputs: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df["ctr"] = inputs["clicks"] / inputs["impressions"].replace(0, float('nan'))
    return df
```

#### Databricks - Separate Spark SQL
```python
# No inline derived features - must create separate table
spark.sql("""
    CREATE OR REPLACE TABLE ml.recommendation.user_signals_derived AS
    SELECT
        *,
        clicks / NULLIF(impressions, 0) as ctr
    FROM ml.recommendation.user_signals
""")

# Then register as feature table
fe.create_table(
    name="ml.recommendation.user_signals_derived",
    primary_keys=["user_id"],
    df=spark.table("ml.recommendation.user_signals_derived"),
)
```

#### SageMaker - Manual Pre-computation
```python
# No derived features - must compute before ingestion
df["ctr"] = df["clicks"] / df["impressions"].replace(0, float('nan'))

# Then ingest the pre-computed DataFrame
feature_group.ingest(data_frame=df, max_workers=3, wait=True)
```

### 3. Bulk Operations

#### Raise
```python
# From schema dict
group.create_features_from_schema({
    "clicks": "int64",
    "impressions": "int64",
    "ctr": "float64",
})

# From YAML file
group.create_features_from_file("features.yaml")

# From list with full specs
group.create_features([
    {"name": "clicks", "dtype": "int64", "description": "Total clicks"},
    {"name": "ctr", "dtype": "float64", "derived_from": "clicks/impressions"},
])
```

#### Feast
```python
# No bulk API - must define each FeatureView with schema list
# Then run feast apply to register all at once
```

#### Databricks
```python
# Bulk via DataFrame schema inference
# But no declarative bulk creation
```

#### SageMaker
```python
# Must define FeatureDefinition list manually
# No schema inference or bulk helpers
```

### 4. Analytics on Features

#### Raise - Built-in Analytics
```python
from raise_ import Aggregation, Distribution, Condition

# Declarative analytics
result = group.analyze(
    Aggregation(feature="ctr", metrics=["avg", "p50", "p99", "null_rate"])
)

# Distribution analysis
dist = group.analyze(Distribution(feature="clicks", bins=50))

# Create alert
fs.create_alert(
    name="ctr-drift",
    analysis=Aggregation(feature="ctr", metrics=["avg"]),
    condition=Condition.LESS_THAN(0.01),
    notify=["team@example.com"],
)

# Live tables (materialized views)
group.create_live_table(
    name="daily_metrics",
    analysis=Aggregation(feature="clicks", metrics=["sum"], group_by=["date"]),
    refresh="hourly",
)
```

#### Feast
```python
# No built-in analytics
# Must export data and use external tools
training_df = store.get_historical_features(...).to_df()
# Then use pandas/spark for analytics
```

#### Databricks
```python
# Use Spark SQL for analytics (not integrated with feature store)
spark.sql("SELECT AVG(ctr), PERCENTILE(ctr, 0.99) FROM feature_table")
```

#### SageMaker
```python
# No built-in analytics
# Must use Athena or other AWS services
```

### 5. Inference Pipelines

#### Raise - Declarative Inference
```python
from raise_ import embedding_inference, GPUType, Schedule

job = fs.create_job(
    name="generate_embeddings",
    sources=[FeatureGroupSource(group="user-profiles")],
    transform=embedding_inference(
        model_uri="hf://sentence-transformers/all-MiniLM-L6-v2",
        input_column="bio",
        output_column="bio_embedding",
        batch_size=256,
        gpu_type=GPUType.NVIDIA_T4,
    ),
    target=Target(feature_group="user-embeddings"),
    schedule=Schedule.daily(hour=2),
)
```

#### Feast
```python
# No built-in inference support
# Must use external orchestration (Airflow, etc.)
```

#### Tecton
```python
# Inference via feature services, not declarative transforms
# Must deploy model separately
```

### 6. End-to-End Example Comparison

**Task**: Create a feature group with user engagement metrics, a derived CTR feature, and set up hourly aggregation monitoring.

#### Raise (15 lines)
```python
from raise_ import FeatureStore, Aggregation, Schedule, Condition

fs = FeatureStore("acme/ml/engagement")
group = fs.create_feature_group("user-signals")

group.create_features_from_schema({
    "clicks": "int64",
    "impressions": "int64",
})
group.create_feature("ctr", dtype="float64", derived_from="clicks / NULLIF(impressions, 0)")

group.create_live_table("hourly_metrics",
    analysis=Aggregation(feature="ctr", metrics=["avg", "p99"]),
    refresh="hourly")

fs.create_alert("ctr-drop", Aggregation(feature="ctr", metrics=["avg"]),
    Condition.LESS_THAN(0.01), notify=["team@example.com"])
```

#### Feast (40+ lines)
```python
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, Project, FeatureStore
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float64, Int64
import pandas as pd

# definitions.py
project = Project(name="engagement")
user = Entity(name="user", join_keys=["user_id"])

source = FileSource(
    name="user_signals_source",
    path="/path/to/data.parquet",
    timestamp_field="event_timestamp",
)

user_signals_fv = FeatureView(
    name="user_signals",
    entities=[user],
    ttl=timedelta(days=1),
    schema=[
        Field(name="clicks", dtype=Int64),
        Field(name="impressions", dtype=Int64),
    ],
    source=source,
)

@on_demand_feature_view(
    sources=[user_signals_fv],
    schema=[Field(name="ctr", dtype=Float64)],
)
def ctr_features(inputs: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df["ctr"] = inputs["clicks"] / inputs["impressions"].replace(0, float('nan'))
    return df

# Run: feast apply

# For monitoring - must use external tools (Great Expectations, etc.)
# For alerts - must set up separately (CloudWatch, PagerDuty, etc.)
# For hourly aggregation - must set up Airflow DAG separately
```

#### SageMaker (60+ lines)
```python
import boto3
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_definition import FeatureDefinition, FeatureTypeEnum

# Setup
session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = session.default_bucket()

# Feature definitions
feature_definitions = [
    FeatureDefinition(feature_name="user_id", feature_type=FeatureTypeEnum.STRING),
    FeatureDefinition(feature_name="event_time", feature_type=FeatureTypeEnum.STRING),
    FeatureDefinition(feature_name="clicks", feature_type=FeatureTypeEnum.INTEGRAL),
    FeatureDefinition(feature_name="impressions", feature_type=FeatureTypeEnum.INTEGRAL),
    FeatureDefinition(feature_name="ctr", feature_type=FeatureTypeEnum.FRACTIONAL),  # Pre-computed
]

# Create feature group
feature_group = FeatureGroup(
    name="user-signals",
    sagemaker_session=session,
    feature_definitions=feature_definitions,
)

feature_group.create(
    s3_uri=f"s3://{bucket}/feature-store/",
    record_identifier_name="user_id",
    event_time_feature_name="event_time",
    role_arn=role,
    enable_online_store=True,
)

# Pre-compute CTR before ingestion
df["ctr"] = df["clicks"] / df["impressions"].replace(0, float('nan'))
feature_group.ingest(data_frame=df, max_workers=3, wait=True)

# For monitoring - set up CloudWatch separately
# For alerts - set up CloudWatch Alarms separately
# For aggregation - set up Athena queries + EventBridge separately
```

---

## Quantitative Comparison

### Lines of Code for Common Tasks

| Task | Raise | Feast | Databricks | SageMaker |
|------|-------|-------|------------|-----------|
| Define 5 features | 5 | 15 | 10 | 20 |
| Add derived feature | 1 | 10 | 8 | N/A* |
| Bulk create from schema | 3 | N/A | N/A | N/A |
| Set up analytics | 3 | N/A | 5** | N/A |
| Create alert | 4 | N/A | N/A | 15*** |
| Define inference job | 10 | N/A | 15 | 20 |

*SageMaker requires pre-computation
**Databricks uses Spark SQL, not integrated
***SageMaker uses CloudWatch, not integrated

### Concepts to Learn

| Platform | Core Concepts | Additional Concepts |
|----------|---------------|---------------------|
| Raise | 4 | +2 for transforms |
| Feast | 6 | +4 for production |
| Databricks | 5 | +3 for Unity Catalog |
| SageMaker | 8 | +5 for AWS services |
| Vertex AI | 6 | +4 for GCP services |

---

## Where Raise Falls Short

### 1. Production Infrastructure
- **Feast/Tecton**: Battle-tested serving infrastructure, proven at scale
- **Raise**: Mock implementation, no actual storage backend

### 2. Online Feature Serving
- **Tecton**: Sub-10ms p99 latency at 100K QPS
- **Feast**: Redis/DynamoDB backends with low latency
- **Raise**: No online serving implementation

### 3. Enterprise Features
- **Databricks**: Unity Catalog governance, lineage, audit
- **SageMaker**: AWS IAM integration, VPC support
- **Raise**: Mock ACL, no actual auth integration

### 4. Ecosystem Integration
- **Feast**: Spark, Snowflake, BigQuery, Redshift connectors
- **Tecton**: Native Databricks, Snowflake, EMR integration
- **Raise**: No actual data source connectors

### 5. Proven at Scale
- **Feast**: Used at Gojek, Twitter, Salesforce
- **Tecton**: Used at Atlassian, Coinbase
- **Raise**: No production deployments

---

## Conclusions

### Raise is Better For:
1. **Rapid prototyping** in notebook environments
2. **Teaching/learning** feature store concepts
3. **Declarative feature definitions** without boilerplate
4. **Inline derived features** with SQL expressions
5. **Integrated analytics** without external tools
6. **Inference pipelines** with GPU/TPU support (API design)

### Raise is Worse For:
1. **Production deployments** (no infrastructure)
2. **Low-latency online serving** (not implemented)
3. **Enterprise governance** (mock implementation)
4. **Large-scale data** (no actual storage)
5. **Ecosystem integration** (no connectors)

### Recommendation

Raise's API design demonstrates meaningful ergonomic improvements for researcher workflows. The value proposition is strongest when:

1. **Target users are researchers**, not ML engineers
2. **Notebook-first** workflow is primary
3. **Speed of iteration** matters more than production scale
4. **Integrated analytics** reduces tool sprawl

However, without production infrastructure, Raise is currently a **design prototype** rather than a **deployable solution**. The API patterns could be:

1. **Adopted by existing feature stores** as a higher-level SDK
2. **Implemented on top of Feast** as a compatibility layer
3. **Built out with real backends** for production use

---

## Sources

- [Feast Documentation](https://docs.feast.dev/)
- [Feast Quickstart](https://docs.feast.dev/getting-started/quickstart)
- [Tecton Documentation](https://docs.tecton.ai/)
- [Tecton Feature Store](https://www.tecton.ai/feature-store/)
- [Databricks Feature Store](https://docs.databricks.com/aws/en/machine-learning/feature-store/)
- [Databricks Unity Catalog Feature Tables](https://docs.databricks.com/aws/en/machine-learning/feature-store/uc/feature-tables-uc)
- [AWS SageMaker Feature Store](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_featurestore.html)
- [SageMaker Feature Store Introduction](https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store-getting-started.html)
- [Vertex AI Feature Store](https://docs.cloud.google.com/vertex-ai/docs/featurestore/latest/overview)
- [Top 5 Feature Stores in 2025](https://www.gocodeo.com/post/top-5-feature-stores-in-2025-tecton-feast-and-beyond)
