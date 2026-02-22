"""
Example 7: Data Transformations and ETL

Demonstrates creating transformation jobs that populate features from
various data sources using SQL and Python transforms.
"""

from raise_ import (
    FeatureStore,
    # Sources
    ObjectStorage,
    FileSystem,
    ColumnarSource,
    FeatureGroupSource,
    # Schedules
    Schedule,
    # Transforms
    SQLTransform,
    PythonTransform,
    # Jobs
    Job,
    Target,
    IncrementalConfig,
    # Orchestrators
    AirflowConfig,
    generate_airflow_dag,
    # Quality Checks
    NullCheck,
    RangeCheck,
    RowCountCheck,
)
from raise_.transforms import (
    ProcessingMode,
    CheckSeverity,
    python_transform,
)

# =============================================================================
# Setup
# =============================================================================

fs = FeatureStore("acme/mlplatform/recommendation")

# Create feature groups for our transformed features
user_signals = fs.create_feature_group(
    "user-signals",
    description="User engagement signals computed from events",
    if_exists="skip",
)

# Create features that will be populated by transforms
user_signals.create_features_from_schema({
    "daily_clicks": "int64",
    "daily_impressions": "int64",
    "daily_revenue": "float64",
    "click_through_rate": "float64",
    "user_segment": "string",
}, if_exists="skip")

print("Setup complete")

# =============================================================================
# SQL-Based Transformation
# =============================================================================

print("\n" + "=" * 60)
print("SQL TRANSFORMATION")
print("=" * 60)

# Define source: clickstream data in S3
clickstream_source = ObjectStorage(
    name="clickstream",
    path="s3://data-lake/events/clickstream/",
    format="parquet",
    partition_columns=["date", "hour"],
)

# Create a SQL transformation job
daily_clicks_job = fs.create_job(
    name="compute_daily_clicks",
    description="Compute daily click aggregates from clickstream",
    sources=[clickstream_source],
    transform=SQLTransform(
        name="daily_clicks_transform",
        sql="""
            SELECT
                user_id,
                DATE(event_time) as date,
                COUNT(*) as daily_clicks,
                SUM(CASE WHEN event_type = 'impression' THEN 1 ELSE 0 END) as daily_impressions,
                SUM(revenue) as daily_revenue
            FROM clickstream
            WHERE event_time >= '{{checkpoint}}'
            GROUP BY user_id, DATE(event_time)
        """,
    ),
    target=Target(
        feature_group="user-signals",
        features={
            "daily_clicks": "daily_clicks",
            "daily_impressions": "daily_impressions",
            "daily_revenue": "daily_revenue",
        },
        write_mode="upsert",
        key_columns=["user_id", "date"],
    ),
    schedule=Schedule.daily(hour=2, minute=0),
    incremental=IncrementalConfig.incremental(
        checkpoint_column="event_time",
        lookback="1h",  # Handle late-arriving data
    ),
    owner="data-team@acme.com",
    tags=["daily", "engagement"],
)

print(f"\nCreated job: {daily_clicks_job.name}")
print(f"  Status: {daily_clicks_job.status.value}")
print(f"  Schedule: {daily_clicks_job.schedule}")
print(f"  Incremental: {daily_clicks_job.incremental.mode.value}")
print(f"  Target: {daily_clicks_job.target.feature_group}")

# =============================================================================
# Python-Based Transformation
# =============================================================================

print("\n" + "=" * 60)
print("PYTHON TRANSFORMATION")
print("=" * 60)


# Define a Python transform function
@python_transform(name="segment_users", dependencies=["pandas", "numpy"])
def segment_users(context, data):
    """
    Segment users based on their engagement metrics.

    Uses business rules to assign users to segments.
    """
    import pandas as pd

    df = pd.DataFrame(data)

    # Business logic for segmentation
    def assign_segment(row):
        if row["daily_revenue"] > 100:
            return "high_value"
        elif row["daily_clicks"] > 50:
            return "engaged"
        elif row["daily_clicks"] > 10:
            return "active"
        else:
            return "casual"

    df["user_segment"] = df.apply(assign_segment, axis=1)

    # Log metrics
    context.log_metric("users_processed", len(df))
    context.log_metric("high_value_users", (df["user_segment"] == "high_value").sum())

    return df.to_dict("records")


# Create a Python transformation job
segmentation_job = fs.create_job(
    name="segment_users",
    description="Assign user segments based on engagement",
    sources=[
        FeatureGroupSource(
            name="user_metrics",
            feature_group="acme/mlplatform/recommendation/user-signals",
            features=["daily_clicks", "daily_impressions", "daily_revenue"],
        )
    ],
    transform=segment_users,  # Use the decorated function
    target=Target(
        feature_group="user-signals",
        features={"user_segment": "user_segment"},
        write_mode="upsert",
        key_columns=["user_id"],
    ),
    schedule=Schedule.daily(hour=4),  # Run after clicks job
    incremental=IncrementalConfig.full(),  # Full refresh for segmentation
    owner="ml-team@acme.com",
    tags=["segmentation", "ml"],
)

print(f"\nCreated job: {segmentation_job.name}")
print(f"  Transform type: {segmentation_job.transform.transform_type.value}")
print(f"  Dependencies: {segmentation_job.transform.dependencies}")

# =============================================================================
# CTR Computation (Derived Feature via Transform)
# =============================================================================

print("\n" + "=" * 60)
print("DERIVED FEATURE TRANSFORM")
print("=" * 60)

ctr_job = fs.create_job(
    name="compute_ctr",
    description="Compute click-through rate from clicks and impressions",
    sources=[
        FeatureGroupSource(
            feature_group="acme/mlplatform/recommendation/user-signals",
            features=["daily_clicks", "daily_impressions"],
        )
    ],
    transform=SQLTransform(
        name="ctr_transform",
        sql="""
            SELECT
                user_id,
                date,
                daily_clicks,
                daily_impressions,
                CASE
                    WHEN daily_impressions > 0
                    THEN CAST(daily_clicks AS FLOAT) / daily_impressions
                    ELSE 0.0
                END as click_through_rate
            FROM source
        """,
    ),
    target=Target(
        feature_group="user-signals",
        features={"click_through_rate": "click_through_rate"},
        write_mode="upsert",
        key_columns=["user_id", "date"],
    ),
    schedule=Schedule.hourly(minute=30),  # Update hourly
    incremental=IncrementalConfig.incremental("date"),
)

print(f"\nCreated job: {ctr_job.name}")
print(f"  Schedule: {ctr_job.schedule}")

# =============================================================================
# Different Source Types
# =============================================================================

print("\n" + "=" * 60)
print("VARIOUS SOURCE TYPES")
print("=" * 60)

# Object Storage source
s3_source = ObjectStorage(
    name="s3_events",
    path="s3://bucket/events/",
    format="parquet",
    partition_columns=["year", "month", "day"],
)
print(f"S3 Source: {s3_source.path}")

# File system source
local_source = FileSystem(
    name="local_data",
    path="/data/exports/",
    format="csv",
    glob_pattern="*.csv",
)
print(f"File Source: {local_source.path}")

# Columnar/warehouse source
warehouse_source = ColumnarSource(
    name="warehouse_table",
    catalog="analytics",
    database="production",
    table="user_events",
    columns=["user_id", "event_type", "timestamp", "value"],
    filter="timestamp >= CURRENT_DATE - INTERVAL '7 days'",
)
print(f"Warehouse Source: {warehouse_source.qualified_name}")

# Feature group source (read from other features)
feature_source = FeatureGroupSource(
    name="upstream_features",
    feature_group="acme/mlplatform/recommendation/user-signals",
    features=["daily_clicks", "daily_revenue"],
    version="v1",
)
print(f"Feature Source: {feature_source.feature_group}")

# =============================================================================
# Schedule Types
# =============================================================================

print("\n" + "=" * 60)
print("SCHEDULE TYPES")
print("=" * 60)

# Cron schedule
cron_schedule = Schedule.cron("0 */6 * * *")  # Every 6 hours
print(f"Cron: {cron_schedule}")

# Daily schedule
daily_schedule = Schedule.daily(hour=3, minute=30)
print(f"Daily: {daily_schedule}")

# Hourly schedule
hourly_schedule = Schedule.hourly(minute=15)
print(f"Hourly: {hourly_schedule}")

# Interval schedule
interval_schedule = Schedule.every("30m")
print(f"Interval: {interval_schedule}")

# On-change (CDC) schedule
cdc_schedule = Schedule.on_change(sources=["clickstream"])
print(f"CDC: {cdc_schedule}")

# Manual only
manual_schedule = Schedule.manual()
print(f"Manual: {manual_schedule}")

# =============================================================================
# Incremental Processing Modes
# =============================================================================

print("\n" + "=" * 60)
print("INCREMENTAL PROCESSING")
print("=" * 60)

# Full refresh
full_config = IncrementalConfig.full()
print(f"Full refresh: mode={full_config.mode.value}")

# Timestamp-based incremental
ts_config = IncrementalConfig.incremental(
    checkpoint_column="event_timestamp",
    lookback="1h",
)
print(f"Timestamp incremental: column={ts_config.checkpoint_column}, lookback={ts_config.lookback}")

# Append-only
append_config = IncrementalConfig.append(checkpoint_column="id")
print(f"Append-only: column={append_config.checkpoint_column}")

# Upsert with keys
upsert_config = IncrementalConfig.upsert(
    key_columns=["user_id", "date"],
    checkpoint_column="updated_at",
)
print(f"Upsert: keys={upsert_config.key_columns}")

# =============================================================================
# Job Execution
# =============================================================================

print("\n" + "=" * 60)
print("JOB EXECUTION")
print("=" * 60)

# Activate and run a job
daily_clicks_job.activate()
print(f"Job status after activation: {daily_clicks_job.status.value}")

# Execute manually
from datetime import datetime
run = daily_clicks_job.run(execution_date=datetime(2026, 2, 21))
print(f"\nRun result:")
print(f"  Run ID: {run.id}")
print(f"  Status: {run.status.value}")
print(f"  Rows read: {run.rows_read}")
print(f"  Rows written: {run.rows_written}")
print(f"  Duration: {run.duration_seconds:.2f}s")

# Get run history
runs = daily_clicks_job.get_runs(limit=5)
print(f"\nRun history: {len(runs)} runs")

# =============================================================================
# Quality Checks
# =============================================================================

print("\n" + "=" * 60)
print("QUALITY CHECKS")
print("=" * 60)

from raise_.transforms import QualityCheck, QualityReport

# Define quality checks
null_check = NullCheck(
    name="check_user_id_nulls",
    column="user_id",
    max_null_rate=0.0,  # No nulls allowed
    severity=CheckSeverity.ERROR,
)

range_check = RangeCheck(
    name="check_ctr_range",
    column="click_through_rate",
    min_value=0.0,
    max_value=1.0,
    severity=CheckSeverity.ERROR,
)

row_count_check = RowCountCheck(
    name="check_minimum_rows",
    min_rows=1000,
    severity=CheckSeverity.WARNING,
)

# Run checks (mock)
for check in [null_check, range_check, row_count_check]:
    result = check.check(None)  # Mock data
    print(f"{check.name}: {result.result.value} - {result.message}")

# =============================================================================
# Airflow DAG Generation
# =============================================================================

print("\n" + "=" * 60)
print("AIRFLOW DAG GENERATION")
print("=" * 60)

# Configure Airflow
fs.transforms.use_airflow(
    config=AirflowConfig(
        dag_folder="/opt/airflow/dags",
        catchup=False,
        max_active_runs=1,
        tags=["raise", "feature-store", "ml"],
    ),
    airflow_url="http://airflow.acme.com",
)

# Generate DAG code
dag_code = fs.transforms.generate_dag(daily_clicks_job)
print(f"Generated DAG code ({len(dag_code)} chars)")
print("\nFirst 50 lines:")
print("\n".join(dag_code.split("\n")[:50]))
print("...")

# Deploy to Airflow
result = fs.transforms.deploy(daily_clicks_job)
print(f"\nDeployment result:")
print(f"  Success: {result.success}")
print(f"  Orchestrator ID: {result.orchestrator_id}")
print(f"  URL: {result.url}")

# =============================================================================
# Lineage
# =============================================================================

print("\n" + "=" * 60)
print("LINEAGE")
print("=" * 60)

# Get job lineage
lineage = fs.transforms.get_job_lineage(daily_clicks_job)
print(f"\nJob: {lineage['job_name']}")
print(f"Sources:")
for source in lineage["sources"]:
    print(f"  - {source['source_type']}: {source['source']}")
print(f"Target:")
if lineage["target"]:
    print(f"  - Feature group: {lineage['target']['feature_group']}")
    print(f"  - Features: {lineage['target']['features']}")

# Find all jobs that produce a feature group
producers = fs.transforms.get_feature_producers("user-signals")
print(f"\nJobs producing 'user-signals': {[j.name for j in producers]}")

# Find all jobs that consume a feature group
consumers = fs.transforms.get_feature_consumers("acme/mlplatform/recommendation/user-signals")
print(f"Jobs consuming 'user-signals': {[j.name for j in consumers]}")

# =============================================================================
# Job Management
# =============================================================================

print("\n" + "=" * 60)
print("JOB MANAGEMENT")
print("=" * 60)

# List all jobs
all_jobs = fs.list_jobs()
print(f"Total jobs: {len(all_jobs)}")
for job in all_jobs:
    print(f"  - {job.name}: {job.status.value}")

# Pause a job
daily_clicks_job.pause()
print(f"\nPaused job: {daily_clicks_job.name} -> {daily_clicks_job.status.value}")

# Resume a job
daily_clicks_job.resume()
print(f"Resumed job: {daily_clicks_job.name} -> {daily_clicks_job.status.value}")

# Reset checkpoint (force full refresh)
daily_clicks_job.reset_checkpoint()
print("Reset checkpoint for full refresh")

print("\n" + "=" * 60)
print("ALL TRANSFORMATION EXAMPLES COMPLETE!")
print("=" * 60)
