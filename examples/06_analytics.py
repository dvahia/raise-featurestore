"""
Example 6: Feature Analytics

Demonstrates running analyses on features, creating dashboards,
setting up alerts, and using live tables.
"""

from raise_ import (
    FeatureStore,
    Aggregation,
    Distribution,
    Correlation,
    VersionDiff,
    StatTest,
    RecordLookup,
    Freshness,
    Dashboard,
    Chart,
    ChartType,
    DashboardParameter,
    ParameterType,
    Condition,
)

# =============================================================================
# Setup
# =============================================================================

fs = FeatureStore("acme/mlplatform/recommendation")

# Create feature group with sample features
user_signals = fs.create_feature_group(
    "user-signals",
    description="User engagement signals",
    if_exists="skip",
)

user_signals.create_features_from_schema({
    "click_count": "int64",
    "impression_count": "int64",
    "revenue": "float64",
    "user_embedding": "float32[512]",
    "user_tier": "string",
    "experiment_group": "string",
    "conversion_rate": "float64",
}, if_exists="skip")

print("Setup complete")

# =============================================================================
# Aggregation Analysis
# =============================================================================

print("\n" + "=" * 60)
print("AGGREGATION ANALYSIS")
print("=" * 60)

# Simple aggregation
result = user_signals.analyze(
    Aggregation(
        feature="click_count",
        metrics=["count", "sum", "avg", "min", "max", "null_rate"],
    )
)

print(f"\nClick count metrics:")
for metric, value in result.metrics.items():
    print(f"  {metric}: {value}")

# Time-windowed aggregation
result = user_signals.analyze(
    Aggregation(
        feature="revenue",
        metrics=["sum", "avg"],
        window="7d",
        group_by="user_tier",
    )
)

print(f"\nRevenue by tier (last 7 days):")
print(f"  Metrics: {result.metrics}")

# Rolling aggregation
result = user_signals.analyze(
    Aggregation(
        feature="click_count",
        metrics=["sum"],
        window="1d",
        rolling=True,
        periods=30,
    )
)

print(f"\n30-day rolling click sum computed")

# =============================================================================
# Distribution Analysis
# =============================================================================

print("\n" + "=" * 60)
print("DISTRIBUTION ANALYSIS")
print("=" * 60)

result = user_signals.analyze(
    Distribution(
        feature="revenue",
        metrics=["histogram", "percentiles"],
        bins=50,
    )
)

print(f"\nRevenue distribution:")
print(f"  Histogram bins: {len(result.histogram.get('bin_edges', []))}")
print(f"  Percentiles: {result.percentiles}")

# Segmented distribution
result = user_signals.analyze(
    Distribution(
        feature="revenue",
        segment_by="user_tier",
        metrics=["histogram", "percentiles"],
    )
)

print(f"\nRevenue distribution by tier computed")

# =============================================================================
# Correlation Analysis
# =============================================================================

print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

result = user_signals.analyze(
    Correlation(
        features=["click_count", "impression_count", "revenue"],
        method="pearson",
    )
)

print(f"\nCorrelation matrix:")
print(f"  Features: {result.data.get('features')}")
print(f"  Matrix shape: {len(result.correlation_matrix)}x{len(result.correlation_matrix[0]) if result.correlation_matrix else 0}")

# Convert to dataframe for better visualization
df = result.to_dataframe()
print(f"\n{df}")

# =============================================================================
# Version Diff
# =============================================================================

print("\n" + "=" * 60)
print("VERSION DIFF")
print("=" * 60)

# Create a new version of user_embedding
user_signals.create_feature(
    "user_embedding",
    dtype="float32[768]",
    version="v2",
    if_exists="skip",
)

result = user_signals.analyze(
    VersionDiff(
        feature="user_embedding",
        version_a="v1",
        version_b="v2",
        compare=["schema", "distribution"],
    )
)

print(f"\nVersion diff for user_embedding:")
print(f"  Schema changes: {result.schema_changes}")
print(f"  Distribution drift: {result.distribution_drift}")

# =============================================================================
# Statistical Testing
# =============================================================================

print("\n" + "=" * 60)
print("STATISTICAL TESTING")
print("=" * 60)

result = user_signals.analyze(
    StatTest(
        feature="conversion_rate",
        test="ttest",
        segment_by="experiment_group",
        control="control",
        treatment="variant_a",
    )
)

print(f"\nA/B test results:")
print(f"  P-value: {result.p_value}")
print(f"  Effect size: {result.effect_size}")
print(f"  Confidence interval: {result.confidence_interval}")
print(f"  Significant: {result.p_value < 0.05 if result.p_value else 'N/A'}")

# =============================================================================
# Record Lookup
# =============================================================================

print("\n" + "=" * 60)
print("RECORD LOOKUP")
print("=" * 60)

# Sample records for inspection
result = user_signals.analyze(
    RecordLookup(
        features=["click_count", "revenue", "user_tier"],
        sample=10,
    )
)

print(f"\nSample records:")
for record in result.records[:5]:
    print(f"  {record}")

# Filter specific records
result = user_signals.analyze(
    RecordLookup(
        features=["click_count", "revenue"],
        filter="user_tier = 'gold'",
        limit=5,
    )
)

print(f"\nFiltered records (gold tier): {len(result.records)} records")

# =============================================================================
# Freshness Control
# =============================================================================

print("\n" + "=" * 60)
print("FRESHNESS CONTROL")
print("=" * 60)

# Real-time (always compute fresh)
result = user_signals.analyze(
    Aggregation(feature="click_count", metrics=["sum"]),
    freshness=Freshness.REAL_TIME,
)
print(f"Real-time result: {result.metrics}")

# Within 1 hour (use cache if fresh enough)
result = user_signals.analyze(
    Aggregation(feature="click_count", metrics=["sum"]),
    freshness=Freshness.WITHIN("1h"),
)
print(f"Within 1h result: {result.metrics}")

# Always use cache
result = user_signals.analyze(
    Aggregation(feature="click_count", metrics=["sum"]),
    freshness=Freshness.CACHED,
)
print(f"Cached result: {result.metrics}")

# =============================================================================
# Live Tables (CDC-based auto-refresh)
# =============================================================================

print("\n" + "=" * 60)
print("LIVE TABLES")
print("=" * 60)

# Create a live table with CDC refresh
daily_metrics = user_signals.create_live_table(
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

print(f"\nCreated live table: {daily_metrics.name}")
print(f"  Status: {daily_metrics.status}")
print(f"  Refresh policy: {daily_metrics.refresh_policy.type}")
print(f"  Last refresh: {daily_metrics.last_refresh}")

# Query the live table
df = daily_metrics.query()
print(f"\nLive table data:\n{df}")

# Query with filter
df = daily_metrics.query(filter="user_tier = 'gold'")
print(f"\nFiltered data (gold):\n{df}")

# Manual refresh
event = daily_metrics.refresh()
print(f"\nManual refresh: {event.status}")

# Get refresh history
history = daily_metrics.refresh_history(limit=5)
print(f"Refresh history: {len(history)} events")

# =============================================================================
# Dashboards
# =============================================================================

print("\n" + "=" * 60)
print("DASHBOARDS")
print("=" * 60)

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

# Add chart linked to live table
dashboard.add_chart(
    Chart(
        title="Live Engagement Metrics",
        live_table="daily_engagement",
        chart_type=ChartType.TABLE,
    )
)

print(f"\nDashboard created: {dashboard.name}")
print(f"  Charts: {len(dashboard.charts)}")
print(f"  Parameters: {len(dashboard.parameters)}")

# Render as JSON spec
spec = dashboard.render(format="json")
print(f"\nDashboard spec (truncated):\n{spec[:500]}...")

# Publish dashboard
url = dashboard.publish()
print(f"\nPublished at: {url}")

# =============================================================================
# Analytics Alerts
# =============================================================================

print("\n" + "=" * 60)
print("ANALYTICS ALERTS")
print("=" * 60)

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

print(f"\nCreated alert: {null_alert.name}")
print(f"  Condition: {null_alert.condition}")
print(f"  Notify: {null_alert.notify}")

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

print(f"\nCreated drift alert: {drift_alert.name}")

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

print(f"\nCreated significance alert: {significance_alert.name}")

# List all alerts
alerts = fs.list_alerts()
print(f"\nAll alerts: {len(alerts)}")
for alert in alerts:
    print(f"  - {alert.name}: {alert.status}")

# =============================================================================
# Async Analysis
# =============================================================================

print("\n" + "=" * 60)
print("ASYNC ANALYSIS")
print("=" * 60)

# Submit async job for large analysis
job = fs.analyze_async(
    Correlation(
        features=["click_count", "impression_count", "revenue", "conversion_rate"],
        method="spearman",
    )
)

print(f"\nSubmitted job: {job.id}")
print(f"  Status: {job.status}")

# Wait for completion
result = job.wait(timeout=60)
print(f"  Completed: {job.status}")
print(f"  Result: {result.to_dict()['data']['features']}")

print("\n" + "=" * 60)
print("ALL ANALYTICS EXAMPLES COMPLETE!")
print("=" * 60)
