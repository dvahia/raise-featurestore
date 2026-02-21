"""
Raise Analytics Module

Analytics, dashboards, and alerting for ML features.
"""

from raise_.analytics.analysis import (
    Analysis,
    Aggregation,
    Distribution,
    Correlation,
    VersionDiff,
    StatTest,
    RecordLookup,
)
from raise_.analytics.freshness import Freshness
from raise_.analytics.result import AnalysisResult, AnalysisJob
from raise_.analytics.live_table import LiveTable, RefreshPolicy, CDCConfig
from raise_.analytics.dashboard import (
    Dashboard,
    Chart,
    ChartType,
    ChartSpec,
    DashboardParameter,
    ParameterType,
)
from raise_.analytics.alert import (
    AnalyticsAlert,
    Condition,
    AlertStatus,
    AlertEvent,
)
from raise_.analytics.client import AnalyticsClient

__all__ = [
    # Analysis types
    "Analysis",
    "Aggregation",
    "Distribution",
    "Correlation",
    "VersionDiff",
    "StatTest",
    "RecordLookup",
    # Freshness
    "Freshness",
    # Results
    "AnalysisResult",
    "AnalysisJob",
    # Live Tables
    "LiveTable",
    "RefreshPolicy",
    "CDCConfig",
    # Dashboards
    "Dashboard",
    "Chart",
    "ChartType",
    "ChartSpec",
    "DashboardParameter",
    "ParameterType",
    # Alerts
    "AnalyticsAlert",
    "Condition",
    "AlertStatus",
    "AlertEvent",
    # Client
    "AnalyticsClient",
]
