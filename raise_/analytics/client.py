"""
Raise Analytics Client

Main client for analytics operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, TYPE_CHECKING

from raise_.analytics.analysis import Analysis
from raise_.analytics.freshness import Freshness
from raise_.analytics.result import AnalysisResult, AnalysisJob, generate_id
from raise_.analytics.live_table import LiveTable, RefreshPolicy
from raise_.analytics.dashboard import Dashboard
from raise_.analytics.alert import AnalyticsAlert, Condition


@dataclass
class AnalyticsClient:
    """
    Client for analytics operations.

    Provides methods for running analyses, managing live tables,
    dashboards, and alerts.

    Attributes:
        feature_store: Parent FeatureStore instance.
    """

    _feature_store: Any = field(repr=False)

    # Internal state
    _results: dict[str, AnalysisResult] = field(default_factory=dict, repr=False)
    _jobs: dict[str, AnalysisJob] = field(default_factory=dict, repr=False)
    _live_tables: dict[str, LiveTable] = field(default_factory=dict, repr=False)
    _dashboards: dict[str, Dashboard] = field(default_factory=dict, repr=False)
    _alerts: dict[str, AnalyticsAlert] = field(default_factory=dict, repr=False)

    # =========================================================================
    # Analysis Execution
    # =========================================================================

    def analyze(
        self,
        analysis: Analysis,
        freshness: Freshness = None,
    ) -> AnalysisResult:
        """
        Run an analysis and return the result.

        Args:
            analysis: The analysis to run.
            freshness: Freshness requirements (default: WITHIN 1h).

        Returns:
            AnalysisResult with the computed data.
        """
        if freshness is None:
            freshness = Freshness.WITHIN("1h")

        # Check cache if freshness allows
        cache_key = self._cache_key(analysis)
        if cache_key in self._results and freshness.policy != "real_time":
            cached = self._results[cache_key]
            age = datetime.now() - cached.created_at
            if freshness.accepts_age(age):
                return cached

        # Execute analysis (in production, this calls the backend)
        result = self._execute_analysis(analysis)

        # Cache result
        self._results[cache_key] = result
        self._results[result.id] = result

        return result

    def analyze_async(self, analysis: Analysis) -> AnalysisJob:
        """
        Submit an analysis for asynchronous execution.

        Args:
            analysis: The analysis to run.

        Returns:
            AnalysisJob for tracking progress.
        """
        job = AnalysisJob(
            id=generate_id("job"),
            analysis=analysis,
            status="pending",
            _client=self,
        )

        self._jobs[job.id] = job

        # In production, this would submit to a job queue
        # For now, we'll simulate immediate completion
        job.status = "running"
        job.started_at = datetime.now()

        try:
            result = self._execute_analysis(analysis)
            job.status = "completed"
            job.completed_at = datetime.now()
            job.result_id = result.id
            self._results[result.id] = result
        except Exception as e:
            job.status = "failed"
            job.error = str(e)

        return job

    def _execute_analysis(self, analysis: Analysis) -> AnalysisResult:
        """Execute an analysis (mock implementation)."""
        result_id = generate_id("result")

        # Generate mock data based on analysis type
        data = self._generate_mock_data(analysis)

        return AnalysisResult(
            id=result_id,
            analysis=analysis,
            data=data,
            created_at=datetime.now(),
            freshness_at=datetime.now(),
        )

    def _generate_mock_data(self, analysis: Analysis) -> dict[str, Any]:
        """Generate mock result data for an analysis."""
        analysis_type = analysis.analysis_type

        if analysis_type == "aggregation":
            return {
                "metrics": {
                    "count": 10000,
                    "sum": 150000,
                    "avg": 15.0,
                    "min": 0,
                    "max": 100,
                    "null_rate": 0.02,
                }
            }
        elif analysis_type == "distribution":
            return {
                "histogram": {
                    "bin_edges": [0, 10, 20, 30, 40, 50],
                    "counts": [100, 250, 400, 180, 70],
                },
                "percentiles": {
                    "p50": 22.5,
                    "p75": 35.0,
                    "p90": 42.0,
                    "p95": 47.0,
                    "p99": 55.0,
                },
            }
        elif analysis_type == "correlation":
            n = len(analysis.features)
            return {
                "features": analysis.features,
                "correlation_matrix": [[1.0 if i == j else 0.5 for j in range(n)] for i in range(n)],
            }
        elif analysis_type == "version_diff":
            return {
                "schema_changes": {
                    "dtype": {"old": "float32[512]", "new": "float32[768]"},
                },
                "distribution_drift": {
                    "psi": 0.15,
                    "kl_divergence": 0.08,
                },
            }
        elif analysis_type == "stat_test":
            return {
                "p_value": 0.023,
                "effect_size": 0.15,
                "confidence_interval": [0.05, 0.25],
                "test_statistic": 2.45,
            }
        elif analysis_type == "record_lookup":
            return {
                "records": [
                    {"id": f"record_{i}", "value": i * 10}
                    for i in range(min(analysis.limit, 10))
                ]
            }
        else:
            return {}

    def _cache_key(self, analysis: Analysis) -> str:
        """Generate a cache key for an analysis."""
        import json
        spec = json.dumps(analysis.to_dict(), sort_keys=True)
        import hashlib
        return hashlib.md5(spec.encode()).hexdigest()

    # =========================================================================
    # Job Management
    # =========================================================================

    def get_job(self, job_id: str) -> AnalysisJob:
        """Get a job by ID."""
        if job_id not in self._jobs:
            raise ValueError(f"Job not found: {job_id}")
        return self._jobs[job_id]

    def list_jobs(
        self,
        status: str | None = None,
        limit: int = 100,
    ) -> list[AnalysisJob]:
        """List analysis jobs."""
        jobs = list(self._jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]

        return sorted(jobs, key=lambda j: j.created_at, reverse=True)[:limit]

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        if job_id in self._jobs:
            job = self._jobs[job_id]
            if job.status in ("pending", "running"):
                job.status = "cancelled"
                return True
        return False

    # =========================================================================
    # Result Management
    # =========================================================================

    def get_result(self, result_id: str) -> AnalysisResult:
        """Get a result by ID."""
        if result_id not in self._results:
            raise ValueError(f"Result not found: {result_id}")
        return self._results[result_id]

    def list_results(
        self,
        feature: str | None = None,
        analysis_type: str | None = None,
        limit: int = 100,
    ) -> list[AnalysisResult]:
        """List analysis results."""
        results = list(self._results.values())

        if analysis_type:
            results = [r for r in results if r.analysis.analysis_type == analysis_type]

        return sorted(results, key=lambda r: r.created_at, reverse=True)[:limit]

    def delete_result(self, result_id: str) -> bool:
        """Delete a result."""
        if result_id in self._results:
            del self._results[result_id]
            return True
        return False

    def delete_results(self, older_than: str | timedelta) -> int:
        """Delete results older than the specified age."""
        if isinstance(older_than, str):
            older_than = Freshness._parse_duration(older_than)

        cutoff = datetime.now() - older_than
        to_delete = [
            r_id for r_id, r in self._results.items()
            if r.created_at < cutoff
        ]

        for r_id in to_delete:
            del self._results[r_id]

        return len(to_delete)

    # =========================================================================
    # Live Tables
    # =========================================================================

    def create_live_table(
        self,
        name: str,
        analysis: Analysis,
        refresh: str | RefreshPolicy = "on_change",
        description: str | None = None,
    ) -> LiveTable:
        """
        Create a live table.

        Args:
            name: Unique name for the live table.
            analysis: Analysis to materialize.
            refresh: Refresh policy ("on_change", "hourly", "daily", or RefreshPolicy).
            description: Description.

        Returns:
            The created LiveTable.
        """
        if isinstance(refresh, str):
            if refresh == "on_change":
                refresh_policy = RefreshPolicy.on_change()
            elif refresh == "hourly":
                refresh_policy = RefreshPolicy.hourly()
            elif refresh == "daily":
                refresh_policy = RefreshPolicy.daily()
            elif refresh == "manual":
                refresh_policy = RefreshPolicy.manual()
            else:
                raise ValueError(f"Unknown refresh policy: {refresh}")
        else:
            refresh_policy = refresh

        live_table = LiveTable(
            name=name,
            analysis=analysis,
            refresh_policy=refresh_policy,
            description=description,
            _client=self,
        )

        # Initial data load
        result = self._execute_analysis(analysis)
        if analysis.analysis_type == "aggregation":
            live_table._data = [result.data.get("metrics", {})]
        elif analysis.analysis_type == "record_lookup":
            live_table._data = result.data.get("records", [])
        else:
            live_table._data = [result.data]

        live_table.row_count = len(live_table._data)
        live_table.last_refresh = datetime.now()

        self._live_tables[name] = live_table
        return live_table

    def get_live_table(self, name: str) -> LiveTable:
        """Get a live table by name."""
        if name not in self._live_tables:
            raise ValueError(f"Live table not found: {name}")
        return self._live_tables[name]

    def list_live_tables(self) -> list[LiveTable]:
        """List all live tables."""
        return list(self._live_tables.values())

    def delete_live_table(self, name: str) -> bool:
        """Delete a live table."""
        if name in self._live_tables:
            del self._live_tables[name]
            return True
        return False

    # =========================================================================
    # Dashboards
    # =========================================================================

    def create_dashboard(
        self,
        name: str,
        description: str | None = None,
    ) -> Dashboard:
        """
        Create a dashboard.

        Args:
            name: Dashboard name.
            description: Description.

        Returns:
            The created Dashboard.
        """
        dashboard = Dashboard(
            name=name,
            description=description,
            _client=self,
        )
        self._dashboards[name] = dashboard
        return dashboard

    def get_dashboard(self, name: str) -> Dashboard:
        """Get a dashboard by name."""
        if name not in self._dashboards:
            raise ValueError(f"Dashboard not found: {name}")
        return self._dashboards[name]

    def list_dashboards(self) -> list[Dashboard]:
        """List all dashboards."""
        return list(self._dashboards.values())

    def delete_dashboard(self, name: str) -> bool:
        """Delete a dashboard."""
        if name in self._dashboards:
            del self._dashboards[name]
            return True
        return False

    # =========================================================================
    # Alerts
    # =========================================================================

    def create_alert(
        self,
        name: str,
        analysis: Analysis,
        condition: Condition,
        notify: list[str],
        channels: list[str] | None = None,
        check_interval: str = "1h",
    ) -> AnalyticsAlert:
        """
        Create an analytics alert.

        Args:
            name: Alert name.
            analysis: Analysis to monitor.
            condition: Condition that triggers the alert.
            notify: Notification recipients.
            channels: Notification channels.
            check_interval: How often to check.

        Returns:
            The created AnalyticsAlert.
        """
        alert = AnalyticsAlert(
            name=name,
            analysis=analysis,
            condition=condition,
            notify=notify,
            channels=channels or ["email"],
            check_interval=check_interval,
            _client=self,
        )
        self._alerts[name] = alert
        return alert

    def get_alert(self, name: str) -> AnalyticsAlert:
        """Get an alert by name."""
        if name not in self._alerts:
            raise ValueError(f"Alert not found: {name}")
        return self._alerts[name]

    def list_alerts(self) -> list[AnalyticsAlert]:
        """List all analytics alerts."""
        return list(self._alerts.values())

    def delete_analytics_alert(self, name: str) -> bool:
        """Delete an analytics alert."""
        if name in self._alerts:
            del self._alerts[name]
            return True
        return False
