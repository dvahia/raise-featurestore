"""
Raise Live Tables

Auto-updating materialized views of analyses using CDC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from raise_.analytics.analysis import Analysis


RefreshPolicyType = Literal["on_change", "hourly", "daily", "weekly", "manual"]


@dataclass
class CDCConfig:
    """
    Change Data Capture configuration for live tables.

    Attributes:
        enabled: Whether CDC is enabled.
        source_table: Source table to monitor for changes.
        track_columns: Specific columns to track (None = all).
        debounce_seconds: Minimum time between refreshes.
        batch_size: Max changes to batch before refresh.
    """

    enabled: bool = True
    source_table: str | None = None
    track_columns: list[str] | None = None
    debounce_seconds: int = 60
    batch_size: int = 1000

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "source_table": self.source_table,
            "track_columns": self.track_columns,
            "debounce_seconds": self.debounce_seconds,
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CDCConfig:
        return cls(
            enabled=data.get("enabled", True),
            source_table=data.get("source_table"),
            track_columns=data.get("track_columns"),
            debounce_seconds=data.get("debounce_seconds", 60),
            batch_size=data.get("batch_size", 1000),
        )


@dataclass
class RefreshPolicy:
    """
    Refresh policy for live tables.

    Attributes:
        type: Policy type (on_change, hourly, daily, weekly, manual).
        cdc_config: CDC configuration for on_change policy.
        schedule_time: Time of day for scheduled refreshes (HH:MM).
        timezone: Timezone for scheduled refreshes.
    """

    type: RefreshPolicyType
    cdc_config: CDCConfig | None = None
    schedule_time: str | None = None  # "HH:MM" format
    timezone: str = "UTC"

    @classmethod
    def on_change(
        cls,
        debounce_seconds: int = 60,
        batch_size: int = 1000,
    ) -> RefreshPolicy:
        """Create an on_change refresh policy with CDC."""
        return cls(
            type="on_change",
            cdc_config=CDCConfig(
                enabled=True,
                debounce_seconds=debounce_seconds,
                batch_size=batch_size,
            ),
        )

    @classmethod
    def hourly(cls) -> RefreshPolicy:
        """Create an hourly refresh policy."""
        return cls(type="hourly")

    @classmethod
    def daily(cls, at: str = "00:00", timezone: str = "UTC") -> RefreshPolicy:
        """Create a daily refresh policy."""
        return cls(type="daily", schedule_time=at, timezone=timezone)

    @classmethod
    def weekly(cls, at: str = "00:00", timezone: str = "UTC") -> RefreshPolicy:
        """Create a weekly refresh policy."""
        return cls(type="weekly", schedule_time=at, timezone=timezone)

    @classmethod
    def manual(cls) -> RefreshPolicy:
        """Create a manual-only refresh policy."""
        return cls(type="manual")

    def to_dict(self) -> dict:
        result = {"type": self.type}
        if self.cdc_config:
            result["cdc_config"] = self.cdc_config.to_dict()
        if self.schedule_time:
            result["schedule_time"] = self.schedule_time
        result["timezone"] = self.timezone
        return result

    @classmethod
    def from_dict(cls, data: dict) -> RefreshPolicy:
        cdc_config = None
        if "cdc_config" in data:
            cdc_config = CDCConfig.from_dict(data["cdc_config"])

        return cls(
            type=data["type"],
            cdc_config=cdc_config,
            schedule_time=data.get("schedule_time"),
            timezone=data.get("timezone", "UTC"),
        )


@dataclass
class RefreshEvent:
    """Record of a live table refresh."""

    id: str
    triggered_at: datetime
    completed_at: datetime | None = None
    trigger: str = "cdc"  # cdc, scheduled, manual
    status: str = "completed"  # completed, failed, running
    rows_affected: int = 0
    error: str | None = None
    changes_detected: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "triggered_at": self.triggered_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "trigger": self.trigger,
            "status": self.status,
            "rows_affected": self.rows_affected,
            "error": self.error,
            "changes_detected": self.changes_detected,
        }


@dataclass
class LiveTable:
    """
    An auto-updating materialized view of an analysis.

    Live tables automatically refresh when source data changes
    (via CDC) or on a schedule.

    Attributes:
        name: Unique name for the live table.
        analysis: The analysis to materialize.
        refresh_policy: How/when to refresh.
        description: Human-readable description.
        created_at: When the table was created.
        last_refresh: When the table was last refreshed.
        status: Current status (active, paused, error).
        row_count: Number of rows in the materialized view.

    Examples:
        >>> live = group.create_live_table(
        ...     name="daily_metrics",
        ...     analysis=Aggregation(
        ...         feature="click_count",
        ...         metrics=["sum", "avg"],
        ...         window="1d",
        ...         rolling=True,
        ...         periods=30,
        ...     ),
        ...     refresh="on_change",
        ... )
        >>> df = live.query()
    """

    name: str
    analysis: Analysis
    refresh_policy: RefreshPolicy
    qualified_name: str = ""
    description: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    last_refresh: datetime | None = None
    next_refresh: datetime | None = None
    status: str = "active"  # active, paused, error
    row_count: int = 0
    error: str | None = None

    # Internal state
    _data: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _refresh_history: list[RefreshEvent] = field(default_factory=list, repr=False)
    _client: Any = field(default=None, repr=False)

    def query(
        self,
        filter: str | None = None,
        order_by: str | None = None,
        limit: int | None = None,
    ) -> Any:
        """
        Query the live table data.

        Args:
            filter: SQL-like filter expression.
            order_by: Column to order by.
            limit: Maximum rows to return.

        Returns:
            pandas.DataFrame with query results.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for query(). "
                "Install it with: pip install pandas"
            )

        df = pd.DataFrame(self._data)

        if filter and len(df) > 0:
            # Simple filter parsing (in production, use proper SQL parser)
            df = self._apply_filter(df, filter)

        if order_by and len(df) > 0:
            ascending = True
            if order_by.startswith("-"):
                order_by = order_by[1:]
                ascending = False
            if order_by in df.columns:
                df = df.sort_values(order_by, ascending=ascending)

        if limit:
            df = df.head(limit)

        return df

    def _apply_filter(self, df: Any, filter: str) -> Any:
        """Apply a simple filter to the dataframe."""
        # Very basic filter parsing - in production use proper SQL
        import re

        # Handle simple equality: "column = 'value'" or "column = value"
        match = re.match(r"(\w+)\s*=\s*['\"]?([^'\"]+)['\"]?", filter)
        if match:
            col, val = match.groups()
            if col in df.columns:
                # Try numeric comparison first
                try:
                    val_numeric = float(val)
                    return df[df[col] == val_numeric]
                except ValueError:
                    return df[df[col] == val]
        return df

    def refresh(self, force: bool = False) -> RefreshEvent:
        """
        Manually trigger a refresh.

        Args:
            force: Force refresh even if data hasn't changed.

        Returns:
            RefreshEvent with details of the refresh.
        """
        from raise_.analytics.result import generate_id

        event = RefreshEvent(
            id=generate_id("refresh"),
            triggered_at=datetime.now(),
            trigger="manual",
            status="running",
        )

        try:
            # In production, this would execute the analysis
            # and update the materialized data
            event.status = "completed"
            event.completed_at = datetime.now()
            event.rows_affected = len(self._data)
            self.last_refresh = datetime.now()
        except Exception as e:
            event.status = "failed"
            event.error = str(e)
            self.status = "error"
            self.error = str(e)

        self._refresh_history.append(event)
        return event

    def refresh_history(
        self,
        limit: int = 100,
        status: str | None = None,
    ) -> list[RefreshEvent]:
        """
        Get the refresh history.

        Args:
            limit: Maximum events to return.
            status: Filter by status.

        Returns:
            List of RefreshEvent objects.
        """
        history = self._refresh_history

        if status:
            history = [e for e in history if e.status == status]

        return sorted(history, key=lambda e: e.triggered_at, reverse=True)[:limit]

    def pause(self) -> None:
        """Pause automatic refreshes."""
        self.status = "paused"

    def resume(self) -> None:
        """Resume automatic refreshes."""
        if self.status == "paused":
            self.status = "active"
            self.error = None

    def delete(self) -> None:
        """Delete this live table."""
        if self._client:
            self._client.delete_live_table(self.name)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "analysis": self.analysis.to_dict(),
            "refresh_policy": self.refresh_policy.to_dict(),
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "last_refresh": self.last_refresh.isoformat() if self.last_refresh else None,
            "next_refresh": self.next_refresh.isoformat() if self.next_refresh else None,
            "status": self.status,
            "row_count": self.row_count,
            "error": self.error,
        }

    def __repr__(self) -> str:
        return (
            f"LiveTable(name={self.name!r}, status={self.status!r}, "
            f"rows={self.row_count}, last_refresh={self.last_refresh})"
        )
