"""
Raise Analytics Alerts

Threshold-based alerting on analysis results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from raise_.analytics.analysis import Analysis


class ConditionOperator(Enum):
    """Condition operators for alerts."""

    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN_OR_EQUAL = "lte"
    BETWEEN = "between"
    OUTSIDE = "outside"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"

    # Statistical conditions
    PSI_GREATER_THAN = "psi_gt"
    KL_DIVERGENCE_GREATER_THAN = "kl_gt"
    P_VALUE_LESS_THAN = "p_value_lt"
    CHANGE_PERCENT_GREATER_THAN = "change_pct_gt"


@dataclass
class Condition:
    """
    A condition for triggering an alert.

    Conditions compare analysis results against thresholds.

    Attributes:
        operator: The comparison operator.
        value: The threshold value(s).
        field: The result field to check (default: primary metric).

    Examples:
        >>> Condition.GREATER_THAN(0.05)
        >>> Condition.BETWEEN(0.01, 0.10)
        >>> Condition.P_VALUE_LESS_THAN(0.05)
    """

    operator: ConditionOperator
    value: Any
    value2: Any = None  # For BETWEEN/OUTSIDE
    field: str | None = None

    def evaluate(self, result_value: Any) -> bool:
        """
        Evaluate the condition against a result value.

        Args:
            result_value: The value from the analysis result.

        Returns:
            True if condition is met (alert should trigger).
        """
        if result_value is None:
            return False

        op = self.operator

        if op == ConditionOperator.GREATER_THAN:
            return result_value > self.value
        elif op == ConditionOperator.LESS_THAN:
            return result_value < self.value
        elif op == ConditionOperator.EQUALS:
            return result_value == self.value
        elif op == ConditionOperator.NOT_EQUALS:
            return result_value != self.value
        elif op == ConditionOperator.GREATER_THAN_OR_EQUAL:
            return result_value >= self.value
        elif op == ConditionOperator.LESS_THAN_OR_EQUAL:
            return result_value <= self.value
        elif op == ConditionOperator.BETWEEN:
            return self.value <= result_value <= self.value2
        elif op == ConditionOperator.OUTSIDE:
            return result_value < self.value or result_value > self.value2
        elif op == ConditionOperator.CONTAINS:
            return self.value in result_value
        elif op == ConditionOperator.NOT_CONTAINS:
            return self.value not in result_value
        # Statistical conditions use same logic as their base operators
        elif op in (
            ConditionOperator.PSI_GREATER_THAN,
            ConditionOperator.KL_DIVERGENCE_GREATER_THAN,
            ConditionOperator.CHANGE_PERCENT_GREATER_THAN,
        ):
            return result_value > self.value
        elif op == ConditionOperator.P_VALUE_LESS_THAN:
            return result_value < self.value

        return False

    # Factory methods for common conditions
    @classmethod
    def GREATER_THAN(cls, value: float, field: str | None = None) -> Condition:
        """Create a greater-than condition."""
        return cls(ConditionOperator.GREATER_THAN, value, field=field)

    @classmethod
    def LESS_THAN(cls, value: float, field: str | None = None) -> Condition:
        """Create a less-than condition."""
        return cls(ConditionOperator.LESS_THAN, value, field=field)

    @classmethod
    def EQUALS(cls, value: Any, field: str | None = None) -> Condition:
        """Create an equals condition."""
        return cls(ConditionOperator.EQUALS, value, field=field)

    @classmethod
    def NOT_EQUALS(cls, value: Any, field: str | None = None) -> Condition:
        """Create a not-equals condition."""
        return cls(ConditionOperator.NOT_EQUALS, value, field=field)

    @classmethod
    def BETWEEN(cls, low: float, high: float, field: str | None = None) -> Condition:
        """Create a between condition (inclusive)."""
        return cls(ConditionOperator.BETWEEN, low, high, field=field)

    @classmethod
    def OUTSIDE(cls, low: float, high: float, field: str | None = None) -> Condition:
        """Create an outside condition."""
        return cls(ConditionOperator.OUTSIDE, low, high, field=field)

    @classmethod
    def PSI_GREATER_THAN(cls, value: float) -> Condition:
        """Create a PSI (drift) threshold condition."""
        return cls(ConditionOperator.PSI_GREATER_THAN, value, field="psi")

    @classmethod
    def KL_DIVERGENCE_GREATER_THAN(cls, value: float) -> Condition:
        """Create a KL divergence threshold condition."""
        return cls(ConditionOperator.KL_DIVERGENCE_GREATER_THAN, value, field="kl_divergence")

    @classmethod
    def P_VALUE_LESS_THAN(cls, value: float) -> Condition:
        """Create a p-value significance condition."""
        return cls(ConditionOperator.P_VALUE_LESS_THAN, value, field="p_value")

    @classmethod
    def CHANGE_PERCENT_GREATER_THAN(cls, value: float) -> Condition:
        """Create a percentage change threshold condition."""
        return cls(ConditionOperator.CHANGE_PERCENT_GREATER_THAN, value, field="change_percent")

    def to_dict(self) -> dict:
        return {
            "operator": self.operator.value,
            "value": self.value,
            "value2": self.value2,
            "field": self.field,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Condition:
        return cls(
            operator=ConditionOperator(data["operator"]),
            value=data["value"],
            value2=data.get("value2"),
            field=data.get("field"),
        )

    def __str__(self) -> str:
        field_str = f"{self.field} " if self.field else ""
        if self.operator in (ConditionOperator.BETWEEN, ConditionOperator.OUTSIDE):
            return f"{field_str}{self.operator.name}({self.value}, {self.value2})"
        return f"{field_str}{self.operator.name}({self.value})"


AlertStatus = Literal["active", "paused", "triggered", "resolved"]


@dataclass
class AlertEvent:
    """
    An event when an alert is triggered.

    Attributes:
        id: Unique event identifier.
        alert_name: Name of the alert that triggered.
        triggered_at: When the alert was triggered.
        resolved_at: When the alert was resolved (if applicable).
        result_value: The value that triggered the alert.
        threshold: The threshold that was exceeded.
        message: Alert message.
        notified: List of recipients notified.
    """

    id: str
    alert_name: str
    triggered_at: datetime
    resolved_at: datetime | None = None
    result_value: Any = None
    threshold: Any = None
    message: str | None = None
    notified: list[str] = field(default_factory=list)

    @property
    def is_resolved(self) -> bool:
        """Check if the alert has been resolved."""
        return self.resolved_at is not None

    @property
    def duration(self) -> float | None:
        """Get duration in seconds (if resolved)."""
        if self.resolved_at:
            return (self.resolved_at - self.triggered_at).total_seconds()
        return None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "alert_name": self.alert_name,
            "triggered_at": self.triggered_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "result_value": self.result_value,
            "threshold": self.threshold,
            "message": self.message,
            "notified": self.notified,
        }


@dataclass
class AnalyticsAlert:
    """
    An alert on analysis results.

    Alerts check analysis results against conditions and
    send notifications when thresholds are exceeded.

    Attributes:
        name: Unique alert name.
        analysis: The analysis to monitor.
        condition: Condition that triggers the alert.
        notify: List of notification recipients.
        channels: Notification channels (email, slack, webhook).
        check_interval: How often to check (e.g., "1h", "5m").
        message_template: Custom message template.
        enabled: Whether the alert is active.

    Examples:
        >>> AnalyticsAlert(
        ...     name="high-null-rate",
        ...     analysis=Aggregation(feature="embedding", metrics=["null_rate"]),
        ...     condition=Condition.GREATER_THAN(0.05),
        ...     notify=["data-quality@acme.com"],
        ...     check_interval="1h",
        ... )
    """

    name: str
    analysis: Analysis
    condition: Condition
    notify: list[str]
    channels: list[str] = field(default_factory=lambda: ["email"])
    check_interval: str = "1h"
    message_template: str | None = None
    enabled: bool = True

    # State
    status: AlertStatus = "active"
    last_checked: datetime | None = None
    last_triggered: datetime | None = None
    trigger_count: int = 0

    # Internal
    _events: list[AlertEvent] = field(default_factory=list, repr=False)
    _client: Any = field(default=None, repr=False)

    def check(self) -> AlertEvent | None:
        """
        Manually check the alert condition.

        Returns:
            AlertEvent if triggered, None otherwise.
        """
        # In production, this would run the analysis and check the condition
        self.last_checked = datetime.now()
        return None

    def pause(self) -> None:
        """Pause the alert."""
        self.status = "paused"
        self.enabled = False

    def resume(self) -> None:
        """Resume the alert."""
        self.status = "active"
        self.enabled = True

    def delete(self) -> None:
        """Delete this alert."""
        if self._client:
            self._client.delete_analytics_alert(self.name)

    def events(
        self,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[AlertEvent]:
        """
        Get alert events history.

        Args:
            limit: Maximum events to return.
            since: Only return events after this time.

        Returns:
            List of AlertEvent objects.
        """
        events = self._events

        if since:
            events = [e for e in events if e.triggered_at >= since]

        return sorted(events, key=lambda e: e.triggered_at, reverse=True)[:limit]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "analysis": self.analysis.to_dict(),
            "condition": self.condition.to_dict(),
            "notify": self.notify,
            "channels": self.channels,
            "check_interval": self.check_interval,
            "message_template": self.message_template,
            "enabled": self.enabled,
            "status": self.status,
            "last_checked": self.last_checked.isoformat() if self.last_checked else None,
            "last_triggered": self.last_triggered.isoformat() if self.last_triggered else None,
            "trigger_count": self.trigger_count,
        }

    def __repr__(self) -> str:
        return (
            f"AnalyticsAlert(name={self.name!r}, status={self.status!r}, "
            f"enabled={self.enabled}, triggers={self.trigger_count})"
        )
