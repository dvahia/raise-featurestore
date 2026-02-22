"""
Raise Transform Schedules

Schedule definitions for transformation jobs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from enum import Enum
from typing import Any


class ScheduleType(Enum):
    """Schedule type enumeration."""
    CRON = "cron"
    INTERVAL = "interval"
    ON_CHANGE = "on_change"
    MANUAL = "manual"
    ONCE = "once"


@dataclass
class Schedule(ABC):
    """
    Base class for schedules.

    Schedules define when transformations should run.
    """

    timezone: str = "UTC"
    start_date: datetime | None = None
    end_date: datetime | None = None
    catchup: bool = False

    @property
    @abstractmethod
    def schedule_type(self) -> ScheduleType:
        """Return the schedule type."""
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize schedule to dictionary."""
        pass

    @abstractmethod
    def to_cron(self) -> str | None:
        """Convert to cron expression (if applicable)."""
        pass

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Schedule:
        """Deserialize schedule from dictionary."""
        schedule_type = data.get("schedule_type")
        if schedule_type == "cron":
            return CronSchedule.from_dict(data)
        elif schedule_type == "interval":
            return IntervalSchedule.from_dict(data)
        elif schedule_type == "on_change":
            return OnChangeSchedule.from_dict(data)
        elif schedule_type == "manual":
            return ManualSchedule.from_dict(data)
        elif schedule_type == "once":
            return OnceSchedule.from_dict(data)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

    # Convenience factory methods
    @staticmethod
    def cron(expression: str, **kwargs) -> CronSchedule:
        """Create a cron schedule."""
        return CronSchedule(expression=expression, **kwargs)

    @staticmethod
    def hourly(minute: int = 0, **kwargs) -> CronSchedule:
        """Create an hourly schedule."""
        return CronSchedule(expression=f"{minute} * * * *", **kwargs)

    @staticmethod
    def daily(hour: int = 0, minute: int = 0, **kwargs) -> CronSchedule:
        """Create a daily schedule."""
        return CronSchedule(expression=f"{minute} {hour} * * *", **kwargs)

    @staticmethod
    def weekly(day: int = 0, hour: int = 0, minute: int = 0, **kwargs) -> CronSchedule:
        """Create a weekly schedule (day: 0=Sunday, 6=Saturday)."""
        return CronSchedule(expression=f"{minute} {hour} * * {day}", **kwargs)

    @staticmethod
    def monthly(day: int = 1, hour: int = 0, minute: int = 0, **kwargs) -> CronSchedule:
        """Create a monthly schedule."""
        return CronSchedule(expression=f"{minute} {hour} {day} * *", **kwargs)

    @staticmethod
    def every(interval: str | timedelta, **kwargs) -> IntervalSchedule:
        """Create an interval schedule."""
        return IntervalSchedule(interval=interval, **kwargs)

    @staticmethod
    def on_change(sources: list[str] | None = None, **kwargs) -> OnChangeSchedule:
        """Create an on-change schedule (CDC-triggered)."""
        return OnChangeSchedule(sources=sources or [], **kwargs)

    @staticmethod
    def manual(**kwargs) -> ManualSchedule:
        """Create a manual-only schedule."""
        return ManualSchedule(**kwargs)

    @staticmethod
    def once(run_at: datetime | None = None, **kwargs) -> OnceSchedule:
        """Create a one-time schedule."""
        return OnceSchedule(run_at=run_at, **kwargs)


@dataclass
class CronSchedule(Schedule):
    """
    Cron-based schedule.

    Attributes:
        expression: Cron expression (e.g., "0 2 * * *" for daily at 2am)
    """

    expression: str = "0 0 * * *"

    @property
    def schedule_type(self) -> ScheduleType:
        return ScheduleType.CRON

    def to_cron(self) -> str:
        return self.expression

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_type": "cron",
            "expression": self.expression,
            "timezone": self.timezone,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "catchup": self.catchup,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CronSchedule:
        return cls(
            expression=data.get("expression", "0 0 * * *"),
            timezone=data.get("timezone", "UTC"),
            start_date=datetime.fromisoformat(data["start_date"]) if data.get("start_date") else None,
            end_date=datetime.fromisoformat(data["end_date"]) if data.get("end_date") else None,
            catchup=data.get("catchup", False),
        )

    def __str__(self) -> str:
        return f"cron({self.expression})"


@dataclass
class IntervalSchedule(Schedule):
    """
    Interval-based schedule.

    Attributes:
        interval: Time interval (e.g., "1h", "30m", timedelta)
    """

    interval: str | timedelta = "1h"

    def __post_init__(self):
        if isinstance(self.interval, str):
            self._interval_td = self._parse_interval(self.interval)
        else:
            self._interval_td = self.interval

    @staticmethod
    def _parse_interval(interval: str) -> timedelta:
        """Parse interval string to timedelta."""
        unit = interval[-1].lower()
        value = int(interval[:-1])

        if unit == "s":
            return timedelta(seconds=value)
        elif unit == "m":
            return timedelta(minutes=value)
        elif unit == "h":
            return timedelta(hours=value)
        elif unit == "d":
            return timedelta(days=value)
        elif unit == "w":
            return timedelta(weeks=value)
        else:
            raise ValueError(f"Unknown interval unit: {unit}")

    @property
    def schedule_type(self) -> ScheduleType:
        return ScheduleType.INTERVAL

    @property
    def timedelta(self) -> timedelta:
        """Return interval as timedelta."""
        return self._interval_td

    def to_cron(self) -> str | None:
        """Convert to cron if possible (only for common intervals)."""
        td = self._interval_td
        total_minutes = int(td.total_seconds() / 60)

        if total_minutes == 1:
            return "* * * * *"
        elif total_minutes < 60 and 60 % total_minutes == 0:
            return f"*/{total_minutes} * * * *"
        elif total_minutes == 60:
            return "0 * * * *"
        elif total_minutes < 1440 and total_minutes % 60 == 0:
            hours = total_minutes // 60
            if 24 % hours == 0:
                return f"0 */{hours} * * *"
        elif total_minutes == 1440:
            return "0 0 * * *"

        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_type": "interval",
            "interval": self.interval if isinstance(self.interval, str) else str(self.interval),
            "timezone": self.timezone,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "catchup": self.catchup,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IntervalSchedule:
        return cls(
            interval=data.get("interval", "1h"),
            timezone=data.get("timezone", "UTC"),
            start_date=datetime.fromisoformat(data["start_date"]) if data.get("start_date") else None,
            end_date=datetime.fromisoformat(data["end_date"]) if data.get("end_date") else None,
            catchup=data.get("catchup", False),
        )

    def __str__(self) -> str:
        return f"every({self.interval})"


@dataclass
class OnChangeSchedule(Schedule):
    """
    CDC-triggered schedule.

    Runs when source data changes are detected.

    Attributes:
        sources: List of source names to watch for changes
        debounce: Minimum time between runs
        max_delay: Maximum time to wait for batching changes
    """

    sources: list[str] = field(default_factory=list)
    debounce: str | timedelta = "5m"
    max_delay: str | timedelta = "1h"

    @property
    def schedule_type(self) -> ScheduleType:
        return ScheduleType.ON_CHANGE

    def to_cron(self) -> str | None:
        return None  # Not cron-based

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_type": "on_change",
            "sources": self.sources,
            "debounce": self.debounce if isinstance(self.debounce, str) else str(self.debounce),
            "max_delay": self.max_delay if isinstance(self.max_delay, str) else str(self.max_delay),
            "timezone": self.timezone,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OnChangeSchedule:
        return cls(
            sources=data.get("sources", []),
            debounce=data.get("debounce", "5m"),
            max_delay=data.get("max_delay", "1h"),
            timezone=data.get("timezone", "UTC"),
        )

    def __str__(self) -> str:
        return f"on_change({', '.join(self.sources) if self.sources else 'all'})"


@dataclass
class ManualSchedule(Schedule):
    """
    Manual-only schedule (no automatic runs).

    Jobs with this schedule only run when explicitly triggered.
    """

    @property
    def schedule_type(self) -> ScheduleType:
        return ScheduleType.MANUAL

    def to_cron(self) -> str | None:
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_type": "manual",
            "timezone": self.timezone,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManualSchedule:
        return cls(timezone=data.get("timezone", "UTC"))

    def __str__(self) -> str:
        return "manual"


@dataclass
class OnceSchedule(Schedule):
    """
    One-time schedule.

    Runs once at a specified time (or immediately if not specified).

    Attributes:
        run_at: When to run (None = immediately when deployed)
    """

    run_at: datetime | None = None

    @property
    def schedule_type(self) -> ScheduleType:
        return ScheduleType.ONCE

    def to_cron(self) -> str | None:
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_type": "once",
            "run_at": self.run_at.isoformat() if self.run_at else None,
            "timezone": self.timezone,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OnceSchedule:
        return cls(
            run_at=datetime.fromisoformat(data["run_at"]) if data.get("run_at") else None,
            timezone=data.get("timezone", "UTC"),
        )

    def __str__(self) -> str:
        if self.run_at:
            return f"once({self.run_at.isoformat()})"
        return "once(immediate)"
