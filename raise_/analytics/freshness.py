"""
Raise Freshness Control

Controls data freshness requirements for analyses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal


@dataclass(frozen=True)
class Freshness:
    """
    Specifies data freshness requirements for analysis.

    Controls whether to use cached results or compute fresh data.

    Attributes:
        policy: Freshness policy type.
        max_age: Maximum age for cached results (for WITHIN policy).

    Examples:
        >>> Freshness.REAL_TIME      # Always compute fresh
        >>> Freshness.WITHIN("1h")   # Accept cache if < 1 hour old
        >>> Freshness.CACHED         # Always use cache if available
    """

    policy: Literal["real_time", "within", "cached"]
    max_age: timedelta | None = None

    # Pre-defined freshness levels
    @classmethod
    @property
    def REAL_TIME(cls) -> Freshness:
        """Always compute fresh data, never use cache."""
        return cls(policy="real_time")

    @classmethod
    @property
    def CACHED(cls) -> Freshness:
        """Always use cached data if available (fastest)."""
        return cls(policy="cached")

    @classmethod
    def WITHIN(cls, duration: str | timedelta) -> Freshness:
        """
        Accept cached data if it's within the specified age.

        Args:
            duration: Max age as string ("5m", "1h", "1d") or timedelta.

        Returns:
            Freshness with WITHIN policy.

        Examples:
            >>> Freshness.WITHIN("5m")   # Within 5 minutes
            >>> Freshness.WITHIN("1h")   # Within 1 hour
            >>> Freshness.WITHIN("1d")   # Within 1 day
        """
        if isinstance(duration, str):
            duration = cls._parse_duration(duration)
        return cls(policy="within", max_age=duration)

    @staticmethod
    def _parse_duration(duration: str) -> timedelta:
        """Parse a duration string like '5m', '1h', '1d' into a timedelta."""
        pattern = r"^(\d+)(s|m|h|d|w)$"
        match = re.match(pattern, duration.lower().strip())
        if not match:
            raise ValueError(
                f"Invalid duration format: '{duration}'. "
                "Use format like '5m', '1h', '1d', '1w'"
            )

        value = int(match.group(1))
        unit = match.group(2)

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
            raise ValueError(f"Unknown time unit: {unit}")

    def accepts_age(self, age: timedelta) -> bool:
        """
        Check if a cached result with the given age is acceptable.

        Args:
            age: Age of the cached result.

        Returns:
            True if the cached result is acceptable.
        """
        if self.policy == "real_time":
            return False
        if self.policy == "cached":
            return True
        if self.policy == "within":
            return age <= self.max_age
        return False

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        result = {"policy": self.policy}
        if self.max_age:
            result["max_age_seconds"] = self.max_age.total_seconds()
        return result

    @classmethod
    def from_dict(cls, data: dict) -> Freshness:
        """Create Freshness from dictionary."""
        policy = data["policy"]
        if policy == "real_time":
            return cls.REAL_TIME
        elif policy == "cached":
            return cls.CACHED
        elif policy == "within":
            max_age = timedelta(seconds=data["max_age_seconds"])
            return cls(policy="within", max_age=max_age)
        else:
            raise ValueError(f"Unknown freshness policy: {policy}")

    def __str__(self) -> str:
        if self.policy == "real_time":
            return "Freshness.REAL_TIME"
        elif self.policy == "cached":
            return "Freshness.CACHED"
        elif self.policy == "within":
            return f"Freshness.WITHIN({self.max_age})"
        return f"Freshness({self.policy})"
