"""
Raise Transform Checkpoints

Checkpoint management for incremental processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class CheckpointType(Enum):
    """Checkpoint type enumeration."""
    TIMESTAMP = "timestamp"
    OFFSET = "offset"
    SEQUENCE = "sequence"
    WATERMARK = "watermark"
    COMPOSITE = "composite"


class ProcessingMode(Enum):
    """Processing mode for transformations."""
    FULL = "full"           # Full recompute every run
    INCREMENTAL = "incremental"  # Only process new/changed data
    APPEND = "append"       # Only append new data (no updates)
    UPSERT = "upsert"       # Insert or update based on key


@dataclass
class Checkpoint:
    """
    Checkpoint for tracking incremental processing state.

    Attributes:
        job_id: Associated job ID
        checkpoint_type: Type of checkpoint
        value: Current checkpoint value
        column: Column used for checkpointing (if applicable)
        metadata: Additional checkpoint metadata
        created_at: When checkpoint was created
        updated_at: When checkpoint was last updated
    """

    job_id: str
    checkpoint_type: CheckpointType = CheckpointType.TIMESTAMP
    value: Any = None
    column: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def advance(self, new_value: Any) -> None:
        """Advance checkpoint to new value."""
        self.value = new_value
        self.updated_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Serialize checkpoint to dictionary."""
        return {
            "job_id": self.job_id,
            "checkpoint_type": self.checkpoint_type.value,
            "value": self._serialize_value(self.value),
            "column": self.column,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        """Deserialize checkpoint from dictionary."""
        return cls(
            job_id=data["job_id"],
            checkpoint_type=CheckpointType(data.get("checkpoint_type", "timestamp")),
            value=cls._deserialize_value(data.get("value"), data.get("checkpoint_type", "timestamp")),
            column=data.get("column"),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(),
        )

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Serialize checkpoint value."""
        if isinstance(value, datetime):
            return {"type": "datetime", "value": value.isoformat()}
        return value

    @staticmethod
    def _deserialize_value(value: Any, checkpoint_type: str) -> Any:
        """Deserialize checkpoint value."""
        if isinstance(value, dict) and value.get("type") == "datetime":
            return datetime.fromisoformat(value["value"])
        return value


@dataclass
class IncrementalConfig:
    """
    Configuration for incremental processing.

    Attributes:
        mode: Processing mode (full, incremental, append, upsert)
        checkpoint_column: Column to use for checkpointing
        checkpoint_type: Type of checkpoint (timestamp, offset, etc.)
        key_columns: Columns that form the unique key (for upsert)
        full_refresh_schedule: Optional schedule for full refresh
        lookback: How far back to look for late-arriving data
    """

    mode: ProcessingMode = ProcessingMode.INCREMENTAL
    checkpoint_column: str | None = None
    checkpoint_type: CheckpointType = CheckpointType.TIMESTAMP
    key_columns: list[str] = field(default_factory=list)
    full_refresh_schedule: str | None = None  # e.g., "weekly", "monthly"
    lookback: str | None = None  # e.g., "1h", "1d"

    def __post_init__(self):
        if isinstance(self.mode, str):
            self.mode = ProcessingMode(self.mode)
        if isinstance(self.checkpoint_type, str):
            self.checkpoint_type = CheckpointType(self.checkpoint_type)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dictionary."""
        return {
            "mode": self.mode.value,
            "checkpoint_column": self.checkpoint_column,
            "checkpoint_type": self.checkpoint_type.value,
            "key_columns": self.key_columns,
            "full_refresh_schedule": self.full_refresh_schedule,
            "lookback": self.lookback,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IncrementalConfig:
        """Deserialize config from dictionary."""
        return cls(
            mode=data.get("mode", "incremental"),
            checkpoint_column=data.get("checkpoint_column"),
            checkpoint_type=data.get("checkpoint_type", "timestamp"),
            key_columns=data.get("key_columns", []),
            full_refresh_schedule=data.get("full_refresh_schedule"),
            lookback=data.get("lookback"),
        )

    # Convenience factory methods
    @staticmethod
    def full() -> IncrementalConfig:
        """Create full refresh config."""
        return IncrementalConfig(mode=ProcessingMode.FULL)

    @staticmethod
    def incremental(
        checkpoint_column: str,
        checkpoint_type: CheckpointType | str = CheckpointType.TIMESTAMP,
        lookback: str | None = None,
    ) -> IncrementalConfig:
        """Create incremental config."""
        return IncrementalConfig(
            mode=ProcessingMode.INCREMENTAL,
            checkpoint_column=checkpoint_column,
            checkpoint_type=checkpoint_type if isinstance(checkpoint_type, CheckpointType) else CheckpointType(checkpoint_type),
            lookback=lookback,
        )

    @staticmethod
    def append(checkpoint_column: str) -> IncrementalConfig:
        """Create append-only config."""
        return IncrementalConfig(
            mode=ProcessingMode.APPEND,
            checkpoint_column=checkpoint_column,
        )

    @staticmethod
    def upsert(
        key_columns: list[str],
        checkpoint_column: str | None = None,
    ) -> IncrementalConfig:
        """Create upsert config."""
        return IncrementalConfig(
            mode=ProcessingMode.UPSERT,
            key_columns=key_columns,
            checkpoint_column=checkpoint_column,
        )


@dataclass
class CheckpointStore:
    """
    Store for managing checkpoints.

    In production, this would be backed by a database.
    """

    _checkpoints: dict[str, Checkpoint] = field(default_factory=dict)

    def get(self, job_id: str) -> Checkpoint | None:
        """Get checkpoint for a job."""
        return self._checkpoints.get(job_id)

    def save(self, checkpoint: Checkpoint) -> None:
        """Save or update a checkpoint."""
        self._checkpoints[checkpoint.job_id] = checkpoint

    def delete(self, job_id: str) -> bool:
        """Delete checkpoint for a job."""
        if job_id in self._checkpoints:
            del self._checkpoints[job_id]
            return True
        return False

    def list(self) -> list[Checkpoint]:
        """List all checkpoints."""
        return list(self._checkpoints.values())

    def reset(self, job_id: str) -> bool:
        """Reset checkpoint for a job (triggers full refresh)."""
        if job_id in self._checkpoints:
            self._checkpoints[job_id].value = None
            self._checkpoints[job_id].updated_at = datetime.now()
            return True
        return False
