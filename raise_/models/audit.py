"""
Raise Audit

Audit logging for tracking access and changes to features.
Captures metadata only (not actual data values).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator, Literal


AuditCategory = Literal["access", "schema", "acl", "lineage", "admin"]
AuditAction = Literal[
    # Access
    "READ",
    "WRITE",
    "QUERY",
    # Schema
    "CREATE",
    "UPDATE_SCHEMA",
    "UPDATE_METADATA",
    "DELETE",
    "DEPRECATE",
    "CREATE_VERSION",
    # ACL
    "UPDATE_ACL",
    "GRANT_EXTERNAL",
    "REVOKE_EXTERNAL",
    # Lineage
    "UPDATE_DERIVATION",
    "LINEAGE_BREAK",
]


@dataclass
class AuditEntry:
    """
    A single audit log entry.

    Captures metadata about who did what, when, and to which resource.
    Does not capture actual data values for privacy.

    Attributes:
        id: Unique log entry ID.
        timestamp: When the action occurred.
        actor: User or service account that performed the action.
        actor_org: Organization of the actor.
        actor_ip: IP address (if available).
        actor_user_agent: Client information.
        action: The action performed (CREATE, READ, etc.).
        category: The category of action (access, schema, acl, etc.).
        resource: Fully qualified resource path.
        resource_type: Type of resource (feature, feature_group, etc.).
        details: Action-specific metadata.
        previous_state: For updates, the state before the change.
        new_state: For updates, the state after the change.
        request_id: Correlation ID for request tracing.
        source: How the action was triggered (api, console, pipeline).
    """

    id: str
    timestamp: datetime
    actor: str
    actor_org: str
    action: AuditAction
    category: AuditCategory
    resource: str
    resource_type: str
    details: dict[str, Any] = field(default_factory=dict)
    actor_ip: str | None = None
    actor_user_agent: str | None = None
    previous_state: dict[str, Any] | None = None
    new_state: dict[str, Any] | None = None
    request_id: str | None = None
    source: str = "api"

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "actor": self.actor,
            "actor_org": self.actor_org,
            "actor_ip": self.actor_ip,
            "actor_user_agent": self.actor_user_agent,
            "action": self.action,
            "category": self.category,
            "resource": self.resource,
            "resource_type": self.resource_type,
            "details": self.details,
            "previous_state": self.previous_state,
            "new_state": self.new_state,
            "request_id": self.request_id,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AuditEntry:
        """Create an AuditEntry from a dictionary."""
        timestamp = data["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return cls(
            id=data["id"],
            timestamp=timestamp,
            actor=data["actor"],
            actor_org=data["actor_org"],
            actor_ip=data.get("actor_ip"),
            actor_user_agent=data.get("actor_user_agent"),
            action=data["action"],
            category=data["category"],
            resource=data["resource"],
            resource_type=data["resource_type"],
            details=data.get("details", {}),
            previous_state=data.get("previous_state"),
            new_state=data.get("new_state"),
            request_id=data.get("request_id"),
            source=data.get("source", "api"),
        )


@dataclass
class AuditQuery:
    """
    Query parameters for searching audit logs.

    All parameters are optional and combined with AND logic.

    Attributes:
        resource: Resource path pattern (supports * wildcard).
        actions: List of actions to filter by.
        category: Category to filter by.
        actor: Specific actor to filter by.
        actor_org: Filter by actor's organization.
        exclude_actor_orgs: Exclude specific organizations.
        since: Start of time range.
        until: End of time range.
        limit: Maximum number of entries to return.
        cursor: Pagination cursor for next page.
    """

    resource: str | None = None
    actions: list[AuditAction] | None = None
    category: AuditCategory | None = None
    actor: str | None = None
    actor_org: str | None = None
    exclude_actor_orgs: list[str] | None = None
    since: datetime | None = None
    until: datetime | None = None
    limit: int = 100
    cursor: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        result = {}
        if self.resource:
            result["resource"] = self.resource
        if self.actions:
            result["actions"] = self.actions
        if self.category:
            result["category"] = self.category
        if self.actor:
            result["actor"] = self.actor
        if self.actor_org:
            result["actor_org"] = self.actor_org
        if self.exclude_actor_orgs:
            result["exclude_actor_orgs"] = self.exclude_actor_orgs
        if self.since:
            result["since"] = self.since.isoformat()
        if self.until:
            result["until"] = self.until.isoformat()
        result["limit"] = self.limit
        if self.cursor:
            result["cursor"] = self.cursor
        return result


@dataclass
class AuditQueryResult:
    """
    Result of an audit query with pagination support.
    """

    entries: list[AuditEntry]
    total_count: int
    next_cursor: str | None = None
    has_more: bool = False

    def __iter__(self) -> Iterator[AuditEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


@dataclass
class AuditAlert:
    """
    An alert configuration for audit events.

    Alerts trigger notifications when matching audit events occur.

    Attributes:
        name: Unique name for this alert.
        query: The query that triggers this alert.
        notify: List of email addresses or endpoints to notify.
        channels: Notification channels (email, slack, webhook).
        enabled: Whether the alert is active.
        created_at: When the alert was created.
        created_by: Who created the alert.
    """

    name: str
    query: AuditQuery
    notify: list[str]
    channels: list[str] = field(default_factory=lambda: ["email"])
    enabled: bool = True
    created_at: datetime | None = None
    created_by: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "query": self.query.to_dict(),
            "notify": self.notify,
            "channels": self.channels,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AuditAlert:
        """Create an AuditAlert from a dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            name=data["name"],
            query=AuditQuery(**data["query"]),
            notify=data["notify"],
            channels=data.get("channels", ["email"]),
            enabled=data.get("enabled", True),
            created_at=created_at,
            created_by=data.get("created_by"),
        )


@dataclass
class AuditConfig:
    """
    Organization-level audit configuration.

    Attributes:
        retention_days: How long to keep audit logs.
        immutable: Whether logs can be modified/deleted.
        export_destination: Where to export logs before deletion.
    """

    retention_days: int = 365
    immutable: bool = True
    export_destination: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "retention_days": self.retention_days,
            "immutable": self.immutable,
            "export_destination": self.export_destination,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AuditConfig:
        """Create an AuditConfig from a dictionary."""
        return cls(
            retention_days=data.get("retention_days", 365),
            immutable=data.get("immutable", True),
            export_destination=data.get("export_destination"),
        )


class AuditClient:
    """
    Client for querying and managing audit logs.
    """

    def __init__(self, feature_store: Any):
        """
        Initialize the audit client.

        Args:
            feature_store: The parent FeatureStore instance.
        """
        self._fs = feature_store
        self._alerts: dict[str, AuditAlert] = {}

    def query(
        self,
        resource: str | None = None,
        actions: list[AuditAction] | None = None,
        category: AuditCategory | None = None,
        actor: str | None = None,
        actor_org: str | None = None,
        exclude_actor_orgs: list[str] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> AuditQueryResult:
        """
        Query audit logs with flexible filters.

        Args:
            resource: Resource path pattern (supports * wildcard).
            actions: List of actions to filter by.
            category: Category to filter by.
            actor: Specific actor to filter by.
            actor_org: Filter by actor's organization.
            exclude_actor_orgs: Exclude specific organizations.
            since: Start of time range.
            until: End of time range.
            limit: Maximum number of entries to return.
            cursor: Pagination cursor for next page.

        Returns:
            AuditQueryResult with matching entries.
        """
        query = AuditQuery(
            resource=resource,
            actions=actions,
            category=category,
            actor=actor,
            actor_org=actor_org,
            exclude_actor_orgs=exclude_actor_orgs,
            since=since,
            until=until,
            limit=limit,
            cursor=cursor,
        )
        # In a real implementation, this would call the backend
        return AuditQueryResult(entries=[], total_count=0)

    def create_alert(
        self,
        name: str,
        query: AuditQuery,
        notify: list[str],
        channels: list[str] | None = None,
    ) -> AuditAlert:
        """
        Create an alert for audit events matching a query.

        Args:
            name: Unique name for this alert.
            query: The query that triggers this alert.
            notify: List of email addresses or endpoints to notify.
            channels: Notification channels (email, slack, webhook).

        Returns:
            The created AuditAlert.
        """
        alert = AuditAlert(
            name=name,
            query=query,
            notify=notify,
            channels=channels or ["email"],
            enabled=True,
            created_at=datetime.now(),
            created_by=self._fs._current_user,
        )
        self._alerts[name] = alert
        return alert

    def list_alerts(self) -> list[AuditAlert]:
        """List all configured alerts."""
        return list(self._alerts.values())

    def get_alert(self, name: str) -> AuditAlert | None:
        """Get an alert by name."""
        return self._alerts.get(name)

    def delete_alert(self, name: str) -> bool:
        """Delete an alert by name."""
        if name in self._alerts:
            del self._alerts[name]
            return True
        return False

    def export(
        self,
        query: AuditQuery,
        format: Literal["jsonl", "csv", "parquet"] = "jsonl",
        destination: str | None = None,
    ) -> str:
        """
        Export audit logs matching a query.

        Args:
            query: The query to filter logs.
            format: Export format (jsonl, csv, parquet).
            destination: Destination path (e.g., s3://bucket/path/).

        Returns:
            The path where logs were exported.
        """
        # In a real implementation, this would export to the destination
        return destination or f"/tmp/audit_export.{format}"

    def stream(self, query: AuditQuery) -> AuditStream:
        """
        Stream audit logs for large exports.

        Args:
            query: The query to filter logs.

        Returns:
            An AuditStream for batch processing.
        """
        return AuditStream(self, query)


class AuditStream:
    """
    Streaming interface for large audit exports.
    """

    def __init__(self, client: AuditClient, query: AuditQuery):
        self._client = client
        self._query = query
        self._cursor: str | None = None

    def __enter__(self) -> AuditStream:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def batches(self, size: int = 1000) -> Iterator[list[AuditEntry]]:
        """
        Iterate over batches of audit entries.

        Args:
            size: Batch size.

        Yields:
            Lists of AuditEntry objects.
        """
        while True:
            result = self._client.query(
                resource=self._query.resource,
                actions=self._query.actions,
                category=self._query.category,
                actor=self._query.actor,
                actor_org=self._query.actor_org,
                exclude_actor_orgs=self._query.exclude_actor_orgs,
                since=self._query.since,
                until=self._query.until,
                limit=size,
                cursor=self._cursor,
            )

            if not result.entries:
                break

            yield result.entries

            if not result.has_more:
                break

            self._cursor = result.next_cursor
