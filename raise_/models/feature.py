"""
Raise Feature

The core Feature model representing an ML feature in the feature store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TYPE_CHECKING

from raise_.models.types import FeatureType, parse_dtype
from raise_.models.acl import ACL
from raise_.models.lineage import Lineage
from raise_.models.audit import AuditEntry, AuditQueryResult

if TYPE_CHECKING:
    from raise_.models.feature_group import FeatureGroup


@dataclass
class Feature:
    """
    An ML feature in the feature store.

    Features are logical abstractions over physical storage columns.
    They can be base features (backed by physical storage) or derived
    features (computed from other features using SQL expressions).

    Attributes:
        name: The feature name (unique within a feature group).
        qualified_name: Full path including org/domain/project/group/name@version.
        dtype: The data type of the feature.
        version: Version string (e.g., "v1", "v2").
        description: Human-readable description.
        tags: List of tags for categorization.
        owner: Owner identifier (user or group).
        acl: Access control list.
        nullable: Whether the feature can be null.
        default: Default value when null.
        derived_from: SQL expression for derived features.
        metadata: Arbitrary user-defined metadata.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
        status: Feature status (active, deprecated, archived).

    Examples:
        >>> # Access feature properties
        >>> feature.name
        'user_embedding'
        >>> feature.dtype
        Embedding('float32[512]')
        >>> feature.qualified_name
        'acme/mlplatform/recommendation/user-signals/user_embedding@v1'

        >>> # Check lineage
        >>> lineage = feature.get_lineage()
        >>> lineage.upstream
        [click_count, impression_count]

        >>> # Deprecate a feature
        >>> feature.deprecate("Replaced by user_embedding_v2")
    """

    name: str
    qualified_name: str
    dtype: FeatureType
    version: str
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    owner: str | None = None
    acl: ACL = field(default_factory=ACL)
    nullable: bool = True
    default: Any = None
    derived_from: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: str = "active"

    # Internal references (not serialized)
    _feature_group: FeatureGroup | None = field(default=None, repr=False)
    _lineage: Lineage | None = field(default=None, repr=False)

    @property
    def org(self) -> str:
        """Get the organization name from the qualified name."""
        return self.qualified_name.split("/")[0].lstrip("@")

    @property
    def domain(self) -> str:
        """Get the domain name from the qualified name."""
        return self.qualified_name.split("/")[1]

    @property
    def project(self) -> str:
        """Get the project name from the qualified name."""
        return self.qualified_name.split("/")[2]

    @property
    def feature_group_name(self) -> str:
        """Get the feature group name from the qualified name."""
        parts = self.qualified_name.split("/")
        return parts[3].split(".")[0]

    @property
    def is_derived(self) -> bool:
        """Check if this is a derived feature."""
        return self.derived_from is not None

    def update(
        self,
        description: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        nullable: bool | None = None,
        default: Any = ...,  # Sentinel to distinguish None from not-set
    ) -> Feature:
        """
        Update mutable feature properties.

        Note: dtype and derived_from cannot be changed. Create a new version instead.

        Args:
            description: New description.
            tags: New tags (replaces existing).
            metadata: New metadata (replaces existing).
            nullable: New nullable setting.
            default: New default value.

        Returns:
            The updated Feature.
        """
        if description is not None:
            self.description = description
        if tags is not None:
            self.tags = tags
        if metadata is not None:
            self.metadata = metadata
        if nullable is not None:
            self.nullable = nullable
        if default is not ...:
            self.default = default

        self.updated_at = datetime.now()
        return self

    def delete(self) -> None:
        """
        Delete this feature.

        Raises:
            RaiseError: If the feature has downstream dependencies.
        """
        if self._feature_group:
            self._feature_group._delete_feature(self.name)

    def deprecate(self, message: str | None = None) -> None:
        """
        Mark this feature as deprecated.

        Deprecated features can still be read but emit warnings.

        Args:
            message: Deprecation message (e.g., "Use feature_v2 instead").
        """
        self.status = "deprecated"
        if message:
            self.metadata["deprecation_message"] = message
        self.updated_at = datetime.now()

    def archive(self) -> None:
        """
        Archive this feature.

        Archived features are hidden from listings but preserved for lineage.
        """
        self.status = "archived"
        self.updated_at = datetime.now()

    def get_lineage(self) -> Lineage:
        """
        Get the lineage information for this feature.

        Returns:
            Lineage object with upstream and downstream features.
        """
        if self._lineage is None:
            self._lineage = Lineage(feature=self)
        return self._lineage

    def get_effective_acl(self) -> ACL:
        """
        Get the effective ACL including inherited permissions.

        Returns:
            ACL with all inherited permissions resolved.
        """
        if not self.acl.inherit or not self._feature_group:
            return self.acl

        parent_acl = self._feature_group.get_effective_acl()
        return parent_acl.merge(self.acl)

    def get_acl_chain(self) -> list[ACL]:
        """
        Get the full ACL inheritance chain.

        Returns:
            List of ACLs from root to this feature.
        """
        chain = []
        if self._feature_group:
            chain.extend(self._feature_group.get_acl_chain())
        chain.append(self.acl)
        return chain

    def set_acl(self, acl: ACL) -> None:
        """
        Set the ACL for this feature.

        Args:
            acl: The new ACL.
        """
        self.acl = acl
        self.updated_at = datetime.now()

    def list_versions(self) -> list[Feature]:
        """
        List all versions of this feature.

        Returns:
            List of Feature objects for all versions.
        """
        if self._feature_group:
            return self._feature_group._list_feature_versions(self.name)
        return [self]

    def audit_log(
        self,
        actions: list[str] | None = None,
        category: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> AuditQueryResult:
        """
        Get audit logs for this feature.

        Args:
            actions: Filter by actions (READ, WRITE, etc.).
            category: Filter by category (access, schema, etc.).
            since: Start of time range.
            until: End of time range.
            limit: Maximum entries to return.

        Returns:
            AuditQueryResult with matching entries.
        """
        # In a real implementation, this would query the audit backend
        return AuditQueryResult(entries=[], total_count=0)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "dtype": self.dtype.to_string(),
            "version": self.version,
            "description": self.description,
            "tags": self.tags,
            "owner": self.owner,
            "acl": self.acl.to_dict(),
            "nullable": self.nullable,
            "default": self.default,
            "derived_from": self.derived_from,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict, feature_group: FeatureGroup | None = None) -> Feature:
        """Create a Feature from a dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.now()

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        else:
            updated_at = datetime.now()

        return cls(
            name=data["name"],
            qualified_name=data["qualified_name"],
            dtype=parse_dtype(data["dtype"]),
            version=data["version"],
            description=data.get("description"),
            tags=data.get("tags", []),
            owner=data.get("owner"),
            acl=ACL.from_dict(data["acl"]) if "acl" in data else ACL(),
            nullable=data.get("nullable", True),
            default=data.get("default"),
            derived_from=data.get("derived_from"),
            metadata=data.get("metadata", {}),
            created_at=created_at,
            updated_at=updated_at,
            status=data.get("status", "active"),
            _feature_group=feature_group,
        )

    def __str__(self) -> str:
        return f"Feature({self.qualified_name})"

    def __repr__(self) -> str:
        return (
            f"Feature(name={self.name!r}, dtype={self.dtype}, "
            f"version={self.version!r}, status={self.status!r})"
        )
