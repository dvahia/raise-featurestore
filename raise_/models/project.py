"""
Raise Project

A project within a domain, containing feature groups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TYPE_CHECKING, Literal

from raise_.models.acl import ACL
from raise_.models.feature_group import FeatureGroup
from raise_.models.audit import AuditQueryResult
from raise_.exceptions import FeatureGroupNotFoundError

if TYPE_CHECKING:
    from raise_.models.domain import Domain


@dataclass
class Project:
    """
    A project containing feature groups.

    Projects organize feature groups within a domain and represent
    specific initiatives or models (e.g., "recommendation", "fraud-detection").

    Attributes:
        name: The project name (unique within a domain).
        qualified_name: Full path including org/domain/project.
        description: Human-readable description.
        tags: List of tags for categorization.
        owner: Owner identifier (user or group).
        acl: Access control list.
        metadata: Arbitrary user-defined metadata.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
    """

    name: str
    qualified_name: str
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    owner: str | None = None
    acl: ACL = field(default_factory=ACL)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Internal state
    _domain: Domain | None = field(default=None, repr=False)
    _feature_groups: dict[str, FeatureGroup] = field(default_factory=dict, repr=False)

    @property
    def org(self) -> str:
        """Get the organization name."""
        return self.qualified_name.split("/")[0]

    @property
    def domain_name(self) -> str:
        """Get the domain name."""
        return self.qualified_name.split("/")[1]

    def create_feature_group(
        self,
        name: str,
        *,
        description: str | None = None,
        tags: list[str] | None = None,
        owner: str | None = None,
        acl: ACL | None = None,
        metadata: dict[str, Any] | None = None,
        if_exists: Literal["error", "skip", "update"] = "error",
    ) -> FeatureGroup:
        """
        Create a new feature group in this project.

        Args:
            name: The feature group name.
            description: Human-readable description.
            tags: List of tags.
            owner: Owner identifier (defaults to project owner).
            acl: Access control (defaults to inherit from project).
            metadata: Arbitrary metadata.
            if_exists: Behavior if group exists ("error", "skip", "update").

        Returns:
            The created FeatureGroup.
        """
        existing = self._feature_groups.get(name)
        if existing:
            if if_exists == "error":
                from raise_.exceptions import RaiseError
                raise RaiseError(f"Feature group '{name}' already exists")
            elif if_exists == "skip":
                return existing

        qualified_name = f"{self.qualified_name}/{name}"

        group = FeatureGroup(
            name=name,
            qualified_name=qualified_name,
            description=description,
            tags=tags or [],
            owner=owner or self.owner,
            acl=acl or ACL(),
            metadata=metadata or {},
            _project=self,
        )

        self._feature_groups[name] = group
        self.updated_at = datetime.now()
        return group

    def feature_group(self, name: str) -> FeatureGroup:
        """
        Get a feature group by name.

        Args:
            name: The feature group name.

        Returns:
            The FeatureGroup.

        Raises:
            FeatureGroupNotFoundError: If group doesn't exist.
        """
        if name in self._feature_groups:
            return self._feature_groups[name]
        raise FeatureGroupNotFoundError(name)

    def list_feature_groups(
        self,
        tags: list[str] | None = None,
    ) -> list[FeatureGroup]:
        """
        List feature groups in this project.

        Args:
            tags: Filter by tags.

        Returns:
            List of matching FeatureGroup objects.
        """
        result = []
        for group in self._feature_groups.values():
            if tags and not all(t in group.tags for t in tags):
                continue
            result.append(group)
        return sorted(result, key=lambda g: g.name)

    def _delete_feature_group(self, name: str) -> None:
        """Internal method to delete a feature group."""
        if name in self._feature_groups:
            del self._feature_groups[name]
            self.updated_at = datetime.now()

    def get_effective_acl(self) -> ACL:
        """Get the effective ACL including inherited permissions."""
        if not self.acl.inherit or not self._domain:
            return self.acl
        parent_acl = self._domain.get_effective_acl()
        return parent_acl.merge(self.acl)

    def get_acl_chain(self) -> list[ACL]:
        """Get the full ACL inheritance chain."""
        chain = []
        if self._domain:
            chain.extend(self._domain.get_acl_chain())
        chain.append(self.acl)
        return chain

    def set_acl(self, acl: ACL) -> None:
        """Set the ACL for this project."""
        self.acl = acl
        self.updated_at = datetime.now()

    def audit_log(
        self,
        actions: list[str] | None = None,
        category: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> AuditQueryResult:
        """Get audit logs for this project."""
        return AuditQueryResult(entries=[], total_count=0)

    def delete(self) -> None:
        """Delete this project."""
        if self._domain:
            self._domain._delete_project(self.name)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "description": self.description,
            "tags": self.tags,
            "owner": self.owner,
            "acl": self.acl.to_dict(),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "feature_groups": [g.to_dict() for g in self._feature_groups.values()],
        }

    @classmethod
    def from_dict(cls, data: dict, domain: Domain | None = None) -> Project:
        """Create a Project from a dictionary."""
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

        project = cls(
            name=data["name"],
            qualified_name=data["qualified_name"],
            description=data.get("description"),
            tags=data.get("tags", []),
            owner=data.get("owner"),
            acl=ACL.from_dict(data["acl"]) if "acl" in data else ACL(),
            metadata=data.get("metadata", {}),
            created_at=created_at,
            updated_at=updated_at,
            _domain=domain,
        )

        for group_data in data.get("feature_groups", []):
            group = FeatureGroup.from_dict(group_data, project)
            project._feature_groups[group.name] = group

        return project

    def __str__(self) -> str:
        return f"Project({self.qualified_name})"

    def __repr__(self) -> str:
        return f"Project(name={self.name!r}, feature_groups={len(self._feature_groups)})"
