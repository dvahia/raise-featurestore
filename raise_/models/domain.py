"""
Raise Domain

A domain within an organization, containing projects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TYPE_CHECKING, Literal

from raise_.models.acl import ACL
from raise_.models.project import Project
from raise_.models.audit import AuditQueryResult
from raise_.exceptions import ProjectNotFoundError

if TYPE_CHECKING:
    from raise_.models.organization import Organization


@dataclass
class Domain:
    """
    A domain containing projects.

    Domains organize projects within an organization and represent
    business or technical areas (e.g., "mlplatform", "search", "ads").

    Attributes:
        name: The domain name (unique within an organization).
        qualified_name: Full path including org/domain.
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
    _organization: Organization | None = field(default=None, repr=False)
    _projects: dict[str, Project] = field(default_factory=dict, repr=False)

    @property
    def org(self) -> str:
        """Get the organization name."""
        return self.qualified_name.split("/")[0]

    def create_project(
        self,
        name: str,
        *,
        description: str | None = None,
        tags: list[str] | None = None,
        owner: str | None = None,
        acl: ACL | None = None,
        metadata: dict[str, Any] | None = None,
        if_exists: Literal["error", "skip", "update"] = "error",
    ) -> Project:
        """
        Create a new project in this domain.

        Args:
            name: The project name.
            description: Human-readable description.
            tags: List of tags.
            owner: Owner identifier (defaults to domain owner).
            acl: Access control (defaults to inherit from domain).
            metadata: Arbitrary metadata.
            if_exists: Behavior if project exists ("error", "skip", "update").

        Returns:
            The created Project.
        """
        existing = self._projects.get(name)
        if existing:
            if if_exists == "error":
                from raise_.exceptions import RaiseError
                raise RaiseError(f"Project '{name}' already exists")
            elif if_exists == "skip":
                return existing

        qualified_name = f"{self.qualified_name}/{name}"

        project = Project(
            name=name,
            qualified_name=qualified_name,
            description=description,
            tags=tags or [],
            owner=owner or self.owner,
            acl=acl or ACL(),
            metadata=metadata or {},
            _domain=self,
        )

        self._projects[name] = project
        self.updated_at = datetime.now()
        return project

    def project(self, name: str) -> Project:
        """
        Get a project by name.

        Args:
            name: The project name.

        Returns:
            The Project.

        Raises:
            ProjectNotFoundError: If project doesn't exist.
        """
        if name in self._projects:
            return self._projects[name]
        raise ProjectNotFoundError(name)

    def list_projects(
        self,
        tags: list[str] | None = None,
    ) -> list[Project]:
        """
        List projects in this domain.

        Args:
            tags: Filter by tags.

        Returns:
            List of matching Project objects.
        """
        result = []
        for project in self._projects.values():
            if tags and not all(t in project.tags for t in tags):
                continue
            result.append(project)
        return sorted(result, key=lambda p: p.name)

    def _delete_project(self, name: str) -> None:
        """Internal method to delete a project."""
        if name in self._projects:
            del self._projects[name]
            self.updated_at = datetime.now()

    def get_effective_acl(self) -> ACL:
        """Get the effective ACL including inherited permissions."""
        if not self.acl.inherit or not self._organization:
            return self.acl
        parent_acl = self._organization.get_effective_acl()
        return parent_acl.merge(self.acl)

    def get_acl_chain(self) -> list[ACL]:
        """Get the full ACL inheritance chain."""
        chain = []
        if self._organization:
            chain.extend(self._organization.get_acl_chain())
        chain.append(self.acl)
        return chain

    def set_acl(self, acl: ACL) -> None:
        """Set the ACL for this domain."""
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
        """Get audit logs for this domain."""
        return AuditQueryResult(entries=[], total_count=0)

    def delete(self) -> None:
        """Delete this domain."""
        if self._organization:
            self._organization._delete_domain(self.name)

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
            "projects": [p.to_dict() for p in self._projects.values()],
        }

    @classmethod
    def from_dict(cls, data: dict, organization: Organization | None = None) -> Domain:
        """Create a Domain from a dictionary."""
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

        domain = cls(
            name=data["name"],
            qualified_name=data["qualified_name"],
            description=data.get("description"),
            tags=data.get("tags", []),
            owner=data.get("owner"),
            acl=ACL.from_dict(data["acl"]) if "acl" in data else ACL(),
            metadata=data.get("metadata", {}),
            created_at=created_at,
            updated_at=updated_at,
            _organization=organization,
        )

        for project_data in data.get("projects", []):
            project = Project.from_dict(project_data, domain)
            domain._projects[project.name] = project

        return domain

    def __str__(self) -> str:
        return f"Domain({self.qualified_name})"

    def __repr__(self) -> str:
        return f"Domain(name={self.name!r}, projects={len(self._projects)})"
