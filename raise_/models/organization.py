"""
Raise Organization

The top-level container in the namespace hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from raise_.models.acl import ACL
from raise_.models.domain import Domain
from raise_.models.audit import AuditConfig, AuditQueryResult
from raise_.exceptions import DomainNotFoundError


@dataclass
class Organization:
    """
    An organization - the top level of the namespace hierarchy.

    Organizations are the billing and access boundary, containing
    domains which in turn contain projects.

    Attributes:
        name: The organization name (globally unique).
        description: Human-readable description.
        owner: Owner identifier (user or group).
        acl: Access control list.
        metadata: Arbitrary user-defined metadata.
        audit_config: Audit logging configuration.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
    """

    name: str
    description: str | None = None
    owner: str | None = None
    acl: ACL = field(default_factory=ACL)
    metadata: dict[str, Any] = field(default_factory=dict)
    audit_config: AuditConfig = field(default_factory=AuditConfig)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Internal state
    _domains: dict[str, Domain] = field(default_factory=dict, repr=False)

    @property
    def qualified_name(self) -> str:
        """Get the qualified name (same as name for organizations)."""
        return self.name

    def create_domain(
        self,
        name: str,
        *,
        description: str | None = None,
        tags: list[str] | None = None,
        owner: str | None = None,
        acl: ACL | None = None,
        metadata: dict[str, Any] | None = None,
        if_exists: Literal["error", "skip", "update"] = "error",
    ) -> Domain:
        """
        Create a new domain in this organization.

        Args:
            name: The domain name.
            description: Human-readable description.
            tags: List of tags.
            owner: Owner identifier (defaults to org owner).
            acl: Access control (defaults to inherit from org).
            metadata: Arbitrary metadata.
            if_exists: Behavior if domain exists ("error", "skip", "update").

        Returns:
            The created Domain.
        """
        existing = self._domains.get(name)
        if existing:
            if if_exists == "error":
                from raise_.exceptions import RaiseError
                raise RaiseError(f"Domain '{name}' already exists")
            elif if_exists == "skip":
                return existing

        qualified_name = f"{self.name}/{name}"

        domain = Domain(
            name=name,
            qualified_name=qualified_name,
            description=description,
            tags=tags or [],
            owner=owner or self.owner,
            acl=acl or ACL(),
            metadata=metadata or {},
            _organization=self,
        )

        self._domains[name] = domain
        self.updated_at = datetime.now()
        return domain

    def domain(self, name: str) -> Domain:
        """
        Get a domain by name.

        Args:
            name: The domain name.

        Returns:
            The Domain.

        Raises:
            DomainNotFoundError: If domain doesn't exist.
        """
        if name in self._domains:
            return self._domains[name]
        raise DomainNotFoundError(name)

    def list_domains(
        self,
        tags: list[str] | None = None,
    ) -> list[Domain]:
        """
        List domains in this organization.

        Args:
            tags: Filter by tags.

        Returns:
            List of matching Domain objects.
        """
        result = []
        for domain in self._domains.values():
            if tags and not all(t in domain.tags for t in tags):
                continue
            result.append(domain)
        return sorted(result, key=lambda d: d.name)

    def _delete_domain(self, name: str) -> None:
        """Internal method to delete a domain."""
        if name in self._domains:
            del self._domains[name]
            self.updated_at = datetime.now()

    def get_effective_acl(self) -> ACL:
        """Get the effective ACL (no parent for organizations)."""
        return self.acl

    def get_acl_chain(self) -> list[ACL]:
        """Get the ACL chain (just this org's ACL)."""
        return [self.acl]

    def set_acl(self, acl: ACL) -> None:
        """Set the ACL for this organization."""
        self.acl = acl
        self.updated_at = datetime.now()

    def set_audit_config(
        self,
        retention_days: int | None = None,
        immutable: bool | None = None,
        export_destination: str | None = None,
    ) -> AuditConfig:
        """
        Configure audit logging for this organization.

        Args:
            retention_days: How long to keep audit logs.
            immutable: Whether logs can be modified/deleted.
            export_destination: Where to export logs before deletion.

        Returns:
            The updated AuditConfig.
        """
        if retention_days is not None:
            self.audit_config.retention_days = retention_days
        if immutable is not None:
            self.audit_config.immutable = immutable
        if export_destination is not None:
            self.audit_config.export_destination = export_destination

        self.updated_at = datetime.now()
        return self.audit_config

    def get_audit_config(self) -> AuditConfig:
        """Get the audit configuration for this organization."""
        return self.audit_config

    def audit_log(
        self,
        actions: list[str] | None = None,
        category: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> AuditQueryResult:
        """Get audit logs for this organization."""
        return AuditQueryResult(entries=[], total_count=0)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "owner": self.owner,
            "acl": self.acl.to_dict(),
            "metadata": self.metadata,
            "audit_config": self.audit_config.to_dict(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "domains": [d.to_dict() for d in self._domains.values()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Organization:
        """Create an Organization from a dictionary."""
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

        org = cls(
            name=data["name"],
            description=data.get("description"),
            owner=data.get("owner"),
            acl=ACL.from_dict(data["acl"]) if "acl" in data else ACL(),
            metadata=data.get("metadata", {}),
            audit_config=AuditConfig.from_dict(data["audit_config"]) if "audit_config" in data else AuditConfig(),
            created_at=created_at,
            updated_at=updated_at,
        )

        for domain_data in data.get("domains", []):
            domain = Domain.from_dict(domain_data, org)
            org._domains[domain.name] = domain

        return org

    def __str__(self) -> str:
        return f"Organization({self.name})"

    def __repr__(self) -> str:
        return f"Organization(name={self.name!r}, domains={len(self._domains)})"
