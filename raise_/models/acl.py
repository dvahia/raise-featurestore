"""
Raise ACL (Access Control List)

Defines access control for features and feature groups.
ACLs cascade from parent to child by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Permission = Literal["read", "write", "admin"]


@dataclass
class ACL:
    """
    Access Control List for a resource.

    ACLs define who can read, write, or administer a resource.
    By default, ACLs inherit from parent resources (cascade down).

    Args:
        readers: Users/groups with read access.
        writers: Users/groups with write access.
        admins: Users/groups with admin access (can modify ACL).
        inherit: Whether to inherit permissions from parent (default: True).

    Examples:
        >>> # Basic ACL
        >>> acl = ACL(
        ...     readers=["ml-engineers@acme.com"],
        ...     writers=["ml-team@acme.com"],
        ...     admins=["ml-leads@acme.com"],
        ... )

        >>> # Override without inheritance
        >>> restricted_acl = ACL(
        ...     readers=["restricted-team@acme.com"],
        ...     inherit=False,
        ... )
    """

    readers: list[str] = field(default_factory=list)
    writers: list[str] = field(default_factory=list)
    admins: list[str] = field(default_factory=list)
    inherit: bool = True

    def has_permission(self, user: str, permission: Permission) -> bool:
        """
        Check if a user has a specific permission.

        Admin permission implies write, write implies read.

        Args:
            user: The user identifier to check.
            permission: The permission level to check.

        Returns:
            True if the user has the permission.
        """
        if user in self.admins:
            return True
        if permission == "admin":
            return False

        if user in self.writers:
            return True
        if permission == "write":
            return False

        return user in self.readers

    def add_reader(self, user: str) -> ACL:
        """Add a reader and return a new ACL."""
        if user not in self.readers:
            return ACL(
                readers=self.readers + [user],
                writers=self.writers.copy(),
                admins=self.admins.copy(),
                inherit=self.inherit,
            )
        return self

    def add_writer(self, user: str) -> ACL:
        """Add a writer and return a new ACL."""
        if user not in self.writers:
            return ACL(
                readers=self.readers.copy(),
                writers=self.writers + [user],
                admins=self.admins.copy(),
                inherit=self.inherit,
            )
        return self

    def add_admin(self, user: str) -> ACL:
        """Add an admin and return a new ACL."""
        if user not in self.admins:
            return ACL(
                readers=self.readers.copy(),
                writers=self.writers.copy(),
                admins=self.admins + [user],
                inherit=self.inherit,
            )
        return self

    def remove_user(self, user: str) -> ACL:
        """Remove a user from all permission lists."""
        return ACL(
            readers=[r for r in self.readers if r != user],
            writers=[w for w in self.writers if w != user],
            admins=[a for a in self.admins if a != user],
            inherit=self.inherit,
        )

    def merge(self, other: ACL) -> ACL:
        """
        Merge with another ACL (union of all permissions).

        Used internally for resolving inherited ACLs.
        """
        return ACL(
            readers=list(set(self.readers + other.readers)),
            writers=list(set(self.writers + other.writers)),
            admins=list(set(self.admins + other.admins)),
            inherit=self.inherit,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "readers": self.readers,
            "writers": self.writers,
            "admins": self.admins,
            "inherit": self.inherit,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ACL:
        """Create an ACL from a dictionary."""
        return cls(
            readers=data.get("readers", []),
            writers=data.get("writers", []),
            admins=data.get("admins", []),
            inherit=data.get("inherit", True),
        )


@dataclass
class ExternalGrant:
    """
    Represents a cross-organization access grant.

    Args:
        org: The organization being granted access.
        features: List of feature names the grant applies to.
        permission: The permission level granted.
        granted_by: The user who created the grant.
        granted_at: When the grant was created.
        expires_at: Optional expiration datetime.
    """

    org: str
    features: list[str]
    permission: Permission
    granted_by: str
    granted_at: str  # ISO 8601 datetime string
    expires_at: str | None = None

    def is_expired(self) -> bool:
        """Check if this grant has expired."""
        if self.expires_at is None:
            return False
        from datetime import datetime

        return datetime.fromisoformat(self.expires_at) < datetime.now()

    def covers_feature(self, feature_name: str) -> bool:
        """Check if this grant covers a specific feature."""
        return feature_name in self.features or "*" in self.features

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "org": self.org,
            "features": self.features,
            "permission": self.permission,
            "granted_by": self.granted_by,
            "granted_at": self.granted_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ExternalGrant:
        """Create an ExternalGrant from a dictionary."""
        return cls(
            org=data["org"],
            features=data["features"],
            permission=data["permission"],
            granted_by=data["granted_by"],
            granted_at=data["granted_at"],
            expires_at=data.get("expires_at"),
        )
