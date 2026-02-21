"""
Raise Feature Group

A logical grouping of related features within a project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING, Literal

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from raise_.models.types import FeatureType, parse_dtype
from raise_.models.acl import ACL, ExternalGrant
from raise_.models.feature import Feature
from raise_.models.audit import AuditQueryResult
from raise_.exceptions import (
    FeatureExistsError,
    FeatureNotFoundError,
    ValidationError,
)

if TYPE_CHECKING:
    from raise_.models.project import Project
    from raise_.validation import ValidationResult


@dataclass
class FeatureGroup:
    """
    A logical grouping of related features.

    Feature groups organize features within a project and provide
    a level for ACL inheritance and batch operations.

    Attributes:
        name: The feature group name (unique within a project).
        qualified_name: Full path including org/domain/project/group.
        description: Human-readable description.
        tags: List of tags for categorization.
        owner: Owner identifier (user or group).
        acl: Access control list.
        metadata: Arbitrary user-defined metadata.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.

    Examples:
        >>> # Create features in a group
        >>> group = project.create_feature_group("user-signals")
        >>> group.create_feature("clicks", dtype="int64")

        >>> # Bulk create
        >>> group.create_features_from_schema({
        ...     "clicks": "int64",
        ...     "impressions": "int64",
        ... })

        >>> # List features
        >>> for feature in group.list_features():
        ...     print(feature.name)
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
    _project: Project | None = field(default=None, repr=False)
    _features: dict[str, Feature] = field(default_factory=dict, repr=False)
    _feature_versions: dict[str, dict[str, Feature]] = field(default_factory=dict, repr=False)
    _external_grants: list[ExternalGrant] = field(default_factory=list, repr=False)

    @property
    def org(self) -> str:
        """Get the organization name."""
        return self.qualified_name.split("/")[0]

    @property
    def domain(self) -> str:
        """Get the domain name."""
        return self.qualified_name.split("/")[1]

    @property
    def project_name(self) -> str:
        """Get the project name."""
        return self.qualified_name.split("/")[2]

    def create_feature(
        self,
        name: str,
        dtype: str | FeatureType,
        *,
        description: str | None = None,
        tags: list[str] | None = None,
        owner: str | None = None,
        acl: ACL | None = None,
        nullable: bool = True,
        default: Any = None,
        derived_from: str | None = None,
        version: str | None = None,
        if_exists: Literal["error", "skip", "update"] = "error",
        validation: Literal["strict", "standard", "permissive"] = "standard",
        metadata: dict[str, Any] | None = None,
    ) -> Feature:
        """
        Create a new feature in this group.

        Args:
            name: The feature name.
            dtype: Data type (string or FeatureType).
            description: Human-readable description.
            tags: List of tags.
            owner: Owner identifier (defaults to current user).
            acl: Access control (defaults to inherit from group).
            nullable: Whether the feature can be null.
            default: Default value when null.
            derived_from: SQL expression for derived features.
            version: Version string (auto-generated if None).
            if_exists: Behavior if feature exists ("error", "skip", "update").
            validation: Validation level for derived_from expression.
            metadata: Arbitrary metadata.

        Returns:
            The created Feature.

        Raises:
            FeatureExistsError: If feature exists and if_exists="error".
            ValidationError: If derived_from expression is invalid.
        """
        # Check if feature exists
        existing = self._features.get(name)

        # If explicit version is provided, check if that specific version exists
        if version and name in self._feature_versions:
            if version in self._feature_versions[name]:
                existing_version = self._feature_versions[name][version]
                if if_exists == "error":
                    raise FeatureExistsError(name, existing_version.qualified_name)
                elif if_exists == "skip":
                    return existing_version
            # New version number, allow creation
            existing = None

        if existing and version is None:
            if if_exists == "error":
                raise FeatureExistsError(name, existing.qualified_name)
            elif if_exists == "skip":
                return existing
            # if_exists == "update" falls through to create new version

        # Parse dtype
        parsed_dtype = parse_dtype(dtype)

        # Validate derived_from expression
        if derived_from:
            result = self._validate_expression(derived_from, validation)
            if not result.valid:
                raise ValidationError(
                    result.errors[0].message,
                    result.errors[0].code,
                    result.errors[0].position,
                )

        # Determine version
        if version is None:
            if name in self._feature_versions:
                version_num = len(self._feature_versions[name]) + 1
            else:
                version_num = 1
            version = f"v{version_num}"

        # Create qualified name
        qualified_name = f"{self.qualified_name}/{name}@{version}"

        # Create feature
        feature = Feature(
            name=name,
            qualified_name=qualified_name,
            dtype=parsed_dtype,
            version=version,
            description=description,
            tags=tags or [],
            owner=owner or self.owner,
            acl=acl or ACL(),
            nullable=nullable,
            default=default,
            derived_from=derived_from,
            metadata=metadata or {},
            _feature_group=self,
        )

        # Store feature
        self._features[name] = feature
        if name not in self._feature_versions:
            self._feature_versions[name] = {}
        self._feature_versions[name][version] = feature

        self.updated_at = datetime.now()
        return feature

    def create_features(
        self,
        features: list[dict[str, Any]],
        if_exists: Literal["error", "skip", "update"] = "error",
    ) -> list[Feature]:
        """
        Create multiple features at once.

        Args:
            features: List of feature definitions (dicts with name, dtype, etc.).
            if_exists: Behavior if feature exists.

        Returns:
            List of created Feature objects.

        Example:
            >>> group.create_features([
            ...     {"name": "clicks", "dtype": "int64"},
            ...     {"name": "ctr", "dtype": "float64", "derived_from": "clicks / impressions"},
            ... ])
        """
        result = []
        for spec in features:
            name = spec.pop("name")
            feature = self.create_feature(name, if_exists=if_exists, **spec)
            result.append(feature)
        return result

    def create_features_from_schema(
        self,
        schema: dict[str, str | FeatureType],
        if_exists: Literal["error", "skip", "update"] = "error",
    ) -> list[Feature]:
        """
        Create features from a simple name-to-type mapping.

        Args:
            schema: Dictionary mapping feature names to dtypes.
            if_exists: Behavior if feature exists.

        Returns:
            List of created Feature objects.

        Example:
            >>> group.create_features_from_schema({
            ...     "clicks": "int64",
            ...     "impressions": "int64",
            ...     "user_embedding": "float32[512]",
            ... })
        """
        result = []
        for name, dtype in schema.items():
            feature = self.create_feature(name, dtype, if_exists=if_exists)
            result.append(feature)
        return result

    def create_features_from_file(
        self,
        path: str | Path,
        if_exists: Literal["error", "skip", "update"] = "error",
    ) -> list[Feature]:
        """
        Create features from a YAML file.

        Args:
            path: Path to the YAML file.
            if_exists: Behavior if feature exists.

        Returns:
            List of created Feature objects.

        YAML format:
            features:
              - name: clicks
                dtype: int64
                description: Total clicks
              - name: ctr
                dtype: float64
                derived_from: clicks / impressions

        Raises:
            ImportError: If PyYAML is not installed.
        """
        if not HAS_YAML:
            raise ImportError(
                "PyYAML is required for create_features_from_file(). "
                "Install it with: pip install pyyaml"
            )

        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        features = data.get("features", [])
        return self.create_features(features, if_exists=if_exists)

    def get_or_create_feature(
        self,
        name: str,
        dtype: str | FeatureType,
        **kwargs,
    ) -> Feature:
        """
        Get an existing feature or create it if it doesn't exist.

        Args:
            name: The feature name.
            dtype: Data type for creation.
            **kwargs: Additional arguments for creation.

        Returns:
            The existing or newly created Feature.
        """
        return self.create_feature(name, dtype, if_exists="skip", **kwargs)

    def feature(self, name: str) -> Feature:
        """
        Get a feature by name.

        Supports version suffix: "feature_name@v1"

        Args:
            name: Feature name, optionally with @version suffix.

        Returns:
            The Feature.

        Raises:
            FeatureNotFoundError: If feature doesn't exist.
        """
        # Parse version suffix
        if "@" in name:
            base_name, version = name.rsplit("@", 1)
            if base_name in self._feature_versions:
                if version in self._feature_versions[base_name]:
                    return self._feature_versions[base_name][version]
            raise FeatureNotFoundError(name)

        if name in self._features:
            return self._features[name]

        suggestions = [n for n in self._features.keys() if n.startswith(name[:3])]
        raise FeatureNotFoundError(name, suggestions)

    def list_features(
        self,
        tags: list[str] | None = None,
        status: str | None = None,
        include_deprecated: bool = False,
    ) -> list[Feature]:
        """
        List features in this group.

        Args:
            tags: Filter by tags (features must have all specified tags).
            status: Filter by status (active, deprecated, archived).
            include_deprecated: Include deprecated features in results.

        Returns:
            List of matching Feature objects.
        """
        result = []
        for feature in self._features.values():
            # Filter by status
            if status and feature.status != status:
                continue
            if not include_deprecated and feature.status == "deprecated":
                continue

            # Filter by tags
            if tags and not all(t in feature.tags for t in tags):
                continue

            result.append(feature)

        return sorted(result, key=lambda f: f.name)

    def validate_feature(
        self,
        name: str,
        dtype: str | FeatureType,
        derived_from: str | None = None,
        **kwargs,
    ) -> ValidationResult:
        """
        Validate a feature definition without creating it.

        Args:
            name: The feature name.
            dtype: Data type.
            derived_from: SQL expression for derived features.
            **kwargs: Other feature properties (ignored for validation).

        Returns:
            ValidationResult with validation status and any errors.
        """
        from raise_.validation import ValidationResult, validate_expression

        if derived_from is None:
            return ValidationResult(valid=True)

        return validate_expression(
            derived_from,
            context=self._get_context(),
            available_features=self._features,
        )

    def _validate_expression(
        self,
        expression: str,
        level: str = "standard",
    ) -> ValidationResult:
        """Internal method to validate a derived_from expression."""
        from raise_.validation import ValidationResult, validate_expression

        return validate_expression(
            expression,
            context=self._get_context(),
            available_features=self._features,
            level=level,
        )

    def _get_context(self) -> dict[str, str]:
        """Get the context for path resolution."""
        return {
            "org": self.org,
            "domain": self.domain,
            "project": self.project_name,
            "feature_group": self.name,
        }

    def _delete_feature(self, name: str) -> None:
        """Internal method to delete a feature."""
        if name in self._features:
            del self._features[name]
        if name in self._feature_versions:
            del self._feature_versions[name]
        self.updated_at = datetime.now()

    def _list_feature_versions(self, name: str) -> list[Feature]:
        """Internal method to list all versions of a feature."""
        if name in self._feature_versions:
            return list(self._feature_versions[name].values())
        return []

    def get_effective_acl(self) -> ACL:
        """Get the effective ACL including inherited permissions."""
        if not self.acl.inherit or not self._project:
            return self.acl
        parent_acl = self._project.get_effective_acl()
        return parent_acl.merge(self.acl)

    def get_acl_chain(self) -> list[ACL]:
        """Get the full ACL inheritance chain."""
        chain = []
        if self._project:
            chain.extend(self._project.get_acl_chain())
        chain.append(self.acl)
        return chain

    def set_acl(self, acl: ACL) -> None:
        """Set the ACL for this group."""
        self.acl = acl
        self.updated_at = datetime.now()

    def grant_external_access(
        self,
        org: str,
        features: list[str],
        permission: Literal["read", "write"] = "read",
        expires_at: datetime | None = None,
    ) -> ExternalGrant:
        """
        Grant access to an external organization.

        Args:
            org: The organization to grant access to.
            features: List of feature names (or ["*"] for all).
            permission: Permission level to grant.
            expires_at: Optional expiration datetime.

        Returns:
            The created ExternalGrant.
        """
        grant = ExternalGrant(
            org=org,
            features=features,
            permission=permission,
            granted_by=self.owner or "unknown",
            granted_at=datetime.now().isoformat(),
            expires_at=expires_at.isoformat() if expires_at else None,
        )
        self._external_grants.append(grant)
        self.updated_at = datetime.now()
        return grant

    def list_external_grants(self) -> list[ExternalGrant]:
        """List all external access grants."""
        return [g for g in self._external_grants if not g.is_expired()]

    def revoke_external_access(self, org: str) -> bool:
        """
        Revoke access for an external organization.

        Args:
            org: The organization to revoke access from.

        Returns:
            True if a grant was revoked.
        """
        original_count = len(self._external_grants)
        self._external_grants = [g for g in self._external_grants if g.org != org]
        if len(self._external_grants) < original_count:
            self.updated_at = datetime.now()
            return True
        return False

    def audit_log(
        self,
        actions: list[str] | None = None,
        category: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> AuditQueryResult:
        """Get audit logs for this feature group."""
        return AuditQueryResult(entries=[], total_count=0)

    # =========================================================================
    # Analytics methods
    # =========================================================================

    def analyze(self, analysis: Any, freshness: Any = None) -> Any:
        """
        Run an analysis on features in this group.

        Args:
            analysis: The analysis to run.
            freshness: Freshness requirements.

        Returns:
            AnalysisResult with computed data.
        """
        # Import here to avoid circular imports
        from raise_.analytics.client import AnalyticsClient

        # Get or create analytics client
        if not hasattr(self, '_analytics') or self._analytics is None:
            self._analytics = AnalyticsClient(self)

        return self._analytics.analyze(analysis, freshness)

    def create_live_table(
        self,
        name: str,
        analysis: Any,
        refresh: str = "on_change",
        description: str | None = None,
    ) -> Any:
        """
        Create a live table backed by an analysis.

        Args:
            name: Live table name.
            analysis: Analysis to materialize.
            refresh: Refresh policy ("on_change", "hourly", "daily", "manual").
            description: Description.

        Returns:
            The created LiveTable.
        """
        from raise_.analytics.client import AnalyticsClient

        if not hasattr(self, '_analytics') or self._analytics is None:
            self._analytics = AnalyticsClient(self)

        return self._analytics.create_live_table(name, analysis, refresh, description)

    def delete(self) -> None:
        """Delete this feature group."""
        if self._project:
            self._project._delete_feature_group(self.name)

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
            "features": [f.to_dict() for f in self._features.values()],
        }

    @classmethod
    def from_dict(cls, data: dict, project: Project | None = None) -> FeatureGroup:
        """Create a FeatureGroup from a dictionary."""
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

        group = cls(
            name=data["name"],
            qualified_name=data["qualified_name"],
            description=data.get("description"),
            tags=data.get("tags", []),
            owner=data.get("owner"),
            acl=ACL.from_dict(data["acl"]) if "acl" in data else ACL(),
            metadata=data.get("metadata", {}),
            created_at=created_at,
            updated_at=updated_at,
            _project=project,
        )

        # Load features
        for feature_data in data.get("features", []):
            feature = Feature.from_dict(feature_data, group)
            group._features[feature.name] = feature
            if feature.name not in group._feature_versions:
                group._feature_versions[feature.name] = {}
            group._feature_versions[feature.name][feature.version] = feature

        return group

    def __str__(self) -> str:
        return f"FeatureGroup({self.qualified_name})"

    def __repr__(self) -> str:
        return f"FeatureGroup(name={self.name!r}, features={len(self._features)})"
