"""
Raise FeatureStore Client

The main entry point for interacting with the Raise Feature Store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from raise_.models.acl import ACL
from raise_.models.organization import Organization
from raise_.models.domain import Domain
from raise_.models.project import Project
from raise_.models.feature_group import FeatureGroup
from raise_.models.feature import Feature
from raise_.models.types import FeatureType, parse_dtype
from raise_.models.audit import AuditClient, AuditQuery
from raise_.analytics.client import AnalyticsClient
from raise_.analytics.analysis import Analysis
from raise_.analytics.dashboard import Dashboard
from raise_.exceptions import (
    OrganizationNotFoundError,
    DomainNotFoundError,
    ProjectNotFoundError,
    FeatureGroupNotFoundError,
    FeatureNotFoundError,
)


@dataclass
class FeatureStore:
    """
    The main client for interacting with the Raise Feature Store.

    Provides access to the namespace hierarchy (org/domain/project/group/feature)
    and convenience methods for common operations.

    Args:
        path: Path string in format "org/domain/project" or "org/domain" or "org".
        org: Organization name (alternative to path).
        domain: Domain name (requires org).
        project: Project name (requires org and domain).

    Examples:
        >>> # Connect with path string
        >>> fs = FeatureStore("acme/mlplatform/recommendation")

        >>> # Connect with explicit parameters
        >>> fs = FeatureStore(org="acme", domain="mlplatform", project="recommendation")

        >>> # Create and work with features
        >>> group = fs.create_feature_group("user-signals")
        >>> group.create_feature("clicks", dtype="int64")

        >>> # Access features directly
        >>> feature = fs.feature("user-signals/clicks")
    """

    # Connection parameters
    _org_name: str | None = field(default=None, repr=False)
    _domain_name: str | None = field(default=None, repr=False)
    _project_name: str | None = field(default=None, repr=False)

    # Cached hierarchy
    _organizations: dict[str, Organization] = field(default_factory=dict, repr=False)

    # Current user (would be set from auth in production)
    _current_user: str = field(default="user@example.com", repr=False)

    # Audit client
    audit: AuditClient = field(init=False, repr=False)

    # Analytics client
    analytics: AnalyticsClient = field(init=False, repr=False)

    def __init__(
        self,
        path: str | None = None,
        *,
        org: str | None = None,
        domain: str | None = None,
        project: str | None = None,
    ):
        """
        Initialize the FeatureStore client.

        Args:
            path: Path string in format "org/domain/project".
            org: Organization name.
            domain: Domain name.
            project: Project name.
        """
        self._organizations = {}

        if path:
            parts = path.split("/")
            if len(parts) >= 1:
                org = parts[0]
            if len(parts) >= 2:
                domain = parts[1]
            if len(parts) >= 3:
                project = parts[2]

        self._org_name = org
        self._domain_name = domain
        self._project_name = project

        # Initialize audit client
        self.audit = AuditClient(self)

        # Initialize analytics client
        self.analytics = AnalyticsClient(self)

        # Auto-create context if specified
        if org:
            self._get_or_create_org(org)
            if domain:
                self._get_or_create_domain(org, domain)
                if project:
                    self._get_or_create_project(org, domain, project)

    def with_context(
        self,
        org: str | None = None,
        domain: str | None = None,
        project: str | None = None,
    ) -> FeatureStore:
        """
        Create a new FeatureStore with updated context.

        Args:
            org: New organization (or keep current if None).
            domain: New domain (or keep current if None).
            project: New project (or keep current if None).

        Returns:
            A new FeatureStore with the updated context.
        """
        new_fs = FeatureStore(
            org=org or self._org_name,
            domain=domain or self._domain_name,
            project=project or self._project_name,
        )
        new_fs._organizations = self._organizations
        new_fs._current_user = self._current_user
        return new_fs

    # =========================================================================
    # Organization methods
    # =========================================================================

    def create_organization(
        self,
        name: str,
        *,
        description: str | None = None,
        owner: str | None = None,
        acl: ACL | None = None,
        metadata: dict[str, Any] | None = None,
        if_exists: Literal["error", "skip", "update"] = "error",
    ) -> Organization:
        """
        Create a new organization.

        Args:
            name: Organization name (globally unique).
            description: Human-readable description.
            owner: Owner identifier.
            acl: Access control list.
            metadata: Arbitrary metadata.
            if_exists: Behavior if org exists.

        Returns:
            The created Organization.
        """
        if name in self._organizations:
            if if_exists == "error":
                from raise_.exceptions import RaiseError
                raise RaiseError(f"Organization '{name}' already exists")
            elif if_exists == "skip":
                return self._organizations[name]

        org = Organization(
            name=name,
            description=description,
            owner=owner or self._current_user,
            acl=acl or ACL(),
            metadata=metadata or {},
        )
        self._organizations[name] = org
        return org

    def organization(self, name: str) -> Organization:
        """
        Get an organization by name.

        Args:
            name: Organization name.

        Returns:
            The Organization.

        Raises:
            OrganizationNotFoundError: If org doesn't exist.
        """
        if name in self._organizations:
            return self._organizations[name]
        raise OrganizationNotFoundError(name)

    def list_organizations(self) -> list[Organization]:
        """List all organizations."""
        return sorted(self._organizations.values(), key=lambda o: o.name)

    # =========================================================================
    # Domain methods (shortcuts using current context)
    # =========================================================================

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
        Create a domain in the current organization.

        Args:
            name: Domain name.
            description: Human-readable description.
            tags: List of tags.
            owner: Owner identifier.
            acl: Access control list.
            metadata: Arbitrary metadata.
            if_exists: Behavior if domain exists.

        Returns:
            The created Domain.
        """
        org = self._require_org()
        return org.create_domain(
            name,
            description=description,
            tags=tags,
            owner=owner,
            acl=acl,
            metadata=metadata,
            if_exists=if_exists,
        )

    def domain(self, name: str | None = None) -> Domain:
        """
        Get a domain by name (or current domain if name is None).

        Args:
            name: Domain name (uses current context if None).

        Returns:
            The Domain.
        """
        org = self._require_org()
        domain_name = name or self._domain_name
        if not domain_name:
            raise ValueError("No domain specified and no default domain in context")
        return org.domain(domain_name)

    def list_domains(self, tags: list[str] | None = None) -> list[Domain]:
        """List domains in the current organization."""
        org = self._require_org()
        return org.list_domains(tags=tags)

    # =========================================================================
    # Project methods (shortcuts using current context)
    # =========================================================================

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
        Create a project in the current domain.

        Args:
            name: Project name.
            description: Human-readable description.
            tags: List of tags.
            owner: Owner identifier.
            acl: Access control list.
            metadata: Arbitrary metadata.
            if_exists: Behavior if project exists.

        Returns:
            The created Project.
        """
        domain = self._require_domain()
        return domain.create_project(
            name,
            description=description,
            tags=tags,
            owner=owner,
            acl=acl,
            metadata=metadata,
            if_exists=if_exists,
        )

    def project(self, name: str | None = None) -> Project:
        """
        Get a project by name (or current project if name is None).

        Args:
            name: Project name (uses current context if None).

        Returns:
            The Project.
        """
        domain = self._require_domain()
        project_name = name or self._project_name
        if not project_name:
            raise ValueError("No project specified and no default project in context")
        return domain.project(project_name)

    def list_projects(self, tags: list[str] | None = None) -> list[Project]:
        """List projects in the current domain."""
        domain = self._require_domain()
        return domain.list_projects(tags=tags)

    # =========================================================================
    # Feature Group methods (shortcuts using current context)
    # =========================================================================

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
        Create a feature group in the current project.

        Args:
            name: Feature group name.
            description: Human-readable description.
            tags: List of tags.
            owner: Owner identifier.
            acl: Access control list.
            metadata: Arbitrary metadata.
            if_exists: Behavior if group exists.

        Returns:
            The created FeatureGroup.
        """
        project = self._require_project()
        return project.create_feature_group(
            name,
            description=description,
            tags=tags,
            owner=owner,
            acl=acl,
            metadata=metadata,
            if_exists=if_exists,
        )

    def feature_group(self, name: str) -> FeatureGroup:
        """
        Get a feature group by name.

        Args:
            name: Feature group name.

        Returns:
            The FeatureGroup.
        """
        project = self._require_project()
        return project.feature_group(name)

    def list_feature_groups(self, tags: list[str] | None = None) -> list[FeatureGroup]:
        """List feature groups in the current project."""
        project = self._require_project()
        return project.list_feature_groups(tags=tags)

    # =========================================================================
    # Feature methods (shortcuts using path syntax)
    # =========================================================================

    def create_feature(
        self,
        path: str,
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
        Create a feature using path syntax.

        Args:
            path: Path in format "group/feature" or just "feature" (requires default group).
            dtype: Data type.
            description: Human-readable description.
            tags: List of tags.
            owner: Owner identifier.
            acl: Access control list.
            nullable: Whether feature can be null.
            default: Default value.
            derived_from: SQL expression for derived features.
            version: Version string.
            if_exists: Behavior if feature exists.
            validation: Validation level for derived_from.
            metadata: Arbitrary metadata.

        Returns:
            The created Feature.
        """
        # Parse path
        if "/" in path:
            group_name, feature_name = path.rsplit("/", 1)
        else:
            raise ValueError("Path must be in format 'group/feature'")

        group = self.feature_group(group_name)
        return group.create_feature(
            feature_name,
            dtype,
            description=description,
            tags=tags,
            owner=owner,
            acl=acl,
            nullable=nullable,
            default=default,
            derived_from=derived_from,
            version=version,
            if_exists=if_exists,
            validation=validation,
            metadata=metadata,
        )

    def feature(self, path: str) -> Feature:
        """
        Get a feature using path syntax.

        Args:
            path: Path in format "group/feature" or "group/feature@version".

        Returns:
            The Feature.
        """
        # Parse path
        if "/" in path:
            group_name, feature_name = path.rsplit("/", 1)
        else:
            raise ValueError("Path must be in format 'group/feature'")

        group = self.feature_group(group_name)
        return group.feature(feature_name)

    def search_features(
        self,
        query: str | None = None,
        dtype: str | None = None,
        tags: list[str] | None = None,
        include_external: bool = False,
        limit: int = 100,
    ) -> list[Feature]:
        """
        Search for features across the current context.

        Args:
            query: Search string to match against feature names/descriptions.
            dtype: Filter by data type (supports wildcards like "float32[*]").
            tags: Filter by tags.
            include_external: Include features from other organizations.
            limit: Maximum number of results.

        Returns:
            List of matching Feature objects.
        """
        results = []
        project = self._require_project()

        for group in project.list_feature_groups():
            for feature in group.list_features(include_deprecated=False):
                # Apply filters
                if query and query.lower() not in feature.name.lower():
                    if not feature.description or query.lower() not in feature.description.lower():
                        continue

                if tags and not all(t in feature.tags for t in tags):
                    continue

                if dtype:
                    # Simple wildcard matching
                    dtype_str = feature.dtype.to_string()
                    if "*" in dtype:
                        pattern = dtype.replace("*", ".*").replace("[", r"\[").replace("]", r"\]")
                        import re
                        if not re.match(pattern, dtype_str):
                            continue
                    elif dtype_str != dtype:
                        continue

                results.append(feature)

                if len(results) >= limit:
                    return results

        return results

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _require_org(self) -> Organization:
        """Get the current organization or raise an error."""
        if not self._org_name:
            raise ValueError("No organization in context. Use FeatureStore('org/...') or with_context(org='...')")
        return self.organization(self._org_name)

    def _require_domain(self) -> Domain:
        """Get the current domain or raise an error."""
        if not self._domain_name:
            raise ValueError("No domain in context. Use FeatureStore('org/domain/...') or with_context(domain='...')")
        org = self._require_org()
        return org.domain(self._domain_name)

    def _require_project(self) -> Project:
        """Get the current project or raise an error."""
        if not self._project_name:
            raise ValueError("No project in context. Use FeatureStore('org/domain/project') or with_context(project='...')")
        domain = self._require_domain()
        return domain.project(self._project_name)

    def _get_or_create_org(self, name: str) -> Organization:
        """Get or create an organization."""
        if name not in self._organizations:
            self._organizations[name] = Organization(name=name, owner=self._current_user)
        return self._organizations[name]

    def _get_or_create_domain(self, org_name: str, domain_name: str) -> Domain:
        """Get or create a domain."""
        org = self._get_or_create_org(org_name)
        try:
            return org.domain(domain_name)
        except DomainNotFoundError:
            return org.create_domain(domain_name, owner=self._current_user)

    def _get_or_create_project(self, org_name: str, domain_name: str, project_name: str) -> Project:
        """Get or create a project."""
        domain = self._get_or_create_domain(org_name, domain_name)
        try:
            return domain.project(project_name)
        except ProjectNotFoundError:
            return domain.create_project(project_name, owner=self._current_user)

    # =========================================================================
    # Analytics methods
    # =========================================================================

    def analyze(
        self,
        analysis: Analysis,
        freshness: Any = None,
    ) -> Any:
        """
        Run an analysis.

        Args:
            analysis: The analysis to run.
            freshness: Freshness requirements.

        Returns:
            AnalysisResult with computed data.
        """
        return self.analytics.analyze(analysis, freshness)

    def analyze_async(self, analysis: Analysis) -> Any:
        """
        Submit an analysis for async execution.

        Args:
            analysis: The analysis to run.

        Returns:
            AnalysisJob for tracking progress.
        """
        return self.analytics.analyze_async(analysis)

    def create_dashboard(
        self,
        name: str,
        description: str | None = None,
    ) -> Dashboard:
        """
        Create a dashboard.

        Args:
            name: Dashboard name.
            description: Description.

        Returns:
            The created Dashboard.
        """
        return self.analytics.create_dashboard(name, description)

    def list_dashboards(self) -> list[Dashboard]:
        """List all dashboards."""
        return self.analytics.list_dashboards()

    def create_alert(
        self,
        name: str,
        analysis: Analysis,
        condition: Any,
        notify: list[str],
        channels: list[str] | None = None,
        check_interval: str = "1h",
    ) -> Any:
        """
        Create an analytics alert.

        Args:
            name: Alert name.
            analysis: Analysis to monitor.
            condition: Condition that triggers the alert.
            notify: Notification recipients.
            channels: Notification channels.
            check_interval: How often to check.

        Returns:
            The created AnalyticsAlert.
        """
        return self.analytics.create_alert(
            name, analysis, condition, notify, channels, check_interval
        )

    def list_alerts(self) -> list[Any]:
        """List all analytics alerts."""
        return self.analytics.list_alerts()

    def __repr__(self) -> str:
        context_parts = []
        if self._org_name:
            context_parts.append(self._org_name)
        if self._domain_name:
            context_parts.append(self._domain_name)
        if self._project_name:
            context_parts.append(self._project_name)

        context = "/".join(context_parts) if context_parts else "no context"
        return f"FeatureStore({context})"
