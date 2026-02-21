"""
Raise Exceptions

Custom exception types for the Raise Feature Store API.
"""

from __future__ import annotations

from typing import Any


class RaiseError(Exception):
    """Base exception for all Raise errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class FeatureExistsError(RaiseError):
    """Raised when attempting to create a feature that already exists."""

    def __init__(self, feature_name: str, qualified_name: str):
        super().__init__(
            f"Feature '{feature_name}' already exists: {qualified_name}",
            {"feature_name": feature_name, "qualified_name": qualified_name},
        )
        self.feature_name = feature_name
        self.qualified_name = qualified_name


class FeatureNotFoundError(RaiseError):
    """Raised when a requested feature does not exist."""

    def __init__(self, feature_name: str, suggestions: list[str] | None = None):
        message = f"Feature '{feature_name}' not found"
        if suggestions:
            message += f". Did you mean: {', '.join(suggestions)}?"
        super().__init__(message, {"feature_name": feature_name, "suggestions": suggestions})
        self.feature_name = feature_name
        self.suggestions = suggestions


class FeatureGroupNotFoundError(RaiseError):
    """Raised when a requested feature group does not exist."""

    def __init__(self, group_name: str):
        super().__init__(f"Feature group '{group_name}' not found", {"group_name": group_name})
        self.group_name = group_name


class ProjectNotFoundError(RaiseError):
    """Raised when a requested project does not exist."""

    def __init__(self, project_name: str):
        super().__init__(f"Project '{project_name}' not found", {"project_name": project_name})
        self.project_name = project_name


class DomainNotFoundError(RaiseError):
    """Raised when a requested domain does not exist."""

    def __init__(self, domain_name: str):
        super().__init__(f"Domain '{domain_name}' not found", {"domain_name": domain_name})
        self.domain_name = domain_name


class OrganizationNotFoundError(RaiseError):
    """Raised when a requested organization does not exist."""

    def __init__(self, org_name: str):
        super().__init__(f"Organization '{org_name}' not found", {"org_name": org_name})
        self.org_name = org_name


class ValidationError(RaiseError):
    """Raised when feature validation fails."""

    def __init__(
        self,
        message: str,
        code: str,
        position: int | None = None,
        suggestion: str | None = None,
    ):
        super().__init__(message, {"code": code, "position": position, "suggestion": suggestion})
        self.code = code
        self.position = position
        self.suggestion = suggestion


class AccessDeniedError(RaiseError):
    """Raised when the user does not have permission to perform an action."""

    def __init__(self, resource: str, action: str, user: str):
        super().__init__(
            f"Access denied: '{user}' cannot '{action}' on '{resource}'",
            {"resource": resource, "action": action, "user": user},
        )
        self.resource = resource
        self.action = action
        self.user = user


class CrossOrgAccessError(RaiseError):
    """Raised when cross-organization access is not permitted."""

    def __init__(self, source_org: str, target_org: str, resource: str):
        super().__init__(
            f"Cross-org access denied: '{source_org}' cannot access '{resource}' in '{target_org}'",
            {"source_org": source_org, "target_org": target_org, "resource": resource},
        )
        self.source_org = source_org
        self.target_org = target_org
        self.resource = resource


class CircularDependencyError(RaiseError):
    """Raised when a circular dependency is detected in derived features."""

    def __init__(self, cycle: list[str]):
        cycle_str = " → ".join(cycle)
        super().__init__(f"Circular dependency detected: {cycle_str}", {"cycle": cycle})
        self.cycle = cycle


class InvalidExpressionError(RaiseError):
    """Raised when a derived_from SQL expression is invalid."""

    def __init__(self, expression: str, error: str, position: int | None = None):
        super().__init__(
            f"Invalid expression: {error}",
            {"expression": expression, "error": error, "position": position},
        )
        self.expression = expression
        self.error = error
        self.position = position


class TypeMismatchError(RaiseError):
    """Raised when types are incompatible in an expression."""

    def __init__(self, expected: str, actual: str, context: str):
        super().__init__(
            f"Type mismatch in {context}: expected '{expected}', got '{actual}'",
            {"expected": expected, "actual": actual, "context": context},
        )
        self.expected = expected
        self.actual = actual
        self.context = context
