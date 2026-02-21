"""
Raise Expression Validator

Validates SQL-like expressions for derived features.
Checks syntax, feature references, type compatibility, and circular dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from raise_.models.feature import Feature
    from raise_.models.lineage import FeatureReference


@dataclass
class ValidationError:
    """
    A validation error.

    Attributes:
        code: Error code (e.g., "UNKNOWN_REFERENCE", "TYPE_MISMATCH").
        message: Human-readable error message.
        position: Character position in the expression.
        suggestion: Suggested fix.
    """

    code: str
    message: str
    position: int | None = None
    suggestion: str | None = None


@dataclass
class ValidationWarning:
    """
    A validation warning.

    Warnings don't prevent creation but indicate potential issues.

    Attributes:
        code: Warning code (e.g., "POSSIBLE_DIVISION_BY_ZERO").
        message: Human-readable warning message.
        position: Character position in the expression.
    """

    code: str
    message: str
    position: int | None = None


@dataclass
class ValidationResult:
    """
    Result of expression validation.

    Attributes:
        valid: Whether the expression is valid.
        errors: List of validation errors.
        warnings: List of validation warnings.
        inferred_type: The inferred result type of the expression.
        references: All feature references found in the expression.
        cross_org_references: Feature references to other organizations.
    """

    valid: bool = True
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationWarning] = field(default_factory=list)
    inferred_type: Any = None
    references: list[FeatureReference] = field(default_factory=list)
    cross_org_references: list[FeatureReference] = field(default_factory=list)

    def add_error(
        self,
        code: str,
        message: str,
        position: int | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Add a validation error."""
        self.valid = False
        self.errors.append(ValidationError(code, message, position, suggestion))

    def add_warning(
        self,
        code: str,
        message: str,
        position: int | None = None,
    ) -> None:
        """Add a validation warning."""
        self.warnings.append(ValidationWarning(code, message, position))


# Supported SQL functions and their signatures
SUPPORTED_FUNCTIONS = {
    # Aggregations
    "AVG": {"args": 1, "returns": "float64"},
    "SUM": {"args": 1, "returns": "numeric"},
    "MIN": {"args": 1, "returns": "same"},
    "MAX": {"args": 1, "returns": "same"},
    "COUNT": {"args": 1, "returns": "int64"},
    "STDDEV": {"args": 1, "returns": "float64"},
    "VARIANCE": {"args": 1, "returns": "float64"},
    # Math
    "ABS": {"args": 1, "returns": "same"},
    "CEIL": {"args": 1, "returns": "int64"},
    "FLOOR": {"args": 1, "returns": "int64"},
    "ROUND": {"args": (1, 2), "returns": "numeric"},
    "LOG": {"args": 1, "returns": "float64"},
    "EXP": {"args": 1, "returns": "float64"},
    "POWER": {"args": 2, "returns": "float64"},
    "SQRT": {"args": 1, "returns": "float64"},
    # Vector operations
    "DOT": {"args": 2, "returns": "float32", "vector": True},
    "COSINE_SIMILARITY": {"args": 2, "returns": "float32", "vector": True},
    "L2_DISTANCE": {"args": 2, "returns": "float32", "vector": True},
    "NORM": {"args": 1, "returns": "float32", "vector": True},
    # String
    "CONCAT": {"args": (2, None), "returns": "string"},
    "LOWER": {"args": 1, "returns": "string"},
    "UPPER": {"args": 1, "returns": "string"},
    "TRIM": {"args": 1, "returns": "string"},
    "SUBSTRING": {"args": (2, 3), "returns": "string"},
    "LENGTH": {"args": 1, "returns": "int64"},
    # Conditional
    "COALESCE": {"args": (2, None), "returns": "same"},
    "NULLIF": {"args": 2, "returns": "same"},
    "IF": {"args": 3, "returns": "same"},
}

# SQL keywords
SQL_KEYWORDS = {
    "CASE", "WHEN", "THEN", "ELSE", "END",
    "AND", "OR", "NOT", "IS", "NULL", "TRUE", "FALSE",
    "OVER", "PARTITION", "BY", "ORDER", "ROWS", "RANGE",
    "PRECEDING", "FOLLOWING", "CURRENT", "ROW", "UNBOUNDED",
}


class ExpressionParser:
    """
    Parser for SQL-like expressions.

    Extracts feature references, validates syntax, and infers types.
    """

    def __init__(
        self,
        context: dict[str, str],
        available_features: dict[str, Feature],
    ):
        """
        Initialize the parser.

        Args:
            context: Current namespace context.
            available_features: Features available for reference.
        """
        self.context = context
        self.available_features = available_features
        self.references: list[FeatureReference] = []

    def parse(self, expression: str) -> ValidationResult:
        """
        Parse and validate an expression.

        Args:
            expression: The SQL-like expression to parse.

        Returns:
            ValidationResult with errors, warnings, and references.
        """
        from raise_.models.lineage import FeatureReference

        result = ValidationResult()

        # Basic syntax validation
        if not expression or not expression.strip():
            result.add_error("EMPTY_EXPRESSION", "Expression cannot be empty")
            return result

        # Check for balanced parentheses
        paren_count = 0
        for i, char in enumerate(expression):
            if char == "(":
                paren_count += 1
            elif char == ")":
                paren_count -= 1
            if paren_count < 0:
                result.add_error(
                    "UNBALANCED_PARENS",
                    "Unbalanced parentheses: unexpected ')'",
                    position=i,
                )
                return result

        if paren_count != 0:
            result.add_error(
                "UNBALANCED_PARENS",
                f"Unbalanced parentheses: {paren_count} unclosed '('",
            )
            return result

        # Extract and validate feature references
        # First, remove all quoted strings to avoid matching content inside them
        expr_no_strings = re.sub(r"'[^']*'", "", expression)
        expr_no_strings = re.sub(r'"[^"]*"', "", expr_no_strings)

        # Pattern matches: word characters, dots for cross-group, @ for cross-org
        reference_pattern = r"(@?[a-zA-Z_][a-zA-Z0-9_\-]*(?:[./][a-zA-Z_][a-zA-Z0-9_\-]*)*)"

        for match in re.finditer(reference_pattern, expr_no_strings):
            token = match.group(1)

            # Skip SQL keywords and functions
            if token.upper() in SQL_KEYWORDS or token.upper() in SUPPORTED_FUNCTIONS:
                continue

            # Skip numeric literals
            if re.match(r"^\d+\.?\d*$", token):
                continue

            try:
                ref = FeatureReference.parse(token, self.context)
                result.references.append(ref)

                if ref.is_cross_org:
                    result.cross_org_references.append(ref)

                # Check if referenced feature exists (for local references)
                if not ref.is_cross_org and ref.feature_group == self.context.get("feature_group"):
                    if ref.name not in self.available_features:
                        # Find similar names for suggestion
                        similar = [
                            n for n in self.available_features.keys()
                            if n.startswith(ref.name[:2]) or ref.name.startswith(n[:2])
                        ]
                        suggestion = f"Did you mean: {', '.join(similar)}?" if similar else None

                        result.add_error(
                            "UNKNOWN_REFERENCE",
                            f"Unknown feature reference: '{token}'",
                            position=match.start(),
                            suggestion=suggestion,
                        )

            except ValueError as e:
                result.add_error(
                    "INVALID_REFERENCE",
                    f"Invalid reference format: '{token}' - {e}",
                    position=match.start(),
                )

        # Validate function calls
        func_pattern = r"(\w+)\s*\("
        for match in re.finditer(func_pattern, expression):
            func_name = match.group(1).upper()
            if func_name not in SUPPORTED_FUNCTIONS and func_name not in SQL_KEYWORDS:
                result.add_error(
                    "UNKNOWN_FUNCTION",
                    f"Unknown function: '{match.group(1)}'",
                    position=match.start(),
                )

        # Check for potential issues (warnings)
        if "/" in expression and "NULLIF" not in expression.upper():
            # Potential division by zero
            result.add_warning(
                "POSSIBLE_DIVISION_BY_ZERO",
                "Division detected without NULLIF protection. Consider using NULLIF(divisor, 0).",
            )

        # Infer result type (simplified)
        result.inferred_type = self._infer_type(expression)

        return result

    def _infer_type(self, expression: str) -> str:
        """Infer the result type of an expression (simplified)."""
        from raise_.models.types import Float64, Int64, String, Bool

        expression_upper = expression.upper()

        # Check for specific function return types
        for func, spec in SUPPORTED_FUNCTIONS.items():
            if func in expression_upper:
                if spec["returns"] == "float64":
                    return "float64"
                elif spec["returns"] == "float32":
                    return "float32"
                elif spec["returns"] == "int64":
                    return "int64"
                elif spec["returns"] == "string":
                    return "string"

        # Check for CASE expression (returns based on THEN clauses)
        if "CASE" in expression_upper:
            return "dynamic"

        # Check for comparison operators
        if any(op in expression for op in ["=", "!=", "<", ">", "<=", ">="]):
            if "AND" in expression_upper or "OR" in expression_upper:
                return "bool"

        # Check for string operations
        if "||" in expression or "CONCAT" in expression_upper:
            return "string"

        # Default to float64 for arithmetic
        if any(op in expression for op in ["+", "-", "*", "/"]):
            return "float64"

        return "dynamic"


def validate_expression(
    expression: str,
    context: dict[str, str],
    available_features: dict[str, Any],
    level: str = "standard",
) -> ValidationResult:
    """
    Validate a derived_from expression.

    Args:
        expression: The SQL-like expression to validate.
        context: Current namespace context.
        available_features: Features available for reference.
        level: Validation level ("strict", "standard", "permissive").

    Returns:
        ValidationResult with errors, warnings, and inferred type.
    """
    parser = ExpressionParser(context, available_features)
    result = parser.parse(expression)

    # Apply strictness level
    if level == "strict" and result.warnings:
        # Promote warnings to errors in strict mode
        for warning in result.warnings:
            result.add_error(warning.code, warning.message, warning.position)
        result.warnings = []
    elif level == "permissive":
        # Clear all but syntax errors
        result.errors = [e for e in result.errors if e.code in ("UNBALANCED_PARENS", "EMPTY_EXPRESSION")]
        result.valid = len(result.errors) == 0

    return result


def detect_circular_dependency(
    feature_name: str,
    expression: str,
    existing_features: dict[str, Any],
    context: dict[str, str],
) -> list[str] | None:
    """
    Detect circular dependencies in derived features.

    Args:
        feature_name: Name of the feature being created.
        expression: The derived_from expression.
        existing_features: Existing features in the group.
        context: Current namespace context.

    Returns:
        List of feature names forming a cycle, or None if no cycle.
    """
    from raise_.models.lineage import FeatureReference

    # Extract references from expression
    reference_pattern = r"(?<!['\"])(@?[\w][\w\-]*(?:[./][\w][\w\-]*)*)(?!['\"])"
    refs = []

    for match in re.finditer(reference_pattern, expression):
        token = match.group(1)
        if token.upper() not in SQL_KEYWORDS and token.upper() not in SUPPORTED_FUNCTIONS:
            try:
                ref = FeatureReference.parse(token, context)
                if ref.feature_group == context.get("feature_group"):
                    refs.append(ref.name)
            except ValueError:
                pass

    # DFS to detect cycles
    def has_cycle(current: str, visited: set, path: list) -> list | None:
        if current in path:
            cycle_start = path.index(current)
            return path[cycle_start:] + [current]

        if current in visited:
            return None

        visited.add(current)
        path.append(current)

        # Get dependencies of current feature
        if current in existing_features:
            feature = existing_features[current]
            if feature.derived_from:
                for match in re.finditer(reference_pattern, feature.derived_from):
                    token = match.group(1)
                    try:
                        ref = FeatureReference.parse(token, context)
                        if ref.feature_group == context.get("feature_group"):
                            cycle = has_cycle(ref.name, visited, path.copy())
                            if cycle:
                                return cycle
                    except ValueError:
                        pass

        return None

    # Check if adding this feature creates a cycle
    for ref_name in refs:
        if ref_name == feature_name:
            return [feature_name, feature_name]

        cycle = has_cycle(ref_name, {feature_name}, [feature_name])
        if cycle:
            return cycle

    return None
