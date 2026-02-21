"""
Raise Data Types

Type definitions for feature schemas.
Supports both string shortcuts (e.g., "float32[512]") and typed constructors.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class FeatureType(ABC):
    """Base class for all feature types."""

    @abstractmethod
    def to_string(self) -> str:
        """Return the string representation of this type."""
        pass

    @abstractmethod
    def is_compatible(self, other: FeatureType) -> bool:
        """Check if this type is compatible with another type."""
        pass

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_string()!r})"


@dataclass(frozen=True)
class Int64(FeatureType):
    """64-bit integer type."""

    def to_string(self) -> str:
        return "int64"

    def is_compatible(self, other: FeatureType) -> bool:
        return isinstance(other, (Int64, Float32, Float64))


@dataclass(frozen=True)
class Float32(FeatureType):
    """32-bit floating point type."""

    def to_string(self) -> str:
        return "float32"

    def is_compatible(self, other: FeatureType) -> bool:
        return isinstance(other, (Float32, Float64, Int64))


@dataclass(frozen=True)
class Float64(FeatureType):
    """64-bit floating point type."""

    def to_string(self) -> str:
        return "float64"

    def is_compatible(self, other: FeatureType) -> bool:
        return isinstance(other, (Float64, Float32, Int64))


@dataclass(frozen=True)
class Bool(FeatureType):
    """Boolean type."""

    def to_string(self) -> str:
        return "bool"

    def is_compatible(self, other: FeatureType) -> bool:
        return isinstance(other, Bool)


@dataclass(frozen=True)
class String(FeatureType):
    """String type with optional max length."""

    max_length: int | None = None

    def to_string(self) -> str:
        if self.max_length:
            return f"string[{self.max_length}]"
        return "string"

    def is_compatible(self, other: FeatureType) -> bool:
        if not isinstance(other, String):
            return False
        if self.max_length is None:
            return True
        if other.max_length is None:
            return False
        return other.max_length <= self.max_length


@dataclass(frozen=True)
class Bytes(FeatureType):
    """Binary bytes type."""

    def to_string(self) -> str:
        return "bytes"

    def is_compatible(self, other: FeatureType) -> bool:
        return isinstance(other, Bytes)


@dataclass(frozen=True)
class Timestamp(FeatureType):
    """Timestamp type."""

    def to_string(self) -> str:
        return "timestamp"

    def is_compatible(self, other: FeatureType) -> bool:
        return isinstance(other, Timestamp)


@dataclass(frozen=True)
class Embedding(FeatureType):
    """
    Fixed-size embedding/vector type.

    Args:
        dim: The dimension of the embedding vector.
        dtype: The element type (default: "float32").
    """

    dim: int
    dtype: str = "float32"

    def __post_init__(self):
        if self.dim <= 0:
            raise ValueError(f"Embedding dimension must be positive, got {self.dim}")
        if self.dtype not in ("float16", "float32", "float64"):
            raise ValueError(f"Embedding dtype must be float16/32/64, got {self.dtype}")

    def to_string(self) -> str:
        return f"{self.dtype}[{self.dim}]"

    def is_compatible(self, other: FeatureType) -> bool:
        if not isinstance(other, Embedding):
            return False
        return self.dim == other.dim and self.dtype == other.dtype


@dataclass(frozen=True)
class Array(FeatureType):
    """
    Variable-length array type.

    Args:
        element_type: The type of elements in the array.
        max_length: Optional maximum length constraint.
    """

    element_type: FeatureType
    max_length: int | None = None

    def to_string(self) -> str:
        base = self.element_type.to_string()
        if self.max_length:
            return f"{base}[:{self.max_length}]"
        return f"{base}[]"

    def is_compatible(self, other: FeatureType) -> bool:
        if not isinstance(other, Array):
            return False
        if not self.element_type.is_compatible(other.element_type):
            return False
        if self.max_length is None:
            return True
        if other.max_length is None:
            return False
        return other.max_length <= self.max_length


@dataclass(frozen=True)
class Struct(FeatureType):
    """
    Structured/nested type with named fields.

    Args:
        fields: A dictionary mapping field names to their types.
    """

    fields: dict[str, FeatureType] = field(default_factory=dict)

    def __post_init__(self):
        # Convert dict to immutable for hashing
        if isinstance(self.fields, dict):
            object.__setattr__(self, "fields", tuple(sorted(self.fields.items())))

    def to_string(self) -> str:
        if isinstance(self.fields, tuple):
            fields_str = ", ".join(f"{k}: {v.to_string()}" for k, v in self.fields)
        else:
            fields_str = ", ".join(f"{k}: {v.to_string()}" for k, v in self.fields.items())
        return f"struct<{fields_str}>"

    def is_compatible(self, other: FeatureType) -> bool:
        if not isinstance(other, Struct):
            return False
        self_fields = dict(self.fields) if isinstance(self.fields, tuple) else self.fields
        other_fields = dict(other.fields) if isinstance(other.fields, tuple) else other.fields
        if set(self_fields.keys()) != set(other_fields.keys()):
            return False
        return all(self_fields[k].is_compatible(other_fields[k]) for k in self_fields)


# Type string pattern matching
_TYPE_PATTERNS = {
    r"^int64$": lambda m: Int64(),
    r"^float32$": lambda m: Float32(),
    r"^float64$": lambda m: Float64(),
    r"^bool$": lambda m: Bool(),
    r"^string$": lambda m: String(),
    r"^string\[(\d+)\]$": lambda m: String(max_length=int(m.group(1))),
    r"^bytes$": lambda m: Bytes(),
    r"^timestamp$": lambda m: Timestamp(),
    r"^(float16|float32|float64)\[(\d+)\]$": lambda m: Embedding(dim=int(m.group(2)), dtype=m.group(1)),
    r"^(int64|float32|float64)\[\]$": lambda m: Array(element_type=parse_dtype(m.group(1))),
    r"^(int64|float32|float64)\[:(\d+)\]$": lambda m: Array(
        element_type=parse_dtype(m.group(1)), max_length=int(m.group(2))
    ),
}


def parse_dtype(dtype: str | FeatureType) -> FeatureType:
    """
    Parse a dtype string into a FeatureType object.

    Args:
        dtype: Either a string like "float32[512]" or a FeatureType instance.

    Returns:
        A FeatureType instance.

    Raises:
        ValueError: If the dtype string is not recognized.

    Examples:
        >>> parse_dtype("int64")
        Int64('int64')
        >>> parse_dtype("float32[512]")
        Embedding('float32[512]')
        >>> parse_dtype(Float32())
        Float32('float32')
    """
    if isinstance(dtype, FeatureType):
        return dtype

    dtype = dtype.strip()

    for pattern, factory in _TYPE_PATTERNS.items():
        match = re.match(pattern, dtype)
        if match:
            return factory(match)

    raise ValueError(
        f"Unknown dtype: '{dtype}'. Supported types: int64, float32, float64, bool, "
        f"string, string[N], bytes, timestamp, float32[N] (embedding), type[] (array)"
    )


def infer_result_type(left: FeatureType, operator: str, right: FeatureType) -> FeatureType:
    """
    Infer the result type of a binary operation.

    Args:
        left: The left operand type.
        operator: The operator (+, -, *, /, etc.).
        right: The right operand type.

    Returns:
        The inferred result type.
    """
    numeric_types = (Int64, Float32, Float64)

    if operator in ("+", "-", "*", "/"):
        if isinstance(left, numeric_types) and isinstance(right, numeric_types):
            # Promote to highest precision
            if isinstance(left, Float64) or isinstance(right, Float64):
                return Float64()
            if isinstance(left, Float32) or isinstance(right, Float32):
                return Float32()
            if operator == "/":
                return Float64()  # Division always returns float
            return Int64()

    if operator in ("=", "!=", "<", ">", "<=", ">=", "AND", "OR"):
        return Bool()

    if operator == "||":  # String concatenation
        if isinstance(left, String) and isinstance(right, String):
            return String()

    raise ValueError(f"Cannot apply operator '{operator}' to types {left} and {right}")
