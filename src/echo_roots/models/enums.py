"""Enumerations for EchoRoots models."""

from enum import Enum


class Domain(str, Enum):
    """Domain types for categories and attributes."""
    PRODUCT = "product"
    KNOWLEDGE_BASE = "knowledge_base"


class AttrType(str, Enum):
    """Attribute value types."""
    TEXT = "text"
    KEYWORD = "keyword"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATETIME = "datetime"


class CategoryStatus(str, Enum):
    """Category status."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"


class AttributeStatus(str, Enum):
    """Attribute status."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"


class ValueStatus(str, Enum):
    """Attribute value status."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    MERGED = "merged"