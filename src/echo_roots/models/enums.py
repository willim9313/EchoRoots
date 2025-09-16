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


class NormStatus(str, Enum):
    """Normalization status for ItemNorm"""
    OK = "ok"
    PARTIAL = "partial"
    FAILED = "failed"


class ReviewStatus(str, Enum):
    """Review status for outlier records"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class SState(str, Enum):
    """State for semantic clusters"""
    STAGING = "STAGING"
    HOT = "HOT"
    ARCHIVE = "ARCHIVE"


# LLM-related enums
class LLMTaskType(str, Enum):
    """LLM task types."""
    CATEGORY_CLASSIFICATION = "category_classification"
    ATTRIBUTE_EXTRACTION = "attribute_extraction"


class AttributeCardinality(str, Enum):
    """Attribute cardinality for LLM tasks."""
    SINGLE = "single"
    MULTIPLE = "multiple"


class ProcessingStatus(str, Enum):
    """Processing status for LLM tasks."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"