"""EchoRoots data models package."""

from .enums import AttrType, Domain, CategoryStatus, AttributeStatus, ValueStatus
from .taxonomy import Category, Attribute, AttributeValue, ValueMapping
from .base import BaseModel

from .data import (
    GovernanceEvent,
    ImpactedSet,
    ItemNorm,
    ItemRaw,
    OutlierRecord,
    SCluster,
    SStagingObs,
)

from .enums import (
    Domain, 
    NormStatus, 
    ReviewStatus, 
    SState,
    LLMTaskType,
    AttributeCardinality,
    ProcessingStatus,
)

# Add LLM I/O models
from .llm_io import (
    LLMCommonInput,
    Evidence,
    CategoryOption,
    CategoryClassificationInput,
    CategoryClassificationOutput,
    AttributeValueOption,
    AttributeOption,
    AttributeExtractionInput,
    AttributeSelection,
    AttributeExtractionOutput,
    ResidualValue,
    CategoryProcessingResult,
    AttributeProcessingResult,
    LLMBatchRequest,
    LLMBatchResponse
)

__all__ = [
    "AttrType",
    "Domain", 
    "CategoryStatus",
    "AttributeStatus",
    "ValueStatus",
    "Category",
    "Attribute", 
    "AttributeValue",
    "ValueMapping",
    "BaseModel",
    # Data models
    "ItemRaw",
    "ItemNorm",
    "OutlierRecord",
    "SStagingObs",
    "SCluster",
    "GovernanceEvent",
    "ImpactedSet",
    # Additional enums
    "NormStatus",
    "ReviewStatus",
    "SState",
    # LLM I/O models
    "LLMCommonInput",
    "Evidence",
    "CategoryOption",
    "CategoryClassificationInput", 
    "CategoryClassificationOutput",
    "AttributeValueOption",
    "AttributeOption",
    "AttributeExtractionInput",
    "AttributeSelection",
    "AttributeExtractionOutput",
    "ResidualValue",
    "CategoryProcessingResult",
    "AttributeProcessingResult",
    "LLMBatchRequest",
    "LLMBatchResponse",
    "LLMTaskType",
    "AttributeCardinality",
    "ProcessingStatus"
]