"""LLM Input/Output models for EchoRoots system."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

from pydantic import BaseModel, Field

from .base import BaseModel as EchoRootsBaseModel
from .enums import Domain, LLMTaskType, AttributeCardinality, ProcessingStatus


# ===============================
# Common Input Models
# ===============================

class LLMCommonInput(EchoRootsBaseModel):
    """Common input fields for all LLM tasks."""

    item_id: UUID
    domain: Domain
    lang: Optional[str] = "zh-TW"  # ISO language code
    title: str = Field(min_length=1)
    description: Optional[str] = None
    specs: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    raw_category_path: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)


# ===============================
# Evidence Models
# ===============================

class Evidence(BaseModel):
    """Evidence model for LLM extraction results."""

    source: str = Field(description="Source field like 'title', 'description', 'specs.storage'")
    span: Tuple[int, int] = Field(description="Start and end position in source text")
    text: str = Field(description="Extracted text")
    score: float = Field(ge=0.0, le=1.0, description="Confidence score")
    model_id: str = Field(description="Model identifier like 'gpt-4o-mini', 'sentence-transformers/...'")


# ===============================
# Category Classification Models
# ===============================

class CategoryOption(BaseModel):
    """Category option for LLM selection."""
    
    option_id: int = Field(ge=1, description="Option number for LLM selection")
    category_id: UUID = Field(description="Internal category UUID")
    path: List[str] = Field(description="Category path like ['Electronics', 'Phones', 'Smartphones']")
    labels: Dict[str, str] = Field(default_factory=dict, description="Multi-language labels")


class CategoryClassificationInput(LLMCommonInput):
    """Input for category classification task."""

    task_type: LLMTaskType = LLMTaskType.CATEGORY_CLASSIFICATION
    options: List[CategoryOption] = Field(min_length=1, description="Category options for selection")
    max_selections: int = Field(default=1, ge=1, description="Maximum number of categories to select")


class CategoryClassificationOutput(BaseModel):
    """Output for category classification task."""
    
    selected_option: int = Field(ge=1, description="Selected option number")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    evidence: List[Evidence] = Field(default_factory=list, description="Supporting evidence")
    reasoning: Optional[str] = Field(None, description="LLM reasoning (optional)")


# ===============================
# Attribute Extraction Models  
# ===============================

class AttributeValueOption(BaseModel):
    """Attribute value option for LLM selection."""
    
    option_id: int = Field(ge=1, description="Option number for LLM selection")
    value_id: UUID = Field(description="Internal value UUID")
    value: str = Field(description="Display value")
    aliases: List[str] = Field(default_factory=list, description="Alternative forms")


class AttributeOption(BaseModel):
    """Attribute option with its possible values."""
    
    attr_key: str = Field(description="Attribute key like 'A1', 'A2'")
    attr_id: UUID = Field(description="Internal attribute UUID")
    attr_name: str = Field(description="Attribute name like 'color', 'storage'")
    attr_labels: Dict[str, str] = Field(default_factory=dict, description="Multi-language labels")
    values: Dict[str, AttributeValueOption] = Field(description="Value options keyed by option_id")
    cardinality: AttributeCardinality = Field(default=AttributeCardinality.SINGLE, description="'single' or 'multiple'")
    required: bool = Field(default=False, description="Whether this attribute is required")


class AttributeExtractionInput(LLMCommonInput):
    """Input for attribute extraction task."""

    task_type: LLMTaskType = LLMTaskType.ATTRIBUTE_EXTRACTION
    category_id: UUID = Field(description="Confirmed category ID")
    category_path: List[str] = Field(description="Category path for context")
    attributes: Dict[str, AttributeOption] = Field(description="Attributes keyed by attr_key")


class AttributeSelection(BaseModel):
    """Selected values for a single attribute."""
    
    selected_options: List[int] = Field(description="Selected option numbers")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence for this attribute")
    evidence: List[Evidence] = Field(default_factory=list, description="Supporting evidence")


class AttributeExtractionOutput(BaseModel):
    """Output for attribute extraction task."""
    
    selected_options: Dict[str, AttributeSelection] = Field(
        description="Selections keyed by attr_key like 'A1', 'A2'"
    )
    overall_confidence: float = Field(ge=0.0, le=1.0, description="Overall task confidence")
    reasoning: Optional[str] = Field(None, description="LLM reasoning (optional)")


# ===============================
# Residual Value Models
# ===============================

class ResidualValue(BaseModel):
    """Residual value that goes to S-STAGING."""

    obs_id: UUID
    attr_hint: Optional[str] = Field(description="Attribute hint like 'color', 'brand'")
    value_text: str = Field(description="Extracted text value")
    item_id: UUID
    category_id: Optional[UUID] = None
    state: str = Field(default="STAGING")
    model_id: str = Field(description="Source model identifier")
    extraction_context: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Context about how this value was extracted"
    )


# ===============================
# Processing Results Models
# ===============================

class CategoryProcessingResult(BaseModel):
    """Result after processing category classification."""

    item_id: UUID
    selected_category_id: UUID
    category_path: List[str]
    confidence: float
    evidence: List[Evidence]
    model_id: str
    processed_at: datetime = Field(default_factory=datetime.utcnow)


class AttributeProcessingResult(BaseModel):
    """Result after processing attribute extraction."""
    
    item_id: UUID
    category_id: UUID
    extracted_attrs: Dict[str, Any] = Field(
        description="Extracted attributes in format {'color': {'value_id': '...', 'value': '綠色'}}"
    )
    evidence: Dict[str, List[Evidence]] = Field(
        description="Evidence keyed by attribute name"
    )
    residual_values: List[ResidualValue] = Field(
        default_factory=list,
        description="Values that couldn't be mapped and go to S-STAGING"
    )
    model_id: str
    processed_at: datetime = Field(default_factory=datetime.utcnow)


# ===============================
# Batch Processing Models
# ===============================

class LLMBatchRequest(BaseModel):
    """Batch request for LLM processing."""
    
    batch_id: UUID
    task_type: str = Field(description="'category_classification' or 'attribute_extraction'")
    items: List[Union[CategoryClassificationInput, AttributeExtractionInput]]
    model_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="LLM configuration like temperature, max_tokens"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


class LLMBatchResponse(BaseModel):
    """Batch response from LLM processing."""
    
    batch_id: UUID
    task_type: str
    results: List[Union[CategoryProcessingResult, AttributeProcessingResult]]
    failed_items: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Items that failed processing with error details"
    )
    model_id: str
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time_seconds: Optional[float] = None