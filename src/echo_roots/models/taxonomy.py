"""Taxonomy and Normalized Attributes models."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID, uuid4

from pydantic import Field, field_validator, model_validator

from .base import BaseModel
from .enums import AttrType, AttributeStatus, CategoryStatus, Domain, ValueStatus
# BaseModel中已經有created_at和updated_at欄位

class Category(BaseModel):
    """Category (T-layer) model representing taxonomy structure."""
    
    category_id: UUID = Field(default_factory=uuid4)
    domain: Domain
    name: str = Field(min_length=1, max_length=200)
    labels: Dict[str, str] = Field(default_factory=dict)
    parent_id: Optional[UUID] = None
    path: List[str] = Field(default_factory=list)
    path_ids: List[UUID] = Field(default_factory=list)
    status: CategoryStatus = CategoryStatus.ACTIVE
    bound_attr_ids: List[UUID] = Field(default_factory=list)
    t_version: int = Field(default=1, ge=1)
    
    @field_validator('labels')
    @classmethod
    def validate_labels(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate language labels."""
        if not isinstance(v, dict):
            raise ValueError("Labels must be a dictionary")
        for lang, label in v.items():
            if not isinstance(lang, str) or not isinstance(label, str):
                raise ValueError("All label keys and values must be strings")
            if len(label.strip()) == 0:
                raise ValueError("Label values cannot be empty")
        return v
    
    @model_validator(mode='after')
    def validate_hierarchy(self) -> 'Category':
        """Validate category hierarchy consistency."""
        if self.parent_id is None:
            # Root category
            if len(self.path) != 1 or self.path[0] != self.name:
                raise ValueError("Root category path must contain only its name")
            if len(self.path_ids) != 1 or self.path_ids[0] != self.category_id:
                raise ValueError("Root category path_ids must contain only its ID")
        else:
            # Child category
            if len(self.path) == 0:
                raise ValueError("Child category must have non-empty path")
            if self.path[-1] != self.name:
                raise ValueError("Category name must be the last element in path")
            if len(self.path_ids) != len(self.path):
                raise ValueError("path_ids length must match path length")
            if self.path_ids[-1] != self.category_id:
                raise ValueError("Category ID must be the last element in path_ids")
        
        return self


class Attribute(BaseModel):
    """Attribute (N-layer) model for normalized attributes."""
    
    attr_id: UUID = Field(default_factory=uuid4)
    domain: Domain
    name: str = Field(min_length=1, max_length=200)
    labels: Dict[str, str] = Field(default_factory=dict)
    type: AttrType = AttrType.KEYWORD
    description: Optional[str] = None
    allowed_value_ids: List[UUID] = Field(default_factory=list)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    scope: Union[str, UUID] = Field(default="global")
    status: AttributeStatus = AttributeStatus.ACTIVE
    n_version: int = Field(default=1, ge=1)
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate attribute name follows snake_case convention."""
        # 這邊要注意一下，可能以後會有問題，屬性名稱應該不能這樣限制？
        import re
        if not re.match(r'^[a-z][a-z0-9_]*[a-z0-9]$|^[a-z]$', v):
            raise ValueError("Attribute name should follow snake_case convention")
        return v
    
    @field_validator('labels')
    @classmethod
    def validate_labels(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate language labels."""
        if not isinstance(v, dict):
            raise ValueError("Labels must be a dictionary")
        for lang, label in v.items():
            if not isinstance(lang, str) or not isinstance(label, str):
                raise ValueError("All label keys and values must be strings")
            if len(label.strip()) == 0:
                raise ValueError("Label values cannot be empty")
        return v
    
    @field_validator('constraints')
    @classmethod
    def validate_constraints(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate attribute constraints."""
        allowed_constraint_keys = {
            "cardinality",  # "single" | "multiple"
            "min_length", 
            "max_length",
            "min_value",
            "max_value",
            "required",
            "pattern"
        }
        
        for key in v.keys():
            if key not in allowed_constraint_keys:
                raise ValueError(f"Unknown constraint key: {key}")
        
        # Validate cardinality
        if "cardinality" in v:
            if v["cardinality"] not in ["single", "multiple"]:
                raise ValueError("cardinality must be 'single' or 'multiple'")
        
        return v


class AttributeValue(BaseModel):
    """Attribute value (N-layer) model for normalized attribute values."""
    
    value_id: UUID = Field(default_factory=uuid4)
    attribute_id: UUID
    value: str = Field(min_length=1, max_length=200)
    labels: Dict[str, str] = Field(default_factory=dict)
    aliases: Set[str] = Field(default_factory=set)
    status: ValueStatus = ValueStatus.ACTIVE
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('labels')
    @classmethod
    def validate_labels(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate language labels."""
        if not isinstance(v, dict):
            raise ValueError("Labels must be a dictionary")
        for lang, label in v.items():
            if not isinstance(lang, str) or not isinstance(label, str):
                raise ValueError("All label keys and values must be strings")
            if len(label.strip()) == 0:
                raise ValueError("Label values cannot be empty")
        return v
    
    @field_validator('aliases')
    @classmethod
    def validate_aliases(cls, v: Set[str]) -> Set[str]:
        """Validate aliases are non-empty strings."""
        for alias in v:
            if not isinstance(alias, str) or len(alias.strip()) == 0:
                raise ValueError("All aliases must be non-empty strings")
        return v


class ValueMapping(BaseModel):
    """Value mapping model for tracking value merge/split operations."""
    # created_at欄位在BaseModel已經有了，不需要重複定義，但是有生成所以先放著
    mapping_id: UUID = Field(default_factory=uuid4)
    from_value_id: UUID
    to_value_id: UUID
    reason: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @model_validator(mode='after')
    def validate_mapping(self) -> 'ValueMapping':
        """Validate mapping doesn't map to itself."""
        if self.from_value_id == self.to_value_id:
            raise ValueError("Cannot map value to itself")
        return self