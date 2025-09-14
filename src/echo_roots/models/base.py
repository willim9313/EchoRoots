"""Base model configurations."""

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel as PydanticBaseModel, ConfigDict, Field


class BaseModel(PydanticBaseModel):
    """Base model with common configurations."""
    
    model_config = ConfigDict(
        # Enable validation on assignment
        validate_assignment=True,
        # Use enum values instead of enum objects
        use_enum_values=True,
        # Populate by name for field aliases
        populate_by_name=True,
        # Validate default values
        validate_default=True,
        # Allow arbitrary types for complex fields
        arbitrary_types_allowed=True,
    )
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def model_dump_for_db(self, **kwargs) -> Dict[str, Any]:
        """Dump model data for database storage."""
        data = self.model_dump(**kwargs)
        # Convert datetime to ISO string for database storage
        for field_name, field_value in data.items():
            if isinstance(field_value, datetime):
                data[field_name] = field_value.isoformat()
        return data