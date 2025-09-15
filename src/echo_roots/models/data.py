"""
Data Models for EchoRoots system.
Implements D (Domain Data), OD (Overflow/Outlier), S (Semantic Layer), G (Governance Events).
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from pydantic import BaseModel, Field

from .enums import Domain, NormStatus, ReviewStatus, SState


# ===============================
# D (Domain Data)
# ===============================

class ItemRaw(BaseModel):
    """D_raw - Raw item data from various sources"""
    
    item_id: UUID
    domain: Domain
    title: str
    description: Optional[str] = None
    raw_category_path: List[str] = Field(default_factory=list)
    lang: Optional[str] = None  # ISO code like "zh-TW", "en"
    source: Optional[str] = None
    
    # Product-specific fields
    brand: Optional[str] = None  # 應該要移除，包含在specs裡面，對應data.md也要修改
    price: Optional[float] = None  # 建議移除，包含在specs裡面，對應data.md也要修改
    specs: Dict[str, Any] = Field(default_factory=dict)
    
    # Knowledge base-specific fields
    text_tags: List[str] = Field(default_factory=list)  # 同上，應該要在specs裡面
    text_uri: Optional[str] = None  # 同上，應該要在specs裡面

    created_at: datetime = Field(default_factory=datetime.now)  # base.py已經有created_at欄位，這裡不需要再定義
    updated_at: datetime = Field(default_factory=datetime.now)  # base.py已經有updated_at欄位，這裡不需要再定義

    class Config:
        json_schema_extra = {
            "example": {
                "item_id": "123e4567-e89b-12d3-a456-426614174000",
                "domain": "product",
                "title": "iPhone 15 Pro",
                "description": "Latest iPhone with advanced features",
                "raw_category_path": ["Electronics", "Phones", "Smartphones"],
                "lang": "zh-TW",
                "source": "apple_store",
                "brand": "Apple",
                "price": 39900.0,
                "specs": {"color": "Natural Titanium", "storage": "256GB"}
            }
        }


class ItemNorm(BaseModel):
    """D_norm - Normalized item data with category and attributes mapping"""
    
    item_id: UUID
    domain: Domain
    category_id: Optional[UUID] = None  # Reference to T layer
    attrs: Dict[str, Any] = Field(default_factory=dict)  # e.g., {"color": {"value_id": "...", "value": "綠色"}}
    evidence: Dict[str, Any] = Field(default_factory=dict)  # span/score/source fields
    
    # Version tracking
    t_version: Optional[int] = None
    n_version: Optional[int] = None
    norm_version: Optional[int] = None
    
    status: NormStatus = NormStatus.OK
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "item_id": "123e4567-e89b-12d3-a456-426614174000",
                "domain": "product",
                "category_id": "987fcdeb-51d2-4567-8901-123456789abc",
                "attrs": {
                    "color": {
                        "value_id": "color_green_001",
                        "value": "綠色"
                    }
                },
                "evidence": {
                    "color": {
                        "span": [45, 47],
                        "score": 0.95,
                        "source_field": "specs.color"
                    }
                },
                "status": "ok"
            }
        }


# ===============================
# OD (Overflow/Outlier Data)
# ===============================

class OutlierRecord(BaseModel):
    """Records for items that couldn't be processed normally"""
    
    outlier_id: UUID
    item_id: UUID
    domain: Domain
    reason: str  # e.g., "no_category", "no_attr_mapping"
    suggestion: Optional[str] = None
    review_status: ReviewStatus = ReviewStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "outlier_id": "456e7890-e89b-12d3-a456-426614174001",
                "item_id": "123e4567-e89b-12d3-a456-426614174000",
                "domain": "product",
                "reason": "no_category",
                "suggestion": "Consider creating new category for this item type",
                "review_status": "pending"
            }
        }


# ===============================
# S (Semantic Layer)
# ===============================

class SStagingObs(BaseModel):
    """S_staging_obs - Staging observations for semantic processing"""
    
    obs_id: UUID
    domain: Domain
    item_id: UUID
    category_id: Optional[UUID] = None
    attr_hint: Optional[str] = None
    value_text: str
    lang: Optional[str] = None
    model_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "obs_id": "789e0123-e89b-12d3-a456-426614174002",
                "domain": "product",
                "item_id": "123e4567-e89b-12d3-a456-426614174000",
                "category_id": "987fcdeb-51d2-4567-8901-123456789abc",
                "attr_hint": "color",
                "value_text": "深綠色",
                "lang": "zh-TW",
                "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            }
        }


class SCluster(BaseModel):
    """S_cluster - Semantic clusters for value normalization"""
    
    cluster_id: UUID
    domain: Domain
    category_id: Optional[UUID] = None
    state: SState = SState.HOT
    rep_term: str  # Representative term
    aliases: Set[str] = Field(default_factory=set)
    support: int = 0  # Number of supporting observations
    attr_distribution: Dict[str, int] = Field(default_factory=dict)
    propose_attr_id: Optional[UUID] = None
    propose_value: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "cluster_id": "abc1234-e89b-12d3-a456-426614174003",
                "domain": "product",
                "category_id": "987fcdeb-51d2-4567-8901-123456789abc",
                "state": "HOT",
                "rep_term": "綠色",
                "aliases": {"深綠色", "草綠色", "翠綠色"},
                "support": 15,
                "attr_distribution": {"color": 15},
                "propose_attr_id": "attr_color_001",
                "propose_value": "綠色"
            }
        }


# ===============================
# G (Governance Events)
# ===============================

class GovernanceEvent(BaseModel):
    """Records governance operations and state changes"""
    
    event_id: UUID
    op: str  # "snapshot", "diff", "rollback", "merge_attr", "split_attr", "deprecate_value", "update_mapping"
    actor: str
    ts: datetime = Field(default_factory=datetime.now)
    before_ref: Optional[str] = None
    after_ref: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "def5678-e89b-12d3-a456-426614174004",
                "op": "merge_attr",
                "actor": "admin@echoroots.com",
                "ts": "2025-09-15T10:30:00Z",
                "before_ref": "attr_001,attr_002",
                "after_ref": "attr_merged_001",
                "payload": {
                    "merged_attrs": ["color", "colour"],
                    "target_attr": "color",
                    "reason": "duplicate attribute consolidation"
                }
            }
        }


class ImpactedSet(BaseModel):
    """Tracks items impacted by governance operations"""
    
    batch_id: UUID
    item_ids: List[UUID] = Field(default_factory=list)
    reason: str
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "batch_id": "ghi9012-e89b-12d3-a456-426614174005",
                "item_ids": [
                    "123e4567-e89b-12d3-a456-426614174000",
                    "456e7890-e89b-12d3-a456-426614174001"
                ],
                "reason": "category_merge_operation",
                "created_at": "2025-09-15T10:30:00Z"
            }
        }