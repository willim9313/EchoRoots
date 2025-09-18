from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime
from dataclasses import dataclass

from .common import BatchKey, VersionInfo, ProcessingResult, GovernanceEvent
from ...models import SStagingObs, ItemRaw, ItemNorm, OutlierRecord, Domain

# 配置和請求/回應保持 Pydantic (邊界層)
class W3ConfigKnobs(BaseModel):
    """W3 workflow configuration"""
    route_to_s_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    allow_partial_norm: bool = Field(default=True)
    max_batch_size: int = Field(default=10000, gt=0)
    enable_semantic_staging: bool = Field(default=True)

class W3InputValidation(BaseModel):
    """Input validation for W3 workflow"""
    domain: Domain
    lang: str
    raw_category_path: Optional[str] = None
    version_info: VersionInfo
    
    @validator('lang')
    def validate_lang(cls, v):
        if len(v) != 2:
            raise ValueError('Language code must be 2 characters')
        return v.lower()

class W3RawBatch(BaseModel):
    """Raw data batch for ingestion"""
    batch_key: BatchKey
    items: List[ItemRaw]
    validation: W3InputValidation
    metadata: Dict[str, Any] = Field(default_factory=dict)

# 路由決策改為 dataclass (內部計算)
class W3RoutingDecision(str, Enum):
    """Routing decisions for items"""
    TAXONOMY = "T"
    NORM = "N"
    OUTLIER = "OD"
    STAGING = "S"

@dataclass
class W3ItemRoute:
    """Individual item routing decision - 內部計算物件"""
    item_id: str
    decision: W3RoutingDecision
    reason: str
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class W3RoutingResult:
    """Result of routing decisions - 內部計算結果"""
    total_items: int
    routes: List[W3ItemRoute]
    
    @property
    def route_counts(self) -> Dict[W3RoutingDecision, int]:
        """Count items by routing decision"""
        counts = {route: 0 for route in W3RoutingDecision}
        for item_route in self.routes:
            counts[item_route.decision] += 1
        return counts

# 處理結果改為 dataclass (內部計算結果)
@dataclass
class W3SemanticStagingResult:
    """Result of semantic staging process - 內部計算結果"""
    observations_created: int
    qdrant_points_written: int
    staging_records: List[SStagingObs]
    processing_time_ms: float
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

@dataclass
class W3NormalizationResult:
    """Result of normalization process - 內部計算結果"""
    normalized_items: List[ItemNorm]
    partial_normalizations: int
    failed_normalizations: int
    processing_time_ms: float

@dataclass
class W3OutlierResult:
    """Result of outlier detection - 內部計算結果"""
    outlier_records: List[OutlierRecord]
    processing_time_ms: float

# 輸出保持 Pydantic (API 邊界)
class W3IngestOutput(BaseModel):
    """Complete output of W3 ingest workflow"""
    batch_key: BatchKey
    normalization_result: Optional[W3NormalizationResult] = None
    semantic_result: Optional[W3SemanticStagingResult] = None
    outlier_result: Optional[W3OutlierResult] = None
    routing_result: W3RoutingResult
    governance_events: List[GovernanceEvent]
    snapshots_created: List[str] = Field(default_factory=list)
    overall_result: ProcessingResult
    
    class Config:
        arbitrary_types_allowed = True  # 允許 dataclass 類型

class W3IngestRequest(BaseModel):
    """Complete request for W3 ingest workflow"""
    raw_batch: W3RawBatch
    config: W3ConfigKnobs = Field(default_factory=W3ConfigKnobs)
    force_reprocess: bool = Field(default=False)
    dry_run: bool = Field(default=False)

class W3IngestResponse(BaseModel):
    """Response from W3 ingest workflow"""
    request_id: str
    timestamp: datetime
    output: W3IngestOutput
    execution_time_ms: float
    idempotency_key: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True

# 檢查結果改为 dataclass (內部檢查邏輯)
@dataclass
class W3PreCheckResult:
    """Result of pre-flight checks - 內部檢查結果"""
    version_aligned: bool
    batch_valid: bool
    dependencies_available: bool
    estimated_processing_time_ms: Optional[float] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
    
    @property
    def can_proceed(self) -> bool:
        """Check if workflow can proceed"""
        return self.version_aligned and self.batch_valid and self.dependencies_available

# 冪等性檢查保持 Pydantic (涉及外部狀態)
class W3IdempotencyCheck(BaseModel):
    """Idempotency control for W3 workflow"""
    batch_key: BatchKey
    event_id: str
    checksum: str
    previous_result: Optional[W3IngestResponse] = None
    
    @property
    def is_duplicate(self) -> bool:
        """Check if this is a duplicate request"""
        return self.previous_result is not None

# 回滾操作保持 Pydantic (邊界操作)
class W3RollbackRequest(BaseModel):
    """Request to rollback W3 operations"""
    batch_key: BatchKey
    target_snapshot: str
    reason: str
    force: bool = Field(default=False)

class W3RollbackResponse(BaseModel):
    """Response from rollback operation"""
    success: bool
    rolled_back_operations: List[str]
    restored_snapshot: str
    timestamp: datetime
    errors: List[str] = Field(default_factory=list)