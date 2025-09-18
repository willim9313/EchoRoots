"""
Workflow W5 - Maintenance & Governance Contracts
維護與治理工作流程合約定義
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set, Union
from datetime import datetime
from enum import Enum
from uuid import UUID
from pydantic import BaseModel, validator, Field

from .common import (
    WorkflowConfig,
    WorkflowResult,
    ValidationResult,
    BatchMetadata,
    BaseProcessor,
    BaseValidator,
    GovernanceEvent as CommonGovernanceEvent,
    GovernanceEventType
)
from echo_roots.models.data import GovernanceEvent as ModelGovernanceEvent


# W5 專屬枚舉
class ProposalAction(str, Enum):
    """提案審核動作"""
    ACCEPT = "accept"
    REJECT = "reject"
    DEFER = "defer"


class ChangeType(str, Enum):
    """分類法變更類型"""
    ATTR_MERGE = "ATTR_MERGE"
    ATTR_SPLIT = "ATTR_SPLIT"
    VALUE_DEPRECATE = "VALUE_DEPRECATE"
    MAPPING_UPDATE = "MAPPING_UPDATE"
    CLASSIFICATION_UPDATE = "CLASSIFICATION_UPDATE"


class ConflictResolutionStrategy(str, Enum):
    """衝突解決策略"""
    FAIL = "fail"
    SKIP = "skip"
    MERGE = "merge"


class MaintenanceOperationType(str, Enum):
    """維護操作類型"""
    PROPOSAL_REVIEW = "PROPOSAL_REVIEW"
    TAXONOMY_UPDATE = "TAXONOMY_UPDATE"
    MAPPING_SYNC = "MAPPING_SYNC"
    SNAPSHOT_CREATE = "SNAPSHOT_CREATE"
    ROLLBACK = "ROLLBACK"


# 內部計算層 - 使用 dataclass
@dataclass
class ProposalImpactAnalysis:
    """提案影響分析 - 內部計算物件"""
    proposal_id: UUID
    affected_taxonomy_nodes: Set[str] = field(default_factory=set)
    affected_data_nodes: int = 0
    estimated_backfill_cost: int = 0
    conflict_risks: List[str] = field(default_factory=list)
    dependency_chain: List[str] = field(default_factory=list)
    
    def has_high_impact(self, threshold: int = 1000) -> bool:
        """檢查是否為高影響度變更"""
        return self.affected_data_nodes > threshold or len(self.conflict_risks) > 0


@dataclass
class ConflictCheck:
    """衝突檢查結果 - 內部計算物件"""
    has_conflicts: bool = False
    conflicting_proposals: List[UUID] = field(default_factory=list)
    version_conflicts: List[str] = field(default_factory=list)
    data_integrity_risks: List[str] = field(default_factory=list)
    recommended_resolution: Optional[str] = None
    
    def add_conflict(self, proposal_id: UUID, reason: str) -> None:
        """添加衝突"""
        self.has_conflicts = True
        self.conflicting_proposals.append(proposal_id)
        self.data_integrity_risks.append(reason)


@dataclass
class ChangeExecution:
    """變更執行計畫 - 內部計算物件"""
    execution_order: List[UUID] = field(default_factory=list)
    neo4j_operations: List[Dict[str, str]] = field(default_factory=list)
    value_mapping_updates: Dict[str, str] = field(default_factory=dict)
    rollback_plan: List[str] = field(default_factory=list)
    estimated_duration_seconds: int = 0
    
    def add_operation(self, operation_type: str, query: str, rollback_query: str) -> None:
        """添加操作"""
        self.neo4j_operations.append({
            "type": operation_type,
            "query": query
        })
        self.rollback_plan.append(rollback_query)


@dataclass
class SnapshotMetadata:
    """快照元數據 - 內部計算物件"""
    snapshot_id: str
    version_point_id: str
    timestamp: datetime
    neo4j_node_count: int = 0
    neo4j_relationship_count: int = 0
    value_mapping_record_count: int = 0
    checksum: str = ""
    
    def is_consistent_with(self, other: 'SnapshotMetadata') -> bool:
        """檢查與另一快照的一致性"""
        return (self.neo4j_node_count == other.neo4j_node_count and
                self.value_mapping_record_count == other.value_mapping_record_count)


@dataclass
class VersionPoint:
    """版本點 - 內部計算物件"""
    version_id: str
    timestamp: datetime
    neo4j_checksum: str = ""
    value_mapping_checksum: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_consistent: bool = True
    
    def validate_consistency(self) -> ValidationResult:
        """驗證版本點一致性"""
        result = ValidationResult(is_valid=True)
        if not self.neo4j_checksum or not self.value_mapping_checksum:
            result.add_error("Missing checksum data")
            self.is_consistent = False
        return result


# 邊界層 - 使用 Pydantic BaseModel
class ProposalReviewInput(BaseModel):
    """提案審核輸入 - API 邊界層"""
    proposal_id: UUID
    action: ProposalAction
    reviewer_id: str
    review_comment: Optional[str] = None
    review_timestamp: datetime = Field(default_factory=datetime.now)
    
    @validator('reviewer_id')
    def validate_reviewer_id(cls, v):
        if not v or not v.strip():
            raise ValueError("reviewer_id is required")
        return v.strip()


class TaxonomyChangeResult(BaseModel):
    """分類法變更結果 - DB I/O 邊界層"""
    change_type: ChangeType
    affected_nodes: List[str] = Field(default_factory=list)
    affected_relationships: List[str] = Field(default_factory=list)
    neo4j_queries_executed: List[str] = Field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None


class MaintenanceGovernanceEvent(BaseModel):
    """維護治理事件記錄 - DB I/O 邊界層"""
    event_id: str
    event_type: MaintenanceOperationType
    timestamp: datetime = Field(default_factory=datetime.now)
    affected_proposals: List[UUID] = Field(default_factory=list)
    changes: List[TaxonomyChangeResult] = Field(default_factory=list)
    version_point_before: str
    version_point_after: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_summary(self) -> Dict[str, Any]:
        """獲取事件摘要"""
        return {
            "total_proposals": len(self.affected_proposals),
            "total_changes": len(self.changes),
            "successful_changes": sum(1 for c in self.changes if c.success),
            "failed_changes": sum(1 for c in self.changes if not c.success),
            "execution_duration": sum(c.execution_time_ms or 0 for c in self.changes)
        }


class SideEffectEvent(BaseModel):
    """副作用事件記錄 - DB I/O 邊界層"""
    effect_type: MaintenanceOperationType
    timestamp: datetime = Field(default_factory=datetime.now)
    details: Dict[str, Union[str, int, List[str]]] = Field(default_factory=dict)
    event_id: str
    
    @validator('event_id')
    def validate_event_id(cls, v):
        if not v:
            raise ValueError("event_id is required")
        return v


class W5Config(BaseModel):
    """W5 維護配置 - API 邊界層"""
    workflow_id: str
    domain: str
    dry_run: bool = False
    auto_alias_from_values: bool = True
    max_proposals_per_batch: int = Field(default=10, ge=1, le=100)
    require_pre_change_snapshot: bool = True
    require_post_change_snapshot: bool = True
    conflict_resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.FAIL
    timeout_seconds: int = Field(default=7200, gt=0)  # 2 hours default
    
    @validator('domain')
    def validate_domain_not_empty(cls, v):
        if not v:
            raise ValueError("domain is required")
        return v
    
    class Config:
        extra = "forbid"


class W5Input(BaseModel):
    """W5 輸入合約 - API 邊界層"""
    proposals: List[ProposalReviewInput]
    batch_metadata: BatchMetadata
    version_point_id: Optional[str] = None
    event_metadata: Dict[str, str] = Field(default_factory=dict)
    
    @validator('proposals')
    def validate_proposals_not_empty(cls, v):
        if not v:
            raise ValueError("proposals cannot be empty")
        if len(v) > 100:  # 防止過大批次
            raise ValueError("too many proposals in single batch (max 100)")
        return v
    
    @validator('proposals', each_item=True)
    def validate_each_proposal(cls, v):
        # Pydantic 會自動驗證 ProposalReviewInput
        return v
    
    class Config:
        arbitrary_types_allowed = True


class W5Output(BaseModel):
    """W5 輸出合約 - API 邊界層"""
    governance_events: List[MaintenanceGovernanceEvent] = Field(default_factory=list)
    updated_value_mapping_count: int = 0
    neo4j_changes_applied: List[TaxonomyChangeResult] = Field(default_factory=list)
    backfill_plan: Optional[Dict[str, List[str]]] = None
    version_consistency_verified: bool = False
    side_effects: List[SideEffectEvent] = Field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    
    def get_maintenance_summary(self) -> Dict[str, Any]:
        """獲取維護摘要統計"""
        total_successful_changes = sum(1 for c in self.neo4j_changes_applied if c.success)
        total_failed_changes = sum(1 for c in self.neo4j_changes_applied if not c.success)
        
        return {
            "total_governance_events": len(self.governance_events),
            "total_proposals_processed": sum(len(e.affected_proposals) for e in self.governance_events),
            "total_changes_applied": len(self.neo4j_changes_applied),
            "successful_changes": total_successful_changes,
            "failed_changes": total_failed_changes,
            "value_mapping_updates": self.updated_value_mapping_count,
            "version_consistency": self.version_consistency_verified,
            "side_effects_count": len(self.side_effects),
            "has_backfill_plan": self.backfill_plan is not None
        }
    
    @validator('governance_events', 'neo4j_changes_applied', 'side_effects')
    def validate_lists_consistent(cls, v):
        return v or []
    
    class Config:
        arbitrary_types_allowed = True


# 錯誤合約
class MaintenanceError(BaseModel):
    """維護錯誤 - API 邊界層"""
    error_type: str
    proposal_id: Optional[UUID] = None
    error_message: str
    recovery_suggestions: List[str] = Field(default_factory=list)
    is_recoverable: bool = True
    timestamp: datetime = Field(default_factory=datetime.now)


class MaintenanceValidationError(MaintenanceError):
    """維護驗證錯誤 - API 邊界層"""
    validation_failures: List[str] = Field(default_factory=list)
    affected_fields: List[str] = Field(default_factory=list)


# 處理器介面
class ProposalImpactAnalyzer(BaseValidator):
    """提案影響分析器"""
    
    def analyze_proposal_impact(self, proposal: ProposalReviewInput, config: W5Config) -> ProposalImpactAnalysis:
        """分析提案影響"""
        raise NotImplementedError
    
    def check_conflicts(self, proposals: List[ProposalReviewInput]) -> ConflictCheck:
        """檢查提案間衝突"""
        raise NotImplementedError


class ChangeExecutor(BaseProcessor):
    """變更執行器"""
    
    def __init__(self, config: W5Config):
        super().__init__(config)
        self.config: W5Config = config
    
    def plan_execution(self, proposals: List[ProposalReviewInput]) -> ChangeExecution:
        """規劃執行計畫"""
        raise NotImplementedError
    
    def execute_changes(self, execution_plan: ChangeExecution) -> List[TaxonomyChangeResult]:
        """執行變更"""
        raise NotImplementedError
    
    def create_rollback_point(self) -> VersionPoint:
        """建立回滾點"""
        raise NotImplementedError


class VersionManager(BaseValidator):
    """版本管理器"""
    
    def create_version_point(self, reason: str) -> VersionPoint:
        """建立版本點"""
        raise NotImplementedError
    
    def verify_consistency(self, version_point: VersionPoint) -> ValidationResult:
        """驗證一致性"""
        raise NotImplementedError
    
    def compare_versions(self, before: VersionPoint, after: VersionPoint) -> Dict[str, Any]:
        """比較版本差異"""
        raise NotImplementedError


class W5Processor(BaseProcessor):
    """W5 主處理器"""
    
    def __init__(self, config: W5Config):
        super().__init__(config)
        self.config: W5Config = config
        self.impact_analyzer = ProposalImpactAnalyzer(config)
        self.change_executor = ChangeExecutor(config)
        self.version_manager = VersionManager(config)
    
    def pre_process_validation(self, input_data: W5Input) -> ValidationResult:
        """預處理驗證"""
        result = ValidationResult(is_valid=True)
        
        # 檢查衝突
        conflict_check = self.impact_analyzer.check_conflicts(input_data.proposals)
        if conflict_check.has_conflicts:
            if self.config.conflict_resolution_strategy == ConflictResolutionStrategy.FAIL:
                result.add_error(f"Conflicts detected: {conflict_check.data_integrity_risks}")
            else:
                result.add_warning(f"Conflicts detected but will be resolved: {conflict_check.recommended_resolution}")
        
        return result
    
    def generate_governance_events(self, 
                                 proposals: List[ProposalReviewInput], 
                                 changes: List[TaxonomyChangeResult]) -> List[MaintenanceGovernanceEvent]:
        """生成治理事件"""
        events = []
        
        # 主要維護事件
        main_event = MaintenanceGovernanceEvent(
            event_id=f"maintenance_{self.config.workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            event_type=MaintenanceOperationType.TAXONOMY_UPDATE,
            affected_proposals=[p.proposal_id for p in proposals],
            changes=changes,
            version_point_before="",  # 會在執行時填入
            version_point_after=""    # 會在執行時填入
        )
        events.append(main_event)
        
        return events
    
    def check_idempotency(self, batch_metadata: BatchMetadata) -> bool:
        """檢查冪等性"""
        raise NotImplementedError
    
    def process(self, input_data: W5Input) -> W5Output:
        """執行完整的 W5 workflow"""
        raise NotImplementedError


# 輔助函數
def create_w5_config(
    workflow_id: str,
    domain: str,
    **kwargs
) -> W5Config:
    """建立 W5 配置"""
    return W5Config(
        workflow_id=workflow_id,
        domain=domain,
        **kwargs
    )


def validate_w5_preconditions(input_data: W5Input, config: W5Config) -> ValidationResult:
    """驗證 W5 前置條件"""
    result = ValidationResult(is_valid=True)
    
    try:
        # 檢查批次大小
        if len(input_data.proposals) > config.max_proposals_per_batch:
            result.add_error(f"Too many proposals: {len(input_data.proposals)} > {config.max_proposals_per_batch}")
        
        # 檢查領域一致性
        if input_data.batch_metadata.domain != config.domain:
            result.add_error(f"Domain mismatch: {input_data.batch_metadata.domain} != {config.domain}")
        
    except Exception as e:
        result.add_error(f"Validation failed: {str(e)}")
    
    return result


def create_proposal_review(
    proposal_id: UUID,
    action: ProposalAction,
    reviewer_id: str,
    comment: Optional[str] = None
) -> ProposalReviewInput:
    """建立提案審核"""
    return ProposalReviewInput(
        proposal_id=proposal_id,
        action=action,
        reviewer_id=reviewer_id,
        review_comment=comment,
        review_timestamp=datetime.now()
    )


def calculate_backfill_plan(changes: List[TaxonomyChangeResult]) -> Dict[str, List[str]]:
    """計算回填計畫"""
    plan = {}
    for change in changes:
        if change.success and change.affected_nodes:
            change_key = f"{change.change_type.value}_{len(change.affected_nodes)}_nodes"
            plan[change_key] = change.affected_nodes
    return plan