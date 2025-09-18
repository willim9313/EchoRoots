"""
Workflow W4 - Proposal Creation Contracts
上浮提案工作流程合約定義
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
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
    SupportMetrics,
    ClusterStatus,
    GovernanceEvent
)
from echo_roots.models.data import GovernanceEvent as ModelGovernanceEvent


# W4 專屬枚舉
class ProposalType(str, Enum):
    """提案類型"""
    ATTRIBUTE = "ATTRIBUTE"
    VALUE = "VALUE"
    CLASSIFICATION = "CLASSIFICATION"


class ProposalStatus(str, Enum):
    """提案狀態"""
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    WITHDRAWN = "WITHDRAWN"


@dataclass
class ClusterCandidate:
    """聚類候選項 - 來自 S-HOT 的候選"""
    cluster_id: str
    status: ClusterStatus
    support_metrics: SupportMetrics
    aliases: List[str] = field(default_factory=list)
    representative_value: str = ""
    domain: str = ""
    
    def validate_hot_status(self) -> ValidationResult:
        """驗證是否為 HOT 狀態"""
        result = ValidationResult(is_valid=True)
        if self.status != ClusterStatus.HOT:
            result.add_error("Only HOT clusters can be candidates")
        return result


@dataclass
class ProposalContent:
    """提案內容基類"""
    cluster_id: str
    support_count: int
    aliases: List[str] = field(default_factory=list)
    impacted_set: List[str] = field(default_factory=list)  # 影響的實體ID列表
    confidence_score: float = 0.0
    
    def validate_confidence(self) -> ValidationResult:
        """驗證信心分數"""
        result = ValidationResult(is_valid=True)
        if not (0.0 <= self.confidence_score <= 1.0):
            result.add_error("confidence_score must be between 0.0 and 1.0")
        return result


@dataclass
class AttributeProposalContent(ProposalContent):
    """屬性提案內容"""
    propose_attr_id: str = ""
    attr_name: str = ""
    attr_type: str = ""


@dataclass
class ValueProposalContent(ProposalContent):
    """值提案內容"""
    propose_attr_id: str = ""
    propose_value: str = ""
    canonical_form: str = ""


@dataclass
class ClassificationProposalContent(ProposalContent):
    """分類提案內容"""
    propose_class_id: str = ""
    class_name: str = ""
    parent_class_id: Optional[str] = None


@dataclass
class Proposal:
    """提案模型"""
    proposal_id: str
    proposal_type: ProposalType
    content: Dict[str, Any]
    domain: str
    status: ProposalStatus = ProposalStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "proposal_id": self.proposal_id,
            "proposal_type": self.proposal_type.value,
            "status": self.status.value,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "domain": self.domain
        }


class W4Config(BaseModel):
    """W4 配置合約 - 邊界層使用 Pydantic"""
    workflow_id: str
    domain: str
    target_domain: str
    min_support: int = Field(default=3, ge=1)
    min_uniqueness: float = Field(default=0.7, ge=0.0, le=1.0)
    max_proposals_per_run: int = Field(default=100, ge=1)
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    
    @validator('target_domain')
    def validate_target_domain(cls, v):
        if not v:
            raise ValueError("target_domain is required")
        return v
    
    class Config:
        extra = "forbid"  # 禁止額外字段


class W4Input(BaseModel):
    """W4 輸入合約 - 邊界層使用 Pydantic"""
    s_hot_candidates: List[ClusterCandidate]
    domain: str
    batch_metadata: BatchMetadata
    
    @validator('s_hot_candidates')
    def validate_candidates_not_empty(cls, v):
        if not v:
            raise ValueError("s_hot_candidates cannot be empty")
        return v
    
    @validator('domain')
    def validate_domain_not_empty(cls, v):
        if not v:
            raise ValueError("domain is required")
        return v
    
    @validator('s_hot_candidates', each_item=True)
    def validate_each_candidate(cls, v):
        validation_result = v.validate_hot_status()
        if not validation_result.is_valid:
            raise ValueError(f"Invalid candidate: {validation_result.errors}")
        return v
    
    class Config:
        arbitrary_types_allowed = True  # 允許自定義類型如 ClusterCandidate


class W4Output(BaseModel):
    """W4 輸出合約 - 邊界層使用 Pydantic"""
    s_proposals: List[Proposal] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    governance_events: List[ModelGovernanceEvent] = Field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    
    def get_proposal_summary(self) -> Dict[str, Any]:
        """獲取提案摘要統計"""
        if not self.s_proposals:
            return {
                "total_proposals": 0,
                "by_type": {},
                "by_status": {},
                "total_candidates_processed": 0,
                "acceptance_rate": 0.0
            }
        
        by_type = {}
        by_status = {}
        
        for proposal in self.s_proposals:
            # 統計類型
            prop_type = proposal.proposal_type.value
            by_type[prop_type] = by_type.get(prop_type, 0) + 1
            
            # 統計狀態
            prop_status = proposal.status.value
            by_status[prop_status] = by_status.get(prop_status, 0) + 1
        
        return {
            "total_proposals": len(self.s_proposals),
            "by_type": by_type,
            "by_status": by_status,
            "total_candidates_processed": len(set(p.content.get("cluster_id", "") for p in self.s_proposals)),
            "acceptance_rate": by_status.get("APPROVED", 0) / len(self.s_proposals) if self.s_proposals else 0.0
        }
    
    @validator('s_proposals')
    def validate_proposals_not_empty(cls, v):
        if not v:
            raise ValueError("At least one proposal must be generated")
        return v
    
    @validator('s_proposals', each_item=True)
    def validate_each_proposal(cls, v):
        if not v.proposal_id:
            raise ValueError("Proposal missing proposal_id")
        if not v.content:
            raise ValueError(f"Proposal {v.proposal_id} missing content")
        return v
    
    class Config:
        arbitrary_types_allowed = True


# 處理器介面
class ClusterAnalyzer(BaseValidator):
    """聚類分析器 - 分析 HOT 聚類候選"""
    
    def analyze_cluster_quality(self, candidate: ClusterCandidate) -> ValidationResult:
        """分析聚類品質"""
        raise NotImplementedError
    
    def check_support_threshold(self, candidate: ClusterCandidate, min_support: int) -> bool:
        """檢查支援度門檻"""
        return candidate.support_metrics.support_count >= min_support
    
    def check_uniqueness_threshold(self, candidate: ClusterCandidate, min_uniqueness: float) -> bool:
        """檢查純度/唯一性門檻"""
        return candidate.support_metrics.confidence_score >= min_uniqueness


class ProposalGenerator(BaseProcessor):
    """提案生成器 - 將聚類轉換為提案"""
    
    def __init__(self, config: W4Config):
        super().__init__(config)
        self.config: W4Config = config
    
    def generate_proposals_from_candidates(self, candidates: List[ClusterCandidate]) -> List[Proposal]:
        """從候選聚類生成提案"""
        raise NotImplementedError
    
    def determine_proposal_type(self, candidate: ClusterCandidate) -> ProposalType:
        """確定提案類型"""
        raise NotImplementedError
    
    def create_proposal_content(self, candidate: ClusterCandidate, proposal_type: ProposalType) -> Dict[str, Any]:
        """創建提案內容"""
        raise NotImplementedError


class W4Processor(BaseProcessor):
    """W4 主處理器"""
    
    def __init__(self, config: W4Config):
        super().__init__(config)
        self.config: W4Config = config
        self.cluster_analyzer = ClusterAnalyzer(config)
        self.proposal_generator = ProposalGenerator(config)
    
    def filter_candidates(self, candidates: List[ClusterCandidate]) -> List[ClusterCandidate]:
        """篩選候選 - 檢查支援度/純度門檻"""
        filtered = []
        for candidate in candidates:
            if (self.cluster_analyzer.check_support_threshold(candidate, self.config.min_support) and
                self.cluster_analyzer.check_uniqueness_threshold(candidate, self.config.min_uniqueness)):
                filtered.append(candidate)
        return filtered
    
    def generate_governance_events(self, proposals: List[Proposal]) -> List[ModelGovernanceEvent]:
        """生成治理事件 - PROPOSAL_CREATE"""
        events = []
        for proposal in proposals:
            event = GovernanceEvent(
                event_id=f"proposal_create_{proposal.proposal_id}",
                event_type="PROPOSAL_CREATE",
                workflow_id="w4_propose",
                domain=proposal.domain,
                timestamp=datetime.now(),
                event_data={
                    "proposal_id": proposal.proposal_id,
                    "proposal_type": proposal.proposal_type.value,
                    "cluster_id": proposal.content.get("cluster_id"),
                    "support_count": proposal.content.get("support_count", 0)
                }
            )
            events.append(event)
        return events
    
    def check_idempotency(self, batch_metadata: BatchMetadata) -> bool:
        """檢查冪等性 - 同一候選群不得重複產生提案"""
        raise NotImplementedError
    
    def process(self, input_data: W4Input) -> W4Output:
        """執行完整的 W4 workflow"""
        raise NotImplementedError


# 輔助函數
def create_w4_config(
    workflow_id: str,
    target_domain: str,
    **kwargs
) -> W4Config:
    """建立 W4 配置"""
    return W4Config(
        workflow_id=workflow_id,
        domain=target_domain,
        target_domain=target_domain,
        **kwargs
    )


def validate_w4_preconditions(input_data: W4Input, config: W4Config) -> ValidationResult:
    """驗證 W4 前置條件"""
    result = ValidationResult(is_valid=True)
    
    try:
        # Pydantic 會在創建時自動驗證
        # 這裡可以添加額外的業務邏輯驗證
        pass
    except Exception as e:
        result.add_error(f"Validation failed: {str(e)}")
    
    return result


def create_proposal_id(cluster_id: str, domain: str) -> str:
    """生成提案 ID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"prop_{domain}_{cluster_id}_{timestamp}"