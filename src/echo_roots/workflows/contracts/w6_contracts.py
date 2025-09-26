from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum

from .common import (
    WorkflowConfig, WorkflowResult, ValidationResult, ProcessingResult,
    BatchMetadata, GovernanceEvent, SnapshotInfo, 
    ClusterStatus, SemanticEventType, ClusteringMetrics, SimilarityMetrics,
    BaseValidator, BaseProcessor, SupportMetrics
)

# 重用 models 中的定義
from ...models.enums import Language, DataSource
from ...models.base import BaseEntity, TimestampMixin
from ...models.data import Observation, Cluster

# ============================================================================
# W6 專用 Enums (非重複的)
# ============================================================================

class ClusteringAlgorithm(str, Enum):
    """聚類演算法類型"""
    NEAREST_NEIGHBOR = "nearest_neighbor"
    DENSITY_BASED = "density_based"
    HIERARCHICAL = "hierarchical"
    COMMUNITY_DETECTION = "community_detection"


class ProposalStatus(str, Enum):
    """提案狀態"""
    DRAFT = "draft"
    PENDING = "pending" 
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"


class GovernanceDecisionType(str, Enum):
    """治理決策類型"""
    ACCEPT = "accept"
    REJECT = "reject"
    MERGE = "merge"
    SPLIT = "split"
    MODIFY = "modify"


# ============================================================================
# 內部計算層 - dataclass (繼承 models 基礎類)
# ============================================================================

@dataclass
class SObservation:
    """STAGING 觀測 - 內部計算物件，基於 models.Observation 擴展"""
    observation_id: str
    domain: str
    category_id: str
    attr_hint: str
    raw_value: str
    normalized_value: Optional[str] = None
    confidence_score: float = 0.0
    cluster_id: Optional[str] = None
    state: ClusterStatus = ClusterStatus.COLD
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """驗證必要欄位"""
        if not self.observation_id:
            raise ValueError("observation_id is required")
        if not self.domain or not self.category_id:
            raise ValueError("domain and category_id are required")
    
    def to_base_observation(self) -> Observation:
        """轉換為基礎 Observation 物件"""
        return Observation(
            id=self.observation_id,
            domain=self.domain,
            category_id=self.category_id,
            raw_value=self.raw_value,
            normalized_value=self.normalized_value,
            confidence_score=self.confidence_score,
            metadata=self.metadata
        )


@dataclass
class SCluster:
    """語意聚類 - 內部計算物件，基於 models.Cluster 擴展"""
    cluster_id: str
    rep_term: str  # 代表詞
    aliases: Set[str] = field(default_factory=set)  # 同義詞集合
    support: int = 0  # 支持數量
    observations: List[SObservation] = field(default_factory=list)
    confidence: float = 0.0
    purity_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_observation(self, obs: SObservation) -> None:
        """添加觀測到聚類"""
        self.observations.append(obs)
        self.support += 1
        obs.cluster_id = self.cluster_id
        obs.state = ClusterStatus.HOT
        
        # 更新別名集合
        if obs.normalized_value:
            self.aliases.add(obs.normalized_value)
    
    def get_support_metrics(self) -> SupportMetrics:
        """計算支援度指標"""        
        total_obs = len(self.observations)
        coverage = len(self.aliases) / max(total_obs, 1)
        
        return SupportMetrics(
            support_count=self.support,
            confidence_score=self.confidence,
            coverage_ratio=coverage
        )
    
    @property
    def is_singleton(self) -> bool:
        """檢查是否為單例聚類"""
        return self.support == 1
    
    def to_base_cluster(self) -> Cluster:
        """轉換為基礎 Cluster 物件"""
        return Cluster(
            id=self.cluster_id,
            name=self.rep_term,
            center_point=self.rep_term,
            size=self.support,
            confidence=self.confidence,
            metadata={
                **self.metadata,
                "aliases": list(self.aliases),
                "purity_score": self.purity_score
            }
        )


@dataclass
class ClusteringPlan:
    """聚類計畫 - 內部計算物件"""
    domain: str
    category_id: str
    algorithm: ClusteringAlgorithm
    parameters: Dict[str, Any] = field(default_factory=dict)
    min_support: int = 2
    purity_threshold: float = 0.8
    similarity_threshold: float = 0.7
    max_clusters: Optional[int] = None
    
    def validate_parameters(self) -> ValidationResult:
        """驗證計畫參數"""
        result = ValidationResult(is_valid=True)
        
        if self.min_support < 1:
            result.add_error("min_support must be >= 1")
        
        if not (0.0 < self.purity_threshold <= 1.0):
            result.add_error("purity_threshold must be between 0.0 and 1.0")
            
        if not (0.0 < self.similarity_threshold <= 1.0):
            result.add_error("similarity_threshold must be between 0.0 and 1.0")
            
        return result


@dataclass
class ProposalCandidate:
    """提案候選 - 內部計算物件"""
    propose_attr_id: str
    propose_value: str
    source_cluster: SCluster
    confidence_score: float = 0.0
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_proposal_strength(self) -> float:
        """計算提案強度"""
        support_weight = min(self.source_cluster.support / 10.0, 1.0)  # 支持度權重
        purity_weight = self.source_cluster.purity_score  # 純度權重  
        confidence_weight = self.confidence_score  # 信心權重
        
        return (support_weight * 0.4 + purity_weight * 0.3 + confidence_weight * 0.3)


@dataclass
class SemanticAnalysisResult:
    """語意分析結果 - 內部計算結果"""
    domain: str
    category_id: str
    total_observations: int
    clusters: List[SCluster] = field(default_factory=list)
    singletons: List[SObservation] = field(default_factory=list)
    proposal_candidates: List[ProposalCandidate] = field(default_factory=list)
    clustering_metrics: Optional[ClusteringMetrics] = None
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    
    def get_hot_clusters(self) -> List[SCluster]:
        """取得HOT狀態聚類"""
        return [cluster for cluster in self.clusters 
                if not cluster.is_singleton and cluster.support >= 2]
    
    def calculate_metrics(self) -> ClusteringMetrics:
        """計算聚類指標"""
        if not self.clustering_metrics:
            non_singleton_clusters = [c for c in self.clusters if not c.is_singleton]
            cluster_sizes = [c.support for c in self.clusters]
            
            self.clustering_metrics = ClusteringMetrics(
                total_observations=self.total_observations,
                total_clusters=len(self.clusters),
                avg_cluster_size=sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0.0,
                max_cluster_size=max(cluster_sizes) if cluster_sizes else 0,
                min_cluster_size=min(cluster_sizes) if cluster_sizes else 0,
                singleton_count=len(self.singletons),
                purity_score=sum(c.purity_score for c in non_singleton_clusters) / 
                            len(non_singleton_clusters) if non_singleton_clusters else 0.0
            )
        
        return self.clustering_metrics


# ============================================================================
# 邊界層 - Pydantic BaseModel (API, DB I/O) - 繼承 models 基礎類
# ============================================================================

class W6Config(WorkflowConfig):
    """W6 語意治理配置 - API 邊界層"""
    clustering_algorithm: ClusteringAlgorithm = ClusteringAlgorithm.NEAREST_NEIGHBOR
    min_support: int = Field(default=2, gt=0)
    purity_threshold: float = Field(default=0.8, gt=0.0, le=1.0)
    similarity_threshold: float = Field(default=0.7, gt=0.0, le=1.0)
    max_clusters_per_category: Optional[int] = Field(default=None, gt=0)
    enable_auto_proposal: bool = True
    proposal_confidence_threshold: float = Field(default=0.6, gt=0.0, le=1.0)
    
    @field_validator('purity_threshold', 'similarity_threshold', 'proposal_confidence_threshold')
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError("Threshold must be between 0.0 and 1.0")
        return v


class StagingQuery(BaseModel):
    """STAGING 查詢請求 - API 邊界層"""
    domain: str
    category_id: Optional[str] = None
    attr_hint: Optional[str] = None
    limit: Optional[int] = Field(default=None, gt=0)
    offset: Optional[int] = Field(default=0, ge=0)
    state_filter: Optional[ClusterStatus] = None
    
    def build_where_clause(self) -> Dict[str, Any]:
        """建構WHERE條件"""
        conditions = {"domain": self.domain}
        
        if self.category_id:
            conditions["category_id"] = self.category_id
        if self.attr_hint:
            conditions["attr_hint"] = self.attr_hint
        if self.state_filter:
            conditions["state"] = self.state_filter.value
            
        return conditions


class SemanticProposal(BaseEntity):
    """語意提案 - DB I/O 邊界層，繼承 BaseEntity"""
    proposal_id: str
    domain: str
    category_id: str
    propose_attr_id: str
    propose_value: str
    source_cluster_id: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    support_count: int = Field(gt=0)
    evidence: List[str] = Field(default_factory=list)
    status: ProposalStatus = ProposalStatus.DRAFT
    reviewed_at: Optional[datetime] = None
    reviewer_id: Optional[str] = None
    
    @field_validator('proposal_id')
    @classmethod
    def validate_proposal_id(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError("proposal_id cannot be empty")
        return v.strip()


class ClusterUpdateRequest(BaseModel):
    """聚類更新請求 - API 邊界層"""
    cluster_id: str
    observations: List[str]  # observation IDs
    action: str  # "add", "remove", "merge", "split"
    target_cluster_id: Optional[str] = None  # for merge operations
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('action')
    @classmethod
    def validate_action(cls, v: str) -> str:
        valid_actions = {"add", "remove", "merge", "split"}
        if v not in valid_actions:
            raise ValueError(f"Action must be one of: {valid_actions}")
        return v


class GovernanceDecision(BaseEntity):
    """治理決策 - API 邊界層，繼承 BaseEntity"""
    decision_id: str
    proposal_id: str
    decision_type: GovernanceDecisionType
    reviewer_id: str
    reason: str
    implemented_value: Optional[str] = None
    effective_date: datetime = Field(default_factory=datetime.now)
    rollback_snapshot_id: Optional[str] = None


class W6Input(BaseModel):
    """W6 工作流輸入 - API 邊界層"""
    config: W6Config
    batch_metadata: BatchMetadata
    staging_query: StagingQuery
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class W6Output(BaseModel):
    """W6 工作流輸出 - API 邊界層"""
    workflow_result: ProcessingResult
    hot_clusters_count: int
    proposals_count: int
    clustering_metrics: Dict[str, Any]  # ClusteringMetrics serialized
    snapshot_info: Optional[SnapshotInfo] = None
    governance_events: List[GovernanceEvent] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SemanticSyncRequest(BaseModel):
    """語意同步請求 - API 邊界層"""
    domain: str
    approved_proposals: List[str]  # proposal IDs
    target_norm_table: str = "d_norm"
    create_backup: bool = True
    batch_size: int = Field(default=1000, gt=0)


# ============================================================================
# 驗證器和處理器 (保持不變)
# ============================================================================

class W6Validator(BaseValidator):
    """W6 語意治理驗證器"""
    
    def __init__(self, config: W6Config):
        self.config = config
    
    def validate(self, data: W6Input) -> ValidationResult:
        """驗證W6輸入"""
        result = ValidationResult(is_valid=True)
        
        # 驗證配置
        if self.config.min_support < 1:
            result.add_error("min_support must be >= 1")
        
        # 驗證查詢參數
        if not data.staging_query.domain:
            result.add_error("domain is required in staging_query")
        
        # 驗證批次元數據
        if not data.batch_metadata.event_id:
            result.add_error("event_id is required in batch_metadata")
            
        return result
    
    def validate_clustering_plan(self, plan: ClusteringPlan) -> ValidationResult:
        """驗證聚類計畫"""
        return plan.validate_parameters()
    
    def validate_proposals(self, proposals: List[ProposalCandidate]) -> ValidationResult:
        """驗證提案列表"""
        result = ValidationResult(is_valid=True)
        
        for i, proposal in enumerate(proposals):
            if proposal.confidence_score < self.config.proposal_confidence_threshold:
                result.add_warning(f"Proposal {i} has low confidence: {proposal.confidence_score}")
        
        return result


class W6SemanticProcessor(BaseProcessor):
    """W6 語意治理處理器"""
    
    def __init__(self, config: W6Config):
        super().__init__(config)
        self.w6_config = config
        self.validator = W6Validator(config)
    
    def process(self, input_data: W6Input) -> WorkflowResult:
        """執行語意治理處理"""
        from .common import create_workflow_result, WorkflowStatus
        
        # 預檢驗證
        validation_result = self.validator.validate(input_data)
        if not validation_result.is_valid:
            result = create_workflow_result(
                workflow_id=self.config.workflow_id,
                status=WorkflowStatus.FAILED,
                message="Validation failed"
            )
            result.errors.extend(validation_result.errors)
            return result
        
        try:
            # 1. 載入STAGING觀測
            observations = self._load_staging_observations(input_data.staging_query)
            
            # 2. 執行聚類分析  
            analysis_result = self._perform_clustering_analysis(
                observations, input_data.config
            )
            
            # 3. 產生提案
            proposals = self._generate_proposals(analysis_result)
            
            # 4. 更新觀測狀態
            self._update_observation_states(analysis_result)
            
            # 5. 建立輸出
            result = create_workflow_result(
                workflow_id=self.config.workflow_id,
                status=WorkflowStatus.SUCCESS
            )
            
            result.data = {
                "hot_clusters_count": len(analysis_result.get_hot_clusters()),
                "proposals_count": len(proposals),
                "total_observations": analysis_result.total_observations,
                "clustering_metrics": analysis_result.calculate_metrics().__dict__
            }
            
            return result
            
        except Exception as e:
            result = create_workflow_result(
                workflow_id=self.config.workflow_id,
                status=WorkflowStatus.FAILED,
                message=str(e)
            )
            result.add_error(f"Processing failed: {str(e)}")
            return result
    
    def _load_staging_observations(self, query: StagingQuery) -> List[SObservation]:
        """載入STAGING觀測 (stub implementation)"""
        # TODO: 實際的資料庫查詢實作
        return []
    
    def _perform_clustering_analysis(self, 
                                   observations: List[SObservation],
                                   config: W6Config) -> SemanticAnalysisResult:
        """執行聚類分析 (stub implementation)"""
        # TODO: 實際的聚類演算法實作
        return SemanticAnalysisResult(
            domain=config.domain,
            category_id="test",
            total_observations=len(observations)
        )
    
    def _generate_proposals(self, 
                          analysis_result: SemanticAnalysisResult) -> List[ProposalCandidate]:
        """產生提案 (stub implementation)"""
        # TODO: 實際的提案產生邏輯
        return []
    
    def _update_observation_states(self, analysis_result: SemanticAnalysisResult) -> None:
        """更新觀測狀態 (stub implementation)"""
        # TODO: 實際的狀態更新邏輯
        pass


# ============================================================================
# 輔助函數
# ============================================================================

def create_clustering_plan(domain: str, 
                          category_id: str,
                          algorithm: ClusteringAlgorithm = ClusteringAlgorithm.NEAREST_NEIGHBOR,
                          **kwargs) -> ClusteringPlan:
    """建立聚類計畫"""
    return ClusteringPlan(
        domain=domain,
        category_id=category_id,
        algorithm=algorithm,
        **kwargs
    )


def create_semantic_proposal(cluster: SCluster,
                           propose_attr_id: str,
                           propose_value: str) -> SemanticProposal:
    """從聚類建立語意提案"""
    return SemanticProposal(
        id=f"{cluster.cluster_id}_{propose_attr_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        proposal_id=f"{cluster.cluster_id}_{propose_attr_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        domain=cluster.observations[0].domain if cluster.observations else "",
        category_id=cluster.observations[0].category_id if cluster.observations else "",
        propose_attr_id=propose_attr_id,
        propose_value=propose_value,
        source_cluster_id=cluster.cluster_id,
        confidence_score=cluster.confidence,
        support_count=cluster.support,
        evidence=[obs.raw_value for obs in cluster.observations[:5]]  # 取前5個作為證據
    )


def calculate_cluster_similarity(cluster1: SCluster, cluster2: SCluster) -> SimilarityMetrics:
    """計算兩個聚類間的相似度"""
    # 簡化的Jaccard相似度計算
    set1 = cluster1.aliases
    set2 = cluster2.aliases
    
    if not set1 and not set2:
        similarity = 1.0
    elif not set1 or not set2:
        similarity = 0.0
    else:
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        similarity = intersection / union if union > 0 else 0.0
    
    return SimilarityMetrics(
        similarity_score=similarity,
        method="jaccard",
        confidence=min(cluster1.confidence, cluster2.confidence)
    )