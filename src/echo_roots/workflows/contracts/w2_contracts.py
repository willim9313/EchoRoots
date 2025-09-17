from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
from uuid import UUID

from .common import (
    WorkflowConfig, WorkflowResult, ValidationResult, BatchMetadata,
    BaseProcessor, BaseValidator, DataQualityMetrics, SnapshotInfo,
    WorkflowStatus, GovernanceEvent, GovernanceEventType
)
from ...models import (
    ItemRaw, ItemNorm, OutlierRecord, GovernanceEvent as ModelGovernanceEvent,
    NormStatus, Domain
)


class BackfillStrategy(Enum):
    """回填策略"""
    INCREMENTAL = "incremental"
    BATCH_BY_BATCH = "batch_by_batch" 
    FULL_DOMAIN = "full_domain"


class NormalizationConflictResolution(Enum):
    """正規化衝突解決策略"""
    USE_LATEST_TN = "use_latest_tn"
    PRESERVE_ORIGINAL = "preserve_original"
    MANUAL_REVIEW = "manual_review"
    AUTO_MERGE = "auto_merge"


@dataclass
class TNVersionDiff:
    """T/N 版本差異 - 對應 w2_backfill.md 的版本差異集要求"""
    old_version: str
    new_version: str
    added_categories: List[UUID] = field(default_factory=list)
    removed_categories: List[UUID] = field(default_factory=list)
    modified_categories: List[UUID] = field(default_factory=list)
    added_value_mappings: Dict[UUID, UUID] = field(default_factory=dict)  # old_value_id -> new_value_id
    removed_value_mappings: Dict[UUID, UUID] = field(default_factory=dict)
    modified_value_mappings: Dict[UUID, UUID] = field(default_factory=dict)
    
    def get_change_summary(self) -> Dict[str, int]:
        """獲取變更摘要統計"""
        return {
            "added_categories": len(self.added_categories),
            "removed_categories": len(self.removed_categories),
            "modified_categories": len(self.modified_categories),
            "added_mappings": len(self.added_value_mappings),
            "removed_mappings": len(self.removed_value_mappings),
            "modified_mappings": len(self.modified_value_mappings)
        }
    
    def has_breaking_changes(self) -> bool:
        """檢查是否有破壞性變更"""
        return len(self.removed_categories) > 0 or len(self.removed_value_mappings) > 0


@dataclass 
class BackfillBatch:
    """回填批次 - 對應 w2_backfill.md 的選批要求"""
    batch_id: str
    domain: Domain
    raw_data_path: str  # data/{domain}/d_raw/dt=.../batch_id=...
    raw_items: List[ItemRaw] = field(default_factory=list)
    
    # 處理狀態
    total_records: int = 0
    processed_records: int = 0
    success_records: int = 0
    failed_records: int = 0
    outlier_records: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def is_complete(self) -> bool:
        """檢查批次是否完成"""
        return self.processed_records >= self.total_records
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.processed_records == 0:
            return 0.0
        return self.success_records / self.processed_records


@dataclass
class AttributeParsingResult:
    """屬性解析結果 - 對應 w2_backfill.md 的屬性解析步驟"""
    item_id: UUID
    raw_attributes: Dict[str, Any]  # 從 ItemRaw.specs 等欄位提取
    parsed_attributes: Dict[str, Any] = field(default_factory=dict)  # 符合 ItemNorm.attrs 格式
    keyword_matches: Dict[str, UUID] = field(default_factory=dict)  # keyword -> value_id
    parsing_errors: List[str] = field(default_factory=list)
    parsing_warnings: List[str] = field(default_factory=list)
    confidence_score: float = 1.0
    
    def to_item_norm_attrs(self) -> Dict[str, Any]:
        """轉換為 ItemNorm.attrs 格式"""
        return self.parsed_attributes


@dataclass
class CategoryClassificationResult:
    """分類掛載結果 - 對應 w2_backfill.md 的分類掛載步驟"""
    item_id: UUID
    raw_category_path: List[str]  # 來自 ItemRaw.raw_category_path
    classified_category_id: Optional[UUID] = None
    classification_method: str = "unknown"  # heuristics, mapping, manual
    confidence_score: float = 0.0
    alternative_categories: List[UUID] = field(default_factory=list)
    classification_metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_successful(self) -> bool:
        """分類是否成功"""
        return self.classified_category_id is not None and self.confidence_score > 0.5


@dataclass
class BackfillItemResult:
    """單一項目回填結果"""
    item_raw: ItemRaw
    attribute_parsing: Optional[AttributeParsingResult] = None
    category_classification: Optional[CategoryClassificationResult] = None
    
    # 最終結果 - 使用 models 中的類型
    item_norm: Optional[ItemNorm] = None
    outlier_record: Optional[OutlierRecord] = None
    
    processing_status: NormStatus = NormStatus.OK
    error_messages: List[str] = field(default_factory=list)
    processing_time: Optional[datetime] = None
    
    def mark_as_success(self, normalized_item: ItemNorm) -> None:
        """標記為成功處理"""
        self.item_norm = normalized_item
        self.processing_status = NormStatus.OK
        self.processing_time = datetime.now()
    
    def mark_as_failed(self, error: str) -> None:
        """標記為處理失敗"""
        self.error_messages.append(error)
        self.processing_status = NormStatus.FAILED
        self.processing_time = datetime.now()
    
    def mark_as_outlier(self, outlier: OutlierRecord) -> None:
        """標記為異常值"""
        self.outlier_record = outlier
        self.processing_status = NormStatus.FAILED
        self.processing_time = datetime.now()


@dataclass
class BackfillConfig(WorkflowConfig):
    """回填工作流配置"""
    target_domain: Domain
    old_tn_version: str
    new_tn_version: str
    tn_version_diff: TNVersionDiff
    
    # W2 特定配置
    backfill_strategy: BackfillStrategy = BackfillStrategy.INCREMENTAL
    batch_size: int = 1000  # 對應 w2_backfill.md 的 Config Knobs
    stop_on_diff_gt: float = 0.05  # 對應 w2_backfill.md 的誤差上限
    conflict_resolution: NormalizationConflictResolution = NormalizationConflictResolution.USE_LATEST_TN
    
    # 異常檢測設定
    enable_outlier_detection: bool = True
    outlier_threshold: float = 0.3
    max_retry_per_batch: int = 3
    
    # 快照與治理
    enable_pre_commit_snapshot: bool = True  # 對應 w2_backfill.md 的 Rollback
    enable_governance_events: bool = True
    
    def validate(self) -> ValidationResult:
        """驗證回填配置"""
        result = super().validate()
        
        if not self.target_domain:
            result.add_error("target_domain is required")
        
        if not self.old_tn_version:
            result.add_error("old_tn_version is required")
        
        if not self.new_tn_version:
            result.add_error("new_tn_version is required")
        
        if self.old_tn_version == self.new_tn_version:
            result.add_warning("old_tn_version and new_tn_version are identical")
        
        if self.batch_size <= 0:
            result.add_error("batch_size must be positive")
        
        if not (0.0 <= self.stop_on_diff_gt <= 1.0):
            result.add_error("stop_on_diff_gt must be between 0.0 and 1.0")
        
        return result


@dataclass
class BackfillResult(WorkflowResult):
    """回填結果 - 對應 w2_backfill.md 的 Outputs 和 Side Effects"""
    
    # 主要輸出
    updated_d_norm_items: List[ItemNorm] = field(default_factory=list)  # 新版 d_norm
    outlier_records: List[OutlierRecord] = field(default_factory=list)  # 寫入 d_outlier 的項目
    
    # 處理統計
    processed_batches: List[BackfillBatch] = field(default_factory=list)
    data_quality_metrics: Optional[DataQualityMetrics] = None
    
    # Side Effects
    pre_commit_snapshot: Optional[SnapshotInfo] = None  # SNAPSHOT_CREATE (D)
    post_commit_snapshot: Optional[SnapshotInfo] = None
    governance_events: List[ModelGovernanceEvent] = field(default_factory=list)  # BACKFILL_COMMIT
    
    def get_overall_success_rate(self) -> float:
        """獲取整體成功率"""
        if not self.processed_batches:
            return 0.0
        
        total_records = sum(batch.total_records for batch in self.processed_batches)
        success_records = sum(batch.success_records for batch in self.processed_batches)
        
        if total_records == 0:
            return 0.0
        
        return success_records / total_records


# 處理器介面
class TNDiffAnalyzer(BaseValidator):
    """T/N 差異分析器 - 對應版本差異集分析"""
    
    def analyze_tn_changes(self, old_version: str, new_version: str) -> TNVersionDiff:
        """分析 T/N 版本變更"""
        raise NotImplementedError
    
    def validate_diff_safety(self, diff: TNVersionDiff) -> ValidationResult:
        """驗證差異變更的安全性 - 對應 Pre-checks"""
        raise NotImplementedError


class AttributeParser(BaseProcessor):
    """屬性解析器 - 對應 w2_backfill.md 步驟2"""
    
    def parse_attributes(self, item_raw: ItemRaw, 
                        value_mapping: Dict[str, Any]) -> AttributeParsingResult:
        """解析原始資料屬性，優先嘗試 N 值匹配（alias/labels）"""
        raise NotImplementedError


class CategoryClassifier(BaseProcessor):  
    """分類掛載器 - 對應 w2_backfill.md 步驟3"""
    
    def classify_category(self, item_raw: ItemRaw,
                         existing_mappings: Dict[str, UUID]) -> CategoryClassificationResult:
        """依 raw_category_path 使用 heuristics 或既有映射"""
        raise NotImplementedError


class BackfillProcessor(BaseProcessor):
    """回填處理器主類 - 實現完整的 W2 workflow"""
    
    def __init__(self, config: BackfillConfig):
        super().__init__(config)
        self.config: BackfillConfig = config
    
    def select_batches(self) -> List[str]:
        """選批 - 以 data/{domain}/d_raw/dt=.../batch_id=... 為單位"""
        raise NotImplementedError
    
    def process_batch(self, batch_path: str) -> BackfillBatch:
        """處理單一批次"""
        raise NotImplementedError
    
    def process_item(self, item_raw: ItemRaw) -> BackfillItemResult:
        """處理單一項目 - 整合屬性解析與分類掛載"""
        raise NotImplementedError
    
    def create_pre_commit_snapshot(self) -> SnapshotInfo:
        """建立預提交快照 - 對應 Rollback 需求"""
        raise NotImplementedError
    
    def generate_governance_events(self, result: BackfillResult) -> List[ModelGovernanceEvent]:
        """生成治理事件 - BACKFILL_COMMIT, SNAPSHOT_CREATE"""
        raise NotImplementedError
    
    def check_idempotency(self, batch_metadata: BatchMetadata) -> bool:
        """檢查冪等性 - 同版本重跑結果一致"""
        raise NotImplementedError


# 輔助函數
def create_backfill_config(
    workflow_id: str,
    target_domain: Domain,
    old_tn_version: str, 
    new_tn_version: str,
    tn_version_diff: TNVersionDiff,
    **kwargs
) -> BackfillConfig:
    """建立回填配置"""
    return BackfillConfig(
        workflow_id=workflow_id,
        domain=target_domain.value,
        target_domain=target_domain,
        old_tn_version=old_tn_version,
        new_tn_version=new_tn_version,
        tn_version_diff=tn_version_diff,
        **kwargs
    )


def validate_backfill_preconditions(config: BackfillConfig) -> ValidationResult:
    """驗證回填前置條件 - 對應 Pre-checks 要求"""
    result = ValidationResult(is_valid=True)
    
    # 檢查 T/N 版本差異是否合理
    if config.tn_version_diff.has_breaking_changes():
        result.add_warning("Detected breaking changes in T/N version diff")
    
    # 檢查配置有效性
    config_validation = config.validate()
    result.merge(config_validation)
    
    return result