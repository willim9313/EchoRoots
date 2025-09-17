from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

from .common import (
    WorkflowConfig, WorkflowResult, ValidationResult, BatchMetadata,
    BaseProcessor, BaseValidator, DataQualityMetrics, SnapshotInfo,
    WorkflowStatus, DataSource, Language
)


class BackfillStrategy(Enum):
    """回填策略"""
    INCREMENTAL = "incremental"  # 增量回填
    BATCH_BY_BATCH = "batch_by_batch"  # 逐批處理
    FULL_DOMAIN = "full_domain"  # 全域回填


class NormalizationConflictResolution(Enum):
    """正規化衝突解決策略"""
    USE_LATEST_TN = "use_latest_tn"  # 使用最新 T/N 規則
    PRESERVE_ORIGINAL = "preserve_original"  # 保留原始值
    MANUAL_REVIEW = "manual_review"  # 手動審查
    AUTO_MERGE = "auto_merge"  # 自動合併


@dataclass
class TNVersionDiff:
    """T/N 版本差異"""
    old_version: str
    new_version: str
    added_categories: List[str] = field(default_factory=list)
    removed_categories: List[str] = field(default_factory=list)
    modified_categories: List[str] = field(default_factory=list)
    added_mappings: Dict[str, Any] = field(default_factory=dict)
    removed_mappings: Dict[str, Any] = field(default_factory=dict)
    modified_mappings: Dict[str, Any] = field(default_factory=dict)
    
    def get_change_summary(self) -> Dict[str, int]:
        """獲取變更摘要統計"""
        return {
            "added_categories": len(self.added_categories),
            "removed_categories": len(self.removed_categories),
            "modified_categories": len(self.modified_categories),
            "added_mappings": len(self.added_mappings),
            "removed_mappings": len(self.removed_mappings),
            "modified_mappings": len(self.modified_mappings)
        }
    
    def has_breaking_changes(self) -> bool:
        """檢查是否有破壞性變更"""
        return len(self.removed_categories) > 0 or len(self.removed_mappings) > 0


@dataclass
class BackfillBatch:
    """回填批次"""
    batch_id: str
    domain: str
    raw_data_path: str
    total_records: int
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
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "batch_id": self.batch_id,
            "domain": self.domain,
            "raw_data_path": self.raw_data_path,
            "total_records": self.total_records,
            "processed_records": self.processed_records,
            "success_records": self.success_records,
            "failed_records": self.failed_records,
            "outlier_records": self.outlier_records,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "success_rate": self.success_rate
        }


@dataclass
class BackfillProgress:
    """回填進度"""
    workflow_id: str
    domain: str
    total_batches: int
    processed_batches: int = 0
    successful_batches: int = 0
    failed_batches: int = 0
    current_batch: Optional[str] = None
    start_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    @property
    def completion_rate(self) -> float:
        """完成率"""
        if self.total_batches == 0:
            return 1.0
        return self.processed_batches / self.total_batches
    
    @property
    def is_complete(self) -> bool:
        """是否完成"""
        return self.processed_batches >= self.total_batches


@dataclass
class AttributeParsingResult:
    """屬性解析結果"""
    raw_attributes: Dict[str, Any]
    parsed_attributes: Dict[str, Any] = field(default_factory=dict)
    keyword_matches: Dict[str, str] = field(default_factory=dict)  # keyword -> normalized_value
    parsing_errors: List[str] = field(default_factory=list)
    parsing_warnings: List[str] = field(default_factory=list)
    confidence_score: float = 1.0
    
    def add_parsing_error(self, error: str) -> None:
        """添加解析錯誤"""
        self.parsing_errors.append(error)
        self.confidence_score *= 0.8  # 降低信心分數
    
    def add_parsing_warning(self, warning: str) -> None:
        """添加解析警告"""
        self.parsing_warnings.append(warning)
        self.confidence_score *= 0.95  # 輕微降低信心分數


@dataclass
class CategoryClassificationResult:
    """分類掛載結果"""
    raw_category_path: str
    classified_category: Optional[str] = None
    classification_method: str = "unknown"  # heuristics, mapping, manual
    confidence_score: float = 0.0
    alternative_categories: List[str] = field(default_factory=list)
    classification_metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_successful(self) -> bool:
        """分類是否成功"""
        return self.classified_category is not None and self.confidence_score > 0.5


@dataclass
class BackfillRecord:
    """回填記錄"""
    record_id: str
    batch_id: str
    raw_data: Dict[str, Any]
    attribute_parsing: Optional[AttributeParsingResult] = None
    category_classification: Optional[CategoryClassificationResult] = None
    normalization_result: Optional[Dict[str, Any]] = None
    processing_status: str = "pending"  # pending, success, failed, outlier
    error_messages: List[str] = field(default_factory=list)
    processing_time: Optional[datetime] = None
    
    def mark_as_success(self, normalized_data: Dict[str, Any]) -> None:
        """標記為成功處理"""
        self.normalization_result = normalized_data
        self.processing_status = "success"
        self.processing_time = datetime.now()
    
    def mark_as_failed(self, error: str) -> None:
        """標記為處理失敗"""
        self.error_messages.append(error)
        self.processing_status = "failed"
        self.processing_time = datetime.now()
    
    def mark_as_outlier(self, reason: str) -> None:
        """標記為異常值"""
        self.error_messages.append(f"Outlier: {reason}")
        self.processing_status = "outlier"
        self.processing_time = datetime.now()


@dataclass
class BackfillConfig(WorkflowConfig):
    """回填工作流配置"""
    target_domain: str
    old_tn_version: str
    new_tn_version: str
    tn_version_diff: TNVersionDiff
    backfill_strategy: BackfillStrategy = BackfillStrategy.INCREMENTAL
    batch_size: int = 1000
    stop_on_diff_gt: float = 0.05  # 5% 差異率上限
    conflict_resolution: NormalizationConflictResolution = NormalizationConflictResolution.USE_LATEST_TN
    enable_outlier_detection: bool = True
    outlier_threshold: float = 0.3  # 低於此信心分數視為異常
    max_retry_per_batch: int = 3
    enable_pre_commit_snapshot: bool = True
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
        
        if not (0.0 <= self.outlier_threshold <= 1.0):
            result.add_error("outlier_threshold must be between 0.0 and 1.0")
        
        return result


@dataclass
class BackfillResult(WorkflowResult):
    """回填結果"""
    backfill_progress: Optional[BackfillProgress] = None
    processed_batches: List[BackfillBatch] = field(default_factory=list)
    data_quality_metrics: Optional[DataQualityMetrics] = None
    pre_commit_snapshot: Optional[SnapshotInfo] = None
    post_commit_snapshot: Optional[SnapshotInfo] = None
    governance_events: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_overall_success_rate(self) -> float:
        """獲取整體成功率"""
        if not self.processed_batches:
            return 0.0
        
        total_records = sum(batch.total_records for batch in self.processed_batches)
        success_records = sum(batch.success_records for batch in self.processed_batches)
        
        if total_records == 0:
            return 0.0
        
        return success_records / total_records
    
    def get_batch_summary(self) -> Dict[str, Any]:
        """獲取批次處理摘要"""
        return {
            "total_batches": len(self.processed_batches),
            "successful_batches": len([b for b in self.processed_batches if b.success_rate >= 0.95]),
            "failed_batches": len([b for b in self.processed_batches if b.success_rate < 0.5]),
            "total_records": sum(b.total_records for b in self.processed_batches),
            "success_records": sum(b.success_records for b in self.processed_batches),
            "failed_records": sum(b.failed_records for b in self.processed_batches),
            "outlier_records": sum(b.outlier_records for b in self.processed_batches),
            "overall_success_rate": self.get_overall_success_rate()
        }


# 抽象類別定義
class TNDiffAnalyzer(BaseValidator):
    """T/N 差異分析器"""
    
    def analyze_tn_changes(self, old_version: str, new_version: str) -> TNVersionDiff:
        """分析 T/N 版本變更"""
        raise NotImplementedError
    
    def validate_diff_safety(self, diff: TNVersionDiff) -> ValidationResult:
        """驗證差異變更的安全性"""
        raise NotImplementedError


class AttributeParser(BaseProcessor):
    """屬性解析器"""
    
    def parse_attributes(self, raw_data: Dict[str, Any], 
                        value_mapping: Dict[str, Any]) -> AttributeParsingResult:
        """解析原始資料屬性"""
        raise NotImplementedError
    
    def match_keywords(self, keywords: List[str], 
                      value_mapping: Dict[str, Any]) -> Dict[str, str]:
        """匹配關鍵字到正規化值"""
        raise NotImplementedError


class CategoryClassifier(BaseProcessor):
    """分類掛載器"""
    
    def classify_category(self, raw_category_path: str,
                         existing_mappings: Dict[str, str]) -> CategoryClassificationResult:
        """分類掛載原始分類路徑"""
        raise NotImplementedError
    
    def apply_heuristics(self, raw_category_path: str) -> CategoryClassificationResult:
        """應用啟發式規則進行分類"""
        raise NotImplementedError


class BackfillProcessor(BaseProcessor):
    """回填處理器主類"""
    
    def __init__(self, config: BackfillConfig):
        super().__init__(config)
        self.config: BackfillConfig = config
    
    def select_batches(self) -> List[str]:
        """選擇需要回填的批次"""
        raise NotImplementedError
    
    def process_batch(self, batch_path: str) -> BackfillBatch:
        """處理單一批次"""
        raise NotImplementedError
    
    def create_pre_commit_snapshot(self) -> SnapshotInfo:
        """建立預提交快照"""
        raise NotImplementedError
    
    def generate_governance_events(self, result: BackfillResult) -> List[Dict[str, Any]]:
        """生成治理事件"""
        raise NotImplementedError


# 輔助函數
def create_backfill_config(
    workflow_id: str,
    target_domain: str,
    old_tn_version: str,
    new_tn_version: str,
    tn_version_diff: TNVersionDiff,
    **kwargs
) -> BackfillConfig:
    """建立回填配置"""
    return BackfillConfig(
        workflow_id=workflow_id,
        domain=target_domain,
        target_domain=target_domain,
        old_tn_version=old_tn_version,
        new_tn_version=new_tn_version,
        tn_version_diff=tn_version_diff,
        **kwargs
    )


def validate_backfill_preconditions(config: BackfillConfig) -> ValidationResult:
    """驗證回填前置條件"""
    result = ValidationResult(is_valid=True)
    
    # 檢查 T/N 版本差異是否合理
    if config.tn_version_diff.has_breaking_changes():
        result.add_warning("Detected breaking changes in T/N version diff")
    
    # 檢查配置有效性
    config_validation = config.validate()
    result.merge(config_validation)
    
    return result