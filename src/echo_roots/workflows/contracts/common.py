from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod


# 基礎枚舉類型 (保持不變)
class WorkflowStatus(Enum):
    """工作流狀態"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLBACK = "rollback"


class DataSource(Enum):
    """資料來源類型"""
    API = "api"
    FILE = "file"
    DATABASE = "database"
    MANUAL = "manual"


class Language(Enum):
    """支援的語言"""
    EN = "en"
    ZH_TW = "zh_tw"
    ZH_CN = "zh_cn"
    JA = "ja"


class DatabaseType(Enum):
    """資料庫類型"""
    DUCKDB = "duckdb"
    NEO4J = "neo4j"
    QDRANT = "qdrant"


# 內部計算層 - 使用 dataclass
@dataclass
class ValidationResult:
    """驗證結果 - 內部計算物件"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, error: str) -> None:
        """添加錯誤訊息"""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """添加警告訊息"""
        self.warnings.append(warning)
    
    def merge(self, other: 'ValidationResult') -> None:
        """合併其他驗證結果"""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.is_valid = self.is_valid and other.is_valid


@dataclass
class WorkflowResult:
    """工作流執行結果 - 內部計算物件"""
    status: WorkflowStatus
    workflow_id: str
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time_ms: Optional[int] = None
    
    def add_error(self, error: str) -> None:
        """添加錯誤訊息並標記為失敗"""
        self.errors.append(error)
        self.status = WorkflowStatus.FAILED
    
    def add_warning(self, warning: str) -> None:
        """添加警告訊息"""
        self.warnings.append(warning)
    
    def is_successful(self) -> bool:
        """檢查是否成功"""
        return self.status == WorkflowStatus.SUCCESS


@dataclass
class SnapshotConfig:
    """快照配置 - 內部配置物件"""
    create_before: bool = True
    create_after: bool = True
    snapshot_name_prefix: str = ""
    auto_cleanup: bool = True
    max_snapshots: int = 10
    
    def get_snapshot_name(self, stage: str, timestamp: datetime) -> str:
        """生成快照名稱"""
        prefix = f"{self.snapshot_name_prefix}_" if self.snapshot_name_prefix else ""
        time_str = timestamp.strftime("%Y%m%d_%H%M%S")
        return f"{prefix}{stage}_{time_str}"


@dataclass
class DataQualityMetrics:
    """資料品質指標 - 內部計算結果"""
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    duplicate_records: int = 0
    missing_values: Dict[str, int] = field(default_factory=dict)
    
    @property
    def validity_rate(self) -> float:
        """有效率"""
        if self.total_records == 0:
            return 0.0
        return self.valid_records / self.total_records
    
    @property
    def duplicate_rate(self) -> float:
        """重複率"""
        if self.total_records == 0:
            return 0.0
        return self.duplicate_records / self.total_records


@dataclass
class DiffMetrics:
    """差異指標 - 內部計算結果"""
    total_compared: int = 0
    identical: int = 0
    modified: int = 0
    added: int = 0
    removed: int = 0
    
    @property
    def change_rate(self) -> float:
        """變更率"""
        if self.total_compared == 0:
            return 0.0
        return (self.modified + self.added + self.removed) / self.total_compared
    
    @property
    def stability_rate(self) -> float:
        """穩定率"""
        if self.total_compared == 0:
            return 1.0
        return self.identical / self.total_compared


@dataclass
class SupportMetrics:
    """支援度指標 - 演算法計算結果"""
    support_count: int
    confidence_score: float
    coverage_ratio: float
    
    def __post_init__(self):
        """驗證數值範圍"""
        if self.support_count < 0:
            raise ValueError("support_count must be >= 0")
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError("confidence_score must be between 0.0 and 1.0")
        if not (0.0 <= self.coverage_ratio <= 1.0):
            raise ValueError("coverage_ratio must be between 0.0 and 1.0")


# 邊界層 - 使用 Pydantic BaseModel
class BatchMetadata(BaseModel):
    """批次元數據 - 外部資料交換"""
    event_id: str
    batch_key: str
    timestamp: datetime
    source: DataSource
    lang: Language
    domain: str
    version: str = "1.0"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_idempotency_key(self) -> str:
        """生成冪等性檢查鍵"""
        return f"{self.event_id}_{self.batch_key}_{self.domain}"


class DatabaseConnection(BaseModel):
    """資料庫連接資訊 - DB I/O 邊界層"""
    db_type: DatabaseType
    host: str
    port: int
    database: str
    username: Optional[str] = None
    password: Optional[str] = None
    connection_params: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('port')
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    def get_connection_string(self) -> str:
        """獲取連接字串"""
        if self.db_type == DatabaseType.DUCKDB:
            return self.database
        elif self.db_type == DatabaseType.NEO4J:
            return f"bolt://{self.host}:{self.port}"
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")


class SnapshotInfo(BaseModel):
    """快照資訊 - DB I/O 邊界層"""
    snapshot_id: str
    name: str
    timestamp: datetime
    db_type: DatabaseType
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowConfig(BaseModel):
    """基礎工作流配置 - API 邊界層"""
    workflow_id: str
    domain: str
    timeout_seconds: int = Field(default=3600, gt=0)
    retry_attempts: int = Field(default=3, ge=0)
    enable_snapshots: bool = True
    snapshot_config: SnapshotConfig = Field(default_factory=SnapshotConfig)
    duckdb_connection: Optional[DatabaseConnection] = None
    neo4j_connection: Optional[DatabaseConnection] = None


class GovernanceEvent(BaseModel):
    """治理事件 - DB I/O 邊界層"""
    event_id: str
    event_type: "GovernanceEventType"
    workflow_id: str
    domain: str
    timestamp: datetime
    event_data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessingResult(BaseModel):
    """通用處理結果 - API 邊界層"""
    success: bool
    processed_count: int
    error_count: int
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchKey(BaseModel):
    """批次識別 - 外部資料交換"""
    domain: str
    batch_id: str
    timestamp: datetime
    source_path: Optional[str] = None


class VersionInfo(BaseModel):
    """版本對齊資訊 - DB I/O 邊界層"""
    norm_version: str
    n_version: str
    t_version: Optional[str] = None
    
    def is_aligned(self) -> bool:
        """Check if norm and n versions are aligned"""
        return self.norm_version == self.n_version


class SnapshotRequest(BaseModel):
    """建立快照請求 - API 邊界層"""
    snapshot_type: "SnapshotType"
    reason: str
    batch_key: BatchKey
    metadata: Dict[str, Any] = Field(default_factory=dict)


# 治理相關枚舉
class GovernanceEventType(Enum):
    """治理事件類型"""
    SNAPSHOT_CREATE = "snapshot_create"
    SNAPSHOT_RESTORE = "snapshot_restore"
    BACKFILL_COMMIT = "backfill_commit"
    DATA_QUALITY_CHECK = "data_quality_check"
    SCHEMA_CHANGE = "schema_change"
    WORKFLOW_START = "workflow_start"
    WORKFLOW_COMPLETE = "workflow_complete"
    WORKFLOW_FAILED = "workflow_failed"
    


class EventType(str, Enum):
    """Common event types across workflows"""
    INGEST_COMMIT = "INGEST_COMMIT"
    SNAPSHOT_CREATE = "SNAPSHOT_CREATE"
    NORM_UPDATE = "NORM_UPDATE"
    PROPOSAL_SUBMIT = "PROPOSAL_SUBMIT"
    CLUSTER_UPDATE = "CLUSTER_UPDATE"


class SnapshotType(str, Enum):
    """Snapshot types"""
    DATA = "D"
    TAXONOMY = "T"
    NORM = "N"
    SEMANTIC = "S"


class ClusterStatus(str, Enum):
    """聚類狀態枚舉"""
    COLD = "COLD"
    WARM = "WARM" 
    HOT = "HOT"
    ARCHIVED = "ARCHIVED"


# 抽象基類
class BaseValidator(ABC):
    """基礎驗證器"""
    
    @abstractmethod
    def validate(self, data: Any) -> ValidationResult:
        """驗證資料"""
        pass


class BaseProcessor(ABC):
    """基礎處理器"""
    
    def __init__(self, config: WorkflowConfig):
        self.config = config
    
    @abstractmethod
    def process(self, input_data: Any) -> WorkflowResult:
        """處理資料"""
        pass
    
    def pre_process(self, input_data: Any) -> ValidationResult:
        """預處理驗證"""
        return ValidationResult(is_valid=True)
    
    def post_process(self, result: WorkflowResult) -> WorkflowResult:
        """後處理"""
        return result


class SnapshotManager(ABC):
    """快照管理器抽象類"""
    
    @abstractmethod
    def create_snapshot(self, name: str, db_type: DatabaseType) -> SnapshotInfo:
        """建立快照"""
        pass
    
    @abstractmethod
    def restore_snapshot(self, snapshot_id: str) -> bool:
        """恢復快照"""
        pass
    
    @abstractmethod
    def list_snapshots(self, db_type: Optional[DatabaseType] = None) -> List[SnapshotInfo]:
        """列出快照"""
        pass
    
    @abstractmethod
    def delete_snapshot(self, snapshot_id: str) -> bool:
        """刪除快照"""
        pass


# 輔助函數
def create_batch_metadata(event_id: str, 
                         batch_key: str, 
                         source: Union[DataSource, str],
                         lang: Union[Language, str],
                         domain: str) -> BatchMetadata:
    """建立批次元數據"""
    if isinstance(source, str):
        source = DataSource(source)
    if isinstance(lang, str):
        lang = Language(lang)
    
    return BatchMetadata(
        event_id=event_id,
        batch_key=batch_key,
        timestamp=datetime.now(),
        source=source,
        lang=lang,
        domain=domain
    )


def merge_validation_results(results: List[ValidationResult]) -> ValidationResult:
    """合併多個驗證結果"""
    merged = ValidationResult(is_valid=True)
    for result in results:
        merged.merge(result)
    return merged


def create_workflow_result(
    workflow_id: str, 
    status: WorkflowStatus = WorkflowStatus.SUCCESS,
    message: str = ""
) -> WorkflowResult:
    """建立工作流結果"""
    return WorkflowResult(
        status=status,
        workflow_id=workflow_id,
        message=message
    )


def create_governance_event(
    event_type: GovernanceEventType,
    workflow_id: str,
    domain: str,
    event_data: Dict[str, Any] = None,
    metadata: Dict[str, Any] = None
) -> GovernanceEvent:
    """建立治理事件"""
    return GovernanceEvent(
        event_id=f"{workflow_id}_{event_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        event_type=event_type,
        workflow_id=workflow_id,
        domain=domain,
        timestamp=datetime.now(),
        event_data=event_data or {},
        metadata=metadata or {}
    )


def calculate_diff_metrics(old_data: List[Any], new_data: List[Any]) -> DiffMetrics:
    """計算差異指標"""
    metrics = DiffMetrics()
    
    old_set = set(str(item) for item in old_data)
    new_set = set(str(item) for item in new_data)
    
    metrics.total_compared = len(old_set | new_set)
    metrics.identical = len(old_set & new_set)
    metrics.removed = len(old_set - new_set)
    metrics.added = len(new_set - old_set)
    
    return metrics

# W5 專用共用元件追加到檔案末尾

class MaintenanceEventType(str, Enum):
    """維護事件類型 - W5 專用"""
    PROPOSAL_REVIEW_BATCH = "PROPOSAL_REVIEW_BATCH"
    TAXONOMY_SYNC = "TAXONOMY_SYNC"
    VALUE_MAPPING_SYNC = "VALUE_MAPPING_SYNC"
    VERSION_POINT_CREATE = "VERSION_POINT_CREATE"
    CONSISTENCY_CHECK = "CONSISTENCY_CHECK"


@dataclass
class BackfillPlan:
    """回填計畫 - W5 內部計算物件"""
    target_entities: List[str] = field(default_factory=list)
    estimated_records: int = 0
    priority: int = 1  # 1=high, 2=medium, 3=low
    estimated_duration_minutes: int = 0
    dependencies: List[str] = field(default_factory=list)
    
    def get_execution_order(self) -> int:
        """獲取執行順序 (priority 越小越優先)"""
        return self.priority


def create_maintenance_batch_metadata(
    event_id: str,
    reviewer_id: str,
    domain: str,
    proposal_count: int
) -> BatchMetadata:
    """建立維護批次元數據"""
    return BatchMetadata(
        event_id=event_id,
        batch_key=f"maintenance_{reviewer_id}_{proposal_count}",
        timestamp=datetime.now(),
        source=DataSource.MANUAL,
        lang=Language.ZH_TW,  # 預設繁中
        domain=domain,
        metadata={
            "reviewer_id": reviewer_id,
            "proposal_count": str(proposal_count),
            "operation_type": "maintenance"
        }
    )

# end of W5 additions

# W6 專用共用元件追加到檔案末尾

class SemanticEventType(str, Enum):
    """語意治理事件類型 - W6 專用"""
    CANDIDATE_MERGE = "CANDIDATE_MERGE"
    PROPOSAL_CREATE = "PROPOSAL_CREATE" 
    CLUSTER_ANALYSIS = "CLUSTER_ANALYSIS"
    SEMANTIC_MAPPING = "SEMANTIC_MAPPING"
    GOVERNANCE_DECISION = "GOVERNANCE_DECISION"


@dataclass
class ClusteringMetrics:
    """聚類指標 - W6 內部計算結果"""
    total_observations: int = 0
    total_clusters: int = 0
    avg_cluster_size: float = 0.0
    max_cluster_size: int = 0
    min_cluster_size: int = 0
    singleton_count: int = 0
    purity_score: float = 0.0
    
    @property
    def clustering_efficiency(self) -> float:
        """聚類效率 (非單例的比例)"""
        if self.total_observations == 0:
            return 0.0
        return (self.total_observations - self.singleton_count) / self.total_observations
    
    @property
    def avg_support_per_cluster(self) -> float:
        """平均每個聚類的支援度"""
        if self.total_clusters == 0:
            return 0.0
        return self.total_observations / self.total_clusters


@dataclass  
class SimilarityMetrics:
    """相似度計算指標 - W6 內部計算"""
    similarity_score: float
    method: str  # "cosine", "jaccard", "edit_distance" etc.
    confidence: float = 0.0
    
    def __post_init__(self):
        """驗證數值範圍"""
        if not (0.0 <= self.similarity_score <= 1.0):
            raise ValueError("similarity_score must be between 0.0 and 1.0")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")


def create_semantic_batch_metadata(
    event_id: str,
    domain: str,
    category_id: str,
    observation_count: int
) -> BatchMetadata:
    """建立語意治理批次元數據"""
    return BatchMetadata(
        event_id=event_id,
        batch_key=f"semantic_{domain}_{category_id}_{observation_count}",
        timestamp=datetime.now(),
        source=DataSource.DATABASE,
        lang=Language.ZH_TW,
        domain=domain,
        metadata={
            "category_id": category_id,
            "observation_count": str(observation_count),
            "operation_type": "semantic_governance"
        }
    )

# end of W6 additions