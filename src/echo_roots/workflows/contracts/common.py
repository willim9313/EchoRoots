from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod


# 基礎枚舉類型
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


# 基礎資料結構
@dataclass
class ValidationResult:
    """驗證結果"""
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
class BatchMetadata:
    """批次元數據"""
    event_id: str
    batch_key: str
    timestamp: datetime
    source: DataSource
    lang: Language
    domain: str
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_idempotency_key(self) -> str:
        """生成冪等性檢查鍵"""
        return f"{self.event_id}_{self.batch_key}_{self.domain}"
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "event_id": self.event_id,
            "batch_key": self.batch_key,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source.value,
            "lang": self.lang.value,
            "domain": self.domain,
            "version": self.version,
            "metadata": self.metadata
        }


@dataclass
class WorkflowResult:
    """工作流執行結果"""
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
class DatabaseConnection:
    """資料庫連接資訊"""
    db_type: DatabaseType
    host: str
    port: int
    database: str
    username: Optional[str] = None
    password: Optional[str] = None
    connection_params: Dict[str, Any] = field(default_factory=dict)
    
    def get_connection_string(self) -> str:
        """獲取連接字串"""
        if self.db_type == DatabaseType.DUCKDB:
            return self.database
        elif self.db_type == DatabaseType.NEO4J:
            return f"bolt://{self.host}:{self.port}"
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")


@dataclass
class SnapshotInfo:
    """快照資訊"""
    snapshot_id: str
    name: str
    timestamp: datetime
    db_type: DatabaseType
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "snapshot_id": self.snapshot_id,
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "db_type": self.db_type.value,
            "metadata": self.metadata
        }


@dataclass
class SnapshotConfig:
    """快照配置"""
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
    """資料品質指標"""
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
class WorkflowConfig:
    """基礎工作流配置"""
    workflow_id: str
    domain: str
    timeout_seconds: int = 3600
    retry_attempts: int = 3
    enable_snapshots: bool = True
    snapshot_config: SnapshotConfig = field(default_factory=SnapshotConfig)
    duckdb_connection: Optional[DatabaseConnection] = None
    neo4j_connection: Optional[DatabaseConnection] = None
    
    def validate(self) -> ValidationResult:
        """驗證配置"""
        result = ValidationResult(is_valid=True)
        
        if not self.workflow_id:
            result.add_error("workflow_id is required")
        
        if not self.domain:
            result.add_error("domain is required")
        
        if self.timeout_seconds <= 0:
            result.add_error("timeout_seconds must be positive")
        
        if self.retry_attempts < 0:
            result.add_error("retry_attempts cannot be negative")
        
        return result


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

# workflow 2 append new 

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


@dataclass
class GovernanceEvent:
    """治理事件"""
    event_id: str
    event_type: GovernanceEventType
    workflow_id: str
    domain: str
    timestamp: datetime
    event_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "workflow_id": self.workflow_id,
            "domain": self.domain,
            "timestamp": self.timestamp.isoformat(),
            "event_data": self.event_data,
            "metadata": self.metadata
        }


@dataclass
class DiffMetrics:
    """差異指標"""
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
    # 這是一個簡化的實作，實際實作需要根據資料類型進行比較
    metrics = DiffMetrics()
    
    old_set = set(str(item) for item in old_data)
    new_set = set(str(item) for item in new_data)
    
    metrics.total_compared = len(old_set | new_set)
    metrics.identical = len(old_set & new_set)
    metrics.removed = len(old_set - new_set)
    metrics.added = len(new_set - old_set)
    
    return metrics

# end of workflow 2 related

# workflow 3 append new 
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

class ProcessingResult(BaseModel):
    """Common processing result structure"""
    success: bool
    processed_count: int
    error_count: int
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BatchKey(BaseModel):
    """Common batch identification"""
    domain: str
    batch_id: str
    timestamp: datetime
    source_path: Optional[str] = None

class VersionInfo(BaseModel):
    """Version alignment information"""
    norm_version: str
    n_version: str
    t_version: Optional[str] = None
    
    def is_aligned(self) -> bool:
        """Check if norm and n versions are aligned"""
        return self.norm_version == self.n_version

class GovernanceEvent(BaseModel):
    """Common governance event structure"""
    event_type: EventType
    event_id: str
    batch_key: BatchKey
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SnapshotRequest(BaseModel):
    """Request to create a snapshot"""
    snapshot_type: SnapshotType
    reason: str
    batch_key: BatchKey
    metadata: Dict[str, Any] = Field(default_factory=dict)

# end of workflow 3 related