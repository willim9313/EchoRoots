from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

from .common import BatchMetadata, ValidationResult, WorkflowResult, SnapshotConfig


@dataclass
class CategoryHint:
    """分類提示資訊"""
    name: str
    parent_name: Optional[str] = None
    level: int = 0
    description: Optional[str] = None


@dataclass
class InitialCategory:
    """初始分類節點"""
    id: str
    name: str
    parent_id: Optional[str] = None
    level: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InitialAttribute:
    """初始屬性"""
    id: str
    name: str
    category_id: str
    data_type: str
    is_required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InitialValue:
    """初始值"""
    id: str
    attribute_id: str
    raw_value: str
    normalized_value: str
    frequency: int = 1


@dataclass
class ValueMapping:
    """值映射"""
    raw_value: str
    normalized_value: str
    attribute_id: str
    confidence: float = 1.0


@dataclass
class W1InitConfig:
    """W1 初始化配置"""
    domain: str
    min_category_support: int = 10
    snapshot_config: SnapshotConfig = field(default_factory=SnapshotConfig)
    duckdb_init_script: str = "duckdb/init/00_init.sql"
    neo4j_init_script: str = "neo4j/init/00_init.cypher"
    t_init_file: Optional[str] = None
    n_init_file: Optional[str] = None


@dataclass
class W1InitInput:
    """W1 初始化輸入"""
    config: W1InitConfig
    batch_metadata: BatchMetadata
    d_raw_data: List[Dict[str, Any]]
    category_hints: List[CategoryHint] = field(default_factory=list)


@dataclass
class W1PreCheckResult:
    """W1 預檢結果"""
    validation_result: ValidationResult
    batch_integrity_ok: bool
    lang_source_valid: bool
    data_sample_count: int


@dataclass
class W1DatabaseInitResult:
    """資料庫初始化結果"""
    duckdb_initialized: bool
    neo4j_initialized: bool
    errors: List[str] = field(default_factory=list)


@dataclass
class W1StructureImportResult:
    """結構匯入結果"""
    categories_created: int
    attributes_created: int
    values_created: int
    categories: List[InitialCategory] = field(default_factory=list)
    attributes: List[InitialAttribute] = field(default_factory=list)
    values: List[InitialValue] = field(default_factory=list)


@dataclass
class W1ValueMappingResult:
    """值映射結果"""
    mappings_created: int
    mappings: List[ValueMapping] = field(default_factory=list)


@dataclass
class W1InitOutput:
    """W1 初始化輸出"""
    workflow_result: WorkflowResult
    pre_check_result: W1PreCheckResult
    database_init_result: W1DatabaseInitResult
    structure_import_result: W1StructureImportResult
    value_mapping_result: W1ValueMappingResult
    snapshot_ids: List[str] = field(default_factory=list)
    d_norm_created: bool = False


# 介面定義
class W1InitProcessor:
    """W1 初始化處理器介面"""
    
    def pre_check(self, input_data: W1InitInput) -> W1PreCheckResult:
        """執行預檢查"""
        raise NotImplementedError
    
    def initialize_databases(self, config: W1InitConfig) -> W1DatabaseInitResult:
        """初始化資料庫"""
        raise NotImplementedError
    
    def import_initial_structure(self, 
                                input_data: W1InitInput) -> W1StructureImportResult:
        """匯入初始結構"""
        raise NotImplementedError
    
    def create_value_mapping(self, 
                            input_data: W1InitInput,
                            structure_result: W1StructureImportResult) -> W1ValueMappingResult:
        """建立值映射"""
        raise NotImplementedError
    
    def create_snapshots(self, 
                        config: W1InitConfig, 
                        stage: str) -> List[str]:
        """建立快照"""
        raise NotImplementedError
    
    def rollback(self, snapshot_id: str) -> bool:
        """回滾到指定快照"""
        raise NotImplementedError
    
    def check_idempotency(self, batch_metadata: BatchMetadata) -> bool:
        """檢查冪等性"""
        raise NotImplementedError
    
    def process(self, input_data: W1InitInput) -> W1InitOutput:
        """執行完整的 W1 初始化流程"""
        raise NotImplementedError


class W1InitValidator:
    """W1 初始化驗證器介面"""
    
    def validate_d_raw_integrity(self, d_raw_data: List[Dict[str, Any]]) -> ValidationResult:
        """驗證 d_raw 批次完整性"""
        raise NotImplementedError
    
    def validate_lang_source(self, batch_metadata: BatchMetadata) -> ValidationResult:
        """驗證 lang 與 source 值"""
        raise NotImplementedError
    
    def validate_category_hints(self, category_hints: List[CategoryHint]) -> ValidationResult:
        """驗證分類提示"""
        raise NotImplementedError
    
    def validate_definition_of_done(self, output: W1InitOutput) -> ValidationResult:
        """驗證 Definition of Done"""
        raise NotImplementedError