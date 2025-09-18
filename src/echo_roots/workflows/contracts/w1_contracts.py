from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field

from .common import BatchMetadata, ValidationResult, WorkflowResult, SnapshotConfig, GovernanceEvent
from ...models import (
    Category, Attribute, AttributeValue, ValueMapping, ItemRaw, ItemNorm, 
    Domain, GovernanceEvent as ModelGovernanceEvent
)


# 內部計算層 - 使用 dataclass
@dataclass
class CategoryHint:
    """分類提示資訊 - 對應 w1_init.md 的 category_hint 輸入 - 內部計算物件"""
    name: str
    parent_name: Optional[str] = None
    level: int = 0
    domain: Domain
    description: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)  # 多語標籤


@dataclass
class W1InitConfig:
    """W1 初始化配置 - 內部配置物件"""
    domain: Domain  # 使用 models 中的 Domain enum
    min_category_support: int = 10  # 對應 w1_init.md 的 min_category_support
    snapshot_config: SnapshotConfig = field(default_factory=SnapshotConfig)
    duckdb_init_script: str = "duckdb/init/00_init.sql"
    neo4j_init_script: str = "neo4j/init/00_init.cypher"
    t_init_file: Optional[str] = None  # data/{domain}/t_init.json
    n_init_file: Optional[str] = None  # data/{domain}/n_init.json
    
    # W1 特定設定
    enable_value_mapping_creation: bool = True  # 對應步驟3的可選建立 value_mapping
    auto_create_categories_from_raw: bool = True


@dataclass
class W1PreCheckResult:
    """W1 預檢結果 - 內部計算結果"""
    validation_result: ValidationResult
    batch_integrity_ok: bool  # d_raw 批次完整性檢查
    lang_source_valid: bool   # lang 與 source 值是否合法
    data_sample_count: int
    version_compatibility: bool = True  # 檢查現有版本相容性
    estimated_categories: int = 0
    estimated_attributes: int = 0


@dataclass
class W1DatabaseInitResult:
    """資料庫初始化結果 - 內部計算結果"""
    duckdb_initialized: bool
    neo4j_initialized: bool
    duckdb_script_executed: bool
    neo4j_script_executed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class W1StructureImportResult:
    """結構匯入結果 - 內部計算結果"""
    categories_created: int
    attributes_created: int
    values_created: int
    
    # 使用 models 中的實際類型
    categories: List[Category] = field(default_factory=list)
    attributes: List[Attribute] = field(default_factory=list)  
    values: List[AttributeValue] = field(default_factory=list)
    
    # 匯入統計
    t_init_imported: bool = False
    n_init_imported: bool = False
    auto_generated_count: int = 0


@dataclass
class W1ValueMappingResult:
    """值映射結果 - 內部計算結果"""
    mappings_created: int
    mappings: List[ValueMapping] = field(default_factory=list)  # 使用 models 中的 ValueMapping
    base_mappings_imported: bool = False


@dataclass
class W1IdempotencyCheck:
    """冪等性檢查 - 內部計算結果"""
    event_id: str
    batch_key: str  # 批次鍵
    domain: Domain
    previous_run_detected: bool = False
    can_skip: bool = False
    conflicts: List[str] = field(default_factory=list)


@dataclass
class W1RollbackInfo:
    """回滾資訊 - 內部計算結果"""
    pre_neo4j_snapshot_id: Optional[str] = None   # 寫入 Neo4j 前快照
    post_neo4j_snapshot_id: Optional[str] = None  # 寫入 Neo4j 後快照
    rollback_available: bool = False


# 邊界層 - 使用 Pydantic BaseModel（API 輸入輸出）
class W1InitInput(BaseModel):
    """W1 初始化輸入 - API 邊界層"""
    config: W1InitConfig
    batch_metadata: BatchMetadata
    d_raw_data: List[ItemRaw]  # 使用 models 中的 ItemRaw 而非 Dict
    category_hints: List[CategoryHint] = Field(default_factory=list)


class W1InitOutput(BaseModel):
    """W1 初始化輸出 - API 邊界層"""
    workflow_result: WorkflowResult
    
    # 步驟結果
    pre_check_result: W1PreCheckResult
    database_init_result: W1DatabaseInitResult
    structure_import_result: W1StructureImportResult
    value_mapping_result: W1ValueMappingResult
    
    # 主要輸出 - 對應 w1_init.md 的 Outputs
    neo4j_categories_created: bool = False  # Neo4j: Category 節點
    duckdb_d_norm_created: bool = False     # DuckDB: 空或極簡的 d_norm_{domain}
    
    # Side Effects - 對應 w1_init.md 的 SNAPSHOT_CREATE (T,N)
    snapshot_ids: List[str] = Field(default_factory=list)
    governance_events: List[ModelGovernanceEvent] = Field(default_factory=list)
    
    # Definition of Done 檢查結果
    taxonomy_structure_browsable: bool = False  # 可瀏覽的根→葉分類結構
    d_norm_sample_mappable: bool = False       # d_norm 可對應上少量樣本
    
    # 版本資訊
    t_version_created: int = 1
    n_version_created: int = 1


# 介面定義 - 對應 w1_init.md 的完整步驟
class W1InitProcessor:
    """W1 初始化處理器介面"""
    
    def pre_check(self, input_data: W1InitInput) -> W1PreCheckResult:
        """執行預檢查 - 對應 Pre-checks"""
        raise NotImplementedError
    
    def check_idempotency(self, batch_metadata: BatchMetadata) -> W1IdempotencyCheck:
        """檢查冪等性 - 以 event_id + 批次鍵 防止重複寫入"""
        raise NotImplementedError
    
    def initialize_databases(self, config: W1InitConfig) -> W1DatabaseInitResult:
        """初始化資料庫 - 對應步驟1"""
        raise NotImplementedError
    
    def import_initial_structure(self, 
                                input_data: W1InitInput) -> W1StructureImportResult:
        """匯入初始結構 - 對應步驟2"""
        raise NotImplementedError
    
    def create_value_mapping(self, 
                            input_data: W1InitInput,
                            structure_result: W1StructureImportResult) -> W1ValueMappingResult:
        """建立值映射 - 對應步驟3（可選）"""
        raise NotImplementedError
    
    def create_snapshots(self, 
                        config: W1InitConfig, 
                        stage: str) -> List[str]:
        """建立快照 - 對應 Rollback 需求"""
        raise NotImplementedError
    
    def rollback(self, snapshot_id: str) -> bool:
        """回滾到指定快照"""
        raise NotImplementedError
    
    def validate_definition_of_done(self, output: W1InitOutput) -> ValidationResult:
        """驗證 Definition of Done - 檢查根→葉結構和樣本映射"""
        raise NotImplementedError
    
    def process(self, input_data: W1InitInput) -> W1InitOutput:
        """執行完整的 W1 初始化流程"""
        raise NotImplementedError


class W1InitValidator:
    """W1 初始化驗證器介面"""
    
    def validate_d_raw_integrity(self, d_raw_data: List[ItemRaw]) -> ValidationResult:
        """驗證 d_raw 批次完整性檢查"""
        raise NotImplementedError
    
    def validate_lang_source(self, batch_metadata: BatchMetadata) -> ValidationResult:
        """驗證 lang 與 source 值是否合法"""
        raise NotImplementedError
    
    def validate_category_hints(self, category_hints: List[CategoryHint]) -> ValidationResult:
        """驗證分類提示"""
        raise NotImplementedError
    
    def validate_min_category_support(self, 
                                     categories: List[Category], 
                                     min_support: int) -> ValidationResult:
        """驗證最小分類支援閾值"""
        raise NotImplementedError


# 輔助函數
def create_w1_config(domain: Domain, **kwargs) -> W1InitConfig:
    """建立 W1 配置"""
    return W1InitConfig(domain=domain, **kwargs)


def extract_category_hints_from_raw(d_raw_data: List[ItemRaw]) -> List[CategoryHint]:
    """從原始資料提取分類提示"""
    hints = {}
    for item in d_raw_data:
        for i, category_name in enumerate(item.raw_category_path):
            if category_name not in hints:
                parent_name = item.raw_category_path[i-1] if i > 0 else None
                hints[category_name] = CategoryHint(
                    name=category_name,
                    parent_name=parent_name,
                    level=i,
                    domain=item.domain
                )
    return list(hints.values())


def validate_w1_output_completeness(output: W1InitOutput) -> ValidationResult:
    """驗證 W1 輸出完整性 - 確保符合 Definition of Done"""
    result = ValidationResult(is_valid=True)
    
    if not output.neo4j_categories_created:
        result.add_error("Neo4j categories not created")
    
    if not output.taxonomy_structure_browsable:
        result.add_error("Taxonomy structure not browsable (root → leaf)")
    
    if output.structure_import_result.categories_created == 0:
        result.add_error("No categories were created")
    
    return result