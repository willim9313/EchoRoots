"""W1 初始化工作流主程式 - 核心邏輯實作"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from uuid import uuid4
from datetime import datetime

from src.echo_roots.models import Category, Attribute, AttributeValue, ItemRaw, Domain
from src.echo_roots.workflows.contracts.w1_contracts import (
    W1InitInput, W1InitOutput, W1InitProcessor, W1InitConfig,
    W1PreCheckResult, W1DatabaseInitResult, W1StructureImportResult, 
    W1ValueMappingResult, W1IdempotencyCheck, CategoryHint
)
from src.echo_roots.workflows.contracts.common import (
    WorkflowResult, WorkflowStatus, ValidationResult, GovernanceEvent,
    BatchMetadata
)
from .w1_database import W1DatabaseManager
from .w1_validator import W1Validator

logger = logging.getLogger(__name__)


class W1InitProcessorImpl(W1InitProcessor):
    """W1 初始化處理器實作 - 簡化版"""

    def __init__(self, db_manager: W1DatabaseManager, validator: W1Validator):
        self.db_manager = db_manager
        self.validator = validator
        self.snapshots_created = []
        self.event_id = str(uuid4())

    def pre_check(self, input_data: W1InitInput) -> W1PreCheckResult:
        """前置檢查 - 驗證輸入資料完整性"""
        logger.info("開始前置檢查，批次: %s", input_data.batch_metadata.batch_key)

        # 檢查 d_raw 批次完整性
        raw_validation = self.validator.validate_d_raw_integrity(input_data.d_raw_data)

        # 檢查 lang 與 source 值
        lang_source_validation = self.validator.validate_lang_source(input_data.batch_metadata)

        # 檢查最小分類支援閾值
        # category_support_ok = self._check_min_category_support(input_data)

        # extract error messages
        is_valid = raw_validation.is_valid and lang_source_validation.is_valid
        error_messages = [
            msg for validation in [raw_validation, lang_source_validation] 
            for msg in validation.errors if not validation.is_valid
        ]

        return W1PreCheckResult(
            validation_result=ValidationResult(
                is_valid=is_valid,
                errors=error_messages,
                warnings=[]
            ),
            batch_integrity_ok=raw_validation.is_valid,
            lang_source_valid=lang_source_validation.is_valid,
            data_sample_count=len(input_data.d_raw_data)
        )

    def check_idempotency(self, batch_metadata: BatchMetadata) -> W1IdempotencyCheck:
        """檢查冪等性 - 避免重複執行"""
        idempotency_key = batch_metadata.get_idempotency_key()

        # 檢查是否已經執行過
        already_processed = self.db_manager.check_batch_processed(idempotency_key)

        return W1IdempotencyCheck(
            event_id=self.event_id,
            batch_key=batch_metadata.batch_key,
            domain=batch_metadata.domain,
            previous_run_detected=already_processed,
            can_skip=already_processed,
            conflicts=[] #previous_result_id=None  # TODO: 如果已處理，回傳之前的結果ID
        )

    def initialize_databases(self, config: W1InitConfig) -> W1DatabaseInitResult:
        """初始化資料庫 - 執行初始化腳本"""
        logger.info("開始初始化資料庫")

        try:
            # 執行 DuckDB 初始化
            duckdb_success = self.db_manager.init_duckdb()

            # 執行 Neo4j 初始化
            neo4j_success = self.db_manager.init_neo4j()

            return W1DatabaseInitResult(
                duckdb_initialized=duckdb_success,
                neo4j_initialized=neo4j_success,
                duckdb_script_executed=duckdb_success,
                neo4j_script_executed=neo4j_success,
                errors=[] if (duckdb_success and neo4j_success) else ["資料庫初始化失敗"],
                warnings=[]
            )

        except Exception as e:
            logger.error("資料庫初始化失敗: %s", e)
            return W1DatabaseInitResult(
                duckdb_initialized=False,
                neo4j_initialized=False,
                duckdb_script_executed=False,
                neo4j_script_executed=False,
                errors=[str(e)],
                warnings=[]
            )

    def import_initial_structure(self, input_data: W1InitInput) -> W1StructureImportResult:
        """匯入初始結構 - 從 t_init.json, n_init.json 匯入"""
        logger.info("開始匯入初始結構")

        try:
            # 讀取初始分類結構
            categories = self._load_initial_categories(input_data.config.domain)
            categories_imported = self.db_manager.import_categories(categories)

            # 讀取初始屬性結構  
            attributes = self._load_initial_attributes(input_data.config.domain)
            attributes_imported = self.db_manager.import_attributes(attributes)

            # 處理 category_hints（如果有）, 但是實際上應該沒辦法直接正確對應，會需要做過一次分類處理
            hints_processed = self._process_category_hints(input_data.category_hints)

            return W1StructureImportResult(
                categories_created=categories_imported,
                attributes_created=attributes_imported,
                values_created=0,
                categories=categories,
                attributes=attributes,
                values=[],  # 這應該會是attribute values, 但上面還沒有處理好
                t_init_imported=True if categories_imported else False,
                n_init_imported=True if attributes_imported else False,
                auto_generated_count=0
            )

        except Exception as e:
            logger.error("初始結構匯入失敗: %s", e)
            return W1StructureImportResult(
                categories_created=0,
                attributes_created=0,
                values_created=0,
                categories=[],
                attributes=[],
                values=[],  # 這應該會是attribute values, 但上面還沒有處理好
                t_init_imported=False,
                n_init_imported=False,
                auto_generated_count=0,
                error_messages=[str(e)]
            )

    def create_value_mapping(
        self,
        input_data: W1InitInput,
        structure_result: W1StructureImportResult
    ) -> W1ValueMappingResult:
        """建立值映射 - 可選步驟"""
        logger.info("開始建立基礎值映射")

        try:
            # TODO: 實作基礎值映射邏輯
            # 目前先回傳空的結果
            return W1ValueMappingResult(
                mappings_created=0,
                mappings=[],  # 應該會是[ValueMapping]
                base_mappings_imported=True
            )

        except Exception as e:
            logger.error("值映射建立失敗: %s", e)
            return W1ValueMappingResult(
                mappings_created=0,
                mappings=[],
                base_mappings_imported=False
            )

    def create_snapshots(self, config: W1InitConfig, stage: str) -> List[str]:
        """建立快照 - 用於 Rollback"""
        logger.info("建立快照: %s", stage)

        try:
            snapshot_id = f"w1_{stage}_{datetime.now().isoformat()}"

            # 建立 Neo4j 快照
            neo4j_snapshot = self.db_manager.create_neo4j_snapshot(snapshot_id)

            # 建立 DuckDB 快照
            duckdb_snapshot = self.db_manager.create_duckdb_snapshot(snapshot_id)

            if neo4j_snapshot and duckdb_snapshot:
                self.snapshots_created.append(snapshot_id)
                return [snapshot_id]

            return []

        except Exception as e:
            logger.error("快照建立失敗: %s", e)
            return []

    def rollback(self, snapshot_id: str) -> bool:
        """回滾到指定快照"""
        logger.warning("開始回滾到快照: %s", snapshot_id)

        try:
            neo4j_rollback = self.db_manager.restore_neo4j_snapshot(snapshot_id)
            duckdb_rollback = self.db_manager.restore_duckdb_snapshot(snapshot_id)

            return neo4j_rollback and duckdb_rollback

        except Exception as e:
            logger.error("回滾失敗: %s", e)
            return False

    def validate_definition_of_done(self, output: W1InitOutput) -> ValidationResult:
        """驗證 Definition of Done"""
        errors = []

        # 檢查可瀏覽的分類結構
        if not output.taxonomy_structure_browsable:
            errors.append("分類結構不可瀏覽")

        # 檢查 d_norm 樣本映射
        if not output.d_norm_sample_mappable:
            errors.append("d_norm 無法映射樣本")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=[]
        )

    def process(self, input_data: W1InitInput) -> W1InitOutput:
        """主要處理流程 - W1 完整執行"""
        logger.info("開始執行 W1 初始化流程，批次: %s", input_data.batch_metadata.batch_key)

        # 初始化輸出結果
        output = W1InitOutput(
            workflow_result=WorkflowResult(
                status=WorkflowStatus.RUNNING,
                workflow_id="W1",
                message="W1 初始化流程開始",
                data=Dict(),
                errors=[],
                warnings=[],
                execution_time_ms=datetime.now()
            ),
            pre_check_result=None,
            database_init_result=None,
            structure_import_result=None,
            value_mapping_result=None,
            snapshot_ids=[]
        )

        try:
            # 步驟 1: 前置檢查
            output.pre_check_result = self.pre_check(input_data)
            if not self._all_pre_checks_passed(output.pre_check_result):
                output.workflow_result.status = WorkflowStatus.FAILED
                output.workflow_result.message = "前置檢查失敗"
                return output

            # 步驟 2: 檢查冪等性
            idempotency_check = self.check_idempotency(input_data.batch_metadata)
            if idempotency_check.previous_run_detected:
                logger.info("批次已處理過，跳過執行")
                output.workflow_result.status = WorkflowStatus.SUCCESS
                output.workflow_result.message = "批次已處理過"
                return output

            # 步驟 3: 建立前置快照
            pre_snapshots = self.create_snapshots(input_data.config, "pre")
            if not hasattr(output, 'snapshot_ids') or output.snapshot_ids is None:
                output.snapshot_ids = []
            output.snapshot_ids.extend(pre_snapshots)

            # 步驟 4: 初始化資料庫
            output.database_init_result = self.initialize_databases(input_data.config)
            if not self._database_init_success(output.database_init_result):
                self._handle_failure(output, "資料庫初始化失敗")
                return output

            # 步驟 5: 匯入初始結構
            output.structure_import_result = self.import_initial_structure(input_data)
            if output.structure_import_result.error_messages:
                self._handle_failure(output, "初始結構匯入失敗")
                return output

            # 步驟 6: 建立值映射（可選）
            output.value_mapping_result = self.create_value_mapping(
                input_data, output.structure_import_result
            )

            # 步驟 7: 建立後置快照
            post_snapshots = self.create_snapshots(input_data.config, "post")
            if not hasattr(output, 'snapshot_ids') or output.snapshot_ids is None:
                output.snapshot_ids = []
            output.snapshot_ids.extend(post_snapshots)

            # 步驟 8: 更新輸出狀態
            output.neo4j_categories_created = output.structure_import_result.categories_created > 0
            output.duckdb_d_norm_created = output.database_init_result.duckdb_initialized

            # 步驟 9: 驗證 Definition of Done
            dod_validation = self.validate_definition_of_done(output)
            output.taxonomy_structure_browsable = self._check_taxonomy_browsable()
            output.d_norm_sample_mappable = self._check_d_norm_mappable(input_data.d_raw_data[:5])  # 檢查前5個樣本

            # 步驟 10: 完成
            output.workflow_result.status = WorkflowStatus.SUCCESS
            output.workflow_result.message = "W1 初始化完成"
            output.workflow_result.execution_time_ms = int(datetime.now().timestamp() * 1000)

            logger.info("W1 初始化流程執行成功")
            return output

        except Exception as e:
            logger.error("W1 執行失敗: %s", e)
            self._handle_failure(output, f"執行異常: {str(e)}")
            return output

    # # 私有輔助方法
    # def _check_min_category_support(self, input_data: W1InitInput) -> bool:
    #     """檢查最小分類支援閾值"""
    #     # TODO: 實作分類支援檢查邏輯
    #     return len(input_data.d_raw_data) >= input_data.config.min_category_support

    def _load_initial_categories(self, domain: Domain) -> List[Category]:
        """載入初始分類結構"""
        init_file = Path(f"data/init/{domain.value}_t_init.json")
        if not init_file.exists():
            logger.warning("初始分類檔案不存在: %s", init_file)
            return []

        with open(init_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return [Category(**cat_data) for cat_data in data.get('categories', [])]

    def _load_initial_attributes(self, domain: Domain) -> List[Attribute]:
        """載入初始屬性結構"""
        init_file = Path(f"data/init/{domain.value}_n_init.json")
        if not init_file.exists():
            logger.warning("初始屬性檔案不存在: %s", init_file)
            return []

        with open(init_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return [Attribute(**attr_data) for attr_data in data.get('attributes', [])]

    def _process_category_hints(self, hints: List[CategoryHint]) -> int:
        """處理分類提示"""
        # TODO: 實作分類提示處理邏輯
        return len(hints)

    def _all_pre_checks_passed(self, pre_check_result: W1PreCheckResult) -> bool:
        """檢查所有前置檢查是否通過"""
        return (pre_check_result.batch_integrity_ok and
                pre_check_result.lang_source_valid and
                pre_check_result.version_compatibility)

    def _database_init_success(self, db_result: W1DatabaseInitResult) -> bool:
        """檢查資料庫初始化是否成功"""
        return db_result.duckdb_initialized and db_result.neo4j_initialized

    def _check_taxonomy_browsable(self) -> bool:
        """檢查分類結構是否可瀏覽"""
        # TODO: 實作分類結構瀏覽檢查
        return self.db_manager.check_taxonomy_structure()

    def _check_d_norm_mappable(self, sample_data: List[ItemRaw]) -> bool:
        """檢查 d_norm 是否可映射樣本"""
        # TODO: 實作樣本映射檢查
        return len(sample_data) > 0

    def _handle_failure(self, output: W1InitOutput, message: str):
        """處理失敗情況"""
        output.workflow_result.status = WorkflowStatus.FAILED
        output.workflow_result.message = message
        output.workflow_result.execution_time_ms = int(datetime.now().timestamp() * 1000)
        
        # 嘗試回滾
        if self.snapshots_created:
            last_snapshot = self.snapshots_created[-1]
            if self.rollback(last_snapshot):
                logger.info("成功回滾到快照: %s", last_snapshot)
            else:
                logger.error("回滾失敗: %s", last_snapshot)


# 工廠函數
def create_w1_processor(
    duckdb_path: str, 
    neo4j_config: Dict[str, Any]
) -> W1InitProcessorImpl:
    """建立 W1 處理器實例"""
    db_manager = W1DatabaseManager(duckdb_path, neo4j_config)
    validator = W1Validator()
    return W1InitProcessorImpl(db_manager, validator)
