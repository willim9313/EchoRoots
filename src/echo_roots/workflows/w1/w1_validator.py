"""W1 驗證器 - 資料驗證功能"""

import logging
from typing import List, Set, Dict, Any
import re

from src.echo_roots.models import ItemRaw
from src.echo_roots.workflows.contracts.common import (
    ValidationResult,
    BatchMetadata,
    Language
)

logger = logging.getLogger(__name__)

class W1Validator:
    """W1 驗證器 - 簡化版"""
    
    # 支援的語言代碼
    SUPPORTED_LANGUAGES = {lang.value for lang in Language}
    
    # 有效的來源格式
    VALID_SOURCE_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
    
    def validate_d_raw_integrity(self, d_raw_data: List[ItemRaw]) -> ValidationResult:
        """驗證 d_raw 批次完整性"""
        errors = []
        warnings = []
        
        if not d_raw_data:
            errors.append("d_raw_data 不能為空")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        # 檢查必要欄位
        seen_item_ids = set()
        
        for i, item in enumerate(d_raw_data):
            # 檢查重複的 item_id
            if item.item_id in seen_item_ids:
                errors.append(f"重複的 item_id: {item.item_id} (位置: {i})")
            seen_item_ids.add(item.item_id)
            
            # 檢查必要欄位
            if not item.title or not item.title.strip():
                errors.append(f"item_id {item.item_id}: title 不能為空")
            
            if not item.domain:
                errors.append(f"item_id {item.item_id}: domain 不能為空")
            
            # 檢查 specs 格式
            if item.specs and not isinstance(item.specs, dict):
                errors.append(f"item_id {item.item_id}: specs 必須是字典格式")
            
            # 檢查 tags 格式
            if item.tags and not isinstance(item.tags, list):
                errors.append(f"item_id {item.item_id}: tags 必須是列表格式")
        
        # 資料量檢查
        if len(d_raw_data) < 10:
            warnings.append(f"資料量較少 ({len(d_raw_data)} 筆)，可能影響分類品質")
        
        logger.info(f"d_raw 驗證完成: {len(d_raw_data)} 筆資料, {len(errors)} 個錯誤, {len(warnings)} 個警告")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_lang_source(self, batch_metadata: BatchMetadata) -> ValidationResult:
        """驗證 lang 與 source 值的合法性"""
        errors = []
        warnings = []
        
        # 檢查語言代碼
        if batch_metadata.lang:
            lang_value = batch_metadata.lang.value if hasattr(batch_metadata.lang, 'value') else str(batch_metadata.lang)
            if lang_value not in self.SUPPORTED_LANGUAGES:
                errors.append(f"不支援的語言代碼: {lang_value}，支援的語言: {', '.join(self.SUPPORTED_LANGUAGES)}")
        else:
            warnings.append("語言代碼未設定，將使用預設值")
        
        # 檢查來源格式
        if batch_metadata.source:
            source_value = batch_metadata.source.value if hasattr(batch_metadata.source, 'value') else str(batch_metadata.source)
            if not self.VALID_SOURCE_PATTERN.match(source_value):
                errors.append(f"來源格式不正確: {source_value}，只允許字母、數字、底線和連字號")
        else:
            errors.append("來源 (source) 不能為空")
        
        # 檢查 domain
        if not batch_metadata.domain:
            errors.append("域 (domain) 不能為空")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_category_hints(self, hints: List[Dict[str, Any]]) -> ValidationResult:
        """驗證分類提示的格式"""
        errors = []
        warnings = []
        
        for i, hint in enumerate(hints):
            if not isinstance(hint, dict):
                errors.append(f"分類提示 {i}: 必須是字典格式")
                continue
            
            # 檢查必要欄位
            if 'path' not in hint:
                errors.append(f"分類提示 {i}: 缺少 path 欄位")
            elif not isinstance(hint['path'], list):
                errors.append(f"分類提示 {i}: path 必須是列表格式")
            elif len(hint['path']) == 0:
                errors.append(f"分類提示 {i}: path 不能為空")
            
            if 'support_count' in hint and not isinstance(hint['support_count'], int):
                errors.append(f"分類提示 {i}: support_count 必須是整數")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_min_category_support(self, data_count: int, min_support: int) -> ValidationResult:
        """驗證最小分類支援閾值"""
        errors = []
        warnings = []
        
        if data_count < min_support:
            errors.append(f"資料量 ({data_count}) 低於最小分類支援閾值 ({min_support})")
        elif data_count < min_support * 2:
            warnings.append(f"資料量 ({data_count}) 接近最小分類支援閾值 ({min_support})，建議增加更多資料")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_initial_structure_files(self, domain: str) -> ValidationResult:
        """驗證初始結構檔案是否存在且格式正確"""
        errors = []
        warnings = []
        
        from pathlib import Path
        import json
        
        # 檢查 t_init.json
        t_init_path = Path(f"data/init/{domain}_t_init.json")
        if not t_init_path.exists():
            warnings.append(f"初始分類檔案不存在: {t_init_path}")
        else:
            try:
                with open(t_init_path, 'r', encoding='utf-8') as f:
                    t_data = json.load(f)
                if 'categories' not in t_data:
                    errors.append(f"{t_init_path}: 缺少 categories 欄位")
            except json.JSONDecodeError as e:
                errors.append(f"{t_init_path}: JSON 格式錯誤 - {e}")
        
        # 檢查 n_init.json
        n_init_path = Path(f"data/init/{domain}_n_init.json")
        if not n_init_path.exists():
            warnings.append(f"初始屬性檔案不存在: {n_init_path}")
        else:
            try:
                with open(n_init_path, 'r', encoding='utf-8') as f:
                    n_data = json.load(f)
                if 'attributes' not in n_data:
                    errors.append(f"{n_init_path}: 缺少 attributes 欄位")
            except json.JSONDecodeError as e:
                errors.append(f"{n_init_path}: JSON 格式錯誤 - {e}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )