"""W1 API 接口 - 對外服務"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import ValidationError

from src.echo_roots.workflows.contracts.w1_contracts import W1InitInput, W1InitOutput
from src.echo_roots.workflows.contracts.common import WorkflowStatus
from src.echo_roots.workflows.w1.w1_main import create_w1_processor
from src.echo_roots.workflows.w1.w1_main import W1InitProcessorImpl

logger = logging.getLogger(__name__)

# 建立 FastAPI 路由器
router = APIRouter(prefix="/workflows/w1", tags=["W1 - 初始化"])


# 設定依賴注入（簡化版）
def get_w1_processor() -> W1InitProcessorImpl:
    """取得 W1 處理器實例"""
    # TODO: 從設定檔讀取資料庫設定
    duckdb_path = "data/echoroots.duckdb"
    neo4j_config = {
        'uri': "bolt://localhost:7687",
        'username': "neo4j",
        'password': "password"
    }
    return create_w1_processor(duckdb_path, neo4j_config)


@router.post("/init", response_model=W1InitOutput)
async def w1_init(
    request: W1InitInput,
    processor: W1InitProcessorImpl = Depends(get_w1_processor)
) -> W1InitOutput:
    """
    W1 初始化 API 端點
    
    執行 W1 初始化工作流：
    1. 初始化資料庫
    2. 匯入初始結構
    3. 建立基礎 value_mapping
    4. 建立快照
    """
    try:
        logger.info("收到 W1 初始化請求，批次: %s", request.batch_metadata.batch_key)

        # 執行 W1 處理流程
        result = processor.process(request)
        
        # 檢查執行結果
        if result.workflow_result.status == WorkflowStatus.FAILED:
            raise HTTPException(
                status_code=500,
                detail=f"W1 執行失敗: {result.workflow_result.message}"
            )

        logger.info("W1 初始化完成，批次: %s", request.batch_metadata.batch_key)
        return result

    except ValidationError as e:
        logger.error("W1 請求驗證失敗: %s", e)
        raise HTTPException(status_code=400, detail="請求格式錯誤: %s" % e)

    except Exception as e:
        logger.error("W1 執行異常: %s", e)
        raise HTTPException(status_code=500, detail="內部伺服器錯誤: %s" % str(e))

@router.get("/status/{batch_key}")
async def get_w1_status(
    batch_key: str,
    processor: W1InitProcessorImpl = Depends(get_w1_processor)
) -> Dict[str, Any]:
    """
    查詢 W1 批次執行狀態
    """
    try:
        # TODO: 實作狀態查詢邏輯
        # 目前簡化版，直接檢查資料庫

        return {
            "batch_key": batch_key,
            "status": "unknown",
            "message": "狀態查詢功能尚未實作"
        }

    except Exception as e:
        logger.error("查詢 W1 狀態失敗: %s", e)
        raise HTTPException(status_code=500, detail="查詢失敗: %s" % str(e)) from e


@router.post("/rollback/{snapshot_id}")
async def w1_rollback(
    snapshot_id: str,
    processor: W1InitProcessorImpl = Depends(get_w1_processor)
) -> Dict[str, Any]:
    """
    回滾 W1 到指定快照
    """
    try:
        logger.warning("開始 W1 回滾操作，快照: %s", snapshot_id)

        success = processor.rollback(snapshot_id)

        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"回滾失敗: {snapshot_id}"
            )

        return {
            "snapshot_id": snapshot_id,
            "status": "success",
            "message": "回滾完成"
        }

    except Exception as e:
        logger.error("W1 回滾失敗: %s", e)
        raise HTTPException(status_code=500, detail="回滾失敗: %s" % str(e)) from e


# 健康檢查端點
@router.get("/health")
async def w1_health_check() -> Dict[str, str]:
    """W1 健康檢查"""
    return {
        "service": "W1 初始化服務",
        "status": "healthy",
        "version": "1.0.0"
    }
