"""
W1 資料庫管理器 - 處理 DuckDB 和 Neo4j 操作
note: 目前db初始化路徑直接寫死於此
"""
import logging
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime
from uuid import UUID

import duckdb
from neo4j import GraphDatabase, Query

from src.echo_roots.models import Category, Attribute, AttributeValue

logger = logging.getLogger(__name__)


class DatabaseInitializationError(Exception):
    """資料庫初始化異常"""
    pass


class W1DatabaseManager:
    """W1 資料庫管理器 - 簡化版"""

    def __init__(self, duckdb_path: str, neo4j_config: Dict[str, Any]):
        self.duckdb_path = duckdb_path
        self.neo4j_config = neo4j_config
        self._neo4j_driver = None

    @property
    def neo4j_driver(self):
        """延遲初始化 Neo4j 驅動"""
        if self._neo4j_driver is None:
            self._neo4j_driver = GraphDatabase.driver(
                self.neo4j_config['uri'],
                auth=(self.neo4j_config['username'], self.neo4j_config['password'])
            )
        return self._neo4j_driver

    def init_duckdb(self) -> bool:
        """初始化 DuckDB - 執行初始化腳本"""
        try:
            # 讀取初始化 SQL
            init_sql_path = Path("scripts/duckdb/init/00_init.sql")
            if not init_sql_path.exists():
                logger.error("DuckDB 初始化腳本不存在: %s", init_sql_path)
                return False

            with open(init_sql_path, 'r', encoding='utf-8') as f:
                init_sql = f.read()

            # 執行初始化
            with duckdb.connect(self.duckdb_path) as conn:
                conn.execute(init_sql)
                logger.info("DuckDB 初始化完成")
                return True

        except PermissionError as e:
            logger.error("檔案權限不足: %s", e)
            return False
        except UnicodeDecodeError as e:
            logger.error("SQL 檔案編碼錯誤: %s", e)
            return False
        except duckdb.Error as e:
            logger.error("DuckDB 執行錯誤: %s", e)
            return False
        except OSError as e:
            logger.error("系統 I/O 錯誤: %s", e)
            return False
        except DatabaseInitializationError as e:
            # 只在最後保留一個通用異常處理，用於處理未預期的錯誤
            logger.error("DuckDB 初始化發生未預期錯誤: %s", e, exc_info=True)
            return False

    def init_neo4j(self) -> bool:
        """初始化 Neo4j - 執行初始化腳本"""
        try:
            # 讀取初始化 Cypher
            init_cypher_path = Path("scripts/neo4j/init/00_init.cypher")
            if not init_cypher_path.exists():
                logger.error("Neo4j 初始化腳本不存在: %s", init_cypher_path)
                return False

            with open(init_cypher_path, 'r', encoding='utf-8') as f:
                init_cypher = f.read()

            # 執行初始化
            with self.neo4j_driver.session() as session:
                session.run(init_cypher)
                logger.info("Neo4j 初始化完成")
                return True
        except PermissionError as e:
            logger.error("檔案權限不足: %s", e)
            return False
        except UnicodeDecodeError as e:
            logger.error("SQL 檔案編碼錯誤: %s", e)
            return False
        except duckdb.Error as e:
            logger.error("DuckDB 執行錯誤: %s", e)
            return False
        except OSError as e:
            logger.error("系統 I/O 錯誤: %s", e)
            return False
        except DatabaseInitializationError as e:
            logger.error("Neo4j 初始化失敗: %s", e)
            return False

    def import_categories(self, categories: List[Category]) -> int:
        """匯入分類到 Neo4j"""
        if not categories:
            return 0

        try:
            with self.neo4j_driver.session() as session:
                imported_count = 0

                for category in categories:
                    # 建立分類節點
                    query = """
                    MERGE (c:Category {category_id: $category_id})
                    SET c.domain = $domain,
                        c.name = $name,
                        c.labels = $labels,
                        c.parent_id = $parent_id,
                        c.path = $path,
                        c.path_ids = $path_ids,
                        c.status = $status,
                        c.t_version = $t_version,
                        c.updated_at = datetime()
                    """

                    session.run(query, {
                        'category_id': str(category.category_id),
                        'domain': category.domain.value,
                        'name': category.name,
                        'labels': json.dumps(category.labels),
                        'parent_id': (str(category.parent_id) 
                                    if category.parent_id else None),
                        'path': category.path,
                        'path_ids': [str(pid) for pid in category.path_ids],
                        'status': category.status.value,
                        't_version': category.t_version
                    })

                    imported_count += 1

                # 建立父子關係
                for category in categories:
                    if category.parent_id:
                        relationship_query = """
                        MATCH (child:Category {category_id: $child_id})
                        MATCH (parent:Category {category_id: $parent_id})
                        MERGE (parent)-[:HAS_CHILD]->(child)
                        """

                        session.run(relationship_query, {
                            'child_id': str(category.category_id),
                            'parent_id': str(category.parent_id)
                        })

                logger.info("成功匯入 %s 個分類", imported_count)
                return imported_count

        except Exception as e:
            logger.error("分類匯入失敗: %s", e)
            return 0

    def import_attributes(self, attributes: List[Attribute]) -> int:
        """匯入屬性到 Neo4j"""
        if not attributes:
            return 0

        try:
            with self.neo4j_driver.session() as session:
                imported_count = 0

                for attribute in attributes:
                    # 建立屬性節點
                    query = """
                    MERGE (a:Attribute {attr_id: $attr_id})
                    SET a.domain = $domain,
                        a.name = $name,
                        a.labels = $labels,
                        a.type = $type,
                        a.description = $description,
                        a.constraints = $constraints,
                        a.scope = $scope,
                        a.status = $status,
                        a.n_version = $n_version,
                        a.updated_at = datetime()
                    """

                    session.run(query, {
                        'attr_id': str(attribute.attr_id),
                        'domain': attribute.domain.value,
                        'name': attribute.name,
                        'labels': json.dumps(attribute.labels),
                        'type': attribute.type.value,
                        'description': attribute.description,
                        'constraints': json.dumps(attribute.constraints),
                        'scope': str(attribute.scope) if isinstance(attribute.scope, UUID) else attribute.scope,
                        'status': attribute.status.value,
                        'n_version': attribute.n_version
                    })

                    imported_count += 1

                logger.info("成功匯入 %s 個屬性", imported_count)
                return imported_count

        except Exception as e:
            logger.error("屬性匯入失敗: %s", e)
            return 0

    def check_batch_processed(self, idempotency_key: str) -> bool:
        """檢查批次是否已處理過"""
        try:
            with duckdb.connect(self.duckdb_path) as conn:
                result = conn.execute(
                    "SELECT COUNT(*) FROM processed_batches WHERE idempotency_key = ?",
                    (idempotency_key,)
                ).fetchone()

                return result[0] > 0 if result else False

        except Exception as e:
            logger.error("檢查批次處理狀態失敗: %s", e)
            return False

    def mark_batch_processed(self, idempotency_key: str) -> bool:
        """標記批次為已處理"""
        try:
            with duckdb.connect(self.duckdb_path) as conn:
                conn.execute(
                    "INSERT INTO processed_batches (idempotency_key, processed_at) VALUES (?, ?)",
                    (idempotency_key, datetime.now())
                )
                return True

        except Exception as e:
            logger.error("標記批次處理狀態失敗: %s", e)
            return False

    def create_neo4j_snapshot(self, snapshot_id: str) -> bool:
        """建立 Neo4j 快照"""
        try:
            # 簡化版：記錄快照資訊
            with self.neo4j_driver.session() as session:
                query = """
                CREATE (s:Snapshot {
                    snapshot_id: $snapshot_id,
                    type: 'neo4j',
                    created_at: datetime(),
                    status: 'active'
                })
                """
                session.run(query, {'snapshot_id': snapshot_id})
                logger.info("Neo4j 快照建立完成: %s", snapshot_id)
                return True

        except Exception as e:
            logger.error("Neo4j 快照建立失敗: %s", e)
            return False

    def create_duckdb_snapshot(self, snapshot_id: str) -> bool:
        """建立 DuckDB 快照"""
        try:
            # 簡化版：複製資料庫檔案
            import shutil
            snapshot_path = f"{self.duckdb_path}.{snapshot_id}.backup"
            shutil.copy2(self.duckdb_path, snapshot_path)

            logger.info("DuckDB 快照建立完成: %s", snapshot_path)
            return True

        except Exception as e:
            logger.error("DuckDB 快照建立失敗: %s", e)
            return False

    def restore_neo4j_snapshot(self, snapshot_id: str) -> bool:
        """還原 Neo4j 快照"""
        try:
            # TODO: 實作 Neo4j 快照還原邏輯
            logger.warning("Neo4j 快照還原功能尚未實作: %s", snapshot_id)
            return True

        except Exception as e:
            logger.error("Neo4j 快照還原失敗: %s", e)
            return False

    def restore_duckdb_snapshot(self, snapshot_id: str) -> bool:
        """還原 DuckDB 快照"""
        try:
            import shutil
            snapshot_path = f"{self.duckdb_path}.{snapshot_id}.backup"
            if Path(snapshot_path).exists():
                shutil.copy2(snapshot_path, self.duckdb_path)
                logger.info("DuckDB 快照還原完成: %s", snapshot_id)
                return True
            else:
                logger.error("快照檔案不存在: %s", snapshot_path)
                return False

        except Exception as e:
            logger.error("DuckDB 快照還原失敗: %s", e)
            return False

    def check_taxonomy_structure(self) -> bool:
        """檢查分類結構是否可瀏覽"""
        try:
            with self.neo4j_driver.session() as session:
                # 檢查是否有根節點
                root_check = session.run(
                    "MATCH (c:Category) "
                    "WHERE c.parent_id IS NULL "
                    "RETURN count(c) as root_count"
                ).single()

                if not root_check or root_check['root_count'] == 0:
                    return False

                # 檢查是否有完整的路徑結構
                path_check = session.run(
                    "MATCH (c:Category) "
                    "WHERE size(c.path) > 0 "
                    "RETURN count(c) as path_count"
                ).single()

                return path_check and path_check['path_count'] > 0

        except Exception as e:
            logger.error("檢查分類結構失敗: %s", e)
            return False

    def close(self):
        """關閉資料庫連線"""
        if self._neo4j_driver:
            self._neo4j_driver.close()
