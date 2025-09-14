# ADR-0001: Core Storage & Data Model Foundation

## Status

Accepted
Date: 2025-09-12
Supersedes: *ADR-0001 (Taxonomy Storage Model, deprecated)*

## Context

EchoRoots 系統需要一個統一的基礎資料模型，能夠同時支持：

* **多領域資料**：商品分類、知識庫文本等。
* **多層結構**：分類骨架 (T)、正規化屬性 (N)、語意候選 (S)。
* **多型態儲存**：結構化查詢、圖狀關聯、向量檢索。
* **可演進性**：先支持本地開發 (MVP)，後續能擴充至分散式。

## Decision

建立 **Hybrid Storage + Layered Model** 作為基礎設計：

* **DuckDB**

  * 角色：本地分析、批次處理、快照儲存。
  * 使用場景：測試資料集、臨時 ETL、日誌聚合。
  * 儲存單位：Parquet/JSON（依照 `data/{domain}/{table}/dt=...` 路徑規範）。

* **Neo4j**

  * 角色：階層結構、屬性治理、版本化關聯。
  * 使用場景：A 層 (分類骨架)、N 層 (正規化屬性)、治理操作 (merge/rollback)。
  * 儲存單位：Category、Attribute、Value、Mapping 節點 + 關聯。

* **Qdrant**

  * 角色：語意檢索、模糊比對、候選推薦。
  * 使用場景：S 層（semantic candidates）、embedding-based search。
  * 儲存單位：Collection per domain，schema 記錄於 `qdrant/collections.md`。

* **資料夾規劃**

  ```
  echo-roots/
  ├─ data/         # 測試資料 (Parquet/JSON)
  ├─ duckdb/       # 本地 DuckDB DB + init SQL
  ├─ neo4j/        # Cypher 腳本 (init, migration)
  ├─ qdrant/       # 各 collection 定義
  ├─ src/echo_roots/
  │  ├─ models/    # Pydantic 資料契約
  │  ├─ storage/   # DuckDB/Neo4j/Qdrant interface
  │  └─ workflows/ # Pipeline 定義
  ```

## Consequences

* 優點

  * 輕量、可本地快速開發。
  * 資料依層次對應最合適的儲存模型。
  * 架構彈性，後續可替換 DuckDB → BigQuery，或 Qdrant → Vespa。

* 缺點

  * 多儲存技術並存，需維護統一 API。
  * MVP 初期需要先定義最小契約 (models, interfaces)。

## Alternatives Considered

1. **單一 Relational DB (Postgres/MySQL)**：結構清楚，但不擅長 graph traversal 與 semantic search。
2. **單一 Document DB (MongoDB)**：靈活，但難以處理 taxonomy 與屬性治理。
3. **純 Graph DB (Neo4j)**：適合階層，但無法取代向量檢索或高效批次分析。
