# Workflow W1 — 初始化 (Init T/N)

## Purpose

* 從 **Domain Data (D\_raw)** 建立最小的 **分類樹 (T)** 與 **屬性/值初稿 (N)**
* 作為系統的 **T/N 初始版本**，後續工作流 (W2\~W6) 的基礎

---

## Inputs

* `d_raw_{domain}`
* 可用的 `category_hint`

---

## Outputs

* **Neo4j**: `Category` 節點
* **DuckDB**: 空或極簡的 `d_norm_{domain}`
* **Side Effects**: `SNAPSHOT_CREATE (T,N)`

---

## Steps

1. **初始化資料庫**

   * DuckDB：執行 `duckdb/init/00_init.sql`
   * Neo4j：執行 `neo4j/init/00_init.cypher`

2. **匯入初始結構**

   * 匯入初始 `Category/Attribute/Value`
   * 來源可為 `data/{domain}/t_init.json`, `n_init.json`

3. **（可選）建立初稿 value\_mapping**

   * `value_mapping` 表可填入基礎映射

---

## Pre-checks

* `d_raw` 批次完整性檢查
* `lang` 與 `source` 值是否合法

---

## Idempotency

* 以 `event_id + 批次鍵` 防止重複寫入

---

## Rollback

* 在 **寫入 Neo4j 前後** 各建立一個快照

---

## Config Knobs

* `min_category_support` — 最小分類支援閾值

---

## Definition of Done (DoD)

* 可瀏覽的 **根 → 葉** 分類結構已建立
* `d_norm` 可對應上少量樣本

---

## Out-of-Scope

* 屬性聚合演算法
* 值治理

---

