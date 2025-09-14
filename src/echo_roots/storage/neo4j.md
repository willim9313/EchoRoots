# Neo4j Storage Contract

## Purpose

* 承載 **Taxonomy (Category)**、**Normalized Attributes (Attribute/Value)** 與其關係
* 提供 **樹 / 網路導向的操作**（路徑、繼承、映射）
* 不直接存放 `Item`；`d_norm` 以 `category_id`、`attrs.*.value_id` 指回圖上的 ID

---

## Data Mapping

### Nodes

* `Category`
* `Attribute`
* `Value`

### Relationships

* `CHILD_OF` — 類別樹階層
* `HAS_ATTR`（或 `BINDS_ATTR`）— 類別綁定屬性
* `ALLOWS_VALUE`（或 `HAS_VALUE`）— 屬性允許值
* `ALIAS_OF`（選擇性）— 值之間的同義關係
* `MAPS_TO` — 值映射（舊值 → 新值）

---

## Required Capabilities

### Read

* **路徑查詢**：取得分類完整路徑與子樹
* **子樹統計**：計算某分類下的節點/屬性覆蓋
* **繼承檢視**：檢查屬性在父子分類間的繼承/覆蓋情況
* **值映射查詢**：由 `from_value_id` 找到 `to_value_id`

### Write

* 建立 / 更新 Category、Attribute、Value 節點
* 節點移動、合併、標記狀態 (`active|deprecated|merged`)
* 維護關係（`CHILD_OF`, `HAS_ATTR`, `HAS_VALUE`, `MAPS_TO`）

### Snapshot

* 以 `event_id` 為批次鍵
* 可導出 **CSV / Parquet / GraphML** 作為版本快照
* 快照可用於回滾或審計

---

## Idempotency & Consistency

* **event\_id 為唯一鍵**：所有寫入以 `event_id` 標記
* **冪等性**：同一個 `event_id` 重放，不應造成重複變更
* **一致性**：所有節點與關係更新需附帶 `event_id`

---

## Failure & Rollback

* 若變更失敗 → 回退至最近快照
* 所有變更皆需關聯 `event_id`，以支援審計與回滾

---
