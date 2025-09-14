# Workflow W5 — 維護 (Governance & Maintenance)

## Purpose

* 人工審核來自 W4 的提案
* 將決策落實到 **T/N 結構**（分類 / 屬性 / 值）
* 同步更新 **值映射 (value\_mapping)** 與 **治理事件 (g\_events)**
* 確保版本點一致性

---

## Inputs

* `s_proposals` — 待審提案
* Neo4j 當前狀態 (T/N 結構)
* `value_mapping`

---

## Outputs

* Neo4j (T/N 更新後狀態)
* 更新後的 `value_mapping`
* `g_events` — 對應治理事件

---

## Side Effects

* `ATTR_MERGE`
* `ATTR_SPLIT`
* `VALUE_DEPRECATE`
* `MAPPING_UPDATE`
* `SNAPSHOT_CREATE`

---

## Steps

1. **提案審核**

   * 從 `s_proposals` 選擇接受 / 拒絕
   * 檢查衝突與影響範圍

2. **執行決策**

   * 對 Neo4j 進行合併 / 拆分 / 新增 / 淘汰
   * 更新 `value_mapping`

3. **記錄事件**

   * 建立治理事件 (`g_events`)
   * 每次維護作業需綁定 `event_id`

4. **產生快照**

   * 變更前後各一版本點
   * 確保一致性與可回溯性

---

## Pre-checks

* 影響面 diff（確認受影響範圍）
* 衝突檢查（避免版本衝突 / 資料覆蓋）

---

## Idempotency

* 每次維護作業需綁定唯一 `event_id`

---

## Rollback

* 變更前後各建立一個版本點快照

---

## Config Knobs

* `dry_run` — 僅模擬，不實際寫入
* `auto_alias_from_values` — 是否自動將值作為 alias

---

## Definition of Done (DoD)

* 圖 (Neo4j) / 映射 (DuckDB.value\_mapping) / 版本點一致
* 對應的回填計畫產出（由 W2 執行）

---

## Out-of-Scope

* 回填操作本身（交由 W2）

---