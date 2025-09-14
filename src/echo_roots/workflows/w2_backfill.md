# Workflow W2 — 回填 (Backfill Normalization)

## Purpose

* 當 **T/N 有變更** 時，對既有 Domain Data 進行 **增量正規化**
* 保證歷史資料與最新 T/N 結構一致，並記錄治理事件

---

## Inputs

* 舊版 `d_norm_{domain}`
* `value_mapping`
* **版本差異集**（T/N 改動摘要）

---

## Outputs

* 新版 `d_norm_{domain}`（`norm_version` 已更新）

---

## Side Effects

* `BACKFILL_COMMIT`
* `SNAPSHOT_CREATE (D)`

---

## Steps

1. **選批**

   * 以 `data/{domain}/d_raw/dt=.../batch_id=...` 為單位

2. **解析屬性 (attrs)**

   * 對 keyword 類型，優先嘗試 **N 值匹配**（alias/labels）

3. **分類掛載**

   * 依 `raw_category_path`
   * 使用 heuristics 或既有映射

4. **寫入結果**

   * 成功 → 寫入 `d_norm`
   * 失敗 → 寫入 `d_outlier`

5. **治理事件**

   * 產出 `g_events: snapshot / diff`（可選）

---

## Pre-checks

* 差異集正確性檢查
* 匯總對帳

---

## Idempotency

* 同版本重跑 → 結果應一致

---

## Rollback

* 回填前建立 `pre_commit_snapshot`

---

## Config Knobs

* `batch_size` — 單次處理大小
* `stop_on_diff_gt` — 允許誤差上限，超過即中止

---

## Definition of Done (DoD)

* 指標穩定
* 抽查通過
* 事件記錄完整

---

## Out-of-Scope

* 全量重算

---
