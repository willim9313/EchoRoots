# Workflow W3 — 新流入 (Ingest)

## Purpose

* 接收新批資料
* 能掛 T 的掛入分類；能對 N 的對應屬性值；其餘進 **OD** 或 **S-STAGING**
* 保證每筆記錄都有一個落腳點

---

## Inputs

* `d_raw_{domain}` 新批（來源於分區路徑）

---

## Outputs

* `d_norm_{domain}`（部分可為空）
* `od_inbox`
* `s_staging_{domain}`

---

## Side Effects

* `INGEST_COMMIT`
* 必要時：`SNAPSHOT_CREATE (D)`

---

## Steps

1. **驗證輸入**

   * 檢查 `domain`、`lang`、`raw_category_path`
   * 確認 `norm_version == n_version`

2. **寫入 D\_raw**

   * 批次落地到 DuckDB 分區

3. **語意觀測 (S\_staging)**

   * 從 `title / description / specs / text_uri` 產生多筆 `SStagingObs`
   * 寫入 Qdrant points（payload + 向量）

4. **規則路由**

   * 能對應到 T/N → 寫入 `d_norm`
   * 無法對應 → 寫入 `od_inbox`
   * 殘餘屬性值 → 寫入 `s_staging_{domain}`

---

## Pre-checks

* 版本對齊：`norm_version == n_version`

---

## Idempotency

* 以 **批次鍵 + event\_id** 作為冪等控制

---

## Rollback

* 在寫 `d_norm` 前後各建立一個快照

---

## Config Knobs

* `route_to_s_threshold` — 決定何時進 S-STAGING
* `allow_partial_norm` — 是否允許部分正規化

---

## Definition of Done (DoD)

* 每筆新資料都有路徑落點：

  * T
  * N
  * OD
  * S
* 四者必居其一

---

## Out-of-Scope

* 聚類 (clustering)
* 上浮提案 (proposal)

---
