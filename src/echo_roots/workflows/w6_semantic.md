# Workflow W6 — 語意治理 (Semantic Governance)

## Purpose

* 在 **單一分類節點** 範圍內，將 STAGING 觀測進行去重與聚類
* 產生 **HOT clusters** 與 **上浮提案**
* 支援屬性/值的治理決策，並可回寫至 `d_norm`

---

## Inputs

* `s_staging_{domain}` — 指定分類相關的 STAGING 樣本

---

## Outputs

* `s_hot_{domain}` — 聚類後的候選群
* `s_proposals` — 提案集合（DuckDB 表或檔案）

---

## Side Effects

* `CANDIDATE_MERGE`
* `PROPOSAL_CREATE`

---

## Steps

1. **取樣本集合**

   * 選取同 `domain / category_id / attr_hint` 的 STAGING 觀測

2. **聚類處理**

   * 使用近鄰圖 + 門檻法進行聚類
   * 產生 `SCluster`：

     * `rep_term`（代表詞）
     * `aliases`（同義詞集合）
     * `support`（支持數量）

3. **更新觀測狀態**

   * 將觀測 payload 寫入 `cluster_id`
   * 更新 `state: HOT`

4. **產生提案**

   * 提取 `propose_attr_id`、`propose_value`
   * 寫入 `s_proposals`
   * 建立治理事件 `PROPOSAL_CREATE`

5. **回寫 (選擇性)**

   * 當新值被治理並映射後，可回填至 `d_norm.attrs`

---

## Pre-checks

* 節點存在檢查
* 樣本量達最低門檻

---

## Idempotency

* 以 **(category\_id + 視窗期)** 為鍵，避免重複合併

---

## Rollback

* 在提案產生點建立快照

---

## Config Knobs

* `min_support` — 最小支持度
* `purity_threshold` — 最小純度閾值

---

## Definition of Done (DoD)

* HOT clusters 維持乾淨
* 提案可讀且格式正確

---

## Out-of-Scope

* 全網級聚類（僅針對單節點局部）

---