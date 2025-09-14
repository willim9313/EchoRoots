# Workflow W4 — 上浮提案 (Proposal Creation)

## Purpose

* 將 **S-HOT** 中已經穩定的候選，轉換為 **可審核的提案**
* 提案類型可能涉及 **屬性 / 值 / 分類** 的新增或合併
* 作為 **治理決策 (W5)** 的前置輸入

---

## Inputs

* `s_hot_{domain}` — 已聚類且狀態為 HOT 的候選

---

## Outputs

* `s_proposals` — 提案集合，內含：

  * 提案內容（屬性 / 值 / 分類）
  * 影響面清單 (impacted set)

---

## Side Effects

* `PROPOSAL_CREATE` — 治理事件

---

## Steps

1. 從 **S-HOT clusters** 篩選候選
2. 檢查支援度 / 純度門檻
3. 轉換為提案格式：

   * `cluster_id`
   * `propose_attr_id / propose_value`
   * `support / aliases / impacted_set`
4. 寫入 `s_proposals`
5. 產生治理事件：`PROPOSAL_CREATE`

---

## Pre-checks

* 支援度門檻 (min\_support)
* 純度門檻 (min\_uniqueness)

---

## Idempotency

* 同一候選群不得重複產生提案

---

## Rollback

* 提案檔案快照

---

## Config Knobs

* `min_support` — 最小支援度
* `min_uniqueness` — 最小純度/唯一性

---

## Definition of Done (DoD)

* 提案集合完整
* 格式正確，可由 **W5 直接執行**

---

## Out-of-Scope

* 自動接受提案
