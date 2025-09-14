# Qdrant Storage Contract

## Purpose

* 作為 **S 層觀測與語意聚類** 的主要存放地
* 支援 **狀態治理**：`STAGING → HOT → ARCHIVE`
* 提供近鄰檢索與聚類能力
* **不承擔權威事實**；最終決策寫回由 **W4/W5** 負責

---

## Collections（最小集合）

* `s_staging_{domain}`
* `s_hot_{domain}`
* （可擴充 `s_archive_{domain}` 作長期保存）

---

## Vectors & Payload

每筆觀測點 (point) 至少包含：

* `embedding_model: str` — 向量模型標識
* `vector: List[float]` — 嵌入向量
* `text: str` — 原始文字
* `lang: str` — 語言代碼
* `record_refs: List[UUID]` — 對應的 item/record 參考
* `support_count: int` — 支援度（聚類後累積）
* `purity: float` — 純度指標

---

## 必要操作

### Upsert

* `s_values_{domain}`（points 寫入或更新）

### Query

* 近鄰檢索：依

  * `value_text`
  * `attr_hint`
  * `category_id`

### Clustering（W6）

1. 取同一 `domain/category_id/attr_hint` 子集合
2. 使用近鄰圖 / 密度法聚類
3. 產生 `SCluster`：

   * `rep_term`（代表詞）
   * `aliases`（同義詞集合）
   * `support`（支持數量）
4. 形成 **上浮提案**（可寫入 `g_events`）

---

## Ops Summary

* **Dedup**：文字同義去重
* **Attach**：候選值附掛至既有節點
* **Stats**：支援度統計
* **Propose**：形成上浮提案（對應治理事件）

---

## Boundaries

* Qdrant 僅存放 **候選 / 聚類結果**
* **不作為權威事實庫**
* 最終寫回（T/N 更新）由 **治理流程 W4/W5** 負責

---
