# DuckDB Storage Contract

## Purpose

DuckDB 作為 **本地分析 / 對帳 / 快照主場**，並同時存放 **治理事件流水 (G-events)**。
在 W1/W2/W3/W6 workflow 中作為主要資料面，承擔 **查詢、轉換、聚合** 與 **版本快照** 的角色。

---

## Tables（最小集合）

* **Domain Data**

  * `d_raw_{domain}`
  * `d_norm_{domain}`
* **Outlier**

  * `od_inbox`（跨 domain）
* **Semantic**

  * `s_staging_obs`
  * `s_clusters`
* **Governance**

  * `g_events`（見 G-event 格式）
  * `impacted_set`
* **Mapping**

  * `value_mapping`（屬性值映射，含 `mapping_version`）

---

## Partition & Files

* 以 **Parquet 分區檔案** 儲存
* 路徑規範：

```
data/{domain}/{table}/dt=YYYY-MM-DD/batch_id={uuid}/*.parquet
```

---

## Ops & Guarantees

### Transaction 策略

* **批次 (batch\_id)** 為單位
* 每次寫入包含：

  1. **pre\_commit\_snapshot**：批次前快照
  2. **commit**：檢查通過後正式寫入
  3. **impacted\_set**：記錄受影響的 item\_ids

---

## 必要操作

* **Upsert**

  * `d_raw`
  * `d_norm`
  * `s_staging_obs`
  * `s_clusters`
  * `g_events`

* **Query**

  * **W2**:

    * 由 `d_raw` 產生 `d_norm`
    * 包含 evidence 聚合
  * **W3**:

    * 新增 `d_raw`
    * 同步寫入 `s_staging_obs`
  * **W6**:

    * 由 `s_staging_obs` 聚合成 `s_clusters`
    * 回寫 cluster 資訊（含狀態 HOT/ARCHIVE）

---

## Role in Workflows

* **W1 初始化**: 輸入 D\_raw，建立初稿 T/N
* **W2 回填**: DuckDB 負責 D\_raw → D\_norm 映射與比對
* **W3 新流入**: 寫入 D\_raw，並將觀測值進入 S-staging
* **W6 語意治理**: 聚合 S-staging → clusters，回寫治理事件

---
