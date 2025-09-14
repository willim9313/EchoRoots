# Governance Events (G Layer)

## Purpose

所有會改變 **T / N / D / S** 狀態或版本的動作，都需記錄為事件，用於 **審計、回溯與回滾**。

---

## Event Schema（統一欄位）

```json
{
  "event_id": "uuid",                        // 事件 ID
  "event_type": "string",                    // 事件類型
  "actor": "user|process",                   // 觸發人或流程
  "created_at": "timestamp",
  "target": {
    "layer": "T|N|D|S|G",                    // 目標層
    "id": "*",                               // 目標 ID
    "scope": "optional"                      // 範圍（如 item_set, domain）
  },
  "before_version": "string|null",           // 事件前版本
  "after_version": "string|null",            // 事件後版本
  "payload": {},                             // 差異或規則
  "notes": "string|null",                    // 備註
  "related_event_ids": ["uuid", "..."]       // 相關事件（可選）
}
```

---

## Event Types（最小集合）

### T (Taxonomy)

* `CATEGORY_CREATE`
* `CATEGORY_RENAME`
* `CATEGORY_MOVE`
* `CATEGORY_MERGE`
* `CATEGORY_DEPRECATE`

### N (Normalized Attributes)

* `ATTR_CREATE`
* `ATTR_MERGE`
* `ATTR_SPLIT`
* `VALUE_ADD`
* `VALUE_MERGE`
* `VALUE_DEPRECATE`

### D (Domain Data)

* `BACKFILL_COMMIT`
* `INGEST_COMMIT`

### S (Semantic Layer)

* `CANDIDATE_ATTACH`
* `CANDIDATE_MERGE`
* `PROPOSAL_CREATE`

### 共用

* `SNAPSHOT_CREATE`
* `ROLLBACK_APPLY`
* `MAPPING_UPDATE`

---

## Rollback 原則

* 僅 **版本點 (version tag)** 支援完整回滾
* 非版本點事件需以 **逆操作** 或 **重算** 處理

---

## Version Tag 命名

* `T-<major>.<minor>.<patch>`
* `N-<major>.<minor>.<patch>`

  * 初期可僅使用 `major`

---

## 補充建議

* 事件應可掛鉤 **batch\_id**，並記錄 `impacted_set.item_ids`
* `payload` 中應包含 **前後影響範圍** 與 **規則版本**（例如 `t_version / n_version / norm_version`）
* 事件類型可對應 **snapshot / diff / rollback** 這三大類別

---