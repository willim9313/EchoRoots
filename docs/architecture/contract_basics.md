**Purpose**
統一欄位命名、版本與審計規範，避免各模組各自為政。

**Field Naming（建議）**

* 主鍵：`*_id`（字串/UUID）
* 版本：`*_version`（整數）
* 時間：`*_at`（ISO8601，UTC 或 +08 明確標註）
* 狀態：`status ∈ {active, deprecated, merged, draft}`

**Audit & Provenance**

* `created_at / updated_at / created_by / updated_by`
* `source`（資料來源，e.g. `official_store`, `crawler_x`）
* `evidence`（可選，原始片段、URL、行號等）

**Versioning 原則**

* **T**（分類樹）改動 → `t_version += 1`
* **N**（正規化層）改動（欄位/值/規則）→ `n_version += 1`
* **D** 正規化結果標記當下 `norm_version = n_version`
* **Mapping**（值映射表）改 → `mapping_version += 1`

**Idempotency**

* 相同輸入、相同版本 → 相同輸出；寫回前產生快照（見治理區段）。