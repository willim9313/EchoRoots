# Taxonomy & Normalized Attributes (T/N Models)

## Purpose

定義 **分類樹 (T)** 與 **正規化屬性/屬性值 (N)** 的最小結構與治理邊界。
適用於 **商品 (product)** 與 **知識庫 (knowledge\_base)** 兩類 domain。

任何 T/N 改動需建立 **治理事件 (G-event)**，並提升對應版本號。

---

## Enumerations

* **Domain**: `"product" | "knowledge_base"`
* **AttrType**: `"text" | "keyword" | "number" | "boolean" | "datetime"`

---

## Entities & Keys

### Category (T)

* `category_id: UUID` — 唯一 ID
* `domain: Domain`
* `name: str`
* `labels: Dict[str, str]` — 多語標籤
* `parent_id: UUID?`
* `path: List[str]` — root → leaf
* `path_ids: List[UUID]`
* `status: "active" | "deprecated"`（預設 `"active"`)
* `bound_attr_ids: List[UUID]` — 綁定到 N 層的屬性
* `t_version: int = 1`
* `created_at, updated_at: datetime`

---

### Attribute (N)

* `attr_id: UUID`
* `domain: Domain`
* `name: str` — 建議 snake\_case 英文
* `labels: Dict[str, str]` — 多語標籤
* `type: AttrType = "keyword"`
* `description: str?`
* `allowed_value_ids: List[UUID]` — keyword 常用
* `constraints: Dict[str, Any]` — 例如 `{"cardinality": "single"}`
* `scope: "global" | category_id` — 指定範圍
* `status: "active" | "deprecated"`
* `n_version: int = 1`
* `created_at, updated_at: datetime`

---

### AttributeValue (N)

* `value_id: UUID`
* `attribute_id: UUID`
* `value: str` (1–200)
* `labels: Dict[str, str]` — 多語標籤，如 `{"en":"Green","zh-TW":"綠色"}`
* `aliases: Set[str]` — 可能出現的別名
* `status: "active" | "deprecated" | "merged"`
* `metadata: Dict[str, Any]`
* `created_at, updated_at: datetime`

---

### ValueMapping

* `mapping_id: UUID`
* `from_value_id: UUID`
* `to_value_id: UUID`
* `reason: str`
* `created_at: datetime`

---

## Constraints

* `Category.path` 為唯一識別路徑
* 同層 `Category.name` 不可重複
* `Attribute.scope` 可為 `global` 或指定 `category_id`
* 覆蓋/擴充策略需在治理規範中定義

---

## Allowed Operations

* **Category**: `create | rename | move | merge | deprecate`
* **Attribute**: `create | merge | split | deprecate`
* **Value**: `add | merge | deprecate`（保留映射 via ValueMapping）

---

## Versioning & Rollback

* 任何 T/N 改動需建立 **治理事件 (G-event)**
* 提升對應版本號（`t_version` / `n_version`）
* 僅版本點支援完整回滾

---

## 建議命名規範

* **屬性名稱 (Attribute.name)**: 使用英文 snake\_case
* **屬性值 (Value)**: 保留原文，並在 `labels` 補齊多語

---
