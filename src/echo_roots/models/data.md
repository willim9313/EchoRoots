# Data Models: D / OD / S / G

## Purpose

規範 **資料流入 (D)**、**暫置/異常 (OD)**、**治理事件 (G)**、**語意層 (S)** 的最小欄位集合，
以支援 **商品 (product)** 與 **文本 (knowledge\_base)** 兩種主要 domain。
適用於 W1 / W2 / W3 / W6 workflow。

---

## 共用列舉

* **Domain**: `"product" | "knowledge_base"`
* **SState**: `"STAGING" | "HOT" | "ARCHIVE"`
* **NormStatus**: `"ok" | "partial" | "failed"`
* **ReviewStatus**: `"pending" | "approved" | "rejected"`

---

## D（Domain Data）

### D\_raw (ItemRaw)

* `item_id: UUID`
* `domain: Domain`
* `title: str`
* `description: str?`
* `raw_category_path: List[str]`
* `lang: str?` — ISO (`"zh-TW"`, `"en"`)
* `source: str?`
* **product 專屬**:

  * `brand: str?`
  * `price: float?`
  * `specs: Dict[str, Any]`
* **knowledge\_base 專屬**:

  * `text_tags: List[str]`
  * `text_uri: str?`
* `created_at, updated_at: datetime`

### D\_norm (ItemNorm)

* `item_id: UUID`
* `domain: Domain`
* `category_id: UUID?` — 對應 T 層
* `attrs: Dict[str, Any]`

  * 例：`{"color": {"value_id": "...", "value": "綠色"}}`
* `evidence: Dict[str, Any]` — span / score / 來源欄位
* `t_version, n_version, norm_version: int?`
* `status: NormStatus`
* `created_at, updated_at: datetime`

---

## OD（Overflow / Outlier）

### OutlierRecord

* `outlier_id: UUID`
* `item_id: UUID`
* `domain: Domain`
* `reason: str` — e.g., `no_category`, `no_attr_mapping`
* `suggestion: str?`
* `review_status: ReviewStatus`
* `created_at, updated_at: datetime`

---

## S（Semantic Layer）

### S\_staging\_obs (SStagingObs)

* `obs_id: UUID`
* `domain: Domain`
* `item_id: UUID`
* `category_id: UUID?`
* `attr_hint: str?`
* `value_text: str`
* `lang: str?`
* `model_id: str?`
* `created_at, updated_at: datetime`

### S\_cluster (SCluster)

* `cluster_id: UUID`
* `domain: Domain`
* `category_id: UUID?`
* `state: SState` — 預設 `"HOT"`
* `rep_term: str`
* `aliases: Set[str]`
* `support: int`
* `attr_distribution: Dict[str, int]`
* `propose_attr_id: UUID?`
* `propose_value: str?`
* `created_at, updated_at: datetime`

---

## G（Governance Events）

### GovernanceEvent

* `event_id: UUID`
* `op: "snapshot" | "diff" | "rollback" | "merge_attr" | "split_attr" | "deprecate_value" | "update_mapping"`
* `actor: str`
* `ts: datetime`
* `before_ref: str?`
* `after_ref: str?`
* `payload: Dict[str, Any]`
* **要求**: 所有寫回動作需關聯 `event_id`

### ImpactedSet

* `batch_id: UUID`
* `item_ids: List[UUID]`
* `reason: str`
* `created_at: datetime`

---