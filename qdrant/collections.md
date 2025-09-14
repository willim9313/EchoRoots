# Qdrant Collections (S Layer)

## 命名規則

* `s_values_{domain}`
  例如：

  * `s_values_product`
  * `s_values_kb`（或 `s_values_knowledge_base`）

## 向量維度

* 依所選模型決定（常見：384 / 768 / 1024）
* 模型名稱或版本記錄於 `model_id` 欄位

## Payload 最小結構

```json
{
  "obs_id": "uuid",                // 單一觀測值 ID
  "item_id": "uuid",               // 所屬 item
  "category_id": "uuid|null",      // 所屬分類，可為 null
  "attr_hint": "string|null",      // 屬性提示，如 color|brand|topic
  "value_text": "string",          // 原始文字值
  "lang": "zh-TW|en|...",          // 語言代碼
  "model_id": "sentence-transformers/... or LLM tag", // 嵌入來源
  "state": "STAGING|HOT|ARCHIVE",  // 狀態
  "cluster_id": "uuid|null"        // 聚類所屬，可為 null
}
```

## Collection 設定範例（YAML）

```yaml
name: s_values_product
vectors:
  size: 768         # 視模型而定（384 / 768 / 1024）
  distance: Cosine
optimizers_config:
  default_segment_number: 2
quantization_config: null   # MVP 先不量化
hnsw_config:
  m: 16
  ef_construct: 100
on_disk_payload: true
```

## 狀態轉換流程

* **STAGING**：新觀測值直接寫入
* **HOT**：形成 cluster 後，更新 `cluster_id` 並將該 cluster 設為 HOT
* **ARCHIVE**：低活躍或被合併的 cluster，移入 ARCHIVE（保留追溯用）
