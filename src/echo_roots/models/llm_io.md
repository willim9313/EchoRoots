# LLM I/O Models

## Purpose

確保 LLM 相關 **輸入/輸出欄位** 在不同任務中穩定，
支持 **分次驅動 (multi-stage)**，並降低幻覺風險與提高驗證效率。

---

## Workflow 任務類型

1. **Category Classification**

   * 前置：embedding model 先計算 item 向量，檢索 top-k 候選分類
   * LLM 接收「候選分類編號列表」並回傳選中的數字
   * 系統再將數字映射回 `category_id`

2. **Attribute Extraction**

   * 前置：分類已確認
   * prompt 提交時，每個屬性提供候選值編號
   * LLM 僅需回傳「選號組合」
   * 系統再映射為結構化屬性值

3. **Residual Handling**

   * 屬性任務中未被選中的候選值 → 自動寫入 **S-STAGING**
   * 不再需要獨立的「Semantic Spill 任務」

---

## Common Inputs

```json
{
  "item_id": "uuid",
  "domain": "product|knowledge_base",
  "lang": "zh-TW|en",
  "title": "string",
  "description": "string|null",
  "specs_or_tags": {"...": "..."},
  "raw_category_path": ["..."],
  "context": {"...": "..."}
}
```

---

## Task-Specific Outputs

### 1. Category Classification

Prompt 會帶入編號候選，例如：

```
1. Electronics > Phones > Smartphones
2. Electronics > Accessories > Cases
3. Home > Kitchen > Appliances
```

LLM 輸出：

```json
{
  "selected_option": 1,
  "confidence": 0.87,
  "evidence": [<Evidence>...]
}
```

系統再將 `1` 對應回 `category_id = uuid`。

---

### 2. Attribute Extraction

Prompt 會帶入每個屬性的候選值編號，例如：

```
{
  "A1": {
    "attr": "color",
    "values": {
      "1": "天然鈦",
      "2": "黑色",
      "3": "白色"
    }
  },
  "A2": {
    "attr": "storage",
    "values": {
      "1": "128GB",
      "2": "256GB",
      "3": "512GB"
    }
  }
}
```

LLM 輸出：

```json
{
  "selected_options": {
    "A1": [1], 
    "A2": [2] 
  },
  "evidence": {
    "A1": [<Evidence>...],
    "A2": [<Evidence>...]
  }
}
```

系統再映射為：

```json
{
  "color": {"value": "天然鈦", "value_id": "uuid"},
  "storage": {"value": "256GB"}
}
```

---

### 3. Residual Handling

* 在屬性任務中，所有 **未被選中的候選值** → 自動記錄到 **S-STAGING**
* 結構如下：

```json
{
  "obs_id": "uuid",
  "attr_hint": "color",
  "value_text": "黑色",
  "item_id": "uuid",
  "state": "STAGING",
  "model_id": "gpt-4o-mini"
}
```

---

## Evidence Schema

```json
{
  "source": "title|description|specs.storage|text_uri",
  "span": [0, 5],
  "text": "天然鈦",
  "score": 0.92,
  "model_id": "st-emb-xxx|gpt|gemini"
}
```

---

## Guardrails

* **分類必須經過 embedding 縮減 + LLM 選號**
* **屬性僅允許從候選中選號**（避免生成不存在的值）
* **殘餘值自動進入 S-STAGING**，不再需要獨立任務
* **寫回 D\_norm 時需記錄 norm\_version**
* **所有 evidence 必須可追溯至來源與模型 ID**

---