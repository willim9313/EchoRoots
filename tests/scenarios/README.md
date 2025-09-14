# 測試劇本（MVP）

## 劇本 1：W1 初始化
- 以小型 T/N JSON 匯入，驗證 DuckDB/Neo4j 結構建立

## 劇本 2：W3 新流入
- 以 5 筆 product、5 筆 knowledge_base 的 `d_raw`，產生對應 `s_staging_obs`

## 劇本 3：W2 回填
- 將劇本 2 產生的 `d_raw` 正規化為 `d_norm`，無法掛載者寫入 `d_outlier`

建議測試資料（可放 data/，測試指向）

data/product/d_raw/dt=2025-09-12/batch_id=demo/raw.json

data/knowledge_base/d_raw/dt=2025-09-12/batch_id=demo/raw.json

data/common/t_init.json、data/common/n_init.json