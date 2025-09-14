* 批次路徑：`data/{domain}/{table}/dt=YYYY-MM-DD/batch_id={id}/...`
* 版本標籤：`T-<n>`、`N-<n>`（先用整數，必要時再引入 `<n>.<m>`）
* 事件 ID：`EVT-YYYYMMDD-HHMMSS-<shortuuid>`
* 表/集合命名：`d_raw_* / d_norm_* / s_staging_* / s_hot_* / od_inbox / g_events / value_mapping`