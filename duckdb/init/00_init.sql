-- DuckDB 最小 DDL（可跑 W1/W2/W3/W6）

CREATE TABLE d_raw (
  item_id TEXT PRIMARY KEY,
  domain TEXT, title TEXT, description TEXT,
  raw_category_path JSON, lang TEXT, source TEXT,
  specs JSON, tags JSON,
  created_at TIMESTAMP, updated_at TIMESTAMP
);

CREATE TABLE d_norm (
  item_id TEXT PRIMARY KEY,
  domain TEXT,
  category_id TEXT,
  attrs JSON,
  evidence JSON,
  t_version INTEGER, n_version INTEGER, norm_version INTEGER,
  status TEXT,
  created_at TIMESTAMP, updated_at TIMESTAMP
);

CREATE TABLE d_outlier (
  outlier_id TEXT PRIMARY KEY,
  item_id TEXT, domain TEXT, reason TEXT, suggestion TEXT,
  review_status TEXT,
  created_at TIMESTAMP, updated_at TIMESTAMP
);

CREATE TABLE n_attributes (
  attr_id TEXT PRIMARY KEY,
  domain TEXT, name TEXT, type TEXT, description TEXT,
  allowed_value_ids JSON, constraints JSON, status TEXT, n_version INTEGER
);

CREATE TABLE n_values (
  value_id TEXT PRIMARY KEY,
  attribute_id TEXT, value TEXT, labels JSON, aliases JSON,
  status TEXT, metadata JSON
);

CREATE TABLE value_mapping (
  mapping_id TEXT PRIMARY KEY,
  from_value_id TEXT, to_value_id TEXT, reason TEXT, created_at TIMESTAMP
);

CREATE TABLE t_categories (
  category_id TEXT PRIMARY KEY,
  domain TEXT, name TEXT, parent_id TEXT,
  path JSON, path_ids JSON, status TEXT, t_version INTEGER
);

CREATE TABLE s_staging_obs (
  obs_id TEXT PRIMARY KEY,
  domain TEXT, item_id TEXT, category_id TEXT,
  attr_hint TEXT, value_text TEXT, lang TEXT, model_id TEXT,
  created_at TIMESTAMP, updated_at TIMESTAMP
);

CREATE TABLE s_clusters (
  cluster_id TEXT PRIMARY KEY,
  domain TEXT, category_id TEXT, state TEXT,
  rep_term TEXT, aliases JSON, support INTEGER,
  attr_distribution JSON,
  created_at TIMESTAMP, updated_at TIMESTAMP
);

CREATE TABLE g_events (
  event_id TEXT PRIMARY KEY,
  op TEXT, actor TEXT, ts TIMESTAMP,
  before_ref TEXT, after_ref TEXT, payload JSON
);

CREATE TABLE impacted_set (
  batch_id TEXT PRIMARY KEY,
  item_ids JSON, reason TEXT, created_at TIMESTAMP
);