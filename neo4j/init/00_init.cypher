// 最小圖約束與索引
CREATE CONSTRAINT category_id IF NOT EXISTS FOR (c:Category) REQUIRE c.category_id IS UNIQUE;
CREATE CONSTRAINT attr_id IF NOT EXISTS FOR (a:Attribute) REQUIRE a.attr_id IS UNIQUE;
CREATE CONSTRAINT value_id IF NOT EXISTS FOR (v:Value) REQUIRE v.value_id IS UNIQUE;

// 節點屬性參考
// (:Category {category_id, domain, name, t_version, status})
// (:Attribute {attr_id, domain, name, type, n_version, status})
// (:Value {value_id, value, status})

// 關係型別
// (:Category)-[:CHILD_OF]->(:Category)
// (:Category)-[:BINDS_ATTR]->(:Attribute)
// (:Attribute)-[:HAS_VALUE]->(:Value)
// (:Value)-[:MAPS_TO]->(:Value)

// 建立關係範例（插入時機依 W1 初始化腳本控制）
/*
MERGE (parent:Category {category_id:$parent_id})
MERGE (child:Category  {category_id:$child_id})
MERGE (child)-[:CHILD_OF]->(parent);

MERGE (c:Category {category_id:$cat_id})
MERGE (a:Attribute {attr_id:$attr_id})
MERGE (c)-[:BINDS_ATTR]->(a);

MERGE (a:Attribute {attr_id:$attr_id})
MERGE (v:Value {value_id:$val_id})
MERGE (a)-[:HAS_VALUE]->(v);

MERGE (src:Value {value_id:$from_id})
MERGE (dst:Value {value_id:$to_id})
MERGE (src)-[:MAPS_TO]->(dst);
*/
