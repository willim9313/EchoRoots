"""測試三個核心儲存系統的連接"""
import duckdb
import neo4j
import qdrant_client
from qdrant_client.models import Distance, VectorParams

def test_duckdb():
    """測試 DuckDB 連接"""
    try:
        conn = duckdb.connect("duckdb/db/echo_roots.duckdb")
        result = conn.execute("SELECT 'DuckDB connected!' as status").fetchone()
        print(f"DuckDB: {result[0]}")
        conn.close()
        return True
    except Exception as e:
        print(f"DuckDB error: {e}")
        return False

def test_neo4j():
    """測試 Neo4j 連接"""
    try:
        driver = neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password123"))
        with driver.session() as session:
            result = session.run("RETURN 'Neo4j connected!' as status")
            record = result.single()
            print(f"Neo4j: {record['status']}")
        driver.close()
        return True
    except Exception as e:
        print(f"Neo4j error: {e}")
        return False

def test_qdrant():
    """測試 Qdrant 連接"""
    try:
        client = qdrant_client.QdrantClient(host="localhost", port=6333)
        collections = client.get_collections()
        print(f"Qdrant: Connected! Collections count: {len(collections.collections)}")
        return True
    except Exception as e:
        print(f"Qdrant error: {e}")
        return False

if __name__ == "__main__":
    print("Testing storage connections...")
    results = [
        test_duckdb(),
        test_neo4j(), 
        test_qdrant()
    ]
    
    if all(results):
        print("All systems connected successfully!")
    else:
        print("Some connections failed.")