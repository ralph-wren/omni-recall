import os
import requests
import json
import psycopg2
from datetime import datetime, timedelta
import sys

class OmniRecallManager:
    """
    Omni-Recall: Neural Knowledge & Long-Term Context Engine
    Manages long-term agent memory using Supabase (pgvector) and APIYI (Embeddings).
    """
    def __init__(self):
        self.apiyi_token = os.environ.get('APIYI_TOKEN')
        self.supabase_password = os.environ.get('SUPABASE_PASSWORD')
        # Standard configuration for TechStack-Handbook infrastructure
        self.db_config = {
            "dbname": "postgres",
            "user": "postgres.tliqsgnqduosgjwkhuyw",
            "host": "aws-1-ap-south-1.pooler.supabase.com",
            "port": "6543"
        }

    def _get_embedding(self, text):
        if not self.apiyi_token:
            raise ValueError("Environment variable 'APIYI_TOKEN' is required for neural encoding.")
        
        url = "https://api.apiyi.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {self.apiyi_token}", "Content-Type": "application/json"}
        data = {"input": text.strip(), "model": "text-embedding-3-small"}
        
        res = requests.post(url, headers=headers, json=data)
        if res.status_code != 200:
            raise Exception(f"Neural encoding failed: {res.text}")
        return res.json()['data'][0]['embedding']

    def sync(self, content, source="omni-recall-sync"):
        """Synchronizes content to the neural knowledge base."""
        if not self.supabase_password:
            raise ValueError("Environment variable 'SUPABASE_PASSWORD' is required for database uplink.")
            
        print(f"Encoding content into vector space...")
        embedding = self._get_embedding(content)
        
        print(f"Uplinking to Supabase knowledge cluster...")
        conn = psycopg2.connect(password=self.supabase_password, **self.db_config)
        cur = conn.cursor()
        
        metadata = {
            "engine": "omni-recall-v1",
            "model": "text-embedding-3-small",
            "timestamp": datetime.now().isoformat()
        }
        
        cur.execute("""
            INSERT INTO memories (content, embedding, metadata, source)
            VALUES (%s, %s, %s, %s)
        """, (content, embedding, json.dumps(metadata), source))
        
        conn.commit()
        cur.close()
        conn.close()
        return True

    def fetch(self, days=10, limit=None):
        """Retrieves historical context from the neural knowledge base."""
        if not self.supabase_password:
            raise ValueError("Environment variable 'SUPABASE_PASSWORD' is required for context retrieval.")
            
        conn = psycopg2.connect(password=self.supabase_password, **self.db_config)
        cur = conn.cursor()
        
        since_date = datetime.now() - timedelta(days=days)
        
        query = "SELECT content, created_at, source, metadata FROM memories WHERE created_at >= %s ORDER BY created_at DESC"
        params = [since_date]
        
        if limit:
            query += " LIMIT %s"
            params.append(limit)
            
        cur.execute(query, tuple(params))
        rows = cur.fetchall()
        
        cur.close()
        conn.close()
        return rows

if __name__ == "__main__":
    manager = OmniRecallManager()
    if len(sys.argv) < 2:
        print("Omni-Recall Engine CLI")
        print("Usage: python3 omni_ops.py sync 'content' [source] | fetch [days] [limit]")
        sys.exit(1)

    action = sys.argv[1]
    
    try:
        if action == "sync" and len(sys.argv) > 2:
            content = sys.argv[2]
            source = sys.argv[3] if len(sys.argv) > 3 else "omni-manual-uplink"
            if manager.sync(content, source):
                print("SUCCESS: Context synchronized to neural base.")
        elif action == "fetch":
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
            memories = manager.fetch(days, limit)
            print(json.dumps([{"content": m[0], "time": str(m[1]), "source": m[2], "metadata": m[3]} for m in memories], ensure_ascii=False))
        else:
            print("Invalid command or missing parameters.")
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        sys.exit(1)
