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

    def sync(self, content, source="omni-recall-sync", threshold=0.9):
        """Synchronizes content to the neural knowledge base with duplicate detection."""
        if not self.supabase_password:
            raise ValueError("Environment variable 'SUPABASE_PASSWORD' is required for database uplink.")
            
        print(f"Encoding content into vector space...")
        embedding = self._get_embedding(content)
        
        print(f"Checking for high-similarity duplicates (threshold={threshold})...")
        conn = psycopg2.connect(password=self.supabase_password, **self.db_config)
        cur = conn.cursor()
        
        # pgvector: 1 - (embedding <=> %s) is cosine similarity
        cur.execute("""
            SELECT content, 1 - (embedding <=> %s::vector) as similarity 
            FROM memories 
            ORDER BY embedding <=> %s::vector 
            LIMIT 1
        """, (embedding, embedding))
        
        result = cur.fetchone()
        if result:
            existing_content, similarity = result
            if similarity >= threshold:
                print(f"SKIP: High similarity detected ({similarity:.4f}). Content already exists in neural base.")
                cur.close()
                conn.close()
                return False

        print(f"Uplinking to Supabase knowledge cluster...")
        metadata = {
            "engine": "omni-recall-v1",
            "model": "text-embedding-3-small",
            "timestamp": datetime.now().isoformat(),
            "language": "zh-CN"
        }
        
        cur.execute("""
            INSERT INTO memories (content, embedding, metadata, source)
            VALUES (%s, %s, %s, %s)
        """, (content, embedding, json.dumps(metadata), source))
        
        conn.commit()
        cur.close()
        conn.close()
        return True

    def fetch(self, days=10, limit=None, keywords=None):
        """Retrieves historical context from the neural knowledge base."""
        if not self.supabase_password:
            raise ValueError("Environment variable 'SUPABASE_PASSWORD' is required for context retrieval.")
            
        conn = psycopg2.connect(password=self.supabase_password, **self.db_config)
        cur = conn.cursor()
        
        since_date = datetime.now() - timedelta(days=days)
        
        query = "SELECT content, created_at, source, metadata FROM memories WHERE created_at >= %s"
        params = [since_date]

        if keywords:
            if isinstance(keywords, str):
                keywords = [keywords]
            for kw in keywords:
                query += " AND content ILIKE %s"
                params.append(f"%{kw}%")
            
        query += " ORDER BY created_at DESC"
        
        if limit:
            query += " LIMIT %s"
            params.append(limit)
            
        cur.execute(query, tuple(params))
        rows = cur.fetchall()
        
        cur.close()
        conn.close()
        return rows

    def sync_instruction(self, category, content, threshold=0.9):
        """Synchronizes AI instruction (tone, workflow, rule) with duplicate detection."""
        if not self.supabase_password:
            raise ValueError("Environment variable 'SUPABASE_PASSWORD' is required for database uplink.")
            
        print(f"Encoding instruction '{category}' into vector space...")
        embedding = self._get_embedding(content)
        
        conn = psycopg2.connect(password=self.supabase_password, **self.db_config)
        cur = conn.cursor()
        
        # Check for duplicates within the same category
        cur.execute("""
            SELECT content, 1 - (embedding <=> %s::vector) as similarity 
            FROM instructions 
            WHERE category = %s
            ORDER BY embedding <=> %s::vector 
            LIMIT 1
        """, (embedding, category, embedding))
        
        result = cur.fetchone()
        if result:
            existing_content, similarity = result
            if similarity >= threshold:
                print(f"SKIP: High similarity detected in '{category}' ({similarity:.4f}). Instruction already exists.")
                cur.close()
                conn.close()
                return False

        print(f"Uplinking to Supabase instruction cluster...")
        metadata = {
            "engine": "omni-recall-v1",
            "model": "text-embedding-3-small",
            "timestamp": datetime.now().isoformat(),
            "language": "zh-CN"
        }
        
        cur.execute("""
            INSERT INTO instructions (category, content, embedding, metadata)
            VALUES (%s, %s, %s, %s)
        """, (category, content, embedding, json.dumps(metadata)))
        
        conn.commit()
        cur.close()
        conn.close()
        return True

    def fetch_instruction(self, category=None, keywords=None):
        """Retrieves AI instructions, optionally filtered by category and keywords."""
        if not self.supabase_password:
            raise ValueError("Environment variable 'SUPABASE_PASSWORD' is required for context retrieval.")
            
        conn = psycopg2.connect(password=self.supabase_password, **self.db_config)
        cur = conn.cursor()
        
        query = "SELECT category, content, metadata FROM instructions WHERE 1=1"
        params = []

        if category:
            query += " AND category = %s"
            params.append(category)

        if keywords:
            if isinstance(keywords, str):
                keywords = [keywords]
            for kw in keywords:
                query += " AND content ILIKE %s"
                params.append(f"%{kw}%")
            
        query += " ORDER BY category ASC, created_at DESC"
            
        cur.execute(query, tuple(params))
        rows = cur.fetchall()
        
        cur.close()
        conn.close()
        return rows

    def fetch_full_context(self, days=10, limit=None):
        """Combines all user profiles, AI instructions, and recent memories into a comprehensive context."""
        profiles = self.fetch_profile()
        instructions = self.fetch_instruction()
        memories = self.fetch(days=days, limit=limit)
        
        return {
            "profiles": [{"category": p[0], "content": p[1]} for p in profiles],
            "ai_instructions": [{"category": i[0], "content": i[1]} for i in instructions],
            "recent_memories": [{"content": m[0], "time": str(m[1]), "source": m[2]} for m in memories]
        }

    def sync_profile(self, category, content, threshold=0.9):
        """Synchronizes user profile (role, preference, setting) with duplicate detection."""
        if not self.supabase_password:
            raise ValueError("Environment variable 'SUPABASE_PASSWORD' is required for database uplink.")
            
        print(f"Encoding profile '{category}' into vector space...")
        embedding = self._get_embedding(content)
        
        conn = psycopg2.connect(password=self.supabase_password, **self.db_config)
        cur = conn.cursor()
        
        # Check for duplicates within the same category
        cur.execute("""
            SELECT content, 1 - (embedding <=> %s::vector) as similarity 
            FROM profiles 
            WHERE category = %s
            ORDER BY embedding <=> %s::vector 
            LIMIT 1
        """, (embedding, category, embedding))
        
        result = cur.fetchone()
        if result:
            existing_content, similarity = result
            if similarity >= threshold:
                print(f"SKIP: High similarity detected in '{category}' ({similarity:.4f}). Profile already exists.")
                cur.close()
                conn.close()
                return False

        print(f"Uplinking to Supabase profile cluster...")
        metadata = {
            "engine": "omni-recall-v1",
            "model": "text-embedding-3-small",
            "timestamp": datetime.now().isoformat(),
            "language": "zh-CN"
        }
        
        cur.execute("""
            INSERT INTO profiles (category, content, embedding, metadata)
            VALUES (%s, %s, %s, %s)
        """, (category, content, embedding, json.dumps(metadata)))
        
        conn.commit()
        cur.close()
        conn.close()
        return True

    def fetch_profile(self, category=None, keywords=None):
        """Retrieves user profiles, optionally filtered by category and keywords."""
        if not self.supabase_password:
            raise ValueError("Environment variable 'SUPABASE_PASSWORD' is required for context retrieval.")
            
        conn = psycopg2.connect(password=self.supabase_password, **self.db_config)
        cur = conn.cursor()
        
        query = "SELECT category, content, metadata FROM profiles WHERE 1=1"
        params = []

        if category:
            query += " AND category = %s"
            params.append(category)

        if keywords:
            if isinstance(keywords, str):
                keywords = [keywords]
            for kw in keywords:
                query += " AND content ILIKE %s"
                params.append(f"%{kw}%")
            
        query += " ORDER BY category ASC, created_at DESC"
            
        cur.execute(query, tuple(params))
        rows = cur.fetchall()
        
        cur.close()
        conn.close()
        return rows

if __name__ == "__main__":
    manager = OmniRecallManager()
    if len(sys.argv) < 2:
        print("Omni-Recall Engine CLI (Default Language: zh-CN)")
        print("Usage:")
        print("  python3 omni_ops.py sync 'content' [source] [threshold]")
        print("  python3 omni_ops.py fetch [days] [limit] [keyword1] [keyword2] ...")
        print("  python3 omni_ops.py sync-profile <category> <content> [threshold]")
        print("  python3 omni_ops.py fetch-profile [category] [keyword1] [keyword2] ...")
        print("  python3 omni_ops.py sync-instruction <category> <content> [threshold]")
        print("  python3 omni_ops.py fetch-instruction [category] [keyword1] [keyword2] ...")
        print("  python3 omni_ops.py fetch-full-context [days] [limit]")
        sys.exit(1)

    action = sys.argv[1]
    
    try:
        if action == "sync" and len(sys.argv) > 2:
            content = sys.argv[2]
            source = sys.argv[3] if len(sys.argv) > 3 else "omni-manual-uplink"
            threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.9
            if manager.sync(content, source, threshold):
                print("SUCCESS: Context synchronized to neural base.")
        elif action == "sync-profile" and len(sys.argv) > 3:
            category = sys.argv[2]
            content = sys.argv[3]
            threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.9
            if manager.sync_profile(category, content, threshold):
                print(f"SUCCESS: Profile '{category}' synchronized.")
        elif action == "fetch":
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            limit = int(sys.argv[3]) if (len(sys.argv) > 3 and sys.argv[3].lower() != 'none') else None
            keywords = sys.argv[4:] if len(sys.argv) > 4 else None
            memories = manager.fetch(days, limit, keywords)
            print(json.dumps([{"content": m[0], "time": str(m[1]), "source": m[2], "metadata": m[3]} for m in memories], ensure_ascii=False))
        elif action == "fetch-profile":
            category = sys.argv[2] if (len(sys.argv) > 2 and sys.argv[2].lower() != 'none') else None
            keywords = sys.argv[3:] if len(sys.argv) > 3 else None
            profiles = manager.fetch_profile(category, keywords)
            print(json.dumps([{"category": p[0], "content": p[1], "metadata": p[2]} for p in profiles], ensure_ascii=False))
        elif action == "sync-instruction" and len(sys.argv) > 3:
            category = sys.argv[2]
            content = sys.argv[3]
            threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.9
            if manager.sync_instruction(category, content, threshold):
                print(f"SUCCESS: Instruction '{category}' synchronized.")
        elif action == "fetch-instruction":
            category = sys.argv[2] if (len(sys.argv) > 2 and sys.argv[2].lower() != 'none') else None
            keywords = sys.argv[3:] if len(sys.argv) > 3 else None
            instructions = manager.fetch_instruction(category, keywords)
            print(json.dumps([{"category": i[0], "content": i[1], "metadata": i[2]} for i in instructions], ensure_ascii=False))
        elif action == "fetch-full-context":
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            limit = int(sys.argv[3]) if (len(sys.argv) > 3 and sys.argv[3].lower() != 'none') else None
            context = manager.fetch_full_context(days, limit)
            print(json.dumps(context, ensure_ascii=False))
        else:
            print("Invalid command or missing parameters.")
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        sys.exit(1)
