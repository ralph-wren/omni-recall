import os
import requests
import json
import psycopg2
import re
import base64
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import sys

class OmniRecallManager:
    """
    Omni-Recall: Neural Knowledge & Long-Term Context Engine
    Manages long-term agent memory using Supabase (pgvector) and APIYI (Embeddings).
    """
    def __init__(self):
        self.apiyi_token = os.environ.get('APIYI_TOKEN')
        self.supabase_password = os.environ.get('SUPABASE_PASSWORD')
        self.salt = os.environ.get('SUPABASE_SALT')
        self._cipher = None
        # Standard configuration for TechStack-Handbook infrastructure
        self.db_config = {
            "dbname": "postgres",
            "user": "postgres.tliqsgnqduosgjwkhuyw",
            "host": "aws-1-ap-south-1.pooler.supabase.com",
            "port": "6543"
        }

    def _get_cipher(self):
        if self._cipher:
            return self._cipher
        if not self.salt:
            raise ValueError("Environment variable 'SUPABASE_SALT' is required for encryption/decryption.")
        
        # Derive a 32-byte key from the salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'omni-recall-fixed-salt', # Use a fixed internal salt for KDF
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.salt.encode()))
        self._cipher = Fernet(key)
        return self._cipher

    def encrypt(self, text):
        if not text: return None
        cipher = self._get_cipher()
        return cipher.encrypt(text.encode()).decode()

    def decrypt(self, token):
        if not token: return None
        cipher = self._get_cipher()
        return cipher.decrypt(token.encode()).decode()

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

    def sync_vault(self, key, value):
        """
        Stores an encrypted value in the vault table.
        """
        if not self.supabase_password:
            raise ValueError("Environment variable 'SUPABASE_PASSWORD' is required for vault uplink.")
            
        encrypted_value = self.encrypt(value)
        conn = psycopg2.connect(password=self.supabase_password, **self.db_config)
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO vault (key, value, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (key) 
                    DO UPDATE SET value = EXCLUDED.value, updated_at = NOW();
                """, (key, encrypted_value))
            conn.commit()
            print(f"SUCCESS: Key '{key}' stored securely in vault.")
        finally:
            conn.close()

    def fetch_vault(self, key=None):
        """
        Retrieves and decrypts value(s) from the vault.
        """
        if not self.supabase_password:
            raise ValueError("Environment variable 'SUPABASE_PASSWORD' is required for vault retrieval.")
            
        conn = psycopg2.connect(password=self.supabase_password, **self.db_config)
        try:
            with conn.cursor() as cur:
                if key:
                    cur.execute("SELECT key, value FROM vault WHERE key = %s", (key,))
                else:
                    cur.execute("SELECT key, value FROM vault ORDER BY key ASC")
                rows = cur.fetchall()
                
            results = []
            for r_key, r_val in rows:
                try:
                    decrypted = self.decrypt(r_val)
                    results.append((r_key, decrypted))
                except Exception as e:
                    results.append((r_key, f"[DECRYPTION_ERROR: {str(e)}]"))
            return results
        finally:
            conn.close()

    def fetch_full_context(self, days=10, limit=None):
        """
        Combines ALL user profiles, ALL AI instructions, and recent memories into a comprehensive context.
        Note: Profiles and Instructions are fetched in full (no time limit) to maintain core identity/rules.
        """
        profiles = self.fetch_profile() # Always full retrieval
        instructions = self.fetch_instruction() # Always full retrieval
        memories = self.fetch(days=days, limit=limit) # Time-filtered retrieval
        
        return {
            "profiles": [{"category": p[0], "content": p[1]} for p in profiles],
            "ai_instructions": [{"category": i[0], "content": i[1]} for i in instructions],
            "recent_memories": [{"content": m[0], "time": str(m[1]), "source": m[2]} for m in memories]
        }

    def _split_markdown(self, content):
        """
        Splits markdown content into logical chunks based on headers (up to level 5).
        """
        # Matches #, ##, ###, ####, ##### headers at the start of a line
        header_pattern = r'\n(?=#{1,5} )'
        sections = re.split(header_pattern, content)
        chunks = []
        
        current_chunk = ""
        for section in sections:
            if not section.strip():
                continue
            
            # If a single section is still very large (> 2000 chars), we might want to split it further
            # but usually, H1-H5 headers cover most logical breaks.
            if len(section) > 2000:
                # Fallback: if no smaller headers, split by paragraphs
                sub_sections = re.split(r'\n\n', section)
                for sub in sub_sections:
                    if not sub.strip():
                        continue
                    if len(current_chunk) + len(sub) > 2000:
                        if current_chunk: chunks.append(current_chunk.strip())
                        current_chunk = sub
                    else:
                        current_chunk += "\n\n" + sub
            else:
                if len(current_chunk) + len(section) > 2000:
                    if current_chunk: chunks.append(current_chunk.strip())
                    current_chunk = section
                else:
                    current_chunk += "\n" + section
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def _split_general_text(self, text, chunk_size=1000, overlap=200):
        """
        Recursive character splitting for general text with overlap.
        """
        if len(text) <= chunk_size:
            return [text.strip()]
        
        separators = ["\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", "; ", " ", ""]
        chunks = []
        
        def recursive_split(content, current_chunk_size, current_overlap):
            if len(content) <= current_chunk_size:
                return [content.strip()]
            
            # Find the best separator
            selected_sep = ""
            for sep in separators:
                if sep in content:
                    selected_sep = sep
                    break
            
            # Split by selected separator
            raw_splits = content.split(selected_sep) if selected_sep else list(content)
            
            final_chunks = []
            current_buffer = ""
            
            for s in raw_splits:
                # Add separator back except for the last one
                item = s + selected_sep
                
                if len(current_buffer) + len(item) <= current_chunk_size:
                    current_buffer += item
                else:
                    if current_buffer:
                        final_chunks.append(current_buffer.strip())
                    
                    # Handle overlap: take last 'overlap' characters from current_buffer
                    overlap_text = current_buffer[-current_overlap:] if len(current_buffer) > current_overlap else current_buffer
                    current_buffer = overlap_text + item
            
            if current_buffer:
                final_chunks.append(current_buffer.strip())
            
            return final_chunks

        return recursive_split(text, chunk_size, overlap)

    def batch_sync_doc(self, input_source, source_tag=None, threshold=0.9, cookie=None):
        """
        Reads a markdown file OR fetches a URL, splits it, and syncs chunks to memories.
        Supports .md, .txt, .log and web URLs.
        """
        content = ""
        is_url = input_source.startswith(('http://', 'https://'))
        
        if is_url:
            # Handle GitHub blob URLs by converting them to raw URLs
            if "github.com" in input_source and "/blob/" in input_source:
                input_source = input_source.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            
            print(f"Fetching content from URL: {input_source}...")
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Referer": "https://www.zhihu.com/",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
            if cookie:
                headers["Cookie"] = cookie
                
            res = requests.get(input_source, headers=headers, timeout=15)
            res.raise_for_status()
            content = res.text
            if not source_tag:
                source_tag = input_source.split('/')[-1] or "web-page"
        else:
            if not os.path.exists(input_source):
                raise FileNotFoundError(f"File not found: {input_source}")
            with open(input_source, 'r', encoding='utf-8') as f:
                content = f.read()
            if not source_tag:
                source_tag = os.path.basename(input_source).replace('.md', '').replace('.txt', '').lower()

        # Choose splitter based on file type or content
        if not is_url and input_source.endswith('.md'):
            print(f"Using Markdown splitter (H1-H5)...")
            chunks = self._split_markdown(content)
        else:
            print(f"Using General Text splitter (Recursive + Overlap)...")
            chunks = self._split_general_text(content)
            
        print(f"Found {len(chunks)} logical chunks.")

        success_count = 0
        skip_count = 0

        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...", end="\r")
            if self.sync(chunk, source=source_tag, threshold=threshold):
                success_count += 1
            else:
                skip_count += 1
        
        print(f"\nBatch sync completed: {success_count} synced, {skip_count} skipped.")
        return success_count, skip_count

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
        print("  python3 omni_ops.py sync-vault <key> <value>")
        print("  python3 omni_ops.py fetch-vault [key]")
        print("  python3 omni_ops.py fetch-full-context [days] [limit]")
        print("  python3 omni_ops.py batch-sync-doc <file_path> [source_tag] [threshold]")
        sys.exit(1)

    action = sys.argv[1]
    
    try:
        if action == "sync" and len(sys.argv) > 2:
            content = sys.argv[2]
            source = sys.argv[3] if len(sys.argv) > 3 else "omni-manual-uplink"
            threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.9
            if manager.sync(content, source, threshold):
                print("SUCCESS: Context synchronized to neural base.")
        elif action == "batch-sync-doc":
            if len(sys.argv) < 3:
                print("Usage: python3 omni_ops.py batch-sync-doc <file_path_or_url> [source_tag] [threshold] [cookie]")
                sys.exit(1)
            file_path = sys.argv[2]
            source_tag = sys.argv[3] if len(sys.argv) > 3 else None
            threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.9
            cookie = sys.argv[5] if len(sys.argv) > 5 else None
            manager.batch_sync_doc(file_path, source_tag, threshold, cookie=cookie)
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
        elif action == "sync-vault" and len(sys.argv) > 3:
            key = sys.argv[2]
            value = sys.argv[3]
            manager.sync_vault(key, value)
        elif action == "fetch-vault":
            key = sys.argv[2] if len(sys.argv) > 2 else None
            results = manager.fetch_vault(key)
            for k, v in results:
                print(f"[{k}]: {v}")
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
