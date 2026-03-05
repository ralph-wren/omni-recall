---
name: "omni-recall"
description: "Omni-Recall: Neural Knowledge & Long-Term Context Engine with Vector Semantic Search. Manages cross-session agent memory via Supabase (pgvector + HNSW) and APIYI. Supports intelligent natural language queries with similarity scoring."
---

# Omni-Recall: Neural Knowledge & Long-Term Context Engine

Omni-Recall is a high-performance memory management skill designed for AI agents. It enables persistent, cross-session awareness by transforming conversation history and technical insights into high-dimensional vector embeddings, stored in a Supabase (PostgreSQL + pgvector) knowledge cluster with HNSW indexing for fast semantic search.

## 🚀 Core Capabilities

1. **Vector Semantic Search (`fetch` with `query_text`)**:
   Intelligent natural language queries using vector similarity. Finds semantically related content even with different wording. Returns results ranked by similarity score (0-1). Default threshold: 0.6 (balanced accuracy and recall).

2. **Neural Synchronization (`sync`)**:
   Encodes current session state, user preferences, and operational steps into 1536-dimensional vectors using OpenAI's `text-embedding-3-small` via APIYI. **Includes automatic duplicate detection** (skips if cosine similarity > 0.9). Supports optional `category` and `importance` fields.

3. **Contextual Retrieval (`fetch`)**:
   Pulls historical neural records using natural language queries or time-based filters. Supports similarity threshold tuning (0.5-0.9) and category filtering.

4. **User Profile Management (`sync-profile` / `fetch-profile`)**:
   Manages user roles, preferences, settings, and personas in a dedicated `profiles` matrix with vector search support.

5. **AI Instruction Management (`sync-instruction` / `fetch-instruction`)**:
   Stores operational requirements for the AI with semantic search capabilities.

---

## 🛠 Usage Examples

### Vector Semantic Search (Recommended)
```bash
# Search with natural language (default threshold: 0.6)
python3 scripts/omni_ops.py fetch "如何优化数据库性能" none 10

# Search with custom similarity threshold
python3 scripts/omni_ops.py fetch "pgvector 索引优化" none 10 none 0.7

# Search last 7 days for AI-related content
python3 scripts/omni_ops.py fetch "AI Agent 开发" 7 10

# Search instructions with semantic understanding
python3 scripts/omni_ops.py fetch-instruction "代码风格规范" none 0.6 5

# Search profiles
python3 scripts/omni_ops.py fetch-profile "用户技能背景" none 0.6 5

# List all records (use 'none' as query)
python3 scripts/omni_ops.py fetch none 30 10
```

### Similarity Threshold Guide
| Threshold | Description | Use Case |
|-----------|-------------|----------|
| **0.6** | **Balanced (Default)** ⭐ | General search, best for most cases |
| 0.7-0.8 | High precision | Exact matches, specific queries |
| 0.5 | Exploratory | Broad search, discover related content |
| 0.8+ | Very precise | Almost exact matches only |

### Synchronize Session Context
```bash
# Basic sync
python3 scripts/omni_ops.py sync "User is interested in Python optimization." "session-tag" 0.9

# Sync with category and importance
python3 scripts/omni_ops.py sync "New tech stack insight" "research" 0.9 "technical" 0.8
```

### Synchronize User Profile
```bash
# Set a persona
python3 scripts/omni_ops.py sync-profile "persona" "Experienced Senior Backend Engineer, favors Go and Python."

# Set a preference
python3 scripts/omni_ops.py sync-profile "preference" "Prefers concise code without excessive comments."
```

### Synchronize AI Instructions
```bash
# Set tone
python3 scripts/omni_ops.py sync-instruction "tone" "Professional yet friendly, use 'Partner' as my nickname."

# Set workflow steps
python3 scripts/omni_ops.py sync-instruction "workflow" "1. Plan -> 2. Implementation -> 3. Verification -> 4. Summary."
```

### Encrypted Nsfw Memory (Sensitive Context)
```bash
# Sync sensitive content (Encrypted at rest + Vector embedding)
python3 scripts/omni_ops.py sync-nsfw "Sensitive information here" "private-tag" 0.9

# Fetch with semantic search
python3 scripts/omni_ops.py fetch-nsfw "敏感查询" 30 10 none 0.6

# Fetch full context including nsfw records
python3 scripts/omni_ops.py fetch-full-context 10 none true
```

### Encrypted Vault (Key-Value Storage)
```bash
# Store an encrypted value
python3 scripts/omni_ops.py sync-vault "ZHIHU_COOKIE" "your_long_cookie_string"

# Fetch and decrypt a value
python3 scripts/omni_ops.py fetch-vault "ZHIHU_COOKIE"
```

### Batch Synchronize Document/URL
```bash
# Sync a markdown file (H1-H5 Splitting)
python3 scripts/omni_ops.py batch-sync-doc "/path/to/doc.md" "tag" 0.9

# Sync a web page via URL
python3 scripts/omni_ops.py batch-sync-doc "https://example.com/article" "web-source" 0.9
```

### Fetch Full Context (Recommended for First Recall)
**Priority Order**: 1. `instructions` (Persona/Rules) > 2. `profiles` (User Info/Preferences) > 3. `memories` (Session History).
```bash
# Get ALL instructions + ALL profiles + memories from last 10 days
python3 scripts/omni_ops.py fetch-full-context 10
```

---

## 🏗 Schema Setup (Supabase / Postgres)

### 1. Supabase Knowledge Cluster
Execute the following SQL in your Supabase project to initialize the neural storage layer:

```sql
-- Enable the pgvector extension for high-dimensional search
create extension if not exists vector;

-- Create the neural memory matrix with halfvec for efficiency
create table if not exists public.memories (
  id bigint primary key generated always as identity,
  content text not null,          -- Raw neural content
  embedding vector(1536),        -- Neural vector (text-embedding-3-small)
  metadata jsonb,                -- Engine & session metadata
  source text,                   -- Uplink source identifier
  category text default 'general',-- Memory category
  importance real default 0.5,   -- Memory importance weight (0.0-1.0)
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- HNSW index for fast approximate nearest neighbor search (recommended for production)
create index on public.memories using hnsw (embedding vector_cosine_ops) 
  with (m = 16, ef_construction = 64);

-- Create the user profiles matrix
create table if not exists public.profiles (
  id uuid primary key default gen_random_uuid(),
  category text not null,        -- 'role', 'preference', 'setting', 'persona'
  content text not null,         -- Profile description
  embedding vector(1536),       -- Neural vector
  metadata jsonb,               -- Versioning & source
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- HNSW index for profiles
create index on public.profiles using hnsw (embedding vector_cosine_ops)
  with (m = 16, ef_construction = 64);

-- Create the AI instructions matrix
create table if not exists public.instructions (
  id uuid primary key default gen_random_uuid(),
  category text not null,        -- 'tone', 'workflow', 'rule', 'naming'
  content text not null,         -- Instruction detail
  embedding vector(1536),       -- Neural vector
  metadata jsonb,               -- Versioning & source
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- HNSW index for instructions
create index on public.instructions using hnsw (embedding vector_cosine_ops)
  with (m = 16, ef_construction = 64);

-- Create the nsfw matrix (Encrypted Sensitive Memories)
create table if not exists public.nsfw_memories (
  id bigint primary key generated always as identity,
  content text not null,          -- Encrypted neural content (AES-256)
  embedding vector(1536),        -- Neural vector (unencrypted for search)
  source text,                   -- Uplink source identifier
  category text default 'general',-- Memory category
  importance real default 0.5,   -- Memory importance weight (0.0-1.0)
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- HNSW index for nsfw_memories
create index on public.nsfw_memories using hnsw (embedding vector_cosine_ops)
  with (m = 16, ef_construction = 64);

-- Create the encrypted vault table
create table if not exists public.vault (
  key text primary key,          -- Unique variable name
  value text not null,           -- Encrypted content (AES-256)
  updated_at timestamptz default now()
);
```

### 2. Environment Configuration
Required variables for the neural uplink:
- `APIYI_TOKEN`: Authorization for the Neural Encoding API ([apiyi.com](https://api.apiyi.com))
- `SUPABASE_PASSWORD`: Credentials for the PostgreSQL Knowledge Base

## 🧠 Engineering Principles
- **Dimensionality**: 1536-D Vector Space (text-embedding-3-small)
- **Indexing**: HNSW (Hierarchical Navigable Small World) for production-grade performance
- **Search**: Cosine similarity with configurable thresholds (default: 0.6)
- **Protocol**: HTTPS / WebSockets (via Psycopg2)
- **Latency**: Optimized for real-time sub-second synchronization
- **Context Prioritization**: `instructions` > `profiles` > `memories`

## 📚 Documentation
- **Quick Start**: `QUICK_START.md` - Get started in 5 minutes
- **CLI Usage**: `CLI_USAGE_EXAMPLES.md` - Comprehensive examples
- **API Reference**: `VECTOR_SEARCH_API.md` - Complete API docs
- **Threshold Guide**: `DEFAULT_THRESHOLD_RECOMMENDATION.md` - Tuning guide

## ⚠️ Notes
- Ensure `psycopg2` and `requests` are present in the host environment
- Always `fetch-full-context` at the start of a mission to align with historical objectives
- Perform a `sync` upon milestone completion to ensure neural persistence
- Use vector search (query_text) for intelligent semantic queries
- Adjust similarity threshold based on your needs (0.6 = balanced, 0.7 = precise, 0.5 = exploratory)
