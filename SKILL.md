---
name: "omni-recall"
description: "Omni-Recall: Neural Knowledge & Long-Term Context Engine. Manages cross-session agent memory via Supabase (pgvector) and APIYI. Invoke to store/retrieve persistent context."
---

# Omni-Recall: Neural Knowledge & Long-Term Context Engine

Omni-Recall is a high-performance memory management skill designed for AI agents. It enables persistent, cross-session awareness by transforming conversation history and technical insights into high-dimensional vector embeddings, stored in a decentralized Supabase (PostgreSQL + pgvector) knowledge cluster.

## 🚀 Core Capabilities

1.  **Neural Synchronization (`sync`)**:
    Encodes current session state, user preferences, and operational steps into 1536-dimensional vectors using OpenAI's `text-embedding-3-small` via APIYI. **Includes automatic duplicate detection** (skips if cosine similarity > 0.9).
2.  **Contextual Retrieval (`fetch`)**:
    Pulls historical neural records from the last N days to re-establish the agent's mental model and context. Supports optional multiple keyword filtering (AND logic).
3.  **User Profile Management (`sync-profile` / `fetch-profile`)**:
    Manages user roles, preferences, settings, and personas in a dedicated `profiles` matrix. Unlike `memories`, this table stores stable personal attributes rather than session logs.
4.  **AI Instruction Management (`sync-instruction` / `fetch-instruction`)**:
    Stores operational requirements for the AI, such as response tone, nickname, attitude, and mandatory working steps in the `instructions` table.

---

## 🛠 Usage Examples

### Synchronize Session Context
```bash
python3 scripts/omni_ops.py sync "User is interested in Python optimization." "session-tag" 0.9
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

### Batch Synchronize Document (Multi-level Header Splitting)
```bash
# Sync a markdown file, splitting it by headers (H1-H5)
# Parameters: <file_path> [source_tag] [threshold]
python3 scripts/omni_ops.py batch-sync-doc "/path/to/doc.md" "tech-stack" 0.9
```

### Fetch Full Context (Identity + Behavior + Recent History)， use this when first time to recall
```bash
# Get ALL profiles + ALL instructions + memories from last 10 days
# (Profiles and Instructions are always fully retrieved regardless of 'days' parameter)
python3 scripts/omni_ops.py fetch-full-context 10
```


### Fetch History (Context Recall)
```bash
# Last 30 days, no limit, keywords "Python" and "optimization"
python3 scripts/omni_ops.py fetch 30 none "Python" "optimization"
```

### Fetch Profiles (Identity Recall)
```bash
# Get all 'preference' category profiles
python3 scripts/omni_ops.py fetch-profile "preference"
```

### Fetch Instructions (Behavior Recall)
```bash
# Get all 'workflow' category instructions
python3 scripts/omni_ops.py fetch-instruction "workflow"
```

---

## 🏗 Schema Setup (Supabase / Postgres)

### 1. Supabase Knowledge Cluster
Execute the following SQL in your Supabase project to initialize the neural storage layer:

```sql
-- Enable the pgvector extension for high-dimensional search
create extension if not exists vector;

-- Create the neural memory matrix
create table if not exists public.memories (
  id uuid primary key default gen_random_uuid(),
  content text not null,          -- Raw neural content
  embedding vector(1536),        -- Neural vector (text-embedding-3-small)
  metadata jsonb,                -- Engine & session metadata
  source text,                   -- Uplink source identifier
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- Optimized index for cosine similarity search
create index on public.memories using ivfflat (embedding vector_cosine_ops);

-- Create the user profiles matrix (Roles, Preferences, Personas)
create table if not exists public.profiles (
  id uuid primary key default gen_random_uuid(),
  category text not null,        -- 'role', 'preference', 'setting', 'persona'
  content text not null,         -- Profile description
  embedding vector(1536),       -- Neural vector
  metadata jsonb,               -- Versioning & source
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- Index for profiles similarity search
create index on public.profiles using ivfflat (embedding vector_cosine_ops);

-- Create the AI instructions matrix (Tone, Workflow, Rules)
create table if not exists public.instructions (
  id uuid primary key default gen_random_uuid(),
  category text not null,        -- 'tone', 'workflow', 'rule', 'naming'
  content text not null,         -- Instruction detail
  embedding vector(1536),       -- Neural vector
  metadata jsonb,               -- Versioning & source
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- Index for instructions similarity search
create index on public.instructions using ivfflat (embedding vector_cosine_ops);
```

### 2. Environment Configuration
Required variables for the neural uplink:
- `APIYI_TOKEN`: Authorization for the Neural Encoding API ([apiyi.com](https://api.apiyi.com)).
- `SUPABASE_PASSWORD`: Credentials for the PostgreSQL Knowledge Base.

## 🧠 Engineering Principles
- **Dimensionality**: 1536-D Vector Space.
- **Protocol**: HTTPS / WebSockets (via Psycopg2).
- **Latency**: Optimized for real-time sub-second synchronization.

## ⚠️ Notes
- Ensure `psycopg2` and `requests` are present in the host environment.
- Always `fetch` at the start of a mission to align with historical objectives.
- Perform a `sync` upon milestone completion to ensure neural persistence.
