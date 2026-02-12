# Omni-Recall: Neural Knowledge & Long-Term Context Engine

> **Elevate your AI Agent with persistent neural memory.**

Omni-Recall is a standardized Agent Skill for Trae, Cursor, and other AI coding assistants. It provides a robust, vector-based long-term memory system using Supabase (PostgreSQL + pgvector) and APIYI (Embeddings).

## 🌟 Features

- **Cross-Session Memory**: Never lose context between chat restarts.
- **Neural Embeddings**: High-dimensional vector storage for semantic search.
- **Automated Sync**: Easily integrate memory synchronization into your agent's workflow.
- **Enterprise Ready**: Built on top of PostgreSQL and standardized Agent Skill specifications.

## 📦 Installation

To install **Omni-Recall** in your current project, run the following command in your terminal:

```bash
npx skills add ralph-wren/omni-recall
```

## ⚙️ Configuration

1. **Database Setup**: Run the SQL script found in `SKILL.md` in your Supabase SQL editor.
2. **Environment Variables**:
   - `APIYI_TOKEN`: Your API key from [apiyi.com](https://api.apiyi.com).
   - `SUPABASE_PASSWORD`: Your database password.

## 🧠 Core Methodology

Omni-Recall operates on a **Tri-Matrix Architecture**:
1.  **`memories`**: Temporal session logs (What happened).
2.  **`profiles`**: Persistent user identity (Who you are).
3.  **`instructions`**: Behavioral constraints (How I behave).

**CRITICAL**: Always prefer `fetch-full-context` over simple `fetch` to ensure the AI is fully aligned with your persona, rules, and history.

## 🛠 Usage Examples

### 1. Full Context Realignment (Recommended)
```bash
# Retrieve ALL Profiles + ALL Instructions + Memories from last 7 days
# (Profiles and Instructions have NO time/limit constraints)
python3 scripts/omni_ops.py fetch-full-context 7
```

### 2. Manual Synchronization
```bash
# Sync session state
python3 scripts/omni_ops.py sync "Detailed summary" "session_tag"

# Sync user profile (Role/Preference)
python3 scripts/omni_ops.py sync-profile "persona" "Senior AI Engineer"

# Sync AI instructions (Tone/Rules)
python3 scripts/omni_ops.py sync-instruction "tone" "Professional and gentle"
```

### 3. Batch Sync (Files & URLs)
```bash
# Automatically split markdown by H1-H5 headers
python3 scripts/omni_ops.py batch-sync-doc "docs/spark_optimization.md"

# Sync web pages directly via URL
python3 scripts/omni_ops.py batch-sync-doc "https://clickhouse.com/docs/en/optimize"
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT
