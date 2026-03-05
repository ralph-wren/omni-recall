# Omni-Recall: Neural Knowledge & Long-Term Context Engine

> **Elevate your AI Agent with persistent neural memory and intelligent semantic search.**

Omni-Recall is a standardized Agent Skill for Trae, Cursor, and other AI coding assistants. It provides a robust, vector-based long-term memory system using Supabase (PostgreSQL + pgvector) and APIYI (Embeddings).

## 🌟 Features

- **Vector Semantic Search**: Intelligent natural language queries with similarity scoring
- **Cross-Session Memory**: Never lose context between chat restarts
- **Neural Embeddings**: High-dimensional vector storage (1536-dim) for semantic understanding
- **Tri-Matrix Architecture**: Separate storage for memories, profiles, and instructions
- **HNSW Indexing**: Production-grade approximate nearest neighbor search
- **Automated Sync**: Easily integrate memory synchronization into your agent's workflow
- **Enterprise Ready**: Built on PostgreSQL with ACID compliance and point-in-time recovery

## 📦 Installation

To install **Omni-Recall** in your current project, run the following command in your terminal:

```bash
npx skills add ralph-wren/omni-recall
```

## ⚙️ Configuration

1. **Database Setup**: Run the SQL script found in `SKILL.md` in your Supabase SQL editor.
2. **Environment Variables**:
   - `APIYI_TOKEN`: Your API key from [apiyi.com](https://api.apiyi.com)
   - `SUPABASE_PASSWORD`: Your database password

## 🧠 Core Methodology

Omni-Recall operates on a **Tri-Matrix Architecture**:
1. **`memories`**: Temporal session logs (What happened)
2. **`profiles`**: Persistent user identity (Who you are)
3. **`instructions`**: Behavioral constraints (How I behave)

**CRITICAL**: Always prefer `fetch-full-context` over simple `fetch` to ensure the AI is fully aligned with your persona, rules, and history.

## 🛠 Usage Examples

### 1. Full Context Realignment (Recommended)
```bash
# Retrieve ALL Profiles + ALL Instructions + Memories from last 7 days
python3 scripts/omni_ops.py fetch-full-context 7
```

### 2. Vector Semantic Search (Command Line)
```bash
# Search memories with natural language query (default threshold: 0.6)
python3 scripts/omni_ops.py fetch "如何优化数据库性能" none 10
# Parameters: <query_text> [days] [limit] [category] [similarity_threshold]

# Search with custom similarity threshold
python3 scripts/omni_ops.py fetch "pgvector 索引优化" none 10 none 0.7

# Search instructions
python3 scripts/omni_ops.py fetch-instruction "代码风格规范" none 0.6 5
# Parameters: <query_text> [category] [similarity_threshold] [limit]

# Search profiles
python3 scripts/omni_ops.py fetch-profile "用户技能背景" none 0.6 5
# Parameters: <query_text> [category] [similarity_threshold] [limit]

# List all records without filtering (use 'none' as query)
python3 scripts/omni_ops.py fetch none 30 10
```

### 3. Semantic Vector Search (Python API)
```python
from scripts.omni_ops import OmniRecallManager

manager = OmniRecallManager()

# Vector-based semantic search - finds similar content by meaning
results = manager.fetch(
    query_text="如何优化数据库查询性能",  # Natural language query
    similarity_threshold=0.6,  # Default: 0.6 (recommended)
    limit=10,
    days=30
)

# Results include similarity scores
for content, created_at, source, metadata, category, importance, similarity in results:
    print(f"Similarity: {similarity:.3f} - {content[:100]}...")

# Search with different thresholds
instructions = manager.fetch_instruction(
    query_text="代码风格规范",
    similarity_threshold=0.6  # 0.6 = balanced, 0.7 = precise, 0.5 = exploratory
)
```

### 4. Manual Synchronization
```bash
# Sync session state
python3 scripts/omni_ops.py sync "Detailed summary" "session_tag"

# Sync user profile (Role/Preference)
python3 scripts/omni_ops.py sync-profile "persona" "Senior AI Engineer"

# Sync AI instructions (Tone/Rules)
python3 scripts/omni_ops.py sync-instruction "tone" "Professional and gentle"
```

### 5. Batch Sync (Files & URLs)
```bash
# Automatically split markdown by H1-H5 headers
python3 scripts/omni_ops.py batch-sync-doc "docs/spark_optimization.md"

# Sync web pages directly via URL
python3 scripts/omni_ops.py batch-sync-doc "https://clickhouse.com/docs/en/optimize"
```

## 🔍 Similarity Threshold Guide

| Threshold | Description | Use Case |
|-----------|-------------|----------|
| **0.6** | **Balanced (Default)** ⭐ | General search, best for most cases |
| 0.7-0.8 | High precision | Exact matches, specific queries |
| 0.5 | Exploratory | Broad search, discover related content |
| 0.8+ | Very precise | Almost exact matches only |

**Why 0.6?** Based on real-world testing, Chinese semantic search typically scores 0.55-0.65 for highly relevant content. Using 0.7 would filter out many relevant results.

## 🎯 Vector Search vs Keyword Search

### Vector Search (Default - Recommended)
- ✅ **Semantic understanding** - finds similar meaning, not just exact words
- ✅ **Language-agnostic** - works across different phrasings
- ✅ **Similarity scores** - ranked results by relevance
- ✅ **Natural language** - query like you speak

**Example:**
```
Query: "如何让数据库跑得更快"
Finds: "PostgreSQL 性能优化" ✅
       "数据库索引优化技巧" ✅
       "SQL 查询加速方法" ✅
```

### Legacy Keyword Search (Deprecated)
- ⚠️ Exact text matching only
- ⚠️ Must contain all specified keywords
- ⚠️ No semantic understanding
- Only available via Python API with `keywords` parameter

## 📚 Documentation

- **Quick Start**: `QUICK_START.md` - Get started in 5 minutes
- **CLI Usage**: `CLI_USAGE_EXAMPLES.md` - Comprehensive command-line examples
- **API Reference**: `VECTOR_SEARCH_API.md` - Complete API documentation
- **Search Comparison**: `SEARCH_FLOW_COMPARISON.md` - Vector vs keyword search
- **Threshold Guide**: `DEFAULT_THRESHOLD_RECOMMENDATION.md` - Similarity threshold tuning
- **Upgrade Summary**: `UPGRADE_SUMMARY.md` - What's new in vector search

## 🚀 Quick Examples

```bash
# Find content about database optimization
python3 scripts/omni_ops.py fetch "数据库优化" none 10

# Search last 7 days for AI-related content
python3 scripts/omni_ops.py fetch "AI Agent 开发" 7 10

# Precise search with higher threshold
python3 scripts/omni_ops.py fetch "pgvector HNSW" none 5 none 0.75

# Exploratory search with lower threshold
python3 scripts/omni_ops.py fetch "机器学习" none 20 none 0.5
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT
