# Changelog

## [2.0.0] - 2026-03-03

### 🎉 Major Update: Vector Semantic Search

#### Added
- **Vector Semantic Search**: All `fetch` commands now support intelligent natural language queries
- **Similarity Scoring**: Results include similarity scores (0-1) for relevance ranking
- **Configurable Thresholds**: Adjustable similarity thresholds (default: 0.6)
- **HNSW Indexing**: Production-grade approximate nearest neighbor search support

#### Changed
- **Default Similarity Threshold**: Changed from 0.7 to 0.6 based on real-world testing
- **Command Line Interface**: Updated all fetch commands to use `query_text` parameter
- **Return Format**: Vector search results include additional `similarity` field

#### Command Format Changes

**Before (v1.x):**
```bash
python3 scripts/omni_ops.py fetch 30 10 none keyword1 keyword2
```

**After (v2.0):**
```bash
python3 scripts/omni_ops.py fetch "natural language query" 30 10 none 0.6
```

#### New Features

1. **Semantic Understanding**
   - Finds content by meaning, not just exact words
   - Works across different phrasings
   - Language-agnostic search

2. **Similarity Threshold Tuning**
   - 0.6: Balanced (default, recommended)
   - 0.7-0.8: High precision
   - 0.5: Exploratory search
   - 0.8+: Very precise

3. **Enhanced Return Values**
   - `fetch()`: 7 fields (added similarity)
   - `fetch_instruction()`: 4 fields (added similarity)
   - `fetch_profile()`: 4 fields (added similarity)
   - `fetch_nsfw()`: 6 fields (added similarity)

#### Backward Compatibility

- Python API still supports `keywords` parameter for legacy keyword search
- Command line interface fully migrated to vector search
- Use `query_text='none'` to list all records without filtering

#### Performance

- Vector search requires embedding API call (slight latency increase)
- HNSW indexing provides fast approximate nearest neighbor search
- Recommended for production use with proper index configuration

#### Documentation

- Updated README.md with vector search examples
- Updated SKILL.md with new command formats
- Added similarity threshold recommendations

### Migration Guide

**Python API:**
```python
# Old (still works)
results = manager.fetch(keywords=["optimization"], days=30)

# New (recommended)
results = manager.fetch(query_text="optimization", days=30, similarity_threshold=0.6)
```

**Command Line:**
```bash
# Old (deprecated)
python3 scripts/omni_ops.py fetch 30 10 none optimization

# New
python3 scripts/omni_ops.py fetch "optimization" 30 10 none 0.6
```

---

## [1.0.0] - 2026-02-11

### Initial Release
- Neural synchronization with duplicate detection
- Tri-Matrix Architecture (memories, profiles, instructions)
- Encrypted NSFW memory support
- Encrypted vault for key-value storage
- Batch document synchronization
- Full context retrieval
