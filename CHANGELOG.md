# Changelog

## [2.0.2] - 2026-03-05

### Changed
- **Default similarity threshold changed from 0.6 to 0.5**
- Improved recall: returns more results (average 3-10 vs 1-2)
- Better support for both phrase queries and single-word queries
- Updated all documentation to reflect new default

### Rationale
- Threshold 0.5 provides better balance between precision and recall
- Works well for diverse query types (phrases and single words)
- Users can still increase threshold for more precise results
- More forgiving for exploratory searches

## [2.0.1] - 2026-03-05

### Analysis
- Completed comprehensive similarity threshold testing
- Analyzed query patterns: phrase queries vs single-word queries
- Confirmed 0.6 as optimal default threshold for phrase queries
- Documented query best practices

### Documentation
- Added `THRESHOLD_ANALYSIS.md` with detailed test results
- Updated threshold guidance in README.md and SKILL.md
- Added query best practices (phrase queries recommended)
- Clarified that single-word queries need lower threshold (0.5)

### Key Findings
- Phrase queries (3-5 words): similarity 0.60-0.66 ✅
- Single-word queries: similarity 0.53-0.57 (need threshold 0.5)
- **Default changed to 0.5** for better recall (3-10 results vs 1-2)
- 0.55 returns 3.5 results (exploratory), 0.65 returns 0.2 results (precise)

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
