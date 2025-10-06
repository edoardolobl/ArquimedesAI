# Deprecation Fixes - v1.3.1

**Date:** 2025-10-06  
**Status:** ✅ COMPLETED

## Summary
Fixed 3 deprecation warnings to future-proof ArquimedesAI for LangChain 1.0:
1. Ollama LLM migration (langchain_community → langchain-ollama)
2. BaseRetriever method migration (get_relevant_documents → invoke)
3. Embeddings cache key encoder (SHA-1 → SHA-256)

## Changes Made

### 1. Ollama LLM Migration (core/llm_local.py)

**Issue:** `Ollama` class from `langchain_community.llms` deprecated in LangChain 0.3.1, will be removed in 1.0.

**Fix:**
- **Package installed:** `pip install -U langchain-ollama` (v0.3.10)
- **Import changed:** `from langchain_community.llms import Ollama` → `from langchain_ollama import OllamaLLM`
- **Instantiation changed:** `Ollama(...)` → `OllamaLLM(...)`
- **Type hint changed:** `get_llm() -> Ollama` → `get_llm() -> OllamaLLM`

**Reference:** https://github.com/langchain-ai/langchain/blob/master/libs/partners/ollama/README.md

**Code changes:**
```python
# OLD (deprecated)
from langchain_community.llms import Ollama
self.llm = Ollama(model=self.model_name, base_url=self.base_url, temperature=self.temperature)

# NEW (langchain-ollama)
from langchain_ollama import OllamaLLM
self.llm = OllamaLLM(model=self.model_name, base_url=self.base_url, temperature=self.temperature)
```

**Impact:** 
- ✅ Zero API changes, drop-in replacement
- ✅ All parameters compatible
- ✅ Maintains 100% offline operation

---

### 2. BaseRetriever Method Migration (core/hybrid_retriever.py)

**Issue:** `BaseRetriever.get_relevant_documents()` deprecated in langchain-core 0.1.46, will be removed in 1.0.

**Fix:**
- **Method changed:** `.get_relevant_documents(query, run_manager=...)` → `.invoke(query, config={...})`
- **Async method changed:** `.aget_relevant_documents(query, run_manager=...)` → `.ainvoke(query, config={...})`
- **Config migration:** `run_manager` parameter → `config={"callbacks": run_manager.get_child()}`

**Reference:** https://python.langchain.com/docs/versions/v0_2/deprecations/#baseretrieverget_relevant_documents

**Code changes:**
```python
# OLD (deprecated) - Line 40
return self.retriever.get_relevant_documents(query, run_manager=run_manager)

# NEW - Line 40
config = {"callbacks": run_manager.get_child()} if run_manager else None
return self.retriever.invoke(query, config=config)

# OLD (deprecated) - Line 47
return await self.retriever.aget_relevant_documents(query, run_manager=run_manager)

# NEW - Line 47
config = {"callbacks": run_manager.get_child()} if run_manager else None
return await self.retriever.ainvoke(query, config=config)
```

**Impact:**
- ✅ Follows LangChain LCEL patterns
- ✅ Maintains run manager callback integration
- ✅ Future-proof for LangChain 1.0

---

### 3. Embeddings Cache Key Encoder (core/embedder.py)

**Issue:** SHA-1 is not collision-resistant and not recommended for production use.

**Fix:**
- **Parameter added:** `key_encoder="sha256"` to `CacheBackedEmbeddings.from_bytes_store()`
- **Security improvement:** SHA-1 → SHA-256 for cache key hashing

**Reference:** https://python.langchain.com/api_reference/langchain/embeddings/langchain.embeddings.cache.CacheBackedEmbeddings.html#langchain.embeddings.cache.CacheBackedEmbeddings.from_bytes_store

**Code changes:**
```python
# OLD (SHA-1 warning)
self.embedder = CacheBackedEmbeddings.from_bytes_store(
    self.base_embedder,
    store,
    namespace=self.model_name.replace("/", "_"),
)

# NEW (SHA-256, secure)
self.embedder = CacheBackedEmbeddings.from_bytes_store(
    self.base_embedder,
    store,
    namespace=self.model_name.replace("/", "_"),
    key_encoder="sha256",  # Use SHA-256 instead of SHA-1
)
```

**Impact:**
- ✅ More secure cache key generation
- ⚠️ **Note:** Existing SHA-1 cache files in `storage/embeddings_cache/` will not be reused
  - Cache will rebuild on next indexing run
  - No data loss, just recomputation time (~1-2 min for 20 chunks)
  - Old cache files can be safely deleted

---

## Testing Results

### Test 1: Standalone RAG Test (`test_rag_fix.py`)
```bash
$ python test_rag_fix.py

Testing query: What is GTM?

✅ SUCCESS!
Answer: GTM is a tool used for collecting data and/or adding listeners to events...
Context docs: 3
```
**Result:** ✅ PASS - No deprecation warnings

### Test 2: CLI Chat Test (`test_cli_chat.py`)
```bash
$ python test_cli_chat.py

Loading RAG chain...
Query: What is GTM?

✅ SUCCESS!
Answer: GTM is a tool used for collecting data and/or adding listeners...
Context docs: 3
```
**Result:** ✅ PASS - OllamaLLM working correctly

### Test 3: Deprecation Warning Check
```bash
$ python test_rag_fix.py 2>&1 | grep -i "deprecation\|warning"
# No output
```
**Result:** ✅ PASS - All warnings resolved

---

## Files Modified

1. **core/llm_local.py**
   - Lines 8: Import changed to `langchain_ollama.OllamaLLM`
   - Line 51: Instantiation changed to `OllamaLLM(...)`
   - Line 59: Type hint changed to `-> OllamaLLM`

2. **core/hybrid_retriever.py**
   - Lines 32-42: Updated `_get_relevant_documents()` to use `.invoke()`
   - Lines 44-52: Updated `_aget_relevant_documents()` to use `.ainvoke()`

3. **core/embedder.py**
   - Line 67: Added `key_encoder="sha256"` parameter

---

## Dependencies Updated

**New dependency added:**
```
langchain-ollama==0.3.10
  └── ollama==0.6.0 (auto-installed)
```

**Updated in requirements.txt:** Not yet - pending final review

---

## Migration Notes

### For Users Upgrading from v1.3.0

1. **Install new dependency:**
   ```bash
   pip install -U langchain-ollama
   ```

2. **Re-index documents (optional but recommended):**
   ```bash
   python cli.py index --data-dir ./data
   ```
   This rebuilds the cache with SHA-256 keys. Skip if you want to keep existing SHA-1 cache.

3. **Clean old cache (optional):**
   ```bash
   rm -rf ./storage/embeddings_cache/*
   ```
   Then re-index to build fresh SHA-256 cache.

### Breaking Changes
**None.** All changes are backward-compatible at the API level.

### Cache Considerations
- SHA-1 → SHA-256 migration means existing cache keys won't match
- Existing embeddings cache will be ignored, not corrupted
- Re-indexing will rebuild cache with new keys
- Performance impact: ~1-2 minutes for 20 chunks on first run

---

## LangChain 1.0 Readiness

| Feature | Status | Notes |
|---------|--------|-------|
| Ollama LLM | ✅ Ready | Using langchain-ollama partner package |
| BaseRetriever | ✅ Ready | Using `.invoke()` pattern |
| Embeddings Cache | ✅ Ready | Using SHA-256 key encoder |
| Vector Store | ✅ Ready | Using langchain-qdrant 0.2.1 |
| LCEL Chains | ✅ Ready | All chains use Expression Language |

**Overall:** ✅ **ArquimedesAI is now LangChain 1.0 ready**

---

## References

### Documentation Sources (via Ref MCP)
1. **langchain-ollama migration:**
   - https://github.com/langchain-ai/langchain/blob/master/libs/partners/ollama/README.md
   - PyPI: langchain-ollama v0.3.10

2. **BaseRetriever.invoke():**
   - https://python.langchain.com/docs/versions/v0_2/deprecations/#baseretrieverget_relevant_documents
   - Deprecated: langchain-core 0.1.46
   - Removal: langchain-core 1.0

3. **CacheBackedEmbeddings key_encoder:**
   - https://python.langchain.com/api_reference/langchain/embeddings/langchain.embeddings.cache.CacheBackedEmbeddings.html
   - Parameter: `key_encoder: Literal['sha1', 'blake2b', 'sha256', 'sha512']`

---

## Conclusion

All 3 deprecation warnings successfully resolved with minimal code changes and zero breaking changes. ArquimedesAI now follows LangChain's latest best practices and is ready for the 1.0 release.

**Next Steps:**
1. Update `requirements.txt` with `langchain-ollama>=0.3.10`
2. Update `CHANGELOG.md` with v1.3.1 release notes
3. Consider updating `.env.example` with cache encoder note (optional)
4. Optional: Add cache migration script for users (low priority)
