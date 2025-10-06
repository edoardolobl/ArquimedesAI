# Changelog

All notable changes to ArquimedesAI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Phase 1 (v1.4): Conversational Memory**
  - New `enable_conversation_memory` setting (default: false) to enable in-session conversation history
  - New `max_history_messages` setting (default: 20) to control conversation history size
  - Added `RunnableWithMessageHistory` integration in `core/rag_chain.py`
  - New `history_store` dict for session-based chat history storage
  - New `get_session_history()` method for managing session histories
  - New `create_conversational_chain()` method wrapping retrieval chain with message history
  - CLI chat now supports `--conversational` / `-c` flag to enable conversational mode
  - Session tracking with UUID for conversation isolation
  - Turn counter displays session stats on exit

- **Phase 1 (v1.4): Structured Citations Foundation**
  - New Pydantic schemas in `prompts/templates.py`:
    - `Citation`: Source ID + verbatim quote model
    - `QuotedAnswer`: Answer with list of citations
    - `CitedAnswer`: Lighter schema with source IDs only
  - New `format_docs_with_id()` helper function for citation tracking
  - New `use_structured_citations` setting (default: false) for Pydantic schema-based citations
  - New `citation_style` setting ("quoted" or "id") to control citation format
  - `core/llm_local.py` migrated from `OllamaLLM` to `ChatOllama` for `.with_structured_output()` support
  - Foundation prepared for future `.with_structured_output(QuotedAnswer)` integration

- **Configuration & Debugging**
  - Debug logging added to `settings.py` on module import to display loaded configuration
  - Logs Ollama model, base URL, conversation memory, structured citations, paths on startup
  - Helpful for troubleshooting `.env` loading issues

### Changed
- **Documentation: Gemma2 → Gemma3 Model Upgrade**
  - Updated all references from Gemma2:1b (~500MB, 1B params) to Gemma3:4b (~3GB, 4B params)
  - Changed default `ollama_model` in `settings.py`: `gemma2:1b` → `gemma3:latest`
  - Updated 47+ occurrences across 21 files:
    - Core docs: `README.md`, `.env.example`, `.github/copilot-instructions.md`
    - Code comments: `core/llm_local.py`, `settings.py`
    - Setup guides: `QUICKSTART.md`, `SETUP.md`, `MIGRATION.md`, `ARCHITECTURE.md`
    - Reference docs: `CHANGELOG.md`, `REFACTORING_SUMMARY.md`
    - Serena MCP memories: `.serena/memories/{tech_stack.md, suggested_commands.md, project_overview.md}`
  - Model pull commands updated: `ollama pull gemma3:latest` (was `ollama pull gemma2:1b`)
  - Model size comments updated: ~3GB (was ~500MB)
  - Created `GEMMA3_DOCUMENTATION_UPDATE.md` documenting all changes

- **Configuration Files: v1.4 Features Added**
  - `.env.example`: Added conversational memory and structured citations settings (disabled by default for dev)
  - `.env.production`: Updated to `gemma3:latest` + added v1.4 features (enabled for production)
  - Both files now include Phase 1 usage notes and examples

- **CLI Enhancements**
  - `cli.py chat` command now displays conversational mode status in header
  - Session ID shown when conversational mode is active (first 8 chars)
  - Turn count displayed on exit in conversational mode
  - Better user feedback for mode selection

- **RAG Chain Architecture**
  - `core/rag_chain.py` now supports optional `use_structured_output` parameter
  - Docstrings updated to document v1.4 features
  - Better separation between single-turn and conversational modes

### Fixed
- **LLM Integration**
  - Migrated from `OllamaLLM` to `ChatOllama` in `core/llm_local.py` to support structured output
  - Maintains 100% offline operation and backward compatibility
  - Zero API changes, drop-in replacement

### Documentation
- Created `GEMMA3_DOCUMENTATION_UPDATE.md` with comprehensive change tracking
- Created `ENV_LOADING_FIX.md` documenting Python bytecode cache troubleshooting
- Created `PHASE1_CONFIGURATION_COMPLETE.md` verifying v1.4 configuration
- Created `PHASE1_v1.4_IMPLEMENTATION.md` documenting implementation details
- Updated `.gitignore` to exclude:
  - Development documentation files (spec, migration guides, implementation notes)
  - Helper scripts (`show_*.py`, `verify_*.py`, `test_*.py`)
  - Backup files (`*.old`, `*.broken`, `*.corrupted`, `*.clean`, `*.new`)
  - README backups (`README.md.*` pattern)
  - `.serena/` directory (MCP configuration and memories)

### Notes
- **Phase 1 Implementation Status**: Foundation complete, ready for testing
  - Conversational memory: Implemented and functional with `--conversational` flag
  - Structured citations: Schemas defined, LLM ready, chain integration pending
- **Breaking Changes**: None - all v1.4 features are opt-in
- **Migration**: No action required - defaults maintain v1.3.1 behavior
- **Testing**: Use `python cli.py chat --conversational` to test conversational mode
- **Model Upgrade**: Run `ollama pull gemma3:latest` to upgrade from Gemma2:1b

## [1.3.1] - 2025-10-06

### Added
- **Qdrant HNSW Optimization** (Significant Quality Improvement)
  - Configurable HNSW parameters for better retrieval accuracy
  - New settings in `settings.py`: `qdrant_hnsw_m`, `qdrant_hnsw_ef_construct`, `qdrant_on_disk`
  - Production defaults: `m=32`, `ef_construct=256` for "High Precision + Low Memory" strategy
  - Development defaults: `m=16`, `ef_construct=100` for balanced speed/quality
  - `on_disk=true` for production: lower memory footprint (good for 8-16GB RAM target)
  - `on_disk=false` for development: faster queries during testing
  - Updated `core/vector_store.py` to use `HnswConfigDiff` on collection creation

- **Image Format Documentation**
  - Clarified image support (PNG, JPG, JPEG, TIFF, BMP) in `ingest/loaders.py`
  - Images automatically processed with OCR when `ARQ_DOCLING_OCR=true`
  - Noted `ImageFormatOption` as future enhancement for advanced image settings

### Changed
- **LangChain 1.0 Compatibility** (Deprecation Fixes)
  - **Ollama LLM Migration**: Updated `core/llm_local.py` to use `langchain-ollama` package
    - Changed import: `langchain_community.llms.Ollama` → `langchain_ollama.OllamaLLM`
    - Added `langchain-ollama>=0.3.10` to `requirements.txt`
    - Zero API changes, drop-in replacement maintaining 100% offline operation
  - **BaseRetriever Method Migration**: Updated `core/hybrid_retriever.py`
    - Changed `.get_relevant_documents()` → `.invoke()` (sync)
    - Changed `.aget_relevant_documents()` → `.ainvoke()` (async)
    - Migrated `run_manager` parameter to `config={"callbacks": ...}` pattern
  - **Embeddings Cache Security**: Updated `core/embedder.py`
    - Changed cache key encoder from SHA-1 to SHA-256 for collision resistance
    - Added `key_encoder="sha256"` to `CacheBackedEmbeddings.from_bytes_store()`
    - ⚠️ Note: Existing SHA-1 cache will be ignored, cache will rebuild on next indexing

- **Configuration Files Cleanup**
  - `.env.example`: Removed deprecated settings, added HNSW optimization (dev/speed-focused)
  - `.env.production`: Removed deprecated settings, added HNSW optimization (prod/quality-focused)
  - Updated performance notes: removed outdated semantic chunking reference
  - Clarified Docling HybridChunker as current chunking method

### Removed
- **Deprecated Settings Purged**
  - Removed `ARQ_SEMANTIC_CHUNKING` from all `.env` files (no backward compatibility needed)
  - Removed `ARQ_SEMANTIC_BREAKPOINT_TYPE` from all `.env` files
  - Removed `ARQ_CHUNK_SIZE`, `ARQ_CHUNK_OVERLAP` from production config
  - Updated `settings.py`: marked `chunk_size`, `chunk_overlap` as DEPRECATED
  - Removed `semantic_chunking`, `semantic_breakpoint_type` fields entirely

### Fixed
- ✅ All LangChain deprecation warnings resolved (future-proof for LangChain 1.0)
- ✅ Ollama integration now uses official `langchain-ollama` partner package
- ✅ BaseRetriever follows LCEL `.invoke()` pattern
- ✅ Embeddings cache uses secure SHA-256 key encoder

### Migration Notes
- **Breaking**: If using `.env` file, remove deprecated semantic chunking settings
- **Action**: Install new dependency: `pip install -U langchain-ollama`
- **Action**: Add new Qdrant HNSW settings to your `.env` (see `.env.example` or `.env.production`)
- **Recommended**: Rebuild index for HNSW optimization and SHA-256 cache: `python cli.py index --rebuild`
- Old indexes continue to work but won't have HNSW benefits until rebuilt
- SHA-1 → SHA-256 cache migration: Existing cache will be ignored (not corrupted), re-indexing rebuilds with new keys

### Documentation & Cleanup
- **Added**: `DEPRECATION_FIXES_v1.3.1.md` - Detailed documentation of LangChain 1.0 migration
- **Removed obsolete files**:
  - `ingest/chunking.py` (deprecated stub, no active code uses it)
  - `DEPRECATED.md` (checklist for already-deleted v0.2 files)
  - `.env.v1.2.reference` (superseded by `.env.production`)
- **Updated documentation to v1.3.1**:
  - `.github/copilot-instructions.md` - Added Docling HybridChunker, HNSW optimization patterns
  - `ARCHITECTURE.md` - Updated from v1.0 to v1.3.1 with full pipeline details
- **Added to `.gitignore`**:
  - Development documentation (spec, migration guides, testing notes)
  - Reference files (`*.reference` pattern)

## [1.3.0] - 2025-10-05

### Added
- **Docling HybridChunker Integration** (Major Quality Improvement)
  - Official `langchain-docling` integration replaces manual markdown export
  - Uses Docling's HybridChunker for structure-aware, tokenization-optimized chunking
  - Preserves document structure: headings, sections, tables, hierarchies
  - Rich metadata: page numbers, bounding boxes, section context for precise citations
  - Configurable OCR support via `ARQ_DOCLING_OCR=true` for scanned PDFs/images
  - Accurate table extraction with `TableFormerMode.ACCURATE` by default
  - New settings: `ARQ_DOCLING_OCR`, `ARQ_DOCLING_TABLE_MODE` in `settings.py`
  - Complete rewrite of `ingest/loaders.py` using `DoclingLoader` + `ExportType.DOC_CHUNKS`
  - Supersedes both RecursiveCharacterTextSplitter and SemanticChunker

- **Enhanced Citation Grounding**
  - Retrieved documents now include section headings in metadata
  - Page numbers and bounding box coordinates preserved
  - RAG answers can reference specific sections: "According to Section 3.2 on page 3..."
  - Better document provenance tracking

### Changed
- **Document Loading Pipeline Simplified**
  - Removed separate chunking step from `cli.py` index command
  - DoclingLoader handles both loading AND chunking in one step
  - CLI now shows "Step 1/3: Loading & chunking" (was 1/4 and 2/4 before)
  - Pipeline flow: Load+Chunk → Embed → Store (was: Load → Chunk → Embed → Store)

- **Settings Reorganization**
  - `ARQ_SEMANTIC_CHUNKING`: Marked DEPRECATED (use HybridChunker)
  - `ARQ_SEMANTIC_BREAKPOINT_TYPE`: Marked DEPRECATED
  - `chunk_size`, `chunk_overlap`: Retained as fallback but not actively used

### Deprecated
- **`ingest/chunking.py` - DocumentChunker class**
  - Replaced by Docling's HybridChunker (superior in all aspects)
  - Class retained for backward compatibility, displays warnings
  - `chunk_documents()` now passes through documents unchanged
  - HybridChunker is: hierarchical, tokenization-aware, structure-preserving, production-ready
  
- **Semantic Chunking Feature (v1.2)**
  - `ARQ_SEMANTIC_CHUNKING` still exists but not recommended
  - HybridChunker supersedes SemanticChunker (experimental → production)
  - Better context preservation through document hierarchy awareness

### Documentation
- **New**: `DOCLING_UPGRADE_v1.3.md` - Complete migration guide
- **Updated**: `.env.example` with Docling settings (OCR=false for dev)
- **Updated**: `.env.production` with Docling settings (OCR=true for prod)
- **Updated**: Copilot instructions reflect v1.3 architecture

### Migration Notes
- **Backward Compatible**: Old indexed documents still work (missing rich metadata)
- **Recommended**: Re-index with `python cli.py index --rebuild` for best results
- **OCR Performance**: Enable `ARQ_DOCLING_OCR=true` for scanned PDFs (~1.5-2x slower, worth it)
- **No Breaking Changes**: Existing code continues to work without modification

## [1.2.0] - 2025-10-05

### Added
- **Cross-Encoder Reranking** (Optional, Spec §8 enhancement)
  - Integration with HuggingFace cross-encoder models
  - Default model: `BAAI/bge-reranker-v2-m3` (multilingual support)
  - Uses LangChain ContextualCompressionRetriever pattern
  - Reranks top-K results based on query-document relevance
  - Configurable via `ARQ_RERANK_ENABLED=true`, `ARQ_RERANK_MODEL`, `ARQ_RERANK_TOP_N`
  - Significantly improves result relevance when enabled
  - New `core/reranker.py` module with RerankerManager class

- **Semantic Chunking** (Optional, experimental)
  - Embedding-based chunking using LangChain SemanticChunker
  - Splits text based on semantic similarity instead of character count
  - Better context preservation than character-based splitting
  - Configurable via `ARQ_SEMANTIC_CHUNKING=true`
  - Three breakpoint types: percentile, standard_deviation, interquartile
  - Uses existing BGE-M3 embeddings for consistency
  - Updated `ingest/chunking.py` to support both strategies

- **Multi-Mode CLI Chat** (`--mode` parameter)
  - Four chat modes available:
    - `grounded` (default): Detailed answers with explicit citations
    - `concise`: Brief 1-3 sentence answers
    - `critic`: Verify if context supports claims
    - `explain`: Show reasoning steps
  - Usage: `python cli.py chat --mode concise`
  - Maps to existing prompt templates from `prompts/templates.py`
  - Enables testing different prompt strategies easily

### Changed
- **RAGChain now supports optional reranking**
  - Wraps base retriever with ContextualCompressionRetriever when enabled
  - Reranking happens after hybrid retrieval, before LLM generation
  - Flow: Hybrid → Rerank → LLM (when both enabled)

- **DocumentChunker supports dual strategies**
  - Can use RecursiveCharacterTextSplitter (default) or SemanticChunker
  - Controlled by `use_semantic` parameter or settings
  - Graceful error handling with helpful messages

- **Settings expanded for v1.2 features**
  - `rerank_enabled`: Enable/disable reranking (default: False)
  - `rerank_model`: Cross-encoder model ID (default: BAAI/bge-reranker-v2-m3)
  - `rerank_top_n`: Final result count after reranking (default: 3)
  - `semantic_chunking`: Use semantic chunking (default: False)
  - `semantic_breakpoint_type`: Breakpoint algorithm (default: percentile)

### Dependencies
- Added `langchain-experimental>=0.3.0` for SemanticChunker
- Added `sentence-transformers>=2.2.0` for cross-encoder models
- All existing dependencies remain compatible

### Performance Notes
- **Reranking**: Adds ~100-300ms latency but significantly improves relevance
- **Semantic Chunking**: Slower indexing (~2x) but better context preservation
- Both features are opt-in (disabled by default) for backward compatibility

---

## [1.1.0] - 2025-10-05

### Added
- **CLI Chat Interface** (`python cli.py chat`)
  - Interactive terminal-based testing interface
  - Rich-formatted output for better readability
  - Support for exit commands (exit/quit/q)
  - Shows source document count for transparency
  
- **Hybrid Retrieval System** (Spec §8 requirement)
  - BM25 sparse retriever for keyword matching
  - Dense semantic retriever (BGE-M3 embeddings)
  - EnsembleRetriever combining both with Reciprocal Rank Fusion (RRF)
  - Configurable weights via `ARQ_BM25_WEIGHT` and `ARQ_DENSE_WEIGHT`
  - Default: 40% BM25, 60% Dense (proven effective balance)
  - Can disable hybrid mode via `ARQ_HYBRID=false` for dense-only

- **Enhanced Prompt Engineering**
  - Explicit citation format instructions in GROUNDED_PROMPT
  - Stronger refusal instructions to prevent hallucination
  - Numbered instruction format for clarity
  - Quote-based citation pattern ("quoted text")

- **Configuration Improvements**
  - `bm25_weight`: Control BM25 retriever weight (default: 0.4)
  - `dense_weight`: Control dense retriever weight (default: 0.6)
  - `hybrid`: Enable/disable hybrid retrieval (default: True)
  - All settings configurable via .env file

### Changed
- **RAGChain now uses prompt templates** from `prompts/templates.py`
  - Replaced hardcoded prompt with GROUNDED_PROMPT
  - Supports custom prompts via `prompt_template` parameter
  - Enables future multi-mode support (concise/critic/explain)

- **Retrieval architecture refactored**
  - New `core/hybrid_retriever.py` module
  - `create_hybrid_retriever()` function auto-selects based on settings
  - Graceful fallback to dense-only if BM25 fails
  - Loads documents from Qdrant using scroll API for BM25 indexing

### Removed
- **Deprecated v0.2 files** (replaced by modular architecture)
  - `main.py` → `cli.py`
  - `indexing.py` → `ingest/loaders.py` + `ingest/chunking.py`
  - `retrieval_generation.py` → `core/rag_chain.py`
  - `discord_chatbot.py` → `bots/discord_bot.py`

### Dependencies
- Added `rank-bm25>=0.2.2` for BM25 sparse retrieval

### Performance
- Hybrid retrieval improves recall by combining keyword and semantic search
- BM25 catches exact term matches that embeddings might miss
- Dense retriever handles synonyms and paraphrases
- RRF algorithm optimally combines both result sets

## [1.0.0] - 2025-10-04

### Added
- Complete architectural redesign per spec v1.1
- Modular directory structure (core/, ingest/, bots/, prompts/)
- Pydantic Settings for environment-based configuration
- Qdrant vector store with local persistence
- Docling document loader (PDF/DOCX/PPTX/XLSX/MD/HTML)
- BGE-M3 multilingual embeddings (PT/EN/ES)
- Ollama LLM integration (Gemma3:4b)
- Discord bot with async message handling
- Typer-based CLI (index/discord/status commands)
- Comprehensive documentation suite

### Changed
- Migrated from LangChain 0.1.0 to 0.3.0+ (LCEL patterns)
- Replaced FAISS (in-memory) with Qdrant (persistent)
- Replaced WebBaseLoader with Docling (multi-format support)
- Upgraded embeddings from distiluse to BGE-M3
- Modern dependencies (Pydantic v2, Discord.py 2.4+)

### Documentation
- README.md: Complete v1.0 architecture overview
- MIGRATION.md: v0.2 → v1.0 upgrade guide
- QUICKSTART.md: Step-by-step setup checklist
- ARCHITECTURE.md: Visual architecture diagrams
- REFACTORING_SUMMARY.md: Detailed refactoring report
- DEPRECATED.md: Old file cleanup guide
- .github/copilot-instructions.md: AI agent guidelines

## [0.2.0] - 2024-01-15 (Deprecated)

Initial monolithic implementation with basic RAG functionality.
