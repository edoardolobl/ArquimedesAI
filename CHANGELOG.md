# Changelog

All notable changes to ArquimedesAI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-10-07

### Added

- **Semantic Routing System** (`core/prompt_router.py`)
  - Two-stage routing: keyword pre-filtering + semantic classification
  - Four routes: `general_chat`, `gtm_qa`, `gtm_generation`, `gtm_validation`
  - Hybrid routing: BM25 (70%) + BGE-M3 (30%) with alpha=0.3
  - Route confidence scoring and visual indicators (📚 Q&A, 🛠️ Generation, ✅ Validation, 💬 General)
  - 89.5% classification accuracy on test queries
  - CLI integration: `--enable-routing` / `-r` flag

- **GTM Domain Expertise** (`prompts/gtm_prompts.py`)
  - Domain-specific prompt templates for Google Tag Manager taxonomy
  - Three specialized modes:
    - **Q&A**: GTM taxonomy questions, platform expertise
    - **Generation**: Tag/trigger/variable creation with validation
    - **Validation**: Configuration review and auditing
  - 73 curated training utterances for route classification
  - Split architecture: base system prompts + LangChain template format
  - Brazilian Portuguese-first with technical term preservation

- **Base Prompt System** (`prompts/base_prompts.py`)
  - Reusable prompt components following prompt engineering best practices
  - `create_base_system_prompt()`: Template builder with role, constraints, output format
  - Language enforcement for Brazilian Portuguese
  - Style modifiers: `concise`, `detailed`, `explain_reasoning`, `critic`
  - `format_context_documents()`: Helper for context formatting
  - `apply_style_modifier()`: Dynamic prompt transformation

- **Conversational Memory** (v1.4 Phase 1)
  - Session-based chat history with `RunnableWithMessageHistory`
  - New `enable_conversation_memory` setting (default: false)
  - New `max_history_messages` setting (default: 20)
  - CLI flag: `--conversational` / `-c` for multi-turn conversations
  - Session tracking with UUID for conversation isolation
  - Turn counter displays session stats on exit
  - Compatible with routing and style modes

- **Structured Citations Foundation** (v1.4 Phase 1)
  - Pydantic schemas in `prompts/templates.py`:
    - `Citation`: Source ID + verbatim quote model
    - `QuotedAnswer`: Answer with list of citations
    - `CitedAnswer`: Lighter schema with source IDs only
  - `format_docs_with_id()`: Helper for citation tracking with source IDs
  - New `use_structured_citations` setting (default: false)
  - New `citation_style` setting ("quoted" or "id")
  - Foundation prepared for `.with_structured_output()` integration

- **CLI Enhancements** (`cli.py`)
  - Route-specific visual indicators (emojis) in output
  - Route confidence display: `[Route: 📚 gtm_qa (confidence: 0.95)]`
  - Conversational mode status in chat header
  - Session ID display when conversational mode active
  - Turn count on exit for conversational sessions
  - Mutual exclusivity enforcement: clear errors when combining incompatible flags
  - Updated docstrings with semantic routing and conversational documentation

### Changed

- **Model Upgrade**: Gemma2:1b → Gemma3:4b (Gemma3:latest)
  - Updated default `ollama_model` in `settings.py`
  - Updated 47+ references across 21 files
  - Model size: ~500MB → ~3GB
  - Parameters: 1B → 4B for better reasoning
  - Pull command: `ollama pull gemma3:latest`

- **LLM Integration** (`core/llm_local.py`)
  - Migrated from `OllamaLLM` to `ChatOllama`
  - Enables `.with_structured_output()` support for Pydantic schemas
  - Maintains 100% offline operation
  - Zero API changes, drop-in replacement

- **RAG Chain Architecture** (`core/rag_chain.py`)
  - Now supports optional `use_structured_output` parameter
  - Integrated `QueryRouter` for semantic routing (optional)
  - Added `create_conversational_chain()` for multi-turn conversations
  - `history_store` dict for session-based chat history
  - Better separation between single-turn and conversational modes
  - Routing-aware prompt selection logic

- **Configuration**
  - Debug logging in `settings.py` on module import
  - Logs Ollama model, base URL, conversation memory, structured citations, paths
  - Updated `.env.example` with v1.4 features (disabled by default for dev)
  - Updated `.env.production` with v1.4 features (enabled for production)
  - Reranker optimization: `ARQ_RERANK_TOP_N` 3 → 5 for better context utilization

### Testing

- **Routing Validation**
  - `test_routing_quick.py`: Validates all 4 routes with real queries
  - `test_semantic_routing.py`: Comprehensive routing accuracy tests
  - `test_cli_routing.py`: CLI integration with routing flag
  - `test_v2_complete.py`, `test_v2_final.py`: End-to-end v2.0 validation

- **Conversational Mode**
  - `test_conversational_and_styles.py`: Multi-turn conversation testing
  - 4-turn conversation with history accumulation verified
  - Routing works correctly across conversation turns
  - History preservation validated

### Migration Notes

- **No Breaking Changes**: All v2.0 features are opt-in
- **Routing**: Use `--enable-routing` or `-r` flag in CLI chat
- **Conversational**: Use `--conversational` or `-c` flag in CLI chat
- **Model Upgrade**: Run `ollama pull gemma3:latest` (recommended)
- **Compatibility**: v1.3.1 behavior maintained when new features disabled
- **Dependencies**: Install `semantic-router` for routing features

### Documentation

- Created `CLI_ROUTING_INTEGRATION_COMPLETE.md`: Full routing integration guide
- Created `PHASE1_v1.4_IMPLEMENTATION.md`: v1.4 Phase 1 implementation details
- Created `PHASE1_CONFIGURATION_COMPLETE.md`: v1.4 configuration verification
- Created `GEMMA3_DOCUMENTATION_UPDATE.md`: Model upgrade documentation
- Updated `.gitignore`: Exclude dev docs, test scripts, backup files, `.serena/`
- Updated README.md: v2.0 features, CLI examples, architecture updates
- Updated ARCHITECTURE.md: v2.0 routing layer, query pipeline, component dependencies

## [Unreleased]

### Notes
- Future features and improvements will be documented here

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
