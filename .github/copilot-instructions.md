# ArquimedesAI Copilot Instructions

## Project Overview
ArquimedesAI is a **100% open-source, self-hosted RAG (Retrieval Augmented Generation) chatbot** that runs locally on modest hardware (8-16GB RAM). It uses LangChain, Qdrant vector store, and local LLMs via Ollama to answer questions from user documents with grounded citations.

**Current State (v1.3.1)**: Production-ready architecture with Docling HybridChunker, HNSW-optimized Qdrant, hybrid retrieval (BM25 + dense), optional cross-encoder reranking, multi-mode CLI chat, enhanced prompts, and Discord bot.  
**Implementation Status**: Core spec requirements met. v1.3 adds Docling integration for structure-aware chunking. v1.3.1 adds Qdrant HNSW optimization for better accuracy and memory efficiency.

## Architecture & Data Flow

### Three-Layer Design
1. **Ingestion** (`ingest/`): Docling loaders → HybridChunker (structure-aware) → BGE-M3 embeddings → Qdrant
2. **Retrieval** (`core/`): Query → HybridRetriever (BM25 + Dense via RRF) → Optional Reranking → Top-K chunks
3. **Generation** (`core/`): Retrieved chunks + mode-specific prompt → Ollama LLM (Gemma3:4b) → Grounded answer
4. **Interfaces** (`bots/`, `cli.py`): Discord bot (async) or CLI chat with modes (sync) invoke RAG chain

```
User Query (CLI/Discord) → RAGChain → HybridRetriever → [BM25 + Qdrant] → RRF → Reranker (optional) → Ollama → Answer
```

### Key Components (v1.3.1)
- **`core/vector_store.py`**: QdrantVectorStore with HNSW optimization (m, ef_construct, on_disk)
- **`core/hybrid_retriever.py`**: HybridRetriever combining BM25Retriever + Qdrant via EnsembleRetriever
- **`core/reranker.py`**: RerankerManager with HuggingFace cross-encoder (bge-reranker-v2-m3)
- **`core/rag_chain.py`**: RAGChain using LangChain LCEL, supports custom prompts and reranking
- **`core/llm_local.py`**: LLMManager wrapping Ollama with Gemma3:4b
- **`core/embedder.py`**: EmbeddingManager with BGE-M3 + CacheBackedEmbeddings
- **`ingest/loaders.py`**: DocumentLoader using Docling with HybridChunker (structure-aware, tokenization-optimized)
- **`bots/discord_bot.py`**: DiscordChatbot with async edit-message pattern
- **`cli.py`**: Typer CLI with `index`, `discord`, `chat --mode`, `status` commands
- **`prompts/templates.py`**: GROUNDED_PROMPT, CONCISE_PROMPT, CRITIC_PROMPT, EXPLAIN_PROMPT
- **`settings.py`**: Pydantic Settings with `.env` support, all config centralized

## Critical Patterns & Conventions (v1.3.1)

### 1. LangChain LCEL (Expression Language)
All retrieval/generation pipelines use **LangChain Expression Language** chains:
```python
# Pattern: retriever → document_chain → retrieval_chain
from core.hybrid_retriever import create_hybrid_retriever
from core.reranker import create_reranking_retriever

base_retriever = create_hybrid_retriever(vector_store)  # BM25+Dense or dense-only
retriever = create_reranking_retriever(base_retriever)  # Optionally wrap with reranker
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)
response = retrieval_chain.invoke({"input": query})  # sync
response = await retrieval_chain.ainvoke({"input": query})  # async
```
**Never** manually iterate chunks or call LLM directly—use LCEL chains for composability.

### 2. Hybrid Retrieval Pattern (Spec §8)
**Enabled by default** (`ARQ_HYBRID=true`) for best results:
```python
# BM25 (sparse, keyword) + Dense (semantic) via Reciprocal Rank Fusion
from core.hybrid_retriever import HybridRetriever

hybrid = HybridRetriever(vector_store)  # Loads docs from Qdrant for BM25
retriever = hybrid.as_retriever()  # Returns EnsembleRetriever

# Weights configurable via settings (default: BM25=0.4, Dense=0.6)
```
- BM25 catches exact term matches
- Dense embeddings handle synonyms/paraphrases
- RRF combines results optimally
- Graceful fallback to dense-only if BM25 fails

### 3. Cross-Encoder Reranking Pattern (v1.2)
**Optional** feature that significantly improves relevance:
```python
from core.reranker import create_reranking_retriever

# Wrap any retriever with reranking
base_retriever = create_hybrid_retriever(vector_store)
retriever = create_reranking_retriever(base_retriever, enabled=True)

# Configuration
ARQ_RERANK_ENABLED=true           # Enable reranking
ARQ_RERANK_MODEL=BAAI/bge-reranker-v2-m3  # Cross-encoder model
ARQ_RERANK_TOP_N=3                # Final results after reranking
```
- Uses `ContextualCompressionRetriever` + `CrossEncoderReranker` pattern
- Fetches top-K candidates (e.g., 20), reranks to top-N (e.g., 3)
- Adds ~100-300ms latency but improves relevance significantly
- Requires `sentence-transformers` package

### 4. Docling HybridChunker Pattern (v1.3)
**Structure-aware chunking** replaces generic text splitting:
```python
from langchain_docling import DoclingLoader, ExportType
from docling.chunking import HybridChunker
from docling.datamodel.pipeline_options import PdfPipelineOptions

# Configure PDF pipeline (OCR + table extraction)
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = settings.docling_ocr
pipeline_options.do_table_structure = True

# Load documents with Docling
loader = DoclingLoader(
    file_path=files,
    export_type=ExportType.DOC_CHUNKS,  # Use HybridChunker
    chunker=HybridChunker(),
)
docs = loader.load()

# Configuration
ARQ_DOCLING_OCR=true           # Enable OCR for scanned PDFs/images
ARQ_DOCLING_TABLE_MODE=accurate  # TableFormerMode: fast or accurate
```
- Uses Docling's HybridChunker (hierarchical + tokenization-aware)
- Preserves document structure (headings, tables, sections)
- Better context than RecursiveCharacterTextSplitter or SemanticChunker
- Rich metadata: page numbers, bounding boxes, section context
- Supports PDF, DOCX, PPTX, XLSX, MD, HTML, images

### 5. Qdrant HNSW Optimization Pattern (v1.3.1)
**Optimized HNSW parameters** for better accuracy and memory efficiency:
```python
from qdrant_client.models import HnswConfigDiff, VectorParams, Distance

# Create collection with HNSW optimization
client.create_collection(
    collection_name="arquimedes_chunks",
    vectors_config=VectorParams(
        size=embedding_dimension,
        distance=Distance.COSINE,
        hnsw_config=HnswConfigDiff(
            m=settings.qdrant_hnsw_m,  # Graph connectivity (16-32-64)
            ef_construct=settings.qdrant_hnsw_ef_construct,  # Build quality (100-256-512)
        ),
        on_disk=settings.qdrant_on_disk,  # Memory vs speed tradeoff
    ),
)

# Configuration (production: quality-focused)
ARQ_QDRANT_HNSW_M=32              # Higher = better accuracy
ARQ_QDRANT_HNSW_EF_CONSTRUCT=256  # Higher = better index quality
ARQ_QDRANT_ON_DISK=true           # Lower memory (good for 8-16GB RAM)

# Configuration (development: speed-focused)
ARQ_QDRANT_HNSW_M=16
ARQ_QDRANT_HNSW_EF_CONSTRUCT=100
ARQ_QDRANT_ON_DISK=false          # Faster queries for testing
```
- "High Precision + Low Memory" strategy for production
- Better retrieval accuracy with optimized graph connectivity
- Lower RAM usage with on_disk storage
- Aligns with spec §16 (8-16GB RAM target)

### 6. Embedding Caching Pattern
Embeddings are expensive to compute. **Always** use `CacheBackedEmbeddings`:
```python
from core.embedder import EmbeddingManager

embedder = EmbeddingManager()  # Automatically uses cache
embeddings = embedder.get_embedder()  # BGE-M3 with LocalFileStore cache
```
Cache path: `./storage/embeddings_cache/` (gitignored).

### 7. Ollama Local LLM Integration
- LLM runs **locally via Ollama** HTTP API (default: `http://localhost:11434`)
- Current model: **Gemma3:4b** (gemma3:latest) (configurable via `ARQ_OLLAMA_MODEL`)
- Use `langchain_community.llms.Ollama` class (wrapped in `core/llm_local.py`)
- **Assumption**: Ollama service is running and model is pre-pulled (`ollama pull gemma3:latest`)

### 8. Async Discord Bot Pattern
- Bot uses `discord.py` with `Intents.all()` and `message_content=True`
- **Always** send "processing" message first, then edit with response (avoid timeouts)
- Chain invocation must be **async**: `await retrieval_chain.ainvoke(input_data)`

### 9. Prompt Engineering & Multi-Mode Chat (v1.2)
Use templates from `prompts/templates.py` for consistency:
- **GROUNDED_PROMPT**: Default, explicit citation format ("quote passages in quotation marks")
- **CONCISE_PROMPT**: Brief 1-3 sentence answers
- **CRITIC_PROMPT**: Verify if context supports claims
- **EXPLAIN_PROMPT**: Show reasoning steps

CLI chat now supports mode selection:
```bash
python cli.py chat --mode grounded  # Default
python cli.py chat --mode concise   # Brief answers
python cli.py chat --mode critic    # Verify claims
python cli.py chat --mode explain   # Show reasoning
```

RAGChain accepts optional `prompt_template` parameter for custom prompts.

## Development Workflows

### Setup & Running (v1.3.1)
```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama (separate terminal)
ollama serve
ollama pull gemma3:latest

# Configure (create .env or set environment variables)
cp .env.example .env
# Edit .env with your settings (Discord token, paths, etc.)

# Index documents
python cli.py index --data-dir ./data

# Test with CLI chat (recommended for development)
python cli.py chat

# Start Discord bot (production)
python cli.py discord
```

### New CLI Commands (v1.2)
```bash
python cli.py index                    # Build/update vector index
python cli.py chat                     # Interactive chat (grounded mode)
python cli.py chat --mode concise      # Brief answers
python cli.py chat --mode critic       # Verify claims
python cli.py chat --mode explain      # Show reasoning
python cli.py discord                  # Start Discord bot
python cli.py status                   # Show configuration and stats
```

### Testing Strategy
- **Manual testing**: Use `python cli.py chat` for rapid iteration
- **Multi-mode testing**: Test different modes with `--mode` flag
- **Reranking validation**: Compare results with/without `ARQ_RERANK_ENABLED`
- **Integration testing**: Test Discord bot with real mentions
- Future: Automated `pytest` fixtures with multilingual docs (PT/EN/ES) per spec §18

### v1.3+ Roadmap (Optional Enhancements)
v1.3 implemented: Docling HybridChunker, HNSW optimization. Future improvements:
1. **Sentence-window retrieval**: Implement spec §9 context window expansion
2. **ParentDocumentRetriever**: Better context preservation for long documents
3. **Automated testing**: pytest fixtures with multilingual docs (PT/EN/ES) per spec §18
4. **LLM-based reranking**: Alternative to cross-encoder for more semantic reranking
5. **Query expansion**: Multi-query retrieval for better recall

## Spec-Driven Development (SDD)

### Constitution for AI Agents (§I-V)
All code changes **must** comply with:
1. **Docstrings**: Google-style docstrings for all classes/functions (PEP 257)
2. **Sequential Thinking**: Use Sequential Thinking MCP for planning complex tasks
3. **Knowledge Freshness**: Validate dependencies via Ref MCP before adding
4. **Clean Architecture**: Separate ingestion, retrieval, generation layers—no cross-contamination
5. **KISS & YAGNI**: Solve current validated problems, avoid premature abstraction

### Contract-First Development
Before implementing features from `ArquimedesAI_Spec_Full_v1.1.md`:
1. Check Pydantic schemas in §5.2 for data contracts
2. Verify `.env` variables in §5.1 match implementation
3. Ensure CLI commands in §11 align with actual `cli.py` interface
4. Test against acceptance criteria in §1.3

## Common Pitfalls

### ❌ Don't
- Hardcode URLs/tokens in source files (use `.env` or config)
- Call Ollama REST API directly (use `langchain_community.llms.Ollama`)
- Skip embedding cache (causes slow re-indexing)
- Use synchronous code in Discord bot (causes blocking/timeouts)
- Add cloud-based services (violates 100% self-hosted principle)

### ✅ Do
- Use LangChain abstractions for retrieval/generation
- Cache embeddings with `LocalFileStore`
- Keep Discord responses async with edit pattern
- Follow Google-style docstrings (see existing classes)
- Check spec §2-3 for target architecture before proposing changes

## Key Files & Their Roles (v1.3.1)
- **`cli.py`**: Entry point, Typer CLI with index/chat/discord/status commands, multi-mode support
- **`settings.py`**: Pydantic Settings, all configuration centralized with .env support
- **`core/vector_store.py`**: Qdrant integration with HNSW optimization (v1.3.1)
- **`core/hybrid_retriever.py`**: BM25 + Dense hybrid retrieval with EnsembleRetriever
- **`core/reranker.py`**: Cross-encoder reranking with HuggingFace models (v1.2)
- **`core/rag_chain.py`**: LangChain LCEL chains for retrieval + generation + reranking
- **`core/llm_local.py`**: Ollama LLM wrapper (Gemma3:4b)
- **`core/embedder.py`**: BGE-M3 embeddings with caching
- **`ingest/loaders.py`**: Docling-based document loading with HybridChunker (v1.3)
- **`bots/discord_bot.py`**: Async Discord bot interface
- **`prompts/templates.py`**: Prompt templates (grounded, concise, critic, explain)
- **`ArquimedesAI_Spec_Full_v1.1.md`**: **Source of truth** for architecture, contracts, roadmap
- **`requirements.txt`**: Modern dependencies (LangChain 0.3+, Qdrant 1.12+, sentence-transformers, etc.)
- **`CHANGELOG.md`**: Version history and feature documentation

## Integration Points
- **Ollama** (local): HTTP API at `localhost:11434`, requires manual `ollama pull` for models
- **Qdrant**: Local persistence or remote server, no external cloud dependencies
- **Discord**: Websocket connection via `discord.py`, requires bot token with message content intent
- **HuggingFace**: Downloads models on first run (BGE-M3 embeddings, bge-reranker-v2-m3 cross-encoder)

## Questions to Ask Before Major Changes
1. Does this align with the spec v1.1 roadmap or is it scope creep?
2. Can this run on 8-16GB RAM machines (performance target §16)?
3. Does this maintain 100% local execution (no cloud dependencies)?
4. Are we adding proper docstrings and following the Constitution (§I-V)?
5. Is this being built contract-first with Pydantic schemas (§5)?
