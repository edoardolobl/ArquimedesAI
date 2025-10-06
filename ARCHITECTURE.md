# ArquimedesAI v1.3.1 Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                       ArquimedesAI v1.3.1                            │
│                  Local RAG System (8-16GB RAM)                       │
│          Docling HybridChunker + HNSW-Optimized Qdrant              │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────┐         ┌──────────────────┐
│   User Actions   │         │   External Deps   │
├──────────────────┤         ├──────────────────┤
│ • Add docs       │         │ • Ollama         │
│   to data/       │         │   (gemma3:4b)    │
│ • Run CLI        │         │ • HuggingFace    │
│ • Discord        │         │   (BGE-M3)       │
│   mention        │         └──────────────────┘
└──────────────────┘

         │                           │
         ▼                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        CLI Layer (cli.py)                            │
├─────────────────────────────────────────────────────────────────────┤
│  Commands:                                                           │
│  • python cli.py index      → Build/update vector index             │
│  • python cli.py discord    → Start Discord bot                     │
│  • python cli.py status     → Show system info                      │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Configuration (settings.py)                        │
├─────────────────────────────────────────────────────────────────────┤
│  Pydantic Settings from .env:                                        │
│  • ARQ_DATA_DIR, ARQ_STORAGE_DIR                                    │
│  • ARQ_EMBED_MODEL (BAAI/bge-m3)                                    │
│  • ARQ_OLLAMA_MODEL (gemma3:latest)                                 │
│  • ARQ_TOP_K, ARQ_CHUNK_SIZE                                        │
│  • ARQ_DISCORD_TOKEN                                                │
└─────────────────────────────────────────────────────────────────────┘


════════════════════════════════════════════════════════════════════════
                        INDEXING PIPELINE
════════════════════════════════════════════════════════════════════════

data/
 ├── report.pdf
 ├── slides.pptx
 └── notes.md
      │
      ▼
┌──────────────────┐
│ ingest/loaders.py│  ← Docling + HybridChunker (v1.3)
│ DocumentLoader   │     • PDF with OCR + table extraction
└──────────────────┘     • DOCX, PPTX, XLSX, MD, HTML, images
      │                  • Structure-aware chunking
      │                  • Rich metadata (headings, pages, bboxes)
      │ List[Document Chunks]
      ▼
┌──────────────────┐
│ core/embedder.py │  ← HuggingFace + Cache
│ EmbeddingManager │     • Model: BAAI/bge-m3
└──────────────────┘     • Dimension: 1024
      │                  • Cache: storage/embeddings_cache/
      │ Embeddings
      ▼
┌────────────────────┐
│core/vector_store.py│  ← Qdrant Client (HNSW optimized v1.3.1)
│  QdrantVectorStore │     • Local: storage/qdrant/
└────────────────────┘     • Distance: COSINE
      │                    • HNSW: m=32, ef_construct=256
      │                    • on_disk=true (low memory)
      ▼
  [Persisted]


════════════════════════════════════════════════════════════════════════
                         QUERY PIPELINE
════════════════════════════════════════════════════════════════════════

Discord Mention: "@ArquimedesAI What is X?"
      │
      ▼
┌──────────────────┐
│bots/discord_bot.py│  ← discord.py async
│  DiscordChatbot  │     • on_message event
└──────────────────┘     • Extract query
      │                  • Send "Processing..."
      │ query: str
      ▼
┌──────────────────┐
│core/rag_chain.py │  ← LangChain LCEL
│    RAGChain      │
└──────────────────┘
      │
      ├───────────────────────────────────────────┐
      │                                           │
      ▼                                           ▼
┌──────────────────────┐              ┌──────────────────┐
│core/hybrid_retriever │              │ core/llm_local.py│
│  HybridRetriever     │              │   LLMManager     │
│                      │              │                  │
│ • BM25 + Dense (RRF) │              │ • Ollama client  │
│ • Top-K: 8           │              │ • gemma3:latest  │
└──────────────────────┘              │ • Temperature: 0.3│
      │                                └──────────────────┘
      ▼                                          │
┌──────────────────────┐                        │
│ core/reranker.py     │                        │
│  (optional v1.2)     │                        │
│                      │                        │
│ • Cross-encoder      │                        │
│ • bge-reranker-v2-m3 │                        │
│ • Fetch 50 → Top 3   │                        │
└──────────────────────┘                        │
      │                                          │
      │ context: List[Document]                  │
      └─────────────┬────────────────────────────┘
                    │
                    ▼
            ┌──────────────────┐
            │prompts/templates │  ← ChatPromptTemplate
            │  GROUNDED_PROMPT │     • System message
            └──────────────────┘     • Context + Query
                    │                • Answer format
                    ▼
            [LLM Generation]
                    │
                    │ response: Dict
                    ▼
            ┌──────────────────┐
            │  Format Answer   │
            │  + Citations     │
            └──────────────────┘
                    │
                    ▼
            Discord.edit_message()
            "Here's your answer based on 3 sources..."


════════════════════════════════════════════════════════════════════════
                         DATA FLOW SUMMARY
════════════════════════════════════════════════════════════════════════

Phase 1: INDEXING (One-time or on-demand)
──────────────────────────────────────────
data/*.pdf → Docling HybridChunker → BGE-M3 → Qdrant (HNSW optimized)

Phase 2: QUERYING (Real-time)
─────────────────────────────
Discord query → Hybrid Retrieval (BM25+Dense) → Optional Reranking → Context → Gemma3 LLM → Answer


════════════════════════════════════════════════════════════════════════
                       COMPONENT DEPENDENCIES
════════════════════════════════════════════════════════════════════════

settings.py (config)
    ↓
    ├→ core/embedder.py (BGE-M3)
    ├→ core/llm_local.py (Ollama)
    ├→ core/vector_store.py (Qdrant)
    │      ↑
    │      └─ core/embedder.py
    │
    ├→ ingest/loaders.py (Docling)
    ├→ ingest/chunking.py
    │
    └→ core/rag_chain.py
           ↑
           ├─ core/vector_store.py
           └─ core/llm_local.py
           
bots/discord_bot.py
    ↓
    └→ core/rag_chain.py

cli.py
    ↓
    ├→ ingest/* (for index command)
    └→ bots/* (for discord command)


════════════════════════════════════════════════════════════════════════
                         STORAGE LAYOUT
════════════════════════════════════════════════════════════════════════

storage/
  ├── qdrant/                      # Qdrant persistence
  │   ├── collection/              
  │   │   └── arquimedes_chunks/   # Vector collection
  │   └── meta.json                # Qdrant metadata
  │
  └── embeddings_cache/            # HuggingFace cache
      └── BAAI_bge-m3/             # Model-specific cache
          └── *.pickle             # Cached embeddings


════════════════════════════════════════════════════════════════════════
                      KEY DESIGN DECISIONS
════════════════════════════════════════════════════════════════════════

1. Separation of Concerns
   • ingest/   - Document loading and preprocessing
   • core/     - Embedding, retrieval, generation
   • bots/     - Interface layer (Discord)

2. Configuration Management
   • All settings in .env via Pydantic
   • Type-safe with validation
   • No hardcoded values

3. Caching Strategy
   • Embeddings cached at storage/embeddings_cache/
   • Qdrant persists vectors at storage/qdrant/
   • No re-computation on restart

4. Async Pattern
   • Discord bot uses async/await
   • RAG chain supports both sync/async
   • Edit-message pattern for UX

5. LangChain LCEL
   • Composable chains (retriever + document chain)
   • Future-proof for v1.1 features
   • Standardized API

6. Local-First
   • Ollama for LLM (no API calls)
   • Qdrant local mode (no cloud)
   • HuggingFace local models
   • 100% self-hosted

════════════════════════════════════════════════════════════════════════
