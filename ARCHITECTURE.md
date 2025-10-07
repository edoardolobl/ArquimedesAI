# ArquimedesAI v2.0 Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                       ArquimedesAI v2.0                              │
│                  Local RAG System (8-16GB RAM)                       │
│          Semantic Routing + Docling + HNSW-Optimized Qdrant         │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────┐         ┌──────────────────┐
│   User Actions   │         │   External Deps   │
├──────────────────┤         ├──────────────────┤
│ • Add docs       │         │ • Ollama         │
│   to data/       │         │   (gemma3:4b)    │
│ • Run CLI        │         │ • HuggingFace    │
│   -r (routing)   │         │   (BGE-M3)       │
│   -c (convo)     │         │ • semantic-router│
│ • Discord        │         │   (v2.0)         │
│   mention        │         └──────────────────┘
└──────────────────┘

         │                           │
         ▼                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        CLI Layer (cli.py)                            │
├─────────────────────────────────────────────────────────────────────┤
│  Commands:                                                           │
│  • python cli.py index      → Build/update vector index             │
│  • python cli.py chat -r    → Chat with routing (v2.0)              │
│  • python cli.py chat -c    → Conversational mode (v2.0)            │
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
                         QUERY PIPELINE (v2.0)
════════════════════════════════════════════════════════════════════════

CLI/Discord Query: "O que é uma tag GTM?"
      │
      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   SEMANTIC ROUTER (v2.0 - Optional)                   │
│                     core/prompt_router.py                             │
├──────────────────────────────────────────────────────────────────────┤
│  Two-Stage Routing:                                                   │
│  1. Keyword Pre-filter: GTM context detection                         │
│  2. Semantic Classification: BGE-M3 + BM25 (hybrid, alpha=0.3)       │
│                                                                        │
│  Routes (89.5% accuracy):                                             │
│  ├─ 📚 gtm_qa          → GTM taxonomy questions                       │
│  ├─ 🛠️ gtm_generation  → Tag/trigger/variable creation               │
│  ├─ ✅ gtm_validation  → Configuration review/audit                   │
│  └─ 💬 general_chat    → Fallback for general questions               │
└──────────────────────────────────────────────────────────────────────┘
      │ route: RouteType + confidence
      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  PROMPT SELECTION (v2.0)                              │
│              prompts/{gtm_prompts,base_prompts}.py                    │
├──────────────────────────────────────────────────────────────────────┤
│  IF routing enabled:                                                  │
│    • gtm_qa         → GTM_QA_SYSTEM_PROMPT + style modifier          │
│    • gtm_generation → GTM_GENERATION_SYSTEM_PROMPT + style           │
│    • gtm_validation → GTM_VALIDATION_SYSTEM_PROMPT + style           │
│    • general_chat   → GENERAL_CHAT_SYSTEM_PROMPT + style             │
│  ELSE:                                                                │
│    • Use base GROUNDED_PROMPT / mode-specific prompt                 │
└──────────────────────────────────────────────────────────────────────┘
      │ system_prompt: str
      ▼
┌──────────────────┐
│core/rag_chain.py │  ← LangChain LCEL
│    RAGChain      │     • Conversational memory (v2.0, optional)
└──────────────────┘     • Structured citations (v2.0, foundation)
      │
      ├───────────────────────────────────────────┐
      │                                           │
      ▼                                           ▼
┌──────────────────────┐              ┌──────────────────┐
│core/hybrid_retriever │              │ core/llm_local.py│
│  HybridRetriever     │              │   LLMManager     │
│                      │              │                  │
│ • BM25 + Dense (RRF) │              │ • ChatOllama     │
│ • Top-K: 8           │              │ • gemma3:4b      │
└──────────────────────┘              │ • Temperature:0.3│
      │                                └──────────────────┘
      ▼                                          │
┌──────────────────────┐                        │
│ core/reranker.py     │                        │
│  (optional v1.3)     │                        │
│                      │                        │
│ • Cross-encoder      │                        │
│ • bge-reranker-v2-m3 │                        │
│ • Fetch 50 → Top 5   │                        │
└──────────────────────┘                        │
      │                                          │
      │ context: List[Document]                  │
      └─────────────┬────────────────────────────┘
                    │
                    ▼
            ┌──────────────────┐
            │  Prompt Template │  ← ChatPromptTemplate
            │  (domain-specific│     • System prompt (from router)
            │   or base)       │     • Context + Query
            └──────────────────┘     • Answer format
                    │
                    ▼
            [LLM Generation]
                    │
                    │ response: Dict
                    ▼
            ┌──────────────────┐
            │  Format Answer   │
            │  + Citations     │
            │  + Route Info    │  ← v2.0: Show route & confidence
            └──────────────────┘
                    │
                    ▼
            CLI/Discord Output
            "[Route: 📚 gtm_qa (0.95)]"
            "Uma tag GTM é..."


════════════════════════════════════════════════════════════════════════
                         DATA FLOW SUMMARY (v2.0)
════════════════════════════════════════════════════════════════════════

Phase 1: INDEXING (One-time or on-demand)
──────────────────────────────────────────
data/*.pdf → Docling HybridChunker → BGE-M3 → Qdrant (HNSW optimized)

Phase 2: QUERYING (Real-time)
─────────────────────────────
v1.x: Query → Hybrid Retrieval → Reranking → Gemma3 → Answer
v2.0: Query → Router (domain detection) → Domain Prompt → Retrieval → Gemma3 → Answer

Phase 3: CONVERSATIONAL (v2.0 - Optional)
─────────────────────────────────────────
Session History → Query → Router → RAG → Response → Update History


════════════════════════════════════════════════════════════════════════
                       COMPONENT DEPENDENCIES (v2.0)
════════════════════════════════════════════════════════════════════════

settings.py (config)
    ↓
    ├→ core/embedder.py (BGE-M3)
    ├→ core/llm_local.py (ChatOllama)
    ├→ core/vector_store.py (Qdrant)
    │      ↑
    │      └─ core/embedder.py
    │
    ├→ core/prompt_router.py (v2.0)
    │      ↑
    │      └─ prompts/gtm_prompts.py (utterances)
    │
    ├→ ingest/loaders.py (Docling)
    │
    └→ core/rag_chain.py
           ↑
           ├─ core/vector_store.py
           ├─ core/llm_local.py
           ├─ core/prompt_router.py (v2.0, optional)
           └─ prompts/{gtm_prompts,base_prompts}.py (v2.0)
           
bots/discord_bot.py
    ↓
    └→ core/rag_chain.py

cli.py
    ↓
    ├→ ingest/* (for index command)
    ├→ core/rag_chain.py (for chat command)
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
