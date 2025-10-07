# ArquimedesAI

<p align="center">
  <img src="assets/arquimedesai.jpg" alt="ArquimedesAI Logo" width="35%" height="35%">
</p>

<p align="center">
  <strong>100% Open-Source, Self-Hosted RAG Chatbot</strong><br>
  <em>Runs locally on modest hardware (8-16GB RAM)</em>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#️-architecture">Architecture</a> •
  <a href="#️-configuration">Configuration</a> •
  <a href="#-development-milestones">Development</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## 🆕 What's New in v2.0

### 🎯 Semantic Routing System
- **Intelligent Query Classification**: Automatic domain detection (GTM Q&A, Generation, Validation, General Chat)
- **Two-Stage Routing**: Keyword pre-filtering + semantic classification (89.5% accuracy)
- **Hybrid Routing**: BM25 (70%) + BGE-M3 (30%) for robust classification
- **Visual Indicators**: Route-specific emojis (📚 Q&A, 🛠️ Generation, ✅ Validation, 💬 General)

### 🏷️ GTM Domain Expertise
- **Specialized Prompts**: Domain-specific templates for Google Tag Manager taxonomy
- **Three GTM Modes**: Q&A (questions), Generation (tag/trigger creation), Validation (config review)
- **Portuguese-First**: Brazilian Portuguese with technical term preservation
- **73 Training Utterances**: Curated examples for accurate route classification

### 💬 Conversational Memory (Optional)
- **Session-Based History**: In-session chat context preservation
- **Configurable History**: Control conversation length (default: 20 messages)
- **CLI Flag**: `--conversational` / `-c` to enable multi-turn conversations
- **Turn Tracking**: Session stats displayed on exit

### 📝 Structured Citations Foundation
- **Pydantic Schemas**: `Citation`, `QuotedAnswer`, `CitedAnswer` models ready
- **ChatOllama Migration**: Supports `.with_structured_output()` for future integration
- **ID-Based Tracking**: `format_docs_with_id()` helper for citation grounding

### 🚀 Enhanced CLI
- **Conversational Mode**: `--conversational` / `-c` for multi-turn conversations
- **Routing Always On**: Automatic domain detection (GTM Q&A, Generation, Validation)
- **Mode Selection**: Combine conversational with styles (grounded, concise, critic, explain)
- **Disable Routing**: Set `ARQ_DISABLE_ROUTING=true` in .env if needed
- **Confidence Display**: See routing confidence scores in chat output

### 🔧 Previous Updates (v1.3.1)
- **LangChain 1.0 Ready**: Official `langchain-ollama`, `.invoke()` LCEL pattern, SHA-256 cache
- **HNSW Optimization**: Better accuracy (`m=32`, `ef_construct=256`), lower memory (on-disk)
- **Docling Integration**: Structure-aware chunking, rich metadata, table extraction with OCR

[See full changelog →](CHANGELOG.md)

---

## 🌟 What is ArquimedesAI?

ArquimedesAI is a production-ready **Retrieval Augmented Generation (RAG)** chatbot that runs entirely on your local machine. Built with cutting-edge NLP technologies, it enables you to chat with your documents using state-of-the-art language models—all while maintaining complete privacy and control.

### Key Highlights

- 🔒 **100% Local & Private**: No cloud dependencies, your data never leaves your machine
- 🚀 **Production-Ready**: Built with LangChain 0.3+, Qdrant vector store, and modern architecture
- 🌍 **Multilingual**: Native support for Portuguese, English, Spanish, and 100+ languages
- ⚡ **Efficient**: Optimized for 8-16GB RAM systems with hybrid retrieval and HNSW indexing
- 🎯 **Advanced Retrieval**: Hybrid search (BM25 + Dense) with optional cross-encoder reranking
- 📚 **Rich Document Support**: PDF, DOCX, PPTX, XLSX, Markdown, HTML, and images (with OCR)
- 🤖 **Multiple Interfaces**: CLI chat for testing, Discord bot for production use

---

## 🏗️ Architecture

```
User Documents → Docling Parser → HybridChunker → BGE-M3 Embeddings → Qdrant (HNSW)
                                                                              ↓
User Query → Hybrid Retrieval → [BM25 40% + Dense 60%] → Reranker (optional) → Gemma3 LLM → Answer
```

### Retrieval Pipeline

**Hybrid Retrieval**: Combines BM25 (keyword) and dense (semantic) search
- BM25: Exact term matching, handles acronyms and proper nouns
- Dense: Semantic similarity via BGE-M3 embeddings
- Reciprocal Rank Fusion (RRF) merges results optimally

**Reranking** (optional): Cross-encoder scores top-K candidates
- Model: `bge-reranker-v2-m3` (multilingual)
- Improves relevance significantly (~100-300ms latency)

- **Fetch 50 candidates** (broad net for recall)
- **Rerank with cross-encoder** (precision filtering)
- **Return top 5** (highly relevant results)

**Generation**: Gemma3 4B LLM generates grounded answers with domain-specific prompts

**v2.0 Routing** (optional): Semantic query classification
- Two-stage routing: keyword pre-filtering + semantic classification
- Routes: GTM Q&A, GTM Generation, GTM Validation, General Chat
- Hybrid classification: BM25 (70%) + BGE-M3 (30%) for 89.5% accuracy
- Domain-specific prompts optimize for task type

**v2.0 Conversational** (optional): Multi-turn chat with history
- Session-based message history (configurable length)
- Context preservation across conversation turns
- Compatible with routing and style modes

### Directory Structure

```
arquimedesai/
├── core/                  # RAG components
│   ├── embedder.py       # BGE-M3 with caching
│   ├── llm_local.py      # Ollama (ChatOllama) integration
│   ├── vector_store.py   # Qdrant with HNSW
│   ├── hybrid_retriever.py  # BM25 + Dense
│   ├── reranker.py       # Cross-encoder reranking
│   ├── prompt_router.py  # Semantic routing (v2.0)
│   └── rag_chain.py      # LangChain LCEL chains
├── ingest/               # Document processing
│   └── loaders.py        # Docling integration
├── bots/                 # Interfaces
│   └── discord_bot.py    # Discord bot
├── prompts/              # Prompt templates
│   ├── templates.py      # Base templates & Pydantic schemas
│   ├── base_prompts.py   # Reusable prompt components (v2.0)
│   └── gtm_prompts.py    # GTM domain-specific prompts (v2.0)
├── data/                 # Your documents (gitignored)
├── storage/              # Vector store & cache (gitignored)
├── cli.py                # Command-line interface
├── settings.py           # Pydantic configuration
└── .env                  # Environment variables
```

---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.12+** ([Download](https://www.python.org/downloads/))
2. **Ollama** ([Installation Guide](https://ollama.ai))

   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Windows
   # Download from https://ollama.ai/download/windows
   ```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/edoardolobl/ArquimedesAI.git
cd ArquimedesAI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pull Ollama model
ollama pull gemma3:latest

# 4. Configure environment
cp .env.example .env
# Edit .env with your Discord token (optional) and other settings

# 5. Add your documents
# Place PDF, DOCX, PPTX, XLSX, MD, HTML files in data/ folder

# 6. Build the index
python cli.py index

# 7. Start chatting!
python cli.py chat                              # CLI interface (default: routing enabled)
python cli.py chat --conversational             # Multi-turn conversation (v2.0)
python cli.py chat -c                           # Short flag for conversational
python cli.py chat --mode concise               # Brief answers
python cli.py discord                           # Discord bot
```

### First Query

```
$ python cli.py chat

ArquimedesAI Chat v2.0 (grounded mode, routing enabled)
Type 'exit', 'quit', or 'q' to exit

✓ Chain loaded
✓ Routing enabled (4 routes available)

You: O que é uma tag GTM?

[Route: 📚 gtm_qa (confidence: 0.95)]
ArquimedesAI: Uma tag GTM (Google Tag Manager) é um fragmento de código JavaScript...
```

---

## ⚙️ Configuration

### Environment Variables (.env)

ArquimedesAI uses a `.env` file for configuration. Copy `.env.example` to `.env` and customize:

#### Essential Settings

```bash
# Ollama LLM
ARQ_OLLAMA_BASE=http://localhost:11434
ARQ_OLLAMA_MODEL=gemma3:latest      # Ollama model (try: llama3.1, mistral, etc.)
ARQ_OLLAMA_TEMPERATURE=0.3

# Embeddings
ARQ_EMBED_MODEL=BAAI/bge-m3         # Multilingual embeddings (1024 dim)

# Vector Store
ARQ_QDRANT_PATH=./storage/qdrant    # Local storage path
ARQ_TOP_K=8                         # Results per query
```

#### Hybrid Retrieval (Recommended)

```bash
ARQ_HYBRID=true                     # Enable hybrid search
ARQ_BM25_WEIGHT=0.4                 # Keyword importance
ARQ_DENSE_WEIGHT=0.6                # Semantic importance
```

#### Advanced Features (Optional)

```bash
# Cross-Encoder Reranking (improves relevance)
ARQ_RERANK_ENABLED=true
ARQ_RERANK_MODEL=BAAI/bge-reranker-v2-m3
ARQ_RERANK_TOP_N=5

# Docling OCR (for scanned PDFs/images)
ARQ_DOCLING_OCR=true                # Enable OCR (slower but accurate)
ARQ_DOCLING_TABLE_MODE=accurate     # Table extraction: fast or accurate

# Qdrant HNSW Optimization
ARQ_QDRANT_HNSW_M=32               # Graph connectivity (16-64)
ARQ_QDRANT_HNSW_EF_CONSTRUCT=256   # Build quality (100-512)
ARQ_QDRANT_ON_DISK=true            # Lower memory usage
```

#### Discord Bot

```bash
ARQ_DISCORD_TOKEN=your_bot_token_here
ARQ_DISCORD_PREFIX=!                # Command prefix (optional)
```

### Understanding .env Settings

The `.env` file controls all aspects of ArquimedesAI's behavior. Here's how to customize it for your needs:

#### Performance Profiles

**Development/Testing** (faster, less accurate):
```bash
ARQ_QDRANT_HNSW_M=16
ARQ_QDRANT_HNSW_EF_CONSTRUCT=100
ARQ_QDRANT_ON_DISK=false
ARQ_DOCLING_OCR=false
ARQ_RERANK_ENABLED=false
```

**Production** (slower, more accurate):
```bash
ARQ_QDRANT_HNSW_M=32
ARQ_QDRANT_HNSW_EF_CONSTRUCT=256
ARQ_QDRANT_ON_DISK=true
ARQ_DOCLING_OCR=true
ARQ_RERANK_ENABLED=true
```

#### Model Selection

You can change the LLM model by editing `ARQ_OLLAMA_MODEL`:

```bash
ARQ_OLLAMA_MODEL=gemma3:latest  # Default, 4B parameters (best balance)
ARQ_OLLAMA_MODEL=gemma3:9b      # Higher quality, slower
ARQ_OLLAMA_MODEL=llama3.1:8b    # Alternative high-quality option
ARQ_OLLAMA_MODEL=mistral:7b     # Faster alternative
```

First pull the model: `ollama pull gemma3:9b`

#### Retrieval Tuning

Adjust retrieval behavior:

```bash
ARQ_TOP_K=8                    # Number of chunks to retrieve (3-10)
ARQ_BM25_WEIGHT=0.4           # Increase for keyword-heavy docs
ARQ_DENSE_WEIGHT=0.6          # Increase for semantic queries
ARQ_RERANK_TOP_N=5            # Final results after reranking (1-10)
```

[See `.env.example` for all options →](.env.example)

---

## 📖 Usage

### CLI Commands

```bash
# Index documents
python cli.py index                          # Incremental index
python cli.py index --rebuild                # Full rebuild
python cli.py index --data-dir ./my-docs     # Custom directory

# Chat modes
python cli.py chat                           # Default (grounded, routing enabled)
python cli.py chat --mode concise            # Brief answers
python cli.py chat --mode critic             # Verify claims
python cli.py chat --mode explain            # Show reasoning

# v2.0 Features: Conversational Memory
python cli.py chat --conversational          # Multi-turn conversation
python cli.py chat -c                        # Short flag for conversational
python cli.py chat -c --mode concise         # Conversational + brief answers

# Disable routing (if needed)
# Set ARQ_DISABLE_ROUTING=true in .env, then:
python cli.py chat                           # Routing disabled

# Discord bot
python cli.py discord                        # Start Discord bot

# System status
python cli.py status                         # Show configuration
```

### Discord Bot

1. Invite bot to your server
2. Mention the bot with your question:
   ```
   @ArquimedesAI What is the main topic of the document?
   ```
3. Bot will search indexed documents and respond with grounded answers citing sources

### CLI Chat Modes

Perfect for testing and development:

```bash
# Grounded mode (default): Detailed answers with explicit citations
python cli.py chat

# Concise mode: Brief 1-3 sentence answers
python cli.py chat --mode concise

# Critic mode: Verify if context supports claims
python cli.py chat --mode critic

# Explain mode: Show reasoning steps
python cli.py chat --mode explain
```

---

## 🎯 Features

### Document Processing
- **Docling HybridChunker**: Structure-aware, tokenization-optimized chunking
- **Rich Metadata**: Page numbers, bounding boxes, section headings
- **OCR Support**: Extract text from scanned PDFs and images
- **Table Extraction**: Accurate parsing of complex tables
- **Format Support**: PDF, DOCX, PPTX, XLSX, Markdown, HTML, images

### Retrieval
- **Hybrid Search**: BM25 (keyword) + Dense (semantic) with RRF
- **Cross-Encoder Reranking**: Improve relevance with `bge-reranker-v2-m3`
- **Configurable Weights**: Balance keyword vs. semantic search
- **Multilingual**: BGE-M3 supports 100+ languages natively

### Generation
- **Grounded Answers**: Explicit citations from source documents
- **Multi-Mode Prompts**: 4 modes (grounded, concise, critic, explain)
- **Hallucination Prevention**: Strong prompt engineering
- **Local LLMs**: Gemma3 4B via Ollama (high quality, efficient)
- **Flexible Models**: Easy to swap LLMs (gemma3, llama3.1, mistral)

### Interfaces
- **CLI Chat**: Interactive testing with mode selection
- **Discord Bot**: Production deployment with async support
- **Status Command**: View configuration and system stats
- **Batch Indexing**: Efficient document processing

---

## 🔄 Development Milestones

### v2.0.0 (2025-10-07) - Semantic Routing & GTM Domain Expertise
- 🎯 Semantic routing system (89.5% accuracy, 4 routes, **enabled by default**)
- 🏷️ GTM domain-specific prompts (Q&A, Generation, Validation)
- 💬 Conversational memory (session-based, optional)
- 📝 Structured citations foundation (Pydantic schemas)
- 🚀 Enhanced CLI (--conversational flag, routing always on)

### v1.3.1 (2025-10-06) - LangChain 1.0 Ready
- ✅ Migrated to `langchain-ollama` official package
- ✅ Updated to modern retrieval API (`.invoke()` pattern)
- ✅ Secure embeddings cache (SHA-256)
- ✅ HNSW optimization for better accuracy

### v1.3.0 (2025-10-05) - Docling Integration
- 📄 Structure-aware chunking with HybridChunker
- 🔍 Rich metadata (pages, bounding boxes, sections)
- 📊 Accurate table extraction
- ��️ OCR support for images and scanned PDFs

### v1.2.0 (2025-10-05) - Advanced Retrieval
- 🎯 Cross-encoder reranking
- 🧩 Optional semantic chunking
- 🎨 Multi-mode CLI chat

### v1.1.0 (2025-10-05) - Hybrid Search
- 🔍 BM25 + Dense retrieval with RRF
- 💬 CLI chat interface
- 📝 Enhanced prompts with citations
- ⚙️ Configurable retrieval weights

### v1.0.0 - Modern Foundation
- 🗂️ Qdrant vector store (replaces FAISS)
- 🤖 Gemma3 4B LLM (upgraded from Mistral 7B)
- 🌍 BGE-M3 multilingual embeddings
- ⚡ LangChain 0.3+ with LCEL
- 🎯 Modular architecture

[View detailed changelog →](CHANGELOG.md)

---

## 🧪 Testing & Development

### Test Artifacts

Development test files are excluded from version control (`.gitignore`):
- `test_*.py` - Unit and integration tests
- `verify_*.py` - Validation scripts
- `__pycache__/` - Python bytecode cache

### Code Quality

ArquimedesAI follows strict documentation standards:
- ✅ **PEP 8** code style compliance
- ✅ **PEP 257** docstring conventions
- ✅ **Google-style docstrings** for all public APIs
- ✅ **Type hints** using Python 3.12+ syntax

### Running Tests

```bash
# Verify deprecation fixes
python verify_deprecation_fixes.py

# Test RAG pipeline
python test_rag_fix.py

# Check configuration
python cli.py status
```

### Code Quality Tools

```bash
# Format code
ruff format .

# Lint
ruff check .

# Type checking
mypy .
```

---

## 🤝 Contributing

We welcome contributions to ArquimedesAI! Whether it's:

- 🐛 Bug reports and fixes
- ✨ Feature requests and implementations
- 📚 Documentation improvements
- 🌍 Translations and internationalization

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow code quality standards (PEP 8, Google-style docstrings)
4. Test your changes thoroughly
5. Commit with clear messages (`git commit -m 'Add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Guidelines

- Follow the **Constitution for AI Agents** (see `ArquimedesAI_Spec_Full_v1.1.md` §I-V)
- Add Google-style docstrings to all functions/classes
- Test changes before submitting PR
- Update documentation as needed

---

## 📝 License

ArquimedesAI is licensed under the **MIT License**.

This means you can:
- ✅ Use commercially
- ✅ Modify and distribute
- ✅ Use privately
- ✅ Sublicense

See the [LICENSE](LICENSE) file for full details.

---

## 🙏 Acknowledgments

ArquimedesAI builds upon amazing open-source projects:

- [LangChain](https://github.com/langchain-ai/langchain) - RAG framework and LCEL
- [Qdrant](https://github.com/qdrant/qdrant) - Vector database
- [Ollama](https://ollama.ai) - Local LLM runtime
- [Docling](https://github.com/DS4SD/docling) - Document processing by IBM Research
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) - Multilingual embeddings by BAAI
- [Gemma](https://ai.google.dev/gemma) - Efficient language model by Google

Special thanks to the open-source AI community! 🚀

---

## 📞 Support & Community

- 📖 **Documentation**: [Full docs](ARCHITECTURE.md) | [Quick start](QUICKSTART.md) | [Setup guide](SETUP.md)
- 💬 **Issues**: [GitHub Issues](https://github.com/edoardolobl/ArquimedesAI/issues)
- 🐛 **Bug Reports**: Use the issue template
- 💡 **Feature Requests**: Share your ideas!
- 📚 **Discussions**: [GitHub Discussions](https://github.com/edoardolobl/ArquimedesAI/discussions)

---

Made with ❤️ by [edoardolobl](https://github.com/edoardolobl)  
**100% Open Source • 100% Local • 100% Private**
