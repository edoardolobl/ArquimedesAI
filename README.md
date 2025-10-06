# ArquimedesAI# ArquimedesAI v1.2



<p align="center"><p align="center">

  <img src="assets/arquimedesai.jpg" alt="ArquimedesAI Logo" title="ArquimedesAI" width="35%" height="35%">  <img src="assets/arquimedesai.jpg" alt="ArquimedesAI Logo" title="ArquimedesAI" width="35%" height="35%">

</p></p>



<p align="center">**100% open-source, self-hosted RAG chatbot** that runs locally on modest hardware (8-16GB RAM).

  <strong>100% Open-Source, Self-Hosted RAG Chatbot</strong><br>

  <em>Runs locally on modest hardware (8-16GB RAM)</em>## ✨ What's New in v1.2

</p>

### Advanced Retrieval Features (Optional)

<p align="center">- 🎯 **Cross-Encoder Reranking**: Significantly improve relevance with `bge-reranker-v2-m3`

  <a href="#-quick-start">Quick Start</a> •- 🧩 **Semantic Chunking**: Embedding-based chunking for better context preservation

  <a href="#-features">Features</a> •- 🎨 **Multi-Mode CLI Chat**: Test different prompt strategies (`--mode grounded|concise|critic|explain`)

  <a href="#-architecture">Architecture</a> •

  <a href="#-configuration">Configuration</a> •All v1.2 features are **opt-in** and disabled by default for backward compatibility.

  <a href="#-development-milestones">Development</a> •

  <a href="#-contributing">Contributing</a>### v1.1 Foundation

</p>- 🔍 **Hybrid Retrieval**: BM25 (keyword) + Dense (semantic) search with Reciprocal Rank Fusion

- 💬 **CLI Chat Interface**: Interactive testing mode (`python cli.py chat`)

---- 📝 **Enhanced Prompts**: Explicit citation format and stronger hallucination prevention

- ⚙️ **Configurable Weights**: Adjust BM25/Dense balance via `.env` (`ARQ_BM25_WEIGHT`, `ARQ_DENSE_WEIGHT`)

## 🌟 What is ArquimedesAI?

### v1.0 Foundation

ArquimedesAI is a production-ready **Retrieval Augmented Generation (RAG)** chatbot that runs entirely on your local machine. Built with cutting-edge NLP technologies, it enables you to chat with your documents using state-of-the-art language models—all while maintaining complete privacy and control.- 🗂️ **Advanced document parsing** with Docling (PDF, DOCX, PPTX, XLSX, Markdown, HTML)

- 🚀 **Qdrant vector store** for persistent, production-ready indexing

### Key Highlights- 🌍 **Multilingual support** with BGE-M3 embeddings (Portuguese, English, Spanish, and more)

- ⚡ **Modern tech stack**: LangChain 0.3, Pydantic v2, discord.py 2.4

- 🔒 **100% Local & Private**: No cloud dependencies, your data never leaves your machine- 🤖 **Gemma2 1B** via Ollama for fast, efficient local inference

- 🚀 **Production-Ready**: Built with LangChain 0.3+, Qdrant vector store, and modern architecture- 🎯 **Modular architecture**: Clean separation of ingestion, retrieval, and generation

- 🌍 **Multilingual**: Native support for Portuguese, English, Spanish, and 100+ languages

- ⚡ **Efficient**: Optimized for 8-16GB RAM systems with hybrid retrieval and HNSW indexing## 🏗️ Architecture

- 🎯 **Advanced Retrieval**: Hybrid search (BM25 + Dense) with optional cross-encoder reranking

- 📚 **Rich Document Support**: PDF, DOCX, PPTX, XLSX, Markdown, HTML, and images (with OCR)```

- 🤖 **Multiple Interfaces**: CLI chat for testing, Discord bot for production useUser Documents (data/) → Docling Parser → Chunks (Semantic/Character) → BGE-M3 Embeddings → Qdrant

                                                                                                ↓

---User Query (CLI/Discord) → HybridRetriever → [BM25 + Qdrant via RRF] → Reranker (optional) → Gemma2 LLM → Grounded Answer

```

## 🆕 What's New in v1.3.1

**Retrieval Pipeline (v1.2):**

### LangChain 1.0 Ready1. **Hybrid Retrieval**: BM25 (40%) + Dense (60%) via RRF → Top-20 candidates

- ✅ **Ollama Migration**: Updated to official `langchain-ollama` package2. **Reranking (optional)**: Cross-encoder scores → Top-3 most relevant

- ✅ **Modern Retrieval API**: BaseRetriever using `.invoke()` pattern3. **Generation**: LLM answers using reranked context with citations

- ✅ **Secure Caching**: SHA-256 key encoder for embeddings cache

### Directory Structure

### HNSW Optimization

- 🎯 **Better Accuracy**: Configurable HNSW parameters (`m=32`, `ef_construct=256`)```

- 💾 **Lower Memory**: On-disk storage for production deploymentsarquimedesai/

- ⚡ **Balanced Performance**: Optimized for both speed and quality├── core/              # RAG components (embeddings, LLM, chains, hybrid retrieval)

├── ingest/            # Document loaders and chunking

### Docling Integration (v1.3)├── bots/              # Discord bot interface

- 📄 **Structure-Aware Chunking**: Preserves document hierarchy and formatting├── prompts/           # Prompt templates (grounded, concise, critic, explain)

- 🔍 **Rich Metadata**: Page numbers, bounding boxes, section context├── data/              # Your documents (gitignored)

- 📊 **Table Extraction**: Accurate table parsing with OCR support├── storage/           # Vector store & cache (gitignored)

├── cli.py             # Command-line interface (index/chat/discord/status)

[See full changelog →](CHANGELOG.md)├── settings.py        # Pydantic configuration

└── .env               # Environment variables (create from .env.example)

---```



## 🏗️ Architecture## 🚀 Quick Start



```### Prerequisites

User Documents → Docling Parser → HybridChunker → BGE-M3 Embeddings → Qdrant (HNSW)

                                                                              ↓1. **Python 3.12+**

User Query → Hybrid Retrieval → [BM25 40% + Dense 60%] → Reranker (optional) → Gemma2 LLM → Answer2. **Ollama** ([installation guide](https://ollama.ai))

```

### Installation

### Retrieval Pipeline

```bash

1. **Hybrid Retrieval**: Combines BM25 (keyword) and dense (semantic) search# 1. Clone repository

   - BM25: Exact term matching, handles acronyms and proper nounsgit clone https://github.com/edoardolobl/ArquimedesAI.git

   - Dense: Semantic similarity via BGE-M3 embeddingscd ArquimedesAI

   - Reciprocal Rank Fusion (RRF) merges results optimally

# 2. Install dependencies

2. **Reranking** (optional): Cross-encoder scores top-K candidatespip install -r requirements.txt

   - Model: `bge-reranker-v2-m3` (multilingual)

   - Improves relevance significantly (~100-300ms latency)# 3. Pull Ollama model

ollama pull gemma2:1b

3. **Generation**: Gemma2 1B LLM generates grounded answers

   - Explicit citations from retrieved context# 4. Create .env file

   - Multiple prompt modes: grounded, concise, critic, explaincp .env.example .env

# Edit .env and add your Discord token (optional)

### Directory Structure

# 5. Add documents to data/ folder

```# Supported: PDF, DOCX, PPTX, XLSX, Markdown, HTML

arquimedesai/

├── core/                  # RAG components# 6. Build index

│   ├── embedder.py       # BGE-M3 with cachingpython cli.py index

│   ├── llm_local.py      # Ollama integration

│   ├── vector_store.py   # Qdrant with HNSW# 7. Test with CLI chat (recommended)

│   ├── hybrid_retriever.py  # BM25 + Densepython cli.py chat

│   ├── reranker.py       # Cross-encoder reranking

│   └── rag_chain.py      # LangChain LCEL chains# 8. Or start Discord bot

├── ingest/               # Document processingpython cli.py discord

│   └── loaders.py        # Docling integration```

├── bots/                 # Interfaces

│   └── discord_bot.py    # Discord bot### Configuration

├── prompts/              # Prompt templates

│   └── templates.py      # 4 modes (grounded/concise/critic/explain)Edit `.env` file to customize:

├── data/                 # Your documents (gitignored)

├── storage/              # Vector store & cache (gitignored)```bash

├── cli.py                # Command-line interface# Vector Database

├── settings.py           # Pydantic configurationARQ_DB=qdrant

└── .env                  # Environment variablesARQ_QDRANT_PATH=./storage/qdrant  # Local persistence

```

# Embeddings

---ARQ_EMBED_MODEL=BAAI/bge-m3



## 🚀 Quick Start# Retrieval (v1.1)

ARQ_HYBRID=true            # Enable hybrid retrieval (BM25 + Dense)

### PrerequisitesARQ_BM25_WEIGHT=0.4        # BM25 weight (keyword matching)

ARQ_DENSE_WEIGHT=0.6       # Dense weight (semantic similarity)

1. **Python 3.12+** ([Download](https://www.python.org/downloads/))ARQ_TOP_K=8                # Number of results

2. **Ollama** ([Installation Guide](https://ollama.ai))

   ```bash# Reranking (v1.2 - optional)

   # macOS/LinuxARQ_RERANK_ENABLED=false   # Enable cross-encoder reranking

   curl -fsSL https://ollama.ai/install.sh | shARQ_RERANK_MODEL=BAAI/bge-reranker-v2-m3

   ARQ_RERANK_TOP_N=3         # Final results after reranking

   # Windows

   # Download from https://ollama.ai/download/windows# Chunking

   ```ARQ_CHUNK_SIZE=1000        # Characters per chunk

ARQ_SEMANTIC_CHUNKING=false  # Use semantic chunking (experimental)

### InstallationARQ_SEMANTIC_BREAKPOINT_TYPE=percentile



```bash# Ollama LLM

# 1. Clone the repositoryARQ_OLLAMA_MODEL=gemma2:1b

git clone https://github.com/edoardolobl/ArquimedesAI.gitARQ_OLLAMA_TEMPERATURE=0.3

cd ArquimedesAI

# Discord Bot

# 2. Install dependenciesARQ_DISCORD_TOKEN=your_token_here

pip install -r requirements.txt```



# 3. Pull Ollama model## 📖 Usage

ollama pull gemma2:1b

### CLI Commands

# 4. Configure environment

cp .env.example .env```bash

# Edit .env with your Discord token (optional) and other settings# Build/update index from data/ folder

python cli.py index --data-dir ./data

# 5. Add your documents

# Place PDF, DOCX, PPTX, XLSX, MD, HTML files in data/ folder# Force rebuild (deletes existing index)

python cli.py index --rebuild

# 6. Build the index

python cli.py index# Interactive CLI chat (v1.1 - recommended for testing)

python cli.py chat

# 7. Start chatting!

python cli.py chat                    # CLI interface# Multi-mode chat (v1.2 - test different prompt strategies)

python cli.py chat --mode concise     # Brief answerspython cli.py chat --mode concise    # Brief answers

python cli.py discord                 # Discord botpython cli.py chat --mode critic     # Verify claims

```python cli.py chat --mode explain    # Show reasoning



### First Query# Start Discord bot

python cli.py discord

```bash

$ python cli.py chat# Show system status and configuration

python cli.py status

ArquimedesAI Chat (grounded mode)```

Type 'exit', 'quit', or 'q' to exit

### Discord Bot

✓ Chain loaded

1. Invite bot to your server

You: What are the main topics in my documents?2. Mention the bot with your question:

ArquimedesAI: Based on your documents, the main topics are...   ```

```   @ArquimedesAI What is the main topic of the document?

   ```

---3. Bot will search indexed documents and respond with grounded answers citing sources



## ⚙️ Configuration### CLI Chat Modes (v1.2)



### Environment Variables (.env)Perfect for testing and development:

```bash

ArquimedesAI uses a `.env` file for configuration. Copy `.env.example` to `.env` and customize:# Grounded mode (default): Detailed answers with explicit citations

python cli.py chat

#### Essential Settings

# Concise mode: Brief 1-3 sentence answers

```bashpython cli.py chat --mode concise

# LLM Configuration

ARQ_OLLAMA_MODEL=gemma2:1b          # Ollama model (try: llama3.1, mistral, etc.)# Critic mode: Verify if context supports claims

ARQ_OLLAMA_BASE=http://localhost:11434python cli.py chat --mode critic

ARQ_OLLAMA_TEMPERATURE=0.3          # Lower = more deterministic

# Explain mode: Show reasoning steps

# Embeddingspython cli.py chat --mode explain

ARQ_EMBED_MODEL=BAAI/bge-m3         # Multilingual embeddings (1024 dim)```



# Vector Store## 🔧 Advanced Configuration

ARQ_QDRANT_PATH=./storage/qdrant    # Local storage path

ARQ_TOP_K=5                         # Results per query### Enable Cross-Encoder Reranking (v1.2)

```

Significantly improves result relevance at the cost of ~100-300ms latency:

#### Hybrid Retrieval (Recommended)

```bash

```bash# .env

ARQ_HYBRID=true                     # Enable hybrid searchARQ_RERANK_ENABLED=true

ARQ_BM25_WEIGHT=0.4                 # Keyword importanceARQ_RERANK_MODEL=BAAI/bge-reranker-v2-m3  # Multilingual support

ARQ_DENSE_WEIGHT=0.6                # Semantic importanceARQ_RERANK_TOP_N=3  # Return top 3 after reranking

``````



#### Advanced Features (Optional)**How it works:**

1. Hybrid retrieval fetches top-20 candidates

```bash2. Cross-encoder reranks based on query-document relevance

# Cross-Encoder Reranking (improves relevance)3. Top-3 most relevant documents go to LLM

ARQ_RERANK_ENABLED=true

ARQ_RERANK_MODEL=BAAI/bge-reranker-v2-m3### Enable Semantic Chunking (v1.2 - Experimental)

ARQ_RERANK_TOP_N=3

Better context preservation using embedding-based splitting:

# Docling OCR (for scanned PDFs/images)

ARQ_DOCLING_OCR=true                # Enable OCR (slower but accurate)```bash

ARQ_DOCLING_TABLE_MODE=accurate     # Table extraction: fast or accurate# .env

ARQ_SEMANTIC_CHUNKING=true

# Qdrant HNSW OptimizationARQ_SEMANTIC_BREAKPOINT_TYPE=percentile  # or standard_deviation, interquartile

ARQ_QDRANT_HNSW_M=32               # Graph connectivity (16-64)```

ARQ_QDRANT_HNSW_EF_CONSTRUCT=256   # Build quality (100-512)

ARQ_QDRANT_ON_DISK=true            # Lower memory usage**Trade-offs:**

```- ✅ Better semantic coherence in chunks

- ✅ Preserves context across splits

#### Discord Bot- ⚠️ Slower indexing (~2x time)

- ⚠️ Experimental (requires langchain-experimental)

```bash

ARQ_DISCORD_TOKEN=your_bot_token_here### Using Remote Qdrant

ARQ_DISCORD_PREFIX=!                # Command prefix (optional)

``````bash

# .env

### Understanding .env SettingsARQ_QDRANT_URL=http://localhost:6333

# Or use Docker:

The `.env` file controls all aspects of ArquimedesAI's behavior. Here's how to customize it for your needs:docker run -p 6333:6333 -v $(pwd)/storage/qdrant:/qdrant/storage qdrant/qdrant:latest

```

#### Performance Profiles

### Different LLM Models

**Development/Testing** (faster, less accurate):

```bash```bash

ARQ_QDRANT_HNSW_M=16# Smaller (faster, less RAM)

ARQ_QDRANT_HNSW_EF_CONSTRUCT=100ollama pull phi3:mini

ARQ_QDRANT_ON_DISK=falseARQ_OLLAMA_MODEL=phi3:mini

ARQ_DOCLING_OCR=false

ARQ_RERANK_ENABLED=false# Larger (better quality, more RAM)

```ollama pull llama3.1:8b

ARQ_OLLAMA_MODEL=llama3.1:8b

**Production** (slower, more accurate):```

```bash

ARQ_QDRANT_HNSW_M=32### Embedding Models

ARQ_QDRANT_HNSW_EF_CONSTRUCT=256

ARQ_QDRANT_ON_DISK=true```bash

ARQ_DOCLING_OCR=true# Alternative multilingual embeddings

ARQ_RERANK_ENABLED=trueARQ_EMBED_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2

``````



#### Model Selection## 🧪 Development



You can change the LLM model by editing `ARQ_OLLAMA_MODEL`:### Code Quality

```bash

ARQ_OLLAMA_MODEL=gemma2:1b     # Fast, 1B parameters (default)```bash

ARQ_OLLAMA_MODEL=llama3.1:8b   # Slower, better quality# Format code

ARQ_OLLAMA_MODEL=mistral:7b    # Alternative optionruff format .

```

# Lint

First pull the model: `ollama pull llama3.1:8b`ruff check .



#### Retrieval Tuning# Type checking

mypy .

Adjust retrieval behavior:```

```bash

ARQ_TOP_K=5                    # Number of chunks to retrieve (3-10)### Testing

ARQ_BM25_WEIGHT=0.4           # Increase for keyword-heavy docs

ARQ_DENSE_WEIGHT=0.6          # Increase for semantic queries```bash

ARQ_RERANK_TOP_N=3            # Final results after reranking (1-5)# Run tests

```pytest



[See `.env.example` for all options →](.env.example)# With coverage

pytest --cov=. --cov-report=html

---```



## 🎯 Features## 📚 Key Features



### Document Processing### Document Processing

- **Docling HybridChunker**: Structure-aware, tokenization-optimized chunking- **Docling integration** for superior PDF parsing with table/structure preservation

- **Rich Metadata**: Page numbers, bounding boxes, section headings- **Recursive chunking** with configurable size and overlap

- **OCR Support**: Extract text from scanned PDFs and images- **Metadata preservation** (filename, page, file type)

- **Table Extraction**: Accurate parsing of complex tables

- **Format Support**: PDF, DOCX, PPTX, XLSX, Markdown, HTML, images### Retrieval

- **Qdrant vector store** with HNSW indexing

### Retrieval- **BGE-M3 embeddings** (1024-dim multilingual)

- **Hybrid Search**: BM25 (keyword) + Dense (semantic) with RRF- **Embedding cache** for fast re-indexing

- **Cross-Encoder Reranking**: Improve relevance with `bge-reranker-v2-m3`

- **HNSW Optimization**: High precision with low memory footprint### Generation

- **Configurable Weights**: Balance keyword vs. semantic search- **LangChain LCEL** for composable chains

- **Multilingual**: BGE-M3 supports 100+ languages natively- **Grounded answers** from retrieved context

- **Async Discord integration** for responsive bot

### Generation

- **Grounded Answers**: Explicit citations from source documents## 🛣️ Roadmap

- **Multi-Mode Prompts**: 4 modes (grounded, concise, critic, explain)

- **Hallucination Prevention**: Strong prompt engineering**v1.1** (Next):

- **Local LLMs**: Gemma2 1B via Ollama (fast, efficient)- [ ] Hybrid retrieval (dense + sparse BM25)

- **Flexible Models**: Easy to swap LLMs (gemma2, llama3.1, mistral)- [ ] Cross-encoder re-ranking

- [ ] Citation formatting with source highlighting

### Interfaces- [ ] Multiple answer modes (grounded, concise, critic, explain)

- **CLI Chat**: Interactive testing with mode selection- [ ] Web URL ingestion via Discord commands

- **Discord Bot**: Production deployment with async support

- **Status Command**: View configuration and system stats**v2.0** (Future):

- **Batch Indexing**: Efficient document processing- [ ] Web UI with Streamlit/Gradio

- [ ] Multi-user support

---- [ ] Query analytics dashboard

- [ ] RAG evaluation metrics

## 📊 CLI Commands

## 📝 Documentation

```bash

# Index documents- **Full Specification**: `ArquimedesAI_Spec_Full_v1.1.md`

python cli.py index                          # Incremental index- **AI Agent Guidelines**: `.github/copilot-instructions.md`

python cli.py index --rebuild                # Full rebuild- **API Documentation**: Auto-generated from docstrings

python cli.py index --data-dir ./my-docs     # Custom directory

## 🤝 Contributing

# Chat modes

python cli.py chat                           # Default (grounded)We welcome contributions! Please:

python cli.py chat --mode concise            # Brief answers

python cli.py chat --mode critic             # Verify claims1. Follow the **Constitution for AI Agents** (see spec §I-V)

python cli.py chat --mode explain            # Show reasoning2. Add Google-style docstrings to all functions/classes

3. Test changes before submitting PR

# Discord bot4. Update documentation as needed

python cli.py discord                        # Start Discord bot

## ⚖️ License

# System status

python cli.py status                         # Show configurationMIT License - see LICENSE file for details

```

## 🙏 Acknowledgments

---

- **Docling** by IBM Research for document parsing

## 🔄 Development Milestones- **LangChain** for RAG orchestration

- **Qdrant** for vector storage

### v1.3.1 (2025-10-06) - LangChain 1.0 Ready- **BAAI** for BGE embeddings

- ✅ Migrated to `langchain-ollama` official package- **Ollama** for local LLM inference

- ✅ Updated to modern retrieval API (`.invoke()` pattern)

- ✅ Secure embeddings cache (SHA-256)## 📧 Support

- ✅ HNSW optimization for better accuracy

- **Issues**: [GitHub Issues](https://github.com/edoardolobl/ArquimedesAI/issues)

### v1.3.0 (2025-10-05) - Docling Integration- **Discussions**: [GitHub Discussions](https://github.com/edoardolobl/ArquimedesAI/discussions)

- 📄 Structure-aware chunking with HybridChunker

- 🔍 Rich metadata (pages, bounding boxes, sections)---

- 📊 Accurate table extraction

- 🖼️ OCR support for images and scanned PDFs**Made with ❤️ for the open-source community**

- **LLM Integration**: The `RetrievalGeneration` class handles document retrieval and response generation using the Mistral 7b LLM.

### v1.2.0 (2025-10-05) - Advanced Retrieval- **Discord Integration**: The `DiscordChatbot` class manages interactions with Discord, receiving user messages and sending back LLM-generated responses.

- 🎯 Cross-encoder reranking

- 🧩 Optional semantic chunking## Contributing

- 🎨 Multi-mode CLI chat

Your contributions can shape ArquimedesAI's future! Dive into the contribution guidelines and join the mission.

### v1.1.0 (2025-10-05) - Hybrid Search

- 🔍 BM25 + Dense retrieval with RRF## License

- 💬 CLI chat interface

- 📝 Enhanced prompts with citationsArquimedesAI is protected under the Apache 2.0. Delve into the `LICENSE` file for intricate details.

- ⚙️ Configurable retrieval weights

## Acknowledgments

### v1.0.0 - Modern Foundation
- 🗂️ Qdrant vector store (replaces FAISS)
- 🤖 Gemma2 1B LLM (replaces Mistral 7B)
- 🌍 BGE-M3 multilingual embeddings
- ⚡ LangChain 0.3+ with LCEL
- 🎯 Modular architecture

### v0.2.0 (2024-01) - LangChain Migration
- 🔄 Migrated from Haystack to LangChain
- 📚 RAG implementation with FAISS
- 🤖 Mistral 7B integration
- 🧩 ColBERT contextual compression

### v0.1.0 (2023-08) - Initial Release
- 💬 Basic Q&A with Haystack
- 🔍 BERT-based retrieval
- 💾 SQLite database
- 🤖 Discord interface

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

---

## 📝 License

ArquimedesAI is licensed under the **Apache License 2.0**.

This means you can:
- ✅ Use commercially
- ✅ Modify and distribute
- ✅ Use privately
- ✅ Use for patent grants

See the [LICENSE](LICENSE) file for full details.

---

## 🙏 Acknowledgments

ArquimedesAI builds upon amazing open-source projects:

- **[LangChain](https://github.com/langchain-ai/langchain)** - RAG framework and LCEL
- **[Qdrant](https://github.com/qdrant/qdrant)** - Vector database
- **[Ollama](https://ollama.ai)** - Local LLM runtime
- **[Docling](https://github.com/DS4SD/docling)** - Document processing
- **[BGE-M3](https://huggingface.co/BAAI/bge-m3)** - Multilingual embeddings
- **[Gemma](https://ai.google.dev/gemma)** - Efficient language model

Special thanks to the open-source AI community! 🚀

---

## 📞 Support & Community

- 📖 **Documentation**: [Full docs](ARCHITECTURE.md) | [Quick start](QUICKSTART.md) | [Setup guide](SETUP.md)
- 💬 **Issues**: [GitHub Issues](https://github.com/edoardolobl/ArquimedesAI/issues)
- 🐛 **Bug Reports**: Use the issue template
- 💡 **Feature Requests**: Share your ideas!

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/edoardolobl">edoardolobl</a><br>
  <sub>100% Open Source • 100% Local • 100% Private</sub>
</p>
