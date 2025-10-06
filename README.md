# ArquimedesAI# ArquimedesAI# ArquimedesAI v1.2



<p align="center">

  <img src="assets/arquimedesai.jpg" alt="ArquimedesAI Logo" title="ArquimedesAI" width="35%" height="35%">

</p><p align="center"><p align="center">



<p align="center">  <img src="assets/arquimedesai.jpg" alt="ArquimedesAI Logo" title="ArquimedesAI" width="35%" height="35%">  <img src="assets/arquimedesai.jpg" alt="ArquimedesAI Logo" title="ArquimedesAI" width="35%" height="35%">

  <strong>100% Open-Source, Self-Hosted RAG Chatbot</strong><br>

  <em>Runs locally on modest hardware (8-16GB RAM)</em></p></p>

</p>



<p align="center">

  <a href="#-quick-start">Quick Start</a> •<p align="center">**100% open-source, self-hosted RAG chatbot** that runs locally on modest hardware (8-16GB RAM).

  <a href="#-features">Features</a> •

  <a href="#-architecture">Architecture</a> •  <strong>100% Open-Source, Self-Hosted RAG Chatbot</strong><br>

  <a href="#-configuration">Configuration</a> •

  <a href="#-development-milestones">Development</a> •  <em>Runs locally on modest hardware (8-16GB RAM)</em>## ✨ What's New in v1.2

  <a href="#-contributing">Contributing</a>

</p></p>



---### Advanced Retrieval Features (Optional)



## 🌟 What is ArquimedesAI?<p align="center">- 🎯 **Cross-Encoder Reranking**: Significantly improve relevance with `bge-reranker-v2-m3`



ArquimedesAI is a production-ready **Retrieval Augmented Generation (RAG)** chatbot that runs entirely on your local machine. Built with cutting-edge NLP technologies, it enables you to chat with your documents using state-of-the-art language models—all while maintaining complete privacy and control.  <a href="#-quick-start">Quick Start</a> •- 🧩 **Semantic Chunking**: Embedding-based chunking for better context preservation



### Key Highlights  <a href="#-features">Features</a> •- 🎨 **Multi-Mode CLI Chat**: Test different prompt strategies (`--mode grounded|concise|critic|explain`)



- 🔒 **100% Local & Private**: No cloud dependencies, your data never leaves your machine  <a href="#-architecture">Architecture</a> •

- 🚀 **Production-Ready**: Built with LangChain 0.3+, Qdrant vector store, and modern architecture

- 🌍 **Multilingual**: Native support for Portuguese, English, Spanish, and 100+ languages  <a href="#-configuration">Configuration</a> •All v1.2 features are **opt-in** and disabled by default for backward compatibility.

- ⚡ **Efficient**: Optimized for 8-16GB RAM systems with hybrid retrieval and HNSW indexing

- 🎯 **Advanced Retrieval**: Hybrid search (BM25 + Dense) with optional cross-encoder reranking  <a href="#-development-milestones">Development</a> •

- 📚 **Rich Document Support**: PDF, DOCX, PPTX, XLSX, Markdown, HTML, and images (with OCR)

- 🤖 **Multiple Interfaces**: CLI chat for testing, Discord bot for production use  <a href="#-contributing">Contributing</a>### v1.1 Foundation



---</p>- 🔍 **Hybrid Retrieval**: BM25 (keyword) + Dense (semantic) search with Reciprocal Rank Fusion



## 🆕 What's New in v1.3.1- 💬 **CLI Chat Interface**: Interactive testing mode (`python cli.py chat`)



### LangChain 1.0 Ready---- 📝 **Enhanced Prompts**: Explicit citation format and stronger hallucination prevention

- ✅ **Ollama Migration**: Updated to official `langchain-ollama` package

- ✅ **Modern Retrieval API**: BaseRetriever using `.invoke()` pattern- ⚙️ **Configurable Weights**: Adjust BM25/Dense balance via `.env` (`ARQ_BM25_WEIGHT`, `ARQ_DENSE_WEIGHT`)

- ✅ **Secure Caching**: SHA-256 key encoder for embeddings cache

## 🌟 What is ArquimedesAI?

### HNSW Optimization

- 🎯 **Better Accuracy**: Configurable HNSW parameters (`m=32`, `ef_construct=256`)### v1.0 Foundation

- 💾 **Lower Memory**: On-disk storage for production deployments

- ⚡ **Balanced Performance**: Optimized for both speed and qualityArquimedesAI is a production-ready **Retrieval Augmented Generation (RAG)** chatbot that runs entirely on your local machine. Built with cutting-edge NLP technologies, it enables you to chat with your documents using state-of-the-art language models—all while maintaining complete privacy and control.- 🗂️ **Advanced document parsing** with Docling (PDF, DOCX, PPTX, XLSX, Markdown, HTML)



### Docling Integration (v1.3)- 🚀 **Qdrant vector store** for persistent, production-ready indexing

- 📄 **Structure-Aware Chunking**: Preserves document hierarchy and formatting

- 🔍 **Rich Metadata**: Page numbers, bounding boxes, section context### Key Highlights- 🌍 **Multilingual support** with BGE-M3 embeddings (Portuguese, English, Spanish, and more)

- 📊 **Table Extraction**: Accurate table parsing with OCR support

- ⚡ **Modern tech stack**: LangChain 0.3, Pydantic v2, discord.py 2.4

[See full changelog →](CHANGELOG.md)

- 🔒 **100% Local & Private**: No cloud dependencies, your data never leaves your machine- 🤖 **Gemma2 1B** via Ollama for fast, efficient local inference

---

- 🚀 **Production-Ready**: Built with LangChain 0.3+, Qdrant vector store, and modern architecture- 🎯 **Modular architecture**: Clean separation of ingestion, retrieval, and generation

## 🏗️ Architecture

- 🌍 **Multilingual**: Native support for Portuguese, English, Spanish, and 100+ languages

```

User Documents → Docling Parser → HybridChunker → BGE-M3 Embeddings → Qdrant (HNSW)- ⚡ **Efficient**: Optimized for 8-16GB RAM systems with hybrid retrieval and HNSW indexing## 🏗️ Architecture

                                                                              ↓

User Query → Hybrid Retrieval → [BM25 40% + Dense 60%] → Reranker (optional) → Gemma2 LLM → Answer- 🎯 **Advanced Retrieval**: Hybrid search (BM25 + Dense) with optional cross-encoder reranking

```

- 📚 **Rich Document Support**: PDF, DOCX, PPTX, XLSX, Markdown, HTML, and images (with OCR)```

### Retrieval Pipeline

- 🤖 **Multiple Interfaces**: CLI chat for testing, Discord bot for production useUser Documents (data/) → Docling Parser → Chunks (Semantic/Character) → BGE-M3 Embeddings → Qdrant

1. **Hybrid Retrieval**: Combines BM25 (keyword) and dense (semantic) search

   - BM25: Exact term matching, handles acronyms and proper nouns                                                                                                ↓

   - Dense: Semantic similarity via BGE-M3 embeddings

   - Reciprocal Rank Fusion (RRF) merges results optimally---User Query (CLI/Discord) → HybridRetriever → [BM25 + Qdrant via RRF] → Reranker (optional) → Gemma2 LLM → Grounded Answer



2. **Reranking** (optional): Cross-encoder scores top-K candidates```

   - Model: `bge-reranker-v2-m3` (multilingual)

   - Improves relevance significantly (~100-300ms latency)## 🆕 What's New in v1.3.1



3. **Generation**: Gemma2 1B LLM generates grounded answers**Retrieval Pipeline (v1.2):**

   - Explicit citations from retrieved context

   - Multiple prompt modes: grounded, concise, critic, explain### LangChain 1.0 Ready1. **Hybrid Retrieval**: BM25 (40%) + Dense (60%) via RRF → Top-20 candidates



### Directory Structure- ✅ **Ollama Migration**: Updated to official `langchain-ollama` package2. **Reranking (optional)**: Cross-encoder scores → Top-3 most relevant



```- ✅ **Modern Retrieval API**: BaseRetriever using `.invoke()` pattern3. **Generation**: LLM answers using reranked context with citations

arquimedesai/

├── core/                  # RAG components- ✅ **Secure Caching**: SHA-256 key encoder for embeddings cache

│   ├── embedder.py       # BGE-M3 with caching

│   ├── llm_local.py      # Ollama integration### Directory Structure

│   ├── vector_store.py   # Qdrant with HNSW

│   ├── hybrid_retriever.py  # BM25 + Dense### HNSW Optimization

│   ├── reranker.py       # Cross-encoder reranking

│   └── rag_chain.py      # LangChain LCEL chains- 🎯 **Better Accuracy**: Configurable HNSW parameters (`m=32`, `ef_construct=256`)```

├── ingest/               # Document processing

│   └── loaders.py        # Docling integration- 💾 **Lower Memory**: On-disk storage for production deploymentsarquimedesai/

├── bots/                 # Interfaces

│   └── discord_bot.py    # Discord bot- ⚡ **Balanced Performance**: Optimized for both speed and quality├── core/              # RAG components (embeddings, LLM, chains, hybrid retrieval)

├── prompts/              # Prompt templates

│   └── templates.py      # 4 modes (grounded/concise/critic/explain)├── ingest/            # Document loaders and chunking

├── data/                 # Your documents (gitignored)

├── storage/              # Vector store & cache (gitignored)### Docling Integration (v1.3)├── bots/              # Discord bot interface

├── cli.py                # Command-line interface

├── settings.py           # Pydantic configuration- 📄 **Structure-Aware Chunking**: Preserves document hierarchy and formatting├── prompts/           # Prompt templates (grounded, concise, critic, explain)

└── .env                  # Environment variables

```- 🔍 **Rich Metadata**: Page numbers, bounding boxes, section context├── data/              # Your documents (gitignored)



---- 📊 **Table Extraction**: Accurate table parsing with OCR support├── storage/           # Vector store & cache (gitignored)



## 🚀 Quick Start├── cli.py             # Command-line interface (index/chat/discord/status)



### Prerequisites[See full changelog →](CHANGELOG.md)├── settings.py        # Pydantic configuration



1. **Python 3.12+** ([Download](https://www.python.org/downloads/))└── .env               # Environment variables (create from .env.example)

2. **Ollama** ([Installation Guide](https://ollama.ai))

   ```bash---```

   # macOS/Linux

   curl -fsSL https://ollama.ai/install.sh | sh

   

   # Windows## 🏗️ Architecture## 🚀 Quick Start

   # Download from https://ollama.ai/download/windows

   ```



### Installation```### Prerequisites



```bashUser Documents → Docling Parser → HybridChunker → BGE-M3 Embeddings → Qdrant (HNSW)

# 1. Clone the repository

git clone https://github.com/edoardolobl/ArquimedesAI.git                                                                              ↓1. **Python 3.12+**

cd ArquimedesAI

User Query → Hybrid Retrieval → [BM25 40% + Dense 60%] → Reranker (optional) → Gemma2 LLM → Answer2. **Ollama** ([installation guide](https://ollama.ai))

# 2. Install dependencies

pip install -r requirements.txt```



# 3. Pull Ollama model### Installation

ollama pull gemma2:1b

### Retrieval Pipeline

# 4. Configure environment

cp .env.example .env```bash

# Edit .env with your Discord token (optional) and other settings

1. **Hybrid Retrieval**: Combines BM25 (keyword) and dense (semantic) search# 1. Clone repository

# 5. Add your documents

# Place PDF, DOCX, PPTX, XLSX, MD, HTML files in data/ folder   - BM25: Exact term matching, handles acronyms and proper nounsgit clone https://github.com/edoardolobl/ArquimedesAI.git



# 6. Build the index   - Dense: Semantic similarity via BGE-M3 embeddingscd ArquimedesAI

python cli.py index

   - Reciprocal Rank Fusion (RRF) merges results optimally

# 7. Start chatting!

python cli.py chat                    # CLI interface# 2. Install dependencies

python cli.py chat --mode concise     # Brief answers

python cli.py discord                 # Discord bot2. **Reranking** (optional): Cross-encoder scores top-K candidatespip install -r requirements.txt

```

   - Model: `bge-reranker-v2-m3` (multilingual)

### First Query

   - Improves relevance significantly (~100-300ms latency)# 3. Pull Ollama model

```bash

$ python cli.py chatollama pull gemma2:1b



ArquimedesAI Chat (grounded mode)3. **Generation**: Gemma2 1B LLM generates grounded answers

Type 'exit', 'quit', or 'q' to exit

   - Explicit citations from retrieved context# 4. Create .env file

✓ Chain loaded

   - Multiple prompt modes: grounded, concise, critic, explaincp .env.example .env

You: What are the main topics in my documents?

# Edit .env and add your Discord token (optional)

ArquimedesAI: Based on your documents, the main topics are...

```### Directory Structure



---# 5. Add documents to data/ folder



## ⚙️ Configuration```# Supported: PDF, DOCX, PPTX, XLSX, Markdown, HTML



### Environment Variables (.env)arquimedesai/



ArquimedesAI uses a `.env` file for configuration. Copy `.env.example` to `.env` and customize:├── core/                  # RAG components# 6. Build index



#### Essential Settings│   ├── embedder.py       # BGE-M3 with cachingpython cli.py index



```bash│   ├── llm_local.py      # Ollama integration

# LLM Configuration

ARQ_OLLAMA_MODEL=gemma2:1b          # Ollama model (try: llama3.1, mistral, etc.)│   ├── vector_store.py   # Qdrant with HNSW# 7. Test with CLI chat (recommended)

ARQ_OLLAMA_BASE=http://localhost:11434

ARQ_OLLAMA_TEMPERATURE=0.3          # Lower = more deterministic│   ├── hybrid_retriever.py  # BM25 + Densepython cli.py chat



# Embeddings│   ├── reranker.py       # Cross-encoder reranking

ARQ_EMBED_MODEL=BAAI/bge-m3         # Multilingual embeddings (1024 dim)

│   └── rag_chain.py      # LangChain LCEL chains# 8. Or start Discord bot

# Vector Store

ARQ_QDRANT_PATH=./storage/qdrant    # Local storage path├── ingest/               # Document processingpython cli.py discord

ARQ_TOP_K=5                         # Results per query

```│   └── loaders.py        # Docling integration```



#### Hybrid Retrieval (Recommended)├── bots/                 # Interfaces



```bash│   └── discord_bot.py    # Discord bot### Configuration

ARQ_HYBRID=true                     # Enable hybrid search

ARQ_BM25_WEIGHT=0.4                 # Keyword importance├── prompts/              # Prompt templates

ARQ_DENSE_WEIGHT=0.6                # Semantic importance

```│   └── templates.py      # 4 modes (grounded/concise/critic/explain)Edit `.env` file to customize:



#### Advanced Features (Optional)├── data/                 # Your documents (gitignored)



```bash├── storage/              # Vector store & cache (gitignored)```bash

# Cross-Encoder Reranking (improves relevance)

ARQ_RERANK_ENABLED=true├── cli.py                # Command-line interface# Vector Database

ARQ_RERANK_MODEL=BAAI/bge-reranker-v2-m3

ARQ_RERANK_TOP_N=3├── settings.py           # Pydantic configurationARQ_DB=qdrant



# Docling OCR (for scanned PDFs/images)└── .env                  # Environment variablesARQ_QDRANT_PATH=./storage/qdrant  # Local persistence

ARQ_DOCLING_OCR=true                # Enable OCR (slower but accurate)

ARQ_DOCLING_TABLE_MODE=accurate     # Table extraction: fast or accurate```



# Qdrant HNSW Optimization# Embeddings

ARQ_QDRANT_HNSW_M=32               # Graph connectivity (16-64)

ARQ_QDRANT_HNSW_EF_CONSTRUCT=256   # Build quality (100-512)---ARQ_EMBED_MODEL=BAAI/bge-m3

ARQ_QDRANT_ON_DISK=true            # Lower memory usage

```



#### Discord Bot## 🚀 Quick Start# Retrieval (v1.1)



```bashARQ_HYBRID=true            # Enable hybrid retrieval (BM25 + Dense)

ARQ_DISCORD_TOKEN=your_bot_token_here

ARQ_DISCORD_PREFIX=!                # Command prefix (optional)### PrerequisitesARQ_BM25_WEIGHT=0.4        # BM25 weight (keyword matching)

```

ARQ_DENSE_WEIGHT=0.6       # Dense weight (semantic similarity)

### Understanding .env Settings

1. **Python 3.12+** ([Download](https://www.python.org/downloads/))ARQ_TOP_K=8                # Number of results

The `.env` file controls all aspects of ArquimedesAI's behavior. Here's how to customize it for your needs:

2. **Ollama** ([Installation Guide](https://ollama.ai))

#### Performance Profiles

   ```bash# Reranking (v1.2 - optional)

**Development/Testing** (faster, less accurate):

```bash   # macOS/LinuxARQ_RERANK_ENABLED=false   # Enable cross-encoder reranking

ARQ_QDRANT_HNSW_M=16

ARQ_QDRANT_HNSW_EF_CONSTRUCT=100   curl -fsSL https://ollama.ai/install.sh | shARQ_RERANK_MODEL=BAAI/bge-reranker-v2-m3

ARQ_QDRANT_ON_DISK=false

ARQ_DOCLING_OCR=false   ARQ_RERANK_TOP_N=3         # Final results after reranking

ARQ_RERANK_ENABLED=false

```   # Windows



**Production** (slower, more accurate):   # Download from https://ollama.ai/download/windows# Chunking

```bash

ARQ_QDRANT_HNSW_M=32   ```ARQ_CHUNK_SIZE=1000        # Characters per chunk

ARQ_QDRANT_HNSW_EF_CONSTRUCT=256

ARQ_QDRANT_ON_DISK=trueARQ_SEMANTIC_CHUNKING=false  # Use semantic chunking (experimental)

ARQ_DOCLING_OCR=true

ARQ_RERANK_ENABLED=true### InstallationARQ_SEMANTIC_BREAKPOINT_TYPE=percentile

```



#### Model Selection

```bash# Ollama LLM

You can change the LLM model by editing `ARQ_OLLAMA_MODEL`:

# 1. Clone the repositoryARQ_OLLAMA_MODEL=gemma2:1b

```bash

ARQ_OLLAMA_MODEL=gemma2:1b     # Fast, 1B parameters (default)git clone https://github.com/edoardolobl/ArquimedesAI.gitARQ_OLLAMA_TEMPERATURE=0.3

ARQ_OLLAMA_MODEL=llama3.1:8b   # Slower, better quality

ARQ_OLLAMA_MODEL=mistral:7b    # Alternative optioncd ArquimedesAI

```

# Discord Bot

First pull the model: `ollama pull llama3.1:8b`

# 2. Install dependenciesARQ_DISCORD_TOKEN=your_token_here

#### Retrieval Tuning

pip install -r requirements.txt```

Adjust retrieval behavior:



```bash

ARQ_TOP_K=5                    # Number of chunks to retrieve (3-10)# 3. Pull Ollama model## 📖 Usage

ARQ_BM25_WEIGHT=0.4           # Increase for keyword-heavy docs

ARQ_DENSE_WEIGHT=0.6          # Increase for semantic queriesollama pull gemma2:1b

ARQ_RERANK_TOP_N=3            # Final results after reranking (1-5)

```### CLI Commands



[See `.env.example` for all options →](.env.example)# 4. Configure environment



---cp .env.example .env```bash



## 📖 Usage# Edit .env with your Discord token (optional) and other settings# Build/update index from data/ folder



### CLI Commandspython cli.py index --data-dir ./data



```bash# 5. Add your documents

# Index documents

python cli.py index                          # Incremental index# Place PDF, DOCX, PPTX, XLSX, MD, HTML files in data/ folder# Force rebuild (deletes existing index)

python cli.py index --rebuild                # Full rebuild

python cli.py index --data-dir ./my-docs     # Custom directorypython cli.py index --rebuild



# Chat modes# 6. Build the index

python cli.py chat                           # Default (grounded)

python cli.py chat --mode concise            # Brief answerspython cli.py index# Interactive CLI chat (v1.1 - recommended for testing)

python cli.py chat --mode critic             # Verify claims

python cli.py chat --mode explain            # Show reasoningpython cli.py chat



# Discord bot# 7. Start chatting!

python cli.py discord                        # Start Discord bot

python cli.py chat                    # CLI interface# Multi-mode chat (v1.2 - test different prompt strategies)

# System status

python cli.py status                         # Show configurationpython cli.py chat --mode concise     # Brief answerspython cli.py chat --mode concise    # Brief answers

```

python cli.py discord                 # Discord botpython cli.py chat --mode critic     # Verify claims

### Discord Bot

```python cli.py chat --mode explain    # Show reasoning

1. Invite bot to your server

2. Mention the bot with your question:

   ```

   @ArquimedesAI What is the main topic of the document?### First Query# Start Discord bot

   ```

3. Bot will search indexed documents and respond with grounded answers citing sourcespython cli.py discord



### CLI Chat Modes```bash



Perfect for testing and development:$ python cli.py chat# Show system status and configuration



```bashpython cli.py status

# Grounded mode (default): Detailed answers with explicit citations

python cli.py chatArquimedesAI Chat (grounded mode)```



# Concise mode: Brief 1-3 sentence answersType 'exit', 'quit', or 'q' to exit

python cli.py chat --mode concise

### Discord Bot

# Critic mode: Verify if context supports claims

python cli.py chat --mode critic✓ Chain loaded



# Explain mode: Show reasoning steps1. Invite bot to your server

python cli.py chat --mode explain

```You: What are the main topics in my documents?2. Mention the bot with your question:



---ArquimedesAI: Based on your documents, the main topics are...   ```



## 🎯 Features```   @ArquimedesAI What is the main topic of the document?



### Document Processing   ```

- **Docling HybridChunker**: Structure-aware, tokenization-optimized chunking

- **Rich Metadata**: Page numbers, bounding boxes, section headings---3. Bot will search indexed documents and respond with grounded answers citing sources

- **OCR Support**: Extract text from scanned PDFs and images

- **Table Extraction**: Accurate parsing of complex tables

- **Format Support**: PDF, DOCX, PPTX, XLSX, Markdown, HTML, images

## ⚙️ Configuration### CLI Chat Modes (v1.2)

### Retrieval

- **Hybrid Search**: BM25 (keyword) + Dense (semantic) with RRF

- **Cross-Encoder Reranking**: Improve relevance with `bge-reranker-v2-m3`

- **Configurable Weights**: Balance keyword vs. semantic search### Environment Variables (.env)Perfect for testing and development:

- **Multilingual**: BGE-M3 supports 100+ languages natively

```bash

### Generation

- **Grounded Answers**: Explicit citations from source documentsArquimedesAI uses a `.env` file for configuration. Copy `.env.example` to `.env` and customize:# Grounded mode (default): Detailed answers with explicit citations

- **Multi-Mode Prompts**: 4 modes (grounded, concise, critic, explain)

- **Hallucination Prevention**: Strong prompt engineeringpython cli.py chat

- **Local LLMs**: Gemma2 1B via Ollama (fast, efficient)

- **Flexible Models**: Easy to swap LLMs (gemma2, llama3.1, mistral)#### Essential Settings



### Interfaces# Concise mode: Brief 1-3 sentence answers

- **CLI Chat**: Interactive testing with mode selection

- **Discord Bot**: Production deployment with async support```bashpython cli.py chat --mode concise

- **Status Command**: View configuration and system stats

- **Batch Indexing**: Efficient document processing# LLM Configuration



---ARQ_OLLAMA_MODEL=gemma2:1b          # Ollama model (try: llama3.1, mistral, etc.)# Critic mode: Verify if context supports claims



## 🔄 Development MilestonesARQ_OLLAMA_BASE=http://localhost:11434python cli.py chat --mode critic



### v1.3.1 (2025-10-06) - LangChain 1.0 ReadyARQ_OLLAMA_TEMPERATURE=0.3          # Lower = more deterministic

- ✅ Migrated to `langchain-ollama` official package

- ✅ Updated to modern retrieval API (`.invoke()` pattern)# Explain mode: Show reasoning steps

- ✅ Secure embeddings cache (SHA-256)

- ✅ HNSW optimization for better accuracy# Embeddingspython cli.py chat --mode explain



### v1.3.0 (2025-10-05) - Docling IntegrationARQ_EMBED_MODEL=BAAI/bge-m3         # Multilingual embeddings (1024 dim)```

- 📄 Structure-aware chunking with HybridChunker

- 🔍 Rich metadata (pages, bounding boxes, sections)

- 📊 Accurate table extraction

- 🖼️ OCR support for images and scanned PDFs# Vector Store## 🔧 Advanced Configuration



### v1.2.0 (2025-10-05) - Advanced RetrievalARQ_QDRANT_PATH=./storage/qdrant    # Local storage path

- 🎯 Cross-encoder reranking

- 🧩 Optional semantic chunkingARQ_TOP_K=5                         # Results per query### Enable Cross-Encoder Reranking (v1.2)

- 🎨 Multi-mode CLI chat

```

### v1.1.0 (2025-10-05) - Hybrid Search

- 🔍 BM25 + Dense retrieval with RRFSignificantly improves result relevance at the cost of ~100-300ms latency:

- 💬 CLI chat interface

- 📝 Enhanced prompts with citations#### Hybrid Retrieval (Recommended)

- ⚙️ Configurable retrieval weights

```bash

### v1.0.0 - Modern Foundation

- 🗂️ Qdrant vector store (replaces FAISS)```bash# .env

- 🤖 Gemma2 1B LLM (replaces Mistral 7B)

- 🌍 BGE-M3 multilingual embeddingsARQ_HYBRID=true                     # Enable hybrid searchARQ_RERANK_ENABLED=true

- ⚡ LangChain 0.3+ with LCEL

- 🎯 Modular architectureARQ_BM25_WEIGHT=0.4                 # Keyword importanceARQ_RERANK_MODEL=BAAI/bge-reranker-v2-m3  # Multilingual support



### v0.2.0 (2024-01) - LangChain MigrationARQ_DENSE_WEIGHT=0.6                # Semantic importanceARQ_RERANK_TOP_N=3  # Return top 3 after reranking

- 🔄 Migrated from Haystack to LangChain

- 📚 RAG implementation with FAISS``````

- 🤖 Mistral 7B integration



### v0.1.0 (2023-08) - Initial Release

- 💬 Basic Q&A with Haystack#### Advanced Features (Optional)**How it works:**

- 🔍 BERT-based retrieval

- 💾 SQLite database1. Hybrid retrieval fetches top-20 candidates

- 🤖 Discord interface

```bash2. Cross-encoder reranks based on query-document relevance

[View detailed changelog →](CHANGELOG.md)

# Cross-Encoder Reranking (improves relevance)3. Top-3 most relevant documents go to LLM

---

ARQ_RERANK_ENABLED=true

## 🧪 Testing & Development

ARQ_RERANK_MODEL=BAAI/bge-reranker-v2-m3### Enable Semantic Chunking (v1.2 - Experimental)

### Test Artifacts

ARQ_RERANK_TOP_N=3

Development test files are excluded from version control (`.gitignore`):

- `test_*.py` - Unit and integration testsBetter context preservation using embedding-based splitting:

- `verify_*.py` - Validation scripts

- `__pycache__/` - Python bytecode cache# Docling OCR (for scanned PDFs/images)



### Code QualityARQ_DOCLING_OCR=true                # Enable OCR (slower but accurate)```bash



ArquimedesAI follows strict documentation standards:ARQ_DOCLING_TABLE_MODE=accurate     # Table extraction: fast or accurate# .env

- ✅ **PEP 8** code style compliance

- ✅ **PEP 257** docstring conventionsARQ_SEMANTIC_CHUNKING=true

- ✅ **Google-style docstrings** for all public APIs

- ✅ **Type hints** using Python 3.12+ syntax# Qdrant HNSW OptimizationARQ_SEMANTIC_BREAKPOINT_TYPE=percentile  # or standard_deviation, interquartile



### Running TestsARQ_QDRANT_HNSW_M=32               # Graph connectivity (16-64)```



```bashARQ_QDRANT_HNSW_EF_CONSTRUCT=256   # Build quality (100-512)

# Verify deprecation fixes

python verify_deprecation_fixes.pyARQ_QDRANT_ON_DISK=true            # Lower memory usage**Trade-offs:**



# Test RAG pipeline```- ✅ Better semantic coherence in chunks

python test_rag_fix.py

- ✅ Preserves context across splits

# Check configuration

python cli.py status#### Discord Bot- ⚠️ Slower indexing (~2x time)

```

- ⚠️ Experimental (requires langchain-experimental)

### Code Quality Tools

```bash

```bash

# Format codeARQ_DISCORD_TOKEN=your_bot_token_here### Using Remote Qdrant

ruff format .

ARQ_DISCORD_PREFIX=!                # Command prefix (optional)

# Lint

ruff check .``````bash



# Type checking# .env

mypy .

```### Understanding .env SettingsARQ_QDRANT_URL=http://localhost:6333



---# Or use Docker:



## 🤝 ContributingThe `.env` file controls all aspects of ArquimedesAI's behavior. Here's how to customize it for your needs:docker run -p 6333:6333 -v $(pwd)/storage/qdrant:/qdrant/storage qdrant/qdrant:latest



We welcome contributions to ArquimedesAI! Whether it's:```

- 🐛 Bug reports and fixes

- ✨ Feature requests and implementations#### Performance Profiles

- 📚 Documentation improvements

- 🌍 Translations and internationalization### Different LLM Models



### How to Contribute**Development/Testing** (faster, less accurate):



1. Fork the repository```bash```bash

2. Create a feature branch (`git checkout -b feature/amazing-feature`)

3. Follow code quality standards (PEP 8, Google-style docstrings)ARQ_QDRANT_HNSW_M=16# Smaller (faster, less RAM)

4. Test your changes thoroughly

5. Commit with clear messages (`git commit -m 'Add amazing feature'`)ARQ_QDRANT_HNSW_EF_CONSTRUCT=100ollama pull phi3:mini

6. Push to your branch (`git push origin feature/amazing-feature`)

7. Open a Pull RequestARQ_QDRANT_ON_DISK=falseARQ_OLLAMA_MODEL=phi3:mini



### Development GuidelinesARQ_DOCLING_OCR=false



- Follow the **Constitution for AI Agents** (see `ArquimedesAI_Spec_Full_v1.1.md` §I-V)ARQ_RERANK_ENABLED=false# Larger (better quality, more RAM)

- Add Google-style docstrings to all functions/classes

- Test changes before submitting PR```ollama pull llama3.1:8b

- Update documentation as needed

ARQ_OLLAMA_MODEL=llama3.1:8b

---

**Production** (slower, more accurate):```

## 📝 License

```bash

ArquimedesAI is licensed under the **MIT License**.

ARQ_QDRANT_HNSW_M=32### Embedding Models

This means you can:

- ✅ Use commerciallyARQ_QDRANT_HNSW_EF_CONSTRUCT=256

- ✅ Modify and distribute

- ✅ Use privatelyARQ_QDRANT_ON_DISK=true```bash

- ✅ Sublicense

ARQ_DOCLING_OCR=true# Alternative multilingual embeddings

See the [LICENSE](LICENSE) file for full details.

ARQ_RERANK_ENABLED=trueARQ_EMBED_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2

---

``````

## 🙏 Acknowledgments



ArquimedesAI builds upon amazing open-source projects:

#### Model Selection## 🧪 Development

- **[LangChain](https://github.com/langchain-ai/langchain)** - RAG framework and LCEL

- **[Qdrant](https://github.com/qdrant/qdrant)** - Vector database

- **[Ollama](https://ollama.ai)** - Local LLM runtime

- **[Docling](https://github.com/DS4SD/docling)** - Document processing by IBM ResearchYou can change the LLM model by editing `ARQ_OLLAMA_MODEL`:### Code Quality

- **[BGE-M3](https://huggingface.co/BAAI/bge-m3)** - Multilingual embeddings by BAAI

- **[Gemma](https://ai.google.dev/gemma)** - Efficient language model by Google```bash



Special thanks to the open-source AI community! 🚀ARQ_OLLAMA_MODEL=gemma2:1b     # Fast, 1B parameters (default)```bash



---ARQ_OLLAMA_MODEL=llama3.1:8b   # Slower, better quality# Format code



## 📞 Support & CommunityARQ_OLLAMA_MODEL=mistral:7b    # Alternative optionruff format .



- 📖 **Documentation**: [Full docs](ARCHITECTURE.md) | [Quick start](QUICKSTART.md) | [Setup guide](SETUP.md)```

- 💬 **Issues**: [GitHub Issues](https://github.com/edoardolobl/ArquimedesAI/issues)

- 🐛 **Bug Reports**: Use the issue template# Lint

- 💡 **Feature Requests**: Share your ideas!

- 📚 **Discussions**: [GitHub Discussions](https://github.com/edoardolobl/ArquimedesAI/discussions)First pull the model: `ollama pull llama3.1:8b`ruff check .



---



<p align="center">#### Retrieval Tuning# Type checking

  Made with ❤️ by <a href="https://github.com/edoardolobl">edoardolobl</a><br>

  <sub>100% Open Source • 100% Local • 100% Private</sub>mypy .

</p>

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
