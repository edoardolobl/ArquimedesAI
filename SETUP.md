# ArquimedesAI Setup Guide (v1.2)

Complete setup instructions for getting ArquimedesAI running on your local machine.

## Prerequisites

- **Hardware**: 8-16GB RAM minimum (16GB recommended for v1.2 features)
- **OS**: macOS, Linux, or Windows with WSL2
- **Python**: 3.10 or higher
- **Disk Space**: ~5GB (3GB for models + 2GB for dependencies)

## Quick Start (5 Steps)

### 1. Install Ollama (Manual - Required)

Ollama provides the local LLM (Gemma3:4b) for answer generation.

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows (WSL2):**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

Or download from: https://ollama.ai/download

### 2. Start Ollama & Pull Model

```bash
# Start Ollama service (separate terminal)
ollama serve

# Pull Gemma3 model (~3GB)
ollama pull gemma3:latest

# Verify installation
ollama list
# Should show: gemma3:latest
```

### 3. Install Python Dependencies

```bash
cd ArquimedesAI

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (~2GB download)
pip install -r requirements.txt
```

### 4. Configure Environment

Choose your configuration level:

**Option A: Development (minimal features)**
```bash
cp .env.example .env
# Edit .env with your Discord token (optional)
```

**Option B: Production (all v1.2 features enabled)**
```bash
cp .env.production .env
# Edit .env with your Discord token (optional)
```

**Key settings in `.env`:**
- `ARQ_DISCORD_TOKEN`: Your Discord bot token (get from https://discord.com/developers/applications)
- `ARQ_DATA_DIR`: Path to your documents folder (default: `./data`)
- `ARQ_OLLAMA_BASE`: Ollama URL (default: `http://localhost:11434`)

### 5. Index Your Documents

```bash
# Place documents in ./data folder
mkdir -p data
cp /path/to/your/documents/*.pdf data/

# Build vector index (first run: ~2GB model downloads)
python cli.py index --data-dir ./data
```

**First-run downloads (automatic):**
- BGE-M3 embeddings: ~2GB (HuggingFace)
- bge-reranker-v2-m3: ~500MB (if `ARQ_RERANK_ENABLED=true`)

## Usage

### CLI Chat (Recommended for Testing)

```bash
# Default grounded mode
python cli.py chat

# Try different modes
python cli.py chat --mode concise   # Brief answers
python cli.py chat --mode critic    # Verify claims
python cli.py chat --mode explain   # Show reasoning

# Check configuration
python cli.py status
```

### Discord Bot (Production)

```bash
# Start Discord bot
python cli.py discord

# Mention the bot in Discord
@YourBot What is the main topic of the documents?
```

## Model Downloads Summary

| Model | Type | Size | Download Timing |
|-------|------|------|----------------|
| **Gemma3:4b** | LLM | ~3GB | **Manual** - `ollama pull gemma3:latest` |
| **BGE-M3** | Embeddings | ~2GB | **Automatic** - First `index` command |
| **bge-reranker-v2-m3** | Reranker | ~500MB | **Automatic** - When `ARQ_RERANK_ENABLED=true` |
| **Total** | | **~5.5GB** | ~5-10 min on fast connection |

## Configuration Comparison

### Development (.env.example)
```env
ARQ_HYBRID=true
ARQ_RERANK_ENABLED=false        # Disabled for speed
ARQ_SEMANTIC_CHUNKING=false     # Disabled for speed
ARQ_FETCH_K=50
ARQ_TOP_K=8
```
- **Pros**: Faster indexing (~2x), faster retrieval (~200ms)
- **Cons**: Lower answer quality
- **Use case**: Testing, development, low-resource machines

### Production (.env.production)
```env
ARQ_HYBRID=true
ARQ_RERANK_ENABLED=true         # Enabled for quality
ARQ_SEMANTIC_CHUNKING=true      # Enabled for quality
ARQ_FETCH_K=50
ARQ_RERANK_TOP_N=3
```
- **Pros**: Best answer quality, better citations
- **Cons**: Slower indexing (~2x), adds ~100-300ms latency
- **Use case**: Production deployments, high-quality answers

## Troubleshooting

### "Ollama connection refused"
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve
```

### "Model not found" error
```bash
# Pull the Gemma3 model
ollama pull gemma3:latest

# Verify
ollama list
```

### "Out of memory" during indexing
```bash
# Reduce chunk size in .env
ARQ_CHUNK_SIZE=500
ARQ_SEMANTIC_CHUNKING=false  # Disable semantic chunking
```

### "Slow retrieval" with reranking enabled
```bash
# Reduce reranking candidates
ARQ_FETCH_K=20  # Default: 50
ARQ_RERANK_TOP_N=3  # Keep this
```

### HuggingFace download timeout
```bash
# Models cached in: ~/.cache/huggingface/
# Delete and retry if corrupted
rm -rf ~/.cache/huggingface/hub/models--BAAI--bge-m3
python cli.py index  # Re-download
```

## Performance Tips

1. **Start with development config** - Test functionality first
2. **Enable reranking for quality** - When speed is less critical
3. **Use semantic chunking selectively** - Better for long documents
4. **Monitor memory usage** - 16GB RAM recommended for full features
5. **Cache embeddings** - Stored in `./storage/embeddings_cache/` (gitignored)

## Next Steps

1. **Test CLI chat**: `python cli.py chat`
2. **Try different modes**: `--mode concise`, `--mode critic`, `--mode explain`
3. **Set up Discord bot**: Add token to `.env`, run `python cli.py discord`
4. **Read documentation**:
   - `QUICKSTART.md` - Fast overview
   - `ARCHITECTURE.md` - System design
   - `TESTING_v1.2.md` - Testing guide
   - `ArquimedesAI_Spec_Full_v1.1.md` - Complete specification

## Support

- **Issues**: https://github.com/edoardolobl/ArquimedesAI/issues
- **Spec**: See `ArquimedesAI_Spec_Full_v1.1.md` for architecture details
- **Discord**: Check `bots/discord_bot.py` for bot setup instructions

---

**Ready to start?** Run: `python cli.py status` to verify your configuration!
