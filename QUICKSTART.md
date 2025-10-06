# 🚀 Quick Start Checklist

## Prerequisites Setup

- [ ] **Python 3.12+** installed
  ```bash
  python --version  # Should be 3.12 or higher
  ```

- [ ] **Ollama** installed
  ```bash
  # macOS
  brew install ollama
  
  # Or download from https://ollama.ai
  ```

- [ ] **Git** installed (for cloning repo)

---

## Installation Steps

### 1. Clone & Navigate
```bash
git clone https://github.com/edoardolobl/ArquimedesAI.git
cd ArquimedesAI
```
- [ ] Repository cloned
- [ ] Inside ArquimedesAI directory

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
- [ ] All packages installed without errors
- [ ] Check for any warning messages

### 3. Pull Ollama Model
```bash
ollama pull gemma2:1b
```
- [ ] Model downloaded (~700MB)
- [ ] Test: `ollama list` shows gemma2:1b

### 4. Configure Environment
```bash
cp .env.example .env
```
Then edit `.env`:
```bash
ARQ_DISCORD_TOKEN=your_actual_discord_token_here
```
- [ ] `.env` file created
- [ ] Discord token added
- [ ] Other settings reviewed (optional)

### 5. Prepare Documents
```bash
# Documents go in data/ folder
mkdir -p data
cp /path/to/your/documents/*.pdf data/
```
- [ ] `data/` folder created
- [ ] At least 1 document added
- [ ] Supported formats: PDF, DOCX, PPTX, XLSX, MD, HTML

### 6. Build Index
```bash
python cli.py index
```
Expected output:
```
✓ Loaded X document(s)
✓ Created Y chunk(s)
✓ Collection ready
✓ Embeddings stored
✓ Indexing complete!
```
- [ ] Indexing completed without errors
- [ ] Check `storage/qdrant/` directory created

### 7. Test Status
```bash
python cli.py status
```
- [ ] Configuration table displayed
- [ ] Qdrant status shows vectors > 0
- [ ] No error messages

### 8. Start Discord Bot
```bash
python cli.py discord
```
Expected output:
```
✓ Chain loaded
✓ Bot is ready: ArquimedesAI#1234
Listening for mentions...
```
- [ ] Bot starts without errors
- [ ] Bot shows as online in Discord

---

## First Test

### In Discord:
1. Mention the bot: `@ArquimedesAI What is this document about?`
2. Wait for response
3. Verify answer is based on your documents

- [ ] Bot responds to mention
- [ ] Answer is relevant to documents
- [ ] No timeout errors

---

## Troubleshooting Common Issues

### "No module named 'X'"
```bash
pip install --upgrade -r requirements.txt
```

### "Qdrant collection not found"
```bash
# Run indexing first
python cli.py index
```

### "Discord token not provided"
```bash
# Check .env file exists and has token
cat .env | grep DISCORD_TOKEN
```

### "Ollama connection refused"
```bash
# Start Ollama service
ollama serve

# In another terminal:
ollama list  # Verify model is there
```

### Bot doesn't respond
- Check bot has "Read Messages" permission
- Check bot has "Send Messages" permission
- Check "Message Content Intent" enabled in Discord Developer Portal

---

## Optional: Development Tools

### Run with Makefile
```bash
# Instead of python cli.py commands:
make index          # Build index
make discord        # Start bot
make status         # Check status
make clean          # Clean cache
```

### Enable Debug Logging
In `.env`:
```bash
ARQ_LOG_LEVEL=DEBUG
```

### Try Different Models
```bash
# Smaller/faster
ollama pull phi3:mini
# Update .env: ARQ_OLLAMA_MODEL=phi3:mini

# Larger/better
ollama pull llama3.1:8b  
# Update .env: ARQ_OLLAMA_MODEL=llama3.1:8b
```

---

## Success Indicators ✅

You're ready when:

- ✅ `python cli.py status` shows vectors in collection
- ✅ Discord bot starts and shows "ready" message
- ✅ Bot responds to Discord mentions
- ✅ Answers are grounded in your documents
- ✅ No errors in terminal output

---

## Next Steps

Once everything works:

1. **Add more documents** to `data/` and run `python cli.py index --rebuild`
2. **Customize prompts** in `prompts/templates.py`
3. **Adjust settings** in `.env` (chunk size, top-k, temperature)
4. **Read the spec** in `ArquimedesAI_Spec_Full_v1.1.md`
5. **Check roadmap** in `README.md` for v1.1 features

---

## Getting Help

- **Documentation**: `README.md`, `MIGRATION.md`
- **Architecture**: `ArquimedesAI_Spec_Full_v1.1.md`
- **Issues**: https://github.com/edoardolobl/ArquimedesAI/issues
- **Discussions**: https://github.com/edoardolobl/ArquimedesAI/discussions

---

**Estimated Setup Time**: 10-15 minutes ⏱️

**Happy querying! 🎉**
