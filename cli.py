"""
ArquimedesAI CLI - Command-line interface for indexing and running the bot.

Provides commands for:
- index: Build/update vector index from documents
- discord: Start Discord bot
- chat: Interactive chat with conversational memory (v1.4)
- status: Show system status
"""

import logging
from pathlib import Path
import uuid

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from ingest.loaders import DocumentLoader
from core.vector_store import QdrantVectorStore
from core.rag_chain import load_rag_chain, RAGChain
from prompts.templates import GROUNDED_PROMPT, CONCISE_PROMPT, CRITIC_PROMPT, EXPLAIN_PROMPT
from settings import settings

# Initialize CLI app
app = typer.Typer(
    name="arquimedes",
    help="ArquimedesAI - Local RAG chatbot powered by Ollama and Qdrant",
    add_completion=False,
)

console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, console=console)],
)
logger = logging.getLogger(__name__)


@app.command()
def index(
    data_dir: Path = typer.Option(
        settings.data_dir,
        "--data-dir",
        "-d",
        help="Directory containing documents to index",
    ),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        "-r",
        help="Force rebuild index (deletes existing collection)",
    ),
):
    """
    Build or update the vector index from documents.
    
    Scans data directory for supported files (PDF, DOCX, MD, etc.),
    chunks them, and stores embeddings in Qdrant.
    """
    console.print("[bold cyan]ArquimedesAI Indexing[/bold cyan]")
    console.print(f"Data directory: {data_dir}")
    console.print()
    
    # Ensure directories exist
    settings.ensure_directories()
    
    if not data_dir.exists():
        console.print(f"[red]Error:[/red] Directory not found: {data_dir}")
        console.print("\n[yellow]Tip:[/yellow] Place documents in the data/ folder")
        raise typer.Exit(1)
    
    # Count files
    supported_exts = {".pdf", ".docx", ".pptx", ".xlsx", ".md", ".html", ".htm"}
    files = [f for f in data_dir.rglob("*") if f.suffix.lower() in supported_exts]
    
    if not files:
        console.print(f"[yellow]Warning:[/yellow] No documents found in {data_dir}")
        console.print("\n[yellow]Supported formats:[/yellow] PDF, DOCX, PPTX, XLSX, Markdown, HTML")
        raise typer.Exit(0)
    
    console.print(f"Found {len(files)} document(s)")
    console.print()
    
    try:
        # Load and chunk documents (Docling HybridChunker handles both)
        console.print("[cyan]Step 1/3:[/cyan] Loading & chunking documents...")
        loader = DocumentLoader()
        chunks = loader.load_from_directory(data_dir)
        console.print(f"✓ Loaded and chunked {len(chunks)} chunk(s)\n")
        
        # Initialize vector store
        console.print("[cyan]Step 2/3:[/cyan] Initializing Qdrant...")
        vector_store = QdrantVectorStore()
        vector_store.create_collection(force_recreate=rebuild)
        console.print("✓ Collection ready\n")
        
        # Add documents
        console.print("[cyan]Step 3/3:[/cyan] Storing embeddings...")
        vector_store.add_documents(chunks)
        console.print("✓ Embeddings stored\n")
        
        # Show stats
        stats = vector_store.get_stats()
        console.print("[bold green]✓ Indexing complete![/bold green]")
        console.print(f"  Collection: {stats.get('collection', 'N/A')}")
        console.print(f"  Vectors: {stats.get('vectors_count', 'N/A')}")
        
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        logger.exception("Indexing failed")
        raise typer.Exit(1)


@app.command()
def discord(
    token: str = typer.Option(
        None,
        "--token",
        "-t",
        envvar="ARQ_DISCORD_TOKEN",
        help="Discord bot token",
    ),
):
    """
    Start the Discord bot.
    
    Requires existing index (run 'arquimedes index' first).
    """
    console.print("[bold cyan]ArquimedesAI Discord Bot[/bold cyan]")
    console.print()
    
    # Get token
    bot_token = token or (
        settings.discord_token.get_secret_value() if settings.discord_token else None
    )
    
    if not bot_token:
        console.print("[red]Error:[/red] Discord token not provided")
        console.print("\n[yellow]Set token via:[/yellow]")
        console.print("  1. ARQ_DISCORD_TOKEN environment variable")
        console.print("  2. .env file")
        console.print("  3. --token flag")
        raise typer.Exit(1)
    
    try:
        # Load RAG chain
        console.print("[cyan]Loading RAG chain...[/cyan]")
        rag_chain = load_rag_chain()
        console.print("✓ Chain loaded\n")
        
        # Import and start bot
        from interfaces.bots.discord_bot import DiscordChatbot
        
        console.print("[cyan]Starting Discord bot...[/cyan]")
        console.print("[yellow]Press Ctrl+C to stop[/yellow]\n")
        
        bot = DiscordChatbot(bot_token, rag_chain)
        bot.run()
        
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        logger.exception("Bot startup failed")
        raise typer.Exit(1)


@app.command()
def chat(
    mode: str = typer.Option(
        "grounded",
        "--mode",
        "-m",
        help="Chat mode for general queries: grounded (default), concise, critic, or explain",
    ),
    conversational: bool = typer.Option(
        settings.enable_conversation_memory,
        "--conversational",
        "-c",
        help="Enable conversational memory (remembers context across turns)",
    ),
):
    """
    Interactive chat interface for testing RAG system (v2.0).
    
    Requires existing index (run 'arquimedes index' first).
    Type 'exit', 'quit', or 'q' to exit.
    
    Modes (for general queries):
      - grounded: Detailed answers with citations (default)
      - concise: Brief 1-3 sentence answers
      - critic: Verify if context supports claims
      - explain: Show reasoning steps
      
    Semantic Routing (v2.0 - ALWAYS ON):
      - Automatically detects GTM-specific queries and routes to specialized prompts
      - Routes: gtm_qa (Q&A), gtm_generation (tag creation), gtm_validation (tag review)
      - 89.5% accuracy with hybrid BM25+semantic routing
      - Mode flag only affects general (non-GTM) queries
      - Can be disabled via ARQ_DISABLE_ROUTING=true in .env
      
    Conversational mode (v2.0 - COMPATIBLE WITH ROUTING):
      - Remembers conversation history within session
      - Allows follow-up questions and refinement  
      - Works seamlessly with semantic routing!
      - Use --conversational flag or set ARQ_ENABLE_CONVERSATION_MEMORY=true
    """
    # Map mode to style (for display purposes)
    valid_modes = ["grounded", "concise", "critic", "explain"]
    
    # Validate mode
    mode_lower = mode.lower()
    if mode_lower not in valid_modes:
        console.print(f"[red]Error:[/red] Invalid mode '{mode}'")
        console.print(f"[yellow]Available modes:[/yellow] {', '.join(valid_modes)}")
        raise typer.Exit(1)
    
    # Display header
    routing_status = "disabled" if settings.disable_routing else "enabled"
    header = f"ArquimedesAI Chat v2.0 ({mode_lower} mode, routing {routing_status}"
    if conversational:
        header += ", conversational"
    header += ")"
    console.print(f"[bold cyan]{header}[/bold cyan]")
    console.print("[dim]Type 'exit', 'quit', or 'q' to exit[/dim]\n")
    
    try:
        # Load RAG chain (routing enabled by default in v2.0)
        console.print(f"[cyan]Loading RAG chain ({mode_lower} mode)...[/cyan]")
        
        # Create RAGChain with routing enabled (unless disabled in settings)
        rag_chain = RAGChain(enable_routing=not settings.disable_routing)
        
        if conversational:
            # Conversational mode with routing
            session_id = str(uuid.uuid4())
            conv_chain = rag_chain.create_conversational_chain(style=mode_lower)
            config = {"configurable": {"session_id": session_id}}
            
            if settings.disable_routing:
                console.print(f"✓ Chain loaded with conversational memory (session: {session_id[:8]}..., routing disabled)\n")
            else:
                console.print(f"✓ Chain loaded with conversational memory + routing (session: {session_id[:8]}...)\n")
        else:
            # Single-turn mode with routing
            if settings.disable_routing:
                console.print("✓ Chain loaded (single-turn mode, routing disabled)\n")
            else:
                console.print("✓ Chain loaded with routing (single-turn mode)\n")
        
        # Interactive loop
        turn_count = 0
        while True:
            try:
                # Get user input
                console.print("[bold cyan]You:[/bold cyan]", end=" ")
                user_input = input().strip()
                
                # Check for exit commands
                if user_input.lower() in ["exit", "quit", "q"]:
                    if conversational and turn_count > 0:
                        console.print(f"\n[dim]Session ended after {turn_count} turn(s)[/dim]")
                    console.print("[cyan]Goodbye![/cyan]")
                    break
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Process query
                console.print("[cyan]ArquimedesAI is thinking...[/cyan]")
                
                if conversational:
                    # Use conversational chain (with routing if enabled)
                    response = conv_chain.invoke({"input": user_input}, config)
                else:
                    # Use standard chain (with routing if enabled)
                    response = rag_chain.invoke(user_input, style=mode_lower)
                
                turn_count += 1
                
                # Display answer
                console.print("\n[bold green]ArquimedesAI:[/bold green]")
                console.print(response.get("answer", "No answer generated"))
                
                # Show source documents count
                source_docs = response.get("context", [])
                if source_docs:
                    console.print(f"\n[dim]({len(source_docs)} source document(s) retrieved)[/dim]")
                
                console.print()  # Blank line for readability
                
            except KeyboardInterrupt:
                console.print("\n\n[cyan]Goodbye![/cyan]")
                break
            except Exception as e:
                console.print(f"\n[red]Error processing query:[/red] {e}")
                logger.exception("Query processing failed")
                console.print()
                
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        logger.exception("Chat startup failed")
        raise typer.Exit(1)


@app.command()
def status():
    """
    Show system status and configuration.
    """
    console.print("[bold cyan]ArquimedesAI Status[/bold cyan]\n")
    
    # Configuration table
    config_table = Table(title="Configuration", show_header=True)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    
    config_table.add_row("Data Directory", str(settings.data_dir))
    config_table.add_row("Storage Directory", str(settings.storage_dir))
    config_table.add_row("Vector DB", settings.db)
    config_table.add_row("Qdrant Path", settings.qdrant_path)
    config_table.add_row("Embedding Model", settings.embed_model)
    config_table.add_row("Ollama Model", settings.ollama_model)
    config_table.add_row("Ollama URL", settings.ollama_base)
    config_table.add_row("Top K", str(settings.top_k))
    config_table.add_row("Chunk Size", str(settings.chunk_size))
    
    console.print(config_table)
    console.print()
    
    # Check Qdrant status
    try:
        vector_store = QdrantVectorStore()
        stats = vector_store.get_stats()
        
        status_table = Table(title="Qdrant Status", show_header=True)
        status_table.add_column("Metric", style="cyan")
        status_table.add_column("Value", style="green")
        
        status_table.add_row("Collection", stats.get("collection", "N/A"))
        status_table.add_row("Vectors", str(stats.get("vectors_count", 0)))
        status_table.add_row("Status", stats.get("status", "unknown"))
        
        console.print(status_table)
        
    except Exception as e:
        console.print(f"[yellow]Qdrant:[/yellow] Not accessible ({e})")


if __name__ == "__main__":
    app()
