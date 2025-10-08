"""
ArquimedesAI Streamlit Web Interface.

A user-friendly chat interface for ArquimedesAI built with Streamlit.
Provides visual routing indicators, conversation history, streaming responses,
source citations, and conversation export.

Run with: streamlit run interfaces/streamlit/app.py
"""

import streamlit as st
import uuid
import json
from datetime import datetime
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.rag_chain import RAGChain
from core.vector_store import QdrantVectorStore
from settings import settings

# Route emojis
ROUTE_EMOJIS = {
    "gtm_qa": "📚",
    "gtm_generation": "🛠️",
    "gtm_validation": "✅",
    "general_chat": "💬",
}

# Page configuration
st.set_page_config(
    page_title="ArquimedesAI Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better chat appearance
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .route-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        background-color: #f0f2f6;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def check_ollama_connection():
    """Check if Ollama is running."""
    try:
        import requests
        response = requests.get(f"{settings.ollama_base}/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


@st.cache_data(ttl=60, show_spinner=False)
def check_index_exists():
    """
    Check if Qdrant index exists.
    
    Cached for 60 seconds to avoid re-initializing embeddings on every rerun.
    This prevents HuggingFace Hub connection attempts and improves performance.
    """
    try:
        from qdrant_client import QdrantClient
        
        # Lightweight check - just verify collection exists
        # Don't create full QdrantVectorStore (which initializes embeddings)
        if settings.qdrant_url:
            client = QdrantClient(url=settings.qdrant_url)
        else:
            client = QdrantClient(path=str(settings.qdrant_path))
        
        return client.collection_exists(settings.qdrant_collection_name)
    except Exception as e:
        # Log error for debugging but don't crash
        import logging
        logging.error(f"Index check failed: {e}")
        return False


@st.cache_resource(show_spinner=False)
def load_rag_chain():
    """Load and cache the RAG chain."""
    try:
        chain = RAGChain(enable_routing=not settings.disable_routing)
        return chain
    except Exception as e:
        st.error(f"Failed to load RAG chain: {e}")
        return None


def format_route_info(route: str, confidence: float) -> str:
    """Format route information with emoji and confidence."""
    emoji = ROUTE_EMOJIS.get(route, "💬")
    route_name = route.replace("_", " ").title()
    return f"{emoji} {route_name} ({confidence:.1%})"


def export_conversation_as_json():
    """Export conversation history as JSON."""
    if "messages" not in st.session_state or len(st.session_state.messages) == 0:
        return None
    
    export_data = {
        "session_id": st.session_state.get("session_id", "unknown"),
        "timestamp": datetime.now().isoformat(),
        "messages": st.session_state.messages,
        "settings": {
            "model": settings.ollama_model,
            "routing_enabled": not settings.disable_routing,
        }
    }
    return json.dumps(export_data, indent=2, ensure_ascii=False)


def export_conversation_as_markdown():
    """Export conversation history as Markdown."""
    if "messages" not in st.session_state or len(st.session_state.messages) == 0:
        return None
    
    md_lines = [
        f"# ArquimedesAI Conversation Export",
        f"\n**Session ID:** `{st.session_state.get('session_id', 'unknown')}`",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Model:** {settings.ollama_model}",
        f"**Routing:** {'Enabled' if not settings.disable_routing else 'Disabled'}",
        "\n---\n"
    ]
    
    for i, msg in enumerate(st.session_state.messages, 1):
        role = msg["role"].title()
        content = msg["content"]
        md_lines.append(f"\n## Message {i}: {role}\n")
        md_lines.append(content)
        
        # Add route info if available
        if msg["role"] == "assistant" and msg.get("route"):
            route_text = format_route_info(msg["route"], msg.get("confidence", 0))
            md_lines.append(f"\n*Route:* {route_text}")
        
        md_lines.append("\n---\n")
    
    return "\n".join(md_lines)


async def stream_rag_response(chain, input_data):
    """Stream response from RAG chain token by token."""
    # Note: LangChain's create_retrieval_chain doesn't support streaming the final answer
    # We'll use the non-streaming invoke and simulate streaming for now
    # TODO: Switch to astream_events when RAGChain supports it
    response = await chain.retrieval_chain.ainvoke(input_data)
    
    # Simulate streaming word by word
    answer = response.get("answer", "")
    for word in answer.split():
        yield word + " "


# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ ArquimedesAI")
    st.markdown("### Configuration")
    
    # Status indicators
    st.markdown("### System Status")
    ollama_status = check_ollama_connection()
    
    # Check index only once per session to avoid Qdrant lock conflicts
    # (Multiple QdrantClient instances can't access same file path)
    if "index_checked" not in st.session_state:
        st.session_state.index_checked = True
        st.session_state.index_status = check_index_exists()
    
    index_status = st.session_state.get("index_status", False)
    
    if ollama_status:
        st.success("🟢 Ollama Connected")
    else:
        st.error("🔴 Ollama Disconnected")
        st.caption("Start Ollama: `ollama serve`")
    
    if index_status:
        st.success("🟢 Index Ready")
    else:
        st.warning("🟡 No Index Found")
        st.caption("Build index: `python cli.py index`")
    
    st.markdown("---")
    
    # Chat settings
    mode = st.selectbox(
        "Response Mode",
        ["grounded", "concise", "critic", "explain"],
        index=0,
        help="Choose how the bot should respond"
    )
    
    show_routing = st.checkbox(
        "Show Routing Info",
        value=True,
        help="Display which route was used and confidence score"
    )
    
    show_sources = st.checkbox(
        "Show Sources",
        value=False,
        help="Display retrieved document sources with metadata"
    )
    
    enable_streaming = st.checkbox(
        "Enable Streaming",
        value=True,
        help="Stream responses with typewriter effect (slower but more engaging)"
    )
    
    conversational = st.checkbox(
        "Conversational Mode",
        value=False,
        help="Remember conversation history (experimental)"
    )
    
    st.markdown("---")
    
    # Export conversation
    st.markdown("### Export Conversation")
    
    if "messages" in st.session_state and len(st.session_state.messages) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            json_data = export_conversation_as_json()
            if json_data:
                st.download_button(
                    label="📥 JSON",
                    data=json_data,
                    file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col2:
            md_data = export_conversation_as_markdown()
            if md_data:
                st.download_button(
                    label="📥 MD",
                    data=md_data,
                    file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
    else:
        st.caption("No messages to export yet")
    
    st.markdown("---")
    
    # Session info
    if "session_id" in st.session_state:
        st.caption(f"Session: `{st.session_state.session_id[:8]}...`")
        if "messages" in st.session_state:
            st.caption(f"Messages: {len(st.session_state.messages)}")
    
    # Clear conversation
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    
    st.markdown("---")
    st.caption("ArquimedesAI v2.1.0")
    st.caption("🤖 Powered by Gemma3 + Qdrant")


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


# Main chat interface
st.title("💬 ArquimedesAI Chat")
st.caption("Ask questions about your documents with intelligent routing and GTM expertise")

# Welcome message on first load
if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        st.markdown("""
        👋 **Welcome to ArquimedesAI!**
        
        I'm ready to help you with:
        - 📚 **Questions** about Google Tag Manager and your documents
        - 🛠️ **Generating** tags, triggers, and GTM configurations
        - ✅ **Validating** your existing GTM setups
        - 💬 **General** conversations about your content
        
        I automatically detect what type of help you need and route your question to specialized prompts.
        
        **Try asking:** "What is a GTM tag?" or "How do I create a pageview trigger?"
        """)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Show routing info if available and enabled
        if msg["role"] == "assistant" and show_routing and msg.get("route"):
            route_text = format_route_info(msg["route"], msg.get("confidence", 0))
            st.caption(f"**Route:** {route_text}")
        
        # Show sources if available and enabled (enhanced with metadata)
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            with st.expander(f"📄 View Sources ({len(msg['sources'])} documents)", expanded=False):
                for i, source_data in enumerate(msg["sources"], 1):
                    # Handle both dict and string sources
                    if isinstance(source_data, dict):
                        content = source_data.get("content", "")
                        metadata = source_data.get("metadata", {})
                    else:
                        content = str(source_data)
                        metadata = {}
                    
                    st.markdown(f"**Source {i}**")
                    
                    # Display metadata if available
                    if metadata:
                        meta_cols = st.columns(3)
                        if "source" in metadata:
                            meta_cols[0].caption(f"📁 {Path(metadata['source']).name}")
                        if "page" in metadata:
                            meta_cols[1].caption(f"📄 Page {metadata['page']}")
                        if "score" in metadata:
                            meta_cols[2].caption(f"🎯 Score: {metadata['score']:.3f}")
                    
                    # Display content preview
                    with st.container():
                        preview_length = 300
                        preview = content[:preview_length]
                        if len(content) > preview_length:
                            preview += "..."
                        st.text(preview)
                    
                    if i < len(msg["sources"]):
                        st.divider()


# Chat input
if prompt := st.chat_input("Ask me anything about your documents..."):
    # Check system status
    if not ollama_status:
        st.error("⚠️ Ollama is not running. Please start Ollama: `ollama serve`")
        st.stop()
    
    if not index_status:
        st.warning("⚠️ No index found. Please build the index: `python cli.py index`")
        st.stop()
    
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        try:
            # Load chain
            with st.status("🔄 Initializing...", expanded=True) as status:
                chain = load_rag_chain()
                if not chain:
                    st.error("❌ Failed to load RAG chain. Check logs for details.")
                    st.stop()
                
                status.update(label="🔍 Searching documents...", state="running")
                
                # Get response
                if conversational and len(st.session_state.messages) > 1:
                    status.update(label="💬 Processing with conversation history...", state="running")
                    # Conversational mode (experimental)
                    # Note: Conversational chains expect dict input
                    input_data = {"input": prompt}
                    conv_chain = chain.create_conversational_chain(style=mode)
                    config = {"configurable": {"session_id": st.session_state.session_id}}
                    response = conv_chain.invoke(input_data, config=config)
                else:
                    status.update(label="🤖 Generating response...", state="running")
                    # Single-turn mode - invoke() expects string query + style
                    response = chain.invoke(prompt, style=mode)
                
                status.update(label="✅ Complete!", state="complete", expanded=False)
            
            # Extract response data
            answer = response.get("answer", "No answer generated.")
            route = response.get("route", "general_chat")
            confidence = response.get("confidence", 0.0)
            sources = response.get("context", [])
            
            # Display answer with optional streaming
            if enable_streaming and answer:
                # Simulate streaming word by word
                def response_generator():
                    for word in answer.split():
                        yield word + " "
                
                displayed_answer = st.write_stream(response_generator())
            else:
                displayed_answer = answer
                st.markdown(answer)
            
            # Show routing info if enabled
            if show_routing:
                route_text = format_route_info(route, confidence)
                st.caption(f"**Route:** {route_text}")
            
            # Show sources if enabled (enhanced with metadata)
            if show_sources and sources:
                with st.expander(f"📄 View Sources ({len(sources)} documents)", expanded=False):
                    for i, doc in enumerate(sources, 1):
                        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                        
                        st.markdown(f"**Source {i}**")
                        
                        # Display metadata if available
                        if metadata:
                            meta_cols = st.columns(3)
                            if "source" in metadata:
                                meta_cols[0].caption(f"📁 {Path(metadata['source']).name}")
                            if "page" in metadata:
                                meta_cols[1].caption(f"📄 Page {metadata['page']}")
                            if "_score" in metadata:
                                meta_cols[2].caption(f"🎯 Score: {metadata['_score']:.3f}")
                        
                        # Display content preview
                        with st.container():
                            preview_length = 300
                            preview = content[:preview_length]
                            if len(content) > preview_length:
                                preview += "..."
                            st.text(preview)
                        
                        if i < len(sources):
                            st.divider()
            
            # Prepare sources for storage
            sources_data = []
            for doc in sources:
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                sources_data.append({
                    "content": content,
                    "metadata": metadata
                })
            
            # Save assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": displayed_answer if enable_streaming else answer,
                "route": route,
                "confidence": confidence,
                "sources": sources_data
            })
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.caption("💡 Try rephrasing your question or check if Ollama is running.")
            
            # Log error details
            with st.expander("🔍 Error Details", expanded=False):
                st.code(str(e))


# Footer
st.markdown("---")
st.caption("💡 **Tip:** Routing is enabled by default. The bot automatically detects GTM-specific queries and uses specialized prompts.")
