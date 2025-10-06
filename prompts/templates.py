"""
Prompt templates for different answer modes.

Per spec §10, provides grounded, concise, critic, and explain modes.
Also includes Pydantic schemas for structured citations (Phase 1 v1.4).
"""

from pydantic import BaseModel, Field

# System prompt (common across all modes)
SYSTEM_PROMPT = """You are ArquimedesAI, a helpful RAG (Retrieval-Augmented Generation) assistant.

Your role is to answer questions based ONLY on the provided context from documents.
You must never make up information or use knowledge not present in the context."""

# Grounded mode - default, citation-focused
GROUNDED_PROMPT = """Answer the following question based only on the provided context.

**Important instructions:**
1. Quote exact passages from the context in quotation marks when making claims
2. If the answer is not fully contained in the context, respond: "I cannot fully answer this based on the provided documents."
3. Never make up information or use knowledge outside the context
4. Be accurate and cite specific passages

<context>
{context}
</context>

Question: {input}

Answer:"""

# Concise mode - brief answers
CONCISE_PROMPT = """Answer the following question based only on the provided context.

Provide a brief, direct answer in 1-3 sentences. Focus on the key facts.

<context>
{context}
</context>

Question: {input}

Brief answer:"""

# Critic mode - verify claims
CRITIC_PROMPT = """Analyze whether the following question can be answered from the context.

For each potential claim:
1. Check if it's supported by the context
2. Flag any gaps or ambiguities
3. Note what additional information would be needed

<context>
{context}
</context>

Question: {input}

Analysis:"""

# Explain mode - show reasoning
EXPLAIN_PROMPT = """Answer the question and explain your reasoning.

1. State your answer
2. Explain which parts of the context support it
3. Mention any limitations or uncertainties

<context>
{context}
</context>

Question: {input}

Answer with explanation:"""


# ===================================================================
# Structured Output Schemas (Phase 1 v1.4)
# ===================================================================

class Citation(BaseModel):
    """
    A single citation with source ID and verbatim quote.
    
    Used with .with_structured_output() for Gemma3 models.
    """
    source_id: int = Field(
        ..., 
        description="The integer ID of the source document (0-indexed)"
    )
    quote: str = Field(
        ..., 
        description="VERBATIM quote from the source that supports the answer"
    )


class QuotedAnswer(BaseModel):
    """
    Answer with citations using verbatim quotes.
    
    This is the main schema for structured citations in ArquimedesAI.
    Uses Gemma3's .with_structured_output() for reliable citation extraction.
    """
    answer: str = Field(
        ..., 
        description="The answer to the question, based ONLY on the provided sources"
    )
    citations: list[Citation] = Field(
        ..., 
        description="List of citations with source IDs and verbatim quotes that justify the answer"
    )


class CitedAnswer(BaseModel):
    """
    Simple answer with source IDs (alternative to QuotedAnswer).
    
    Lighter-weight option that just tracks which sources were used,
    without extracting full quotes.
    """
    answer: str = Field(
        ..., 
        description="The answer to the question, based ONLY on the provided sources"
    )
    citations: list[int] = Field(
        ..., 
        description="List of source IDs (integers) that justify the answer"
    )


def format_docs_with_id(docs: list) -> str:
    """
    Format documents with source IDs for citation tracking.
    
    Args:
        docs: List of LangChain Document objects
        
    Returns:
        Formatted string with Source ID, title/source, and content
    """
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Unknown')
        content = doc.page_content
        
        doc_str = f"""Source ID: {i}
Source: {source}
Content: {content}"""
        formatted.append(doc_str)
    
    return "\n\n".join(formatted)
