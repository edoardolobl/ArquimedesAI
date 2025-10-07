"""
Base prompt templates for ArquimedesAI (v1.4).

Provides domain-agnostic, reusable prompt components following 
best practices from prompt engineering research:
- Clear role definition
- Language enforcement (Brazilian Portuguese)
- Structured output format
- Delimiters for clarity
- Task decomposition support

These base components can be extended by domain-specific prompt modules
(e.g., gtm_prompts.py for GTM taxonomy).
"""

from typing import Literal

# ============================================================================
# LANGUAGE ENFORCEMENT (Best Practice: Explicit Language Requirements)
# ============================================================================

LANGUAGE_INSTRUCTION_PT_BR = """
IMPORTANTE: Você DEVE responder SEMPRE em Português Brasileiro.

REGRAS DE IDIOMA:
- Todas as explicações, descrições e textos gerados devem ser em Português Brasileiro
- Termos técnicos podem permanecer em inglês quando apropriado (ex: dataLayer, API, JSON)
- Nomes técnicos de plataformas e ferramentas podem permanecer em inglês (ex: Google Analytics, Meta Pixel)
- Exemplos e justificativas devem ser em Português Brasileiro

Se você não souber responder em Português Brasileiro, diga: "Desculpe, não posso ajudar com isso."
""".strip()


# ============================================================================
# BASE SYSTEM PROMPTS (Best Practice: Clear Role Definition)
# ============================================================================

def create_base_system_prompt(
    role: str,
    task_description: str,
    constraints: list[str] | None = None,
    output_format: str | None = None,
) -> str:
    """
    Create a base system prompt with best practices.
    
    Args:
        role: The AI's role (e.g., "assistente de taxonomia GTM")
        task_description: What the AI should do
        constraints: List of constraints/rules to follow
        output_format: Expected output format description
        
    Returns:
        Formatted system prompt with language enforcement
        
    Example:
        >>> create_base_system_prompt(
        ...     role="assistente RAG",
        ...     task_description="Responder perguntas baseado em documentos fornecidos",
        ...     constraints=["Nunca inventar informações", "Citar fontes"],
        ...     output_format="Resposta em formato de texto com citações"
        ... )
    """
    prompt_parts = [
        f"Você é um {role}.",
        "",
        f"SUA TAREFA:",
        task_description,
        "",
        LANGUAGE_INSTRUCTION_PT_BR,
    ]
    
    if constraints:
        prompt_parts.extend([
            "",
            "RESTRIÇÕES E REGRAS:",
            *[f"- {constraint}" for constraint in constraints],
        ])
    
    if output_format:
        prompt_parts.extend([
            "",
            "FORMATO DE SAÍDA:",
            output_format,
        ])
    
    return "\n".join(prompt_parts)


# ============================================================================
# GENERAL CHAT SYSTEM PROMPT (Domain-Agnostic)
# ============================================================================

GENERAL_CHAT_SYSTEM_PROMPT = create_base_system_prompt(
    role="ArquimedesAI, assistente RAG (Retrieval-Augmented Generation)",
    task_description="""Responder perguntas baseado APENAS no contexto fornecido pelos documentos recuperados.
Você deve ajudar o usuário a encontrar informações nos documentos de forma clara e precisa.""",
    constraints=[
        "Responda APENAS com base no contexto fornecido",
        "NUNCA invente informações ou use conhecimento externo",
        "Se a informação não estiver no contexto, diga claramente que não pode responder",
        "Cite passagens específicas quando fizer afirmações",
        "Seja preciso e objetivo",
    ],
    output_format="Resposta em Português Brasileiro com citações dos documentos quando relevante.",
)


# ============================================================================
# USER PROMPT TEMPLATES (Best Practice: Structured with Delimiters)
# ============================================================================

def create_user_prompt(
    context_documents: str,
    user_query: str,
    instructions: str | None = None,
    examples: str | None = None,
) -> str:
    """
    Create a structured user prompt with clear delimiters.
    
    Args:
        context_documents: Retrieved documents (formatted)
        user_query: User's question/request
        instructions: Optional specific instructions for this query
        examples: Optional few-shot examples
        
    Returns:
        Formatted user prompt with clear sections
        
    Best Practice: Uses delimiters (===, ---, ##) to separate sections
    for better LLM comprehension.
    """
    prompt_parts = []
    
    # Context section (always present)
    prompt_parts.extend([
        "## CONTEXTO RECUPERADO DOS DOCUMENTOS",
        "",
        context_documents,
        "",
        "=" * 80,
    ])
    
    # Examples section (if provided - few-shot prompting)
    if examples:
        prompt_parts.extend([
            "",
            "## EXEMPLOS",
            "",
            examples,
            "",
            "=" * 80,
        ])
    
    # Instructions section (if provided)
    if instructions:
        prompt_parts.extend([
            "",
            "## INSTRUÇÕES ESPECÍFICAS",
            "",
            instructions,
            "",
            "=" * 80,
        ])
    
    # Query section (always present)
    prompt_parts.extend([
        "",
        "## PERGUNTA DO USUÁRIO",
        "",
        user_query,
        "",
        "=" * 80,
        "",
        "## SUA RESPOSTA",
        "",
    ])
    
    return "\n".join(prompt_parts)


# ============================================================================
# STYLE MODIFIERS (Compatible with existing templates.py modes)
# ============================================================================

STYLE_INSTRUCTIONS: dict[Literal["grounded", "concise", "critic", "explain"], str] = {
    "grounded": """
Ao responder:
- Cite passagens exatas do contexto entre aspas quando fizer afirmações
- Indique claramente quando a informação não está no contexto
- Seja preciso e fundamentado
""".strip(),
    
    "concise": """
Ao responder:
- Seja breve e direto (1-3 sentenças)
- Foque nos fatos essenciais
- Evite explicações longas
""".strip(),
    
    "critic": """
Ao responder:
- Verifique se cada afirmação está suportada pelo contexto
- Indique o nível de confiança em cada parte da resposta
- Aponte explicitamente se algo não está no contexto
- Seja crítico e rigoroso na verificação
""".strip(),
    
    "explain": """
Ao responder:
- Mostre seu raciocínio passo a passo
- Explique como chegou à resposta
- Indique quais partes do contexto fundamentam cada conclusão
- Seja didático e detalhado
""".strip(),
}


def apply_style_modifier(
    base_prompt: str,
    style: Literal["grounded", "concise", "critic", "explain"],
) -> str:
    """
    Apply a style modifier to a base prompt.
    
    Args:
        base_prompt: Base system or user prompt
        style: Style to apply (grounded, concise, critic, explain)
        
    Returns:
        Prompt with style instructions appended
        
    Example:
        >>> system = create_base_system_prompt(...)
        >>> styled = apply_style_modifier(system, "concise")
    """
    style_instruction = STYLE_INSTRUCTIONS[style]
    return f"{base_prompt}\n\n{style_instruction}"


# ============================================================================
# CHAIN-OF-THOUGHT PROMPTING (Best Practice for Complex Tasks)
# ============================================================================

COT_INSTRUCTION = """
Antes de responder, pense passo a passo:
1. Analise a pergunta e identifique o que está sendo pedido
2. Revise o contexto fornecido e identifique informações relevantes
3. Organize as informações encontradas
4. Formule sua resposta baseada apenas no que encontrou

Mostre seu raciocínio de forma clara e estruturada.
""".strip()


def add_chain_of_thought(prompt: str) -> str:
    """
    Add chain-of-thought instruction to a prompt.
    
    Best Practice: Improves reasoning for complex tasks like
    validation or multi-step generation.
    
    Args:
        prompt: Base prompt to enhance
        
    Returns:
        Prompt with CoT instruction
    """
    return f"{prompt}\n\n{COT_INSTRUCTION}"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_context_documents(documents: list[dict]) -> str:
    """
    Format retrieved documents for prompt context.
    
    Args:
        documents: List of document dicts with 'page_content' and optional 'metadata'
        
    Returns:
        Formatted string with numbered documents
        
    Best Practice: Clear delimiters and numbering for reference.
    """
    if not documents:
        return "Nenhum documento relevante foi encontrado."
    
    formatted_docs = []
    for i, doc in enumerate(documents, 1):
        content = doc.get("page_content", "")
        metadata = doc.get("metadata", {})
        
        doc_parts = [f"[Documento {i}]"]
        
        # Add source info if available
        source = metadata.get("source", "")
        if source:
            doc_parts.append(f"Fonte: {source}")
        
        # Add page info if available
        page = metadata.get("page", "")
        if page:
            doc_parts.append(f"Página: {page}")
        
        doc_parts.append("")
        doc_parts.append(content)
        doc_parts.append("")
        doc_parts.append("-" * 40)
        
        formatted_docs.append("\n".join(doc_parts))
    
    return "\n\n".join(formatted_docs)
