"""
GTM-specific prompts for ArquimedesAI (v1.4).

Domain-specific prompts for Google Tag Manager (GTM) taxonomy use cases:
- Q&A: Answer questions about GTM concepts, rules, best practices
- Generation: Create GTM tags, triggers, variables following style guide
- Validation: Review and correct GTM configurations against guidelines

Uses synthetic data from monks_gtm_style_guide_v1.0.md v1.1.
"""

from typing import Literal
from prompts.base_prompts import (
    create_base_system_prompt,
    create_user_prompt,
    add_chain_of_thought,
    LANGUAGE_INSTRUCTION_PT_BR,
)

# ============================================================================
# GTM Q&A PROMPT (Answer questions about GTM taxonomy)
# ============================================================================

_GTM_QA_SYSTEM = create_base_system_prompt(
    role="especialista em taxonomia do Google Tag Manager (GTM)",
    task_description="""Responder perguntas sobre conceitos, regras, nomenclaturas e boas práticas de GTM 
baseado APENAS no guia de estilo Monks fornecido no contexto.

Você deve ajudar o usuário a entender:
- Regras de nomenclatura (prefixos, formatos, convenções)
- Dicionários de referência (Páginas, Eventos, Plataformas, Tipos de Variáveis)
- Estrutura e organização de Tags, Triggers e Variáveis
- Boas práticas e padrões recomendados""",
    constraints=[
        "Responda APENAS com base no guia de estilo fornecido no contexto",
        "NUNCA invente regras ou convenções não documentadas no guia",
        "Se a informação não estiver no guia, diga claramente que não pode responder",
        "Cite seções específicas do guia quando fizer afirmações (ex: 'Seção 4.1 - Tags')",
        "Use exemplos do guia quando disponíveis",
        "Seja preciso com nomenclaturas e formatos",
    ],
    output_format="""Resposta estruturada em Português Brasileiro com:
- Explicação clara do conceito
- Citações do guia (seção + passagem)
- Exemplos práticos quando relevante
- Referências aos dicionários quando aplicável""",
)

# Complete prompt template with context and input placeholders for LangChain
GTM_QA_SYSTEM_PROMPT = f"""{_GTM_QA_SYSTEM}

<context>
{{context}}
</context>

Question: {{input}}

Answer:"""


# Few-shot examples for GTM Q&A (extracted from guide)
GTM_QA_EXAMPLES = """
Exemplo 1:
Pergunta: "O que é uma tag?"
Resposta: "Uma tag é um dos três tipos de entidades no GTM (junto com Triggers e Variáveis). 
Segundo o guia (Seção 4.1), tags devem seguir o formato: `<página> - <evento> - <plataforma>`.
Por exemplo: 'Checkout - Submit - GA4' ou 'Various - Click - Meta'."

Exemplo 2:
Pergunta: "Quando usar o prefixo NÃO PUBLICAR?"
Resposta: "Segundo o guia (Seção 4.1), o prefixo 'NÃO PUBLICAR - ' deve ser usado em tags 
que não devem ser publicadas em produção, como tags de teste ou configurações experimentais.
Exemplo: 'NÃO PUBLICAR - Various - Click - GA4'."

Exemplo 3:
Pergunta: "Quais são os tipos de variáveis disponíveis?"
Resposta: "Segundo o dicionário de tipos de variáveis (Seção 8.4), existem 4 tipos principais:
- DL (Data Layer): Acessa diretamente o dataLayer, formato 'NomeVariavel'
- DL v2 (Data Layer Variable v2): Segunda versão, formato 'eventModel.NomeVariavel'
- Cookie: Lê cookies, formato 'NomeCookie'
- Custom JS (JavaScript Personalizado): Código customizado, formato descritivo"
""".strip()


def create_gtm_qa_prompt(context_documents: str, user_query: str) -> str:
    """Create a GTM Q&A prompt with examples."""
    return create_user_prompt(
        context_documents=context_documents,
        user_query=user_query,
        instructions="Use os dicionários de referência e regras do guia para responder.",
        examples=GTM_QA_EXAMPLES,
    )


# ============================================================================
# GTM GENERATION PROMPT (Create tags, triggers, variables)
# ============================================================================

_GTM_GENERATION_SYSTEM = create_base_system_prompt(
    role="especialista em criação de taxonomia GTM seguindo o guia de estilo Monks",
    task_description="""Criar nomes de Tags, Triggers ou Variáveis do Google Tag Manager seguindo 
RIGOROSAMENTE as regras de nomenclatura e formatos do guia de estilo fornecido.

Você deve gerar nomenclaturas que:
- Seguem os formatos exatos do guia (ex: `<página> - <evento> - <plataforma>` para tags)
- Usam valores válidos dos dicionários de referência
- Respeitam as convenções de capitalização e formatação
- Aplicam prefixos quando necessário (ex: 'NÃO PUBLICAR - ', 'DL', 'Custom')""",
    constraints=[
        "SEMPRE siga os formatos EXATOS do guia de estilo",
        "Use APENAS valores dos dicionários de referência (Seções 8.1-8.4) quando aplicável",
        "Verifique os exemplos de conversão (Seção 9) antes de gerar",
        "NUNCA invente valores não documentados nos dicionários",
        "Mantenha capitalização consistente com os exemplos",
        "Se faltar informação crítica, peça esclarecimento ao usuário",
    ],
    output_format="""Resposta em Português Brasileiro com:
- Nome gerado seguindo o formato do guia
- Justificativa referenciando seção e regras aplicadas
- Valores dos dicionários usados (quando aplicável)
- Exemplo similar do guia (quando disponível)""",
)

# Complete prompt template with context and input placeholders for LangChain
GTM_GENERATION_SYSTEM_PROMPT = f"""{_GTM_GENERATION_SYSTEM}

<context>
{{context}}
</context>

Question: {{input}}

Generated GTM Name:"""


# Few-shot examples for GTM Generation (extracted from guide Section 9)
GTM_GENERATION_EXAMPLES = """
Exemplo 1 - Tag Básica:
Input: "Criar tag para evento de carregar carrinho no GA4"
Output: "Cart - Load - GA4"
Justificativa: "Formato <página> - <evento> - <plataforma> (Seção 4.1). 
Página='Cart' (Seção 8.1), Evento='Load' (Seção 8.2), Plataforma='GA4' (Seção 8.3)."

Exemplo 2 - Tag Multi-Plataforma:
Input: "Criar tag para clique em várias páginas enviando para GA4 e Meta"
Output: "Various - Click - GA4 + Meta"
Justificativa: "Usa 'Various' para múltiplas páginas (Seção 8.1), 'Click' para clique (Seção 8.2), 
concatenação de plataformas com '+' (Seção 4.1)."

Exemplo 3 - Variável Data Layer:
Input: "Criar variável para ID do produto no dataLayer"
Output: "productId"
Justificativa: "Tipo 'DL' (Data Layer) usa formato simples sem prefixo (Seção 8.4). 
Nome descritivo em camelCase."

Exemplo 4 - Trigger Básico:
Input: "Criar trigger para envio de formulário na página de checkout"
Output: "Checkout - Submit"
Justificativa: "Formato <página> - <evento> (Seção 4.2). 
Página='Checkout' (Seção 8.1), Evento='Submit' (Seção 8.2)."
""".strip()


def create_gtm_generation_prompt(context_documents: str, user_query: str) -> str:
    """Create a GTM Generation prompt with few-shot examples."""
    return create_user_prompt(
        context_documents=context_documents,
        user_query=user_query,
        instructions="Use os dicionários de referência e exemplos de conversão do guia para gerar nomenclaturas corretas.",
        examples=GTM_GENERATION_EXAMPLES,
    )


# ============================================================================
# GTM VALIDATION PROMPT (Review and correct configurations)
# ============================================================================

_GTM_VALIDATION_SYSTEM = create_base_system_prompt(
    role="auditor de taxonomia GTM especializado no guia de estilo Monks",
    task_description="""Revisar e corrigir nomes de Tags, Triggers ou Variáveis do Google Tag Manager,
identificando desvios das regras do guia de estilo e sugerindo correções.

Você deve analisar:
- Conformidade com formatos obrigatórios
- Uso correto de valores dos dicionários
- Aplicação de prefixos quando necessário
- Capitalização e formatação
- Erros comuns documentados no guia""",
    constraints=[
        "Compare RIGOROSAMENTE com as regras do guia de estilo",
        "Verifique os erros comuns (Seção 10) antes de validar",
        "Cite a seção e regra específica que foi violada",
        "Forneça correção seguindo o formato exato do guia",
        "Use exemplos de conversão (Seção 9) como referência",
        "Se a nomenclatura estiver correta, confirme explicitamente",
    ],
    output_format="""Resposta estruturada em Português Brasileiro com:
1. **Status**: CORRETO ou INCORRETO
2. **Problemas Identificados** (se houver):
   - Descrição do erro
   - Seção violada
   - Tipo de erro (formato, dicionário, capitalização, etc.)
3. **Correção Sugerida** (se houver):
   - Nome corrigido
   - Justificativa da correção
   - Regra/seção aplicada
4. **Exemplo Similar** (quando disponível)""",
)

# Complete prompt template with context and input placeholders for LangChain
GTM_VALIDATION_SYSTEM_PROMPT = f"""{_GTM_VALIDATION_SYSTEM}

<context>
{{context}}
</context>

Question: {{input}}

Validation Analysis:"""


# Few-shot examples for GTM Validation (extracted from guide Section 10)
GTM_VALIDATION_EXAMPLES = """
Exemplo 1 - Erro de Formato de Tag:
Input: "Validar: GA4CartPurchase"
Análise:
1. **Status**: INCORRETO
2. **Problemas Identificados**:
   - Erro de formato: não segue o padrão `<página> - <evento> - <plataforma>` (Seção 4.1)
   - Tipo: Formato de Tag Incorreto (Seção 10.1, Erro #1)
3. **Correção Sugerida**:
   - Nome corrigido: "Cart - Purchase - GA4"
   - Justificativa: Separação por hífen com espaços, ordem correta dos componentes
   - Regra: Seção 4.1 - Formato obrigatório para tags
4. **Exemplo Similar**: "Checkout - Submit - GA4" (Seção 9.1)

Exemplo 2 - Uso de Valor Não Padronizado:
Input: "Validar: Home Page - View - Meta"
Análise:
1. **Status**: INCORRETO
2. **Problemas Identificados**:
   - Página='Home Page' não consta no dicionário (Seção 8.1)
   - Evento='View' não consta no dicionário (Seção 8.2)
   - Tipo: Valores Não Padronizados (Seção 10.1, Erro #2)
3. **Correção Sugerida**:
   - Nome corrigido: "Homepage - Pageview - Meta"
   - Justificativa: Usa valores válidos dos dicionários (Página='Homepage', Evento='Pageview')
   - Regras: Seções 8.1 (Páginas) e 8.2 (Eventos)
4. **Exemplo Similar**: "Homepage - Load - GA4" (Seção 9.1)

Exemplo 3 - Nomenclatura Correta:
Input: "Validar: Checkout - Submit - GA4"
Análise:
1. **Status**: CORRETO
2. **Justificativa**:
   - Formato correto: `<página> - <evento> - <plataforma>` (Seção 4.1)
   - Valores válidos: Página='Checkout' (Seção 8.1), Evento='Submit' (Seção 8.2), Plataforma='GA4' (Seção 8.3)
   - Capitalização correta
3. **Referência**: Exemplo idêntico encontrado na Seção 9.1
""".strip()


def create_gtm_validation_prompt(context_documents: str, user_query: str) -> str:
    """Create a GTM Validation prompt with CoT and examples."""
    base_prompt = create_user_prompt(
        context_documents=context_documents,
        user_query=user_query,
        instructions="Revise a nomenclatura contra as regras e erros comuns do guia. Seja rigoroso e preciso.",
        examples=GTM_VALIDATION_EXAMPLES,
    )
    # Add chain-of-thought for systematic validation
    return add_chain_of_thought(base_prompt)


# ============================================================================
# ROUTING UTTERANCES (PT-BR examples for semantic router)
# ============================================================================

# Extracted from GTM guide - Questions that should route to GTM_QA
GTM_QA_UTTERANCES = [
    # General GTM concepts
    "O que é uma tag?",
    "O que é um trigger?",
    "O que é uma variável?",
    "Como organizar tags no GTM?",
    "Quais são as regras de nomenclatura?",
    
    # Prefixes and formats
    "Quando usar o prefixo NÃO PUBLICAR?",
    "Qual o formato de uma tag?",
    "Qual o formato de um trigger?",
    "Como nomear variáveis?",
    
    # Dictionaries
    "Quais páginas estão no dicionário?",
    "Quais eventos são suportados?",
    "Quais plataformas posso usar?",
    "Quais tipos de variáveis existem?",
    
    # Rules and guidelines
    "Como funciona a capitalização?",
    "Posso usar underscore nos nomes?",
    "Como separar múltiplas plataformas?",
    "O que significa 'Various' em páginas?",
    
    # Best practices
    "Quais são as boas práticas de GTM?",
    "Como evitar erros comuns?",
    "Por que usar o guia de estilo?",
]

# Extracted from GTM guide - Requests that should route to GTM_GENERATION
GTM_GENERATION_UTTERANCES = [
    # Tag creation
    "Criar tag para carrinho ao carregar",
    "Gerar tag de clique no checkout",
    "Preciso de uma tag para purchase no GA4",
    "Como criar tag multi-plataforma?",
    "Criar tag de pageview para Meta",
    "Tag para evento de submit",
    "Gerar tag de infraestrutura",
    
    # Trigger creation
    "Criar trigger para checkout",
    "Gerar trigger de clique",
    "Preciso de um trigger de submit",
    "Como criar trigger de pageview?",
    "Trigger para evento de load",
    
    # Variable creation
    "Criar variável de data layer",
    "Gerar variável de cookie",
    "Preciso de uma variável customizada",
    "Como criar variável JS?",
    "Variável para ID do produto",
    
    # Generic creation
    "Montar nomenclatura GTM",
    "Gerar nome seguindo o guia",
    "Criar estrutura de tag",
]

# Extracted from GTM guide - Requests that should route to GTM_VALIDATION
GTM_VALIDATION_UTTERANCES = [
    # Validation requests
    "Validar tag: Cart - Load - GA4",
    "Revisar nomenclatura: Checkout Submit GA4",
    "Está correto: Homepage - Click - Meta?",
    "Verificar tag GA4CartPurchase",
    "Auditar: Various - Pageview - GA4 + Meta",
    
    # Error checking
    "Tem algum erro nessa tag?",
    "O que está errado?",
    "Corrigir nomenclatura incorreta",
    "Por que essa tag está errada?",
    
    # Specific validations
    "Validar formato de tag",
    "Revisar capitalização",
    "Verificar se usa valores do dicionário",
    "Checar prefixos",
    
    # Generic validation
    "Auditar configuração GTM",
    "Revisar nomenclaturas",
    "Está seguindo o guia?",
]


# ============================================================================
# PROMPT SELECTION LOGIC
# ============================================================================

def get_gtm_prompt(
    mode: Literal["qa", "generation", "validation"],
    context_documents: str,
    user_query: str,
) -> tuple[str, str]:
    """
    Get GTM-specific system and user prompts based on mode.
    
    Args:
        mode: GTM mode (qa, generation, validation)
        context_documents: Retrieved documents formatted as string
        user_query: User's question/request
        
    Returns:
        Tuple of (system_prompt, user_prompt)
        
    Example:
        >>> system, user = get_gtm_prompt("validation", docs, "Validar: Cart - Load - GA4")
    """
    if mode == "qa":
        return (
            GTM_QA_SYSTEM_PROMPT,
            create_gtm_qa_prompt(context_documents, user_query),
        )
    elif mode == "generation":
        return (
            GTM_GENERATION_SYSTEM_PROMPT,
            create_gtm_generation_prompt(context_documents, user_query),
        )
    elif mode == "validation":
        return (
            GTM_VALIDATION_SYSTEM_PROMPT,
            create_gtm_validation_prompt(context_documents, user_query),
        )
    else:
        raise ValueError(f"Invalid GTM mode: {mode}. Must be 'qa', 'generation', or 'validation'")
