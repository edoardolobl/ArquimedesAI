"""
ArquimedesAI Premium Custom Styles v2.2.0

Modern glassmorphism, gradients, and animations for a premium AI chat experience.
Inspired by ChatGPT, Claude, and Perplexity interfaces.
"""


def get_custom_css() -> str:
    """
    Returns custom CSS for premium styling.
    
    Features:
    - Glassmorphism effects on chat messages
    - Gradient backgrounds and accents
    - Custom shadows and glows
    - Smooth animations and transitions
    - Professional typography
    - Logo styling with glow effect
    """
    return """
    <style>
    /* ============================================
       FONT IMPORTS
       ============================================ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* ============================================
       GLOBAL STYLES & VARIABLES
       ============================================ */
    :root {
        /* Color Palette */
        --primary-60: #177ddc;
        --primary-40: #3c9ae8;
        --primary-80: #1554ad;
        
        --bg-primary: #0a0e27;
        --bg-secondary: #141b2d;
        --bg-tertiary: #1e293b;
        --bg-overlay: rgba(30, 41, 59, 0.7);
        
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-tertiary: #64748b;
        
        --success: #6abe39;
        --warning: #e8b339;
        --error: #e84749;
        --info: #65a9f3;
        
        /* Effects */
        --border-subtle: rgba(255, 255, 255, 0.1);
        --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.25);
        --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.37);
        --glow-primary: 0 0 20px rgba(23, 125, 220, 0.3);
        --glow-success: 0 0 20px rgba(106, 190, 57, 0.3);
        
        /* Spacing */
        --space-sm: 0.5rem;
        --space-md: 1rem;
        --space-lg: 1.5rem;
        --space-xl: 2rem;
        
        /* Radius */
        --radius-sm: 0.5rem;
        --radius-md: 1rem;
        --radius-lg: 1.5rem;
    }
    
    /* ============================================
       MAIN APP CONTAINER
       ============================================ */
    .stApp {
        background: linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* ============================================
       SIDEBAR - INCREASED WIDTH
       ============================================ */
    section[data-testid="stSidebar"] {
        width: 28rem !important;
        min-width: 28rem !important;
    }
    
    section[data-testid="stSidebar"] > div {
        width: 28rem !important;
        min-width: 28rem !important;
    }
    
    /* ============================================
       HEADER & LOGO
       ============================================ */
    header[data-testid="stHeader"] {
        background: linear-gradient(180deg, rgba(10, 14, 39, 0.95) 0%, transparent 100%);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid var(--border-subtle);
    }
    
    .logo-container {
        display: inline-flex;
        align-items: center;
        gap: var(--space-md);
        padding: var(--space-sm) var(--space-md);
        background: linear-gradient(145deg, var(--bg-tertiary), var(--bg-secondary));
        border-radius: var(--radius-md);
        border: 2px solid rgba(23, 125, 220, 0.3);
        box-shadow: 0 4px 16px rgba(23, 125, 220, 0.4), var(--glow-primary);
        transition: all 0.3s ease;
    }
    
    .logo-container:hover {
        box-shadow: 0 6px 24px rgba(23, 125, 220, 0.5), 0 0 30px rgba(23, 125, 220, 0.3);
        transform: scale(1.05);
    }
    
    .logo-img {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        object-fit: cover;
    }
    
    .logo-text {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
    }
    
    .logo-version {
        font-size: 0.75rem;
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    /* ============================================
       SIDEBAR STYLING
       ============================================ */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-tertiary) 0%, var(--bg-secondary) 100%);
        border-right: 1px solid var(--border-subtle);
    }
    
    section[data-testid="stSidebar"] > div {
        background: transparent;
    }
    
    /* Sidebar headers */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--text-primary);
        font-weight: 600;
    }
    
    section[data-testid="stSidebar"] h1 {
        font-size: 1.5rem;
        padding-bottom: var(--space-md);
        border-bottom: 2px solid var(--border-subtle);
        margin-bottom: var(--space-lg);
    }
    
    /* ============================================
       CHAT MESSAGES (GLASSMORPHISM)
       ============================================ */
    .stChatMessage {
        background: var(--bg-overlay);
        backdrop-filter: blur(10px) saturate(180%);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: var(--space-lg);
        margin-bottom: var(--space-md);
        box-shadow: var(--shadow-lg);
        animation: slideUp 0.3s ease-out;
    }
    
    /* User message styling */
    .stChatMessage[data-testid*="user"] {
        background: linear-gradient(135deg, var(--bg-tertiary) 0%, #2a3548 100%);
        border-left: 3px solid var(--primary-60);
    }
    
    /* Assistant message styling */
    .stChatMessage[data-testid*="assistant"] {
        background: var(--bg-overlay);
        backdrop-filter: blur(10px);
    }
    
    /* ============================================
       ROUTING BADGES (WITH GLOW)
       ============================================ */
    .badge-gtm-qa {
        display: inline-flex;
        align-items: center;
        gap: 0.375rem;
        background: rgba(106, 190, 57, 0.15);
        color: var(--success);
        border: 1px solid rgba(106, 190, 57, 0.3);
        border-radius: var(--radius-sm);
        padding: 0.375rem 0.75rem;
        font-size: 0.875rem;
        font-weight: 500;
        box-shadow: var(--glow-success);
    }
    
    .badge-validation {
        display: inline-flex;
        align-items: center;
        gap: 0.375rem;
        background: rgba(232, 179, 57, 0.15);
        color: var(--warning);
        border: 1px solid rgba(232, 179, 57, 0.3);
        border-radius: var(--radius-sm);
        padding: 0.375rem 0.75rem;
        font-size: 0.875rem;
        font-weight: 500;
        box-shadow: 0 0 15px rgba(232, 179, 57, 0.2);
    }
    
    .badge-general {
        display: inline-flex;
        align-items: center;
        gap: 0.375rem;
        background: rgba(101, 169, 243, 0.15);
        color: var(--info);
        border: 1px solid rgba(101, 169, 243, 0.3);
        border-radius: var(--radius-sm);
        padding: 0.375rem 0.75rem;
        font-size: 0.875rem;
        font-weight: 500;
        box-shadow: 0 0 15px rgba(101, 169, 243, 0.2);
    }
    
    /* ============================================
       BUTTONS & INPUTS
       ============================================ */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-60) 0%, var(--primary-80) 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: var(--radius-md);
        border: none;
        box-shadow: 0 4px 12px rgba(23, 125, 220, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(23, 125, 220, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Text input */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(30, 41, 59, 0.8);
        border: 2px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        color: var(--text-primary);
        padding: var(--space-md);
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary-60);
        box-shadow: 0 0 0 3px rgba(23, 125, 220, 0.1), 0 4px 16px rgba(0, 0, 0, 0.15);
        background: rgba(30, 41, 59, 0.95);
    }
    
    /* ============================================
       STATUS INDICATORS
       ============================================ */
    .status-connected {
        display: inline-flex;
        align-items: center;
        gap: var(--space-sm);
        background: rgba(106, 190, 57, 0.1);
        color: var(--success);
        padding: var(--space-sm) var(--space-md);
        border-radius: 2rem;
        border: 1px solid rgba(106, 190, 57, 0.3);
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--success);
        box-shadow: 0 0 10px rgba(106, 190, 57, 0.5);
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* ============================================
       CHECKBOXES & SELECT BOXES
       ============================================ */
    .stCheckbox {
        padding: var(--space-sm) 0;
    }
    
    .stCheckbox label {
        color: var(--text-secondary);
        font-size: 0.875rem;
        transition: color 0.2s ease;
    }
    
    .stCheckbox label:hover {
        color: var(--text-primary);
    }
    
    .stSelectbox > div > div {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        background: rgba(30, 41, 59, 0.8);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    /* ============================================
       ANIMATIONS
       ============================================ */
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    @keyframes glowPulse {
        0%, 100% { box-shadow: 0 0 15px rgba(23, 125, 220, 0.2); }
        50% { box-shadow: 0 0 25px rgba(23, 125, 220, 0.4); }
    }
    
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
    
    /* ============================================
       SCROLLBAR CUSTOMIZATION
       ============================================ */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--primary-60), var(--primary-80));
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--primary-40), var(--primary-60));
    }
    
    /* ============================================
       CODE BLOCKS
       ============================================ */
    pre, code {
        background: rgba(10, 14, 39, 0.8);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
    }
    
    pre {
        padding: var(--space-md);
        overflow-x: auto;
    }
    
    code {
        padding: 0.125rem 0.375rem;
        font-size: 0.875rem;
    }
    
    /* ============================================
       RESPONSIVE DESIGN
       ============================================ */
    @media (max-width: 768px) {
        .logo-container {
            padding: var(--space-sm);
        }
        
        .logo-text {
            font-size: 1.25rem;
        }
        
        .stChatMessage {
            padding: var(--space-md);
        }
    }
    
    /* ============================================
       UTILITY CLASSES
       ============================================ */
    .fade-in {
        animation: fadeIn 0.3s ease-out;
    }
    
    .slide-up {
        animation: slideUp 0.3s ease-out;
    }
    
    .glow-primary {
        box-shadow: var(--glow-primary);
    }
    
    .glow-success {
        box-shadow: var(--glow-success);
    }
    
    </style>
    """


def get_logo_component() -> str:
    """
    Returns HTML component for premium logo header with base64-encoded logo.
    
    Returns:
        HTML string with logo component
    """
    # Base64-encoded logo image (embedded to avoid Streamlit static file serving issues)
    import base64
    from pathlib import Path
    
    logo_path = Path(__file__).parent.parent / "assets" / "arquimedesai.jpg"
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        logo_src = f"data:image/jpeg;base64,{logo_data}"
    else:
        # Fallback to placeholder if logo not found
        logo_src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='80' height='80'%3E%3Crect width='80' height='80' fill='%23177ddc'/%3E%3C/svg%3E"
    
    return f"""
    <div class="logo-container">
        <img src="{logo_src}" class="logo-img" alt="ArquimedesAI Logo">
        <div>
            <h1 class="logo-text">ArquimedesAI</h1>
            <span class="logo-version">v2.2.0 • Intelligent GTM Assistant</span>
        </div>
    </div>
    """
