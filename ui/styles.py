"""
Glass UI CSS theme for the Multimodal RAG Streamlit application.
"""


def get_glass_css() -> str:
    """Return the complete Glassmorphism CSS theme."""
    return """
    <style>
    /* ═══════════════════════════════════════════════════════
       MULTIMODAL RAG — GLASSMORPHISM THEME
       ═══════════════════════════════════════════════════════ */

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Root Variables ─────────────────────────────────── */
    :root {
        --bg-primary: #0a0a1a;
        --bg-secondary: #0f0f2a;
        --bg-glass: rgba(255, 255, 255, 0.04);
        --bg-glass-hover: rgba(255, 255, 255, 0.08);
        --border-glass: rgba(255, 255, 255, 0.08);
        --border-glass-hover: rgba(255, 255, 255, 0.15);
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-violet: #8b5cf6;
        --accent-violet-glow: rgba(139, 92, 246, 0.3);
        --accent-blue: #3b82f6;
        --accent-cyan: #06b6d4;
        --accent-emerald: #10b981;
        --accent-rose: #f43f5e;
        --accent-amber: #f59e0b;
        --gradient-primary: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 50%, #06b6d4 100%);
        --gradient-subtle: linear-gradient(135deg, rgba(139,92,246,0.15) 0%, rgba(59,130,246,0.15) 100%);
        --shadow-glass: 0 8px 32px rgba(0, 0, 0, 0.3);
        --shadow-glow: 0 0 40px rgba(139, 92, 246, 0.15);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;
        --blur-glass: 20px;
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ── Base Styles ───────────────────────────────────── */
    .stApp {
        background: var(--bg-primary) !important;
        background-image: 
            radial-gradient(ellipse at 20% 50%, rgba(139, 92, 246, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 20%, rgba(59, 130, 246, 0.06) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 80%, rgba(6, 182, 212, 0.05) 0%, transparent 50%) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary) !important;
    }

    /* ── Hide Streamlit Defaults ───────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden !important;}
    .stDeployButton {display: none !important;}

    /* ── Sidebar ───────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: rgba(15, 15, 35, 0.95) !important;
        backdrop-filter: blur(var(--blur-glass)) !important;
        -webkit-backdrop-filter: blur(var(--blur-glass)) !important;
        border-right: 1px solid var(--border-glass) !important;
    }

    section[data-testid="stSidebar"] > div {
        background: transparent !important;
    }

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }

    /* ── Glass Card Containers ─────────────────────────── */
    .glass-card {
        background: var(--bg-glass);
        backdrop-filter: blur(var(--blur-glass));
        -webkit-backdrop-filter: blur(var(--blur-glass));
        border: 1px solid var(--border-glass);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        box-shadow: var(--shadow-glass);
        transition: var(--transition);
        margin-bottom: 1rem;
    }

    .glass-card:hover {
        background: var(--bg-glass-hover);
        border-color: var(--border-glass-hover);
        box-shadow: var(--shadow-glass), var(--shadow-glow);
        transform: translateY(-2px);
    }

    /* ── Metrics Cards ─────────────────────────────────── */
    .metric-card {
        background: var(--bg-glass);
        backdrop-filter: blur(var(--blur-glass));
        -webkit-backdrop-filter: blur(var(--blur-glass));
        border: 1px solid var(--border-glass);
        border-radius: var(--radius-md);
        padding: 1.2rem;
        text-align: center;
        transition: var(--transition);
    }

    .metric-card:hover {
        border-color: var(--accent-violet);
        box-shadow: 0 0 20px var(--accent-violet-glow);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }

    .metric-label {
        font-size: 0.75rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.3rem;
        font-weight: 500;
    }

    /* ── Chat Messages ─────────────────────────────────── */
    .chat-message {
        padding: 1.25rem 1.5rem;
        border-radius: var(--radius-lg);
        margin: 0.75rem 0;
        line-height: 1.7;
        font-size: 0.95rem;
        animation: fadeIn 0.4s ease-out;
    }

    .chat-user {
        background: linear-gradient(135deg, rgba(139,92,246,0.12) 0%, rgba(59,130,246,0.12) 100%);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-left: 3px solid var(--accent-violet);
    }

    .chat-assistant {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-left: 3px solid var(--accent-cyan);
    }

    .chat-role {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    .chat-role-user { color: var(--accent-violet); }
    .chat-role-assistant { color: var(--accent-cyan); }

    /* ── Streamlit Chat Input ──────────────────────────── */
    .stChatInput > div {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: var(--radius-lg) !important;
        backdrop-filter: blur(var(--blur-glass)) !important;
    }

    .stChatInput > div:focus-within {
        border-color: var(--accent-violet) !important;
        box-shadow: 0 0 20px var(--accent-violet-glow) !important;
    }

    .stChatInput textarea {
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Streamlit Native Chat Messages ────────────────── */
    .stChatMessage {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: var(--radius-lg) !important;
        backdrop-filter: blur(var(--blur-glass)) !important;
        padding: 1rem !important;
    }

    [data-testid="stChatMessageContent"] {
        color: var(--text-primary) !important;
    }

    /* ── Input Fields ──────────────────────────────────── */
    .stTextInput > div > div {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        transition: var(--transition) !important;
    }

    .stTextInput > div > div:focus-within {
        border-color: var(--accent-violet) !important;
        box-shadow: 0 0 15px var(--accent-violet-glow) !important;
    }

    .stTextInput input {
        color: var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
    }

    /* ── Buttons ────────────────────────────────────────── */
    .stButton > button {
        background: var(--gradient-primary) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        transition: var(--transition) !important;
        text-transform: none !important;
        letter-spacing: 0.02em !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px var(--accent-violet-glow) !important;
        filter: brightness(1.1) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── Secondary Buttons ─────────────────────────────── */
    .stDownloadButton > button {
        background: rgba(16, 185, 129, 0.15) !important;
        color: var(--accent-emerald) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        transition: var(--transition) !important;
    }

    .stDownloadButton > button:hover {
        background: rgba(16, 185, 129, 0.25) !important;
        border-color: var(--accent-emerald) !important;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.2) !important;
    }

    /* ── File Uploader ─────────────────────────────────── */
    .stFileUploader > div {
        background: var(--bg-glass) !important;
        border: 2px dashed var(--border-glass) !important;
        border-radius: var(--radius-lg) !important;
        transition: var(--transition) !important;
    }

    .stFileUploader > div:hover {
        border-color: var(--accent-violet) !important;
        background: var(--bg-glass-hover) !important;
    }

    /* ── Selectbox ──────────────────────────────────────── */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
    }

    /* ── Expander ───────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }

    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid var(--border-glass) !important;
        border-top: none !important;
        border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
    }

    /* ── Tabs ───────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--bg-glass);
        border-radius: var(--radius-md);
        padding: 4px;
        border: 1px solid var(--border-glass);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: var(--radius-sm);
        color: var(--text-secondary);
        font-weight: 500;
        padding: 8px 16px;
        transition: var(--transition);
    }

    .stTabs [aria-selected="true"] {
        background: var(--gradient-primary) !important;
        color: white !important;
    }

    /* ── Progress Bar ──────────────────────────────────── */
    .stProgress > div > div {
        background: var(--gradient-primary) !important;
        border-radius: 10px !important;
    }

    /* ── Slider ────────────────────────────────────────── */
    .stSlider > div > div > div {
        color: var(--accent-violet) !important;
    }

    /* ── Markdown Links ────────────────────────────────── */
    a {
        color: var(--accent-violet) !important;
        text-decoration: none !important;
        transition: var(--transition) !important;
    }

    a:hover {
        color: var(--accent-blue) !important;
        text-decoration: underline !important;
    }

    /* ── Code Blocks ───────────────────────────────────── */
    code {
        background: rgba(139, 92, 246, 0.1) !important;
        color: var(--accent-violet) !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85em !important;
    }

    /* ── Divider ───────────────────────────────────────── */
    hr {
        border: none !important;
        height: 1px !important;
        background: var(--gradient-primary) !important;
        opacity: 0.3 !important;
        margin: 1.5rem 0 !important;
    }

    /* ── Animations ────────────────────────────────────── */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    .animate-fade-in {
        animation: fadeIn 0.5s ease-out;
    }

    .animate-pulse {
        animation: pulse 2s infinite;
    }

    /* ── Source Citation Tag ────────────────────────────── */
    .source-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: rgba(139, 92, 246, 0.1);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 20px;
        padding: 0.2rem 0.7rem;
        font-size: 0.72rem;
        color: var(--accent-violet);
        font-weight: 500;
        margin: 0.15rem;
        transition: var(--transition);
    }

    .source-tag:hover {
        background: rgba(139, 92, 246, 0.2);
        border-color: var(--accent-violet);
    }

    /* ── Status Indicators ─────────────────────────────── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .status-active {
        background: rgba(16, 185, 129, 0.15);
        color: var(--accent-emerald);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .status-inactive {
        background: rgba(100, 116, 139, 0.15);
        color: var(--text-muted);
        border: 1px solid rgba(100, 116, 139, 0.3);
    }

    .status-warning {
        background: rgba(245, 158, 11, 0.15);
        color: var(--accent-amber);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* ── Header Title ──────────────────────────────────── */
    .app-header {
        text-align: center;
        padding: 1.5rem 0 1rem;
        animation: fadeIn 0.6s ease-out;
    }

    .app-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }

    .app-subtitle {
        font-size: 0.9rem;
        color: var(--text-secondary);
        font-weight: 400;
    }

    /* ── Scrollbar ──────────────────────────────────────── */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(139, 92, 246, 0.3);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(139, 92, 246, 0.5);
    }

    /* ── Toast / Alert ─────────────────────────────────── */
    .stAlert {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: var(--radius-md) !important;
        backdrop-filter: blur(var(--blur-glass)) !important;
    }

    /* ── Multiselect ───────────────────────────────────── */
    .stMultiSelect > div {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: var(--radius-sm) !important;
    }

    /* ── Tooltip ────────────────────────────────────────── */
    [data-baseweb="tooltip"] {
        background: rgba(15, 15, 35, 0.95) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: var(--radius-sm) !important;
        backdrop-filter: blur(10px) !important;
    }
    </style>
    """


def render_header():
    """Render the application header."""
    return """
    <div class="app-header">
        <div class="app-title">🔮 RAGnarok</div>
        <div class="app-subtitle">Multimodal Retrieval-Augmented Generation • PDF • Audio • Video • Images</div>
    </div>
    """


def render_metric_card(value: str, label: str, icon: str = "") -> str:
    """Render a glass metric card."""
    return f"""
    <div class="metric-card">
        <div class="metric-value">{icon} {value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def render_source_tag(source: str, modality: str) -> str:
    """Render a source citation tag."""
    icons = {"text": "📄", "audio": "🎵", "video": "🎬", "image": "🖼️"}
    icon = icons.get(modality, "📎")
    return f'<span class="source-tag">{icon} {source}</span>'


def render_status_badge(text: str, status: str = "active") -> str:
    """Render a status badge."""
    dot = "🟢" if status == "active" else "🟡" if status == "warning" else "⚪"
    return f'<span class="status-badge status-{status}">{dot} {text}</span>'
