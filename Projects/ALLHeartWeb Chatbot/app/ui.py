import streamlit as st
import time
import os
import sys
from pathlib import Path

# Add project folder to sys.path so we can import our modules
# The ui.py file is in app/ directory, so its parent is the project folder
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.rag_engine import RAGEngine
from app.memory import ConversationMemory
from app.config import DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL, DEFAULT_EMBEDDING_PROVIDER

# Page configuration
st.set_page_config(
    page_title="WHOIS Data Center AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply premium modern UI theme styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* Core Layout Fonts */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    code, pre, [class*="mono"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.9rem !important;
    }

    /* Gradients and Background */
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #111827 50%, #0d1527 100%);
        color: #e2e8f0;
    }

    /* Sidebar aesthetics */
    [data-testid="stSidebar"] {
        background: rgba(10, 15, 30, 0.95) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.15);
        box-shadow: 5px 0 25px rgba(0, 0, 0, 0.5);
    }
    
    [data-testid="stSidebar"] .stMarkdown h2, 
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #818cf8;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    /* Glassmorphism Title Card */
    .title-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-left: 5px solid #6366f1;
    }
    
    .title-card h1 {
        background: linear-gradient(90deg, #818cf8 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin: 0;
        font-size: 2.5rem;
    }
    
    .title-card p {
        color: #94a3b8;
        font-size: 1.1rem;
        margin: 8px 0 0 0;
    }

    /* Chat styling */
    .stChatMessage {
        border-radius: 12px !important;
        margin-bottom: 12px !important;
        padding: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.03) !important;
    }
    
    [data-testid="chatAvatarIcon-user"] {
        background-color: #6366f1 !important;
    }
    
    [data-testid="chatAvatarIcon-assistant"] {
        background-color: #a855f7 !important;
    }

    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background: rgba(30, 41, 59, 0.3) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background: rgba(15, 23, 42, 0.4) !important;
    }

    /* Badge styles for Intent classification */
    .intent-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }
    
    .badge-api {
        background-color: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    .badge-company {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    .badge-mixed {
        background-color: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }

    /* Citations expanders */
    .citation-header {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
        margin-top: 12px;
        margin-bottom: 6px;
    }
    
    .citation-card {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        border-left: 3px solid #818cf8;
    }
    
    .citation-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    
    .citation-source {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 2px;
    }

    /* Metric stats sidebar cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 8px;
        padding: 12px;
        margin-top: 10px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #818cf8;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 2px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# Cache RAG Engine instance to keep database connection persistent
@st.cache_resource
def get_rag_engine(emb_provider, llm_prov, model):
    return RAGEngine(embedding_provider=emb_provider, llm_provider=llm_prov, llm_model=model)

# Initialize Session State
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/bot.png", width=64)
    st.subheader("System Settings")
    
    # LLM Settings
    st.markdown("### LLM Configurations")
    llm_providers = ["gemini", "openai", "groq", "anthropic"]
    default_llm_idx = llm_providers.index(DEFAULT_LLM_PROVIDER) if DEFAULT_LLM_PROVIDER in llm_providers else 0
    llm_provider = st.selectbox("LLM Provider", llm_providers, index=default_llm_idx)
    
    model_options = {
        "gemini": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"],
        "openai": ["gpt-4o", "gpt-4o-mini", "text-davinci-003"],
        "groq": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
        "anthropic": ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"]
    }
    models = model_options.get(llm_provider, [])
    default_model_idx = models.index(DEFAULT_LLM_MODEL) if DEFAULT_LLM_MODEL in models else 0
    llm_model = st.selectbox("Model Name", models, index=default_model_idx)
    
    # Embedding Settings
    st.markdown("### Embedding Configurations")
    embedding_providers = ["gemini", "openai", "huggingface"]
    default_emb_idx = embedding_providers.index(DEFAULT_EMBEDDING_PROVIDER) if DEFAULT_EMBEDDING_PROVIDER in embedding_providers else 0
    embedding_provider = st.selectbox("Embedding Provider", embedding_providers, index=default_emb_idx)

    # Initialize RAG Engine
    try:
        rag_engine = get_rag_engine(embedding_provider, llm_provider, llm_model)
        db_stats = rag_engine.retriever.get_stats()
    except Exception as e:
        st.error(f"Failed to initialize RAG Engine: {e}")
        st.stop()
        
    st.markdown("### Database Status")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{db_stats['api_docs_count']}</div>
            <div class="metric-label">API Chunks</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{db_stats['company_knowledge_count']}</div>
            <div class="metric-label">Knowledge</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("### Chat Controls")
    
    if st.button("🧹 Clear Conversation", use_container_width=True):
        st.session_state.memory.clear()
        st.session_state.chat_messages = []
        st.toast("Chat history cleared!")
        st.rerun()

# Main UI layout
st.markdown("""
<div class="title-card">
    <h1>WHOIS Data Center Customer Support AI</h1>
    <p>Ask technical API queries, check auth instructions, request code snippets, or get general information about our company services.</p>
</div>
""", unsafe_allow_html=True)

# Display conversation messages from session state
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            # Render route badge
            badge_class = f"badge-{msg['routing']}"
            badge_label = msg['routing'].replace('_', ' ')
            st.markdown(f'<span class="intent-badge {badge_class}">{badge_label}</span>', unsafe_allow_html=True)
            
            # Show contextualized query if it differed
            if msg.get("contextualized") and msg["contextualized"] != msg["query"]:
                st.caption(f"🔎 *Internal search rewrite:* \"{msg['contextualized']}\"")
                
            # Show latency metrics
            if msg.get("metrics"):
                m = msg["metrics"]
                if m.get("cached"):
                    st.caption("⚡ **Served from LRU Cache** (Latency: 0.00s)")
                else:
                    st.caption(
                        f"⚡ **Latency: {m['total_response_time']:.2f}s** | "
                        f"Routing: {m['intent_classification_time']:.3f}s | "
                        f"Retrieve: {m['retrieval_time']:.3f}s | "
                        f"Compress: {m['reranking_time']:.3f}s | "
                        f"LLM: {m['llm_generation_time']:.3f}s"
                    )
            
        st.markdown(msg["content"])
        
        # Show citations if any
        if msg["role"] == "assistant" and msg.get("sources"):
            st.markdown('<div class="citation-header">Sources & Citations:</div>', unsafe_allow_html=True)
            for idx, src in enumerate(msg["sources"]):
                filename = src["file_name"]
                path_str = src["source"]
                section = src["section"]
                snippet = src["content"]
                
                category = src.get("category", "General")
                with st.expander(f"📖 [{idx+1}] {filename} ({category}) — {section or 'Main'}"):
                    st.caption(f"Source URL/Path: {path_str}")
                    st.markdown(snippet)

# Chat input and response execution
if user_query := st.chat_input("How can I assist you with WHOIS Data Center today?"):
    # Display user message in chat
    with st.chat_message("user"):
        st.markdown(user_query)
        
    st.session_state.chat_messages.append({"role": "user", "content": user_query})
    
    # Display assistant response with loading indicator
    with st.chat_message("assistant"):
        # Check if database is empty
        if db_stats['api_docs_count'] == 0 and db_stats['company_knowledge_count'] == 0:
            answer = "The vector database is empty. Please run the document ingestion script (`python app/run_ingestion.py`) to index the company documents first!"
            st.write(answer)
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": answer,
                "routing": "mixed",
                "sources": [],
                "query": user_query
            })
        else:
            message_placeholder = st.empty()
            with st.spinner("Classifying intent and retrieving context..."):
                # Process query through RAG engine
                res = rag_engine.answer_query(user_query, st.session_state.memory)
                
            # Format routed badge
            badge_class = f"badge-{res['routing']}"
            badge_label = res['routing'].replace('_', ' ')
            
            # Display rewrite caption if contextualized query differs
            rewrite_html = ""
            if res['contextualized_query'] != user_query:
                rewrite_html = f"🔎 *Internal search rewrite:* \"{res['contextualized_query']}\"\n\n"
                
            message_placeholder.markdown(f'<span class="intent-badge {badge_class}">{badge_label}</span>\n\n{rewrite_html}{res["answer"]}', unsafe_allow_html=True)
            
            # Show latency metrics
            if res.get("metrics"):
                m = res["metrics"]
                if m.get("cached"):
                    st.caption("⚡ **Served from LRU Cache** (Latency: 0.00s)")
                else:
                    st.caption(
                        f"⚡ **Latency: {m['total_response_time']:.2f}s** | "
                        f"Routing: {m['intent_classification_time']:.3f}s | "
                        f"Retrieve: {m['retrieval_time']:.3f}s | "
                        f"Compress: {m['reranking_time']:.3f}s | "
                        f"LLM: {m['llm_generation_time']:.3f}s"
                    )
            
            # Display sources
            if res["sources"]:
                st.markdown('<div class="citation-header">Sources & Citations:</div>', unsafe_allow_html=True)
                for idx, src in enumerate(res["sources"]):
                    filename = src["file_name"]
                    path_str = src["source"]
                    section = src["section"]
                    snippet = src["content"]
                    
                    category = src.get("category", "General")
                    with st.expander(f"📖 [{idx+1}] {filename} ({category}) — {section or 'Main'}"):
                        st.caption(f"Source URL/Path: {path_str}")
                        st.markdown(snippet)
                        
            # Save answer to state
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": res["answer"],
                "routing": res["routing"],
                "sources": res["sources"],
                "contextualized": res["contextualized_query"],
                "query": user_query,
                "metrics": res.get("metrics")
            })
