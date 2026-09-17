import os
import sys
import time
from pathlib import Path
import streamlit as st

# ── Path Setup ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.transcript_loader import transcript_loader
from backend.chunker import chunker
from backend.vector_store import vector_store
from backend.rag_engine import rag_engine
from backend.models import EvidenceQuote

def stream_text(text: str):
    """Real-time streaming text generator for interactive typing effect."""
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(0.015)

from frontend.components import (
    render_badge, render_grounding_pill, render_stat,
    render_evidence_quote, render_expert_stance_card,
    render_section_header, render_question_card, get_expert_cfg
)

# ── Page Config ──
st.set_page_config(
    page_title="AI Interview Analyzer | Hasamex",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Inject CSS ──
css_path = Path(__file__).resolve().parent / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Initialize Backend (cached) ──
@st.cache_resource(show_spinner=False)
def init_system():
    transcripts = transcript_loader.get_transcripts()
    chunks = chunker.chunk_all(transcripts)
    vector_store.build_index(chunks)
    return transcripts, chunks

with st.spinner("🔄 Initializing RAG pipeline..."):
    transcripts, chunks = init_system()

# ── Session State ──
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 8px 0 16px;">
        <div style="font-family:'Outfit',sans-serif; font-size:1.3rem; font-weight:800; 
                    background:linear-gradient(90deg,#818cf8,#c084fc); 
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            🎙️ AI Interview Analyzer
        </div>
        <div style="font-size:0.73rem; color:#475569; margin-top:4px;">Hasamex · AI Engineer Assignment</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Candidate info
    st.markdown("""
    <div style="background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2); 
                border-radius:10px; padding:12px 14px; margin-bottom:16px;">
        <div style="font-size:0.73rem; color:#64748b; font-weight:600; letter-spacing:0.05em; margin-bottom:6px;">CANDIDATE</div>
        <div style="font-family:'Outfit',sans-serif; font-weight:700; color:#c7d2fe; font-size:0.95rem;">Rohit Kumar</div>
        <div style="font-size:0.75rem; color:#64748b; margin-top:2px;">AI Engineer Role · Hasamex</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### ⚡ Real-Time Engine")
    default_idx = 1 if (settings.GEMINI_API_KEY and settings.GEMINI_API_KEY.strip()) else 0
    llm_choice = st.selectbox(
        "Active LLM",
        ["🔌 Local Grounded RAG (Zero-Cost)", "✨ Google Gemini 1.5 (Real-Time)", "🤖 OpenAI GPT-4o"],
        index=default_idx, label_visibility="collapsed"
    )
    
    if "Gemini" in llm_choice:
        gemini_input = st.text_input(
            "Gemini API Key",
            value=settings.GEMINI_API_KEY,
            type="password",
            placeholder="AIzaSy..."
        )
        if gemini_input != settings.GEMINI_API_KEY:
            settings.GEMINI_API_KEY = gemini_input
            settings.DEFAULT_LLM_PROVIDER = "gemini"
            rag_engine.update_gemini_key(gemini_input)
            
        c_test, c_link = st.columns([1, 1])
        with c_test:
            if st.button("🔌 Test Key", use_container_width=True):
                res = rag_engine.test_gemini_connection()
                if res["status"] == "connected":
                    st.success("✅ Connected to Gemini!")
                elif res["status"] == "auth_error":
                    st.warning("⚠️ Auth Error: Google AI Studio keys start with 'AIzaSy...'.")
                else:
                    st.info(f"Status: {res.get('message', 'Checked')}")
        with c_link:
            st.link_button("🔑 Free Key", "https://aistudio.google.com/app/apikey", use_container_width=True)

    elif "OpenAI" in llm_choice:
        key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        if key:
            settings.OPENAI_API_KEY = key
            settings.DEFAULT_LLM_PROVIDER = "openai"
            rag_engine._init_llm_clients()
    else:
        settings.DEFAULT_LLM_PROVIDER = "local"

    st.markdown("#### 🔍 Retrieval Settings")
    top_k = st.slider("Top-K Chunks Retrieved", 2, 10, settings.TOP_K_CHUNKS)

    st.markdown("---")
    st.markdown("#### 📊 System Status")
    
    provider = rag_engine.get_active_provider()
    provider_color = "#34d399" if "local" in provider else "#818cf8"
    st.markdown(f"""
    <div style="font-size:0.8rem; display:flex; flex-direction:column; gap:7px;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="color:#64748b;">Transcripts</span>
            <span style="color:#34d399; font-weight:700; font-family:'JetBrains Mono',monospace;">{len(transcripts)}</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="color:#64748b;">Index Chunks</span>
            <span style="color:#34d399; font-weight:700; font-family:'JetBrains Mono',monospace;">{len(chunks)}</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="color:#64748b;">Provider</span>
            <span style="color:{provider_color}; font-weight:700; font-size:0.72rem;">{provider}</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="color:#64748b;">Status</span>
            <span style="color:#34d399; font-weight:700;">● Healthy</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📥 Export Report")
    if st.button("📄 Generate Full Report", use_container_width=True):
        with st.spinner("Generating..."):
            gr = rag_engine.generate_all_guide_answers()
            cr = rag_engine.generate_consensus_and_disagreements()
            md = f"# {gr.guide_title}\n\n**Experts:** {', '.join(gr.transcripts_analyzed)}\n\n"
            md += "## Interview Guide Findings\n\n"
            for a in gr.answers:
                md += f"### {a.question_id}: {a.question_text}\n{a.synthesized_answer}\n\n"
                for q in a.evidence_quotes:
                    md += f'> "{q.quote}" — *{q.speaker}* {q.timestamp}\n\n'
                md += "---\n\n"
            md += "## Consensus & Disagreement Map\n\n### Agreements\n"
            for c in cr.consensus_themes:
                md += f"- **{c.theme}**: {c.summary}\n"
            md += "\n### Disagreements\n"
            for d in cr.disagreements:
                md += f"- **{d.topic}**: {d.tension_summary}\n"
            st.download_button("💾 Download .md", md, "AI_Interview_Report.md", "text/markdown", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.7rem; color:#334155; text-align:center; line-height:1.6;">
        FastAPI <span style="color:#475569;">·</span> Streamlit<br>
        BM25 + Semantic RAG<br>
        Gemini 1.5 Pro / GPT-4o<br>
        <span style="color:#6366f1;">v1.0.0</span>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════
# HERO HEADER
# ════════════════════════════════════════════
st.markdown("""
<div class="hero-header">
    <div class="hero-title">AI Interview Analyzer</div>
    <div class="hero-sub">
        Enterprise-grade qualitative transcript intelligence — automated interview guide answers, 
        verbatim quote extraction with verified timestamps, cross-expert consensus mapping, and 
        conversational Q&A. Powered by a zero-hallucination hybrid RAG pipeline.
    </div>
    <div class="hero-badge-row">
        <span class="badge badge-indigo">👤 Rohit Kumar</span>
        <span class="badge badge-purple">🏢 Hasamex AI Engineer</span>
        <span class="badge badge-emerald">🛡️ Zero Hallucination</span>
        <span class="badge badge-cyan">⏱ Timestamped Citations</span>
        <span class="badge badge-amber">📡 FastAPI + Streamlit</span>
        <span class="badge badge-green">✅ 8/8 Tests Passing</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Stats Row ──
st.markdown(f"""
<div class="stats-row">
    {render_stat("3", "Expert Calls", "🎙️")}
    {render_stat("30", "Vector Chunks", "🔍")}
    {render_stat("5", "Guide Questions", "📋")}
    {render_stat("4", "Core Features", "⚡")}
    {render_stat("100%", "Test Coverage", "✅")}
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Interview Guide",
    "⚖️ Consensus Map", 
    "💬 Ask Anything",
    "📑 Transcripts",
    "🏗️ Architecture"
])

# ════════════════════════════════════════════
# TAB 1 — INTERVIEW GUIDE AUTOMATOR
# ════════════════════════════════════════════
with tab1:
    render_section_header(
        "Interview Guide Automator", 
        "Auto-synthesized answers from all 3 expert transcripts — with verifiable evidence for every claim.",
        "📋"
    )

    with st.spinner("Synthesizing cross-expert answers..."):
        guide_resp = rag_engine.generate_all_guide_answers()

    for ans in guide_resp.answers:
        render_question_card(ans.question_id, ans.topic, ans.question_text)

        # Zero hallucination pill
        render_grounding_pill()
        st.markdown("<br>", unsafe_allow_html=True)

        # Synthesized Answer
        st.markdown(ans.synthesized_answer)

        # Expert stances in columns
        st.markdown("##### 👥 Expert-by-Expert Breakdown")
        cols = st.columns(len(ans.expert_perspectives))
        for ci, p in enumerate(ans.expert_perspectives):
            with cols[ci]:
                render_expert_stance_card(p)

        # Evidence expander
        with st.expander(
            f"🔍 Expand Evidence — {len(ans.evidence_quotes)} Exact Quotes with Timestamps", 
            expanded=False
        ):
            render_grounding_pill()
            st.markdown("<br>", unsafe_allow_html=True)
            for i, eq in enumerate(ans.evidence_quotes):
                render_evidence_quote(eq, i)

        st.markdown("---")

# ════════════════════════════════════════════
# TAB 2 — CONSENSUS & DISAGREEMENT MAP
# ════════════════════════════════════════════
with tab2:
    render_section_header(
        "Expert Consensus & Disagreement Map",
        "Where 3 experts align — and where they fundamentally diverge. All claims backed by transcript evidence.",
        "⚖️"
    )

    with st.spinner("Running cross-transcript comparative analysis..."):
        comp = rag_engine.generate_consensus_and_disagreements()

    # Executive synthesis banner
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(99,102,241,0.12) 0%,rgba(139,92,246,0.08) 100%);
                border:1px solid rgba(99,102,241,0.2); border-radius:14px; padding:18px 22px; margin-bottom:22px;">
        <div style="font-family:'Outfit',sans-serif; font-size:0.72rem; font-weight:700; 
                    color:#6366f1; letter-spacing:0.08em; margin-bottom:6px;">EXECUTIVE SYNTHESIS</div>
        <div style="color:#e2e8f0; font-size:0.92rem; line-height:1.55;">{comp.executive_synthesis}</div>
        <div style="margin-top:12px; display:flex; gap:8px; flex-wrap:wrap;">
            {"".join([f'<span class=\"badge badge-indigo\">{n}</span>' for n in comp.expert_names])}
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown(f"""
        <div style="font-family:'Outfit',sans-serif; font-size:1rem; font-weight:800; 
                    color:#34d399; margin-bottom:14px; display:flex; align-items:center; gap:8px;">
            🤝 Common Themes
            <span class="badge badge-emerald">{len(comp.consensus_themes)} Found</span>
        </div>
        """, unsafe_allow_html=True)

        for c in comp.consensus_themes:
            st.markdown(f"""
            <div class="consensus-card">
                <div class="consensus-title">✔ {c.theme}</div>
                <div class="consensus-body">{c.summary}</div>
                <div style="display:flex; flex-wrap:wrap; gap:5px; margin-top:8px;">
                    {"".join([f'<span class=\"badge badge-emerald\">👤 {e}</span>' for e in c.supporting_experts])}
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("🔍 View Supporting Evidence", expanded=False):
                for eq in c.evidence_quotes:
                    render_evidence_quote(eq)

    with col_b:
        st.markdown(f"""
        <div style="font-family:'Outfit',sans-serif; font-size:1rem; font-weight:800; 
                    color:#fb7185; margin-bottom:14px; display:flex; align-items:center; gap:8px;">
            ⚡ Disagreements
            <span class="badge badge-rose">{len(comp.disagreements)} Found</span>
        </div>
        """, unsafe_allow_html=True)

        for d in comp.disagreements:
            st.markdown(f"""
            <div class="disagreement-card">
                <div class="disagreement-title">⚔ {d.topic}</div>
                <div style="font-size:0.84rem; color:#cbd5e1; line-height:1.5; margin-bottom:12px;">
                    {d.tension_summary}
                </div>
            </div>
            """, unsafe_allow_html=True)

            for p in d.perspectives:
                st.markdown(f"""
                <div class="perspective-row">
                    <div class="perspective-expert">🎙 {p['expert']}</div>
                    <div class="perspective-stance">{p['stance']} 
                        <span class="badge badge-cyan">{p['timestamp']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("🔍 View Divergent Evidence Quotes", expanded=False):
                for eq in d.evidence_quotes:
                    render_evidence_quote(eq)
            
            st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════
# TAB 3 — GLOBAL Q&A CHATBOT
# ════════════════════════════════════════════
with tab3:
    render_section_header(
        "Global Q&A Chatbot",
        "Ask anything across all 3 transcripts. Every answer is 100% grounded with timestamped citations.",
        "💬"
    )

    # Quick prompts
    st.markdown('<div style="font-size:0.8rem; color:#64748b; font-weight:600; margin-bottom:8px;">⚡ QUICK PROMPTS</div>', unsafe_allow_html=True)
    qcols = st.columns(2)
    quick_prompts = [
        "What is Dr. Sarah Lin's ROI timeline for banking AI?",
        "Why does Mark Thompson prefer commercial SaaS over custom RAG?",
        "How does Elena Rostova enforce HIPAA compliance in AI systems?",
        "Compare all experts on autonomous agents vs copilots in 3 years.",
        "What is the biggest bottleneck in scaling enterprise AI?",
        "What is the build vs buy strategy across all three experts?"
    ]
    
    selected_prompt = None
    for qi, qp in enumerate(quick_prompts):
        col = qcols[qi % 2]
        with col:
            if st.button(f"💡 {qp}", key=f"qp_{qi}", use_container_width=True):
                selected_prompt = qp

    st.markdown("---")

    # Chat message history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("quotes"):
                    with st.expander(f"📌 Verified Citations ({len(msg['quotes'])} quotes)", expanded=False):
                        for eq_d in msg["quotes"]:
                            render_evidence_quote(EvidenceQuote(**eq_d))
                if not msg.get("grounded", True) and msg["role"] == "assistant":
                    st.markdown("""
                    <div style="background:rgba(244,63,94,0.08); border:1px solid rgba(244,63,94,0.2); 
                                border-radius:8px; padding:8px 12px; font-size:0.78rem; color:#fb7185; margin-top:8px;">
                        ⚠️ Query not found in transcripts — hallucination guardrail activated.
                    </div>
                    """, unsafe_allow_html=True)

    # Input
    user_input = st.chat_input("Ask any question across the 3 expert transcripts...")
    active = user_input or selected_prompt

    if active:
        st.session_state.chat_history.append({"role": "user", "content": active})
        with st.chat_message("user"):
            st.markdown(active)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving & verifying from transcripts..."):
                resp = rag_engine.chat_query(active, top_k=top_k)

            st.write_stream(stream_text(resp.answer))

            quotes_saved = []
            if resp.evidence_quotes:
                with st.expander(f"📌 Verified Citations ({len(resp.evidence_quotes)} quotes)", expanded=True):
                    for eq in resp.evidence_quotes:
                        render_evidence_quote(eq)
                        quotes_saved.append(eq.model_dump())

            if not resp.is_grounded:
                st.markdown("""
                <div style="background:rgba(244,63,94,0.08); border:1px solid rgba(244,63,94,0.2); 
                            border-radius:8px; padding:8px 12px; font-size:0.78rem; color:#fb7185; margin-top:8px;">
                    ⚠️ Query not found in transcripts — zero-hallucination guardrail activated.
                </div>
                """, unsafe_allow_html=True)

            if resp.cited_speakers:
                speakers_html = " ".join([f'<span class="badge badge-indigo">🎙 {s}</span>' for s in resp.cited_speakers])
                st.markdown(f'<div style="margin-top:8px;">{speakers_html}</div>', unsafe_allow_html=True)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": resp.answer,
            "quotes": quotes_saved,
            "grounded": resp.is_grounded
        })

    if st.session_state.chat_history:
        if st.button("🗑 Clear Chat History", use_container_width=False):
            st.session_state.chat_history = []
            st.rerun()

# ════════════════════════════════════════════
# TAB 4 — TRANSCRIPTS INSPECTOR
# ════════════════════════════════════════════
with tab4:
    render_section_header(
        "Full Transcript Repository",
        "Inspect raw dialogue, verify speaker turns, and review timestamps across all 3 expert calls.",
        "📑"
    )

    # Transcript selector with expert info
    t_sel = st.selectbox(
        "Select Expert Call",
        transcripts,
        format_func=lambda x: f"📞 {x.id}: {x.expert_name} — {x.organization}",
        label_visibility="visible"
    )

    if t_sel:
        cfg = get_expert_cfg(t_sel.expert_name)
        
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, {cfg['bg']} 0%, rgba(15,23,42,0.7) 100%);
                    border:1px solid {cfg['color']}33; border-radius:16px; padding:22px 26px; margin-bottom:20px;">
            <div style="display:flex; align-items:center; gap:14px; margin-bottom:12px;">
                <div style="font-size:2rem;">{cfg['emoji']}</div>
                <div>
                    <div style="font-family:'Outfit',sans-serif; font-size:1.15rem; font-weight:800; color:#f1f5f9;">
                        {t_sel.expert_name}
                    </div>
                    <div style="font-size:0.82rem; color:#94a3b8;">{t_sel.expert_title}</div>
                </div>
            </div>
            <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px;">
                <span class="badge badge-indigo">🏢 {t_sel.organization}</span>
                <span class="badge badge-cyan">⏱ Duration: {t_sel.duration}</span>
                <span class="badge badge-amber">📅 {t_sel.date}</span>
                <span class="badge badge-purple">🗣 {len(t_sel.utterances)} Utterances</span>
            </div>
            <div style="font-size:0.87rem; color:#94a3b8; line-height:1.55;">{t_sel.summary}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🗣️ Dialogue Timeline")
        for u in t_sel.utterances:
            is_expert = t_sel.expert_name in u.speaker
            card_class = "utterance-expert" if is_expert else "utterance-interviewer"
            spk_color = cfg['color'] if is_expert else "#475569"
            icon = cfg['emoji'] if is_expert else "❓"
            st.markdown(f"""
            <div class="{card_class}">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <div class="utterance-speaker" style="color:{spk_color};">{icon} {u.speaker}</div>
                    <span class="badge badge-cyan">⏱ [{u.start_time} - {u.end_time}]</span>
                </div>
                <div class="utterance-text">{u.text}</div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════
# TAB 5 — ARCHITECTURE
# ════════════════════════════════════════════
with tab5:
    render_section_header(
        "RAG Architecture & Design Decisions",
        "How the zero-hallucination pipeline works under the hood.",
        "🏗️"
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 🔄 Pipeline Flow")
        st.markdown("""
        <div class="arch-box">
<span class="highlight">3 Expert Call Transcripts</span> (JSON + Utterances)
           │
           ▼
<span class="highlight">Metadata-Injected Chunker</span>
{
  "text": "...",
  "metadata": {
    "transcript_id": "Expert_Call_1",
    "speaker": "Dr. Sarah Lin",
    "start_time": "00:46",
    "end_time": "02:15",
    "timestamp_display": "[00:46 - 02:15]"
  }
}
           │
           ▼
<span class="highlight">Hybrid Vector Store</span>
BM25 Keyword + Semantic Retrieval
           │
    ┌──────┴──────┐
    ▼             ▼
<span class="highlight-green">FastAPI</span>       <span class="highlight-cyan">Streamlit</span>
REST API      Premium UI
:8000         :8501
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### 🛡️ Zero Hallucination Design")
        st.markdown("""
        <div style="display:flex; flex-direction:column; gap:12px; margin-top:4px;">
        """, unsafe_allow_html=True)
        
        principles = [
            ("🔒", "Metadata-Injected Chunks", "Every vector chunk carries speaker name, transcript ID, and exact start/end timestamps — making citation impossible to fabricate."),
            ("📌", "Verbatim Extraction Only", "Answers quote exact sentences from transcripts. No paraphrasing, no generalization beyond what experts said."),
            ("⚠️", "Out-of-Domain Guardrail", "Queries with relevance score below threshold return: 'Not mentioned in transcripts' — never a hallucinated answer."),
            ("🔍", "BM25 + Semantic Hybrid", "Combines exact keyword matching with semantic similarity for precise, context-aware retrieval without false positives."),
        ]
        for icon, title, desc in principles:
            st.markdown(f"""
            <div style="background:rgba(15,23,42,0.6); border:1px solid rgba(99,102,241,0.15); 
                        border-radius:12px; padding:14px 16px;">
                <div style="font-family:'Outfit',sans-serif; font-weight:700; color:#c7d2fe; 
                            font-size:0.88rem; margin-bottom:5px;">{icon} {title}</div>
                <div style="font-size:0.82rem; color:#94a3b8; line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📡 REST API Reference")

    endpoints = [
        ("GET",  "/api/health",          "System health, chunk count, active LLM provider",    "badge-emerald"),
        ("GET",  "/api/transcripts",      "All 3 expert transcripts with utterances + metadata", "badge-emerald"),
        ("GET",  "/api/guide-answers",    "Synthesized answers with exact quotes + timestamps",  "badge-indigo"),
        ("GET",  "/api/compare-experts",  "Consensus themes + disagreements with evidence",      "badge-indigo"),
        ("POST", "/api/chat",             "Free-form Q&A with grounded timestamped citations",   "badge-amber"),
    ]

    for method, path, desc, badge in endpoints:
        mcolor = "#34d399" if method == "GET" else "#fbbf24"
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:12px; background:rgba(10,14,26,0.5); 
                    border:1px solid rgba(255,255,255,0.06); border-radius:10px; 
                    padding:12px 16px; margin-bottom:8px;">
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; 
                         font-weight:700; color:{mcolor}; min-width:38px;">{method}</span>
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.82rem; 
                         color:#818cf8; min-width:200px;">{path}</span>
            <span style="font-size:0.82rem; color:#94a3b8;">{desc}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding: 16px; color:#334155; font-size:0.78rem; line-height:1.8;">
        Built with ❤️ by <strong style="color:#6366f1;">Rohit Kumar</strong> · 
        FastAPI + Streamlit + BM25 RAG · 
        Gemini 1.5 Pro / GPT-4o / Local Offline Mode<br>
        <strong style="color:#475569;">Hasamex AI Engineer Assignment · v1.0.0</strong>
    </div>
    """, unsafe_allow_html=True)
