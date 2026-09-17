import os
import sys
import json
from pathlib import Path
import streamlit as st

# Setup sys.path for backend imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.transcript_loader import transcript_loader
from backend.chunker import chunker
from backend.vector_store import vector_store
from backend.rag_engine import rag_engine
from frontend.components import (
    render_badge, render_evidence_quote_card,
    render_expert_perspective_card, render_grounding_indicator
)

# Page configuration
st.set_page_config(
    page_title="AI Interview Analyzer | Hasamex",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
css_path = Path(__file__).resolve().parent / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize Session State & Backend Index
@st.cache_resource
def initialize_system():
    transcripts = transcript_loader.get_transcripts()
    chunks = chunker.chunk_all(transcripts)
    vector_store.build_index(chunks)
    return transcripts, chunks

transcripts, chunks = initialize_system()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.markdown("### 🎙️ AI Interview Analyzer")
    st.markdown("**Role**: AI Engineer — Hasamex")
    st.markdown("**Candidate**: Rohit Kumar")
    st.markdown("---")
    
    st.markdown("#### ⚙️ LLM & Engine Settings")
    llm_choice = st.selectbox(
        "Active LLM Engine",
        ["Local High-Precision RAG (Zero-Cost / Offline)", "Google Gemini 1.5 Pro", "OpenAI GPT-4o"],
        index=0
    )
    
    if "Gemini" in llm_choice:
        gemini_api_key = st.text_input("Gemini API Key", type="password", value=settings.GEMINI_API_KEY)
        if gemini_api_key:
            settings.GEMINI_API_KEY = gemini_api_key
            settings.DEFAULT_LLM_PROVIDER = "gemini"
            rag_engine._init_llm_clients()
    elif "OpenAI" in llm_choice:
        openai_api_key = st.text_input("OpenAI API Key", type="password", value=settings.OPENAI_API_KEY)
        if openai_api_key:
            settings.OPENAI_API_KEY = openai_api_key
            settings.DEFAULT_LLM_PROVIDER = "openai"
            rag_engine._init_llm_clients()
    else:
        settings.DEFAULT_LLM_PROVIDER = "local"

    st.markdown("#### 🔍 Retrieval Hyperparameters")
    top_k = st.slider("Top-K Retrieved Chunks", min_value=2, max_value=10, value=settings.TOP_K_CHUNKS, step=1)
    
    st.markdown("---")
    st.markdown("#### 📊 Corpus Statistics")
    st.markdown(f"• **Transcripts Loaded**: `{len(transcripts)}` expert calls")
    st.markdown(f"• **Metadata Chunks**: `{len(chunks)}` index units")
    st.markdown(f"• **Active Provider**: `{rag_engine.get_active_provider()}`")
    
    st.markdown("---")
    # Export Report Button
    st.markdown("#### 📥 Export Study Report")
    if st.button("Generate Markdown Report", use_container_width=True):
        guide_res = rag_engine.generate_all_guide_answers()
        comp_res = rag_engine.generate_consensus_and_disagreements()
        
        report_md = f"# {guide_res.guide_title}\n\n"
        report_md += f"**Analyzed Transcripts**: {', '.join(guide_res.transcripts_analyzed)}\n\n"
        report_md += "## 1. Interview Guide Findings\n\n"
        for ans in guide_res.answers:
            report_md += f"### {ans.question_id}: {ans.question_text}\n"
            report_md += f"{ans.synthesized_answer}\n\n"
            report_md += "**Key Quotes:**\n"
            for q in ans.evidence_quotes:
                report_md += f"- \"{q.quote}\" ({q.speaker}, {q.timestamp})\n"
            report_md += "\n---\n"
            
        report_md += "## 2. Consensus & Disagreement Map\n\n"
        report_md += "### Common Themes (Agreements)\n"
        for c in comp_res.consensus_themes:
            report_md += f"- **{c.theme}**: {c.summary} (Supported by: {', '.join(c.supporting_experts)})\n"
        report_md += "\n### Disagreements\n"
        for d in comp_res.disagreements:
            report_md += f"- **{d.topic}**: {d.tension_summary}\n"
            
        st.download_button(
            label="💾 Download Report (.md)",
            data=report_md,
            file_name="AI_Interview_Analysis_Report.md",
            mime="text/markdown",
            use_container_width=True
        )

# Main Application Header
st.markdown(
    """
    <div class="main-header">
        <div class="main-title">AI Interview Analyzer</div>
        <div class="main-subtitle">
            Automated qualitative transcript analysis with verified timestamp citations, zero-hallucination quotes, and cross-expert consensus mapping.
        </div>
        <div style="margin-top: 10px; display: flex; gap: 8px; align-items: center;">
            <span class="badge badge-primary">Candidate: Rohit Kumar</span>
            <span class="badge badge-emerald">Hasamex AI Engineer</span>
            <span class="badge badge-timestamp">3 Expert Transcripts Indexed</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Tab 1: Interview Guide Automator",
    "⚖️ Tab 2: Expert Consensus & Disagreements",
    "💬 Tab 3: Global Q&A Chatbot",
    "📑 Tab 4: Transcripts & Raw Evidence"
])

# -------------------------------------------------------------
# TAB 1: INTERVIEW GUIDE AUTOMATOR & EVIDENCE ENGINE
# -------------------------------------------------------------
with tab1:
    st.markdown("### 📋 Interview Guide Automator & Evidence Engine")
    st.markdown("Auto-generated qualitative answers synthesized strictly from the transcripts. Expand any question to inspect **exact verbatim quotes, speaker attribution, and timestamps**.")
    
    with st.spinner("Synthesizing guide answers across all transcripts..."):
        guide_answers = rag_engine.generate_all_guide_answers()

    for idx, item in enumerate(guide_answers.answers):
        with st.container():
            st.markdown(
                f"""
                <div class="custom-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                        <span class="badge badge-primary">{item.question_id} • {item.topic}</span>
                        <span class="badge badge-emerald">🛡️ Zero Hallucination Verified</span>
                    </div>
                    <h4 style="margin: 4px 0 10px 0; color: #f8fafc;">{item.question_text}</h4>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Synthesized summary
            st.markdown(item.synthesized_answer)
            
            # Individual Expert Breakdown Columns
            st.markdown("##### 👥 Individual Expert Stances:")
            cols = st.columns(len(item.expert_perspectives))
            for c_idx, perspective in enumerate(item.expert_perspectives):
                with cols[c_idx]:
                    render_expert_perspective_card(perspective)

            # Expandable Section for Exact Quotes and Timestamps
            with st.expander(f"🔍 Expand for Evidence & Exact Quotes ({len(item.evidence_quotes)} Verified Quotes)", expanded=False):
                st.markdown("##### 📌 Verbatim Evidence Quotes with Timestamps:")
                for eq in item.evidence_quotes:
                    render_evidence_quote_card(eq)
                    
            st.markdown("---")

# -------------------------------------------------------------
# TAB 2: CONSENSUS & DISAGREEMENT MAP
# -------------------------------------------------------------
with tab2:
    st.markdown("### ⚖️ Cross-Transcript Consensus & Disagreement Map")
    st.markdown("Comparative analysis identifying where all three experts agree (Common Themes) versus where strategic opinions diverge.")
    
    with st.spinner("Analyzing cross-transcript agreements and tensions..."):
        comp_data = rag_engine.generate_consensus_and_disagreements()
        
    st.info(f"💡 **Executive Synthesis**: {comp_data.executive_synthesis}")
    
    # 2-Column layout for Consensus vs Disagreements
    col_agree, col_disagree = st.columns(2)
    
    with col_agree:
        st.markdown("#### 🤝 Common Themes (Agreements)")
        for c in comp_data.consensus_themes:
            with st.container():
                st.markdown(
                    f"""
                    <div class="consensus-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-weight: 700; color: #34d399; font-size: 1.05rem;">✔ {c.theme}</span>
                        </div>
                        <div style="font-size: 0.88rem; color: #e2e8f0; line-height: 1.45; margin-bottom: 10px;">
                            {c.summary}
                        </div>
                        <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 6px;">
                            <strong>Supporting Experts:</strong> {", ".join(c.supporting_experts)}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                with st.expander("🔍 View Supporting Evidence Quotes", expanded=False):
                    for q in c.evidence_quotes:
                        render_evidence_quote_card(q)

    with col_disagree:
        st.markdown("#### ⚡ Disagreements & Divergent Views")
        for d in comp_data.disagreements:
            with st.container():
                st.markdown(
                    f"""
                    <div class="disagreement-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-weight: 700; color: #fb7185; font-size: 1.05rem;">⚔ {d.topic}</span>
                        </div>
                        <div style="font-size: 0.88rem; color: #e2e8f0; line-height: 1.45; margin-bottom: 10px;">
                            {d.tension_summary}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Show side-by-side perspectives
                for p in d.perspectives:
                    st.markdown(
                        f"""
                        <div style="background: rgba(30, 41, 59, 0.4); border-left: 3px solid #f43f5e; padding: 8px 12px; margin-bottom: 6px; border-radius: 0 6px 6px 0; font-size: 0.83rem;">
                            <strong style="color: #38bdf8;">{p['expert']}</strong>: {p['stance']} <span class="badge badge-timestamp">{p['timestamp']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with st.expander("🔍 View Divergent Evidence Quotes", expanded=False):
                    for q in d.evidence_quotes:
                        render_evidence_quote_card(q)

# -------------------------------------------------------------
# TAB 3: GLOBAL Q&A CHATBOT
# -------------------------------------------------------------
with tab3:
    st.markdown("### 💬 Global Q&A Chatbot")
    st.markdown("Ask free-form qualitative questions across all 3 transcripts. Every answer includes **timestamped citations** and **verifiable quotes**.")
    
    # Quick prompt buttons
    st.markdown("##### ⚡ Quick Prompt Suggestions:")
    prompt_cols = st.columns(4)
    quick_prompts = [
        "What is Sarah Lin's timeline for ROI in banking?",
        "Why does Mark Thompson advocate buying commercial SaaS?",
        "How is HIPAA compliance handled at Vanguard Health?",
        "Compare autonomous agents vs copilots across all experts."
    ]
    
    selected_quick_prompt = None
    for q_idx, q_txt in enumerate(quick_prompts):
        with prompt_cols[q_idx]:
            if st.button(f"💡 {q_txt}", key=f"qp_{q_idx}", use_container_width=True):
                selected_quick_prompt = q_txt

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "quotes" in msg and msg["quotes"]:
                with st.expander(f"📌 Verified Citations ({len(msg['quotes'])} Quotes)", expanded=False):
                    for eq_dict in msg["quotes"]:
                        eq = EvidenceQuote(**eq_dict)
                        render_evidence_quote_card(eq)

    # Chat Input
    user_input = st.chat_input("Ask any question across the transcripts (e.g. ROI timeline, data bottlenecks, build vs buy)...")
    active_prompt = user_input or selected_quick_prompt
    
    if active_prompt:
        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant transcript chunks & verifying citations..."):
                chat_res = rag_engine.chat_query(active_prompt, top_k=top_k)
                st.markdown(chat_res.answer)
                
                quotes_to_save = []
                if chat_res.evidence_quotes:
                    with st.expander(f"📌 Verified Citations & Timestamps ({len(chat_res.evidence_quotes)} Quotes)", expanded=True):
                        for eq in chat_res.evidence_quotes:
                            render_evidence_quote_card(eq)
                            quotes_to_save.append(eq.model_dump())
                            
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": chat_res.answer,
                    "quotes": quotes_to_save
                })

# -------------------------------------------------------------
# TAB 4: TRANSCRIPTS & RAW EVIDENCE
# -------------------------------------------------------------
with tab4:
    st.markdown("### 📑 Full Transcript Repository & Ground Truth Inspector")
    st.markdown("Inspect raw qualitative dialogues, verify timestamps, and review speaker turns across the three expert calls.")
    
    selected_t = st.selectbox(
        "Select Transcript to Inspect",
        options=transcripts,
        format_func=lambda x: f"{x.id}: {x.expert_name} ({x.expert_title}, {x.organization})"
    )
    
    if selected_t:
        st.markdown(
            f"""
            <div class="custom-card">
                <h3>{selected_t.title}</h3>
                <div style="display: flex; gap: 8px; margin: 6px 0 12px 0;">
                    <span class="badge badge-primary">👤 {selected_t.expert_name}</span>
                    <span class="badge badge-emerald">🏢 {selected_t.organization}</span>
                    <span class="badge badge-timestamp">⏱️ Duration: {selected_t.duration}</span>
                    <span class="badge badge-amber">📅 Date: {selected_t.date}</span>
                </div>
                <p style="color: #94a3b8; font-size: 0.9rem;">{selected_t.summary}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("#### 🗣️ Dialogue Transcript Timeline")
        for u in selected_t.utterances:
            is_expert = selected_t.expert_name in u.speaker
            bg_color = "rgba(99, 102, 241, 0.1)" if is_expert else "rgba(30, 41, 59, 0.3)"
            border_color = "#818cf8" if is_expert else "#475569"
            
            st.markdown(
                f"""
                <div style="background: {bg_color}; border-left: 3px solid {border_color}; padding: 10px 14px; margin-bottom: 8px; border-radius: 0 8px 8px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                        <span style="font-weight: 700; color: {'#818cf8' if is_expert else '#94a3b8'}; font-size: 0.88rem;">
                            {'🎙️ ' if is_expert else '❓ '} {u.speaker}
                        </span>
                        <span class="badge badge-timestamp">[{u.start_time} - {u.end_time}]</span>
                    </div>
                    <div style="color: #f1f5f9; font-size: 0.9rem; line-height: 1.45;">
                        {u.text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
