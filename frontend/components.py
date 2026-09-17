import streamlit as st
from typing import List, Dict, Any
from backend.models import EvidenceQuote, ExpertPerspective

EXPERT_CONFIG = {
    "Dr. Sarah Lin": {"emoji": "💼", "color": "#6366f1", "bg": "rgba(99,102,241,0.15)", "sector": "FinTech"},
    "Mark Thompson":  {"emoji": "☁️", "color": "#06b6d4", "bg": "rgba(6,182,212,0.15)",  "sector": "Cloud/DevOps"},
    "Elena Rostova":  {"emoji": "🏥", "color": "#10b981", "bg": "rgba(16,185,129,0.15)", "sector": "Healthcare"},
}

def get_expert_cfg(name: str) -> dict:
    for k, v in EXPERT_CONFIG.items():
        if k in name:
            return v
    return {"emoji": "👤", "color": "#818cf8", "bg": "rgba(129,140,248,0.15)", "sector": "Expert"}

def render_badge(text: str, btype: str = "indigo") -> str:
    classes = f"badge badge-{btype}"
    return f'<span class="{classes}">{text}</span>'

def render_grounding_pill():
    st.markdown(
        '<div class="grounding-pill">🛡️ Zero Hallucination Verified — 100% Transcript Grounded</div>',
        unsafe_allow_html=True
    )

def render_stat(number: str, label: str, emoji: str = ""):
    return f"""
    <div class="stat-card">
        <span class="stat-number">{number}</span>
        <span class="stat-label">{emoji} {label}</span>
    </div>
    """

def render_evidence_quote(quote: EvidenceQuote, idx: int = 0):
    org_str = f" · {quote.organization}" if quote.organization else ""
    title_str = quote.expert_title or ""
    cfg = get_expert_cfg(quote.speaker)
    
    st.markdown(f"""
    <div class="evidence-box">
        <div class="evidence-quote-text">"{quote.quote}"</div>
        <div class="evidence-meta">
            <span class="badge badge-indigo">{cfg['emoji']} {quote.speaker}</span>
            <span class="badge badge-cyan">⏱ {quote.timestamp}</span>
            <span class="badge badge-purple">📄 {quote.transcript_id}</span>
            {"<span class='badge badge-amber'>" + quote.organization + "</span>" if quote.organization else ""}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_expert_stance_card(p: ExpertPerspective):
    cfg = get_expert_cfg(p.expert_name)
    st.markdown(f"""
    <div class="expert-card">
        <div class="expert-avatar" style="background:{cfg['bg']}; color:{cfg['color']};">
            {cfg['emoji']}
        </div>
        <div class="expert-name">{p.expert_name}</div>
        <div class="expert-org">{p.expert_title} · {p.organization}</div>
        <div class="expert-stance">{p.stance_summary}</div>
    </div>
    """, unsafe_allow_html=True)

def render_section_header(title: str, subtitle: str = "", icon: str = ""):
    st.markdown(f"""
    <div class="section-header">
        <div>
            <div class="section-header-title">{icon} {title}</div>
            {"<div class='section-header-sub'>" + subtitle + "</div>" if subtitle else ""}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_question_card(q_id: str, topic: str, question: str):
    st.markdown(f"""
    <div class="question-card">
        <div class="question-number">{q_id} · RESEARCH QUESTION</div>
        <div class="question-title">{question}</div>
        <div class="question-topic">{render_badge(topic, "purple")}</div>
    </div>
    """, unsafe_allow_html=True)
